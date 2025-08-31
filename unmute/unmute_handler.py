import asyncio
import math
from functools import partial
from logging import getLogger
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import websockets
from fastrtc import (
    AdditionalOutputs,
    AsyncStreamHandler,
    CloseStream,
    audio_to_float32,
    wait_for_item,
)
from pydantic import BaseModel

import unmute.openai_realtime_api_events as ora
from unmute import metrics as mt
from unmute.audio_input_override import AudioInputOverride
from unmute.exceptions import make_ora_error
from unmute.kyutai_constants import (
    FRAME_TIME_SEC,
    RECORDINGS_DIR,
    SAMPLE_RATE,
    SAMPLES_PER_FRAME,
)
from unmute.llm.chatbot import Chatbot
from unmute.llm.llm_utils import (
    INTERRUPTION_CHAR,
    USER_SILENCE_MARKER,
    VLLMStream,
    get_openai_client,
    rechunk_to_words,
)
from unmute.quest_manager import Quest, QuestManager
from unmute.recorder import Recorder
from unmute.service_discovery import find_instance
from unmute.stt.speech_to_text import SpeechToText, STTMarkerMessage
from unmute.timer import Stopwatch
from unmute.tts.text_to_speech import (
    TextToSpeech,
    TTSAudioMessage,
    TTSTextMessage,
)

# TTS_DEBUGGING_TEXT: str | None = "What's 'Hello world'?"
# TTS_DEBUGGING_TEXT: str | None = "What's the difference between a bagel and a donut?"
TTS_DEBUGGING_TEXT = None

# AUDIO_INPUT_OVERRIDE: Path | None = Path.home() / "audio/dog-or-cat-3.mp3"
AUDIO_INPUT_OVERRIDE: Path | None = None
DEBUG_PLOT_HISTORY_SEC = 10.0

USER_SILENCE_TIMEOUT = 7.0
FIRST_MESSAGE_TEMPERATURE = 0.7
FURTHER_MESSAGES_TEMPERATURE = 0.3
# For this much time, the VAD does not interrupt the bot. This is needed because at
# least on Mac, the echo cancellation takes a while to kick in, at the start, so the ASR
# sometimes hears a bit of the TTS audio and interrupts the bot. Only happens on the
# first message.
# A word from the ASR can still interrupt the bot.
UNINTERRUPTIBLE_BY_VAD_TIME_SEC = 3

logger = getLogger(__name__)

HandlerOutput = (
    tuple[int, np.ndarray] | AdditionalOutputs | ora.ServerEvent | CloseStream
)


class GradioUpdate(BaseModel):
    chat_history: list[dict[str, str]]
    debug_dict: dict[str, Any]
    debug_plot_data: list[dict]


class UnmuteHandler(AsyncStreamHandler):
    def __init__(self) -> None:
        super().__init__(
            input_sample_rate=SAMPLE_RATE,
            # IMPORTANT! If set to a higher value, will lead to choppy audio. 🤷‍♂️
            output_frame_size=480,
            output_sample_rate=SAMPLE_RATE,
        )
        self.n_samples_received = 0  # Used for measuring time
        self.output_queue: asyncio.Queue[HandlerOutput] = asyncio.Queue()
        self.recorder = Recorder(RECORDINGS_DIR) if RECORDINGS_DIR else None

        self.quest_manager = QuestManager()

        self.stt_last_message_time: float = 0
        self.stt_end_of_flush_time: float | None = None
        self.stt_flush_timer = Stopwatch()

        self.tts_voice: str | None = None  # Stored separately because TTS is restarted
        self.tts_output_stopwatch = Stopwatch()

        self.chatbot = Chatbot()
        self.openai_client = get_openai_client()

        self.turn_transition_lock = asyncio.Lock()

        self.debug_dict: dict[str, Any] = {
            "timing": {},
            "connection": {},
            "chatbot": {},
        }
        self.debug_plot_data: list[dict] = []
        self.last_additional_output_update = self.audio_received_sec()

        if AUDIO_INPUT_OVERRIDE is not None:
            self.audio_input_override = AudioInputOverride(AUDIO_INPUT_OVERRIDE)
        else:
            self.audio_input_override = None

    async def cleanup(self):
        if self.recorder is not None:
            await self.recorder.shutdown()

    @property
    def stt(self) -> SpeechToText | None:
        try:
            quest = self.quest_manager.quests["stt"]
        except KeyError:
            return None
        return cast(Quest[SpeechToText], quest).get_nowait()

    @property
    def tts(self) -> TextToSpeech | None:
        try:
            quest = self.quest_manager.quests["tts"]
        except KeyError:
            return None
        return cast(Quest[TextToSpeech], quest).get_nowait()

    def get_gradio_update(self):
        self.debug_dict["conversation_state"] = self.chatbot.conversation_state()
        self.debug_dict["connection"]["stt"] = self.stt.state() if self.stt else "none"
        self.debug_dict["connection"]["tts"] = self.tts.state() if self.tts else "none"
        self.debug_dict["tts_voice"] = self.tts.voice if self.tts else "none"
        self.debug_dict["stt_pause_prediction"] = (
            self.stt.pause_prediction.value if self.stt else -1
        )

        # This gets verbose
        # cutoff_time = self.audio_received_sec() - DEBUG_PLOT_HISTORY_SEC
        # self.debug_plot_data = [x for x in self.debug_plot_data if x["t"] > cutoff_time]

        return AdditionalOutputs(
            GradioUpdate(
                chat_history=[
                    # Not trying to hide the system prompt, just making it less verbose
                    m
                    for m in self.chatbot.chat_history
                    if m["role"] != "system"
                ],
                debug_dict=self.debug_dict,
                debug_plot_data=[],
            )
        )

    async def add_chat_message_delta(
        self,
        delta: str,
        role: Literal["user", "assistant"],
        generating_message_i: int | None = None,  # Avoid race conditions
    ):
        is_new_message = await self.chatbot.add_chat_message_delta(
            delta, role, generating_message_i=generating_message_i
        )

        return is_new_message

    async def _generate_response(self):
        logger.info("=== Starting response generation ===")
        # Empty message to signal we've started responding.
        # Do it here in the lock to avoid race conditions
        await self.add_chat_message_delta("", "assistant")
        quest = Quest.from_run_step("llm", self._generate_response_task)
        await self.quest_manager.add(quest)

    async def _generate_response_task(self):
        logger.info("=== Starting response generation task ===")
        generating_message_i = len(self.chatbot.chat_history)
        logger.info(f"Generating message index: {generating_message_i}")

        await self.output_queue.put(
            ora.ResponseCreated(
                response=ora.Response(
                    status="in_progress",
                    voice=self.tts_voice or "missing",
                    chat_history=self.chatbot.chat_history,
                )
            )
        )
        logger.info("Sent ResponseCreated event to output queue")

        llm_stopwatch = Stopwatch()

        logger.info("=== Starting TTS startup ===")
        quest = await self.start_up_tts(generating_message_i)
        logger.info("=== TTS startup completed ===")
        llm = VLLMStream(
            # if generating_message_i is 2, then we have a system prompt + an empty
            # assistant message signalling that we are generating a response.
            self.openai_client,
            temperature=FIRST_MESSAGE_TEMPERATURE
            if generating_message_i == 2
            else FURTHER_MESSAGES_TEMPERATURE,
        )

        messages = self.chatbot.preprocessed_messages()
        logger.info(f"Preprocessed {len(messages)} messages for LLM")

        self.tts_output_stopwatch = Stopwatch(autostart=False)
        tts = None

        response_words = []
        error_from_tts = False
        time_to_first_token = None
        num_words_sent = sum(
            len(message.get("content", "").split()) for message in messages
        )
        logger.info(f"Sending {num_words_sent} words to LLM")
        mt.VLLM_SENT_WORDS.inc(num_words_sent)
        mt.VLLM_REQUEST_LENGTH.observe(num_words_sent)
        mt.VLLM_ACTIVE_SESSIONS.inc()

        try:
            logger.info("=== Starting LLM chat completion stream ===")
            # Wait for TTS instance once before streaming words
            try:
                tts = await quest.get()
                logger.info("=== Got TTS instance successfully ===")
            except Exception as e:
                logger.error(f"=== Failed to get TTS instance: {e} ===")
                error_from_tts = True
                raise

            async for delta in rechunk_to_words(llm.chat_completion(messages)):
                await self.output_queue.put(
                    ora.UnmuteResponseTextDeltaReady(delta=delta)
                )

                mt.VLLM_RECV_WORDS.inc()
                response_words.append(delta)

                if time_to_first_token is None:
                    time_to_first_token = llm_stopwatch.time()
                    self.debug_dict["timing"]["to_first_token"] = time_to_first_token
                    mt.VLLM_TTFT.observe(time_to_first_token)
                    logger.info("Sending first word to TTS: %s. Time to first token: %s", delta, time_to_first_token)

                self.tts_output_stopwatch.start_if_not_started()

                if len(self.chatbot.chat_history) > generating_message_i:
                    logger.info("=== Response interrupted, breaking LLM loop ===")
                    break  # We've been interrupted

                assert isinstance(delta, str)  # make Pyright happy
                logger.info(f"=== Sending word to TTS: '{delta}' ===")
                await tts.send(delta)

            await self.output_queue.put(
                # The words include the whitespace, so no need to add it here
                ora.ResponseTextDone(text="".join(response_words))
            )

            logger.info(f"=== LLM stream completed with {len(response_words)} words ===")
            logger.info("Full LLM response: %s", "".join(response_words))

            if tts is not None:
                logger.info("=== Queuing TTS EOS after text messages ===")
                tts.queue_eos()
                logger.info("=== TTS EOS queued successfully ===")
            else:
                logger.warning("=== TTS is None, cannot queue EOS ===")
        except asyncio.CancelledError:
            mt.VLLM_INTERRUPTS.inc()
            raise
        except Exception:
            if not error_from_tts:
                mt.VLLM_HARD_ERRORS.inc()
            raise
        finally:
            logger.info("End of VLLM, after %d words.", len(response_words))
            mt.VLLM_ACTIVE_SESSIONS.dec()
            mt.VLLM_REPLY_LENGTH.observe(len(response_words))
            mt.VLLM_GEN_DURATION.observe(llm_stopwatch.time())

    def audio_received_sec(self) -> float:
        """How much audio has been received in seconds. Used instead of time.time().

        This is so that we aren't tied to real-time streaming.
        """
        current_audio_time = self.n_samples_received / self.input_sample_rate
        
        # Debug logging for timing diagnostics
        if hasattr(self, '_last_audio_time_log'):
            time_diff = current_audio_time - self._last_audio_time_log
            if time_diff > 2.0:  # Log every 2 seconds of audio time
                logger.info(f"Audio timing: n_samples_received={self.n_samples_received}, audio_received_sec={current_audio_time:.3f}, sample_rate={self.input_sample_rate}")
                self._last_audio_time_log = current_audio_time
        else:
            self._last_audio_time_log = current_audio_time
            
        return current_audio_time

    async def receive(self, frame: tuple[int, np.ndarray]) -> None:
        stt = self.stt
        assert stt is not None
        sr = frame[0]
        assert sr == self.input_sample_rate

        assert frame[1].shape[0] == 1  # Mono
        array = frame[1][0]

        self.n_samples_received += array.shape[0]

        # If this doesn't update, it means the receive loop isn't running because
        # the process is busy with something else, which is bad.
        self.debug_dict["last_receive_time"] = self.audio_received_sec()
        float_audio = audio_to_float32(array)

        self.debug_plot_data.append(
            {
                "t": self.audio_received_sec(),
                "amplitude": float(np.sqrt((float_audio**2).mean())),
                "pause_prediction": stt.pause_prediction.value,
            }
        )

        if self.chatbot.conversation_state() == "bot_speaking":
            # Periodically update this not to trigger the "long silence" accidentally.
            self.waiting_for_user_start_time = self.audio_received_sec()

        if TTS_DEBUGGING_TEXT is not None:
            assert self.audio_input_override is None, (
                "Can't use both TTS_DEBUGGING_TEXT and audio input override."
            )

            # Debugging mode: always send a fixed string when it's the user's turn.
            if self.chatbot.conversation_state() == "waiting_for_user":
                logger.info("Using TTS debugging text. Ignoring microphone.")
                self.chatbot.chat_history.append(
                    {"role": "user", "content": TTS_DEBUGGING_TEXT}
                )
                await self._generate_response()
            return

        if (
            len(self.chatbot.chat_history) == 1
            # Wait until the instructions are updated. A bit hacky
            and self.chatbot.get_instructions() is not None
        ):
            logger.info("Generating initial response.")
            await self._generate_response()

        if self.audio_input_override is not None:
            frame = (frame[0], self.audio_input_override.override(frame[1]))

        if self.chatbot.conversation_state() == "user_speaking":
            self.debug_dict["timing"] = {}

        await stt.send_audio(array)
        if self.stt_end_of_flush_time is None:
            await self.detect_long_silence()

            if self.determine_pause():
                logger.info("Pause detected")
                await self.output_queue.put(ora.InputAudioBufferSpeechStopped())

                self.stt_end_of_flush_time = stt.current_time + stt.delay_sec
                self.stt_flush_timer = Stopwatch()
                num_frames = (
                    int(math.ceil(stt.delay_sec / FRAME_TIME_SEC)) + 1
                )  # some safety margin.
                zero = np.zeros(SAMPLES_PER_FRAME, dtype=np.float32)
                for _ in range(num_frames):
                    await stt.send_audio(zero)
            elif (
                self.chatbot.conversation_state() == "bot_speaking"
                and stt.pause_prediction.value < 0.4
                and self.audio_received_sec() > UNINTERRUPTIBLE_BY_VAD_TIME_SEC
            ):
                logger.info("Interruption by STT-VAD")
                await self.interrupt_bot()
                await self.add_chat_message_delta("", "user")
        else:
            # We do not try to detect interruption here, the STT would be processing
            # a chunk full of 0, so there is little chance the pause score would indicate an interruption.
            if stt.current_time > self.stt_end_of_flush_time:
                self.stt_end_of_flush_time = None
                elapsed = self.stt_flush_timer.time()
                rtf = stt.delay_sec / elapsed
                logger.info(
                    "STT Flushing finished, took %.1f ms, RTF: %.1f", elapsed * 1000, rtf
                )
                await self._generate_response()

    def determine_pause(self) -> bool:
        stt = self.stt
        if stt is None:
            return False
        if self.chatbot.conversation_state() != "user_speaking":
            return False

        # This is how much wall clock time has passed since we received the last ASR
        # message. Assumes the ASR connection is healthy, so that stt.sent_samples is up
        # to date.
        time_since_last_message = (
            stt.sent_samples / self.input_sample_rate
        ) - self.stt_last_message_time
        self.debug_dict["time_since_last_message"] = time_since_last_message

        if stt.pause_prediction.value > 0.6:
            self.debug_dict["timing"]["pause_detection"] = time_since_last_message
            logger.info("Pause detected")
            return True
        else:
            return False

    async def emit(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
    ) -> HandlerOutput | None:
        output_queue_item = await wait_for_item(self.output_queue)

        if output_queue_item is not None:
            return output_queue_item
        else:
            if self.last_additional_output_update < self.audio_received_sec() - 1:
                # If we have nothing to emit, at least update the debug dict.
                # Don't update too often for performance reasons
                self.last_additional_output_update = self.audio_received_sec()
                return self.get_gradio_update()
            else:
                return None

    def copy(self):
        return UnmuteHandler()

    async def __aenter__(self) -> None:
        await self.quest_manager.__aenter__()

    async def start_up(self):
        print("=== UNMUTE_HANDLER: Starting up handler ===")
        await self.start_up_stt()
        print("=== UNMUTE_HANDLER: STT startup completed ===")
        self.waiting_for_user_start_time = self.audio_received_sec()
        print("=== UNMUTE_HANDLER: Handler startup completed ===")

    async def __aexit__(self, *exc: Any) -> None:
        return await self.quest_manager.__aexit__(*exc)

    async def start_up_stt(self):
        print("=== UNMUTE_HANDLER: Starting STT initialization ===")
        async def _init() -> SpeechToText:
            print("=== UNMUTE_HANDLER: Finding STT instance ===")
            # Use longer timeout for Modal services which can take time to cold start
            stt = await find_instance("stt", SpeechToText)
            print("=== UNMUTE_HANDLER: STT instance found ===")
            return stt

        async def _run(stt: SpeechToText):
            print("=== UNMUTE_HANDLER: Starting STT loop ===")
            await self._stt_loop(stt)

        async def _close(stt: SpeechToText):
            print("=== UNMUTE_HANDLER: Shutting down STT ===")
            await stt.shutdown()

        quest = await self.quest_manager.add(Quest("stt", _init, _run, _close))
        print("=== UNMUTE_HANDLER: STT quest added, waiting for initialization ===")
        # We want to be sure to have the STT before starting anything.
        await quest.get()
        print("=== UNMUTE_HANDLER: STT quest initialization completed ===")

    async def _stt_loop(self, stt: SpeechToText):
        try:
            async for data in stt:
                if isinstance(data, STTMarkerMessage):
                    # Ignore the marker messages
                    continue

                await self.output_queue.put(
                    ora.ConversationItemInputAudioTranscriptionDelta(
                        delta=data.text,
                        start_time=data.start_time,
                    )
                )

                # The STT sends an empty string as the first message, but we
                # don't want to add that because it can trigger a pause even
                # if the user hasn't started speaking yet.
                if data.text == "":
                    continue

                if self.chatbot.conversation_state() == "bot_speaking":
                    logger.info("STT-based interruption")
                    await self.interrupt_bot()

                self.stt_last_message_time = data.start_time
                is_new_message = await self.add_chat_message_delta(data.text, "user")
                if is_new_message:
                    # Ensure we don't stop after the first word if the VAD didn't have
                    # time to react.
                    stt.pause_prediction.value = 0.0
                    await self.output_queue.put(ora.InputAudioBufferSpeechStarted())
        except websockets.ConnectionClosed:
            logger.info("STT connection closed while receiving messages.")

    async def start_up_tts(self, generating_message_i: int) -> Quest[TextToSpeech]:
        async def _init() -> TextToSpeech:
            logger.info("=== TTS _init() starting ===")
            factory = partial(
                TextToSpeech,
                recorder=self.recorder,
                get_time=self.audio_received_sec,
                voice=self.tts_voice,
            )
            logger.info(f"Created TTS factory with voice: {self.tts_voice}")
            
            try:
                # find_instance already has its own retry logic, no need to duplicate it here
                logger.info("Calling find_instance for TTS...")
                tts = await find_instance("tts", factory)
                logger.info("=== TTS instance found successfully ===")
                return tts
            except Exception as e:
                logger.error(f"=== TTS connection failed: {e} ===")
                # Send a user-friendly error message
                error = make_ora_error(
                    type="error",
                    message="Unable to connect to text-to-speech service. Please try again.",
                )
                await self.output_queue.put(error)
                raise

        async def _run(tts: TextToSpeech):
            logger.info("=== Starting TTS _run loop ===")
            await self._tts_loop(tts, generating_message_i)
            logger.info("=== TTS _run loop completed ===")

        async def _close(tts: TextToSpeech):
            logger.info("=== TTS _close() starting ===")
            connection_state = tts.state()
            logger.info(f"TTS connection state at shutdown: {connection_state}")
            
            try:
                await tts.shutdown()
                logger.info("=== TTS _close() completed successfully ===")
            except Exception as e:
                logger.error(f"=== TTS _close() failed: {e} ===")
                raise

        logger.info("=== Adding TTS quest to quest manager ===")
        quest = await self.quest_manager.add(Quest("tts", _init, _run, _close))
        logger.info("=== TTS quest added successfully ===")
        return quest

    async def _tts_loop(self, tts: TextToSpeech, generating_message_i: int):
        logger.info("=== TTS loop starting ===")
        # On interruption, we swap the output queue. This will ensure that this worker
        # can never accidentally push to the new queue if it's interrupted.
        output_queue = self.output_queue
        try:
            audio_started = None
            message_count = 0
            last_message_time = asyncio.get_event_loop().time()
            last_text_message_time = None
            last_audio_message_time = None
            text_message_count = 0
            audio_message_count = 0
            
            # TTS Message Flow Watchdog - monitors for stopped message flow
            async def tts_watchdog():
                """Monitor TTS message flow and alert if it stops unexpectedly"""
                while True:
                    await asyncio.sleep(5.0)  # Check every 5 seconds
                    current_watchdog_time = asyncio.get_event_loop().time()
                    
                    # Check if we've been receiving messages
                    time_since_last_message = current_watchdog_time - last_message_time
                    
                    # Alert if no messages for 10+ seconds (indicates potential TTS server issue)
                    if time_since_last_message > 10.0 and message_count > 0:
                        logger.warning(f"=== TTS MESSAGE FLOW WATCHDOG ALERT ===")
                        logger.warning(f"No TTS messages received for {time_since_last_message:.1f}s")
                        logger.warning(f"Last message counts: total={message_count}, text={text_message_count}, audio={audio_message_count}")
                        logger.warning(f"TTS connection state: {tts.state()}")
                        await tts.send("Yeah it broke")
                        
                        # Check if TTS server may have crashed/exited
                        if time_since_last_message > 30.0:
                            logger.error(f"=== SUSPECTED TTS SERVER FAILURE - No messages for {time_since_last_message:.1f}s ===")
                            logger.error("This suggests the TTS server may have crashed or exited unexpectedly")
            
            # Start the watchdog task
            watchdog_task = asyncio.create_task(tts_watchdog())

            logger.info("=== Starting to iterate over TTS messages ===")
            async for message in tts:
                current_time = asyncio.get_event_loop().time()
                message_count += 1
                last_message_time = current_time
                
                logger.info(f"=== Received TTS message #{message_count}: {type(message).__name__} ===")
                
                # Track message types and timing for flow monitoring
                if isinstance(message, TTSTextMessage):
                    text_message_count += 1
                    last_text_message_time = current_time
                elif isinstance(message, TTSAudioMessage):
                    audio_message_count += 1
                    last_audio_message_time = current_time
                
                if audio_started is not None:
                    time_since_start = self.audio_received_sec() - audio_started
                    time_received = tts.received_samples / self.input_sample_rate
                    time_received_yielded = (
                        tts.received_samples_yielded / self.input_sample_rate
                    )
                    assert self.input_sample_rate == SAMPLE_RATE
                    self.debug_dict["tts_throughput"] = {
                        "time_received": round(time_received, 2),
                        "time_received_yielded": round(time_received_yielded, 2),
                        "time_since_start": round(time_since_start, 2),
                        "ratio": round(
                            time_received_yielded / (time_since_start + 0.01), 2
                        ),
                    }

                if len(self.chatbot.chat_history) > generating_message_i:
                    logger.info("=== Response interrupted, breaking TTS loop ===")
                    break

                if isinstance(message, TTSAudioMessage):
                    logger.info(f"=== Processing TTSAudioMessage with {len(message.pcm)} samples ===")
                    t = self.tts_output_stopwatch.stop()
                    if t is not None:
                        self.debug_dict["timing"]["tts_audio"] = t

                    audio = np.array(message.pcm, dtype=np.float32)
                    assert self.output_sample_rate == SAMPLE_RATE

                    logger.info("=== Putting audio in output queue ===")
                    await output_queue.put((SAMPLE_RATE, audio))

                    if audio_started is None:
                        audio_started = self.audio_received_sec()
                        logger.info("=== First audio message received ===")
                elif isinstance(message, TTSTextMessage):
                    logger.info(f"=== Processing TTSTextMessage: '{message.text}' ===")
                    await output_queue.put(ora.ResponseTextDelta(delta=message.text))
                    await self.add_chat_message_delta(
                        message.text,
                        "assistant",
                        generating_message_i=generating_message_i,
                    )
                else:
                    logger.warning("Got unexpected message from TTS: %s", message.type)

        except websockets.ConnectionClosedError as e:
            current_time = asyncio.get_event_loop().time()
            logger.error(f"=== TTS CONNECTION CLOSED WITH ERROR: {e} ===")
            logger.error(f"TTS message flow stats: total={message_count if 'message_count' in locals() else 0}, text={text_message_count if 'text_message_count' in locals() else 0}, audio={audio_message_count if 'audio_message_count' in locals() else 0}")
            if 'last_text_message_time' in locals() and last_text_message_time:
                logger.error(f"Last text message was {current_time - last_text_message_time:.1f}s ago")
            if 'last_audio_message_time' in locals() and last_audio_message_time:
                logger.error(f"Last audio message was {current_time - last_audio_message_time:.1f}s ago")
        finally:
            # Cancel the watchdog task
            if 'watchdog_task' in locals() and watchdog_task is not None:
                watchdog_task.cancel()
                try:
                    await watchdog_task
                except asyncio.CancelledError:
                    pass

        logger.info("=== TTS loop ended, cleaning up ===")
        logger.info(f"TTS session stats: processed {message_count if 'message_count' in locals() else 0} messages ({text_message_count if 'text_message_count' in locals() else 0} text, {audio_message_count if 'audio_message_count' in locals() else 0} audio)")
        
        # Push some silence to flush the Opus state.
        # Not sure that this is actually needed.
        logger.info("=== Pushing silence to flush Opus state ===")
        await output_queue.put(
            (SAMPLE_RATE, np.zeros(SAMPLES_PER_FRAME, dtype=np.float32))
        )

        message = self.chatbot.last_message("assistant")
        if message is None:
            logger.warning("No message to send in TTS shutdown.")
            message = ""

        # It's convenient to have the whole chat history available in the client
        # after the response is done, so send the "gradio update"
        logger.info("=== Sending final gradio update and ResponseAudioDone ===")
        await self.output_queue.put(self.get_gradio_update())
        await self.output_queue.put(ora.ResponseAudioDone())

        # Signal that the turn is over by adding an empty message.
        logger.info("=== Adding empty user message to signal turn end ===")
        await self.add_chat_message_delta("", "user")

        await asyncio.sleep(1)
        await self.check_for_bot_goodbye()
        self.waiting_for_user_start_time = self.audio_received_sec()
        logger.info("=== TTS loop cleanup completed ===")

    async def interrupt_bot(self):
        if self.chatbot.conversation_state() != "bot_speaking":
            logger.error(f"Can't interrupt bot when conversation state is {self.chatbot.conversation_state()}")
            raise RuntimeError(
                "Can't interrupt bot when conversation state is "
                f"{self.chatbot.conversation_state()}"
            )

        await self.add_chat_message_delta(INTERRUPTION_CHAR, "assistant")

        if self._clear_queue is not None:
            # Clear any audio queued up by FastRTC's emit().
            # Not sure under what circumstatnces this is None.
            self._clear_queue()
        logger.info("=== Clearing TTS output queue because bot was interrupted ===")
        self.output_queue = asyncio.Queue()  # Clear our own queue too

        # Push some silence to flush the Opus state.
        # Not sure that this is actually needed.
        await self.output_queue.put(
            (SAMPLE_RATE, np.zeros(SAMPLES_PER_FRAME, dtype=np.float32))
        )

        await self.output_queue.put(ora.UnmuteInterruptedByVAD())

        await self.quest_manager.remove("tts")
        await self.quest_manager.remove("llm")

    async def check_for_bot_goodbye(self):
        last_assistant_message = next(
            (
                msg
                for msg in reversed(self.chatbot.chat_history)
                if msg["role"] == "assistant"
            ),
            {"content": ""},
        )["content"]

        # Using function calling would be a more robust solution, but it would make it
        # harder to swap LLMs.
        if last_assistant_message.lower().endswith("bye!"):
            await self.output_queue.put(
                CloseStream("The assistant ended the conversation. Bye!")
            )

    async def detect_long_silence(self):
        """Handle situations where the user doesn't answer for a while."""
        if (
            self.chatbot.conversation_state() == "waiting_for_user"
            and (self.audio_received_sec() - self.waiting_for_user_start_time)
            > USER_SILENCE_TIMEOUT
        ):
            # This will trigger pause detection because it changes the conversation
            # state to "user_speaking".
            # The system prompt has a rule that tells it how to handle the "..."
            # messages.
            logger.info("Long silence detected.")
            await self.add_chat_message_delta(USER_SILENCE_MARKER, "user")

    async def update_session(self, session: ora.SessionConfig):
        if session.instructions:
            self.chatbot.set_instructions(session.instructions)

        if session.voice:
            self.tts_voice = session.voice

        if not session.allow_recording and self.recorder:
            await self.recorder.add_event("client", ora.SessionUpdate(session=session))
            await self.recorder.shutdown(keep_recording=False)
            self.recorder = None
            logger.info("Recording disabled for a session.")
