"""
Conversation-based architecture where each client has persistent websocket connections
to STT, LLM, and TTS services running on separate threads.
"""

import asyncio
import logging
import math
import numpy as np
from functools import partial
from pathlib import Path
from typing import Any, AsyncIterator, Literal, cast

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
from unmute.exceptions import make_ora_error, WebSocketClosedError
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
from unmute.recorder import Recorder
from unmute.service_discovery import find_instance
from unmute.stt.speech_to_text import SpeechToText, STTMarkerMessage
from unmute.timer import Stopwatch
from unmute.tts.text_to_speech import (
    TextToSpeech,
    TTSAudioMessage,
    TTSTextMessage,
)

# Constants from unmute_handler.py
TTS_DEBUGGING_TEXT = None
AUDIO_INPUT_OVERRIDE: Path | None = None
DEBUG_PLOT_HISTORY_SEC = 10.0
USER_SILENCE_TIMEOUT = 7.0
FIRST_MESSAGE_TEMPERATURE = 0.7
FURTHER_MESSAGES_TEMPERATURE = 0.3
UNINTERRUPTIBLE_BY_VAD_TIME_SEC = 3

logger = logging.getLogger(__name__)

HandlerOutput = (
    tuple[int, np.ndarray] | AdditionalOutputs | ora.ServerEvent | CloseStream
)


class GradioUpdate(BaseModel):
    chat_history: list[dict[str, str]]
    debug_dict: dict[str, Any]
    debug_plot_data: list[dict]


class Conversation(AsyncStreamHandler):
    """
    A conversation manages persistent websocket connections to STT, LLM, and TTS services.
    Each service runs on its own task, with STT triggering LLM, and LLM triggering TTS.
    """

    def __init__(self, conversation_id: str) -> None:
        super().__init__(
            input_sample_rate=SAMPLE_RATE,
            output_frame_size=480,
            output_sample_rate=SAMPLE_RATE,
        )
        self.conversation_id = conversation_id
        self.n_samples_received = 0
        self.output_queue: asyncio.Queue[HandlerOutput] = asyncio.Queue()
        self.recorder = Recorder(RECORDINGS_DIR) if RECORDINGS_DIR else None

        # Service instances
        self.stt: SpeechToText | None = None
        self.tts: TextToSpeech | None = None
        self.openai_client = get_openai_client()

        # Service tasks
        self.stt_task: asyncio.Task | None = None
        self.tts_task: asyncio.Task | None = None
        self.llm_task: asyncio.Task | None = None

        # STT state
        self.stt_last_message_time: float = 0
        self.stt_end_of_flush_time: float | None = None
        self.stt_flush_timer = Stopwatch()

        # TTS state
        self.tts_voice: str | None = None
        self.tts_output_stopwatch = Stopwatch()

        # Chatbot and conversation state
        self.chatbot = Chatbot()
        self.waiting_for_user_start_time: float = 0

        # Synchronization
        self.turn_transition_lock = asyncio.Lock()

        # Debug and monitoring
        self.debug_dict: dict[str, Any] = {
            "timing": {},
            "connection": {},
            "chatbot": {},
        }
        self.debug_plot_data: list[dict] = []
        self.last_additional_output_update = 0.0

        # Audio processing state
        if AUDIO_INPUT_OVERRIDE is not None:
            self.audio_input_override = AudioInputOverride(AUDIO_INPUT_OVERRIDE)
        else:
            self.audio_input_override = None

        # Event for coordinating service shutdown
        self.shutdown_event = asyncio.Event()
        self._services_started = False

    async def start_services(self) -> None:
        """Initialize and start all persistent service connections."""
        if self._services_started:
            return
            
        logger.info(f"=== CONVERSATION {self.conversation_id}: Starting services ===")
        
        # Start STT service
        await self._start_stt_service()
        
        # Set initial waiting time
        self.waiting_for_user_start_time = self.audio_received_sec()
        
        self._services_started = True
        logger.info(f"=== CONVERSATION {self.conversation_id}: All services started ===")

    async def _start_stt_service(self) -> None:
        """Start the STT service and its processing task."""
        logger.info(f"=== CONVERSATION {self.conversation_id}: Starting STT service ===")
        
        try:
            self.stt = await find_instance("stt", SpeechToText)
            await self.stt.start_up()
            
            # Start STT processing task
            self.stt_task = asyncio.create_task(
                self._stt_loop(), 
                name=f"stt_loop_{self.conversation_id}"
            )
            
            logger.info(f"=== CONVERSATION {self.conversation_id}: STT service started ===")
        except Exception as e:
            logger.error(f"=== CONVERSATION {self.conversation_id}: STT startup failed: {e} ===")
            raise

    async def _start_tts_service(self, generating_message_i: int) -> TextToSpeech:
        """Start the TTS service and its processing task."""
        logger.info(f"=== CONVERSATION {self.conversation_id}: Starting TTS service ===")
        
        try:
            factory = partial(
                TextToSpeech,
                recorder=self.recorder,
                get_time=self.audio_received_sec,
                voice=self.tts_voice,
            )
            
            tts = await find_instance("tts", factory)
            await tts.start_up()
            
            # Start TTS processing task
            self.tts_task = asyncio.create_task(
                self._tts_loop(tts, generating_message_i),
                name=f"tts_loop_{self.conversation_id}"
            )
            
            logger.info(f"=== CONVERSATION {self.conversation_id}: TTS service started ===")
            return tts
        except Exception as e:
            logger.error(f"=== CONVERSATION {self.conversation_id}: TTS startup failed: {e} ===")
            # Send error message
            error = make_ora_error(
                type="fatal",
                message="TTS service unavailable. Please try again later.",
            )
            await self.output_queue.put(error)
            raise

    async def _stt_loop(self) -> None:
        """Process STT messages and trigger LLM when text is received."""
        if not self.stt:
            return
            
        try:
            async for data in self.stt:
                if self.shutdown_event.is_set():
                    break
                    
                if isinstance(data, STTMarkerMessage):
                    continue

                logger.info(f"=== CONVERSATION {self.conversation_id}: STT output received: {data.text} ===")
                await self.output_queue.put(
                    ora.ConversationItemInputAudioTranscriptionDelta(
                        delta=data.text,
                        start_time=data.start_time,
                    )
                )

                if data.text == "":
                    continue

                self.stt_last_message_time = self.audio_received_sec()
                
                # Add to chat history
                is_new_message = await self.add_chat_message_delta(data.text, "user")
                if is_new_message:
                    self.stt_end_of_flush_time = None
                    self.stt_flush_timer.start()
                
        except Exception as e:
            logger.error(f"=== CONVERSATION {self.conversation_id}: STT loop error: {e} ===")
            if not self.shutdown_event.is_set():
                raise

    async def _generate_response(self) -> None:
        """Generate a response using LLM and TTS."""
        logger.info(f"=== CONVERSATION {self.conversation_id}: Starting response generation ===")
        
        # Empty message to signal we've started responding
        await self.add_chat_message_delta("", "assistant")
        
        # Start LLM task
        generating_message_i = len(self.chatbot.chat_history)
        self.llm_task = asyncio.create_task(
            self._generate_response_task(generating_message_i),
            name=f"llm_response_{self.conversation_id}"
        )

    async def _generate_response_task(self, generating_message_i: int) -> None:
        """Main response generation task."""
        logger.info(f"=== CONVERSATION {self.conversation_id}: Starting response generation task ===")

        await self.output_queue.put(
            ora.ResponseCreated(
                response=ora.Response(
                    status="in_progress",
                    voice=self.tts_voice or "missing",
                    chat_history=self.chatbot.chat_history,
                )
            )
        )

        llm_stopwatch = Stopwatch()

        # Start TTS service
        self.tts = await self._start_tts_service(generating_message_i)
        
        # Create LLM stream
        llm = VLLMStream(
            self.openai_client,
            temperature=FIRST_MESSAGE_TEMPERATURE
            if generating_message_i == 2
            else FURTHER_MESSAGES_TEMPERATURE,
        )

        messages = self.chatbot.preprocessed_messages()
        logger.info(f"=== CONVERSATION {self.conversation_id}: Preprocessed {len(messages)} messages for LLM ===")

        self.tts_output_stopwatch = Stopwatch(autostart=False)
        response_words = []
        error_from_tts = False
        time_to_first_token = None
        num_words_sent = sum(
            len(message.get("content", "").split()) for message in messages
        )
        
        mt.VLLM_SENT_WORDS.inc(num_words_sent)
        mt.VLLM_REQUEST_LENGTH.observe(num_words_sent)
        mt.VLLM_ACTIVE_SESSIONS.inc()

        try:
            logger.info(f"=== CONVERSATION {self.conversation_id}: Starting LLM chat completion stream ===")

            async for delta in rechunk_to_words(llm.chat_completion(messages)):
                if self.shutdown_event.is_set():
                    break
                    
                await self.output_queue.put(
                    ora.UnmuteResponseTextDeltaReady(delta=delta)
                )

                mt.VLLM_RECV_WORDS.inc()
                response_words.append(delta)

                if time_to_first_token is None:
                    time_to_first_token = llm_stopwatch.time()
                    self.debug_dict["timing"]["to_first_token"] = time_to_first_token
                    mt.VLLM_TTFT.observe(time_to_first_token)
                    logger.info(f"=== CONVERSATION {self.conversation_id}: First token time: {time_to_first_token} ===")

                self.tts_output_stopwatch.start_if_not_started()

                if len(self.chatbot.chat_history) > generating_message_i:
                    logger.info(f"=== CONVERSATION {self.conversation_id}: Response interrupted, breaking LLM loop ===")
                    break

                logger.info(f"=== CONVERSATION {self.conversation_id}: Sending word to TTS: '{delta}' ===")
                await self.tts.send(delta)

            await self.output_queue.put(
                ora.ResponseTextDone(text="".join(response_words))
            )

            logger.info(f"=== CONVERSATION {self.conversation_id}: LLM stream completed with {len(response_words)} words ===")

            if self.tts is not None:
                logger.info(f"=== CONVERSATION {self.conversation_id}: Queuing TTS EOS ===")
                self.tts.queue_eos()
            
        except asyncio.CancelledError:
            mt.VLLM_INTERRUPTS.inc()
            raise
        except Exception:
            if not error_from_tts:
                mt.VLLM_HARD_ERRORS.inc()
            raise
        finally:
            logger.info(f"=== CONVERSATION {self.conversation_id}: LLM completed, {len(response_words)} words ===")
            mt.VLLM_ACTIVE_SESSIONS.dec()
            mt.VLLM_REPLY_LENGTH.observe(len(response_words))
            mt.VLLM_GEN_DURATION.observe(llm_stopwatch.time())

    async def _tts_loop(self, tts: TextToSpeech, generating_message_i: int) -> None:
        """Process TTS messages and output audio."""
        logger.info(f"=== CONVERSATION {self.conversation_id}: Starting TTS loop ===")
        
        try:
            output_queue = self.output_queue
            audio_started = None
            message_count = 0

            async for message in tts:
                if self.shutdown_event.is_set():
                    break
                    
                message_count += 1
                logger.info(f"=== CONVERSATION {self.conversation_id}: TTS message #{message_count}: {type(message).__name__} ===")

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
                    logger.info(f"=== CONVERSATION {self.conversation_id}: TTS interrupted ===")
                    break

                if isinstance(message, TTSAudioMessage):
                    t = self.tts_output_stopwatch.stop()
                    if t is not None:
                        self.debug_dict["timing"]["tts_audio"] = t

                    audio = np.array(message.pcm, dtype=np.float32)
                    await output_queue.put((SAMPLE_RATE, audio))
                    
                    if audio_started is None:
                        audio_started = self.audio_received_sec()
                        logger.info(f"=== CONVERSATION {self.conversation_id}: First audio message received ===")
                        
                elif isinstance(message, TTSTextMessage):
                    logger.info(f"=== CONVERSATION {self.conversation_id}: TTS text: '{message.text}' ===")
                    await output_queue.put(ora.ResponseTextDelta(delta=message.text))
                    await self.add_chat_message_delta(
                        message.text, "assistant", generating_message_i
                    )

            logger.info(f"=== CONVERSATION {self.conversation_id}: TTS loop completed ===")
            
        except Exception as e:
            logger.error(f"=== CONVERSATION {self.conversation_id}: TTS loop error: {e} ===")
            if not self.shutdown_event.is_set():
                raise

    def audio_received_sec(self) -> float:
        """How much audio has been received in seconds. Used instead of time.time()."""
        current_audio_time = self.n_samples_received / self.input_sample_rate
        
        # Debug logging for timing diagnostics
        if hasattr(self, '_last_audio_time_log'):
            time_diff = current_audio_time - self._last_audio_time_log
            if time_diff > 2.0:  # Log every 2 seconds of audio time
                logger.info(f"=== CONVERSATION {self.conversation_id}: Audio timing: n_samples_received={self.n_samples_received}, audio_received_sec={current_audio_time:.3f} ===")
                self._last_audio_time_log = current_audio_time
        else:
            self._last_audio_time_log = current_audio_time
            
        return current_audio_time

    async def receive(self, frame: tuple[int, np.ndarray]) -> None:
        """Process incoming audio frame."""
        # Ensure services are started (for Gradio compatibility)
        if not self._services_started:
            await self.start_services()
            
        stt = self.stt
        if stt is None:
            return
            
        sr = frame[0]
        assert sr == self.input_sample_rate
        assert frame[1].shape[0] == 1  # Mono
        array = frame[1][0]

        self.n_samples_received += array.shape[0]
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

        # Handle TTS debugging mode
        if TTS_DEBUGGING_TEXT is not None:
            if self.chatbot.conversation_state() == "waiting_for_user":
                logger.info(f"=== CONVERSATION {self.conversation_id}: Using TTS debugging text ===")
                self.chatbot.chat_history.append(
                    {"role": "user", "content": TTS_DEBUGGING_TEXT}
                )
                await self._generate_response()
            return

        # Generate initial response if needed
        if (
            len(self.chatbot.chat_history) == 1
            and self.chatbot.get_instructions() is not None
        ):
            logger.info(f"=== CONVERSATION {self.conversation_id}: Generating initial response ===")
            await self._generate_response()

        # Apply audio input override if configured
        if self.audio_input_override is not None:
            frame = (frame[0], self.audio_input_override.override(frame[1]))

        if self.chatbot.conversation_state() == "user_speaking":
            self.debug_dict["timing"] = {}

        # Send audio to STT
        await stt.send_audio(array)
        
        # Handle pause detection and flushing
        if self.stt_end_of_flush_time is None:
            await self.detect_long_silence()

            if self.determine_pause():
                logger.info(f"=== CONVERSATION {self.conversation_id}: Pause detected ===")
                await self.output_queue.put(ora.InputAudioBufferSpeechStopped())

                self.stt_end_of_flush_time = stt.current_time + stt.delay_sec
                self.stt_flush_timer = Stopwatch()
                num_frames = int(math.ceil(stt.delay_sec / FRAME_TIME_SEC)) + 1
                zero = np.zeros(SAMPLES_PER_FRAME, dtype=np.float32)
                for _ in range(num_frames):
                    await stt.send_audio(zero)
            elif (
                self.chatbot.conversation_state() == "bot_speaking"
                and stt.pause_prediction.value < 0.4
                and self.audio_received_sec() > UNINTERRUPTIBLE_BY_VAD_TIME_SEC
            ):
                logger.info(f"=== CONVERSATION {self.conversation_id}: Interruption by STT-VAD ===")
                await self.interrupt_bot()
                await self.add_chat_message_delta("", "user")
        else:
            # STT is flushing
            if stt.current_time > self.stt_end_of_flush_time:
                self.stt_end_of_flush_time = None
                elapsed = self.stt_flush_timer.time()
                rtf = stt.delay_sec / elapsed
                logger.info(f"=== CONVERSATION {self.conversation_id}: STT Flushing finished, took {elapsed * 1000:.1f} ms, RTF: {rtf:.1f} ===")
                await self._generate_response()

    def determine_pause(self) -> bool:
        """Determine if user has paused speaking."""
        stt = self.stt
        if stt is None:
            return False
        if self.chatbot.conversation_state() != "user_speaking":
            return False

        time_since_last_message = (
            stt.sent_samples / self.input_sample_rate
        ) - self.stt_last_message_time
        self.debug_dict["time_since_last_message"] = time_since_last_message

        if stt.pause_prediction.value > 0.6:
            self.debug_dict["timing"]["pause_detection"] = time_since_last_message
            logger.info(f"=== CONVERSATION {self.conversation_id}: Pause detected ===")
            return True
        else:
            return False

    async def emit(self) -> HandlerOutput | None:
        """Emit output from the conversation."""
        output_queue_item = await wait_for_item(self.output_queue)

        if output_queue_item is not None:
            return output_queue_item
        else:
            if self.last_additional_output_update < self.audio_received_sec() - 1:
                self.last_additional_output_update = self.audio_received_sec()
                return self.get_gradio_update()
            else:
                return None

    def get_gradio_update(self) -> AdditionalOutputs:
        """Get debug information for Gradio interface."""
        self.debug_dict["conversation_state"] = self.chatbot.conversation_state()
        self.debug_dict["connection"]["stt"] = self.stt.state() if self.stt else "none"
        self.debug_dict["connection"]["tts"] = self.tts.state() if self.tts else "none"
        self.debug_dict["tts_voice"] = self.tts.voice if self.tts else "none"
        self.debug_dict["stt_pause_prediction"] = (
            self.stt.pause_prediction.value if self.stt else -1
        )

        return AdditionalOutputs(
            GradioUpdate(
                chat_history=[
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
        generating_message_i: int | None = None,
    ) -> bool:
        """Add a partial message to the chat history."""
        return await self.chatbot.add_chat_message_delta(
            delta, role, generating_message_i=generating_message_i
        )

    async def interrupt_bot(self) -> None:
        """Handle user interruption of bot response."""
        if self.chatbot.conversation_state() != "bot_speaking":
            logger.error(f"=== CONVERSATION {self.conversation_id}: Can't interrupt bot when state is {self.chatbot.conversation_state()} ===")
            raise RuntimeError(
                f"Can't interrupt bot when conversation state is {self.chatbot.conversation_state()}"
            )

        await self.add_chat_message_delta(INTERRUPTION_CHAR, "assistant")

        if self._clear_queue is not None:
            self._clear_queue()
            
        logger.info(f"=== CONVERSATION {self.conversation_id}: Clearing TTS output queue because bot was interrupted ===")
        self.output_queue = asyncio.Queue()

        # Push silence to flush Opus state
        await self.output_queue.put(
            (SAMPLE_RATE, np.zeros(SAMPLES_PER_FRAME, dtype=np.float32))
        )
        await self.output_queue.put(ora.UnmuteInterruptedByVAD())

        # Cancel current TTS and LLM tasks
        if self.tts_task and not self.tts_task.done():
            self.tts_task.cancel()
            try:
                await self.tts_task
            except asyncio.CancelledError:
                pass
                
        if self.llm_task and not self.llm_task.done():
            self.llm_task.cancel()
            try:
                await self.llm_task
            except asyncio.CancelledError:
                pass

    async def detect_long_silence(self) -> None:
        """Handle situations where the user doesn't answer for a while."""
        if (
            self.chatbot.conversation_state() == "waiting_for_user"
            and (self.audio_received_sec() - self.waiting_for_user_start_time)
            > USER_SILENCE_TIMEOUT
        ):
            logger.info(f"=== CONVERSATION {self.conversation_id}: Long silence detected ===")
            await self.add_chat_message_delta(USER_SILENCE_MARKER, "user")

    async def check_for_bot_goodbye(self) -> None:
        """Check if the bot said goodbye and close the conversation."""
        last_assistant_message = next(
            (
                msg
                for msg in reversed(self.chatbot.chat_history)
                if msg["role"] == "assistant"
            ),
            {"content": ""},
        )["content"]

        if last_assistant_message.lower().endswith("bye!"):
            await self.output_queue.put(
                CloseStream("The assistant ended the conversation. Bye!")
            )

    async def update_session(self, session: ora.SessionConfig) -> None:
        """Update session configuration."""
        if session.instructions:
            self.chatbot.set_instructions(session.instructions)

        if session.voice:
            self.tts_voice = session.voice

        if not session.allow_recording and self.recorder:
            await self.recorder.add_event("client", ora.SessionUpdate(session=session))
            await self.recorder.shutdown(keep_recording=False)
            self.recorder = None

    def copy(self) -> "Conversation":
        """Create a copy of this conversation (for compatibility)."""
        return Conversation(f"{self.conversation_id}_copy")

    async def cleanup(self) -> None:
        """Clean up all service connections and tasks."""
        logger.info(f"=== CONVERSATION {self.conversation_id}: Starting cleanup ===")
        
        # Signal shutdown to all tasks
        self.shutdown_event.set()
        
        # Cancel and wait for all tasks
        tasks = [self.stt_task, self.tts_task, self.llm_task]
        for task in tasks:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Shutdown services
        if self.stt:
            try:
                await self.stt.shutdown()
            except Exception as e:
                logger.error(f"=== CONVERSATION {self.conversation_id}: STT shutdown error: {e} ===")
                
        if self.tts:
            try:
                await self.tts.shutdown()
            except Exception as e:
                logger.error(f"=== CONVERSATION {self.conversation_id}: TTS shutdown error: {e} ===")

        if self.recorder:
            try:
                await self.recorder.shutdown()
            except Exception as e:
                logger.error(f"=== CONVERSATION {self.conversation_id}: Recorder shutdown error: {e} ===")

        logger.info(f"=== CONVERSATION {self.conversation_id}: Cleanup completed ===")

    async def __aenter__(self) -> None:
        """Context manager entry."""
        await self.start_services()

    async def __aexit__(self, *exc: Any) -> None:
        """Context manager exit."""
        await self.cleanup()


class ConversationManager:
    """
    Manages multiple conversations, creating and cleaning up conversations as clients connect/disconnect.
    """

    def __init__(self):
        self.conversations: dict[str, Conversation] = {}
        self._conversation_counter = 0
        self._lock = asyncio.Lock()

    async def create_conversation(self) -> Conversation:
        """Create a new conversation with a unique ID."""
        async with self._lock:
            self._conversation_counter += 1
            conversation_id = f"conv_{self._conversation_counter}"
            
            logger.info(f"=== CONVERSATION_MANAGER: Creating conversation {conversation_id} ===")
            
            conversation = Conversation(conversation_id)
            self.conversations[conversation_id] = conversation
            
            # Start the conversation services
            await conversation.start_services()
            
            logger.info(f"=== CONVERSATION_MANAGER: Conversation {conversation_id} created and started ===")
            return conversation

    async def remove_conversation(self, conversation_id: str) -> None:
        """Remove and cleanup a conversation."""
        async with self._lock:
            if conversation_id in self.conversations:
                logger.info(f"=== CONVERSATION_MANAGER: Removing conversation {conversation_id} ===")
                conversation = self.conversations.pop(conversation_id)
                await conversation.cleanup()
                logger.info(f"=== CONVERSATION_MANAGER: Conversation {conversation_id} removed ===")

    async def cleanup_all(self) -> None:
        """Clean up all conversations."""
        logger.info("=== CONVERSATION_MANAGER: Cleaning up all conversations ===")
        
        async with self._lock:
            conversation_ids = list(self.conversations.keys())
        
        # Clean up conversations without holding the lock
        for conversation_id in conversation_ids:
            await self.remove_conversation(conversation_id)
            
        logger.info("=== CONVERSATION_MANAGER: All conversations cleaned up ===")

    def get_conversation_count(self) -> int:
        """Get the number of active conversations."""
        return len(self.conversations)

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        """Get a conversation by ID."""
        return self.conversations.get(conversation_id)


# Global conversation manager instance
conversation_manager = ConversationManager()


def create_conversation_for_gradio() -> Conversation:
    """Create a conversation instance for Gradio (services will be started on first use)."""
    return Conversation("gradio_conversation")