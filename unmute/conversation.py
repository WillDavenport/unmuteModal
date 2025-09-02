"""
Conversation-based architecture where each client connection has persistent
websocket connections to STT, LLM, and TTS services.
"""

import asyncio
import logging
from functools import partial
from typing import Any, Literal
from uuid import uuid4

import websockets
from fastrtc import AdditionalOutputs, CloseStream

import unmute.openai_realtime_api_events as ora
from unmute.exceptions import (
    MissingServiceAtCapacity,
    MissingServiceTimeout,
    WebSocketClosedError,
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
from unmute.tts.text_to_speech import TextToSpeech, TTSAudioMessage, TTSTextMessage
from unmute.kyutai_constants import (
    RECORDINGS_DIR, 
    SAMPLE_RATE, 
    SAMPLES_PER_FRAME,
    FRAME_TIME_SEC,
)
from unmute import metrics as mt
import numpy as np

# Constants from unmute_handler
USER_SILENCE_TIMEOUT = 7.0
FIRST_MESSAGE_TEMPERATURE = 0.7
FURTHER_MESSAGES_TEMPERATURE = 0.3
UNINTERRUPTIBLE_BY_VAD_TIME_SEC = 3

logger = logging.getLogger(__name__)


class Conversation:
    """
    A Conversation represents a single client session with persistent websocket
    connections to STT, LLM, and TTS services. Each service runs on its own thread
    and they communicate through async queues and events.
    """

    def __init__(self, conversation_id: str | None = None):
        self.conversation_id = conversation_id or str(uuid4())
        self.output_queue: asyncio.Queue[ora.ServerEvent | AdditionalOutputs | CloseStream] = asyncio.Queue()
        
        # Service connections
        self.stt: SpeechToText | None = None
        self.tts: TextToSpeech | None = None
        self.chatbot = Chatbot()
        self.openai_client = get_openai_client()
        
        # Tasks for each service
        self.stt_task: asyncio.Task | None = None
        self.tts_task: asyncio.Task | None = None
        self.llm_task: asyncio.Task | None = None
        
        # Synchronization
        self.turn_transition_lock = asyncio.Lock()
        self.shutdown_event = asyncio.Event()
        
        # State tracking
        self.recorder = Recorder(RECORDINGS_DIR) if RECORDINGS_DIR else None
        self.stt_last_message_time: float = 0
        self.stt_end_of_flush_time: float | None = None
        self.stt_flush_timer = Stopwatch()
        self.tts_voice: str | None = None
        self.tts_output_stopwatch = Stopwatch()
        self.n_samples_received = 0
        self.waiting_for_user_start_time = 0.0
        self.input_sample_rate = SAMPLE_RATE
        self.output_sample_rate = SAMPLE_RATE
        self._clear_queue: callable | None = None
        
        # Debug and metrics
        self.debug_dict: dict[str, Any] = {
            "timing": {},
            "connection": {},
            "chatbot": {},
        }
        self.debug_plot_data: list[dict] = []
        self.last_additional_output_update = 0.0
        
        # Events for service coordination
        self.stt_finished_event = asyncio.Event()
        self.llm_finished_event = asyncio.Event()
        
        logger.info(f"Created conversation {self.conversation_id}")

    async def start(self):
        """Initialize all service connections and start their tasks."""
        logger.info(f"Starting conversation {self.conversation_id}")
        
        try:
            # Initialize STT connection
            await self._init_stt()
            
            # Start service tasks
            self.stt_task = asyncio.create_task(
                self._stt_loop(), name=f"stt_loop_{self.conversation_id}"
            )
            
            # Initialize waiting time
            self.waiting_for_user_start_time = 0.0  # Will be set when audio starts coming in
            
            logger.info(f"Conversation {self.conversation_id} started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start conversation {self.conversation_id}: {e}")
            await self.cleanup()
            raise

    async def _init_stt(self):
        """Initialize STT websocket connection."""
        logger.info(f"Initializing STT for conversation {self.conversation_id}")
        try:
            self.stt = await find_instance("stt", SpeechToText)
            logger.info(f"STT connection established for conversation {self.conversation_id}")
        except Exception as e:
            logger.error(f"Failed to connect to STT service: {e}")
            raise

    async def _init_tts(self, generating_message_i: int):
        """Initialize TTS websocket connection."""
        logger.info(f"Initializing TTS for conversation {self.conversation_id}")
        try:
            factory = partial(
                TextToSpeech,
                recorder=self.recorder,
                get_time=lambda: self.n_samples_received / SAMPLE_RATE,
                voice=self.tts_voice,
            )
            self.tts = await find_instance("tts", factory)
            logger.info(f"TTS connection established for conversation {self.conversation_id}")
            
            # Start TTS task
            self.tts_task = asyncio.create_task(
                self._tts_loop(generating_message_i), name=f"tts_loop_{self.conversation_id}"
            )
            
        except Exception as e:
            logger.error(f"Failed to connect to TTS service: {e}")
            raise

    async def _stt_loop(self):
        """Main STT processing loop."""
        if not self.stt:
            return
            
        logger.info(f"Starting STT loop for conversation {self.conversation_id}")
        try:
            async for data in self.stt:
                if self.shutdown_event.is_set():
                    break
                    
                if isinstance(data, STTMarkerMessage):
                    continue

                logger.info(f"STT output received: {data.text}, start time: {data.start_time}")
                await self.output_queue.put(
                    ora.ConversationItemInputAudioTranscriptionDelta(
                        delta=data.text,
                        start_time=data.start_time,
                    )
                )

                if data.text == "":
                    continue

                # Check for interruption
                if self.chatbot.conversation_state() == "bot_speaking":
                    logger.info("STT-based interruption detected")
                    await self._interrupt_bot()

                self.stt_last_message_time = data.start_time
                is_new_message = await self._add_chat_message_delta(data.text, "user")
                if is_new_message:
                    if self.stt:
                        self.stt.pause_prediction.value = 0.0
                    await self.output_queue.put(ora.InputAudioBufferSpeechStarted())
                    
                # Pause detection is handled in the conversation_handler
                # based on the audio input processing, not here in the STT loop
                    
        except websockets.ConnectionClosed:
            logger.info(f"STT connection closed for conversation {self.conversation_id}")
        except Exception as e:
            logger.error(f"STT loop error for conversation {self.conversation_id}: {e}")
            raise

    async def _tts_loop(self, generating_message_i: int):
        """Main TTS processing loop."""
        if not self.tts:
            return
            
        logger.info(f"Starting TTS loop for conversation {self.conversation_id}")
        output_queue = self.output_queue
        
        try:
            audio_started = None
            message_count = 0

            async for message in self.tts:
                if self.shutdown_event.is_set():
                    break
                    
                message_count += 1
                logger.info(f"Received TTS message #{message_count}: {type(message).__name__}")

                # Check for interruption
                if len(self.chatbot.chat_history) > generating_message_i:
                    logger.info("Response interrupted, breaking TTS loop")
                    break

                if isinstance(message, TTSAudioMessage):
                    logger.info(f"Processing TTSAudioMessage with {len(message.pcm)} samples")
                    t = self.tts_output_stopwatch.stop()
                    if t is not None:
                        self.debug_dict["timing"]["tts_audio"] = t

                    audio = np.array(message.pcm, dtype=np.float32)
                    
                    # Output as tuple for FastRTC compatibility
                    await output_queue.put((SAMPLE_RATE, audio))

                    if audio_started is None:
                        audio_started = self.n_samples_received / SAMPLE_RATE
                        logger.info("First audio message received")

                elif isinstance(message, TTSTextMessage):
                    logger.info(f"Processing TTSTextMessage: {message.text}")
                    await output_queue.put(ora.ResponseTextDelta(delta=message.text))
                    await self._add_chat_message_delta(
                        message.text,
                        "assistant",
                        generating_message_i=generating_message_i,
                    )
                else:
                    logger.warning("Got unexpected message from TTS: %s", message.type)

        except websockets.ConnectionClosedError as e:
            logger.error(f"TTS CONNECTION CLOSED WITH ERROR: {e}")

        logger.info("TTS loop ended, cleaning up")
        
        # Push some silence to flush the Opus state
        logger.info("Pushing silence to flush Opus state")
        await output_queue.put(
            (SAMPLE_RATE, np.zeros(SAMPLES_PER_FRAME, dtype=np.float32))
        )

        message = self.chatbot.last_message("assistant")
        if message is None:
            logger.warning("No message to send in TTS shutdown.")
            message = ""

        # Send final updates
        logger.info("Sending final gradio update and ResponseAudioDone")
        await self.output_queue.put(self._get_gradio_update())
        await self.output_queue.put(ora.ResponseAudioDone())

        # Signal that the turn is over by adding an empty message
        logger.info("Adding empty user message to signal turn end")
        await self._add_chat_message_delta("", "user")

        await asyncio.sleep(1)
        await self._check_for_bot_goodbye()
        self.waiting_for_user_start_time = self.n_samples_received / SAMPLE_RATE
        logger.info("TTS loop cleanup completed")

    async def _generate_response(self):
        """Generate LLM response and trigger TTS."""
        logger.info(f"Starting response generation for conversation {self.conversation_id}")
        
        async with self.turn_transition_lock:
            # Empty message to signal we've started responding
            await self._add_chat_message_delta("", "assistant")
            
            # Start LLM task
            self.llm_task = asyncio.create_task(
                self._generate_response_task(), name=f"llm_task_{self.conversation_id}"
            )

    async def _generate_response_task(self):
        """Main LLM processing task."""
        logger.info(f"Starting LLM response generation task for conversation {self.conversation_id}")
        generating_message_i = len(self.chatbot.chat_history)
        
        await self.output_queue.put(
            ora.ResponseCreated(
                response=ora.Response(
                    status="in_progress",
                    voice=self.tts_voice or "missing",
                    chat_history=self.chatbot.chat_history,
                )
            )
        )

        # Initialize TTS connection for this response
        await self._init_tts(generating_message_i)

        # Generate LLM response and stream to TTS
        try:
            # This would contain the actual LLM streaming logic
            # For now, placeholder implementation
            await self._stream_llm_to_tts(generating_message_i)
            
        except Exception as e:
            logger.error(f"LLM response generation failed: {e}")
            raise
        finally:
            self.llm_finished_event.set()

    async def _stream_llm_to_tts(self, generating_message_i: int):
        """Stream LLM response to TTS service."""
        logger.info(f"Streaming LLM to TTS for conversation {self.conversation_id}")
        
        llm_stopwatch = Stopwatch()
        llm = VLLMStream(
            self.openai_client,
            temperature=FIRST_MESSAGE_TEMPERATURE
            if generating_message_i == 2
            else FURTHER_MESSAGES_TEMPERATURE,
        )

        messages = self.chatbot.preprocessed_messages()
        logger.info(f"Preprocessed {len(messages)} messages for LLM")

        self.tts_output_stopwatch = Stopwatch(autostart=False)

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
            logger.info("Starting LLM chat completion stream")
            
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
                    logger.info("Response interrupted, breaking LLM loop")
                    break  # We've been interrupted

                assert isinstance(delta, str)
                logger.info(f"Sending word to TTS: '{delta}'")
                if self.tts:
                    await self.tts.send(delta)

            await self.output_queue.put(
                ora.ResponseTextDone(text="".join(response_words))
            )

            logger.info(f"LLM stream completed with {len(response_words)} words")
            logger.info("Full LLM response: %s", "".join(response_words))

            if self.tts is not None:
                logger.info("Queuing TTS EOS after text messages")
                self.tts.queue_eos()
                logger.info("TTS EOS queued successfully")
            else:
                logger.warning("TTS is None, cannot queue EOS")
                
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

    async def _add_chat_message_delta(
        self,
        delta: str,
        role: Literal["user", "assistant"],
        generating_message_i: int | None = None,
    ) -> bool:
        """Add a partial message to chat history."""
        return await self.chatbot.add_chat_message_delta(
            delta, role, generating_message_i=generating_message_i
        )

    def _determine_pause(self) -> bool:
        """Determine if user has paused speaking."""
        if not self.stt:
            return False
        if self.chatbot.conversation_state() != "user_speaking":
            return False

        time_since_last_message = (
            self.stt.sent_samples / SAMPLE_RATE
        ) - self.stt_last_message_time
        self.debug_dict["time_since_last_message"] = time_since_last_message

        if self.stt.pause_prediction.value > 0.6:
            self.debug_dict["timing"]["pause_detection"] = time_since_last_message
            logger.info("Pause detected")
            return True
        else:
            return False

    async def _interrupt_bot(self):
        """Handle bot interruption."""
        if self.chatbot.conversation_state() != "bot_speaking":
            logger.error(f"Can't interrupt bot when conversation state is {self.chatbot.conversation_state()}")
            raise RuntimeError(
                "Can't interrupt bot when conversation state is "
                f"{self.chatbot.conversation_state()}"
            )

        await self._add_chat_message_delta(INTERRUPTION_CHAR, "assistant")

        if self._clear_queue is not None:
            # Clear any audio queued up by FastRTC's emit().
            self._clear_queue()
            
        logger.info("Clearing TTS output queue because bot was interrupted")
        self.output_queue = asyncio.Queue()  # Clear our own queue too

        # Push some silence to flush the Opus state
        await self.output_queue.put(
            (SAMPLE_RATE, np.zeros(SAMPLES_PER_FRAME, dtype=np.float32))
        )

        await self.output_queue.put(ora.UnmuteInterruptedByVAD())
        
        # Cancel current TTS task if running
        if self.tts_task and not self.tts_task.done():
            self.tts_task.cancel()
            
        # Cancel current LLM task if running  
        if self.llm_task and not self.llm_task.done():
            self.llm_task.cancel()
            
        # Reset events
        self.stt_finished_event.clear()
        self.llm_finished_event.clear()

    async def process_audio_input(self, audio_data: np.ndarray):
        """Process incoming audio data and send to STT."""
        if self.stt:
            await self.stt.send_audio(audio_data)
            self.n_samples_received += len(audio_data)

    async def send_text_to_tts(self, text: str):
        """Send text to TTS service."""
        if self.tts:
            await self.tts.send_text(text)

    async def get_output(self):
        """Get the next output event from the conversation."""
        try:
            return await asyncio.wait_for(self.output_queue.get(), timeout=0.1)
        except asyncio.TimeoutError:
            return None

    def get_debug_info(self):
        """Get debug information for the conversation."""
        self.debug_dict["conversation_state"] = self.chatbot.conversation_state()
        self.debug_dict["connection"]["stt"] = self.stt.state() if self.stt else "none"
        self.debug_dict["connection"]["tts"] = self.tts.state() if self.tts else "none"
        self.debug_dict["tts_voice"] = self.tts.voice if self.tts else "none"
        self.debug_dict["stt_pause_prediction"] = (
            self.stt.pause_prediction.value if self.stt else -1
        )
        
        return self.debug_dict

    def _get_gradio_update(self):
        """Get gradio update with current conversation state."""
        # Import GradioUpdate from a shared location to avoid circular imports
        # For now, define it locally to avoid import issues
        from pydantic import BaseModel
        
        class GradioUpdate(BaseModel):
            chat_history: list[dict[str, str]]
            debug_dict: dict[str, Any]
            debug_plot_data: list[dict]
        
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

    async def _check_for_bot_goodbye(self):
        """Check if bot said goodbye and close conversation if so."""
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

    async def detect_long_silence(self):
        """Handle situations where the user doesn't answer for a while."""
        if (
            self.chatbot.conversation_state() == "waiting_for_user"
            and (self.n_samples_received / SAMPLE_RATE - self.waiting_for_user_start_time)
            > USER_SILENCE_TIMEOUT
        ):
            logger.info("Long silence detected.")
            await self._add_chat_message_delta(USER_SILENCE_MARKER, "user")

    async def update_session(self, session: ora.SessionConfig):
        """Update session configuration."""
        if session.instructions:
            self.chatbot.set_instructions(session.instructions)

        if session.voice:
            self.tts_voice = session.voice

        if not session.allow_recording and self.recorder:
            await self.recorder.add_event("client", ora.SessionUpdate(session=session))
            await self.recorder.shutdown(keep_recording=False)
            self.recorder = None
            logger.info("Recording disabled for a session.")

    def audio_received_sec(self) -> float:
        """How much audio has been received in seconds."""
        return self.n_samples_received / self.input_sample_rate

    def set_clear_queue_callback(self, clear_queue_fn):
        """Set the callback to clear FastRTC's audio queue."""
        self._clear_queue = clear_queue_fn

    async def cleanup(self):
        """Clean up all connections and tasks."""
        logger.info(f"Cleaning up conversation {self.conversation_id}")
        
        # Signal shutdown to all loops
        self.shutdown_event.set()
        
        # Cancel all tasks
        tasks_to_cancel = []
        if self.stt_task and not self.stt_task.done():
            tasks_to_cancel.append(self.stt_task)
        if self.tts_task and not self.tts_task.done():
            tasks_to_cancel.append(self.tts_task)
        if self.llm_task and not self.llm_task.done():
            tasks_to_cancel.append(self.llm_task)
            
        for task in tasks_to_cancel:
            task.cancel()
            
        # Wait for tasks to complete cancellation
        if tasks_to_cancel:
            await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
        
        # Close service connections
        try:
            if self.stt:
                await self.stt.shutdown()
        except Exception as e:
            logger.warning(f"Error shutting down STT: {e}")
            
        try:
            if self.tts:
                await self.tts.shutdown()
        except Exception as e:
            logger.warning(f"Error shutting down TTS: {e}")
            
        # Clean up recorder
        if self.recorder:
            await self.recorder.shutdown()
            
        logger.info(f"Conversation {self.conversation_id} cleanup completed")


class ConversationManager:
    """
    Manages multiple Conversation instances, one per client connection.
    """

    def __init__(self):
        self.conversations: dict[str, Conversation] = {}
        self._lock = asyncio.Lock()
        
    async def create_conversation(self, conversation_id: str | None = None) -> Conversation:
        """Create a new conversation and start its services."""
        async with self._lock:
            conversation = Conversation(conversation_id)
            self.conversations[conversation.conversation_id] = conversation
            await conversation.start()
            logger.info(f"Created and started conversation {conversation.conversation_id}")
            return conversation
    
    async def get_conversation(self, conversation_id: str) -> Conversation | None:
        """Get an existing conversation by ID."""
        return self.conversations.get(conversation_id)
    
    async def remove_conversation(self, conversation_id: str):
        """Remove and clean up a conversation."""
        async with self._lock:
            conversation = self.conversations.pop(conversation_id, None)
            if conversation:
                await conversation.cleanup()
                logger.info(f"Removed conversation {conversation_id}")
    
    async def cleanup_all(self):
        """Clean up all conversations."""
        async with self._lock:
            cleanup_tasks = []
            for conversation in self.conversations.values():
                cleanup_tasks.append(conversation.cleanup())
            
            if cleanup_tasks:
                await asyncio.gather(*cleanup_tasks, return_exceptions=True)
            
            self.conversations.clear()
            logger.info("All conversations cleaned up")

    def get_conversation_count(self) -> int:
        """Get the number of active conversations."""
        return len(self.conversations)
