"""
New UnmuteHandler that uses the Conversation architecture.
This replaces the quest_manager-based approach with persistent websocket connections.
"""

import asyncio
import logging
from pathlib import Path
from typing import Any

import numpy as np
from fastrtc import AsyncStreamHandler, AdditionalOutputs, CloseStream, wait_for_item

import unmute.openai_realtime_api_events as ora
from unmute.audio_input_override import AudioInputOverride
from unmute.conversation import Conversation, ConversationManager, UNINTERRUPTIBLE_BY_VAD_TIME_SEC
from unmute.kyutai_constants import SAMPLE_RATE
from unmute.timer import Stopwatch

# Audio input override for testing
AUDIO_INPUT_OVERRIDE: Path | None = None
DEBUG_PLOT_HISTORY_SEC = 10.0

# TTS debugging - set to a string to test TTS without microphone
TTS_DEBUGGING_TEXT = None

logger = logging.getLogger(__name__)

HandlerOutput = (
    tuple[int, np.ndarray] | AdditionalOutputs | ora.ServerEvent | CloseStream
)


class ConversationUnmuteHandler(AsyncStreamHandler):
    """
    New UnmuteHandler that uses the Conversation architecture instead of quest_manager.
    Each handler instance manages a single conversation with persistent connections.
    """

    def __init__(self, conversation_manager: ConversationManager) -> None:
        super().__init__(
            input_sample_rate=SAMPLE_RATE,
            # IMPORTANT! If set to a higher value, will lead to choppy audio. 🤷‍♂️
            output_frame_size=480,
            output_sample_rate=SAMPLE_RATE,
        )
        
        self.conversation_manager = conversation_manager
        self.conversation: Conversation | None = None
        
        # Audio input override for testing
        if AUDIO_INPUT_OVERRIDE is not None:
            self.audio_input_override = AudioInputOverride(AUDIO_INPUT_OVERRIDE)
        else:
            self.audio_input_override = None

    async def start_up(self):
        """Initialize the conversation and its services."""
        logger.info("Starting up ConversationUnmuteHandler")
        
        # Create a new conversation
        self.conversation = await self.conversation_manager.create_conversation()
        
        # Set the clear queue callback for interruptions
        self.conversation.set_clear_queue_callback(self._clear_queue)
        
        # Initialize waiting time now that we're ready to receive audio
        self.conversation.waiting_for_user_start_time = self.conversation.audio_received_sec()
        
        logger.info(f"Handler startup completed for conversation {self.conversation.conversation_id}")

    async def cleanup(self):
        """Clean up the conversation."""
        if self.conversation:
            await self.conversation_manager.remove_conversation(self.conversation.conversation_id)
            self.conversation = None

    async def handle_input_audio(self, audio: np.ndarray) -> None:
        """Handle incoming audio data."""
        if not self.conversation:
            return
            
        # Update sample count
        self.conversation.n_samples_received += len(audio)
        
        # Update debug info
        self.conversation.debug_dict["last_receive_time"] = self.conversation.audio_received_sec()
        
        # Add to debug plot data
        from fastrtc import audio_to_float32
        float_audio = audio_to_float32(audio)
        self.conversation.debug_plot_data.append(
            {
                "t": self.conversation.audio_received_sec(),
                "amplitude": float(np.sqrt((float_audio**2).mean())),
                "pause_prediction": self.conversation.stt.pause_prediction.value if self.conversation.stt else 0,
            }
        )

        # Handle bot speaking state
        if self.conversation.chatbot.conversation_state() == "bot_speaking":
            # Periodically update this not to trigger the "long silence" accidentally
            self.conversation.waiting_for_user_start_time = self.conversation.audio_received_sec()

        # Handle TTS debugging mode
        if TTS_DEBUGGING_TEXT is not None:
            assert self.audio_input_override is None, (
                "Can't use both TTS_DEBUGGING_TEXT and audio input override."
            )
            # Debugging mode: always send a fixed string when it's the user's turn
            if self.conversation.chatbot.conversation_state() == "waiting_for_user":
                logger.info("Using TTS debugging text. Ignoring microphone.")
                self.conversation.chatbot.chat_history.append(
                    {"role": "user", "content": TTS_DEBUGGING_TEXT}
                )
                await self.conversation._generate_response()
            return

        # Remove automatic initial response generation - wait for real user input
        # The system should only respond after receiving actual user audio/text input

        # Handle audio input override for testing
        if self.audio_input_override is not None:
            audio = self.audio_input_override.override(audio[np.newaxis, :])[0]

        # Clear timing debug info when user starts speaking
        if self.conversation.chatbot.conversation_state() == "user_speaking":
            self.conversation.debug_dict["timing"] = {}

        # Send audio to STT
        if self.conversation.stt:
            await self.conversation.stt.send_audio(audio)
            
            # Handle pause detection and flushing
            if self.conversation.stt_end_of_flush_time is None:
                await self.conversation.detect_long_silence()

                if self.conversation._determine_pause():
                    logger.info("Pause detected")
                    await self.conversation.output_queue.put(ora.InputAudioBufferSpeechStopped())

                    stt = self.conversation.stt
                    self.conversation.stt_end_of_flush_time = stt.current_time + stt.delay_sec
                    self.conversation.stt_flush_timer = Stopwatch()
                    
                    # Send silence frames to flush STT
                    import math
                    from unmute.kyutai_constants import FRAME_TIME_SEC, SAMPLES_PER_FRAME
                    num_frames = int(math.ceil(stt.delay_sec / FRAME_TIME_SEC)) + 1
                    zero = np.zeros(SAMPLES_PER_FRAME, dtype=np.float32)
                    for _ in range(num_frames):
                        await stt.send_audio(zero)
                        
                elif (
                    self.conversation.chatbot.conversation_state() == "bot_speaking"
                    and self.conversation.stt.pause_prediction.value < 0.4
                    and self.conversation.audio_received_sec() > UNINTERRUPTIBLE_BY_VAD_TIME_SEC
                ):
                    logger.info("Interruption by STT-VAD")
                    await self.conversation._interrupt_bot()
                    await self.conversation._add_chat_message_delta("", "user")
            else:
                # Handle STT flushing completion
                stt = self.conversation.stt
                if stt.current_time > self.conversation.stt_end_of_flush_time:
                    self.conversation.stt_end_of_flush_time = None
                    elapsed = self.conversation.stt_flush_timer.time()
                    rtf = stt.delay_sec / elapsed
                    logger.info(
                        "STT Flushing finished, took %.1f ms, RTF: %.1f", elapsed * 1000, rtf
                    )
                    await self.conversation._generate_response()

    async def emit(self) -> HandlerOutput | None:
        """Emit the next output event."""
        if not self.conversation:
            return None
            
        output_item = await wait_for_item(self.conversation.output_queue)

        if output_item is not None:
            return output_item
        else:
            current_time = self.conversation.audio_received_sec()
            if self.conversation.last_additional_output_update < current_time - 1:
                # If we have nothing to emit, at least update the debug dict
                self.conversation.last_additional_output_update = current_time
                return self.conversation._get_gradio_update()
            else:
                return None

    def copy(self):
        """Create a copy of this handler."""
        return ConversationUnmuteHandler(self.conversation_manager)

    async def __aenter__(self) -> None:
        """Enter the async context."""
        # No quest_manager context needed anymore
        pass

    async def __aexit__(self, *exc: Any) -> None:
        """Exit the async context."""
        # Cleanup is handled in the cleanup() method
        pass

    # Delegate methods to conversation
    async def update_session(self, session: ora.SessionConfig):
        """Update session configuration."""
        if self.conversation:
            await self.conversation.update_session(session)

    async def interrupt_bot(self):
        """Interrupt the bot."""
        if self.conversation:
            await self.conversation._interrupt_bot()

    def get_gradio_update(self):
        """Get gradio update."""
        if self.conversation:
            return self.conversation._get_gradio_update()
        return None

    @property
    def stt(self):
        """Get STT instance."""
        return self.conversation.stt if self.conversation else None

    @property
    def tts(self):
        """Get TTS instance."""
        return self.conversation.tts if self.conversation else None

    def audio_received_sec(self) -> float:
        """How much audio has been received in seconds."""
        if self.conversation:
            return self.conversation.audio_received_sec()
        return 0.0

    async def add_chat_message_delta(
        self,
        delta: str,
        role: str,
        generating_message_i: int | None = None,
    ) -> bool:
        """Add a chat message delta."""
        if self.conversation:
            return await self.conversation._add_chat_message_delta(
                delta, role, generating_message_i
            )
        return False

    async def receive(self, audio_data: tuple[int, np.ndarray]) -> None:
        """Receive audio data from FastRTC."""
        sample_rate, audio = audio_data
        assert sample_rate == SAMPLE_RATE
        
        # Flatten the audio if it has multiple channels
        if audio.ndim > 1:
            audio = audio.flatten()
            
        await self.handle_input_audio(audio)

    @property
    def recorder(self):
        """Get the recorder instance."""
        return self.conversation.recorder if self.conversation else None
