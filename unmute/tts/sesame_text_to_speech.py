"""
Sesame CSM-based Text-to-Speech service client that connects to Modal's Sesame_TTS_Service.
"""

import asyncio
import base64
import io
import logging
import wave
from typing import AsyncIterator, Callable

import numpy as np
import websockets
from pydantic import BaseModel

# Removed unused import
from unmute.kyutai_constants import HEADERS, SAMPLE_RATE
from unmute.recorder import Recorder
from unmute.service_discovery import ServiceWithStartup
from unmute.timer import Stopwatch
from unmute.tts.text_to_speech import TTSAudioMessage, TTSMessage, TTSTextMessage
from unmute.websocket_utils import WebsocketState

logger = logging.getLogger(__name__)


class SesameGenerateRequest(BaseModel):
    """Request to generate speech from text using Sesame CSM."""
    text: str
    speaker: int = 0
    context: list = []
    max_audio_length_ms: float = 10_000
    temperature: float = 0.9
    topk: int = 50


class SesameGenerateResponse(BaseModel):
    """Response from Sesame TTS service."""
    audio_base64: str | None = None
    sample_rate: int | None = None
    duration_ms: float | None = None
    text: str | None = None
    speaker: int | None = None
    success: bool
    error: str | None = None


class SesameTextToSpeech(ServiceWithStartup):
    """
    Sesame CSM-based TTS client that maintains a websocket connection 
    to the Modal Sesame_TTS_Service for the conversation lifetime.
    """

    def __init__(
        self,
        sesame_base_url: str,
        recorder: Recorder | None = None,
        get_time: Callable[[], float] | None = None,
        voice: str | None = None,
    ):
        self.sesame_base_url = sesame_base_url
        self.recorder = recorder
        self.websocket: websockets.ClientConnection | None = None
        
        # Voice configuration - map to speaker ID
        self.voice = voice
        self.speaker_id = 0  # Default speaker, could map voice to different speakers
        
        # Timing and metrics
        self.time_since_first_text_sent = Stopwatch(autostart=False)
        self.waiting_first_audio: bool = True
        self.received_samples = 0
        self.received_samples_yielded = 0
        
        # Conversation context for CSM
        self.conversation_context: list = []
        
        # Shutdown handling
        self.shutdown_lock = asyncio.Lock()
        self.shutdown_complete = asyncio.Event()
        
        # Queue for streaming text and audio
        self.output_queue: asyncio.Queue[TTSMessage] = asyncio.Queue()
        self.processing_task: asyncio.Task | None = None

    def state(self) -> WebsocketState:
        if not self.websocket:
            return "not_created"
        else:
            d: dict[websockets.protocol.State, WebsocketState] = {
                websockets.protocol.State.CONNECTING: "connecting",
                websockets.protocol.State.OPEN: "connected",
                websockets.protocol.State.CLOSING: "closing",
                websockets.protocol.State.CLOSED: "closed",
            }
            return d[self.websocket.state]

    async def start_up(self):
        """Initialize websocket connection to Sesame TTS service."""
        # For Modal services, connect to the websocket endpoint
        if "modal.run" in self.sesame_base_url:
            url = f"{self.sesame_base_url}/ws"
        else:
            url = f"{self.sesame_base_url}/ws"
            
        logger.info(f"Connecting to Sesame TTS: {url}")
        
        try:
            self.websocket = await websockets.connect(
                url,
                additional_headers=HEADERS,
            )
            logger.info("Connected to Sesame TTS service")
            
            # Start processing task for handling requests
            self.processing_task = asyncio.create_task(
                self._process_requests(), name="sesame_tts_processor"
            )
            
        except Exception as e:
            logger.error(f"Failed to connect to Sesame TTS: {e}")
            if self.websocket:
                await self.websocket.close()
                self.websocket = None
            raise

    async def shutdown(self):
        """Shutdown the Sesame TTS connection."""
        async with self.shutdown_lock:
            if self.shutdown_complete.is_set():
                return
                
            self.shutdown_complete.set()
            
            # Cancel processing task
            if self.processing_task and not self.processing_task.done():
                self.processing_task.cancel()
                try:
                    await self.processing_task
                except asyncio.CancelledError:
                    pass
            
            # Close websocket
            if self.websocket:
                await self.websocket.close()
                self.websocket = None
                
            logger.info("Sesame TTS shutdown completed")

    async def send(self, message: str) -> None:
        """Send text to be converted to speech."""
        if self.shutdown_complete.is_set():
            logger.warning("Can't send - Sesame TTS shutting down")
            return
            
        if not self.websocket:
            logger.warning("Can't send - Sesame TTS websocket not connected")
            return
            
        if message.strip() == "":
            return  # Don't send empty messages
            
        self.time_since_first_text_sent.start_if_not_started()
        
        # Create request for Sesame TTS
        request = SesameGenerateRequest(
            text=message,
            speaker=self.speaker_id,
            context=self.conversation_context.copy(),
            max_audio_length_ms=30_000,  # Allow longer responses for conversation
            temperature=0.9,
            topk=50,
        )
        
        # Send request via websocket
        try:
            await self.websocket.send(request.model_dump_json())
            logger.info(f"Sent text to Sesame TTS: '{message}'")
        except Exception as e:
            logger.error(f"Failed to send text to Sesame TTS: {e}")
            raise

    async def send_text(self, text: str) -> None:
        """Alias for send() method to maintain compatibility."""
        await self.send(text)

    def queue_eos(self) -> None:
        """Queue end-of-sequence marker."""
        # For Sesame TTS, we don't need explicit EOS since each request is complete
        logger.info("EOS queued for Sesame TTS (no-op)")

    async def _process_requests(self):
        """Process incoming responses from Sesame TTS service."""
        if not self.websocket:
            return
            
        try:
            async for message in self.websocket:
                if self.shutdown_complete.is_set():
                    break
                    
                try:
                    # Parse response
                    response_data = SesameGenerateResponse.model_validate_json(message)
                    
                    if not response_data.success:
                        logger.error(f"Sesame TTS error: {response_data.error}")
                        continue
                        
                    # Convert base64 audio to PCM
                    if response_data.audio_base64:
                        await self._process_audio_response(response_data)
                        
                except Exception as e:
                    logger.error(f"Error processing Sesame TTS response: {e}")
                    continue
                    
        except websockets.ConnectionClosedOK:
            logger.info("Sesame TTS connection closed normally")
        except websockets.ConnectionClosedError as e:
            if not self.shutdown_complete.is_set():
                logger.error(f"Sesame TTS connection lost: {e}")
                raise
        except Exception as e:
            logger.error(f"Sesame TTS processing error: {e}")
            raise

    async def _process_audio_response(self, response: SesameGenerateResponse):
        """Convert Sesame TTS response to TTSMessage format."""
        try:
            # Decode base64 audio
            if not response.audio_base64:
                logger.error("No audio data in Sesame TTS response")
                return
            audio_bytes = base64.b64decode(response.audio_base64)
            
            # Read WAV file from bytes
            with io.BytesIO(audio_bytes) as audio_buffer:
                with wave.open(audio_buffer, 'rb') as wav_file:
                    # Read audio data
                    frames = wav_file.readframes(wav_file.getnframes())
                    sample_rate = wav_file.getframerate()
                    channels = wav_file.getnchannels()
                    sample_width = wav_file.getsampwidth()
                    
                    # Convert to numpy array
                    if sample_width == 2:  # 16-bit
                        audio_array = np.frombuffer(frames, dtype=np.int16)
                    elif sample_width == 4:  # 32-bit
                        audio_array = np.frombuffer(frames, dtype=np.int32)
                    else:
                        raise ValueError(f"Unsupported sample width: {sample_width}")
                    
                    # Handle stereo to mono conversion
                    if channels == 2:
                        audio_array = audio_array.reshape(-1, 2).mean(axis=1)
                    
                    # Convert to float32 and normalize
                    if sample_width == 2:
                        audio_float = audio_array.astype(np.float32) / 32768.0
                    else:
                        audio_float = audio_array.astype(np.float32) / 2147483648.0
                    
                    # Resample if needed
                    if sample_rate != SAMPLE_RATE:
                        import librosa
                        audio_float = librosa.resample(
                            audio_float, 
                            orig_sr=sample_rate, 
                            target_sr=SAMPLE_RATE
                        )
            
            # First send text message
            if response.text:
                text_message = TTSTextMessage(
                    type="Text",
                    text=response.text,
                    start_s=0.0,
                    stop_s=len(audio_float) / SAMPLE_RATE,
                )
                await self.output_queue.put(text_message)
            
            # Then send audio message
            audio_message = TTSAudioMessage(
                type="Audio",
                pcm=audio_float.tolist(),
            )
            await self.output_queue.put(audio_message)
            
            # Update conversation context
            if response.text and len(audio_float) > 0:
                self.conversation_context.append({
                    'text': response.text,
                    'speaker': response.speaker or 0,
                    'audio': audio_float.tolist()
                })
                
                # Keep context manageable (last 5 exchanges)
                if len(self.conversation_context) > 5:
                    self.conversation_context = self.conversation_context[-5:]
            
            self.received_samples += len(audio_float)
            
            # Update timing metrics
            if self.waiting_first_audio and self.time_since_first_text_sent.started:
                self.waiting_first_audio = False
                ttft = self.time_since_first_text_sent.time()
                logger.info(f"Sesame TTS time to first token: {ttft * 1000:.1f}ms")
            
            logger.info(f"Processed Sesame TTS response: {len(audio_float)} samples, {len(audio_float) / SAMPLE_RATE:.2f}s")
            
        except Exception as e:
            logger.error(f"Error processing Sesame audio response: {e}")
            raise

    async def __aiter__(self) -> AsyncIterator[TTSMessage]:
        """Async iterator for TTS messages."""
        while not self.shutdown_complete.is_set():
            try:
                # Get message from output queue with timeout
                message = await asyncio.wait_for(
                    self.output_queue.get(), 
                    timeout=0.1
                )
                
                if isinstance(message, TTSAudioMessage):
                    self.received_samples_yielded += len(message.pcm)
                    
                yield message
                
            except asyncio.TimeoutError:
                # No message available, continue loop
                continue
            except Exception as e:
                if not self.shutdown_complete.is_set():
                    logger.error(f"Error in Sesame TTS iterator: {e}")
                    raise
                break
                
        logger.info("Sesame TTS iterator finished")
