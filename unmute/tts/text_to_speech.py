import asyncio
import urllib.parse
from logging import getLogger
from typing import Annotated, Any, AsyncIterator, Callable, Literal, Union, cast

import msgpack
import websockets
from pydantic import BaseModel, Field, TypeAdapter

import unmute.openai_realtime_api_events as ora
from unmute import metrics as mt
from unmute.exceptions import MissingServiceAtCapacity
from unmute.kyutai_constants import (
    FRAME_TIME_SEC,
    HEADERS,
    SAMPLE_RATE,
    TEXT_TO_SPEECH_PATH,
    TTS_SERVER,
)
from unmute.recorder import Recorder
from unmute.service_discovery import ServiceWithStartup
from unmute.timer import Stopwatch
from unmute.tts.realtime_queue import RealtimeQueue
from unmute.tts.voice_cloning import voice_embeddings_cache
from unmute.websocket_utils import WebsocketState

logger = getLogger(__name__)


class TTSClientTextMessage(BaseModel):
    """Message sent to the TTS server saying we to turn this text into speech."""

    type: Literal["Text"] = "Text"
    text: str


class TTSClientVoiceMessage(BaseModel):
    type: Literal["Voice"] = "Voice"
    embeddings: list[float]
    shape: list[int]


class TTSClientEosMessage(BaseModel):
    """Message sent to the TTS server saying we are done sending text."""

    type: Literal["Eos"] = "Eos"


class TTSQueuedEosMessage(BaseModel):
    """Special EOS message that can be queued with a timestamp."""

    type: Literal["QueuedEos"] = "QueuedEos"


TTSClientMessage = Annotated[
    Union[TTSClientTextMessage, TTSClientVoiceMessage, TTSClientEosMessage],
    Field(discriminator="type"),
]
TTSClientMessageAdapter = TypeAdapter(TTSClientMessage)


class TTSTextMessage(BaseModel):
    type: Literal["Text"]
    text: str
    start_s: float
    stop_s: float


class TTSAudioMessage(BaseModel):
    type: Literal["Audio"]
    pcm: list[float]


class TTSErrorMessage(BaseModel):
    type: Literal["Error"]
    message: str


class TTSReadyMessage(BaseModel):
    type: Literal["Ready"]


TTSMessage = Annotated[
    Union[TTSTextMessage, TTSAudioMessage, TTSErrorMessage, TTSReadyMessage],
    Field(discriminator="type"),
]
TTSMessageAdapter = TypeAdapter(TTSMessage)


def url_escape(value: object) -> str:
    return urllib.parse.quote(str(value), safe="")


# Only release the audio such that it's AUDIO_BUFFER_SEC ahead of real time.
# If the value it's too low, it might cause stuttering.
# If it's too high, it's difficult to control the synchronization of the text and the
# audio, because that's controlled by emit() and WebRTC. Note that some
# desynchronization can still occur if the TTS is less than real-time, because WebRTC
# will decide to do some buffering of the audio on the fly.
AUDIO_BUFFER_SEC = FRAME_TIME_SEC * 4


def prepare_text_for_tts(text: str) -> str:
    text = text.strip()

    unpronounceable_chars = "*_`"
    for char in unpronounceable_chars:
        text = text.replace(char, "")

    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace(" : ", " ")

    return text


class TtsStreamingQuery(BaseModel):
    # See moshi-rs/moshi-server/
    seed: int | None = None
    temperature: float | None = None
    top_k: int | None = None
    format: str = "PcmMessagePack"
    voice: str | None = None
    voices: list[str] | None = None
    max_seq_len: int | None = None
    cfg_alpha: float | None = None
    auth_id: str | None = None

    def to_url_params(self) -> str:
        params = self.model_dump()
        return "?" + "&".join(
            f"{key}={url_escape(value)}"
            for key, value in params.items()
            if value is not None
        )


class TextToSpeech(ServiceWithStartup):
    def __init__(
        self,
        tts_base_url: str = TTS_SERVER,
        # For TTS, we do internal queuing, so we pass in the recorder to be able to
        # record the true time of the messages.
        recorder: Recorder | None = None,
        get_time: Callable[[], float] | None = None,
        voice: str | None = None,
    ):
        self.tts_base_url = tts_base_url
        self.recorder = recorder
        self.websocket: websockets.ClientConnection | None = None

        self.time_since_first_text_sent = Stopwatch(autostart=False)
        self.waiting_first_audio: bool = True
        # Number of samples received from the TTS server
        self.received_samples = 0
        # Number of samples that we passed on after waiting for the correct time
        self.received_samples_yielded = 0

        self.voice = voice
        self.query = TtsStreamingQuery(
            voice=self.voice
            # Don't pass in custom voices as a query parameter, we set it later using
            # a message
            if (self.voice and not self.voice.startswith("custom:"))
            else None,
            cfg_alpha=1.5,
        )

        # self.query_parameters = f"?voice={self.voice}&cfg_alpha=2&format=PcmMessagePack"
        self.text_output_queue = RealtimeQueue(get_time=get_time)
        
        # Track the last queued text message timestamp to properly schedule EOS
        self.last_text_message_stop_time: float | None = None

        self.shutdown_lock = asyncio.Lock()
        self.shutdown_complete = asyncio.Event()

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

    async def send(self, message: str | TTSClientMessage) -> None:
        """Send a message to the TTS server.

        Note that raw strings will be preprocessed to remove unpronounceable characters
        etc., but a TTSClientTextMessage will send the text as-is.
        """
        if isinstance(message, str):
            message = TTSClientTextMessage(
                type="Text", text=prepare_text_for_tts(message)
            )

        if self.shutdown_lock.locked():
            logger.warning("Can't send - TTS shutting down")
        elif not self.websocket:
            logger.warning("Can't send - TTS websocket not connected")
        else:
            if isinstance(message, TTSClientTextMessage):
                if message.text == "":
                    return  # Don't send empty messages

                mt.TTS_SENT_FRAMES.inc()
                self.time_since_first_text_sent.start_if_not_started()

            await self.websocket.send(msgpack.packb(message.model_dump()))

    def queue_eos(self) -> None:
        """Queue an EOS message to be sent after all currently queued text messages.
        
        This prevents the TTS server from shutting down before all text messages
        have been processed and released from the queue.
        """
        if self.last_text_message_stop_time is not None:
            # Schedule EOS to be sent 0.1 seconds after the last text message
            # This small buffer ensures all text has been processed
            eos_timestamp = self.last_text_message_stop_time + 0.1
            logger.info(f"Queuing EOS message at timestamp {eos_timestamp:.3f}s (after last text at {self.last_text_message_stop_time:.3f}s)")
            self.text_output_queue.put(TTSQueuedEosMessage(), eos_timestamp)
        else:
            # No text messages have been queued, send EOS immediately
            logger.info("No text messages queued, sending EOS immediately")
            asyncio.create_task(self._send_eos_immediately())

    async def _send_eos_immediately(self) -> None:
        """Send EOS message immediately to the TTS server."""
        await self.send(TTSClientEosMessage())
        

    async def start_up(self):

        url = self.tts_base_url + TEXT_TO_SPEECH_PATH + self.query.to_url_params()

        # For Modal services, connect to /ws instead of /api/tts_streaming
        if "modal.run" in self.tts_base_url:
            url = self.tts_base_url + "/ws" + self.query.to_url_params()
        else:
            url = self.tts_base_url + TEXT_TO_SPEECH_PATH
            
        logger.info(f"Connecting to TTS: {url}")
        self.websocket = await websockets.connect(
            url,
            additional_headers=HEADERS,
        )
        logger.debug("Connected to TTS")

        try:
            if self.voice is not None and self.voice.startswith("custom:"):
                voice_embedding = voice_embeddings_cache.get(self.voice)

                if voice_embedding is not None:
                    await self.websocket.send(voice_embedding)
                else:
                    logger.warning(
                        f"Custom voice {self.voice} not found, not sending it to TTS"
                    )

            for _ in range(10):
                # Due to some race condition in the TTS, we might get packets from a previous TTS client.
                message_bytes = await self.websocket.recv(decode=False)
                message_dict = msgpack.unpackb(message_bytes)
                message = TTSMessageAdapter.validate_python(message_dict)
                if isinstance(message, TTSReadyMessage):
                    return
                elif isinstance(message, TTSErrorMessage):
                    raise MissingServiceAtCapacity("tts")
                else:
                    logger.warning(
                        f"Received unexpected message type from {self.tts_base_url}, {message.type}"
                    )
        except Exception as e:
            logger.error(f"Error during TTS startup: {repr(e)}")
            # Make sure we don't leave a dangling websocket connection
            await self.websocket.close()
            self.websocket = None
            raise

        raise AssertionError("Not supposed to happen.")

    async def shutdown(self):
        async with self.shutdown_lock:
            if self.shutdown_complete.is_set():
                return
            mt.TTS_ACTIVE_SESSIONS.dec()
            mt.TTS_AUDIO_DURATION.observe(self.received_samples / SAMPLE_RATE)
            if self.time_since_first_text_sent.started:
                mt.TTS_GEN_DURATION.observe(self.time_since_first_text_sent.time())

            # Set before closing the websocket so that __aiter__ knows we're closing
            # the connection intentionally
            self.shutdown_complete.set()

            if self.websocket:
                await self.websocket.close()
                self.websocket = None

            logger.info("TTS shutdown() finished")

    async def __aiter__(self) -> AsyncIterator[TTSMessage]:
        if self.websocket is None:
            raise RuntimeError("TTS websocket not connected")
        mt.TTS_SESSIONS.inc()
        mt.TTS_ACTIVE_SESSIONS.inc()

        output_queue: RealtimeQueue[TTSMessage] = RealtimeQueue()
        
        # TTS Connection Health Monitoring
        last_message_time = asyncio.get_event_loop().time()
        message_count = 0
        last_health_check = last_message_time
        
        logger.info("=== TTS __aiter__() starting - connection established ===")

        try:
            async for message_bytes in self.websocket:
                current_time = asyncio.get_event_loop().time()
                message_count += 1
                last_message_time = current_time
                
                # Log health check every 10 seconds or every 50 messages
                if (current_time - last_health_check > 10.0) or (message_count % 50 == 0):
                    logger.info(f"TTS connection health: {message_count} messages received, last message {current_time - last_message_time:.1f}s ago, websocket state: {self.websocket.state.name}")
                    last_health_check = current_time
                # Handle different protocols for Modal vs local services
                if "modal.run" in self.tts_base_url:
                    # For Modal services, we get raw moshi-server protocol messages
                    # Try to parse as msgpack first for text messages, but skip raw audio bytes
                    try:
                        message_dict = msgpack.unpackb(cast(Any, message_bytes))
                        message: TTSMessage = TTSMessageAdapter.validate_python(message_dict)
                    except (msgpack.exceptions.ExtraData, msgpack.exceptions.UnpackException, ValueError):
                        # Raw bytes from moshi-server are PCM audio data that we should skip
                        # The original working version skipped these and it worked fine
                        logger.warning("Received unexpected message type from TTS: %s, value error: %s", message_bytes )
                        logger.debug(f"First 50 bytes: {message_bytes[:50]}")
                        continue
                else:
                    # For local services, expect msgpack format
                    message_dict = msgpack.unpackb(cast(Any, message_bytes))
                    message: TTSMessage = TTSMessageAdapter.validate_python(message_dict)

                if isinstance(message, TTSAudioMessage):
                    # Queue audio messages with proper timing to prevent fast/staticky playback
                    mt.TTS_RECV_FRAMES.inc()
                    if (
                        self.waiting_first_audio
                        and self.time_since_first_text_sent.started
                    ):
                        self.waiting_first_audio = False
                        ttft = self.time_since_first_text_sent.time()
                        mt.TTS_TTFT.observe(ttft)
                        logger.info("Time to first token is %.1f ms", ttft * 1000)
                    output_queue.start_if_not_started()
                    output_queue.put(
                        message, self.received_samples / SAMPLE_RATE - AUDIO_BUFFER_SEC
                    )
                    self.received_samples += len(message.pcm)

                    if self.recorder is not None:
                        await self.recorder.add_event(
                            "server",
                            ora.UnmuteResponseAudioDeltaReady(
                                number_of_samples=len(message.pcm)
                            ),
                        )

                elif isinstance(message, TTSTextMessage):
                    mt.TTS_RECV_WORDS.inc()
                    if message == TTSTextMessage(
                        type="Text", text="", start_s=0, stop_s=0
                    ):
                        # Always emitted by the TTS server, but we don't need it
                        continue

                    # There are two reasons why we don't send the text messages
                    # immediately:
                    # - The text messages have timestamps "from the future" because the
                    # audio stream is delayed by 2s.
                    # - Even so, we receive the audio/text faster than real time. It
                    #   seems difficult to keep track of how much audio data has already
                    #   been streamed (.emit() eats up the inputs immediately,
                    #   apparently it has some internal buffering) so we only send the
                    #   text messages at the real time when they're actually supposed to
                    #   be displayed. Precise timing/buffering is less important here.
                    # By using stop_s instead of start_s, we ensure that anything shown
                    # has already been said, so that if there's an interruption, the
                    # chat history matches what's actually been said.
                    output_queue.put(message, message.stop_s)
                    current_time = output_queue.get_time() if output_queue.start_time else 0
                    time_since_start = current_time - output_queue.start_time if output_queue.start_time else 0
                    logger.info(f"Queued TTSTextMessage '{message.text}' with stop_s={message.stop_s}, current_time_since_start={time_since_start:.3f}, queue size={len(output_queue.queue)}")
                    
                    # Track the latest text message timestamp for EOS scheduling
                    self.last_text_message_stop_time = message.stop_s

                for _, message in output_queue.get_nowait():
                    if isinstance(message, TTSAudioMessage):
                        self.received_samples_yielded += len(message.pcm)
                        yield message
                    elif isinstance(message, TTSQueuedEosMessage):
                        # Time to send the EOS message to the TTS server
                        logger.info("Processing queued EOS message - sending to TTS server")
                        await self.send(TTSClientEosMessage())
                    else:
                        yield message

        except websockets.ConnectionClosedOK:
            logger.info("=== TTS connection closed normally (ConnectionClosedOK) ===")
        except websockets.ConnectionClosedError as e:
            if self.shutdown_complete.is_set():
                # If we closed the websocket in shutdown(), it leads to this exception
                # (not sure why) but it's an intentional exit, so don't raise.
                logger.info("=== TTS connection closed during intentional shutdown ===")
            else:
                logger.error(f"=== TTS CONNECTION LOST UNEXPECTEDLY: {e} ===")
                current_time = asyncio.get_event_loop().time()
                connection_duration = current_time - (last_message_time if 'last_message_time' in locals() else current_time)
                logger.error(f"TTS server disconnected after {message_count if 'message_count' in locals() else 0} messages, connection was active for {connection_duration:.1f}s")
                raise
        except Exception as e:
            logger.error(f"=== TTS CONNECTION ERROR: {type(e).__name__}: {e} ===")
            logger.error(f"TTS failed after {message_count if 'message_count' in locals() else 0} messages")
            raise

        # Empty the queue if the connection is closed - we're releasing the messages
        # in real time, see above.
        remaining_messages = 0
        async for _, message in output_queue:
            if self.shutdown_complete.is_set():
                break
            if isinstance(message, TTSAudioMessage):
                self.received_samples_yielded += len(message.pcm)
            remaining_messages += 1
            yield message

        if remaining_messages > 0:
            logger.info(f"=== TTS __aiter__() finished - released {remaining_messages} remaining queued messages ===")
        else:
            logger.info("=== TTS __aiter__() finished - no remaining messages ===")
        await self.shutdown()
