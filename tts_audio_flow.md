# TTS Audio Flow Documentation

This document provides a comprehensive overview of the Text-to-Speech (TTS) audio pipeline from Orpheus generation through to frontend output, including all potential failure points and the specific code lines involved.

## Architecture Overview

The TTS audio flow consists of the following main components:

1. **Orpheus TTS Generation** (Modal-based)
2. **Backend Audio Processing** (WebSocket streaming)
3. **Frontend Audio Decoding & Playback** (Web Audio API)

## Complete Audio Flow

### 1. Orpheus TTS Generation (Modal)

#### Entry Point
**File**: `/workspace/unmute/tts/orpheus_modal.py`

**Modal App Initialization**:
```python
# Line 11: Create Modal app
orpheus_tts_app = modal.App("orpheus-tts")

# Lines 30-42: Modal class definition with GPU resources
@orpheus_tts_app.cls(
    image=orpheus_image,
    gpu="H100",
    timeout=10 * 60,  # 10 minutes
    volumes={"/cache": model_volume},
    secrets=[modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])],
    min_containers=int(os.environ.get("MIN_CONTAINERS", "0")),
    scaledown_window=600,  # 10 minutes
)
```

**Model Loading**:
```python
# Lines 47-71: Model initialization
@modal.enter()
def initialize(self):
    from orpheus_tts import OrpheusModel
    model_name = os.environ.get("ORPHEUS_MODEL_NAME", "canopylabs/orpheus-tts-0.1-finetune-prod")
    self.model = OrpheusModel(model_name=model_name)
```

**Audio Generation**:
```python
# Lines 74-130: Non-streaming generation
@modal.method()
def generate_speech(self, text: str, voice: str = "tara", response_format: str = "wav") -> bytes:
    # Line 87-90: Generate speech tokens
    syn_tokens = self.model.generate_speech(
        prompt=text,
        voice=voice,
    )
    
    # Lines 95-116: Create WAV file with proper headers
    with wave.open(audio_buffer, "wb") as wf:
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(24000)  # 24kHz sample rate
        
        for audio_chunk in syn_tokens:
            wf.writeframes(audio_chunk)
```

**Streaming Generation**:
```python
# Lines 132-171: Streaming generation method
@modal.method()
def generate_speech_stream(self, text: str, voice: str = "tara") -> Iterator[bytes]:
    # Lines 144-147: Generate streaming tokens
    syn_tokens = self.model.generate_speech(
        prompt=text,
        voice=voice,
    )
    
    # Lines 152-162: Stream raw audio chunks
    for audio_chunk in syn_tokens:
        yield audio_chunk  # Raw 16-bit PCM at 24kHz
```

### 2. Backend TTS Service Integration

#### Service Initialization
**File**: `/workspace/unmute/tts/text_to_speech.py`

**Orpheus TTS Adapter**:
```python
# Lines 434-469: OrpheusTextToSpeech class initialization
class OrpheusTextToSpeech(ServiceWithStartup):
    def __init__(self, tts_base_url: str = TTS_SERVER, recorder: Recorder | None = None, 
                 get_time: Callable[[], float] | None = None, orpheus_service_instance = None):
        self.orpheus_service_instance = orpheus_service_instance
        self.voice = "tara"  # Always use tara voice
        self.audio_queue: asyncio.Queue[TTSMessage] = asyncio.Queue()
```

**Modal Connection Setup**:
```python
# Lines 481-504: Connection to Modal service
async def start_up(self):
    if self.orpheus_service_instance is not None:
        self.modal_function = self.orpheus_service_instance.generate_speech_stream
    else:
        from .orpheus_modal import OrpheusTTS
        self.modal_function = OrpheusTTS().generate_speech_stream
```

#### Text Processing and Audio Generation
**File**: `/workspace/unmute/tts/text_to_speech.py`

**Complete Text Sending**:
```python
# Lines 506-548: Send complete text for generation
async def send_complete_text(self, text: str) -> None:
    # Line 521: Preprocess text
    text = prepare_text_for_tts(text)
    
    # Lines 540-542: Start streaming generation task
    self.current_generation_task = asyncio.create_task(
        self._stream_audio_from_modal(text)
    )
```

**Audio Streaming from Modal**:
```python
# Lines 550-638: Stream audio chunks from Modal
async def _stream_audio_from_modal(self, text: str) -> None:
    # Lines 562-568: Call Modal streaming function
    for audio_chunk in self.modal_function.remote_gen(
        text=text,
        voice=self.voice
    ):
        # Lines 584-602: Convert raw bytes to TTSAudioMessage
        audio_data = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
        audio_message = TTSAudioMessage(
            type="Audio",
            pcm=audio_data.tolist()
        )
        
        # Line 616: Queue the audio message
        await self.audio_queue.put(audio_message)
```

### 3. Conversation Management

#### Conversation Service Integration
**File**: `/workspace/unmute/conversation.py`

**TTS Initialization**:
```python
# Lines 139-158: Initialize Orpheus TTS connection
async def _init_tts(self, generating_message_i: int):
    factory = partial(
        OrpheusTextToSpeech,
        recorder=self.recorder,
        get_time=lambda: self.n_samples_received / SAMPLE_RATE,
    )
    self.tts = await find_instance("tts", factory)
    
    # Lines 152-154: Start TTS task
    self.tts_task = asyncio.create_task(
        self._tts_loop(generating_message_i), name=f"orpheus_tts_loop_{self.conversation_id}"
    )
```

**TTS Message Processing Loop**:
```python
# Lines 217-255: TTS message processing
async def _tts_loop(self, generating_message_i: int):
    async for message in self.tts:
        if isinstance(message, TTSAudioMessage):
            # Line 235: Process audio message
            logger.info(f"Processing TTSAudioMessage with {len(message.pcm)} samples")
            
            # Lines 248-253: Send to output queue
            await self.output_queue.put(
                ora.ResponseAudioDelta(
                    delta=base64.b64encode(opus_bytes).decode("utf-8")
                )
            )
```

### 4. WebSocket Communication

#### Main WebSocket Handler
**File**: `/workspace/unmute/main_websocket.py`

**Audio Message Transmission**:
```python
# Lines 495-503: Convert audio to Opus and send via WebSocket
audio = audio_to_float32(audio)
opus_bytes = await asyncio.to_thread(opus_writer.append_pcm, audio)
if opus_bytes:
    to_emit = ora.ResponseAudioDelta(
        delta=base64.b64encode(opus_bytes).decode("utf-8"),
    )

# Lines 520-521: Send to WebSocket
if isinstance(to_emit, ora.ResponseAudioDelta):
    await websocket.send_text(to_emit.model_dump_json())
```

**WebSocket Event Definition**:
**File**: `/workspace/unmute/openai_realtime_api_events.py`

```python
# Lines 135-136: Response audio delta event
class ResponseAudioDelta(BaseEvent[Literal["response.audio.delta"]]):
    delta: str  # Base64-encoded Opus audio data
```

### 5. Frontend Audio Processing

#### WebSocket Message Reception
**File**: `/workspace/frontend/src/app/Unmute.tsx`

**Audio Delta Processing**:
```typescript
// Lines 180-185: Handle incoming audio delta messages
if (data.type === "response.audio.delta") {
  const opus = base64DecodeOpus(data.delta);
  console.log(`=== FRONTEND: Received audio delta, opus size: ${opus.length} bytes ===`);
  const ap = audioProcessor.current;
  if (!ap) return;
  
  // Line 186: Send to decoder worker
  ap.decoder.postMessage({ command: "decode", pages: [opus] });
}
```

**Base64 Decoding**:
**File**: `/workspace/frontend/src/app/audioUtil.ts`

```typescript
// Lines 10-18: Decode base64 Opus data
export const base64DecodeOpus = (base64String: string): Uint8Array => {
  const binaryString = window.atob(base64String);
  const len = binaryString.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes;
};
```

#### Audio Processing Setup
**File**: `/workspace/frontend/src/app/useAudioProcessor.ts`

**Audio Context and Decoder Setup**:
```typescript
// Lines 36-77: Setup audio processing pipeline
const audioContext = new AudioContext();
const outputWorklet = await getAudioWorkletNode(audioContext, "audio-output-processor");
const decoder = new Worker("/decoderWorker.min.js");

// Lines 60-70: Decoder message handling
decoder.onmessage = (event: MessageEvent<any>) => {
  const frame = event.data[0];
  outputWorklet.port.postMessage({
    frame: frame,
    type: "audio",
    micDuration: micDuration,
  });
};

// Lines 71-77: Decoder initialization
decoder.postMessage({
  command: "init",
  bufferLength: (960 * audioContext.sampleRate) / 24000,
  decoderSampleRate: 24000,
  outputBufferSampleRate: audioContext.sampleRate,
  resampleQuality: 0,
});
```

#### Audio Output Processing
**File**: `/workspace/frontend/public/audio-output-processor.js`

**Audio Worklet Processor**:
```javascript
// Lines 17-102: AudioOutputProcessor class definition
class AudioOutputProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    // Lines 23-36: Buffer configuration
    this.initialBufferSamples = 1 * frameSize;
    this.partialBufferSamples = asSamples(10);
    this.maxBufferSamples = asSamples(DEFAULT_MAX_BUFFER_MS);
    
    // Lines 41-101: Message handling for incoming audio frames
    this.port.onmessage = (event) => {
      let frame = event.data.frame;
      this.frames.push(frame);
      if (this.currentSamples() >= this.initialBufferSamples && !this.started) {
        this.start();
      }
    };
  }
  
  // Lines 164-237: Audio processing and output
  process(inputs, outputs) {
    const output = outputs[0][0];
    if (!this.canPlay()) {
      return true;
    }
    
    // Lines 188-210: Copy audio frames to output buffer
    while (out_idx < output.length && this.frames.length) {
      const first = this.frames[0];
      const to_copy = Math.min(
        first.length - this.offsetInFirstBuffer,
        output.length - out_idx
      );
      const subArray = first.subarray(
        this.offsetInFirstBuffer,
        this.offsetInFirstBuffer + to_copy
      );
      output.set(subArray, out_idx);
    }
  }
}
```

## Potential Failure Points

Based on the codebase analysis and existing documentation, here are the critical failure points that could prevent audio from reaching the frontend:

### 1. Modal/Orpheus TTS Failures

#### Model Loading Issues
**Location**: `/workspace/unmute/tts/orpheus_modal.py:64-71`
```python
try:
    self.model = OrpheusModel(model_name=model_name)
except Exception as e:
    print(f"Error loading Orpheus model: {e}")
    raise
```

**Potential Failures**:
- HuggingFace authentication failures (missing HF_TOKEN)
- Model download timeouts
- GPU memory allocation failures
- Model compatibility issues

#### Modal Function Call Failures
**Location**: `/workspace/unmute/tts/text_to_speech.py:562-568`
```python
for audio_chunk in self.modal_function.remote_gen(
    text=text,
    voice=self.voice
):
```

**Potential Failures**:
- Modal container scaling issues
- Network connectivity to Modal
- Modal timeout (10 minute limit)
- Container resource exhaustion

### 2. Backend Service Failures

#### TTS Service Discovery
**Location**: `/workspace/unmute/conversation.py:148`
```python
self.tts = await find_instance("tts", factory)
```

**Potential Failures**:
- Service discovery timeout
- TTS service not available
- Connection refused errors

#### Audio Processing Pipeline
**Location**: `/workspace/unmute/tts/text_to_speech.py:584-602`
```python
# Audio data conversion failure
audio_data = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
```

**Potential Failures**:
- Invalid audio chunk format
- Buffer size misalignment (not multiple of 2 bytes)
- NumPy conversion errors

#### Message Queue Failures
**Location**: `/workspace/unmute/tts/text_to_speech.py:616`
```python
await self.audio_queue.put(audio_message)
```

**Potential Failures**:
- Queue overflow
- Memory exhaustion
- Task cancellation during shutdown

### 3. WebSocket Communication Failures

#### Connection Issues
**Location**: `/workspace/unmute/main_websocket.py:520-521`
```python
if isinstance(to_emit, ora.ResponseAudioDelta):
    await websocket.send_text(to_emit.model_dump_json())
```

**Potential Failures**:
- WebSocket connection closed
- Network connectivity issues
- Message serialization errors
- Client disconnection

#### Audio Encoding Failures
**Location**: `/workspace/unmute/main_websocket.py:496-502`
```python
opus_bytes = await asyncio.to_thread(opus_writer.append_pcm, audio)
if opus_bytes:
    to_emit = ora.ResponseAudioDelta(
        delta=base64.b64encode(opus_bytes).decode("utf-8"),
    )
```

**Potential Failures**:
- Opus encoding errors
- Base64 encoding failures
- PCM format issues

### 4. Frontend Processing Failures

#### WebSocket Message Handling
**Location**: `/workspace/frontend/src/app/Unmute.tsx:180-186`
```typescript
if (data.type === "response.audio.delta") {
  const opus = base64DecodeOpus(data.delta);
  ap.decoder.postMessage({ command: "decode", pages: [opus] });
}
```

**Potential Failures**:
- JSON parsing errors
- Base64 decoding failures
- Decoder worker not initialized
- Message queue overflow

#### Audio Context Issues
**Location**: `/workspace/frontend/src/app/useAudioProcessor.ts:134`
```typescript
audioProcessorRef.current.audioContext.resume();
```

**Potential Failures**:
- Audio context suspended (browser policy)
- User interaction required for audio
- Audio permissions denied
- Device audio issues

#### Decoder Worker Failures
**Location**: `/workspace/frontend/public/decoderWorker.min.js`

**Potential Failures**:
- Worker initialization failures
- Opus decoding errors
- Memory allocation issues
- Sample rate conversion problems

#### Audio Output Processing
**Location**: `/workspace/frontend/public/audio-output-processor.js:164-237`

**Potential Failures**:
- Buffer underrun/overflow
- Sample rate mismatches
- Audio worklet not supported
- Frame processing errors

### 5. Known Critical Issues

#### TTS Server Premature Shutdown
**Source**: `/workspace/progress.md:88-102`

**Issue**: TTS server (moshi-server) exits after completing `batch_idx=0`
```
TTS moshi-server: tts finished batch_idx=0
TTS moshi-server: send loop exited  
TTS moshi-server: recv loop exited
```

**Impact**: Complete message flow stoppage after first batch
**Detection**: Message flow watchdog alerts after 10+ seconds
**Location**: Service restarts after 65 seconds causing long delays

#### RealtimeQueue Timing Issues
**Source**: `/workspace/progress.md:32-42`

**Issue**: Messages can get stuck in RealtimeQueue if timing calculations drift
**Location**: `/workspace/unmute/tts/text_to_speech.py:387`
```python
for _, message in output_queue.get_nowait():
    if isinstance(message, TTSAudioMessage):
        yield message
```

**Mitigation**: Timeout mechanism implemented to force-release stuck messages

#### STT Audio Delay After Interruption
**Source**: `/workspace/stt_delay_issue.md:28-57`

**Issue**: 10-second delay in audio processing after VAD interruption
**Root Cause**: Synchronous TTS websocket shutdown blocks audio processing
**Location**: `/workspace/unmute/conversation_handler.py` (interrupt_bot method)

## Monitoring and Debugging

### TTS Message Flow Watchdog
**Location**: `/workspace/unmute/conversation_handler.py`

Monitors TTS message flow and alerts when no messages received for 10+ seconds:
```python
# Enhanced monitoring detects TTS server failures
=== TTS MESSAGE FLOW WATCHDOG ALERT ===
No TTS messages received for 15.2s
=== SUSPECTED TTS SERVER FAILURE - No messages for 35.1s ===
```

### Connection Health Monitoring
**Location**: `/workspace/unmute/tts/text_to_speech.py:299-314`

Tracks TTS connection health and message flow:
```python
# Log health check every 10 seconds or every 50 messages
if (current_time - last_health_check > 10.0) or (message_count % 50 == 0):
    logger.info(f"TTS connection health: {message_count} messages received")
```

### Error Recovery Mechanisms

1. **Service Discovery Retry**: Automatic retry with exponential backoff
2. **Message Queue Timeout**: Force-release stuck messages after 5 seconds
3. **Connection Monitoring**: Detect and alert on connection failures
4. **Graceful Degradation**: Continue processing when services are unavailable

## Summary

The TTS audio flow is a complex pipeline involving:
1. **Orpheus Modal TTS** generating raw PCM audio
2. **Backend services** processing and streaming audio
3. **WebSocket communication** transmitting base64-encoded Opus
4. **Frontend decoding** and Web Audio API playback

Critical failure points include Modal container issues, service discovery failures, WebSocket disconnections, and frontend audio context problems. The most critical known issue is TTS server premature shutdown after processing single batches, which causes complete audio flow stoppage.

Comprehensive monitoring and error recovery mechanisms are in place to detect and mitigate these failures, but the core TTS server stability issue requires fixing the batch processing configuration or implementing faster restart mechanisms.