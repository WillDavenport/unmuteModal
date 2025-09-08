# Simplified TTS Audio Pipeline Implementation

This document describes the implementation of the simplified TTS audio pipeline as proposed in the original specification.

## Overview

The simplified pipeline reduces latency and complexity by eliminating intermediate queues and implementing immediate interrupt handling with minimal buffering.

## Architecture Changes

### Backend Changes

#### 1. New SimplifiedOrpheusTextToSpeech Class
- **File**: `unmute/tts/text_to_speech.py`
- **Key Features**:
  - Ring buffer with configurable size (default: 200ms)
  - Immediate Opus encoding when buffer reaches packet size (40ms)
  - Direct interrupt handling with buffer flush
  - Minimal queuing for encoding granularity only

#### 2. Ring Buffer Implementation
```python
# Ring buffer for PCM samples (collections.deque for efficient operations)
self.ring_buffer = collections.deque(maxlen=self.ring_buffer_max_samples)

# Encode packets when we have enough samples
while len(self.ring_buffer) >= self.packet_size_samples:
    packet_samples = np.array([self.ring_buffer.popleft() for _ in range(self.packet_size_samples)])
    opus_bytes = await asyncio.to_thread(self.opus_encoder.append_pcm, packet_samples)
```

#### 3. Interrupt Handling
```python
async def interrupt(self, reason: str = "user_interrupt") -> None:
    """Interrupt current generation and flush buffers."""
    self.is_interrupted = True
    self.interrupt_event.set()
    
    # Cancel current generation
    if self.current_generation_task and not self.current_generation_task.done():
        self.current_generation_task.cancel()
    
    # Flush ring buffer
    async with self.buffer_lock:
        self.ring_buffer.clear()
    
    # Send interrupt message
    await self.output_queue.put(SimplifiedTTSMessage(type="interrupted", reason=reason))
```

### Frontend Changes

#### 1. Audio Worklet Updates
- **File**: `frontend/public/audio-output-processor.js`
- **Changes**:
  - Reduced max buffer from 60 seconds to 120ms
  - Reduced initial buffer for faster audio start
  - Added flush command support for immediate interrupt handling

#### 2. WebSocket Message Handling
- **File**: `frontend/src/app/Unmute.tsx`
- **New Messages**:
  - `response.audio.start`: Audio generation started
  - `response.interrupted`: Audio interrupted, flush buffers

### Control Messages

#### 1. New Event Types
- **File**: `unmute/openai_realtime_api_events.py`
- **Added**:
  - `ResponseAudioStart`: Marks beginning of audio response
  - `ResponseInterrupted`: Signals audio interruption

#### 2. Message Flow
```
response.audio.start → Audio generation begins
response.audio.delta → Opus audio data (as before)
response.interrupted → Immediate buffer flush
response.audio.done → Audio generation complete
```

## Configuration

### Environment Variables
```bash
# Enable simplified pipeline
SIMPLIFIED_TTS_ENABLED=true

# Buffer configuration
SIMPLIFIED_TTS_RING_BUFFER_MS=200        # Backend ring buffer
SIMPLIFIED_TTS_OPUS_PACKET_MS=40         # Opus packet size
SIMPLIFIED_TTS_FRONTEND_INITIAL_MS=80    # Frontend initial buffer
SIMPLIFIED_TTS_FRONTEND_TARGET_MS=100    # Frontend target buffer
SIMPLIFIED_TTS_FRONTEND_MAX_MS=120       # Frontend max buffer
```

### Code Configuration
```python
from unmute.simplified_tts_config import should_use_simplified_tts

if should_use_simplified_tts():
    # Use SimplifiedOrpheusTextToSpeech
else:
    # Use legacy OrpheusTextToSpeech
```

## Key Improvements

### 1. Reduced Latency
- **Before**: Multiple queues with seconds of buffering
- **After**: ≤200ms backend buffer + ≤120ms frontend buffer = ≤320ms total

### 2. Faster Interrupts
- **Before**: Wait for queue drain (multiple seconds)
- **After**: Immediate buffer flush (≤320ms)

### 3. Simplified Architecture
- **Removed**: RealtimeQueue, cadence schedulers, timing shapers
- **Kept**: Opus encoding, WebSocket transport, basic jitter buffer

### 4. Maintained Compatibility
- Same API surface (`send_complete_text`, `interrupt`)
- Same Opus encoding and transport
- Same frontend decoder pipeline

## Performance Characteristics

### Buffer Sizes
| Component | Legacy | Simplified | Improvement |
|-----------|--------|------------|-------------|
| Backend Queue | Unlimited | 200ms | 99%+ reduction |
| Frontend Buffer | 60 seconds | 120ms | 99.7% reduction |
| Interrupt Latency | 2-10 seconds | 320ms | 85-95% reduction |

### Audio Quality
- Same Opus encoding (24kHz, variable bitrate)
- Same 40ms packetization for efficiency
- Minimal impact on audio quality due to reduced buffering

## Migration Guide

### 1. Enable Simplified Pipeline
Set environment variable:
```bash
export SIMPLIFIED_TTS_ENABLED=true
```

### 2. Adjust Buffer Sizes (Optional)
```bash
export SIMPLIFIED_TTS_RING_BUFFER_MS=150  # Reduce for lower latency
export SIMPLIFIED_TTS_FRONTEND_MAX_MS=100  # Reduce for faster interrupts
```

### 3. Monitor Performance
- Check logs for "Simplified Orpheus TTS" initialization
- Monitor interrupt latency in audio debug logs
- Verify audio quality with reduced buffering

### 4. Rollback if Needed
```bash
export SIMPLIFIED_TTS_ENABLED=false
```

## Testing

### 1. Run Test Script
```bash
python test_simplified_tts.py
```

### 2. Manual Testing
1. Start conversation with simplified TTS enabled
2. Test audio quality and latency
3. Test interrupt responsiveness (speak while bot is talking)
4. Monitor logs for buffer management

### 3. Load Testing
- Test with multiple concurrent conversations
- Monitor memory usage (should be lower due to smaller buffers)
- Test interrupt handling under load

## Troubleshooting

### Common Issues

#### 1. Audio Dropouts
- **Cause**: Ring buffer too small for network jitter
- **Solution**: Increase `SIMPLIFIED_TTS_RING_BUFFER_MS`

#### 2. Slow Interrupts
- **Cause**: Frontend buffer too large
- **Solution**: Decrease `SIMPLIFIED_TTS_FRONTEND_MAX_MS`

#### 3. Audio Quality Issues
- **Cause**: Opus packet size too small
- **Solution**: Increase `SIMPLIFIED_TTS_OPUS_PACKET_MS` (multiples of 20ms)

### Debug Logging
Enable detailed logging:
```python
import logging
logging.getLogger('unmute.tts').setLevel(logging.DEBUG)
```

## Future Enhancements

### 1. Adaptive Buffering
- Adjust buffer sizes based on network conditions
- Monitor underruns and automatically increase buffers

### 2. Quality Metrics
- Track interrupt latency statistics
- Monitor buffer utilization
- Measure audio quality metrics

### 3. Advanced Interrupt Handling
- Predictive interruption based on VAD confidence
- Graceful audio fade-out on interrupt
- Smart resume after false interrupts

## Implementation Status

- ✅ Backend ring buffer implementation
- ✅ Interrupt handling with buffer flush
- ✅ Frontend worklet flush support
- ✅ Control message support
- ✅ Configuration system
- ✅ Legacy compatibility
- ✅ Documentation and testing

The simplified TTS pipeline is ready for testing and deployment with the `SIMPLIFIED_TTS_ENABLED=true` flag.