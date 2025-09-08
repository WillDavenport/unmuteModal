# Audio Flow Debugging System

This document explains how to use the comprehensive audio flow debugging system that has been implemented to track audio messages from Orpheus TTS generation through to the frontend.

## Overview

The audio debugging system tracks audio messages through these stages:

1. **Orpheus Modal TTS** → Raw PCM chunks generated
2. **TTS Service Audio Queue** → TTSAudioMessage objects queued
3. **TTS Service Iterator** → TTSAudioMessage objects yielded to conversation
4. **Conversation Output Queue** → Audio tuples sent to WebSocket handler
5. **WebSocket Emit Loop** → Opus-encoded base64 strings sent to frontend
6. **Frontend Reception** → Messages received and processed by browser

## Key Features

### 1. Comprehensive Stage Tracking
- **Modal chunks received**: Raw audio chunks from Orpheus Modal
- **Audio messages queued**: Processed messages in TTS service queue
- **Audio messages yielded**: Messages sent from TTS to conversation
- **Output queue messages**: Messages added to conversation output queue
- **WebSocket messages sent**: Messages transmitted to frontend
- **Frontend messages received**: Messages processed by browser (requires frontend logging)

### 2. Data Volume Tracking
- Total bytes received from Modal
- Total audio samples processed
- Total Opus bytes sent via WebSocket

### 3. Dropoff Analysis
- Automatically identifies stages with significant message loss (>10%)
- Calculates drop percentages between each stage
- Highlights problematic stages with ⚠️ warnings

### 4. Error Tracking
- Conversion errors (PCM processing failures)
- Queue timeouts (TTS service timeouts)
- WebSocket errors (transmission failures)

## How to Use

### 1. Debug Logging in Logs

The system automatically logs detailed information at each stage with `=== AUDIO_DEBUG:` prefixes:

```
=== AUDIO_DEBUG: Generation request #1 ===
=== AUDIO_DEBUG: Received raw chunk 1 from Modal: 1024 bytes ===
=== AUDIO_DEBUG: Queued audio chunk 1 with 512 samples to audio_queue (queue size: 1) ===
=== AUDIO_DEBUG: Yielding audio message with 512 samples (yielded count: 1) ===
=== AUDIO_DEBUG: Processing TTSAudioMessage with 512 samples ===
=== AUDIO_DEBUG: Putting audio data to output queue: 512 samples at 24000Hz (output_queue size: 1) ===
=== AUDIO_DEBUG: Sending audio to realtime websocket: 256 opus bytes from 512 PCM samples ===
=== AUDIO_DEBUG: Sending ResponseAudioDelta to websocket: 344 base64 chars (~256 opus bytes) ===
```

### 2. Periodic Debug Summaries

The system automatically logs comprehensive summaries:
- Every 10 TTS messages received
- Every 5 seconds during active generation
- At the end of each generation

Example summary:
```
=== COMPREHENSIVE AUDIO FLOW DEBUG SUMMARY ===
Stage Counts:
  modal_chunks_received: 25
  audio_messages_queued: 25
  audio_messages_yielded: 22
  output_queue_messages: 22
  websocket_messages_sent: 20
  frontend_messages_received: 18

Dropoff Analysis:
  Modal → TTS Queue: 25 → 25 (0.0% drop) ✅ Normal
  TTS Queue → TTS Yield: 25 → 22 (12.0% drop) ⚠️  SIGNIFICANT DROP
  TTS Yield → Output Queue: 22 → 22 (0.0% drop) ✅ Normal
  Output Queue → WebSocket: 22 → 20 (9.1% drop) ✅ Normal
  WebSocket → Frontend: 20 → 18 (10.0% drop) ✅ Normal
=== END AUDIO FLOW DEBUG SUMMARY ===
```

### 3. Frontend Debug Logging

Frontend logging shows audio reception and processing:

```javascript
=== FRONTEND_AUDIO_DEBUG: Received audio delta, opus size: 256 bytes, base64 length: 344 ===
=== FRONTEND_AUDIO_DEBUG: Sending opus data to decoder worker ===
=== FRONTEND_AUDIO_DEBUG: Decoder worker returned PCM frame with 960 samples ===
=== FRONTEND_AUDIO_DEBUG: Sent PCM frame to output worklet ===
```

### 4. Manual Debug Summary

You can trigger a manual debug summary by calling the TTS service's debug method:

```python
# In conversation or handler code
if hasattr(self.tts, 'log_debug_summary'):
    self.tts.log_debug_summary()
```

## Interpreting the Results

### Normal Operation
- All stages should have similar message counts
- Drop percentages should be < 10%
- No significant errors in error tracking

### Common Issues and Diagnosis

#### 1. High Drop at "Modal → TTS Queue"
- **Symptom**: Many chunks from Modal but few queued
- **Possible Causes**: 
  - PCM conversion errors
  - Buffer size issues
  - TTS service queue overflow
- **Check**: Look for conversion errors in logs

#### 2. High Drop at "TTS Queue → TTS Yield"
- **Symptom**: Messages queued but not yielded
- **Possible Causes**:
  - TTS service timeout/hang
  - Async iterator blocking
  - Service shutdown during generation
- **Check**: Look for timeout errors, check if generation task is still running

#### 3. High Drop at "Output Queue → WebSocket"
- **Symptom**: Messages in output queue but not sent
- **Possible Causes**:
  - WebSocket connection issues
  - Opus encoding failures
  - FastRTC buffering issues
- **Check**: WebSocket connection state, Opus encoding errors

#### 4. High Drop at "WebSocket → Frontend"
- **Symptom**: Messages sent but not received by frontend
- **Possible Causes**:
  - Network connectivity issues
  - Browser WebSocket problems
  - Frontend audio processor not initialized
- **Check**: Browser dev console, network tab, audio processor state

### 5. Audio Cutoff Scenarios

#### Mid-Sentence Cutoff
If audio cuts off mid-sentence, look for:
- Sudden drop to 0 in message counts
- TTS service stopping unexpectedly
- WebSocket disconnection
- Frontend audio processor errors

#### Gradual Degradation
If audio quality degrades over time:
- Increasing queue timeouts
- Growing conversion errors
- Opus encoding issues
- Buffer overflow/underflow

## Code Locations

### Backend Debug Logging
- **TTS Service**: `/workspace/unmute/tts/text_to_speech.py`
- **Conversation**: `/workspace/unmute/conversation.py`
- **WebSocket**: `/workspace/unmute/main_websocket.py`
- **Debug Tracker**: `/workspace/unmute/audio_debug.py`

### Frontend Debug Logging
- **WebSocket Reception**: `/workspace/frontend/src/app/Unmute.tsx`
- **Audio Processing**: `/workspace/frontend/src/app/useAudioProcessor.ts`

## Troubleshooting Tips

### 1. Enable More Detailed Logging
Set log level to DEBUG for more detailed information:
```python
import logging
logging.getLogger('unmute.tts.text_to_speech').setLevel(logging.DEBUG)
logging.getLogger('unmute.conversation').setLevel(logging.DEBUG)
```

### 2. Check Service Health
Monitor the health of each service:
```python
# Check TTS service state
if self.tts:
    logger.info(f"TTS service state: {self.tts.state()}")
    logger.info(f"Current generation task: {self.tts.current_generation_task}")
```

### 3. Monitor Queue Sizes
Watch for queue buildup:
```python
logger.info(f"Audio queue size: {self.tts.audio_queue.qsize()}")
logger.info(f"Output queue size: {self.conversation.output_queue.qsize()}")
```

### 4. Frontend Console Monitoring
Open browser dev console and watch for:
- Audio debug messages
- WebSocket connection status
- Audio processor initialization
- Decoder worker messages

## Performance Impact

The debug logging system is designed to have minimal performance impact:
- Uses INFO level logging (can be disabled in production)
- Lightweight counters and timers
- Asynchronous logging operations
- Optional detailed summaries

To reduce logging in production:
```python
# Set higher log level to reduce verbosity
logging.getLogger('unmute').setLevel(logging.WARNING)
```

## Example Debug Session

Here's a typical debug session flow:

1. **Start Generation**: Look for generation request log
2. **Monitor Modal**: Watch for raw chunks being received
3. **Check Queuing**: Verify chunks are being converted and queued
4. **Watch Yielding**: Confirm messages are being yielded to conversation
5. **Track Output**: See messages added to output queue
6. **Monitor WebSocket**: Verify Opus encoding and transmission
7. **Check Frontend**: Confirm reception and decoding

If audio cuts off, the logs will show exactly where the flow stops, allowing you to focus debugging efforts on the problematic stage.

## Future Enhancements

Potential improvements to the debug system:
- Real-time dashboard showing message flow
- Automatic alerting on significant drops
- Performance metrics (latency, throughput)
- Historical tracking across sessions
- Integration with monitoring systems (Grafana/Prometheus)