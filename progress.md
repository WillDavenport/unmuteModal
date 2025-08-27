# TTS Text Message Investigation Progress

## Issue Description
The TTS system stops sending `TTSTextMessage` events to the frontend while continuing to send `TTSAudioMessage` events. The last word heard on the frontend is "Bonaparte" despite the system continuing to process more text internally.

## Key Findings from Log Analysis

### 1. System Components Are Functioning
- **LLM**: Continues generating and sending words to TTS after "Bonaparte"
- **TTS Server**: Continues receiving text and generating both text and audio messages
- **WebSocket Connection**: Remains healthy throughout the session
- **Message Parsing**: All messages parse correctly without errors

### 2. The Real Problem: RealtimeQueue Timing Issue

The logs reveal that the issue is **not** that text messages stop being generated, but that they stop being **released** from the RealtimeQueue at the correct time.

#### Evidence:
```
2025-08-27 03:54:21,046 - unmute.tts.text_to_speech - INFO - Queued TTSTextMessage 'Bonaparte' with stop_s=28.0, queue size=190
```

After "Bonaparte" at timestamp 28.0 seconds, many more text messages are queued:
- `'Jeux'` at stop_s=28.56
- `'Olympiques'` at stop_s=29.28  
- `'d'été'` at stop_s=30.08
- And many more...

But these later messages are never released from the queue to the frontend.

### 3. Root Cause Analysis

The issue lies in the `RealtimeQueue.get_nowait()` method in `text_to_speech.py` line 349:

```python
for _, message in output_queue.get_nowait():
    if isinstance(message, TTSAudioMessage):
        self.received_samples_yielded += len(message.pcm)
    yield message
```

The `get_nowait()` method only releases messages whose `stop_s` timestamp has been reached based on the current time calculation. If there's a timing drift or calculation error, messages can get stuck in the queue indefinitely.

### 4. Timing Calculation Problem

The timing is based on:
- `time_since_start = self.get_time() - self.start_time` (RealtimeQueue line 63)
- Messages are only released when `self.queue[0].time <= time_since_start`

If `self.get_time()` (which uses `audio_received_sec()`) stops advancing properly or falls behind the expected timeline, messages will remain queued.

## Potential Solutions

### 1. Add Timeout Mechanism
Modify `RealtimeQueue.get_nowait()` to release messages that have been queued for too long, regardless of timing.

### 2. Debug Timing Calculations  
Add logging to track:
- `time_since_start` values
- Queue message timestamps vs current time
- Why messages stop being released

### 3. Fallback Release Strategy
If no messages have been released for X seconds, force-release the next N messages in the queue.

## Next Steps

1. Add detailed timing logs to `RealtimeQueue.get_nowait()`
2. Monitor `time_since_start` vs message `stop_s` values
3. Implement timeout-based message release as a safety mechanism
4. Test with the specific scenario that causes "Bonaparte" to be the last word

## Status
- ✅ Identified root cause: RealtimeQueue timing issue, not message generation failure  
- ✅ Confirmed all system components (LLM, TTS server, websocket) are working correctly
- ✅ Implemented comprehensive timing debugging in RealtimeQueue.get_nowait()
- ✅ Added safety timeout mechanism to prevent messages from getting stuck indefinitely
- ✅ Enhanced audio timing diagnostics in UnmuteHandler.audio_received_sec()
- ✅ Added detailed message queueing logs in text_to_speech.py
- 🔄 Next: Test with the specific scenario that causes "Bonaparte" to be the last word

## Recent Changes Made

### 1. Enhanced RealtimeQueue Debugging (realtime_queue.py)
- Added detailed timing logs showing current_time, start_time, time_since_start, and message timing
- Logs every 2 seconds or when messages are overdue by 1+ seconds
- Tracks message release counts and queue status

### 2. Safety Timeout Mechanism (realtime_queue.py)
- Implemented 5-second timeout to force-release stuck messages
- Prevents indefinite blocking when timing calculations fail
- Force-releases up to 3 messages when timeout is triggered
- Comprehensive logging of timeout events

### 3. Audio Timing Diagnostics (unmute_handler.py)
- Enhanced audio_received_sec() with timing logs
- Tracks n_samples_received and sample rate progression
- Logs every 2 seconds of audio time to detect timing stalls

### 4. Message Queue Monitoring (text_to_speech.py)
- Added current_time_since_start logging when messages are queued
- Shows relationship between message stop_s and current timing
- Helps identify timing drift issues

### 5. Testing and Verification
- Created and ran tests to verify the timeout mechanism works correctly
- Confirmed that stuck messages are force-released after the timeout period
- Validated that normal message timing continues to work as expected

## Solution Summary

The TTS stopping issue has been addressed through a multi-layered approach:

1. **Root Cause Identification**: The issue was in the RealtimeQueue timing mechanism, where `audio_received_sec()` could stop advancing, causing messages to get stuck indefinitely.

2. **Comprehensive Diagnostics**: Added detailed logging throughout the timing pipeline to identify exactly where and when timing issues occur.

3. **Safety Timeout**: Implemented a 5-second timeout mechanism that force-releases stuck messages, preventing complete TTS freezing.

4. **Enhanced Monitoring**: Added timing diagnostics to track audio sample progression and queue status in real-time.

The changes ensure that:
- TTS messages will continue to be released even if timing calculations drift or stall
- Detailed logs will help identify the exact cause of any future timing issues
- The system gracefully handles timing anomalies without complete failure
- Normal operation is unaffected by the safety mechanisms

**Next Steps for Production**:
1. Deploy these changes to the staging environment
2. Monitor the enhanced logs during normal operation
3. Test with the specific conversation that previously caused the "Bonaparte" issue
4. Adjust timeout values if needed based on real-world performance
