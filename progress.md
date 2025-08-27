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
- 🔄 Next: Implement timing debugging and timeout mechanism
