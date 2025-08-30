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

## Next Steps - Updated Priority

### Immediate Actions Required:
1. **Investigate TTS Server Crashes**: Determine why moshi-server exits after completing `batch_idx=0`
2. **Add TTS Connection Monitoring**: Detect when TTS server goes offline
3. **Implement Fast TTS Restart**: Reduce 65-second restart time to <5 seconds  
4. **Add Error Handling**: Proper logging and recovery when TTS server fails

### Previous Actions (Now Completed):
1. ✅ Add detailed timing logs to `RealtimeQueue.get_nowait()`
2. ✅ Monitor `time_since_start` vs message `stop_s` values  
3. ✅ Implement timeout-based message release as a safety mechanism
4. ✅ Test with the specific scenario - **REAL ISSUE DISCOVERED**

## Status - CRITICAL NEW FINDINGS
- ✅ Identified root cause: RealtimeQueue timing issue, not message generation failure  
- ✅ Confirmed all system components (LLM, TTS server, websocket) are working correctly
- ✅ Implemented comprehensive timing debugging and timeout mechanism
- ✅ **DISCOVERED THE REAL ISSUE**: TTS Server shutdown causes complete message flow stoppage

## Latest Investigation Results (August 28, 2025)

### Critical Discovery: TTS Server Premature Shutdown

The new logs reveal the **actual root cause** of the TTS stopping issue. The user reported hearing "Or maybe you're looking" as the last words, and the logs show:

#### Timeline of Events:
1. **22:01:49-22:01:50**: System processes text messages normally:
   - `'tonight?'`, `'me'`, `'asking,'`, `'of'`, `'course.'`, `'Or'`, `'maybe'`, `'you're'`, `'looking'`
   - All messages are queued and released properly with timing: `time_since_start=8.182`

2. **22:01:50.898**: **TTS SERVER SHUTS DOWN UNEXPECTEDLY**:
   ```
   TTS moshi-server: tts finished batch_idx=0
   TTS moshi-server: send loop exited  
   TTS moshi-server: recv loop exited
   ```

3. **22:01:50.931**: Last audio message (#131) processed
4. **22:01:51+**: No more TTS messages generated - only audio timing logs continue
5. **22:02:14**: TTS server restarts after 65 seconds, but conversation has moved on

#### Key Evidence:
- **Queue was functioning perfectly**: Messages were being released normally with 121 items remaining
- **Timing was accurate**: `time_since_start=8.182` vs `next_message_time=8.160` (only 0.022s difference)
- **No timeout mechanism triggered**: System never reached the 5-second timeout threshold
- **Audio processing continued**: `audio_received_sec()` kept advancing normally for minutes

### Root Cause Analysis - Updated

The issue is **NOT** a RealtimeQueue timing problem. The issue is:

1. **TTS Server Premature Shutdown**: The moshi-server process terminates unexpectedly after processing a batch
2. **No Error Handling**: When TTS server exits, the system continues running but stops receiving new TTS messages
3. **Silent Failure**: No error logs indicate the TTS connection was lost
4. **Long Restart Time**: 65 seconds to restart TTS server while user waits

### Why Previous Fixes Didn't Work

The timeout mechanism and timing diagnostics we implemented were solving the wrong problem:
- The RealtimeQueue timing was working correctly
- Messages weren't getting "stuck" - they simply stopped being generated
- The 5-second timeout wouldn't help when the TTS server itself is down

## Actual Solution Required

### 1. TTS Server Stability
- Investigate why moshi-server exits after `batch_idx=0`
- Add TTS server health monitoring
- Implement automatic restart without 65-second delay

### 2. Connection Monitoring  
- Detect when TTS server connection is lost
- Alert the system when no TTS messages received for X seconds
- Attempt immediate reconnection

### 3. Graceful Degradation
- Continue processing queued messages when TTS server is down
- Provide user feedback when TTS is temporarily unavailable
- Buffer text for replay when TTS server reconnects

### 4. Error Handling
- Log TTS server disconnections as errors, not info messages
- Implement retry logic for TTS server communication
- Add circuit breaker pattern for TTS failures

## Enhanced Debugging Capabilities Added (August 28, 2025)

### Comprehensive TTS Monitoring System Implemented

Based on the root cause analysis showing TTS server premature shutdown, I've added extensive logging and monitoring to catch this issue early and provide detailed diagnostics:

#### 1. TTS Connection Health Monitoring (`text_to_speech.py`)
- **Connection lifecycle tracking**: Logs when TTS connections start, health checks every 10 seconds or 50 messages
- **Message flow monitoring**: Tracks total messages received with timing information
- **Connection state logging**: Records websocket state and connection duration
- **Enhanced error handling**: Distinguishes between normal shutdown vs unexpected disconnections

#### 2. TTS Message Flow Watchdog (`unmute_handler.py`)
- **Real-time flow monitoring**: Tracks text vs audio message counts and timing
- **Stall detection**: Alerts when no TTS messages received for 10+ seconds
- **Server failure detection**: Flags suspected TTS server crashes after 30+ seconds of silence
- **Message type tracking**: Separates text and audio message statistics

#### 3. Enhanced Service Discovery Logging (`service_discovery.py`)
- **TTS-specific connection diagnostics**: Detailed logging for TTS connection attempts
- **Failure pattern detection**: Identifies quick failures (server not running) vs timeouts (server hanging)
- **Connection timing analysis**: Tracks connection attempt duration and success rates
- **Restart scenario detection**: Logs patterns that indicate server restart cycles

#### 4. Timing Diagnostics Enhancement (`realtime_queue.py`)
- **Queue timing validation**: Already had comprehensive timing logs and timeout mechanism
- **Message release monitoring**: Tracks when messages are released vs queued
- **Overdue message detection**: Identifies messages stuck in queue beyond expected timing

### New Log Messages to Watch For:

#### TTS Server Shutdown Detection:
```
=== TTS MESSAGE FLOW WATCHDOG ALERT ===
No TTS messages received for 15.2s
=== SUSPECTED TTS SERVER FAILURE - No messages for 35.1s ===
This suggests the TTS server may have crashed or exited unexpectedly
```

#### TTS Connection Issues:
```
=== TTS CONNECTION FAILURE ===
TTS instance: ws://localhost:8002
Error type: ConnectionRefusedError
=== QUICK FAILURE - TTS SERVER MAY NOT BE RUNNING ===
```

#### Server Restart Detection:
```
=== TTS CONNECTION SUCCESSFUL ===
TTS instance: ws://localhost:8002
Connection time: 45.2ms
Total discovery time: 65432.1ms  # Long discovery time indicates restart
```

### Expected Benefits:

1. **Early Detection**: Watchdog will alert within 10 seconds when TTS message flow stops
2. **Root Cause Identification**: Enhanced logging will distinguish between:
   - TTS server crashes/exits
   - Network connectivity issues  
   - Service discovery problems
   - Queue timing issues
3. **Faster Debugging**: Detailed connection lifecycle logs will show exactly when and why TTS fails
4. **Performance Monitoring**: Connection timing data will help identify performance degradation

### Next Steps for Testing:

1. **Reproduce the Issue**: Run the system with these enhanced logs to capture the exact failure sequence
2. **Analyze New Logs**: Look for the new watchdog alerts and connection failure patterns
3. **Identify Fix Strategy**: Based on detailed logs, implement appropriate solution (faster restart, connection pooling, etc.)

The enhanced logging should now provide complete visibility into the TTS server shutdown issue that was previously silent.

## Latest Log Analysis - August 28, 2025 (Session 2)

### New Session Analysis: "Qu'est-ce qui t'a amené" Cutoff

The user reported hearing "Qu'est-ce qui t'a amené" as the last words in a new session. Analysis of the logs reveals the **EXACT SAME PATTERN** as the previous session:

#### Timeline of Events (22:24:00-22:24:04):

1. **22:24:00-22:24:03**: System processes French text messages normally:
   - Queue processes: `'Comment'`, `'ça'`, `'va'`, `'?'`, `'Est-ce'`, `'que'`, `'je'`, `'aujourd'hui'`, `'peux'`, `'te'`, `'vous'`, `'aider'`, `'?'`, `'avec'`, `'quelque'`, `'chose'`, `'?'`, `'Qu'est-ce'`, `'qui'`, `'t'a'`, `'amené'`
   - All messages are queued and released properly through message #97
   - RealtimeQueue timing working correctly: `time_since_start=6.068` vs `next_message_time=6.080`

2. **22:24:03.709**: **IDENTICAL TTS SERVER SHUTDOWN**:
   ```
   TTS moshi-server: tts finished batch_idx=0
   TTS moshi-server: send loop exited  
   TTS moshi-server: recv loop exited
   ```

3. **22:24:03.704**: Last TTS message (#97) processed with text "amené"
4. **22:24:04+**: Complete TTS message flow stoppage - only audio timing logs continue
5. **22:24:15**: TTS Message Flow Watchdog triggers: "No TTS messages received for 11.3s"
6. **22:24:20**: Second watchdog alert: "No TTS messages received for 16.3s"

#### Critical Confirmation:

This **definitively confirms** our root cause analysis:

1. **Consistent Server Behavior**: TTS server exits with identical log pattern in both sessions
2. **Timing System Working**: RealtimeQueue timing calculations are accurate (only 0.012s difference)
3. **Queue Processing Normal**: 87 messages in queue, proper release timing maintained
4. **Watchdog System Working**: New monitoring correctly detects TTS message flow stoppage after 11.3s

### Pattern Analysis: TTS Server "batch_idx=0" Exit

Both sessions show the TTS server completing `batch_idx=0` and then exiting:
- **Session 1**: "looking" was last word before server exit
- **Session 2**: "amené" was last word before server exit  
- **Common Pattern**: Server processes one complete batch then terminates loops

This suggests the TTS server is designed to process single batches but may be misconfigured for continuous operation, or there's a bug causing premature termination after the first batch completion.

### Enhanced Monitoring System Validation

The new watchdog system is working as designed:
- **Detection Time**: 11.3 seconds (within expected 10-15 second range)
- **Alert Frequency**: Every 5 seconds after initial detection
- **Message Tracking**: Accurately reports total=97, text=17, audio=80
- **Connection State**: Correctly identifies TTS connection as "connected" (websocket still open)

### Updated Root Cause

The issue is **confirmed** to be:
1. **TTS Server Single-Batch Limitation**: moshi-server exits after completing first batch
2. **No Automatic Restart**: System doesn't detect server exit or restart quickly
3. **Silent Failure Mode**: Connection remains "connected" but no new messages flow

### Immediate Action Items

1. **Investigate moshi-server Configuration**: 
   - Check if server is configured for continuous operation vs single-batch mode
   - Review batch processing parameters and exit conditions

2. **Implement TTS Server Process Monitoring**:
   - Monitor moshi-server process health, not just websocket connection
   - Detect process exit and restart immediately

3. **Add TTS Server Restart Logic**:
   - Automatic restart when message flow stops for >10 seconds
   - Fast restart without 65-second service discovery delay

### Status: Root Cause Definitively Confirmed
- ✅ **TTS Server Premature Exit**: Confirmed in two separate sessions
- ✅ **Timing System Working**: RealtimeQueue operates correctly  
- ✅ **Monitoring System Functional**: Watchdog detects issues within 11 seconds
- ✅ **Next Step Clear**: Fix TTS server batch processing or implement fast restart
