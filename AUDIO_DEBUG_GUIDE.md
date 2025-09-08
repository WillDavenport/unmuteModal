# Audio Flow Debugging Guide

This guide provides comprehensive debugging tools and logging to trace audio cutoffs in the TTS pipeline from Orpheus generation through to frontend playback.

## Problem Description

Audio from TTS is sometimes being cut off mid-sentence on the frontend. This could be happening at several stages:

1. **Orpheus Modal TTS Generation** - Audio generation stops prematurely
2. **Backend Audio Queue** - Audio chunks get lost in the OrpheusTextToSpeech adapter
3. **Conversation Output Queue** - Audio doesn't make it to the conversation output
4. **WebSocket Emission** - Audio fails to be encoded/transmitted to frontend
5. **Frontend Processing** - Audio fails to decode or play on the frontend

## Debug Logging Added

Comprehensive debug logging has been added at each stage with `=== STAGE: message ===` format:

### 1. Orpheus Modal TTS (`/workspace/unmute/tts/orpheus_modal.py`)
- `=== ORPHEUS MODAL: Starting streaming generation ===`
- `=== ORPHEUS MODAL: Time to first token: X.XXXs ===`
- `=== ORPHEUS MODAL: Yielding chunk N: X bytes ===`
- `=== ORPHEUS MODAL: Completed streaming generation ===`

### 2. OrpheusTextToSpeech Adapter (`/workspace/unmute/tts/text_to_speech.py`)
- `=== ORPHEUS TTS: Starting Modal streaming generation ===`
- `=== ORPHEUS TTS: Received raw chunk N: X bytes from Modal ===`
- `=== ORPHEUS TTS: Queuing audio chunk N with X samples ===`
- `=== ORPHEUS TTS: Yielding audio message #N with X samples ===`

### 3. Conversation Pipeline (`/workspace/unmute/conversation.py`)
- `=== CONVERSATION TTS: Processing TTSAudioMessage with X samples ===`
- `=== CONVERSATION TTS: Putting audio data to output queue ===`
- `=== CONVERSATION TTS: First audio message received and queued to output ===`

### 4. WebSocket Emission (`/workspace/unmute/main_websocket.py`)
- `=== WEBSOCKET EMIT: Received audio tuple: sample_rate=X, audio_length=X ===`
- `=== WEBSOCKET EMIT: Got opus bytes from encoder: X bytes ===`
- `=== WEBSOCKET EMIT: Sending ResponseAudioDelta to frontend via WebSocket ===`

### 5. Frontend Processing (`/workspace/frontend/src/app/Unmute.tsx` & `useAudioProcessor.ts`)
- `=== FRONTEND: Received audio delta at HH:MM:SS.mmm, opus size: X bytes ===`
- `=== FRONTEND DECODER: Decoded audio frame at HH:MM:SS.mmm, length: X samples ===`
- `=== FRONTEND DECODER: Sent decoded frame to output worklet ===`

## Debug Script Usage

The debug script `/workspace/debug_audio_flow.py` tests each stage independently:

### Test Individual Stages

```bash
# Test only Orpheus Modal generation
python debug_audio_flow.py --test-orpheus-only --test-text "Your test message here"

# Test only the OrpheusTextToSpeech adapter
python debug_audio_flow.py --test-adapter-only --test-text "Your test message here"

# Test only the conversation pipeline
python debug_audio_flow.py --test-conversation-only --test-text "Your test message here"

# Test only WebSocket emission simulation
python debug_audio_flow.py --test-websocket-only
```

### Test Complete Pipeline

```bash
# Run all tests in sequence
python debug_audio_flow.py --test-full-pipeline --test-text "Your test message here"

# Run with default test text
python debug_audio_flow.py
```

## Debug Output Analysis

The debug script provides detailed metrics for each stage:

- **Chunks/Messages Received**: How many audio chunks were processed
- **Total Time**: End-to-end processing time
- **Time to First Token (TTFT)**: Latency to first audio output
- **Audio Duration**: Expected duration of generated audio
- **Real-time Factor**: Processing speed vs. audio duration (lower is better)

## Debugging Workflow

### Step 1: Run Full Test Suite
```bash
python debug_audio_flow.py --test-full-pipeline
```

Look for which stages are failing or producing incomplete results.

### Step 2: Isolate the Problem Stage
If the full test reveals issues, run individual stage tests:

```bash
# If Orpheus generation is suspected
python debug_audio_flow.py --test-orpheus-only

# If backend processing is suspected  
python debug_audio_flow.py --test-adapter-only

# If conversation handling is suspected
python debug_audio_flow.py --test-conversation-only
```

### Step 3: Monitor Live System
With the enhanced logging in place, run your normal application and watch the logs:

1. **Backend Logs**: Look for the `=== STAGE: ===` messages in your backend logs
2. **Frontend Console**: Open browser dev tools and watch for frontend debug messages
3. **Modal Logs**: Check Modal dashboard for Orpheus container logs

### Step 4: Identify Cutoff Patterns

Look for these patterns that indicate where cutoffs occur:

#### Pattern 1: Orpheus Generation Stops Early
```
=== ORPHEUS MODAL: Starting streaming generation ===
=== ORPHEUS MODAL: Yielding chunk 1: 1024 bytes ===
=== ORPHEUS MODAL: Yielding chunk 2: 1024 bytes ===
# No more chunks - generation stopped prematurely
```

#### Pattern 2: Backend Queue Processing Stops
```
=== ORPHEUS TTS: Received raw chunk 1: 1024 bytes from Modal ===
=== ORPHEUS TTS: Queuing audio chunk 1 with 512 samples ===
=== ORPHEUS TTS: Received raw chunk 2: 1024 bytes from Modal ===
# No queuing of chunk 2 - backend processing failed
```

#### Pattern 3: Conversation Output Stops
```
=== CONVERSATION TTS: Processing TTSAudioMessage with 512 samples ===
=== CONVERSATION TTS: Putting audio data to output queue ===
# No more conversation messages - TTS loop ended early
```

#### Pattern 4: WebSocket Emission Stops
```
=== WEBSOCKET EMIT: Received audio tuple: sample_rate=24000, audio_length=512 ===
=== WEBSOCKET EMIT: Got opus bytes from encoder: 128 bytes ===
# No WebSocket send - emission failed
```

#### Pattern 5: Frontend Processing Stops
```
=== FRONTEND: Received audio delta at 14:30:15.123, opus size: 128 bytes ===
=== FRONTEND DECODER: Decoded audio frame at 14:30:15.125, length: 512 samples ===
# No more frontend messages - decoding or playback failed
```

## Common Issues and Solutions

### Issue 1: Orpheus Modal Container Timeout
**Symptoms**: Generation stops after a few chunks, Modal logs show timeout
**Solution**: Increase Modal timeout or optimize text length

### Issue 2: Audio Queue Overflow
**Symptoms**: Backend logs show queue size growing, then stops processing
**Solution**: Increase queue size or add backpressure handling

### Issue 3: WebSocket Connection Drops
**Symptoms**: WebSocket emit logs stop, frontend shows connection closed
**Solution**: Add connection retry logic, check network stability

### Issue 4: Frontend Audio Context Suspended
**Symptoms**: Frontend receives data but no audio plays
**Solution**: Ensure user interaction to resume audio context

### Issue 5: Opus Encoding/Decoding Errors
**Symptoms**: Audio data present but encoding/decoding fails
**Solution**: Validate PCM format, check sample rates match

## Performance Monitoring

Monitor these key metrics during debugging:

1. **Time to First Token (TTFT)**: Should be < 2 seconds
2. **Real-time Factor**: Should be < 1.0x for real-time processing
3. **Queue Sizes**: Should remain bounded, not grow indefinitely
4. **WebSocket Message Rate**: Should match expected audio chunk rate
5. **Frontend Decode Rate**: Should match incoming audio delta rate

## Next Steps

After identifying the cutoff location:

1. **Add targeted fixes** for the specific stage causing issues
2. **Implement retry logic** for transient failures
3. **Add monitoring alerts** for production deployment
4. **Consider fallback mechanisms** for graceful degradation

The enhanced logging will remain in place to help with ongoing monitoring and debugging of audio flow issues.