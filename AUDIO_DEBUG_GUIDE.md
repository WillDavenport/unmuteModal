# Audio Flow Debugging Guide

This guide explains how to debug audio cutoff issues in the TTS (Text-to-Speech) pipeline using the comprehensive logging and analysis tools we've implemented.

## Overview

Audio flows through several stages in the system:

1. **Orpheus TTS Generation** - Modal service generates raw PCM audio chunks
2. **Backend TTS Service** - OrpheusTextToSpeech processes and queues audio messages
3. **Conversation TTS Loop** - Conversation service processes TTS messages
4. **WebSocket Stage** - Main websocket encodes to Opus and sends to frontend
5. **Frontend Stage** - Browser receives, decodes, and plays audio

## Quick Start

### 1. Enable Debug Logging

The debug logging is already integrated into the codebase. When you run the system, you'll see detailed logs like:

```
=== ORPHEUS_GENERATION_START ===
[ORPHEUS] Starting speech generation for text: 'Hello, this is a test...'
=== ORPHEUS_CHUNK_GENERATED ===
[ORPHEUS] Generated chunk #1: 4800 bytes, 2400 samples
```

### 2. Collect Logs

Run your system and capture logs to a file:

```bash
# Run your backend with logging
python -m unmute.main_websocket 2>&1 | tee audio_debug.log

# Or if using Docker/other setup, redirect logs
your_command 2>&1 | tee audio_debug.log
```

### 3. Analyze the Logs

Use the debug analysis script:

```bash
# Analyze from log file
python3 debug_audio_flow.py audio_debug.log

# Or analyze from stdin (pipe logs directly)
tail -f audio_debug.log | python3 debug_audio_flow.py

# Test the system with sample data
python3 test_audio_debug.py
```

## Understanding the Output

The debug report shows statistics for each stage:

```
1. ORPHEUS TTS GENERATION STAGE
----------------------------------------
Chunks generated:     5      ← How many chunks Orpheus produced
Total bytes:          24000  ← Raw PCM bytes generated
Total samples:        12000  ← Audio samples (bytes/2 for 16-bit)
Generation complete:  ✓      ← Whether generation finished successfully
Audio duration:       0.50s  ← Duration of generated audio

2. BACKEND TTS SERVICE STAGE
----------------------------------------
Raw chunks received:  5      ← Chunks received from Orpheus
Chunks queued:        5      ← Chunks successfully queued
Messages yielded:     5      ← Messages sent to conversation layer
Backend chunk loss:   0 (0.0%) ← Loss in this stage
```

### Key Metrics to Watch

- **Chunk Loss Percentages**: Shows where audio is being dropped
- **Overall Efficiency**: Should be close to 100% for good performance
- **WebSocket Efficiency**: Opus encoding success rate
- **Error Detection**: Lists any errors that occurred

## Common Issues and Solutions

### 1. Audio Cutoff Mid-Sentence

**Symptoms:**
- Orpheus generates complete audio (high chunk count)
- Backend receives all chunks
- Conversation or WebSocket stage shows losses

**Debug Steps:**
```bash
# Look for these patterns in logs
grep "=== ORPHEUS_GENERATION_COMPLETE ===" audio_debug.log
grep "=== BACKEND_TTS_STREAM_COMPLETE ===" audio_debug.log
grep "=== CONVERSATION_TTS_LOOP_END ===" audio_debug.log
```

**Common Causes:**
- Task cancellation during interruption
- WebSocket disconnection
- Frontend audio context suspended

### 2. No Audio at All

**Symptoms:**
- Low or zero chunk generation
- Early stage failures

**Debug Steps:**
```bash
# Check for generation errors
grep "=== ORPHEUS_GENERATION_ERROR ===" audio_debug.log
grep "=== BACKEND_TTS_STREAM_ERROR ===" audio_debug.log
```

**Common Causes:**
- Modal service not available
- Authentication issues
- Network connectivity problems

### 3. Choppy/Intermittent Audio

**Symptoms:**
- High WebSocket "No opus output" count
- Frontend receives fewer messages than sent

**Debug Steps:**
```bash
# Check Opus encoding issues
grep "=== WEBSOCKET_NO_OPUS_OUTPUT ===" audio_debug.log
grep "=== FRONTEND_AUDIO_RECEIVED ===" audio_debug.log
```

**Common Causes:**
- Opus encoder buffering issues
- Network packet loss
- Browser audio context problems

## Log Message Reference

### Orpheus Stage
- `=== ORPHEUS_GENERATION_START ===` - Generation begins
- `=== ORPHEUS_CHUNK_GENERATED ===` - Each audio chunk produced
- `=== ORPHEUS_GENERATION_COMPLETE ===` - Generation finished successfully
- `=== ORPHEUS_GENERATION_ERROR ===` - Generation failed

### Backend TTS Stage
- `=== BACKEND_TTS_STREAM_START ===` - Backend starts processing
- `=== BACKEND_TTS_RAW_CHUNK_RECEIVED ===` - Raw chunk from Orpheus
- `=== BACKEND_TTS_CHUNK_QUEUED ===` - Chunk successfully queued
- `=== BACKEND_TTS_MESSAGE_YIELDED ===` - Message sent to conversation
- `=== BACKEND_TTS_STREAM_COMPLETE ===` - Backend processing complete

### Conversation Stage
- `=== CONVERSATION_TTS_LOOP_START ===` - Conversation loop begins
- `=== CONVERSATION_TTS_MESSAGE_RECEIVED ===` - Message from backend
- `=== CONVERSATION_TTS_TO_OUTPUT_QUEUE ===` - Audio sent to output
- `=== CONVERSATION_TTS_LOOP_END ===` - Loop finished

### WebSocket Stage
- `=== WEBSOCKET_AUDIO_RECEIVED ===` - Audio tuple from conversation
- `=== WEBSOCKET_OPUS_ENCODED ===` - Successful Opus encoding
- `=== WEBSOCKET_SENDING_AUDIO ===` - Sending to frontend
- `=== WEBSOCKET_NO_OPUS_OUTPUT ===` - Opus encoder didn't output (buffering)

### Frontend Stage
- `=== FRONTEND_AUDIO_RECEIVED ===` - WebSocket message received
- `=== FRONTEND_SENDING_TO_DECODER ===` - Sent to decoder worker
- `=== FRONTEND_DECODER_OUTPUT ===` - Decoder produced audio frame
- `=== FRONTEND_TO_AUDIO_WORKLET ===` - Frame sent to audio worklet

## Advanced Debugging

### Real-time Monitoring

Monitor audio flow in real-time:

```bash
# Monitor specific stages
tail -f audio_debug.log | grep "ORPHEUS_CHUNK_GENERATED\|WEBSOCKET_SENDING_AUDIO"

# Count messages at each stage
tail -f audio_debug.log | grep "=== ORPHEUS_CHUNK_GENERATED ===" | wc -l
tail -f audio_debug.log | grep "=== FRONTEND_AUDIO_RECEIVED ===" | wc -l
```

### Performance Analysis

Check timing between stages:

```bash
# Extract timestamps for analysis
grep "=== ORPHEUS_CHUNK_GENERATED ===" audio_debug.log | head -1
grep "=== FRONTEND_AUDIO_RECEIVED ===" audio_debug.log | head -1
```

### Custom Analysis

The `debug_audio_flow.py` script can be extended for custom analysis:

```python
# Add custom metrics
def parse_log_line(line: str, stats: AudioFlowStats) -> None:
    # Add your custom parsing logic here
    if "MY_CUSTOM_MARKER" in line:
        stats.custom_counter += 1
```

## Integration with Existing Monitoring

This debug system complements existing monitoring:

- **Prometheus Metrics**: The debug logs don't replace metrics but provide detailed flow analysis
- **Error Tracking**: Errors are logged with context for easier debugging
- **Performance Monitoring**: Timing information helps identify bottlenecks

## Troubleshooting the Debug System

### Debug Logs Not Appearing

1. Check log level configuration
2. Ensure you're capturing stderr (use `2>&1`)
3. Verify the logging format matches the patterns

### Analysis Script Issues

1. Make sure you're using Python 3
2. Check file permissions: `chmod +x debug_audio_flow.py`
3. Verify log format matches expected patterns

### Performance Impact

The debug logging adds minimal overhead but can be disabled by:

1. Commenting out debug log statements
2. Using log level filtering
3. Redirecting debug logs to separate files

## Example Workflow

Here's a complete debugging workflow:

```bash
# 1. Start system with logging
python3 -m unmute.main_websocket 2>&1 | tee audio_debug.log &

# 2. Reproduce the audio cutoff issue
# (use your frontend to trigger TTS)

# 3. Stop logging and analyze
kill %1  # Stop the background process
python3 debug_audio_flow.py audio_debug.log

# 4. Look at the report to identify where audio is lost

# 5. Check specific stages based on the report
grep "=== WEBSOCKET_NO_OPUS_OUTPUT ===" audio_debug.log
grep "=== BACKEND_TTS_STREAM_ERROR ===" audio_debug.log
```

This comprehensive debugging system will help you identify exactly where audio messages are being dropped in the pipeline, making it much easier to fix the cutoff issues.