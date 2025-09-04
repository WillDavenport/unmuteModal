# Orpheus TTS Service Testing

This document describes how to test the Orpheus TTS service in the Modal deployment.

## Available Tests

### 1. Direct Service Test (`test_orpheus_tts`)

Tests the Orpheus TTS service by calling its `generate_speech` method directly.

**Usage:**
```bash
modal run modal_app.py::test_orpheus_tts
```

**What it tests:**
- Automatic model loading via `@modal.enter()` decorator
- Direct speech generation using `TTSService.generate_speech()` method
- Audio chunk collection and processing
- File output (both raw PCM and WAV formats)

**Expected Output:**
- Confirmation of model loading
- Audio generation metrics (duration, chunk count, file sizes)
- Two audio files: `orpheus_test_output.raw` and `orpheus_test_output.wav`

### 2. WebSocket Interface Test (`test_orpheus_tts_websocket`)

Tests the Orpheus TTS service through its WebSocket API, simulating how real clients interact with it.

**Usage:**
```bash
modal run modal_app.py::test_orpheus_tts_websocket
```

**What it tests:**
- WebSocket connection to TTS service
- MessagePack protocol communication
- Text-to-speech request/response cycle
- Streaming audio chunk reception
- End-of-stream handling

**Expected Output:**
- WebSocket connection confirmation
- Message exchange logs
- Audio chunk reception metrics
- Audio file: `orpheus_websocket_test_output.raw`

## Test Requirements

### Prerequisites
- Modal CLI installed and configured
- Access to the Modal deployment (correct app name and secrets)
- HuggingFace token configured in Modal secrets

### Dependencies
The tests use the following packages (included in the Modal image):
- `msgpack` - Message serialization
- `websockets` - WebSocket client
- `numpy` - Audio processing
- `wave` - WAV file creation (optional)

## Audio Output

Both tests generate raw PCM audio files that can be played using FFmpeg:

```bash
# Play raw PCM audio (24kHz, 16-bit, mono)
ffplay -f s16le -ar 24000 -ac 1 orpheus_test_output.raw

# Play WAV file (if generated)
ffplay orpheus_test_output.wav
```

## Troubleshooting

### Common Issues

1. **HuggingFace Authentication Error**
   - Ensure the `huggingface-secret` is properly configured in Modal
   - Check that the HF token has access to the Orpheus model

2. **WebSocket Connection Failed**
   - Verify the correct Modal app URL pattern
   - Check that the TTS service is deployed and running
   - Ensure the service is not cold-starting (may take time)

3. **No Audio Generated**
   - Check model loading logs for errors
   - Verify CUDA availability in the Modal environment
   - Check for token generation issues in the Orpheus model

### Performance Notes

- **Model Loading**: First run may take several minutes to download models
- **Cold Start**: WebSocket test may timeout if the service is cold-starting
- **GPU Requirements**: Tests require L40S GPU resources as configured

## Integration with CI/CD

These tests can be integrated into deployment pipelines:

```bash
# Run both tests as part of deployment verification
modal run modal_app.py::test_orpheus_tts
modal run modal_app.py::test_orpheus_tts_websocket
```

## Test Data

**Default Test Text**: "This is a test of the Orpheus TTS service in our Modal deployment."

**Default Voice**: "tara"

**Expected Duration**: Approximately 3-5 seconds of audio

## Comparison with Sesame Tests

The Orpheus tests follow the same pattern as the original Sesame CSM tests but are adapted for:
- Streaming audio generation (vs. batch generation)
- WebSocket protocol (vs. HTTP API)
- MessagePack serialization (vs. JSON)
- Different audio format (24kHz PCM vs. other formats)

This ensures consistent testing patterns across TTS service migrations.
