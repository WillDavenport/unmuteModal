# Orpheus TTS Modal Integration

This directory contains the Orpheus TTS implementation adapted for Modal deployment as the `Orpheus_TTS_Service`.

## Files

- **`orpheus_modal.py`** - Main Modal-adapted implementation
- **`model.py`** - Original Truss model implementation  
- **`config.yaml`** - Original Truss configuration
- **`snac_batching_quantization_dev.py`** - SNAC batching utilities
- **`test_orpheus.py`** - Standalone test script
- **`INTEGRATION_NOTES.md`** - Detailed integration documentation

## Quick Start

### 1. Test the Service Locally (via Modal)

```bash
# Test basic functionality
modal run modal_app.py::test_orpheus_tts

# Test batch processing  
modal run modal_app.py::test_orpheus_batch
```

### 2. Use the Service in Code

```python
# Initialize the service
orpheus_service = Orpheus_TTS_Service()

# Generate speech
audio_bytes = await orpheus_service.generate_speech.aio(
    text="Hello, world!",
    voice="tara"
)

# Save audio
with open("output.wav", "wb") as f:
    f.write(audio_bytes)
```

### 3. Deploy to Modal

```bash
modal deploy modal_app.py
```

## Features

- ✅ **SNAC Audio Codec**: 24kHz, 16-bit audio generation
- ✅ **Batched Processing**: Efficient GPU utilization  
- ✅ **Voice Support**: "tara" voice (default)
- ✅ **Modal Integration**: Full Modal class with entrypoints
- ✅ **Error Handling**: Fallback audio generation
- ⚠️ **Model Integration**: Uses Hugging Face Transformers (not TRT-LLM)

## Performance

- **GPU**: L4 (10GB memory)
- **Batch Size**: Up to 64 concurrent audio generations
- **Input Limit**: 6,144 characters
- **Output Format**: WAV (24kHz, 16-bit, mono)

## Architecture

```
Text → Tokenizer → HF Model → Audio Tokens → SNAC → Audio Bytes
```

## Original Source

Adapted from: https://github.com/basetenlabs/truss-examples/tree/main/orpheus-best-performance