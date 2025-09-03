# Orpheus TTS Integration Notes

## Overview

This directory contains the core files and logic from the Orpheus TTS repository integrated into your Modal project as the `Orpheus_TTS_Service` class.

## Files Copied

1. **`model.py`** - Core TTS model implementation with SNAC batching
2. **`config.yaml`** - Original Truss configuration with dependencies
3. **`snac_batching_quantization_dev.py`** - SNAC model batching utilities
4. **`orpheus_modal.py`** - Modal-adapted implementation (newly created)

## Current Implementation Status

### ✅ Completed
- Modal service class (`Orpheus_TTS_Service`) created
- Image with all required dependencies configured
- SNAC model integration and batching logic
- Audio processing pipeline (token decoding to audio bytes)
- Local entrypoints for testing

### ⚠️ Requires Attention: TensorRT-LLM Engine

The original Orpheus implementation relies on **TensorRT-LLM (TRT-LLM)** for the language model inference that generates the audio tokens. This is currently **not integrated** because:

1. **Missing TRT-LLM Engine**: The original uses a pre-built TensorRT-LLM engine for the `baseten/orpheus-3b-0.1-ft` model
2. **Complex Build Process**: TRT-LLM requires specialized compilation and optimization
3. **Model Weights**: The actual model weights need to be downloaded and converted

### Required Next Steps

To make the Orpheus TTS service fully functional, you need to:

#### Option 1: Use Hugging Face Transformers (Simpler)
```python
# Replace the mock implementation in orpheus_modal.py
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "baseten/orpheus-3b-0.1-ft",
    torch_dtype=torch.float16,
    device_map="auto"
)
```

#### Option 2: Integrate TensorRT-LLM (More Complex, Better Performance)
1. Build TensorRT-LLM engine for the Orpheus model
2. Integrate the engine initialization in the Modal service
3. Update the `generate_speech` method to use the TRT-LLM engine

## Dependencies Configured

The Modal image includes all required dependencies from `config.yaml`:

- **PyTorch**: 2.7.1 with CUDA 12.8 support
- **SNAC**: 1.2.1 for audio codec
- **Batched**: 0.1.4 for efficient batching
- **Transformers**: For model loading
- **Audio Libraries**: librosa, soundfile for audio processing

## Usage

### Testing the Service
```bash
# Test basic functionality
modal run modal_app.py::test_orpheus_tts

# Test batch processing
modal run modal_app.py::test_orpheus_batch
```

### Using in Code
```python
# Initialize the service
orpheus_service = Orpheus_TTS_Service()

# Generate speech
audio_bytes = await orpheus_service.generate_speech.aio(
    text="Hello, world!",
    voice="tara"
)

# Save to file
with open("output.wav", "wb") as f:
    f.write(audio_bytes)
```

## Performance Configuration

- **GPU**: L4 (as specified in original config)
- **Memory**: 10GB (as specified in original config)  
- **Batch Size**: Up to 64 for SNAC processing
- **Concurrent Inputs**: Up to 100

## Model Details

- **Base Model**: `baseten/orpheus-3b-0.1-ft`
- **Audio Codec**: SNAC 24kHz
- **Sample Rate**: 24kHz, 16-bit, mono
- **Max Input**: 6,144 characters
- **Default Voice**: "tara"

## Architecture

```
Text Input → Tokenizer → Language Model → Audio Tokens → SNAC Decoder → Audio Bytes
                                ↑                           ↑
                        (TRT-LLM Engine)            (Batched Processing)
                        [NEEDS INTEGRATION]          [✅ IMPLEMENTED]
```

## Next Steps for Full Integration

1. **Choose Implementation Path**: Decide between HuggingFace Transformers or TensorRT-LLM
2. **Model Integration**: Implement the actual language model inference
3. **Testing**: Verify end-to-end functionality
4. **Optimization**: Fine-tune batching and performance settings
5. **Production**: Deploy and monitor the service

## Original Repository

Source: https://github.com/basetenlabs/truss-examples/tree/main/orpheus-best-performance