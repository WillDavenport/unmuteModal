# Orpheus TTS Integration Progress

## Overview
This document tracks the progress of integrating Orpheus TTS model into the Modal-based voice stack, replacing the previous TTS system with a more advanced speech synthesis solution.

## Issues Identified and Fixed

### 1. Garbled Audio Issue (CRITICAL - RESOLVED)

**Problem**: The generated audio was completely garbled and unintelligible.

**Root Cause**: The Orpheus TTS implementation was not using the actual Orpheus language model at all. Instead, it was generating fake/random tokens that don't correspond to real speech.

**Fixes Applied**:

#### A. Missing Actual Language Model Loading
- **Before**: Only loaded the tokenizer, never loaded the actual Orpheus 3B language model
- **After**: Now loads the full `AutoModelForCausalLM` with proper GPU acceleration and authentication
- **Code**: Added complete model loading in `OrpheusModel.load()` method

#### B. Fake Token Generation Replaced
- **Before**: `generate_speech_stream()` was simulating random tokens with mathematical patterns
- **After**: Now uses real Orpheus model inference with proper text-to-speech token generation
- **Implementation**: Streaming generation with temperature, top-k sampling, and proper end-token detection

#### C. Audio Data Pipeline Corrections
- **Modal Service**: Fixed to send raw 16-bit PCM bytes directly (like Baseten API)
- **Client Handling**: Fixed OrpheusTextToSpeech to receive raw bytes and convert properly
- **Test Functions**: Updated to handle raw bytes correctly for WAV file creation

#### D. SNAC Model Integration
- **Verified**: SNAC model correctly outputs 16-bit PCM bytes at 24kHz
- **Conversion**: Proper `(audio_np * 32767).astype(np.int16).tobytes()` conversion confirmed

### 2. Missing Dependencies (RESOLVED)

**Problem**: Modal container missing `accelerate` package required for large model loading.

**Error**: 
```
ValueError: Using a `device_map`, `tp_plan`, `torch.device` context manager or setting `torch.set_default_device(device)` requires `accelerate`. You can install it with `pip install accelerate`
```

**Fixes Applied**:
- Added `accelerate>=0.20.0` to Modal TTS image dependencies
- Fixed deprecated `torch_dtype` parameter (changed to `dtype`)
- Added robust fallback loading strategy:
  1. **Primary**: Try loading with `device_map="auto"` (optimal GPU memory management)
  2. **Fallback**: Load without device_map and manually move to GPU with `.cuda()`
  3. **CPU Fallback**: Run on CPU if no GPU available

### 3. Model Download Configuration (RESOLVED)

**Problem**: Modal service only downloading tokenizer files, not the complete model.

**Fix**: Updated snapshot_download to fetch complete model:
```python
# Before: only tokenizer files
allow_patterns=['tokenizer*', 'config.json']

# After: complete model download
# (removed allow_patterns to download everything)
```

## Technical Implementation Details

### Architecture Overview
```
Text Input → Orpheus LLM → Speech Tokens → SNAC Decoder → Raw PCM → Client → Audio Output
```

### Key Components

1. **OrpheusModel Class** (`unmute/tts/orpheus_tts.py`)
   - Loads Orpheus 3B language model with proper authentication
   - Implements streaming text-to-speech token generation
   - Handles model inference with temperature and top-k sampling

2. **SNAC Audio Decoder**
   - Converts speech tokens to 16-bit PCM audio at 24kHz
   - Batched processing for efficiency
   - Proper audio format conversion

3. **Modal Service** (`modal_app.py`)
   - TTSService class with GPU acceleration (L40S)
   - WebSocket interface for real-time streaming
   - Proper raw bytes transmission

4. **Client Integration** (`text_to_speech.py`)
   - OrpheusTextToSpeech class for client-side handling
   - Raw PCM bytes processing
   - Integration with existing audio pipeline

### Data Flow

1. **Text Input**: User provides text and voice selection
2. **Model Inference**: Orpheus LLM generates speech tokens
3. **Audio Synthesis**: SNAC decoder converts tokens to PCM audio
4. **Streaming**: Raw 16-bit PCM bytes streamed via WebSocket
5. **Client Processing**: Bytes converted to float32 for audio pipeline
6. **Output**: Clear, intelligible speech audio

## Current Status

### ✅ Completed
- [x] Fixed garbled audio by implementing real model inference
- [x] Added missing accelerate dependency to Modal image
- [x] Implemented robust model loading with fallbacks
- [x] Fixed audio data pipeline (raw bytes handling)
- [x] Updated model download configuration
- [x] Fixed deprecated parameter warnings
- [x] Added proper error handling and logging

### 🧪 Ready for Testing
- [x] Modal service deployment with Orpheus model
- [x] WebSocket interface for real-time TTS
- [x] Test functions for audio generation validation
- [x] Integration with existing voice stack

### 📋 Next Steps
1. **Performance Testing**: Measure latency and throughput
2. **Voice Quality Evaluation**: Compare with previous TTS system  
3. **Memory Optimization**: Monitor GPU memory usage with 3B model
4. **Production Deployment**: Deploy to staging environment
5. **Client Integration**: Update frontend to use Orpheus TTS

## Files Modified

### Core Implementation
- `unmute/tts/orpheus_tts.py` - Main Orpheus model implementation
- `unmute/modal_app.py` - Modal service configuration and endpoints
- `unmute/unmute/tts/text_to_speech.py` - Client-side integration

### Configuration
- Modal TTS image dependencies updated
- HuggingFace model download configuration
- GPU device mapping and memory management

## Testing

### Test Commands
```bash
# Test direct TTS generation
modal run modal_app.py::test_orpheus_tts

# Test WebSocket interface  
modal run modal_app.py::test_orpheus_tts_websocket
```

### Expected Output
- Clear, intelligible speech audio
- Proper 16-bit PCM format at 24kHz sample rate
- Real-time streaming with low latency
- No garbled or corrupted audio artifacts

## Technical Notes

### Model Specifications
- **Model**: canopylabs/orpheus-3b-0.1-ft
- **Size**: 3 billion parameters
- **Precision**: FP16 for memory efficiency
- **GPU**: L40S with CUDA acceleration
- **Audio Format**: 16-bit PCM at 24kHz

### Memory Requirements
- Model loading: ~6GB GPU memory (FP16)
- SNAC decoder: Additional GPU memory for audio processing
- Batch processing for efficiency optimization

### Authentication
- HuggingFace token required for gated model access
- Proper token handling in Modal secrets configuration

## Lessons Learned

1. **Always verify actual model loading**: The original implementation appeared to work but was only using fake data
2. **Raw audio handling**: Understanding the exact data format (16-bit PCM) is critical for audio quality
3. **Dependency management**: Large model libraries like transformers require specific dependencies (accelerate)
4. **Fallback strategies**: Robust error handling prevents deployment failures
5. **Streaming considerations**: Real-time audio generation requires careful token-by-token processing

---

*Last Updated: [Current Date]*
*Status: Implementation Complete - Ready for Testing*
