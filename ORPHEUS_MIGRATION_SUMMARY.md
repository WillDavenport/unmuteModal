# Orpheus TTS Migration Summary

## Overview
Successfully migrated from Kyutai TTS to Orpheus TTS for improved performance and quality. This migration replaces the existing Moshi-based TTS service with a high-performance Orpheus implementation.

## Key Changes Made

### 1. New Orpheus Implementation Files
- **`/workspace/unmute/tts/orpheus_model.py`**: Core Orpheus TTS model implementation with SNAC batching
- **`/workspace/unmute/tts/orpheus_tts.py`**: Orpheus TTS service wrapper for integration
- **`/workspace/orpheus_config.yaml`**: Configuration file based on the optimal performance setup

### 2. Modal App Updates (`modal_app.py`)

#### Dependencies Updated
- Added PyTorch 2.7.1 with CUDA 12.8 support
- Added SNAC 1.2.1 for audio generation
- Added batched 0.1.4 for optimized batching
- Updated model pre-download to include Orpheus tokenizer and SNAC models

#### Service Changes
- **Class Name**: `TTSService` → `OrpheusTTSService`
- **GPU**: Upgraded from L40S to H100_40GB for optimal performance
- **Implementation**: Replaced Moshi-based TTS with native Orpheus implementation
- **Model**: Changed from Kyutai TTS 1.6B to Orpheus 3B with SNAC 24kHz

#### URL Updates
- Development URLs: `ttsservice-web-dev.modal.run` → `orpheusttsservice-web-dev.modal.run`
- Production URLs: Updated accordingly in orchestrator service

### 3. Project Dependencies (`pyproject.toml`)
Added Orpheus-specific dependencies:
- torch>=2.1.0
- transformers>=4.35.0  
- snac==1.2.1
- batched==0.1.4
- huggingface_hub>=0.19.0

### 4. Test Files Updated
- **`test_modal_deployment.py`**: Updated TTS service URL references

### 5. Deployment Scripts Updated  
- **`deploy_modal.py`**: Updated service URL documentation

## Technical Implementation Details

### Orpheus Model Features
- **Batched Processing**: Uses `@batched.dynamically()` for efficient batch processing
- **SNAC Integration**: 24kHz SNAC model for high-quality audio generation
- **Torch Compilation**: Optimized with `torch.compile()` for better performance
- **Voice Support**: Maintains compatibility with existing voice system (default: "tara")

### Performance Optimizations
- **GPU Upgrade**: H100_40GB for optimal performance (as recommended in config)
- **Batch Size**: Up to 64 items with 15ms timeout for optimal throughput
- **Memory Management**: Efficient CUDA stream management and synchronization
- **Model Caching**: Pre-download of models during image build to reduce startup time

### WebSocket Protocol
Maintains compatibility with existing protocol:
- **Ready Message**: Sent on connection establishment
- **Speak Messages**: Process text input with voice specification
- **Audio Messages**: Stream audio chunks back to client
- **Marker Messages**: Echo back for synchronization
- **Error Handling**: Proper error reporting and connection management

## Migration Benefits

1. **Performance**: Orpheus provides superior speech quality and generation speed
2. **Efficiency**: Batched processing reduces GPU utilization and improves throughput
3. **Scalability**: H100 GPU and optimized batching support higher concurrent loads
4. **Quality**: SNAC 24kHz provides higher quality audio output
5. **Compatibility**: Maintains existing API interface for seamless integration

## Next Steps

1. Deploy the updated Modal app: `modal deploy modal_app.py`
2. Test the new Orpheus TTS service with existing clients
3. Monitor performance metrics and adjust batch sizes if needed
4. Consider adding more voice options as needed

## Configuration Notes

The implementation includes optimal performance settings from the Baseten repository:
- TensorRT-LLM with FP8 quantization for inference acceleration
- Chunked context processing for memory efficiency
- Max utilization batch scheduling policy
- Optimized memory allocation (90% KV cache GPU memory fraction)

## Rollback Plan

If needed, the previous Kyutai TTS implementation can be restored by:
1. Reverting the `modal_app.py` changes
2. Removing the Orpheus-specific files
3. Restoring the original dependencies
4. Updating service URLs back to `ttsservice`