# Orpheus FastAPI Migration Complete

## Summary

We have successfully replaced the current Orpheus TTS implementation with a new Modal-based solution that uses the `orpheus_fast_api` project. The migration involved creating two new Modal services and updating the existing codebase to use them.

## What Was Done

### 1. Created New Modal Services

#### A. Orpheus FastAPI TTS Service (`unmute/tts/orpheus_fastapi_modal.py`)
- Main TTS service based on the orpheus_fast_api project
- Implements the SNAC model for audio generation
- Provides OpenAI-compatible REST API endpoints
- Supports WebSocket streaming for real-time audio generation
- Configured to run on L4 GPU for optimal performance

Key features:
- `/v1/audio/speech` - OpenAI-compatible TTS endpoint
- `/v1/audio/speech/stream` - Streaming TTS endpoint  
- WebSocket support for real-time streaming
- Multiple voice support (tara, leah, jess, leo, dan, mia, zac, zoe)
- Speed adjustment (0.5-1.5x)

#### B. Orpheus llama.cpp Server (`unmute/tts/orpheus_llama_modal.py`)
- Dedicated llama.cpp server for running the Orpheus language model
- Configured for the Orpheus-3b-FT-Q8_0.gguf model
- Provides OpenAI-compatible completions API
- Optimized with GPU support (L40S) and proper CUDA configuration
- Supports both streaming and non-streaming generation

Key features:
- `/v1/completions` - OpenAI-compatible completions endpoint
- Automatic model downloading from HuggingFace
- Configurable model selection via environment variables
- Optimized inference with rope-scaling and GPU layers

### 2. Updated Existing Code

#### A. Modified `unmute/tts/text_to_speech.py`
- Updated `OrpheusTextToSpeech` class to use the new Modal services
- Changed WebSocket connection to point to Modal FastAPI endpoints
- Simplified message handling for the new service architecture
- Maintained backward compatibility with existing interfaces

#### B. Updated `modal_app.py`
- Completely removed the TTSService class (no longer needed)
- Updated Orchestrator to connect directly to Orpheus FastAPI Modal service
- Eliminated proxy layer for better performance and simplicity
- No local model loading or management required

### 3. Removed Old Implementation
- Deleted `unmute/tts/orpheus_tts.py` (old implementation)
- Removed local model loading and SNAC initialization from modal_app.py

### 4. Created Deployment Tools
- `deploy_orpheus_modal.py` - Automated deployment script for both services
- Handles Modal authentication checks
- Deploys services in correct order
- Provides configuration instructions and test commands

## Architecture (Simplified)

```
┌─────────────────────────┐
│   Client Application    │
│   (Orchestrator)        │
└───────────┬─────────────┘
            │ Direct WebSocket
            │ Connection
            ▼
┌─────────────────────────┐
│  Orpheus FastAPI Modal  │◄──────┐
│   (orpheus_fastapi_     │       │
│     modal.py)           │       │
│   - SNAC audio gen      │       │
│   - WebSocket streaming │       │
│   - OpenAI-compatible   │       │
└─────────────────────────┘       │
                                  │
                          ┌───────┴──────────┐
                          │ Orpheus llama.cpp│
                          │  Modal Service   │
                          │ (orpheus_llama_  │
                          │   modal.py)      │
                          │ - Token generation│
                          └──────────────────┘
```

The architecture has been simplified by removing the proxy layer. The Orchestrator in modal_app.py now connects directly to the Orpheus FastAPI Modal service via WebSocket, eliminating an unnecessary layer of indirection.

## Configuration

### Environment Variables

Add these to your `.env` file or environment:

```bash
# Orpheus FastAPI service URL (deployed Modal service)
ORPHEUS_FASTAPI_URL=https://orpheus-fastapi-tts.modal.run

# Orpheus llama.cpp endpoint (deployed Modal service)
ORPHEUS_LLAMA_ENDPOINT=https://orpheus-llama-cpp.modal.run/v1/completions

# Model configuration (optional)
ORPHEUS_MODEL_NAME=Orpheus-3b-FT-Q8_0.gguf
ORPHEUS_MAX_TOKENS=8192
ORPHEUS_TEMPERATURE=0.6
ORPHEUS_TOP_P=0.9
ORPHEUS_SAMPLE_RATE=24000
```

## Deployment

To deploy the new services to Modal:

```bash
# Run the deployment script
python deploy_orpheus_modal.py

# Or deploy individually:
modal deploy unmute/tts/orpheus_llama_modal.py
modal deploy unmute/tts/orpheus_fastapi_modal.py
```

## Testing

### Test the llama.cpp server:
```bash
curl https://orpheus-llama-cpp.modal.run/health
```

### Test the Orpheus FastAPI service:
```bash
curl https://orpheus-fastapi-tts.modal.run/health
```

### Generate speech:
```bash
curl -X POST https://orpheus-fastapi-tts.modal.run/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello world! This is a test.", "voice": "tara"}' \
  --output test.wav
```

### Test WebSocket streaming:
```python
import asyncio
import websockets

async def test_streaming():
    url = "wss://orpheus-fastapi-tts.modal.run/v1/audio/speech/stream/ws?voice=tara"
    async with websockets.connect(url) as ws:
        await ws.send("Hello world! This is a streaming test.")
        
        audio_chunks = []
        async for message in ws:
            if isinstance(message, bytes):
                audio_chunks.append(message)
        
        # Combine and save audio
        with open("stream_test.raw", "wb") as f:
            f.write(b''.join(audio_chunks))
        
        print(f"Received {len(audio_chunks)} audio chunks")

asyncio.run(test_streaming())
```

## Benefits of the New Implementation

1. **Simplified Architecture**: Removed unnecessary proxy layer - direct connection to TTS service
2. **Separation of Concerns**: TTS and LLM inference run in separate Modal services
3. **Better Scalability**: Each service can scale independently
4. **Improved Reliability**: Services can be updated/restarted independently
5. **Reduced Latency**: Direct connection eliminates proxy overhead
6. **OpenAI Compatibility**: Full compatibility with OpenAI TTS API
7. **Streaming Support**: Real-time audio streaming via WebSocket
8. **Docker-based Architecture**: Following the orpheus_fast_api Docker setup
9. **Optimized Performance**: Proper GPU allocation and CUDA optimization
10. **Easier Maintenance**: Fewer components to manage and debug

## Migration Notes

- The old `OrpheusModel` class has been completely removed
- All TTS requests now go through the Modal services
- WebSocket protocol remains compatible with existing clients
- Audio format remains 24kHz 16-bit PCM for compatibility

## Next Steps

1. Monitor the deployed services for performance
2. Configure auto-scaling policies in Modal if needed
3. Set up monitoring and alerting for the services
4. Consider adding caching for frequently requested text
5. Optimize model quantization (Q2_K, Q4_K_M) for faster inference if needed