# Modal Deployment Guide

This guide explains how to deploy the voice stack application to Modal, following the serverless architecture with four main services.

## Architecture Overview

The Modal deployment consists of four serverless classes:

1. **OrchestratorService** (CPU) - Main client WebSocket endpoint
2. **STTService** (GPU: L4) - Speech-to-Text using Moshi STT
3. **TTSService** (GPU: L4) - Text-to-Speech using Moshi TTS  
4. **LLMService** (GPU: L40S) - Large Language Model using VLLM

Each service runs in its own container with:
- **Warm containers**: Models loaded once in `@modal.enter()`
- **Long-lived sessions**: WebSocket connections kept alive
- **Auto-scaling**: Modal handles scaling based on demand
- **GPU isolation**: Each service gets its own GPU type

## Prerequisites

1. **Modal Account**: Sign up at [modal.com](https://modal.com)
2. **Modal CLI**: Install and authenticate
   ```bash
   pip install modal
   modal token new
   ```

## Quick Start

1. **Deploy to Modal**:
   ```bash
   python deploy_modal.py deploy
   ```

2. **Local Development**:
   ```bash
   python deploy_modal.py serve
   ```

## Service Details

### OrchestratorService
- **Purpose**: Main client WebSocket endpoint that coordinates all services
- **Resources**: 2 CPU cores, no GPU
- **Image**: Includes the full `unmute` package
- **Endpoint**: `/ws` - accepts client WebSocket connections
- **Logic**: Uses the existing `UnmuteHandler` for full compatibility

### STTService  
- **Purpose**: Speech-to-Text transcription
- **Resources**: L4 GPU
- **Model**: Kyutai STT 1B (English/French)
- **Implementation**: Runs moshi-server as subprocess, proxies WebSocket messages
- **Endpoint**: `/ws` - accepts audio streams, returns transcription

### TTSService
- **Purpose**: Text-to-Speech synthesis  
- **Resources**: L4 GPU
- **Model**: Kyutai TTS 1.6B with voice embeddings
- **Implementation**: Runs moshi-server with Python environment for voice processing
- **Endpoint**: `/ws` - accepts text, returns audio streams

### LLMService
- **Purpose**: Language model for conversational responses
- **Resources**: L40S GPU (higher memory for larger models)
- **Model**: Google Gemma 3 1B (configurable)
- **Implementation**: VLLM AsyncLLMEngine for efficient inference
- **Endpoint**: `/ws` - accepts prompts, returns streaming text

## Configuration

### Environment Variables (Set automatically)
- `KYUTAI_STT_URL`: Points to Modal STT service
- `KYUTAI_TTS_URL`: Points to Modal TTS service  
- `KYUTAI_LLM_URL`: Points to Modal LLM service

### Modal Secrets
- `voice-auth`: Authentication tokens for inter-service communication

### Modal Volumes
- `voice-models`: Persistent storage for model weights and caches

## Service Communication

The services communicate using WebSockets in a fan-out pattern:

```
Client ↔ Orchestrator ↔ STT Service
                    ↔ LLM Service  
                    ↔ TTS Service
```

1. **Client** sends audio to **Orchestrator**
2. **Orchestrator** forwards audio to **STT Service**
3. **STT Service** returns transcription to **Orchestrator**
4. **Orchestrator** sends transcription to **LLM Service**
5. **LLM Service** returns response text to **Orchestrator**
6. **Orchestrator** sends text to **TTS Service**
7. **TTS Service** returns audio to **Orchestrator**
8. **Orchestrator** forwards audio to **Client**

## Scaling and Performance

### Container Lifecycle
- **Cold Start**: ~10-30 seconds for model loading
- **Warm Containers**: Kept alive for 20 minutes idle time
- **Concurrency**: Each service handles 1 request at a time for optimal GPU utilization

### Auto-scaling
- Modal automatically scales containers based on demand
- Each service can scale independently
- GPU containers are more expensive but provide dedicated resources

### Performance Optimizations
- Models loaded once per container in `@modal.enter()`
- WebSocket connections maintained for session duration
- Chunked message handling (< 2MB per message)
- No compression (Modal doesn't support permessage-deflate)

## Monitoring and Debugging

### Logs
```bash
modal logs voice-stack
modal logs voice-stack.OrchestratorService
modal logs voice-stack.STTService
modal logs voice-stack.TTSService
modal logs voice-stack.LLMService
```

### Service Status
```bash
modal app list
modal app logs voice-stack
```

### Local Development
```bash
python deploy_modal.py serve
```
This runs all services locally with hot-reloading.

## Security

### Inter-service Authentication
- Services protected with Modal proxy auth tokens
- `voice-auth` secret contains shared authentication keys
- Only the orchestrator can access STT/TTS/LLM endpoints

### Client Authentication
- Public endpoints for client connections
- Add custom authentication logic in the orchestrator if needed

## Cost Optimization

### GPU Selection
- **L4**: Cost-effective for STT/TTS workloads
- **L40S**: Higher memory for LLM, better performance
- **A10/A100**: Alternative options based on availability and cost

### Container Management
- **keep_warm=1**: Keeps one container warm per service
- **container_idle_timeout=1200**: 20-minute timeout balances cost vs. performance
- **concurrency_limit=1**: Ensures dedicated GPU per request

## Troubleshooting

### Common Issues

1. **Model Download Failures**
   - Check internet connectivity in Modal containers
   - Verify HuggingFace model paths
   - Use Modal volumes for persistent model storage

2. **WebSocket Connection Issues**
   - Verify service URLs in environment variables
   - Check Modal proxy authentication
   - Monitor container logs for connection errors

3. **GPU Memory Issues**
   - Adjust `gpu_memory_utilization` in VLLM config
   - Consider larger GPU types for bigger models
   - Monitor GPU usage in Modal dashboard

4. **Cold Start Latency**
   - Increase `keep_warm` for critical services
   - Pre-warm containers during deployment
   - Consider serverless alternatives for non-critical paths

### Getting Help

- Check Modal documentation: [docs.modal.com](https://docs.modal.com)
- Modal Discord community
- GitHub issues for this repository

## Frontend Integration

Update your frontend to point to the Modal orchestrator endpoint:

```javascript
// Replace localhost with Modal endpoint
const wsUrl = "wss://your-username--voice-stack-orchestratorservice-web.modal.run/ws";
const websocket = new WebSocket(wsUrl);
```

The API remains the same - all existing OpenAI Realtime API events are supported.

## Next Steps

1. **Custom Models**: Replace with your own fine-tuned models
2. **Multi-tenancy**: Add user authentication and isolation
3. **Monitoring**: Set up proper logging and metrics
4. **Caching**: Add Redis for session state and caching
5. **CDN**: Use Modal's built-in CDN for static assets
