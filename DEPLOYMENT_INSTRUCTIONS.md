# Orpheus TTS Modal Deployment Instructions

## Overview

This guide will help you deploy the new Orpheus TTS Modal services that replace the non-working `orpheus_services.py`. The new implementation uses the proven Docker image from `orpheus_fast_api` without requiring code changes.

## Prerequisites

1. **Modal CLI installed and authenticated**:
   ```bash
   pip install modal
   modal token new
   ```

2. **HuggingFace token** (for model downloads):
   ```bash
   modal secret create huggingface-secret HF_TOKEN=your_hf_token_here
   ```

## Files Overview

- `unmute/tts/orpheus_modal.py` - New Modal entrypoint with two apps (Updated for Modal 1.0 API)
- `unmute/modal_app.py` - Updated orchestrator to use new services  
- `deploy_orpheus_modal.py` - Automated deployment script
- `unmute/tts/orpheus_services.py` - Cleared out (old implementation)

## ⚠️ Modal 1.0 API Update

The code has been updated to use the correct Modal 1.0 API:
- Uses `modal.Image.from_dockerfile()` instead of deprecated methods
- Follows the official Modal documentation for Docker image building
- Compatible with current Modal CLI versions

## Deployment Steps

### Option 1: Automated Deployment (Recommended)

```bash
cd /workspace
python deploy_orpheus_modal.py
```

This script will:
1. Verify Modal authentication
2. Deploy the llama.cpp server first
3. Deploy the TTS service
4. Provide testing instructions

### Option 2: Manual Deployment

1. **Deploy the llama.cpp server first** (TTS service depends on it):
   ```bash
   modal deploy unmute/tts/orpheus_modal.py::llama_app
   ```

2. **Deploy the TTS service**:
   ```bash
   modal deploy unmute/tts/orpheus_modal.py::app
   ```

## Testing the Deployment

### 1. Health Checks

```bash
# Test llama.cpp server
curl https://willdavenport--orpheus-llama-server-orpheusllamaserver-asgi-app.modal.run/health

# Test TTS service  
curl https://willdavenport--orpheus-tts-orpheustts-asgi-app.modal.run/health
```

### 2. TTS Generation Test

```bash
curl -X POST https://willdavenport--orpheus-tts-orpheustts-asgi-app.modal.run/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello world! This is a test.", "voice": "tara"}' \
  --output test.wav
```

### 3. llama.cpp Server Test

```bash
curl -X POST https://willdavenport--orpheus-llama-server-orpheusllamaserver-asgi-app.modal.run/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "<|audio|>tara: Hello world<|eot_id|>",
    "max_tokens": 100,
    "temperature": 0.6
  }'
```

## Architecture

The new setup creates two separate Modal apps:

```
┌─────────────────────────┐
│   Orchestrator          │
│   (modal_app.py)        │  
└───────────┬─────────────┘
            │ HTTP REST API
            │ /v1/audio/speech
            ▼
┌─────────────────────────┐
│  Orpheus TTS Modal      │◄──────┐
│  App: orpheus-tts       │       │
│  - Ubuntu 22.04 base    │       │ HTTP API
│  - FastAPI server       │       │ /v1/completions
│  - SNAC model (L4 GPU)  │       │
└─────────────────────────┘       │
                                  │
                          ┌───────┴──────────┐
                          │ Orpheus llama.cpp│
                          │ App: orpheus-     │
                          │   llama-server   │
                          │ - Official image │
                          │ - L40S GPU       │
                          │ - Model download │
                          └──────────────────┘
```

## Configuration

The orchestrator (`modal_app.py`) automatically configures these environment variables:

- **Dev Mode** (`MODAL_DEV_MODE=true`):
  - `KYUTAI_TTS_URL`: `https://willdavenport--orpheus-tts-orpheustts-asgi-app-dev.modal.run/v1/audio/speech`
  - `ORPHEUS_LLAMA_ENDPOINT`: `https://willdavenport--orpheus-llama-server-orpheusllamaserver-asgi-app-dev.modal.run/v1/completions`

- **Production Mode**:
  - `KYUTAI_TTS_URL`: `https://willdavenport--orpheus-tts-orpheustts-asgi-app.modal.run/v1/audio/speech`
  - `ORPHEUS_LLAMA_ENDPOINT`: `https://willdavenport--orpheus-llama-server-orpheusllamaserver-asgi-app.modal.run/v1/completions`

## Troubleshooting

### 1. Modal Authentication Issues
```bash
modal token show  # Check current token
modal token new   # Create new token if needed
```

### 2. HuggingFace Token Issues  
```bash
modal secret create huggingface-secret HF_TOKEN=your_token_here
```

### 3. Service Not Starting
- Check Modal logs: `modal logs your-app-name`
- Verify GPU availability in your Modal account
- Check that the HuggingFace token has access to the Orpheus model

### 4. Model Download Issues
- Ensure your HF token has access to `lex-au/Orpheus-3b-FT-Q8_0.gguf`
- Check Modal volumes: `modal volume list`
- Clear the models volume if corrupted: `modal volume delete orpheus-models`

### 5. TTS Generation Fails
- Verify the llama.cpp server is running and accessible
- Check that `ORPHEUS_API_URL` points to the correct llama server
- Test the llama server independently first

## Key Features

✅ **No Code Changes**: Uses proven `orpheus_fast_api` Docker setup  
✅ **Separation of Concerns**: TTS and LLM services run independently  
✅ **Auto-scaling**: Each service scales independently on Modal  
✅ **GPU Optimized**: L4 for TTS, L40S for LLM  
✅ **OpenAI Compatible**: Standard `/v1/audio/speech` endpoint  
✅ **Volume Persistence**: Models cached between deployments  

## Next Steps

1. Deploy the services using the instructions above
2. Test the endpoints to ensure they're working
3. The orchestrator will automatically use the new Modal services
4. Monitor performance and adjust GPU types if needed
5. Set up monitoring/alerting for the Modal services

## Support

If you encounter issues:
1. Check the Modal dashboard for service status
2. Review Modal logs for error messages  
3. Verify all prerequisites are met
4. Test services individually before integration