# Sesame_TTS Modal Integration

This document describes the integration of SesameAI Labs' CSM (Conversational Speech Model) into the Modal voice stack as a new TTS service.

## Overview

The `SesameTTSService` class provides a Modal-based implementation of the Sesame CSM model for high-quality text-to-speech generation. Unlike traditional TTS models, CSM is designed for conversational speech and can maintain context across multiple utterances.

## Features

- **High-Quality Speech Generation**: Uses the CSM-1B model for natural-sounding speech
- **Conversational Context**: Supports context-aware generation for multi-turn conversations
- **Dual Interface**: Provides both HTTP and WebSocket endpoints
- **GPU Acceleration**: Optimized for L4 GPU instances
- **Watermarking**: Includes built-in audio watermarking for AI-generated content identification

## Architecture

### Modal Image: `sesame_tts_image`

The image includes:
- **Base Dependencies**: Core Python and system packages
- **CSM Requirements**: Specific versions from the CSM repository
  - `torch==2.4.0`
  - `torchaudio==2.4.0`
  - `tokenizers==0.21.0`
  - `transformers==4.49.0`
  - `huggingface_hub==0.28.1`
  - `moshi==0.2.2`
  - `torchtune==0.4.0`
  - `torchao==0.9.0`
- **CSM Repository**: Cloned to `/opt/csm`
- **Environment Configuration**: `NO_TORCH_COMPILE=1` for compatibility

### Service Class: `SesameTTSService`

**Configuration:**
- GPU: L4 instance
- Concurrency: Up to 5 concurrent requests
- Timeout: 15 minutes for model loading
- Volumes: HuggingFace cache for model storage

**Methods:**

#### `generate_speech(text, speaker=0, max_audio_length_ms=10000, temperature=0.9, topk=50)`
Basic TTS generation without conversation context.

**Parameters:**
- `text`: Input text to convert to speech
- `speaker`: Speaker ID (0 or 1) for different voice characteristics
- `max_audio_length_ms`: Maximum output audio length
- `temperature`: Sampling temperature (higher = more varied)
- `topk`: Top-k sampling parameter

**Returns:** Audio data as bytes (WAV format)

#### `generate_speech_with_context(text, speaker, context_segments, ...)`
Context-aware TTS generation for conversational scenarios.

**Parameters:**
- `context_segments`: List of previous conversation segments with:
  - `text`: Previous utterance text
  - `speaker`: Speaker ID for the utterance
  - `audio_tensor`: Audio tensor of the previous utterance

**Returns:** Audio data as bytes (WAV format)

## API Endpoints

### HTTP Endpoint: `/generate`
```bash
curl -X POST "https://your-deployment-url/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello from Sesame TTS!",
    "speaker": 0,
    "max_audio_length_ms": 10000,
    "temperature": 0.9,
    "topk": 50
  }'
```

**Response:**
```json
{
  "audio": "base64-encoded-wav-data",
  "format": "wav",
  "sample_rate": 24000
}
```

### WebSocket Endpoint: `/ws`
```javascript
const ws = new WebSocket('wss://your-deployment-url/ws');

ws.onopen = () => {
  ws.send(JSON.stringify({
    text: "Hello from WebSocket!",
    speaker: 0,
    max_audio_length_ms: 10000,
    temperature: 0.9,
    topk: 50
  }));
};

ws.onmessage = (event) => {
  // event.data contains WAV audio bytes
  const audioBlob = new Blob([event.data], { type: 'audio/wav' });
  // Play or process the audio
};
```

## Deployment

### Prerequisites

1. **Modal Account**: Set up and authenticate with Modal
2. **HuggingFace Token**: Required for accessing CSM models
   ```bash
   modal secret create huggingface-secret HF_TOKEN=your_token_here
   ```

### Deploy the Service

```bash
# Deploy all services including Sesame_TTS
modal deploy modal_app.py

# Or serve for development
modal serve modal_app.py
```

### Access URLs

After deployment, the service will be available at:
- **HTTP**: `https://username--voice-stack-sesamettservice-web.modal.run`
- **WebSocket**: `wss://username--voice-stack-sesamettservice-web.modal.run/ws`

## Usage Examples

### Simple Python Example

```python
import modal

# Connect to deployed service
app = modal.App.lookup("voice-stack")
sesame_tts = app.cls.SesameTTSService()

# Generate speech
audio_bytes = sesame_tts.generate_speech.remote(
    text="Hello from Sesame TTS!",
    speaker=0
)

# Save to file
with open("output.wav", "wb") as f:
    f.write(audio_bytes)
```

### WebSocket Example

See `test_sesame_tts.py` for complete WebSocket and HTTP examples.

## Model Information

### CSM (Conversational Speech Model)

- **Repository**: https://github.com/SesameAILabs/csm
- **Model**: `sesame/csm-1b` (1 billion parameters)
- **Sample Rate**: 24 kHz
- **Architecture**: Llama backbone + audio decoder
- **Audio Codec**: Mimi (32 codebooks)

### Key Features

1. **Conversational Quality**: Designed specifically for natural conversation
2. **Context Awareness**: Can maintain voice characteristics across turns
3. **Speaker Control**: Supports multiple speaker identities
4. **Watermarking**: Built-in audio watermarking for transparency

## Limitations

- **Language Support**: Primarily English (some other languages via data contamination)
- **Compute Requirements**: Requires GPU for reasonable inference speed
- **Model Size**: 1B parameters require significant memory
- **Context Length**: Limited by model's maximum sequence length (2048 tokens)

## Troubleshooting

### Common Issues

1. **Model Loading Timeout**: Increase the timeout in the service configuration
2. **CUDA Memory Errors**: Reduce batch size or use smaller context
3. **HuggingFace Authentication**: Ensure HF_TOKEN is properly configured
4. **Triton Compilation Errors**: The `NO_TORCH_COMPILE=1` environment variable should prevent these

### Debug Information

The service logs detailed information during startup and generation:
- Model loading progress
- Device selection (CUDA/CPU)
- Generation parameters and timing
- Audio output statistics

## Integration with Existing Services

The Sesame_TTS service is designed to work alongside the existing voice stack:

- **STT Service**: Provides speech recognition
- **LLM Service**: Generates text responses  
- **TTS Service**: Original Moshi-based TTS
- **Sesame_TTS Service**: New CSM-based TTS with conversational capabilities
- **Orchestrator**: Coordinates between all services

You can choose between the original TTS service and Sesame_TTS based on your needs:
- Use **TTS Service** for standard text-to-speech with voice cloning
- Use **Sesame_TTS Service** for conversational speech with context awareness

## Contributing

When modifying the Sesame_TTS service:

1. Follow the existing code patterns in `modal_app.py`
2. Maintain compatibility with the CSM repository structure
3. Update this documentation for any API changes
4. Test both HTTP and WebSocket interfaces
5. Verify GPU memory usage and performance

## License

The CSM model is licensed under Apache 2.0. Please review the license terms in the SesameAI Labs repository and maintain proper attribution.

**Important**: The CSM model includes audio watermarking for transparency and responsible AI use. Please keep this functionality intact and use the service ethically.