"""
Modal-based Orpheus FastAPI TTS Service

This module provides a Modal serverless implementation of the Orpheus FastAPI TTS system,
based on the orpheus_fast_api project requirements.
"""

import modal
import os
import logging
from typing import Optional, AsyncIterator
import asyncio
import json
import time
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

# Modal app configuration
MINUTES = 60
HOURS = 60 * MINUTES

# Create Modal app
orpheus_app = modal.App("orpheus-fastapi-tts")

# Create the Modal image with all dependencies from orpheus_fast_api
orpheus_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git",
        "curl", 
        "wget",
        "ffmpeg",
        "libsndfile1",
        "portaudio19-dev",
        "build-essential",
        "cmake",
        "pkg-config"
    )
    # Install CUDA toolkit for GPU support
    .run_commands(
        "wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb",
        "dpkg -i cuda-keyring_1.0-1_all.deb",
        "apt-get update -q",
        "apt-get install -y cuda-toolkit-12-4 || apt-get install -y cuda-toolkit-12-1",
        "rm cuda-keyring_1.0-1_all.deb"
    )
    # Install PyTorch with CUDA support
    .pip_install(
        "torch==2.5.1",
        "torchvision",
        "torchaudio",
        index_url="https://download.pytorch.org/whl/cu124"
    )
    # Install orpheus_fast_api requirements
    .pip_install(
        # Web Server Dependencies
        "fastapi==0.103.1",
        "uvicorn==0.23.2",
        "jinja2==3.1.2",
        "pydantic==2.3.0",
        "python-multipart==0.0.6",
        
        # API and Communication
        "requests==2.31.0",
        "python-dotenv==1.0.0",
        "watchfiles==1.0.4",
        
        # Audio Processing
        "numpy==1.24.0",
        "sounddevice==0.4.6",
        "snac==1.2.1",  # Required for audio generation from tokens
        
        # System Utilities
        "psutil==5.9.0",
        
        # Additional dependencies for Modal integration
        "websockets>=12.0",
        "msgpack>=1.1.0",
        "aiofiles>=24.0.0",
        
        # For llama.cpp integration
        "httpx>=0.27.0",
        "transformers>=4.35.0",
        "tokenizers>=0.15.0",
        "safetensors>=0.4.0",
        "huggingface_hub>=0.19.0",
        "accelerate>=0.25.0",
    )
    # Pre-download SNAC model during image build
    .run_commands(
        "python -c \"from snac import SNAC; SNAC.from_pretrained('hubertsiuzdak/snac_24khz')\"",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "PYTHONUNBUFFERED": "1",
        "USE_GPU": "true",
    })
)

# Create a volume for model storage
model_volume = modal.Volume.from_name("orpheus-models", create_if_missing=True)

# Create network file system for shared state
orpheus_nfs = modal.NetworkFileSystem.from_name("orpheus-nfs", create_if_missing=True)


@orpheus_app.cls(
    image=orpheus_image,
    gpu="l4",  # Use L4 GPU for TTS
    container_idle_timeout=5 * MINUTES,
    timeout=30 * MINUTES,
    volumes={
        "/models": model_volume,
        "/shared": orpheus_nfs,
    },
    secrets=[
        modal.Secret.from_name("huggingface", required=False),
        modal.Secret.from_name("orpheus-config", required=False),
    ],
    allow_concurrent_inputs=10,
)
class OrpheusFastAPIService:
    """Modal service for Orpheus FastAPI TTS"""
    
    def __init__(self):
        self.snac_model = None
        self.llama_client = None
        self.device = None
        self.cuda_stream = None
        
    @modal.enter()
    async def initialize(self):
        """Initialize the Orpheus FastAPI service"""
        import torch
        from snac import SNAC
        import httpx
        
        logger.info("Initializing Orpheus FastAPI service...")
        
        # Set up device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Initialize SNAC model for audio generation
        logger.info("Loading SNAC model...")
        self.snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
        self.snac_model = self.snac_model.to(self.device)
        
        # Set up CUDA stream for parallel processing if available
        if self.device == "cuda":
            self.cuda_stream = torch.cuda.Stream()
            logger.info("CUDA stream initialized for parallel processing")
        
        # Initialize HTTP client for llama.cpp server communication
        # We'll use a separate Modal service for llama.cpp
        self.llama_client = httpx.AsyncClient(
            timeout=httpx.Timeout(120.0),
            limits=httpx.Limits(max_keepalive_connections=5)
        )
        
        # Get configuration from environment/secrets
        self.api_config = {
            "max_tokens": int(os.environ.get("ORPHEUS_MAX_TOKENS", "8192")),
            "temperature": float(os.environ.get("ORPHEUS_TEMPERATURE", "0.6")),
            "top_p": float(os.environ.get("ORPHEUS_TOP_P", "0.9")),
            "repetition_penalty": 1.1,  # Hardcoded as per orpheus_fast_api
            "sample_rate": int(os.environ.get("ORPHEUS_SAMPLE_RATE", "24000")),
        }
        
        logger.info(f"Orpheus FastAPI service initialized with config: {self.api_config}")
        
    @modal.method()
    async def generate_speech(
        self,
        text: str,
        voice: str = "tara",
        model: str = "orpheus",
        response_format: str = "wav",
        speed: float = 1.0,
        llama_endpoint: Optional[str] = None,
    ) -> bytes:
        """
        Generate speech from text using Orpheus TTS
        
        Args:
            text: Text to convert to speech
            voice: Voice to use (default: "tara")
            model: Model name (default: "orpheus")
            response_format: Output format (default: "wav")
            speed: Speech speed multiplier (0.5-1.5)
            llama_endpoint: Optional endpoint for llama.cpp server
            
        Returns:
            Audio data as bytes in WAV format
        """
        import torch
        import numpy as np
        import wave
        import io
        from unmute.orpheus_fast_api.tts_engine import speechpipe, inference
        
        try:
            logger.info(f"Generating speech for text: '{text[:100]}...' with voice: {voice}")
            
            # Format the prompt for Orpheus model
            formatted_prompt = self._format_prompt(text, voice)
            
            # Get the llama.cpp endpoint (from parameter or environment)
            if not llama_endpoint:
                llama_endpoint = os.environ.get("ORPHEUS_LLAMA_ENDPOINT")
                if not llama_endpoint:
                    # Use the companion llama.cpp Modal service
                    llama_endpoint = await self._get_llama_endpoint()
            
            # Generate tokens using llama.cpp API
            tokens = await self._generate_tokens(formatted_prompt, llama_endpoint)
            
            # Convert tokens to audio using SNAC
            audio_data = await self._tokens_to_audio(tokens, speed)
            
            # Convert to WAV format
            wav_data = self._create_wav(audio_data, self.api_config["sample_rate"])
            
            logger.info(f"Successfully generated {len(wav_data)} bytes of audio")
            return wav_data
            
        except Exception as e:
            logger.error(f"Error generating speech: {e}")
            raise
    
    @modal.method()
    async def generate_speech_stream(
        self,
        text: str,
        voice: str = "tara",
        model: str = "orpheus",
        speed: float = 1.0,
        llama_endpoint: Optional[str] = None,
    ) -> AsyncIterator[bytes]:
        """
        Stream speech generation for real-time output
        
        Args:
            text: Text to convert to speech
            voice: Voice to use
            model: Model name  
            speed: Speech speed multiplier
            llama_endpoint: Optional endpoint for llama.cpp server
            
        Yields:
            Audio chunks as bytes
        """
        import torch
        import numpy as np
        
        try:
            logger.info(f"Starting streaming speech generation for: '{text[:100]}...'")
            
            # Format the prompt
            formatted_prompt = self._format_prompt(text, voice)
            
            # Get llama endpoint
            if not llama_endpoint:
                llama_endpoint = os.environ.get("ORPHEUS_LLAMA_ENDPOINT")
                if not llama_endpoint:
                    llama_endpoint = await self._get_llama_endpoint()
            
            # Stream tokens from llama.cpp
            async for token_batch in self._stream_tokens(formatted_prompt, llama_endpoint):
                # Convert token batch to audio
                audio_chunk = await self._tokens_to_audio(token_batch, speed)
                if audio_chunk:
                    yield audio_chunk
                    
            logger.info("Streaming speech generation completed")
            
        except Exception as e:
            logger.error(f"Error in streaming speech generation: {e}")
            raise
    
    def _format_prompt(self, text: str, voice: str) -> str:
        """Format the text prompt for the Orpheus model"""
        # Based on orpheus_fast_api prompt formatting
        start_token = "<custom_token_128259>"
        end_tokens = "<custom_token_128009><custom_token_128260><custom_token_128261><custom_token_128257>"
        
        # Add voice prefix
        prompt = f"{start_token}{voice}: {text}{end_tokens}"
        return prompt
    
    async def _get_llama_endpoint(self) -> str:
        """Get the endpoint for the llama.cpp Modal service"""
        # This would connect to a separate Modal service running llama.cpp
        # For now, return a placeholder - we'll implement the llama.cpp service next
        return "http://orpheus-llama-cpp.modal.run/v1/completions"
    
    async def _generate_tokens(self, prompt: str, endpoint: str) -> list:
        """Generate tokens using llama.cpp API"""
        try:
            # Prepare the request for llama.cpp completions endpoint
            request_data = {
                "prompt": prompt,
                "max_tokens": self.api_config["max_tokens"],
                "temperature": self.api_config["temperature"],
                "top_p": self.api_config["top_p"],
                "repeat_penalty": self.api_config["repetition_penalty"],
                "stop": ["<custom_token_128009>", "<custom_token_128260>", 
                        "<custom_token_128261>", "<custom_token_128257>"],
            }
            
            response = await self.llama_client.post(
                endpoint,
                json=request_data,
            )
            response.raise_for_status()
            
            result = response.json()
            generated_text = result.get("choices", [{}])[0].get("text", "")
            
            # Extract custom tokens from generated text
            tokens = self._extract_tokens(generated_text)
            return tokens
            
        except Exception as e:
            logger.error(f"Error generating tokens: {e}")
            raise
    
    async def _stream_tokens(self, prompt: str, endpoint: str) -> AsyncIterator[list]:
        """Stream tokens from llama.cpp API"""
        try:
            request_data = {
                "prompt": prompt,
                "max_tokens": self.api_config["max_tokens"],
                "temperature": self.api_config["temperature"],
                "top_p": self.api_config["top_p"],
                "repeat_penalty": self.api_config["repetition_penalty"],
                "stop": ["<custom_token_128009>", "<custom_token_128260>",
                        "<custom_token_128261>", "<custom_token_128257>"],
                "stream": True,
            }
            
            async with self.llama_client.stream("POST", endpoint, json=request_data) as response:
                response.raise_for_status()
                buffer = ""
                
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])
                            if "choices" in data:
                                text = data["choices"][0].get("text", "")
                                buffer += text
                                
                                # Extract and yield complete token batches
                                tokens = self._extract_tokens_from_buffer(buffer)
                                if tokens:
                                    yield tokens
                                    # Clear processed tokens from buffer
                                    buffer = self._clean_buffer(buffer)
                                    
                        except json.JSONDecodeError:
                            continue
                            
        except Exception as e:
            logger.error(f"Error streaming tokens: {e}")
            raise
    
    def _extract_tokens(self, text: str) -> list:
        """Extract custom tokens from generated text"""
        import re
        
        # Pattern to match custom tokens
        pattern = r"<custom_token_(\d+)>"
        matches = re.findall(pattern, text)
        
        tokens = []
        for match in matches:
            token_id = int(match)
            if token_id != 0:  # Skip padding tokens
                tokens.append(token_id)
        
        return tokens
    
    def _extract_tokens_from_buffer(self, buffer: str) -> list:
        """Extract complete tokens from streaming buffer"""
        import re
        
        # Only extract complete tokens (with closing >)
        pattern = r"<custom_token_(\d+)>"
        matches = re.findall(pattern, buffer)
        
        tokens = []
        for match in matches:
            token_id = int(match)
            if token_id != 0:
                tokens.append(token_id)
        
        # Return tokens in batches of 7 (one frame)
        if len(tokens) >= 7:
            batch_size = (len(tokens) // 7) * 7
            return tokens[:batch_size]
        
        return []
    
    def _clean_buffer(self, buffer: str) -> str:
        """Remove processed tokens from buffer"""
        import re
        
        # Remove complete tokens from buffer
        pattern = r"<custom_token_\d+>"
        processed = re.sub(pattern, "", buffer, count=7)  # Remove up to 7 tokens
        return processed
    
    async def _tokens_to_audio(self, tokens: list, speed: float = 1.0) -> bytes:
        """Convert tokens to audio using SNAC model"""
        import torch
        import numpy as np
        
        if not tokens or len(tokens) < 7:
            return b""
        
        try:
            # Process tokens into SNAC codes (based on speechpipe.py logic)
            num_frames = len(tokens) // 7
            frame_tokens = tokens[:num_frames * 7]
            
            # Convert tokens to SNAC codes format
            codes_0 = []
            codes_1 = []
            codes_2 = []
            
            for i in range(0, len(frame_tokens), 7):
                frame = frame_tokens[i:i+7]
                if len(frame) == 7:
                    # Map tokens to codes based on orpheus_fast_api logic
                    # Token ID conversion: token_id - 10 - ((index % 7) * 4096)
                    codes_0.append(self._token_to_code(frame[0], 0))
                    codes_1.extend([self._token_to_code(frame[1], 1),
                                   self._token_to_code(frame[4], 4)])
                    codes_2.extend([self._token_to_code(frame[2], 2),
                                   self._token_to_code(frame[3], 3),
                                   self._token_to_code(frame[5], 5),
                                   self._token_to_code(frame[6], 6)])
            
            # Convert to tensors
            codes = [
                torch.tensor(codes_0, dtype=torch.int32, device=self.device).unsqueeze(0),
                torch.tensor(codes_1, dtype=torch.int32, device=self.device).unsqueeze(0),
                torch.tensor(codes_2, dtype=torch.int32, device=self.device).unsqueeze(0),
            ]
            
            # Validate codes are in range
            for code_tensor in codes:
                if torch.any(code_tensor < 0) or torch.any(code_tensor > 4096):
                    logger.warning("Invalid token IDs detected, skipping audio generation")
                    return b""
            
            # Generate audio with SNAC
            with torch.inference_mode():
                if self.cuda_stream:
                    with torch.cuda.stream(self.cuda_stream):
                        audio_hat = self.snac_model.decode(codes)
                else:
                    audio_hat = self.snac_model.decode(codes)
                
                # Extract the relevant audio slice
                audio_slice = audio_hat[:, :, 2048:4096]
                
                # Apply speed adjustment if needed
                if speed != 1.0:
                    # Resample for speed adjustment
                    audio_slice = self._adjust_speed(audio_slice, speed)
                
                # Convert to int16 audio bytes
                if self.device == "cuda":
                    audio_int16 = (audio_slice * 32767).to(torch.int16)
                    audio_bytes = audio_int16.cpu().numpy().tobytes()
                else:
                    audio_np = audio_slice.detach().cpu().numpy()
                    audio_int16 = (audio_np * 32767).astype(np.int16)
                    audio_bytes = audio_int16.tobytes()
                
                return audio_bytes
                
        except Exception as e:
            logger.error(f"Error converting tokens to audio: {e}")
            return b""
    
    def _token_to_code(self, token_id: int, index: int) -> int:
        """Convert token ID to SNAC code"""
        # Based on orpheus_fast_api token conversion logic
        return token_id - 10 - ((index % 7) * 4096)
    
    def _adjust_speed(self, audio_tensor, speed: float):
        """Adjust audio playback speed"""
        import torch.nn.functional as F
        
        if speed == 1.0:
            return audio_tensor
        
        # Simple speed adjustment via resampling
        original_length = audio_tensor.shape[-1]
        target_length = int(original_length / speed)
        
        # Reshape for interpolation
        audio_reshaped = audio_tensor.unsqueeze(1)  # Add channel dimension
        
        # Interpolate to target length
        audio_adjusted = F.interpolate(
            audio_reshaped,
            size=target_length,
            mode='linear',
            align_corners=False
        )
        
        # Remove added dimension
        return audio_adjusted.squeeze(1)
    
    def _create_wav(self, audio_bytes: bytes, sample_rate: int) -> bytes:
        """Create WAV file from raw audio bytes"""
        import wave
        import io
        
        # Create WAV file in memory
        wav_buffer = io.BytesIO()
        
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_bytes)
        
        wav_buffer.seek(0)
        return wav_buffer.read()
    
    @modal.asgi_app()
    def asgi_app(self):
        """Create FastAPI app for HTTP endpoints"""
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import Response, StreamingResponse
        from pydantic import BaseModel
        
        app = FastAPI(title="Orpheus FastAPI TTS")
        
        class SpeechRequest(BaseModel):
            input: str
            model: str = "orpheus"
            voice: str = "tara"
            response_format: str = "wav"
            speed: float = 1.0
        
        @app.post("/v1/audio/speech")
        async def create_speech(request: SpeechRequest):
            """OpenAI-compatible TTS endpoint"""
            try:
                audio_data = await self.generate_speech(
                    text=request.input,
                    voice=request.voice,
                    model=request.model,
                    response_format=request.response_format,
                    speed=request.speed,
                )
                
                return Response(
                    content=audio_data,
                    media_type="audio/wav",
                    headers={
                        "Content-Disposition": f"attachment; filename=speech_{uuid.uuid4().hex[:8]}.wav"
                    }
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/v1/audio/speech/stream")
        async def create_speech_stream(request: SpeechRequest):
            """Streaming TTS endpoint"""
            try:
                async def audio_stream():
                    async for chunk in self.generate_speech_stream(
                        text=request.input,
                        voice=request.voice,
                        model=request.model,
                        speed=request.speed,
                    ):
                        yield chunk
                
                return StreamingResponse(
                    audio_stream(),
                    media_type="audio/wav",
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/health")
        async def health():
            return {"status": "healthy", "service": "orpheus-fastapi-tts"}
        
        return app


# Export the service for use in other modules
__all__ = ["OrpheusFastAPIService", "orpheus_app"]