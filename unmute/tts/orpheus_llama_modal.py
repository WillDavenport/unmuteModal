"""
Modal-based llama.cpp server for Orpheus model inference

This module provides a Modal serverless implementation of llama.cpp server
specifically configured for the Orpheus TTS model.
"""

import modal
import os
import logging
from typing import Optional, Dict, Any, AsyncIterator
import asyncio
import json
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

# Modal app configuration
MINUTES = 60
HOURS = 60 * MINUTES

# Create Modal app
llama_app = modal.App("orpheus-llama-cpp")

# Default model configuration
DEFAULT_MODEL = "lex-au/Orpheus-3b-FT-Q8_0.gguf"
DEFAULT_CTX_SIZE = 8192
DEFAULT_N_GPU_LAYERS = 29

# Create the Modal image with llama.cpp
llama_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git",
        "build-essential",
        "cmake",
        "curl",
        "wget",
        "libcurl4-openssl-dev",
    )
    # Install CUDA toolkit for GPU support
    .run_commands(
        "wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb",
        "dpkg -i cuda-keyring_1.0-1_all.deb",
        "apt-get update -q",
        "apt-get install -y cuda-toolkit-12-4 || apt-get install -y cuda-toolkit-12-1",
        "rm cuda-keyring_1.0-1_all.deb"
    )
    # Build and install llama.cpp with CUDA support
    .run_commands(
        # Clone llama.cpp repository
        "git clone https://github.com/ggerganov/llama.cpp /opt/llama.cpp",
        "cd /opt/llama.cpp && git checkout b4410",  # Use stable version
        
        # Build with CUDA support
        "cd /opt/llama.cpp && mkdir build && cd build && "
        "cmake .. -DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc && "
        "cmake --build . --config Release -j$(nproc)",
        
        # Make binaries accessible
        "ln -s /opt/llama.cpp/build/bin/llama-server /usr/local/bin/llama-server",
        "ln -s /opt/llama.cpp/build/bin/llama-cli /usr/local/bin/llama-cli",
    )
    # Install Python dependencies for model management
    .pip_install(
        "huggingface_hub>=0.19.0",
        "requests>=2.31.0",
        "fastapi>=0.103.1",
        "uvicorn>=0.23.2",
        "httpx>=0.27.0",
        "pydantic>=2.3.0",
        "aiofiles>=24.0.0",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "PYTHONUNBUFFERED": "1",
        "CUDA_VISIBLE_DEVICES": "0",
    })
)

# Create a volume for model storage
model_volume = modal.Volume.from_name("orpheus-models", create_if_missing=True)


@llama_app.cls(
    image=llama_image,
    gpu="l40s",  # Use L40S GPU for better LLM performance
    container_idle_timeout=10 * MINUTES,
    timeout=60 * MINUTES,
    volumes={
        "/models": model_volume,
    },
    secrets=[
        modal.Secret.from_name("huggingface", required=False),
        modal.Secret.from_name("orpheus-config", required=False),
    ],
    allow_concurrent_inputs=5,
)
class OrpheusLlamaCppServer:
    """Modal service for running llama.cpp with Orpheus model"""
    
    def __init__(self):
        self.server_process = None
        self.model_path = None
        self.server_port = 5006
        self.model_name = None
        
    @modal.enter()
    async def initialize(self):
        """Initialize and start the llama.cpp server"""
        from huggingface_hub import hf_hub_download
        import subprocess
        import time
        
        logger.info("Initializing Orpheus llama.cpp server...")
        
        # Get model configuration from environment
        self.model_name = os.environ.get("ORPHEUS_MODEL_NAME", DEFAULT_MODEL)
        ctx_size = int(os.environ.get("ORPHEUS_MAX_TOKENS", DEFAULT_CTX_SIZE))
        n_gpu_layers = int(os.environ.get("ORPHEUS_N_GPU_LAYERS", DEFAULT_N_GPU_LAYERS))
        
        # Parse model repo and filename
        if "/" in self.model_name:
            # Full HuggingFace repo format: owner/repo/filename.gguf
            parts = self.model_name.split("/")
            if len(parts) == 3:
                repo_id = f"{parts[0]}/{parts[1]}"
                filename = parts[2]
            else:
                # Assume format: owner/filename.gguf
                repo_id = f"{parts[0]}/{parts[1].replace('.gguf', '')}"
                filename = parts[1] if parts[1].endswith('.gguf') else f"{parts[1]}.gguf"
        else:
            # Just filename, use default repo
            repo_id = "lex-au/Orpheus-3b-FT-Q8_0"
            filename = self.model_name if self.model_name.endswith('.gguf') else f"{self.model_name}.gguf"
        
        # Check if model exists locally
        local_model_path = Path(f"/models/{filename}")
        
        if not local_model_path.exists():
            logger.info(f"Downloading model {repo_id}/{filename}...")
            try:
                # Download model from HuggingFace
                downloaded_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    cache_dir="/models",
                    local_dir="/models",
                    token=os.environ.get("HF_TOKEN"),
                )
                self.model_path = downloaded_path
                logger.info(f"Model downloaded to {self.model_path}")
            except Exception as e:
                logger.error(f"Failed to download model: {e}")
                # Try alternative download method
                logger.info("Attempting direct download with wget...")
                download_url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
                subprocess.run(
                    ["wget", "-P", "/models", download_url],
                    check=True
                )
                self.model_path = str(local_model_path)
        else:
            self.model_path = str(local_model_path)
            logger.info(f"Using existing model at {self.model_path}")
        
        # Start llama.cpp server
        logger.info("Starting llama.cpp server...")
        
        server_cmd = [
            "llama-server",
            "-m", self.model_path,
            "--port", str(self.server_port),
            "--host", "0.0.0.0",
            "--n-gpu-layers", str(n_gpu_layers),
            "--ctx-size", str(ctx_size),
            "--n-predict", str(ctx_size),
            "--rope-scaling", "linear",
            "--parallel", "4",  # Allow parallel requests
            "--threads", "8",
            "--batch-size", "512",
            "--ubatch-size", "512",
            "--cache-type-k", "f16",
            "--cache-type-v", "f16",
            "--log-disable",  # Disable verbose logging
        ]
        
        # Start server as subprocess
        self.server_process = subprocess.Popen(
            server_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for server to be ready
        logger.info("Waiting for llama.cpp server to be ready...")
        await self._wait_for_server()
        
        logger.info(f"Orpheus llama.cpp server ready on port {self.server_port}")
        
    async def _wait_for_server(self, timeout: int = 60):
        """Wait for the llama.cpp server to be ready"""
        import httpx
        import asyncio
        
        start_time = asyncio.get_event_loop().time()
        
        async with httpx.AsyncClient() as client:
            while asyncio.get_event_loop().time() - start_time < timeout:
                try:
                    response = await client.get(f"http://localhost:{self.server_port}/health")
                    if response.status_code == 200:
                        logger.info("llama.cpp server is ready")
                        return
                except (httpx.ConnectError, httpx.TimeoutException):
                    pass
                
                await asyncio.sleep(1)
        
        raise TimeoutError(f"llama.cpp server did not start within {timeout} seconds")
    
    @modal.exit()
    async def cleanup(self):
        """Clean up the llama.cpp server process"""
        if self.server_process:
            logger.info("Stopping llama.cpp server...")
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
            logger.info("llama.cpp server stopped")
    
    @modal.method()
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 8192,
        temperature: float = 0.6,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        stop: Optional[list] = None,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate text using the Orpheus model
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            repetition_penalty: Repetition penalty
            stop: Stop sequences
            stream: Whether to stream the response
            
        Returns:
            Generated text or streaming response
        """
        import httpx
        
        try:
            # Prepare request for llama.cpp completions endpoint
            request_data = {
                "prompt": prompt,
                "n_predict": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "repeat_penalty": repetition_penalty,
                "stop": stop or [],
                "stream": stream,
                "cache_prompt": True,
                "seed": -1,
            }
            
            async with httpx.AsyncClient(timeout=120.0) as client:
                if stream:
                    # Return streaming response
                    async with client.stream(
                        "POST",
                        f"http://localhost:{self.server_port}/completion",
                        json=request_data,
                    ) as response:
                        response.raise_for_status()
                        
                        result = {
                            "choices": [{"text": "", "finish_reason": None}],
                            "model": self.model_name,
                        }
                        
                        async for line in response.aiter_lines():
                            if line.startswith("data: "):
                                try:
                                    data = json.loads(line[6:])
                                    if "content" in data:
                                        result["choices"][0]["text"] += data["content"]
                                    if data.get("stop", False):
                                        result["choices"][0]["finish_reason"] = "stop"
                                        break
                                except json.JSONDecodeError:
                                    continue
                        
                        return result
                else:
                    # Non-streaming response
                    response = await client.post(
                        f"http://localhost:{self.server_port}/completion",
                        json=request_data,
                    )
                    response.raise_for_status()
                    
                    data = response.json()
                    return {
                        "choices": [{
                            "text": data.get("content", ""),
                            "finish_reason": "stop" if data.get("stop", False) else "length"
                        }],
                        "model": self.model_name,
                    }
                    
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            raise
    
    @modal.method()
    async def stream_generate(
        self,
        prompt: str,
        max_tokens: int = 8192,
        temperature: float = 0.6,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        stop: Optional[list] = None,
    ) -> AsyncIterator[str]:
        """
        Stream text generation using the Orpheus model
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            repetition_penalty: Repetition penalty
            stop: Stop sequences
            
        Yields:
            Generated text chunks
        """
        import httpx
        
        try:
            request_data = {
                "prompt": prompt,
                "n_predict": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "repeat_penalty": repetition_penalty,
                "stop": stop or [],
                "stream": True,
                "cache_prompt": True,
            }
            
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream(
                    "POST",
                    f"http://localhost:{self.server_port}/completion",
                    json=request_data,
                ) as response:
                    response.raise_for_status()
                    
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            try:
                                data = json.loads(line[6:])
                                if "content" in data:
                                    yield data["content"]
                                if data.get("stop", False):
                                    break
                            except json.JSONDecodeError:
                                continue
                                
        except Exception as e:
            logger.error(f"Error in streaming generation: {e}")
            raise
    
    @modal.asgi_app()
    def asgi_app(self):
        """Create FastAPI app for HTTP endpoints (OpenAI-compatible)"""
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import StreamingResponse, JSONResponse
        from pydantic import BaseModel
        from typing import List, Optional
        import json
        
        app = FastAPI(title="Orpheus llama.cpp Server")
        
        class CompletionRequest(BaseModel):
            prompt: str
            max_tokens: int = 8192
            temperature: float = 0.6
            top_p: float = 0.9
            repeat_penalty: float = 1.1
            stop: Optional[List[str]] = None
            stream: bool = False
        
        @app.post("/v1/completions")
        async def create_completion(request: CompletionRequest):
            """OpenAI-compatible completions endpoint"""
            try:
                if request.stream:
                    async def stream_response():
                        async for chunk in self.stream_generate(
                            prompt=request.prompt,
                            max_tokens=request.max_tokens,
                            temperature=request.temperature,
                            top_p=request.top_p,
                            repetition_penalty=request.repeat_penalty,
                            stop=request.stop,
                        ):
                            # Format as SSE
                            data = {
                                "choices": [{"text": chunk, "index": 0}],
                                "model": self.model_name,
                            }
                            yield f"data: {json.dumps(data)}\n\n"
                        
                        # Send final message
                        yield "data: [DONE]\n\n"
                    
                    return StreamingResponse(
                        stream_response(),
                        media_type="text/event-stream",
                    )
                else:
                    result = await self.generate(
                        prompt=request.prompt,
                        max_tokens=request.max_tokens,
                        temperature=request.temperature,
                        top_p=request.top_p,
                        repetition_penalty=request.repeat_penalty,
                        stop=request.stop,
                        stream=False,
                    )
                    return JSONResponse(content=result)
                    
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/health")
        async def health():
            """Health check endpoint"""
            return {"status": "healthy", "model": self.model_name}
        
        @app.get("/v1/models")
        async def list_models():
            """List available models"""
            return {
                "data": [{
                    "id": self.model_name,
                    "object": "model",
                    "owned_by": "orpheus",
                }]
            }
        
        return app


# Export the service
__all__ = ["OrpheusLlamaCppServer", "llama_app"]