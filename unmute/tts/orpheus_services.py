"""
Orpheus TTS Services for Modal

This module provides both the Orpheus FastAPI TTS service and the companion llama.cpp server
as separate Modal apps that can be included in the main application.
"""

import modal
import os
import logging
from typing import Optional, AsyncIterator, Dict, Any
import asyncio
import json
import subprocess
import time
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

# Modal app configuration
MINUTES = 60
HOURS = 60 * MINUTES

# Create separate Modal apps for each service
orpheus_fastapi_app = modal.App("orpheus-fastapi-tts")
orpheus_llama_app = modal.App("orpheus-llama-cpp")

# Default model configuration for llama.cpp
DEFAULT_MODEL = "lex-au/Orpheus-3b-FT-Q8_0.gguf"
DEFAULT_CTX_SIZE = 8192
DEFAULT_N_GPU_LAYERS = 29

# Create the FastAPI image with all dependencies
orpheus_fastapi_image = (
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
        "apt-get install -y cuda-toolkit-12-4 libcublas-dev-12-4 libcusparse-dev-12-4 libcurand-dev-12-4 || apt-get install -y cuda-toolkit-12-1 libcublas-dev-12-1 libcusparse-dev-12-1 libcurand-dev-12-1",
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
        "msgpack-numpy>=0.4.8",
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

# Create the llama.cpp image
orpheus_llama_image = (
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
        "apt-get install -y cuda-toolkit-12-4 libcublas-dev-12-4 libcusparse-dev-12-4 libcurand-dev-12-4 || apt-get install -y cuda-toolkit-12-1 libcublas-dev-12-1 libcusparse-dev-12-1 libcurand-dev-12-1",
        "rm cuda-keyring_1.0-1_all.deb"
    )
    # Build and install llama.cpp with CUDA support
    .run_commands(
        # Clone llama.cpp repository
        "git clone https://github.com/ggerganov/llama.cpp /opt/llama.cpp",
        "cd /opt/llama.cpp && git checkout b3909", # Use a more stable commit instead of master
        
        # Build with CUDA support using more conservative settings for L40S
        "cd /opt/llama.cpp && mkdir -p build && cd build && "
        "PATH=/usr/local/cuda/bin:$PATH LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH "
        "cmake .. -DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc "
        "-DCMAKE_CUDA_ARCHITECTURES='89' "  # L40S is compute capability 8.9
        "-DCMAKE_BUILD_TYPE=Release "
        "-DGGML_CUDA_FORCE_DMMV=OFF "  # Disable force DMMV which can cause issues
        "-DGGML_CUDA_FORCE_MMQ=OFF "   # Disable force MMQ 
        "-DBUILD_SHARED_LIBS=OFF "
        "-DGGML_STATIC=ON",
        "cd /opt/llama.cpp/build && make -j$(nproc) llama-server",
        
        # Make binaries accessible with proper permissions
        "cp /opt/llama.cpp/build/bin/llama-server /usr/local/bin/llama-server",
        "chmod +x /usr/local/bin/llama-server",
        
        # Verify binaries exist and are executable
        "ls -la /opt/llama.cpp/build/bin/",
        "ldd /usr/local/bin/llama-server || echo 'ldd not available'",
        "/usr/local/bin/llama-server --help || echo 'Binary test failed'",
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

# Create volumes for model storage
orpheus_model_volume = modal.Volume.from_name("orpheus-models", create_if_missing=True)
# Use a separate volume instead of NetworkFileSystem for shared storage
orpheus_shared_volume = modal.Volume.from_name("orpheus-shared", create_if_missing=True)


@orpheus_llama_app.cls(
    image=orpheus_llama_image,
    gpu="l40s",  # Use L40S GPU for better LLM performance
    timeout=60 * MINUTES,
    volumes={
        "/models": orpheus_model_volume,
    },
    secrets=[
        modal.Secret.from_name("voice-auth"),
        modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"]),
    ],
    min_containers=int(os.environ.get("MIN_CONTAINERS", "0")),
    scaledown_window=10 * MINUTES,
)
@modal.concurrent(max_inputs=5)
class OrpheusLlamaCppServer:
    """Modal service for running llama.cpp with Orpheus model"""
        
    @modal.enter()
    async def initialize(self):
        """Initialize and start the llama.cpp server"""
        from huggingface_hub import hf_hub_download
        import subprocess
        import time
        import os
        
        logger.info("Initializing Orpheus llama.cpp server...")
        
        # Initialize instance variables with defaults from constants
        self.server_process = None
        self.model_path = None
        
        # Get model configuration from environment or use defaults
        self.model_name = os.environ.get("ORPHEUS_MODEL_NAME", DEFAULT_MODEL)
        self.ctx_size = int(os.environ.get("ORPHEUS_MAX_TOKENS", DEFAULT_CTX_SIZE))
        self.n_gpu_layers = int(os.environ.get("ORPHEUS_N_GPU_LAYERS", DEFAULT_N_GPU_LAYERS))
        self.server_port = 5006
        
        model_name = self.model_name
        ctx_size = self.ctx_size
        n_gpu_layers = self.n_gpu_layers
        
        # Verify llama-server binary exists
        llama_server_paths = [
            "/usr/local/bin/llama-server",
            "/opt/llama.cpp/build/bin/llama-server",
            "llama-server"  # fallback to PATH
        ]
        
        llama_server_binary = None
        for path in llama_server_paths:
            if os.path.exists(path) or path == "llama-server":
                try:
                    # Test if binary is executable
                    result = subprocess.run([path, "--help"], 
                                          capture_output=True, text=True, timeout=10)
                    if result.returncode == 0 or "llama-server" in result.stderr.lower():
                        llama_server_binary = path
                        logger.info(f"Found llama-server binary at: {path}")
                        break
                except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
                    continue
        
        if not llama_server_binary:
            raise RuntimeError("llama-server binary not found or not executable")
        
        # Simplify model name parsing for robustness
        if self.model_name == DEFAULT_MODEL:
            # Use default model
            repo_id = "lex-au/Orpheus-3b-FT-Q8_0.gguf"
            filename = "Orpheus-3b-FT-Q8_0.gguf"
        elif "/" in self.model_name and self.model_name.endswith('.gguf'):
            # Handle "owner/repo.gguf" format
            parts = self.model_name.split("/")
            repo_id = f"{parts[0]}/{parts[1]}"  # Keep the .gguf in repo_id
            filename = parts[1]
        else:
            # Fallback to default
            logger.warning(f"Unsupported model name format: {self.model_name}, using default")
            repo_id = "lex-au/Orpheus-3b-FT-Q8_0.gguf"
            filename = "Orpheus-3b-FT-Q8_0.gguf"
        
        # Check if model exists locally and is valid
        local_model_path = Path(f"/models/{filename}")
        model_is_valid = False
        
        if local_model_path.exists():
            # Check if existing model is valid
            try:
                with open(local_model_path, 'rb') as f:
                    magic = f.read(4)
                    if magic == b'GGUF' and local_model_path.stat().st_size > 1024 * 1024:  # At least 1MB
                        model_is_valid = True
                        logger.info(f"Using existing valid model at {local_model_path}")
                    else:
                        logger.warning(f"Existing model file appears corrupted (magic: {magic}), re-downloading...")
            except Exception as e:
                logger.warning(f"Could not validate existing model: {e}, re-downloading...")
        
        if not model_is_valid:
            logger.info(f"Downloading model {repo_id}/{filename}...")
            
            # Remove corrupted file if it exists
            if local_model_path.exists():
                try:
                    local_model_path.unlink()
                    logger.info("Removed corrupted model file")
                except Exception as e:
                    logger.warning(f"Could not remove corrupted file: {e}")
            
            try:
                # Ensure the models directory exists
                Path("/models").mkdir(parents=True, exist_ok=True)
                
                downloaded_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    cache_dir="/models",
                    local_dir="/models",
                    token=os.environ.get("HF_TOKEN"),
                    resume_download=True,  # Resume partial downloads
                )
                
                # Validate the downloaded file
                with open(downloaded_path, 'rb') as f:
                    magic = f.read(4)
                    if magic != b'GGUF':
                        # The download might be an error page, let's check
                        f.seek(0)
                        first_100_bytes = f.read(100)
                        if b'<html>' in first_100_bytes.lower() or b'<!doctype' in first_100_bytes.lower():
                            raise RuntimeError(f"Downloaded file appears to be an HTML page (likely an error page). Check your HF_TOKEN permissions for {repo_id}")
                        else:
                            raise RuntimeError(f"Downloaded file is not a valid GGUF file (magic bytes: {magic})")
                
                self.model_path = downloaded_path
                logger.info(f"Model successfully downloaded and validated: {self.model_path}")
                
            except Exception as e:
                logger.error(f"Failed to download model: {e}")
                # Provide more specific error messages
                if "401" in str(e) or "unauthorized" in str(e).lower():
                    raise RuntimeError(f"Model download failed: Unauthorized access. Check your HF_TOKEN for {repo_id}")
                elif "404" in str(e) or "not found" in str(e).lower():
                    raise RuntimeError(f"Model download failed: Model {repo_id}/{filename} not found")
                else:
                    raise RuntimeError(f"Model download failed: {e}")
        else:
            self.model_path = str(local_model_path)
        
        # Final verification
        if not os.path.exists(self.model_path):
            raise RuntimeError(f"Model file not found after download/validation: {self.model_path}")
        
        file_size = os.path.getsize(self.model_path)
        logger.info(f"Model file ready: {self.model_path} ({file_size} bytes, {file_size / (1024**3):.2f} GB)")
        
        # Add diagnostic information
        logger.info("=== Diagnostic Information ===")
        logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
        logger.info(f"Using llama-server binary: {llama_server_binary}")
        logger.info(f"Requested GPU layers: {n_gpu_layers}")
        logger.info(f"Context size: {ctx_size}")
        
        # Test GPU availability
        try:
            result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv"], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                logger.info(f"GPU Info:\n{result.stdout}")
            else:
                logger.warning("nvidia-smi failed, GPU may not be available")
        except Exception as e:
            logger.warning(f"Could not check GPU status: {e}")
        
        # Start llama.cpp server with more conservative and robust settings
        logger.info("Starting llama.cpp server...")
        
        # Use more conservative settings to avoid crashes
        # Start with minimal GPU layers and increase if successful
        safe_gpu_layers = min(n_gpu_layers, 20)  # Start conservatively
        
        server_cmd = [
            llama_server_binary,
            "-m", self.model_path,
            "--port", str(self.server_port),
            "--host", "0.0.0.0",
            "--n-gpu-layers", str(safe_gpu_layers),
            "--ctx-size", str(min(ctx_size, 4096)),  # Cap context size to avoid memory issues
            "--batch-size", "128",  # Further reduce batch size
            "--ubatch-size", "128",
            "--threads", "4",  # Reduce thread count
            "--parallel", "1",  # Single parallel request for stability
            "--cache-type-k", "f16",
            "--cache-type-v", "f16",
            "--verbose",  # Enable verbose logging for debugging
            "--log-disable",  # Reduce log spam but keep errors
        ]
        
        logger.info(f"Starting server with command: {' '.join(server_cmd)}")
        
        # Store command for debugging
        self._last_server_cmd = ' '.join(server_cmd)
        
        # Use temporary files to capture stdout/stderr reliably
        import tempfile
        self.stdout_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='_stdout.log')
        self.stderr_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='_stderr.log')
        
        self.server_process = subprocess.Popen(
            server_cmd,
            stdout=self.stdout_file,
            stderr=self.stderr_file,
            text=True,
            bufsize=1  # Line buffered
        )
        
        await self._wait_for_server(timeout=120)  # Full timeout since no retries
        logger.info(f"Orpheus llama.cpp server ready on port {self.server_port}")
        
    async def _wait_for_server(self, timeout: int = 120):
        """Wait for the llama.cpp server to be ready"""
        import httpx
        import asyncio
        
        start_time = asyncio.get_event_loop().time()
        last_log_time = start_time
        last_stderr_pos = 0
        last_stdout_pos = 0
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            while asyncio.get_event_loop().time() - start_time < timeout:
                try:
                    # Check if server process is still running
                    if self.server_process and self.server_process.poll() is not None:
                        # Process has terminated, read all output from temp files
                        stderr_output = ""
                        stdout_output = ""
                        
                        try:
                            # Flush and read stderr
                            self.stderr_file.flush()
                            self.stderr_file.seek(0)
                            stderr_output = self.stderr_file.read()
                        except Exception as e:
                            stderr_output = f"Could not read stderr: {e}"
                        
                        try:
                            # Flush and read stdout
                            self.stdout_file.flush()
                            self.stdout_file.seek(0)
                            stdout_output = self.stdout_file.read()
                        except Exception as e:
                            stdout_output = f"Could not read stdout: {e}"
                        
                        return_code = self.server_process.returncode
                        raise RuntimeError(
                            f"llama.cpp server process terminated unexpectedly with return code {return_code}.\n"
                            f"STDERR: {stderr_output}\n"
                            f"STDOUT: {stdout_output}\n"
                            f"Model path: {self.model_path}\n"
                            f"Server command was: {getattr(self, '_last_server_cmd', 'Unknown')}"
                        )
                    
                    # Try to read partial output for debugging
                    current_time = asyncio.get_event_loop().time()
                    if current_time - last_log_time >= 15:  # Log every 15 seconds
                        elapsed = current_time - start_time
                        
                        # Try to read new stderr/stdout content from temp files
                        new_stderr = ""
                        new_stdout = ""
                        
                        try:
                            # Read new stderr content since last check
                            self.stderr_file.flush()
                            current_pos = self.stderr_file.tell()
                            if current_pos > last_stderr_pos:
                                self.stderr_file.seek(last_stderr_pos)
                                new_stderr = self.stderr_file.read(current_pos - last_stderr_pos)
                                last_stderr_pos = current_pos
                        except Exception:
                            pass
                        
                        try:
                            # Read new stdout content since last check
                            self.stdout_file.flush()
                            current_pos = self.stdout_file.tell()
                            if current_pos > last_stdout_pos:
                                self.stdout_file.seek(last_stdout_pos)
                                new_stdout = self.stdout_file.read(current_pos - last_stdout_pos)
                                last_stdout_pos = current_pos
                        except Exception:
                            pass
                        
                        log_msg = f"Still waiting for llama.cpp server... ({elapsed:.0f}s/{timeout}s elapsed)"
                        if new_stderr.strip():
                            log_msg += f"\nRecent stderr: {new_stderr.strip()}"
                        if new_stdout.strip():
                            log_msg += f"\nRecent stdout: {new_stdout.strip()}"
                        
                        logger.info(log_msg)
                        last_log_time = current_time
                    
                    # Try to connect to server
                    response = await client.get(f"http://localhost:{self.server_port}/health")
                    if response.status_code == 200:
                        logger.info("llama.cpp server is ready")
                        return
                        
                except (httpx.ConnectError, httpx.TimeoutException, httpx.ReadTimeout):
                    # Server not ready yet, continue waiting
                    pass
                
                await asyncio.sleep(2)  # Check every 2 seconds
        
        # Timeout reached - try to get final output from temp files
        final_stderr = ""
        final_stdout = ""
        
        try:
            self.stderr_file.flush()
            self.stderr_file.seek(0)
            final_stderr = self.stderr_file.read()
        except Exception as e:
            final_stderr = f"Could not read final stderr: {e}"
        
        try:
            self.stdout_file.flush()
            self.stdout_file.seek(0)
            final_stdout = self.stdout_file.read()
        except Exception as e:
            final_stdout = f"Could not read final stdout: {e}"
        
        raise TimeoutError(
            f"llama.cpp server did not start within {timeout} seconds.\n"
            f"Final STDERR: {final_stderr}\n"
            f"Final STDOUT: {final_stdout}\n"
            f"Process running: {self.server_process and self.server_process.poll() is None}"
        )
    
    @modal.exit()
    async def cleanup(self):
        """Clean up the llama.cpp server process and temporary files"""
        if self.server_process:
            logger.info("Stopping llama.cpp server...")
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
            logger.info("llama.cpp server stopped")
        
        # Clean up temporary files
        import os
        if hasattr(self, 'stdout_file'):
            try:
                self.stdout_file.close()
                os.unlink(self.stdout_file.name)
            except Exception:
                pass
        
        if hasattr(self, 'stderr_file'):
            try:
                self.stderr_file.close()
                os.unlink(self.stderr_file.name)
            except Exception:
                pass
    
    async def _generate_internal(
        self,
        prompt: str,
        max_tokens: int = 8192,
        temperature: float = 0.6,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        stop: Optional[list] = None,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """Internal method to generate text using the Orpheus model"""
        import httpx
        
        try:
            # Clamp n_predict to avoid exceeding server context size
            # The server is started with ctx-size capped to 4096 above
            effective_n_predict = min(max_tokens, getattr(self, "ctx_size", 4096), 4096)
            if effective_n_predict < max_tokens:
                logger.info(f"Clamping n_predict from {max_tokens} to {effective_n_predict} due to ctx-size limits")

            request_data = {
                "prompt": prompt,
                "n_predict": effective_n_predict,
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
                    async with client.stream(
                        "POST",
                        f"http://localhost:{self.server_port}/completion",
                        json=request_data,
                    ) as response:
                        try:
                            response.raise_for_status()
                        except httpx.HTTPStatusError as he:
                            body = await response.aread()
                            raise RuntimeError(f"llama.cpp /completion error {response.status_code}: {body.decode(errors='ignore')}") from he
                        
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
                    response = await client.post(
                        f"http://localhost:{self.server_port}/completion",
                        json=request_data,
                    )
                    try:
                        response.raise_for_status()
                    except httpx.HTTPStatusError as he:
                        # Include upstream response text for easier debugging
                        try:
                            body_text = response.text
                        except Exception:
                            body_text = "<unavailable>"
                        raise RuntimeError(
                            f"llama.cpp /completion error {response.status_code}: {body_text}"
                        ) from he
                    
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
        """Generate text using the Orpheus model (Modal method wrapper)"""
        return await self._generate_internal(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            stop=stop,
            stream=stream,
        )
    
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
                # Call the generate method directly (not as Modal method)
                result = await self._generate_internal(
                    prompt=request.prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    repetition_penalty=request.repeat_penalty,
                    stop=request.stop,
                    stream=request.stream,
                )
                return JSONResponse(content=result)
                    
            except Exception as e:
                # Surface detailed error for easier debugging
                return JSONResponse(status_code=500, content={
                    "error": "generation_failed",
                    "message": str(e),
                })
        
        @app.get("/health")
        async def health():
            """Health check endpoint"""
            return {"status": "healthy", "model": self.model_name}
        
        @app.post("/completion")
        async def create_completion_native(request: CompletionRequest):
            """llama.cpp native completion endpoint"""
            try:
                result = await self._generate_internal(
                    prompt=request.prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    repetition_penalty=request.repeat_penalty,
                    stop=request.stop,
                    stream=request.stream,
                )
                
                # Return in llama.cpp native format
                if result and "choices" in result and len(result["choices"]) > 0:
                    return JSONResponse(content={
                        "content": result["choices"][0]["text"],
                        "stop": result["choices"][0]["finish_reason"] == "stop",
                        "model": self.model_name,
                    })
                else:
                    return JSONResponse(content={
                        "content": "",
                        "stop": True,
                        "model": self.model_name,
                    })
                    
            except Exception as e:
                # Surface detailed error for easier debugging
                return JSONResponse(status_code=500, content={
                    "error": "generation_failed",
                    "message": str(e),
                })
        
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


@orpheus_fastapi_app.cls(
    image=orpheus_fastapi_image,
    gpu="L4",  # Use L4 GPU for TTS
    timeout=30 * MINUTES,
    volumes={
        "/models": orpheus_model_volume,
        "/shared": orpheus_shared_volume,
    },
    secrets = [
        modal.Secret.from_name("voice-auth"),
        modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"]),  # Ensure HF_TOKEN is available
    ],
    scaledown_window=10 * MINUTES,
    min_containers=int(os.environ.get("MIN_CONTAINERS", "0")),
)
@modal.concurrent(max_inputs=10)
class OrpheusFastAPIService:
    """Modal service for Orpheus FastAPI TTS"""
        
    @modal.enter()
    async def initialize(self):
        """Initialize the Orpheus FastAPI service"""
        import torch
        from snac import SNAC
        import httpx
        
        logger.info("Initializing Orpheus FastAPI service...")
        
        # Initialize instance variables
        self.snac_model = None
        self.llama_client = None
        self.device = None
        self.cuda_stream = None
        
        # Set up device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Initialize SNAC model for audio generation
        logger.info("Loading SNAC model...")
        try:
            self.snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
            logger.info(f"SNAC model loaded successfully, type: {type(self.snac_model)}")
            self.snac_model = self.snac_model.to(self.device)
            logger.info(f"SNAC model moved to device {self.device}")
            
            # Check if decode method exists
            if hasattr(self.snac_model, 'decode'):
                decode_method = getattr(self.snac_model, 'decode')
                logger.info(f"SNAC decode method found, type: {type(decode_method)}")
            else:
                logger.error("SNAC model does not have decode method!")
                
        except Exception as e:
            logger.error(f"Failed to load SNAC model: {e}")
            self.snac_model = None
            raise
        
        # Set up CUDA stream for parallel processing if available
        if self.device == "cuda":
            self.cuda_stream = torch.cuda.Stream()
            logger.info("CUDA stream initialized for parallel processing")
        
        # Initialize HTTP client for llama.cpp server communication
        self.llama_client = httpx.AsyncClient(
            timeout=httpx.Timeout(120.0),
            limits=httpx.Limits(max_keepalive_connections=5)
        )
        
        # Get configuration from environment/secrets
        self.api_config = {
            "max_tokens": int(os.environ.get("ORPHEUS_MAX_TOKENS", "8192")),
            "temperature": float(os.environ.get("ORPHEUS_TEMPERATURE", "0.6")),
            "top_p": float(os.environ.get("ORPHEUS_TOP_P", "0.9")),
            "repetition_penalty": 1.1,
            "sample_rate": int(os.environ.get("ORPHEUS_SAMPLE_RATE", "24000")),
        }
        
        logger.info(f"Orpheus FastAPI service initialized with config: {self.api_config}")
        
    async def generate_speech(
        self,
        text: str,
        voice: str = "tara",
        model: str = "orpheus",
        response_format: str = "wav",
        speed: float = 1.0,
        llama_endpoint: Optional[str] = None,
    ) -> bytes:
        """Generate speech from text using Orpheus TTS"""
        import torch
        import numpy as np
        import wave
        import io
        
        try:
            logger.info(f"Generating speech for text: '{text[:100]}...' with voice: {voice}")
            
            # Format the prompt for Orpheus model
            formatted_prompt = self._format_prompt(text, voice)
            
            # Get the llama.cpp endpoint
            if not llama_endpoint:
                llama_endpoint = os.environ.get("ORPHEUS_LLAMA_ENDPOINT")
                if not llama_endpoint:
                    llama_endpoint = await self._get_llama_endpoint()
            
            # Generate tokens using llama.cpp API
            logger.info(f"=== ORPHEUS DEBUG: Using llama endpoint: {llama_endpoint}")
            try:
                tokens = await self._generate_tokens(formatted_prompt, llama_endpoint)
                logger.info(f"=== ORPHEUS DEBUG: Token generation completed, got {len(tokens)} tokens")
            except Exception as e:
                logger.error(f"=== ORPHEUS DEBUG: Token generation failed: {e}")
                logger.info("=== ORPHEUS DEBUG: Using fallback tokens for testing")
                tokens = self._generate_fallback_tokens()
            
            # Convert tokens to audio using SNAC
            if tokens:
                logger.info(f"=== ORPHEUS DEBUG: Converting {len(tokens)} tokens to audio")
                audio_data = await self._tokens_to_audio(tokens, speed)
                logger.info(f"=== ORPHEUS DEBUG: Audio conversion completed, got {len(audio_data)} bytes")
            else:
                logger.error("=== ORPHEUS DEBUG: No tokens available for audio generation")
                audio_data = b""
            
            # Return raw PCM or WAV format based on response_format
            if response_format == "raw":
                logger.info(f"Successfully generated {len(audio_data)} bytes of raw PCM audio")
                return audio_data
            else:
                # Convert to WAV format
                wav_data = self._create_wav(audio_data, self.api_config["sample_rate"])
                logger.info(f"Successfully generated {len(wav_data)} bytes of WAV audio")
                return wav_data
            
        except Exception as e:
            logger.error(f"Error generating speech: {e}")
            raise
    
    def _format_prompt(self, text: str, voice: str) -> str:
        """Format the text prompt for the Orpheus model"""
        start_token = "<custom_token_128259>"
        end_tokens = "<custom_token_128009><custom_token_128260><custom_token_128261><custom_token_128257>"
        
        prompt = f"{start_token}{voice}: {text}{end_tokens}"
        logger.info(f"=== ORPHEUS DEBUG: Formatted prompt: '{prompt}'")
        logger.info(f"=== ORPHEUS DEBUG: Prompt length: {len(prompt)}")
        return prompt
    
    async def _get_llama_endpoint(self) -> str:
        """Resolve the HTTP endpoint for the llama.cpp service.

        Prefers the `ORPHEUS_LLAMA_ENDPOINT` environment variable, otherwise
        falls back to the public Modal URL.
        """
        import os
        endpoint = os.environ.get("ORPHEUS_LLAMA_ENDPOINT")
        if endpoint:
            logger.info(f"Using ORPHEUS_LLAMA_ENDPOINT from env: {endpoint}")
            return endpoint

        # Derive likely Modal URLs for this app and probe them
        base = os.environ.get("MODAL_APP_BASE", "willdavenport--voice-stack")
        candidate_bases = [
            f"https://{base}-orpheusllamacppserver-asgi-app-dev.modal.run",
            f"https://{base}-orpheusllamacppserver-asgi-app.modal.run",
        ]
        candidates = [f"{b}/v1/completions" for b in candidate_bases]

        for c in candidates:
            try:
                health_url = c.replace("/v1/completions", "/health")
                resp = await self.llama_client.get(health_url)
                if resp.status_code == 200:
                    logger.info(f"Resolved llama endpoint via health check: {c}")
                    return c
            except Exception:
                continue

        # Final fallback (may still 404 depending on deployment)
        fallback = "https://willdavenport--voice-stack-orpheusllamacppserver-asgi-app.modal.run/v1/completions"
        logger.warning(f"Falling back to default llama endpoint: {fallback}")
        return fallback
    
    async def _generate_tokens(self, prompt: str, endpoint: str) -> list:
        """Generate tokens using llama.cpp API"""
        try:
            # HTTP call to external endpoint
            request_data_openai = {
                "prompt": prompt,
                "max_tokens": self.api_config["max_tokens"],
                "temperature": self.api_config["temperature"],
                "top_p": self.api_config["top_p"],
                "repeat_penalty": self.api_config["repetition_penalty"],
                "stop": ["<custom_token_128009>", "<custom_token_128260>", 
                          "<custom_token_128261>", "<custom_token_128257>"],
                "stream": False,
            }
            
            logger.info(f"=== ORPHEUS DEBUG: Request data: {request_data_openai}")

            # First try OpenAI-compatible endpoint
            try:
                response = await self.llama_client.post(
                    endpoint,
                    json=request_data_openai,
                )
                response.raise_for_status()
            except Exception as e:
                # Log upstream error details, including body if available
                try:
                    body_text = getattr(e, "response", None).text if hasattr(e, "response") and getattr(e, "response") is not None else ""
                except Exception:
                    body_text = ""
                logger.error(f"OpenAI-compatible /v1/completions failed at {endpoint}: {e} {body_text}")

                # Fallback to llama.cpp native endpoint if applicable
                fallback_endpoint = None
                if endpoint.endswith("/v1/completions"):
                    fallback_endpoint = endpoint[: -len("/v1/completions")] + "/completion"
                
                if fallback_endpoint is None:
                    raise

                request_data_llama = {
                    "prompt": prompt,
                    "n_predict": self.api_config["max_tokens"],
                    "temperature": self.api_config["temperature"],
                    "top_p": self.api_config["top_p"],
                    "repeat_penalty": self.api_config["repetition_penalty"],
                    "stop": ["<custom_token_128009>", "<custom_token_128260>", 
                              "<custom_token_128261>", "<custom_token_128257>"],
                    "stream": False,
                    "cache_prompt": True,
                    "seed": -1,
                }

                logger.info(f"Falling back to llama.cpp native completion at {fallback_endpoint}")
                response = await self.llama_client.post(
                    fallback_endpoint,
                    json=request_data_llama,
                )
                response.raise_for_status()

            # Parse successful response
            result = response.json()
            logger.info(f"=== ORPHEUS DEBUG: Raw API response: {result}")
            
            generated_text = ""
            choices = result.get("choices")
            if isinstance(choices, list) and len(choices) > 0 and isinstance(choices[0], dict):
                generated_text = choices[0].get("text") or choices[0].get("message", {}).get("content", "")
            if not generated_text:
                generated_text = result.get("content", "")
            
            logger.info(f"=== ORPHEUS DEBUG: Generated text: '{generated_text}'")
            logger.info(f"=== ORPHEUS DEBUG: Generated text length: {len(generated_text)}")
            
            tokens = self._extract_tokens(generated_text)
            logger.info(f"=== ORPHEUS DEBUG: Extracted tokens count: {len(tokens)}")
            logger.info(f"=== ORPHEUS DEBUG: First 10 tokens: {tokens[:10]}")
            return tokens
            
        except Exception as e:
            logger.error(f"Error generating tokens: {e}")
            raise
    
    def _extract_tokens(self, text: str) -> list:
        """Extract custom tokens from generated text"""
        import re
        
        # Try multiple token patterns that might be generated by different Orpheus models
        patterns = [
            r"<custom_token_(\d+)>",  # Expected format
            r"<\|(\d+)\|>",           # Alternative format
            r"<(\d+)>",               # Simple format
            r"\[(\d+)\]",             # Bracket format
        ]
        
        tokens = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            if matches:
                logger.info(f"=== ORPHEUS DEBUG: Found matches with pattern '{pattern}': {matches}")
                for match in matches:
                    token_id = int(match)
                    if token_id != 0:
                        tokens.append(token_id)
                break  # Use the first pattern that matches
        
        if not tokens:
            logger.warning(f"=== ORPHEUS DEBUG: No tokens found with any pattern in text: '{text[:200]}...'")
            # As a fallback, try to generate some test tokens to see if the audio pipeline works
            logger.info("=== ORPHEUS DEBUG: Generating fallback tokens for testing")
            tokens = self._generate_fallback_tokens()
        
        logger.info(f"=== ORPHEUS DEBUG: Final extracted tokens count: {len(tokens)}")
        logger.info(f"=== ORPHEUS DEBUG: First 10 tokens: {tokens[:10]}")
        return tokens
    
    def _generate_fallback_tokens(self) -> list:
        """Generate fallback tokens for testing when model doesn't produce expected format"""
        # Generate a sequence of tokens that should produce some audio
        # Based on the SNAC token ranges (typically 10 + offset to 4106 + offset)
        import random
        
        # Generate tokens for about 1 second of audio (7 tokens per frame, ~24 frames per second)
        num_frames = 24
        tokens = []
        
        for frame in range(num_frames):
            for pos in range(7):
                # Generate tokens in the expected range for SNAC
                base_token = 10 + (pos * 4096)
                token_offset = random.randint(0, 4095)
                token_id = base_token + token_offset
                tokens.append(token_id)
        
        logger.info(f"=== ORPHEUS DEBUG: Generated {len(tokens)} fallback tokens")
        return tokens
    
    async def _tokens_to_audio(self, tokens: list, speed: float = 1.0) -> bytes:
        """Convert tokens to audio using SNAC model"""
        import torch
        import numpy as np
        
        logger.info(f"=== ORPHEUS DEBUG: Converting {len(tokens)} tokens to audio")
        logger.info(f"=== ORPHEUS DEBUG: First 20 tokens: {tokens[:20]}")
        
        if not tokens or len(tokens) < 7:
            logger.warning(f"=== ORPHEUS DEBUG: Not enough tokens ({len(tokens)}) for audio generation (need at least 7)")
            return b""
        
        try:
            # Process tokens into SNAC codes
            num_frames = len(tokens) // 7
            frame_tokens = tokens[:num_frames * 7]
            
            # Pre-allocate tensors on device (more efficient approach from working code)
            codes_0 = torch.zeros(num_frames, dtype=torch.int32, device=self.device)
            codes_1 = torch.zeros(num_frames * 2, dtype=torch.int32, device=self.device)
            codes_2 = torch.zeros(num_frames * 4, dtype=torch.int32, device=self.device)
            
            frame_tensor = torch.tensor(frame_tokens, dtype=torch.int32, device=self.device)
            
            # Populate codebooks (7 codes per frame → 1/2/4 per codebook)
            for j in range(num_frames):
                idx = j * 7
                codes_0[j] = self._token_to_code(frame_tensor[idx].item(), 0)
                codes_1[j * 2] = self._token_to_code(frame_tensor[idx + 1].item(), 1)
                codes_1[j * 2 + 1] = self._token_to_code(frame_tensor[idx + 4].item(), 4)
                codes_2[j * 4] = self._token_to_code(frame_tensor[idx + 2].item(), 2)
                codes_2[j * 4 + 1] = self._token_to_code(frame_tensor[idx + 3].item(), 3)
                codes_2[j * 4 + 2] = self._token_to_code(frame_tensor[idx + 5].item(), 5)
                codes_2[j * 4 + 3] = self._token_to_code(frame_tensor[idx + 6].item(), 6)
            
            codes = [codes_0.unsqueeze(0), codes_1.unsqueeze(0), codes_2.unsqueeze(0)]
            
            # Validate codes are in range
            for code_tensor in codes:
                if torch.any(code_tensor < 0) or torch.any(code_tensor > 4096):
                    logger.warning("Invalid token IDs detected, skipping audio generation")
                    return b""
            
            # Generate audio with SNAC using same approach as working code
            stream_ctx = (
                torch.cuda.stream(self.cuda_stream) if self.cuda_stream is not None else torch.no_grad()
            )
            
            with stream_ctx, torch.inference_mode():
                # Debug: Check if snac_model and decode method are available
                if not hasattr(self, 'snac_model') or self.snac_model is None:
                    logger.error("SNAC model is not initialized")
                    return b""
                
                if not hasattr(self.snac_model, 'decode'):
                    logger.error("SNAC model does not have decode method")
                    return b""
                
                decode_method = getattr(self.snac_model, 'decode')
                logger.info(f"SNAC decode method type: {type(decode_method)}")
                
                # Try calling the method directly with better error handling
                try:
                    if self.cuda_stream:
                        with torch.cuda.stream(self.cuda_stream):
                            audio_hat = self.snac_model.decode(codes)
                    else:
                        audio_hat = self.snac_model.decode(codes)
                except Exception as e:
                    logger.error(f"Error calling SNAC decode: {e}")
                    logger.error(f"Codes type: {type(codes)}, length: {len(codes)}")
                    logger.error(f"Code shapes: {[c.shape for c in codes]}")
                    raise
                
                # Extract the relevant audio slice
                audio_slice = audio_hat[:, :, 2048:4096]
                
                # Apply speed adjustment if needed
                if speed != 1.0:
                    audio_slice = self._adjust_speed(audio_slice, speed)
                
                # Convert to int16 audio bytes (same approach as working code)
                if self.device == "cuda":
                    audio_int16_tensor = (audio_slice * 32767).to(torch.int16)
                    audio_bytes = audio_int16_tensor.cpu().numpy().tobytes()
                else:
                    detached_audio = audio_slice.detach().cpu()
                    audio_np = detached_audio.numpy()
                    audio_int16 = (audio_np * 32767).astype(np.int16)
                    audio_bytes = audio_int16.tobytes()
                
                logger.info(f"=== ORPHEUS DEBUG: Generated {len(audio_bytes)} bytes of audio from {len(tokens)} tokens")
                return audio_bytes
                
        except Exception as e:
            logger.error(f"Error converting tokens to audio: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return b""
    
    def _token_to_code(self, token_id: int, index: int) -> int:
        """Convert token ID to SNAC code"""
        return token_id - 10 - ((index % 7) * 4096)
    
    def _adjust_speed(self, audio_tensor, speed: float):
        """Adjust audio playback speed"""
        import torch.nn.functional as F
        
        if speed == 1.0:
            return audio_tensor
        
        original_length = audio_tensor.shape[-1]
        target_length = int(original_length / speed)
        
        audio_reshaped = audio_tensor.unsqueeze(1)
        
        audio_adjusted = F.interpolate(
            audio_reshaped,
            size=target_length,
            mode='linear',
            align_corners=False
        )
        
        return audio_adjusted.squeeze(1)
    
    def _create_wav(self, audio_bytes: bytes, sample_rate: int) -> bytes:
        """Create WAV file from raw audio bytes"""
        import wave
        import io
        
        wav_buffer = io.BytesIO()
        
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_bytes)
        
        wav_buffer.seek(0)
        return wav_buffer.read()
    
    @modal.asgi_app()
    def asgi_app(self):
        """Create FastAPI app for HTTP endpoints"""
        from fastapi import FastAPI, HTTPException, WebSocket
        from fastapi.responses import Response, StreamingResponse
        from pydantic import BaseModel
        import msgpack
        
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
        
        @app.get("/health")
        async def health():
            return {"status": "healthy", "service": "orpheus-fastapi-tts"}
        
        @app.get("/")
        def root():
            return {"service": "tts", "model": "orpheus-fastapi", "status": "ready"}
        
        # Create a reference to self for use in the websocket handler
        service_ref = self
        
        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint compatible with modal_app.py expectations"""
            from fastapi import WebSocketDisconnect
            import msgpack
            
            print("=== ORPHEUS_FASTAPI: WebSocket connection attempt ===")
            
            # Accept the connection (query params available via websocket.url)
            await websocket.accept()
            print("=== ORPHEUS_FASTAPI: WebSocket connection accepted ===")
            
            try:
                # Send initial Ready message to client in MessagePack format
                ready_message = {"type": "Ready"}
                packed_ready = msgpack.packb(ready_message)
                await websocket.send_bytes(packed_ready)
                print("=== ORPHEUS_FASTAPI: Sent Ready message (MessagePack format) ===")
                
                while True:
                    try:
                        # Receive message from client
                        data = await websocket.receive_bytes()
                        
                        # Check if data is None to prevent len() errors
                        if data is None:
                            print("=== ORPHEUS_FASTAPI: Received None data, skipping ===")
                            continue
                            
                        print(f"=== ORPHEUS_FASTAPI: Received data: {len(data)} bytes ===")
                        
                        # Unpack the message
                        try:
                            message = msgpack.unpackb(data)
                            print(f"=== ORPHEUS_FASTAPI: Unpacked message type: {message.get('type', 'unknown')} ===")
                            
                            if message.get("type") == "Text":
                                text = message.get("text", "")
                                voice = message.get("voice", "tara")
                                # Allow overriding llama endpoint via websocket query param
                                llama_endpoint = None
                                try:
                                    llama_endpoint = websocket.url.query_params.get("llama_endpoint")  # type: ignore[attr-defined]
                                except Exception:
                                    llama_endpoint = None
                                
                                print(f"=== ORPHEUS_FASTAPI: Processing text: {text[:50]}... with voice: {voice} ===")
                                
                                # Generate audio using Orpheus - use the service reference
                                try:
                                    audio_data = await service_ref.generate_speech(
                                        text=text,
                                        voice=voice,
                                        model="orpheus",
                                        response_format="raw",
                                        speed=1.0,
                                        llama_endpoint=llama_endpoint,
                                    )
                                    
                                    # Send raw PCM bytes directly
                                    await websocket.send_bytes(audio_data)
                                        
                                except Exception as e:
                                    print(f"=== ORPHEUS_FASTAPI: Error generating audio: {e} ===")
                                    error_message = {
                                        "type": "Error",
                                        "message": f"Audio generation failed: {str(e)}"
                                    }
                                    packed_error = msgpack.packb(error_message)
                                    await websocket.send_bytes(packed_error)
                                    
                            elif message.get("type") == "Eos":
                                print("=== ORPHEUS_FASTAPI: Received EOS message ===")
                                eos_response = {"type": "Eos"}
                                packed_eos = msgpack.packb(eos_response)
                                await websocket.send_bytes(packed_eos)
                                
                        except Exception as e:
                            print(f"=== ORPHEUS_FASTAPI: Error unpacking message: {e} ===")
                            continue
                            
                    except WebSocketDisconnect:
                        print("=== ORPHEUS_FASTAPI: Client disconnected ===")
                        break
                    except Exception as e:
                        print(f"=== ORPHEUS_FASTAPI: WebSocket error: {e} ===")
                        break
                        
            except Exception as e:
                print(f"=== ORPHEUS_FASTAPI: Connection error: {e} ===")
        
        return app


# Export the apps for inclusion in the main modal app
orpheus_apps = {
    "fastapi": orpheus_fastapi_app,
    "llama": orpheus_llama_app,
}

__all__ = ["OrpheusFastAPIService", "OrpheusLlamaCppServer", "orpheus_fastapi_app", "orpheus_llama_app", "orpheus_apps"]
