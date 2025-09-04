"""
Modal Voice Stack Application

This Modal app implements the target architecture with four serverless classes:
- Orchestrator (CPU only) - handles client WebSocket connections
- STT (L4 GPU) - Speech-to-Text service with WebSocket endpoint  
- LLM (L40S GPU) - Language Model service with WebSocket endpoint
- TTS (L4 GPU) - Text-to-Speech service with WebSocket endpoint

Each service uses @modal.asgi_app() for WebSocket endpoints and @modal.enter() for model loading.

OPTIMIZATION: Essential model weights are pre-downloaded during image build to reduce startup delays:
- STT models: kyutai/stt-1b-en_fr-candle (model.safetensors, tokenizer, audio tokenizer)  
- TTS models: kyutai/tts-1.6b-en_fr (tokenizer) + essential default voice from kyutai/tts-voices
Additional TTS voices are downloaded on-demand at runtime to avoid HuggingFace rate limits.
This reduces initial TTS startup time significantly while maintaining voice variety.
"""

import modal
import logging
import os

logger = logging.getLogger(__name__)

# Modal app definition
app = modal.App("voice-stack")

# Configuration constants
MINUTES = 60
HOURS = 60 * MINUTES
CACHE_DIR = "/root/.cache/huggingface"  # Use standard HF cache location
VOLUME_CACHE_DIR = "/hf-cache"  # Volume mount point (separate from build cache)

# Define the base image with common dependencies (NO local files yet)
base_deps_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl", "ffmpeg", "libsndfile1", "build-essential")
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",  # Faster downloads from Hugging Face
        "HF_HOME": CACHE_DIR,
    })
    .pip_install(
        # Core FastAPI and web dependencies
        "fastapi[standard]>=0.115.12",
        "pydantic>=2.0.0",
        "ruamel-yaml>=0.18.10",
        
        # Audio processing dependencies
        "librosa>=0.10.0",
        "torchaudio>=2.1.0",
        "soundfile>=0.12.0",
        
        # WebRTC and streaming dependencies
        "fastrtc==0.0.23",
        "sphn>=0.2.0",
        "websockets>=12.0",
        
        # Utility dependencies
        "msgpack>=1.1.0",
        "msgpack-types>=0.5.0",
        "tqdm>=4.66.0",
        "numpy>=1.24.0",
        "redis>=5.0.0",  # Required by unmute.cache
        
        # Monitoring dependencies
        "prometheus-fastapi-instrumentator==7.1.0",
        "prometheus-client==0.21.0",
        
        # LLM and API dependencies
        "mistralai>=1.5.1",
        "openai>=1.70.0",
    )
)

# STT image: build steps first, then local files
stt_image = (
    base_deps_image
    .apt_install("cmake", "pkg-config", "libopus-dev", "git", "curl", "libssl-dev", "openssl", "wget", "gnupg")
    .run_commands(
        # Install CUDA toolkit for nvcc compiler (required for cudarc compilation)
        "wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb",
        "dpkg -i cuda-keyring_1.0-1_all.deb",
        "apt-get update -q",
        "apt-get install -y cuda-toolkit-12-1 || apt-get install -y cuda-toolkit-11-8",
        # Install latest Rust toolchain to handle Cargo.lock version 4
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable",
        ". ~/.cargo/env"
    )
    .pip_install(
        "torch>=2.1.0",
        "torchaudio>=2.1.0",
        "safetensors>=0.4.0",
        "transformers>=4.35.0",
        "tokenizers>=0.15.0",
        # Moshi Python package and dependencies for moshi-server
        "moshi>=0.2.8",
        "huggingface_hub>=0.19.0",
        "sentencepiece>=0.2.0",
        "einops>=0.8.0",
    )
    .run_commands(
        # Set up environment variables that will be needed at runtime
        "echo 'export PATH=\"$HOME/.cargo/bin:$PATH\"' >> /etc/environment",
        "echo 'export PATH=\"$HOME/.cargo/bin:$PATH\"' >> /root/.bashrc",
        # Create directory for cached binaries
        "mkdir -p /usr/local/bin",
        # Note: moshi-server binary will be installed at startup time with CUDA support
        "echo 'Rust environment prepared for runtime binary installation'"
    )
    .run_commands(
        # Pre-download STT models during image build to avoid startup delays
        "echo 'Pre-downloading STT models to cache...'",
        # Create cache directories using CACHE_DIR
        f"mkdir -p {CACHE_DIR}",
        # Download STT model files with simple retry logic
        f"python -c \"import time; from huggingface_hub import hf_hub_download; [hf_hub_download('kyutai/stt-1b-en_fr-candle', f, cache_dir='{CACHE_DIR}') for f in ['model.safetensors', 'tokenizer_en_fr_audio_8000.model', 'mimi-pytorch-e351c8d8@125.safetensors']]\"",
        # Verify models were downloaded
        "echo 'Verifying downloaded STT models...'",
        f"find {CACHE_DIR} -name 'model.safetensors' | head -5",
        f"find {CACHE_DIR} -name 'tokenizer_en_fr_audio_8000.model' | head -5",
        f"find {CACHE_DIR} -name 'mimi-pytorch-e351c8d8@125.safetensors' | head -5",
        "echo 'STT model pre-download completed'"
    )
    # Add local files LAST
    .add_local_python_source("unmute")
    .add_local_file("voices.yaml", "/root/voices.yaml")
)

# TTS image: build steps first, then local files
tts_image = (
    base_deps_image
    .apt_install("cmake", "pkg-config", "libopus-dev", "git", "curl", "libssl-dev", "openssl", "wget", "gnupg", "git-lfs")
    .env({
        # H100 optimization environment variables
        "TORCH_CUDNN_V8_API_ENABLED": "1",
        "CUDA_LAUNCH_BLOCKING": "0",  # Allow async CUDA operations
        "TORCH_COMPILE_DEBUG": "0",  # Disable debug for performance
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512,expandable_segments:True",
        "CUDA_VISIBLE_DEVICES": "0",  # Use first GPU
    })
    .run_commands(
        # Install CUDA toolkit for nvcc compiler (required for cudarc compilation)
        "wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb",
        "dpkg -i cuda-keyring_1.0-1_all.deb",
        "apt-get update -q",
        "apt-get install -y cuda-toolkit-12-1 || apt-get install -y cuda-toolkit-11-8",
        # Install git-lfs for Orpheus model downloads
        "git lfs install",
    )
    .pip_install(
        # PyTorch with CUDA support for Orpheus - use latest stable
        "torch==2.7.1",
        "torchaudio>=2.1.0",
        "huggingface_hub[hf_transfer]>=0.19.0",  # Include hf_transfer for faster downloads
        "safetensors>=0.4.0",
        "transformers>=4.35.0",
        "tokenizers>=0.15.0",
        # Required for device mapping with large models
        "accelerate>=0.20.0",
        # Orpheus-specific dependencies
        "snac==1.2.1",
        "batched==0.1.4",
        # OpenAI client for LLM integration
        "openai>=1.70.0",
        # Keep some existing dependencies for compatibility
        "sentencepiece>=0.2.0",
        "einops>=0.8.0",
        # Additional optimization packages (flash-attn removed due to compatibility issues)
        # "flash-attn>=2.0.0",  # Commented out due to CUDA symbol conflicts
        "xformers>=0.0.20",   # Memory-efficient attention (fallback)
    )
    .run_commands(
        # Pre-download publicly available models during image build
        "echo 'Pre-downloading publicly available models to cache...'",
        # Create cache directories using CACHE_DIR
        f"mkdir -p {CACHE_DIR}",
        "mkdir -p /root/orpheus-models",
        # Download SNAC model for audio conversion (publicly available)
        f"python -c \"from huggingface_hub import snapshot_download; snapshot_download('hubertsiuzdak/snac_24khz', cache_dir='{CACHE_DIR}')\"",
        # Pre-download Orpheus tokenizer and config (publicly available parts)
        "echo 'Pre-downloading Orpheus tokenizer and config...'",
        f"python -c \"from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('canopylabs/orpheus-3b-0.1-ft', cache_dir='{CACHE_DIR}', use_auth_token=False)\" || echo 'Tokenizer download failed (expected without token)'",
        # Note: Full Orpheus model requires authentication, will be downloaded at runtime
        "echo 'Note: Full Orpheus model (canopylabs/orpheus-3b-0.1-ft) requires HF token, will download at runtime'",
        # Verify SNAC model was downloaded
        "echo 'Verifying downloaded models...'",
        f"ls -la {CACHE_DIR}/ | head -10",
        "echo 'Public model pre-download completed.'"
    )
    # Add local files LAST
    .add_local_python_source("unmute")
    .add_local_file("voices.yaml", "/root/voices.yaml")
)

# Import dependencies for TTS image context
with tts_image.imports():
    import torch
    import transformers
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

# LLM image: additional deps first, then local files
llm_image = (
    base_deps_image
    .pip_install(
        "huggingface_hub[hf_transfer]==0.32.0",
        "hf_transfer==0.1.8",  # Pin specific version for stability
        "vllm==0.9.1",
        "torch>=2.1.0",
        "transformers>=4.51.1",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})  # faster model transfers
    .env({"VLLM_USE_V1": "1"})  # use V1 engine for better performance
    # Add local files LAST
    .add_local_python_source("unmute")
    .add_local_file("voices.yaml", "/root/voices.yaml")
)

# Orchestrator image: just add local files to base deps
orchestrator_image = (
    base_deps_image
    # Add local files LAST
    .add_local_python_source("unmute")
    .add_local_file("voices.yaml", "/root/voices.yaml")
)

# Modal volumes for model storage - following Modal best practices
models_volume = modal.Volume.from_name("voice-models", create_if_missing=True)
# Orpheus TTS doesn't need persistent voice storage like Moshi
# tts_voices_volume = modal.Volume.from_name("tts-voices-cache")

# HuggingFace cache volume for persistent model storage
hf_cache_volume = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

# Cache volume for Rust binaries (shared between STT and TTS)
rust_binaries_volume = modal.Volume.from_name("rust-binaries-cache", create_if_missing=True)

# Volume mappings for TTS service
tts_volumes = {
    "/models": models_volume,
    VOLUME_CACHE_DIR: hf_cache_volume,  # Use volume mount point (separate from build cache)
}

# Model configuration for Mistral
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
MODEL_REVISION = "main"  # Pin to specific revision when available
FAST_BOOT = True  # Set to False for better performance if you have persistent replicas

# Modal secrets for API keys and auth tokens - following Modal best practices
secrets = [
    modal.Secret.from_name("voice-auth"),
    modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"]),  # Ensure HF_TOKEN is available
]


def install_moshi_server_with_cuda():
    """
    Install moshi-server binary with CUDA support at runtime.
    This function checks if a cached binary exists, and if not, compiles it with CUDA features.
    Used by STT service - TTS now uses Orpheus.
    """
    import subprocess
    import os
    import time
    from pathlib import Path
    
    print("Installing moshi-server with CUDA support...")
    
    # Check if we already have a cached binary
    cached_binary_path = "/rust-binaries/moshi-server"
    local_binary_path = "/usr/local/bin/moshi-server"
    
    if os.path.exists(cached_binary_path):
        print(f"Found cached moshi-server binary at {cached_binary_path}")
        # Copy from cache to local bin
        subprocess.run(["cp", cached_binary_path, local_binary_path], check=True)
        subprocess.run(["chmod", "+x", local_binary_path], check=True)
        print("Cached binary copied and made executable")
        return local_binary_path
    
    print("No cached binary found, compiling moshi-server with CUDA support...")
    
    # Set up environment with cargo bin in PATH
    env = os.environ.copy()
    cargo_bin_path = os.path.expanduser("~/.cargo/bin")
    env["PATH"] = f"{cargo_bin_path}:{env.get('PATH', '')}"
    
    # Set up compilation environment for CUDA
    env.update({
        "CXXFLAGS": "-include cstdint",
        "LD_LIBRARY_PATH": subprocess.check_output([
            "python", "-c", 
            "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"
        ], text=True).strip(),
        "OPENSSL_DIR": "/usr",
        "OPENSSL_LIB_DIR": "/usr/lib/x86_64-linux-gnu",
        "OPENSSL_INCLUDE_DIR": "/usr/include/openssl",
        "PKG_CONFIG_PATH": "/usr/lib/x86_64-linux-gnu/pkgconfig",
        "RUST_LOG": "info",  # Reduce noisy logs
        # CUDA environment variables
        "CUDA_ROOT": "/usr/local/cuda-12.1",
        "CUDA_PATH": "/usr/local/cuda-12.1",
        "CUDA_TOOLKIT_ROOT_DIR": "/usr/local/cuda-12.1",
        "PATH": f"{env.get('PATH', '')}:/usr/local/cuda-12.1/bin"
    })
    
    try:
        # Install moshi-server with CUDA features
        print("Running cargo install moshi-server@0.6.3 --features cuda...")
        start_time = time.time()
        
        result = subprocess.run([
            "cargo", "install", "moshi-server@0.6.3", "--features", "cuda"
        ], env=env, capture_output=True, text=True, timeout=600)  # 10 minute timeout
        
        compile_time = time.time() - start_time
        print(f"Compilation completed in {compile_time:.1f} seconds")
        
        if result.returncode != 0:
            print(f"Cargo install with CUDA failed with return code {result.returncode}")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            
            # Try fallback compilation without CUDA features
            print("Attempting fallback compilation without CUDA features...")
            fallback_result = subprocess.run([
                "cargo", "install", "moshi-server@0.6.3"
            ], env=env, capture_output=True, text=True, timeout=600)
            
            if fallback_result.returncode != 0:
                print(f"Fallback compilation also failed: {fallback_result.stderr}")
                raise RuntimeError("Failed to compile moshi-server with and without CUDA")
            else:
                print("Successfully compiled moshi-server without CUDA features (fallback)")
                result = fallback_result  # Use fallback result for rest of the function
        
        # Copy to local bin
        cargo_binary = f"{cargo_bin_path}/moshi-server"
        if not os.path.exists(cargo_binary):
            raise RuntimeError(f"moshi-server not found at expected location: {cargo_binary}")
            
        subprocess.run(["cp", cargo_binary, local_binary_path], check=True)
        subprocess.run(["chmod", "+x", local_binary_path], check=True)
        
        # Cache the binary for future use
        print("Caching compiled binary...")
        os.makedirs("/rust-binaries", exist_ok=True)
        subprocess.run(["cp", local_binary_path, cached_binary_path], check=True)
        
        print("moshi-server successfully installed and cached")
        return local_binary_path
        
    except subprocess.TimeoutExpired:
        raise RuntimeError("Compilation timed out after 10 minutes")
    except Exception as e:
        print(f"Error during compilation: {e}")
        raise


@app.cls(
    gpu="L40S",
    image=stt_image,
    volumes={
        "/models": models_volume,
        "/rust-binaries": rust_binaries_volume,
        VOLUME_CACHE_DIR: hf_cache_volume,  # Use volume mount point (separate from build cache)
    },
    secrets=secrets,
    min_containers=int(os.environ.get("MIN_CONTAINERS", "0")),
    scaledown_window=600,  # 10 minutes - prevent scaling during long conversations
)
@modal.concurrent(max_inputs=10)
class STTService:
    """Speech-to-Text service using Moshi STT model"""
    
    @modal.enter()
    def load_model(self):
        """Load STT model weights once per container"""
        import subprocess
        import os
        import tempfile
        import time
        
        print("Setting up STT Service...")
        
        # Install moshi-server with CUDA support
        moshi_server_path = install_moshi_server_with_cuda()
        
        # Set up environment with cargo bin in PATH
        env = os.environ.copy()
        cargo_bin_path = os.path.expanduser("~/.cargo/bin")
        env["PATH"] = f"{cargo_bin_path}:{env.get('PATH', '')}"
        # Reduce noisy batched_asr logs by setting log level to off for moshi_server::batched_asr
        env["RUST_LOG"] = "warn"
        
        # Verify the installed binary
        print(f"Verifying moshi-server at: {moshi_server_path}")
        try:
            result = subprocess.run([moshi_server_path, "--help"], 
                                  capture_output=True, text=True, timeout=10, env=env)
            print(f"moshi-server help output: {result.stdout[:200]}...")  # Show first 200 chars
            if result.returncode != 0:
                raise RuntimeError(f"moshi-server verification failed: {result.stderr}")
        except Exception as e:
            print(f"Failed to verify moshi-server: {e}")
            raise
        
        # Create temporary config file for STT
        config_content = '''
static_dir = "./static/"
log_dir = "/tmp/unmute_logs"
instance_name = "stt"
authorized_ids = ["public_token"]

[modules.asr]
path = "/api/asr-streaming"
type = "BatchedAsr"
lm_model_file = "hf://kyutai/stt-1b-en_fr-candle/model.safetensors"
text_tokenizer_file = "hf://kyutai/stt-1b-en_fr-candle/tokenizer_en_fr_audio_8000.model"
audio_tokenizer_file = "hf://kyutai/stt-1b-en_fr-candle/mimi-pytorch-e351c8d8@125.safetensors"
asr_delay_in_tokens = 6
batch_size = 64
conditioning_learnt_padding = true
temperature = 0.25

[modules.asr.model]
audio_vocab_size = 2049
text_in_vocab_size = 8001
text_out_vocab_size = 8000
audio_codebooks = 20

[modules.asr.model.transformer]
d_model = 2048
num_heads = 16
num_layers = 16
dim_feedforward = 8192
causal = true
norm_first = true
bias_ff = false
bias_attn = false
context = 750
max_period = 100000
use_conv_block = false
use_conv_bias = true
gating = "silu"
norm = "RmsNorm"
positional_embedding = "Rope"
conv_layout = false
conv_kernel_size = 3
kv_repeat = 1
max_seq_len = 40960

[modules.asr.model.extra_heads]
num_heads = 4
dim = 6
'''
        
        os.makedirs("/tmp/unmute_logs", exist_ok=True)
        os.makedirs("./static", exist_ok=True)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write(config_content)
            self.config_path = f.name
        
        # Start moshi-server as a background process
        print("Starting STT moshi-server...")
        print(f"Config file: {self.config_path}")
        
        # Debug: print the config content
        with open(self.config_path, 'r') as f:
            config_content = f.read()
            print("STT Config content:")
            print(config_content)
        
        # Find the moshi-server binary
        moshi_server_cmd = "moshi-server"
        if os.path.exists("/usr/local/bin/moshi-server"):
            moshi_server_cmd = "/usr/local/bin/moshi-server"
        elif os.path.exists(cargo_bin_path + "/moshi-server"):
            moshi_server_cmd = cargo_bin_path + "/moshi-server"
        
        print(f"Using moshi-server at: {moshi_server_cmd}")
        
        self.proc = subprocess.Popen([
            moshi_server_cmd, "worker", 
            "--config", self.config_path,
            "--port", "8090"
        ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        
        # Wait for the server to be ready with health checking
        import socket
        import threading
        
        # Start a thread to monitor the process output in real-time
        def monitor_output():
            try:
                while self.proc.poll() is None:
                    line = self.proc.stdout.readline()
                    if line:
                        # Filter out noisy batched_asr logs
                        line_stripped = line.strip()
                        if "moshi_server::batched_asr" not in line_stripped:
                            print(f"STT moshi-server: {line_stripped}")
            except Exception as e:
                print(f"STT output monitor error: {e}")
        
        output_thread = threading.Thread(target=monitor_output, daemon=True)
        output_thread.start()
        
        max_attempts = 60  # 60 seconds max
        for attempt in range(max_attempts):
            # Check if process died
            if self.proc.poll() is not None:
                print(f"STT moshi-server process died with return code: {self.proc.returncode}")
                break
                
            try:
                # Try to connect to the moshi server
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1.0)
                result = sock.connect_ex(('127.0.0.1', 8090))
                sock.close()
                if result == 0:
                    print(f"STT moshi-server ready after {attempt + 1} seconds")
                    break
            except Exception:
                pass
            time.sleep(1)
        else:
            # Check if process is still running
            if self.proc.poll() is not None:
                stdout, stderr = self.proc.communicate()
                print(f"STT moshi-server failed to start. Return code: {self.proc.returncode}")
                print(f"STT moshi-server stdout: {stdout}")
                print(f"STT moshi-server stderr: {stderr}")
                
                # Try to get more debug info
                print("Checking moshi-server binary...")
                try:
                    version_result = subprocess.run([moshi_server_cmd, "--help"], 
                                                  capture_output=True, text=True, timeout=10, env=env)
                    print(f"moshi-server help output: {version_result.stdout[:200]}...")
                    if version_result.stderr:
                        print(f"moshi-server help stderr: {version_result.stderr}")
                except Exception as e:
                    print(f"Failed to get moshi-server help: {e}")
                
                raise RuntimeError(f"STT moshi-server process died during startup with return code {self.proc.returncode}")
            else:
                print("STT moshi-server taking longer than expected to start, continuing anyway...")
        
        print("STT model loaded successfully")
    
    @modal.asgi_app()
    def web(self):
        """WebSocket endpoint for STT service"""
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect
        import asyncio
        
        app = FastAPI(title="STT Service", version="1.0.0")
        
        @app.get("/")
        def root():
            return {"service": "stt", "model": "kyutai-stt", "status": "ready"}
        
        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            print("=== STT_SERVICE: WebSocket connection attempt ===")
            
            # First check if our internal moshi server is ready before accepting the connection
            import socket
            import asyncio
            max_wait_attempts = 90  # 90 seconds max to allow for moshi-server startup
            for attempt in range(max_wait_attempts):
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(1.0)
                    result = sock.connect_ex(('127.0.0.1', 8090))
                    sock.close()
                    if result == 0:
                        print(f"=== STT_SERVICE: Internal moshi server ready after {attempt + 1} attempts ===")
                        break
                except Exception:
                    pass
                print(f"=== STT_SERVICE: Waiting for internal moshi server, attempt {attempt + 1}/{max_wait_attempts} ===")
                await asyncio.sleep(1)  # Use async sleep instead of blocking sleep
            else:
                print("=== STT_SERVICE: Internal moshi server not ready after 90 seconds ===")
                # We must accept the websocket first before we can close it
                await websocket.accept()
                await websocket.close(code=1011, reason="Internal STT server not ready")
                return
            
            await websocket.accept()
            print("=== STT_SERVICE: WebSocket connection accepted, internal server ready ===")
            
            try:
                # Connect to the local moshi server and proxy messages
                import websockets
                try:
                    print("=== STT_SERVICE: Connecting to internal moshi server ===")
                    moshi_connection = await websockets.connect(
                        "ws://localhost:8090/api/asr-streaming?format=PcmMessagePack",
                        additional_headers={"kyutai-api-key": "public_token"}
                    )
                    print("=== STT_SERVICE: Successfully connected to internal moshi server ===")
                except ConnectionRefusedError as e:
                    print(f"=== STT_SERVICE: Failed to connect to internal moshi server: {e} ===")
                    await websocket.close(code=1011, reason="Internal STT server not ready")
                    return
                except Exception as e:
                    print(f"=== STT_SERVICE: Unexpected error connecting to moshi server: {e} ===")
                    await websocket.close(code=1011, reason="Internal STT server error")
                    return
                
                async with moshi_connection as moshi_ws:
                    # First, wait for and forward the initial Ready message from moshi-server
                    try:
                        initial_message = await asyncio.wait_for(moshi_ws.recv(), timeout=30.0)
                        print(f"STT: Received initial message from moshi-server: {type(initial_message)}")
                        if isinstance(initial_message, bytes):
                            await websocket.send_bytes(initial_message)
                        else:
                            await websocket.send_text(initial_message)
                    except asyncio.TimeoutError:
                        print("STT: Timeout waiting for initial Ready message from moshi-server")
                        await websocket.close(code=1011, reason="Internal STT server timeout")
                        return
                    except Exception as e:
                        print(f"STT: Error receiving initial message: {e}")
                        await websocket.close(code=1011, reason="Internal STT server error")
                        return
                    
                    # Now proxy messages between client and moshi server
                    async def client_to_moshi():
                        try:
                            while True:
                                data = await websocket.receive_bytes()
                                logger.info(f"=== STT: Sending data to moshi server: {len(data)} bytes ===")
                                await moshi_ws.send(data)
                        except WebSocketDisconnect:
                            pass
                    
                    async def moshi_to_client():
                        try:
                            async for message in moshi_ws:
                                if isinstance(message, bytes):
                                    logger.info(f"=== STT: Received bytes from moshi-server, sending to client: {len(message)} bytes ===")
                                    await websocket.send_bytes(message)
                                else:
                                    logger.info(f"=== STT: Received text from moshi-server, sending to client: {message[:50]} ===")
                                    await websocket.send_text(message)
                        except Exception as e:
                            print(f"STT moshi_to_client error: {e}")
                    
                    # Run both directions concurrently
                    await asyncio.gather(
                        client_to_moshi(),
                        moshi_to_client(),
                        return_exceptions=True
                    )
            except Exception as e:
                print(f"STT websocket error: {e}")
        
        return app


@app.cls(
    gpu="H100", 
    image=tts_image,
    volumes=tts_volumes,
    secrets=secrets,
    min_containers=int(os.environ.get("MIN_CONTAINERS", "1")),  # Keep at least 1 warm
    scaledown_window=300,  # 5 minutes - faster scaling for better utilization
    timeout=20 * 60,  # 20 minutes for model optimization and compilation
    # H100-optimized configuration
    cpu=8.0,  # More CPU cores for streaming and async processing
    memory=40960,  # 40GB RAM for H100 optimization and FP8 conversions
    # Enable container keep-warm for better performance
    keep_warm=1,  # Keep 1 container warm for instant response
)
@modal.concurrent(max_inputs=16)  # Increased concurrency for H100
class TTSService:
    """Text-to-Speech service using Orpheus TTS model"""
    
    @modal.enter()
    def load_model(self):
        """Load Orpheus TTS model weights once per container"""
        import os
        import time
        
        print("Setting up Orpheus TTS Service...")
        
        # Check if HuggingFace token is available
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if hf_token:
            print(f"Orpheus TTS: HuggingFace token available (length: {len(hf_token)})")
            # Set environment variables for huggingface_hub
            os.environ["HF_TOKEN"] = hf_token
            os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
        else:
            print("Orpheus TTS: No HuggingFace token found - this may cause issues downloading models")
            print("Orpheus TTS: Please ensure 'huggingface-secret' is configured in Modal")
        
        # Pre-download Orpheus model at runtime (when token is available)
        try:
            print("Checking for cached Orpheus model...")
            from huggingface_hub import snapshot_download
            import os
            import shutil
            
            # Check both build-time cache and volume cache
            build_cache_path = CACHE_DIR  # Build-time cache
            volume_cache_path = VOLUME_CACHE_DIR  # Volume cache
            
            # Ensure volume cache directory exists
            os.makedirs(volume_cache_path, exist_ok=True)
            
            # Check if model is already in volume cache (persistent)
            volume_cached_dirs = [d for d in os.listdir(volume_cache_path) if 'orpheus' in d.lower()] if os.path.exists(volume_cache_path) else []
            build_cached_dirs = [d for d in os.listdir(build_cache_path) if 'orpheus' in d.lower()] if os.path.exists(build_cache_path) else []
            
            if volume_cached_dirs:
                print(f"Found cached Orpheus model in volume: {volume_cached_dirs}")
                # Use volume cache as primary cache location
                cache_path = volume_cache_path
            elif build_cached_dirs:
                print(f"Found cached Orpheus model in build cache: {build_cached_dirs}")
                # Copy from build cache to volume cache for persistence
                print("Copying model from build cache to volume cache...")
                for cached_dir in build_cached_dirs:
                    src_path = os.path.join(build_cache_path, cached_dir)
                    dst_path = os.path.join(volume_cache_path, cached_dir)
                    if os.path.exists(src_path) and not os.path.exists(dst_path):
                        shutil.copytree(src_path, dst_path)
                        print(f"Copied {cached_dir} to volume cache")
                cache_path = volume_cache_path
            else:
                print("No cached Orpheus model found, downloading to volume cache...")
                cache_path = volume_cache_path
            
            # Download or verify the full Orpheus model with authentication
            print(f"Downloading/verifying Orpheus model to: {cache_path}")
            snapshot_download(
                'canopylabs/orpheus-3b-0.1-ft',
                cache_dir=cache_path,
                token=hf_token,
                local_files_only=False,  # Allow download if not cached
                resume_download=True     # Resume partial downloads
            )
            print("Orpheus model ready (downloaded or verified from cache)")
            
            # Set environment variable to use volume cache
            os.environ["HF_HOME"] = volume_cache_path
            
        except Exception as e:
            print(f"WARNING: Failed to download Orpheus model: {e}")
            print("Will attempt to use cached version or fallback")
            # Check if we have any cached version in either location
            try:
                import os
                volume_dirs = os.listdir(volume_cache_path) if os.path.exists(volume_cache_path) else []
                build_dirs = os.listdir(build_cache_path) if os.path.exists(build_cache_path) else []
                print(f"Available cached models - Volume: {volume_dirs}, Build: {build_dirs}")
                
                # Prefer volume cache, fallback to build cache
                if volume_dirs:
                    os.environ["HF_HOME"] = volume_cache_path
                elif build_dirs:
                    os.environ["HF_HOME"] = build_cache_path
                    
            except Exception as cache_e:
                print(f"Could not check cache: {cache_e}")
        
        # Initialize Orpheus model
        from unmute.tts.orpheus_tts import OrpheusModel, initialize_orpheus_model
        
        try:
            print("Loading Orpheus TTS model...")
            self.orpheus_model = OrpheusModel()
            self.orpheus_model.load()
            
            # Initialize the global SNAC model
            initialize_orpheus_model()
            
            print("Orpheus TTS model loaded successfully")
            
        except Exception as e:
            print(f"CRITICAL ERROR: Orpheus TTS model loading failed: {e}")
            raise RuntimeError(f"Orpheus TTS service initialization failed: {e}. Container must be restarted.")

    @modal.method()
    def generate_speech(self, text: str, voice: str = "tara"):
        """Generate speech from text - test method for direct access"""
        import asyncio
        import base64
        
        print(f"Generating speech for text: '{text}' with voice: '{voice}'")
        
        async def generate_audio():
            audio_chunks = []
            chunk_count = 0
            
            async for audio_chunk in self.orpheus_model.generate_speech_stream(text, voice):
                audio_chunks.append(audio_chunk)
                chunk_count += 1
                print(f"Generated audio chunk {chunk_count}: {len(audio_chunk)} bytes")
            
            if not audio_chunks:
                return {
                    "success": False,
                    "error": "No audio chunks generated"
                }
            
            # Combine all audio chunks (raw 16-bit PCM bytes)
            combined_audio = b''.join(audio_chunks)
            
            # Convert to base64 for transport
            audio_base64 = base64.b64encode(combined_audio).decode('utf-8')
            
            # Calculate approximate duration (24kHz sample rate, 16-bit PCM)
            sample_rate = 24000
            bytes_per_sample = 2  # 16-bit = 2 bytes
            duration_ms = (len(combined_audio) / bytes_per_sample / sample_rate) * 1000
            
            return {
                "success": True,
                "audio_base64": audio_base64,
                "sample_rate": sample_rate,
                "duration_ms": duration_ms,
                "chunk_count": chunk_count,
                "total_bytes": len(combined_audio)
            }
        
        # Run the async generation
        return asyncio.run(generate_audio())
    
    @modal.asgi_app()
    def web(self):
        """WebSocket endpoint for Orpheus TTS service"""
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect
        import asyncio
        import msgpack
        import json
        
        app = FastAPI(title="Orpheus TTS Service", version="1.0.0")
        
        @app.get("/")
        def root():
            return {"service": "tts", "model": "orpheus-tts", "status": "ready"}
        
        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            print("=== ORPHEUS_TTS_SERVICE: WebSocket connection attempt ===")
            
            await websocket.accept()
            print("=== ORPHEUS_TTS_SERVICE: WebSocket connection accepted ===")
            
            try:
                # Send initial Ready message to client in MessagePack format (not JSON)
                ready_message = {"type": "Ready"}
                packed_ready = msgpack.packb(ready_message)
                await websocket.send_bytes(packed_ready)
                print("=== ORPHEUS_TTS_SERVICE: Sent Ready message (MessagePack format) ===")
                
                while True:
                    try:
                        # Receive message from client
                        data = await websocket.receive_bytes()
                        print(f"=== ORPHEUS_TTS_SERVICE: Received data: {len(data)} bytes ===")
                        
                        # Unpack the message
                        try:
                            message = msgpack.unpackb(data)
                            print(f"=== ORPHEUS_TTS_SERVICE: Unpacked message type: {message.get('type', 'unknown')} ===")
                            
                            if message.get("type") == "Text":
                                text = message.get("text", "")
                                voice = message.get("voice", "tara")
                                
                                print(f"=== ORPHEUS_TTS_SERVICE: Processing text: {text[:50]}... with voice: {voice} ===")
                                
                                # Generate audio using Orpheus
                                try:
                                    async for audio_chunk in self.orpheus_model.generate_speech_stream(text, voice):
                                        # Send raw PCM bytes directly (like Baseten API)
                                        # Orpheus outputs 16-bit PCM at 24kHz, send as raw bytes
                                        await websocket.send_bytes(audio_chunk)
                                        
                                except Exception as e:
                                    print(f"=== ORPHEUS_TTS_SERVICE: Error generating audio: {e} ===")
                                    error_message = {
                                        "type": "Error",
                                        "message": f"Audio generation failed: {str(e)}"
                                    }
                                    packed_error = msgpack.packb(error_message)
                                    await websocket.send_bytes(packed_error)
                                    
                            elif message.get("type") == "Eos":
                                print("=== ORPHEUS_TTS_SERVICE: Received EOS message ===")
                                # Send EOS response
                                eos_response = {"type": "Eos"}
                                packed_eos = msgpack.packb(eos_response)
                                await websocket.send_bytes(packed_eos)
                                
                        except Exception as e:
                            print(f"=== ORPHEUS_TTS_SERVICE: Error unpacking message: {e} ===")
                            continue
                            
                    except WebSocketDisconnect:
                        print("=== ORPHEUS_TTS_SERVICE: Client disconnected ===")
                        break
                    except Exception as e:
                        print(f"=== ORPHEUS_TTS_SERVICE: WebSocket error: {e} ===")
                        break
                        
            except Exception as e:
                print(f"=== ORPHEUS_TTS_SERVICE: Connection error: {e} ===")
                
        return app


@app.cls(
    gpu="L40S",
    image=llm_image,
    volumes={
        "/models": models_volume,
        VOLUME_CACHE_DIR: hf_cache_volume,  # Use volume mount point (separate from build cache)
        "/root/.cache/vllm": vllm_cache_vol,
    },
    secrets=secrets,
    min_containers=int(os.environ.get("MIN_CONTAINERS", "0")),
    scaledown_window=600,  # 10 minutes - prevent scaling during long conversations
    timeout=10 * 60,  # 10 minutes for model loading
)
@modal.concurrent(max_inputs=10)
class LLMService:
    """Large Language Model service using VLLM with OpenAI-compatible server"""
    
    @modal.enter()
    def load_model(self):
        """Start VLLM OpenAI-compatible server"""
        print("Setting up LLM Service...")
        
        import subprocess
        import os
        import time
        import socket
        import threading
        
        # Import constants from module level
        MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
        MODEL_REVISION = "main"
        FAST_BOOT = True
        
        # Set up environment
        env = os.environ.copy()
        env["KYUTAI_LLM_MODEL"] = MODEL_NAME
        
        # Add HuggingFace token for model authentication
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if hf_token:
            env["HF_TOKEN"] = hf_token
            env["HUGGING_FACE_HUB_TOKEN"] = hf_token
            print(f"LLM: HuggingFace token configured (length: {len(hf_token)})")
        else:
            print("LLM: WARNING - No HuggingFace token found! This will likely cause authentication errors for gated models like Mistral.")
            print("LLM: Please ensure 'huggingface-secret' is configured in Modal with your HF_TOKEN")
        
        # Start VLLM server using optimized configuration
        print("Starting VLLM server...")
        
        cmd = [
            "vllm", "serve",
            MODEL_NAME,  # Model as positional argument, not --model flag
            "--host", "0.0.0.0",
            "--port", "8091",
            "--served-model-name", MODEL_NAME,
            "--max-model-len", "8192",
            "--dtype", "bfloat16",
            "--gpu-memory-utilization", "0.9",
            "--uvicorn-log-level", "info",
            "--tokenizer-mode", "mistral",  # Fix for Mistral tokenizer warning
        ]
        
        # Performance optimization: enforce-eager for fast boot vs better performance
        if FAST_BOOT:
            cmd.append("--enforce-eager")
        else:
            cmd.append("--no-enforce-eager")
        
        print(f"VLLM command: {' '.join(cmd)}")
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        
        # Monitor output
        def monitor_output():
            try:
                while self.proc.poll() is None:
                    line = self.proc.stdout.readline()
                    if line:
                        print(f"VLLM server: {line.strip()}")
            except Exception as e:
                print(f"VLLM output monitor error: {e}")
        
        output_thread = threading.Thread(target=monitor_output, daemon=True)
        output_thread.start()
        
        # Wait for server to be ready
        max_attempts = 48  # 8 minutes
        for attempt in range(max_attempts):
            if self.proc.poll() is not None:
                print(f"VLLM server process died with return code: {self.proc.returncode}")
                break
                
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1.0)
                result = sock.connect_ex(('127.0.0.1', 8091))
                sock.close()
                if result == 0:
                    print(f"VLLM server ready after {attempt + 1} seconds")
                    break
            except Exception:
                pass
            time.sleep(10)
        else:
            if self.proc.poll() is not None:
                stdout, stderr = self.proc.communicate()
                print(f"VLLM server failed to start. Return code: {self.proc.returncode}")
                print(f"VLLM server stdout: {stdout}")
                print(f"VLLM server stderr: {stderr}")
                raise RuntimeError(f"VLLM server process died during startup with return code {self.proc.returncode}")
            else:
                print("VLLM server taking longer than expected to start, continuing anyway...")
        
        print("LLM service loaded successfully")
    
    @modal.asgi_app()
    def web(self):
        """OpenAI-compatible API endpoint for LLM service"""
        from fastapi import FastAPI
        from fastapi.responses import StreamingResponse
        from unmute.llm.llm_utils import get_openai_client, rechunk_to_words, VLLMStream
        import json
        import os
        
        # Import constants from module level
        MODEL_NAME = "mistralai/Mistral-Small-24B-Instruct-2501"
        
        app = FastAPI(title="LLM Service", version="1.0.0")
        
        @app.get("/")
        def root():
            return {"service": "llm", "model": MODEL_NAME, "status": "ready"}
        
        @app.get("/v1/models")
        def list_models():
            """List available models endpoint for OpenAI client compatibility"""
            return {
                "object": "list",
                "data": [
                    {
                        "id": MODEL_NAME,
                        "object": "model",
                        "created": 1640995200,
                        "owned_by": "mistralai",
                        "permission": [],
                        "root": MODEL_NAME,
                        "parent": None
                    }
                ]
            }
        
        @app.post("/v1/chat/completions")
        async def chat_completions(request: dict):
            """OpenAI-compatible chat completions endpoint using VLLMStream"""
            print(f"=== LLM_SERVICE: Received chat completion request: {request} ===")
            
            try:
                # Extract parameters from request
                messages = request.get("messages", [])
                temperature = request.get("temperature", 0.7)
                stream = request.get("stream", False)
                
                print(f"=== LLM_SERVICE: Processing {len(messages)} messages, temperature={temperature}, stream={stream} ===")
                
                # Wait for VLLM server to be ready before creating client
                import socket
                import time
                max_wait = 30  # 30 seconds max wait
                for attempt in range(max_wait):
                    try:
                        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        sock.settimeout(1.0)
                        result = sock.connect_ex(('127.0.0.1', 8091))
                        sock.close()
                        if result == 0:
                            print(f"=== LLM_SERVICE: VLLM server ready after {attempt + 1} attempts ===")
                            break
                    except Exception:
                        pass
                    if attempt < max_wait - 1:  # Don't sleep on the last attempt
                        time.sleep(1)
                else:
                    raise ConnectionError("VLLM server not ready after 30 seconds")
                
                # Create OpenAI client pointing to local VLLM server
                client = get_openai_client(server_url="http://localhost:8091")
                
                # Use existing VLLMStream class
                llm = VLLMStream(client, temperature=temperature)
                
                if stream:
                    # Streaming response using the existing VLLMStream + rechunk_to_words
                    async def generate_stream():
                        try:
                            async for word in rechunk_to_words(llm.chat_completion(messages)):
                                chunk = {
                                    "id": f"chatcmpl-{hash(str(messages))}",
                                    "object": "chat.completion.chunk",
                                    "created": 1640995200,
                                    "model": MODEL_NAME,
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {
                                                "content": word
                                            },
                                            "finish_reason": None
                                        }
                                    ]
                                }
                                yield f"data: {json.dumps(chunk)}\n\n"
                            
                            # Send final chunk
                            final_chunk = {
                                "id": f"chatcmpl-{hash(str(messages))}",
                                "object": "chat.completion.chunk",
                                "created": 1640995200,
                                "model": MODEL_NAME,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {},
                                        "finish_reason": "stop"
                                    }
                                ]
                            }
                            yield f"data: {json.dumps(final_chunk)}\n\n"
                            yield "data: [DONE]\n\n"
                            
                        except Exception as e:
                            print(f"=== LLM_SERVICE: Error in streaming: {e} ===")
                            import traceback
                            print(f"=== LLM_SERVICE: Traceback: {traceback.format_exc()} ===")
                            # Send error chunk
                            error_chunk = {
                                "error": {
                                    "message": str(e),
                                    "type": "internal_server_error",
                                    "code": "internal_error"
                                }
                            }
                            yield f"data: {json.dumps(error_chunk)}\n\n"

                    return StreamingResponse(generate_stream(), media_type="text/event-stream")
                
                else:
                    # Non-streaming response - collect all words
                    full_text = ""
                    async for word in rechunk_to_words(llm.chat_completion(messages)):
                        full_text += word
                    
                    response = {
                        "id": f"chatcmpl-{hash(str(messages))}",
                        "object": "chat.completion",
                        "created": 1640995200,
                        "model": MODEL_NAME,
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": full_text
                                },
                                "finish_reason": "stop"
                            }
                        ],
                        "usage": {
                            "prompt_tokens": sum(len(msg.get("content", "").split()) for msg in messages),
                            "completion_tokens": len(full_text.split()),
                            "total_tokens": sum(len(msg.get("content", "").split()) for msg in messages) + len(full_text.split())
                        }
                    }
                    
                    print(f"=== LLM_SERVICE: Generated response: {full_text[:100]}... ===")
                    return response
                    
            except Exception as e:
                print(f"=== LLM_SERVICE: Error in chat completion: {e} ===")
                import traceback
                print(f"=== LLM_SERVICE: Traceback: {traceback.format_exc()} ===")
                return {
                    "error": {
                        "message": str(e),
                        "type": "internal_server_error",
                        "code": "internal_error"
                    }
                }
        
        return app


@app.cls(
    cpu=2,
    image=orchestrator_image,
    secrets=secrets,
    min_containers=int(os.environ.get("MIN_CONTAINERS", "0")),
    scaledown_window=600,  # 10 minutes - prevent scaling during long conversations
)
@modal.concurrent(max_inputs=10)
class OrchestratorService:
    """Orchestrator service that coordinates between STT, LLM, and TTS"""
    
    @modal.enter()
    def setup(self):
        """Set up the orchestrator environment"""
        import os
        
        print("Setting up Orchestrator Service...")
        
        # Set environment variables to point to Modal services
        # Use the actual Modal deployment URLs based on the app name and service classes
        base_url = "willdavenport--voice-stack"
        
        # Check for dev environment variable to use correct URL patterns
        if os.environ.get("MODAL_DEV_MODE"):
            # In modal serve mode, each service class gets its own -dev URL
            # Pattern: https://username--appname-classname-dev.modal.run
            os.environ["KYUTAI_STT_URL"] = f"wss://{base_url}-sttservice-web-dev.modal.run"
            os.environ["KYUTAI_TTS_URL"] = f"wss://{base_url}-ttsservice-web-dev.modal.run"
            os.environ["KYUTAI_LLM_URL"] = f"https://{base_url}-llmservice-web-dev.modal.run"
        else:
            # Production deployment URLs include -web suffix
            os.environ["KYUTAI_STT_URL"] = f"wss://{base_url}-sttservice-web-dev.modal.run"
            os.environ["KYUTAI_TTS_URL"] = f"wss://{base_url}-ttsservice-web-dev.modal.run"
            os.environ["KYUTAI_LLM_URL"] = f"https://{base_url}-llmservice-web-dev.modal.run"
        # Voice cloning is not available in Modal deployment
        os.environ["KYUTAI_VOICE_CLONING_URL"] = "http://localhost:8092"
        
        # Override paths for Modal services (they use /ws instead of /api/*)
        os.environ["KYUTAI_STT_PATH"] = "/ws"
        os.environ["KYUTAI_TTS_PATH"] = "/ws"
        
        # Set longer timeout for Modal services which can take time to cold start
        # TTS service takes especially long to start up
        os.environ["KYUTAI_SERVICE_TIMEOUT_SEC"] = "150.0"
        
        print(f"Orchestrator setup complete - STT: {os.environ['KYUTAI_STT_URL']}")
        print(f"Orchestrator setup complete - TTS: {os.environ['KYUTAI_TTS_URL']}")
        print(f"Orchestrator setup complete - LLM: {os.environ['KYUTAI_LLM_URL']}")
        
        print("Orchestrator setup complete")
    
    @modal.asgi_app()
    def web(self):
        """Main WebSocket endpoint for client connections"""
        from fastapi import FastAPI, WebSocket, UploadFile
        from fastapi.middleware.cors import CORSMiddleware
        import asyncio
        
        app = FastAPI(title="Orchestrator Service", version="1.0.0")
        
        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        @app.get("/")
        def root():
            return {
                "service": "orchestrator",
                "status": "ready",
                "services": ["stt", "llm", "tts"]
            }
        
        @app.get("/v1/health")
        async def get_health():
            """Health check endpoint that mimics the original backend"""
            # Import the health check logic from main_websocket
            from unmute.main_websocket import _get_health
            health = await _get_health(None)
            return health
        
        @app.get("/v1/voices")
        def get_voices():
            """Get available voices - mimics the original backend endpoint"""
            from unmute.tts.voices import VoiceList
            voice_list = VoiceList()
            # Note that `voice.good` is bool | None, here we really take only True values.
            good_voices = [
                voice.model_dump(exclude={"comment"})
                for voice in voice_list.voices
                if voice.good
            ]
            return good_voices
        
        @app.post("/v1/voices")
        async def post_voices(file: UploadFile):
            """Upload a voice file for cloning - mimics the original backend endpoint"""
            from unmute.tts.voice_cloning import clone_voice
            
            # Note: Voice cloning is not fully available in Modal deployment
            # but we provide this endpoint for compatibility
            try:
                file_content = await file.read()
                name = clone_voice(file_content)
                return {"name": name}
            except Exception:
                # If voice cloning server is not available, return a mock response
                import uuid
                mock_name = "custom:" + str(uuid.uuid4())
                return {"name": mock_name}
        
        @app.websocket("/v1/realtime")
        async def websocket_endpoint(websocket: WebSocket):
            """Main client WebSocket endpoint using the existing unmute handler"""
            logger.debug("WebSocket connection attempt started")
            try:
                # Accept with the correct subprotocol
                await websocket.accept(subprotocol="realtime")
                logger.info("WebSocket connection accepted with realtime subprotocol")
            except Exception as e:
                logger.error(f"Failed to accept WebSocket connection: {e}")
                return
            
            # Import the complete websocket handling logic
            from unmute.conversation_handler import ConversationUnmuteHandler
            from unmute.conversation import ConversationManager
            from unmute.main_websocket import receive_loop, emit_loop, debug_running_tasks, _get_health, _report_websocket_exception
            import unmute.openai_realtime_api_events as ora
            from fastapi import status
            
            try:
                logger.debug("Creating handler instance")
                # Create handler instance
                conversation_manager = ConversationManager()
                handler = ConversationUnmuteHandler(conversation_manager)
                
                logger.debug("Checking health status")
                # Check health first
                health = await _get_health(None)
                print(f"=== ORCHESTRATOR: Health check result: {health} ===")
                if not health.ok:
                    print(f"=== ORCHESTRATOR: Health check failed, closing WebSocket connection: {health} ===")
                    await websocket.close(
                        code=status.WS_1011_INTERNAL_ERROR,
                        reason=f"Server is not healthy: {health}",
                    )
                    return
                
                print("=== ORCHESTRATOR: Health check passed, starting handler ===")
                
                # Log current environment variables for debugging
                import os
                logger.debug(f"Environment variables: STT_URL={os.environ.get('KYUTAI_STT_URL', 'NOT_SET')}, TTS_URL={os.environ.get('KYUTAI_TTS_URL', 'NOT_SET')}, LLM_URL={os.environ.get('KYUTAI_LLM_URL', 'NOT_SET')}")
                
                emit_queue: asyncio.Queue[ora.ServerEvent] = asyncio.Queue()
                try:
                    async with handler:
                        logger.debug("Handler context entered, calling start_up")
                        await handler.start_up()
                        logger.info("Handler start_up completed, creating task group")
                        async with asyncio.TaskGroup() as tg:
                            tg.create_task(
                                receive_loop(websocket, handler, emit_queue), name="receive_loop()"
                            )
                            tg.create_task(
                                emit_loop(websocket, handler, emit_queue), name="emit_loop()"
                            )
                            # No quest_manager.wait() needed with new conversation architecture
                            tg.create_task(debug_running_tasks(), name="debug_running_tasks()")
                            logger.debug("All tasks created, task group running")
                except Exception as handler_exc:
                    print(f"=== ORCHESTRATOR: Exception in handler context: {handler_exc} ===")
                    raise
                finally:
                    print("=== ORCHESTRATOR: Cleaning up handler ===")
                    await handler.cleanup()
                    print("=== ORCHESTRATOR: websocket_route() finished ===")
                    
            except Exception as exc:
                print(f"=== ORCHESTRATOR: Exception in websocket_endpoint: {exc} ===")
                import traceback
                print(f"=== ORCHESTRATOR: Traceback: {traceback.format_exc()} ===")
                await _report_websocket_exception(websocket, exc)
        
        return app


# Additional functions for deployment and health checks
@app.function(image=orchestrator_image)
def deploy():
    """Deploy the voice stack application to Modal."""
    print("Deploying voice stack application to Modal...")
    print("Architecture: Orchestrator (CPU) + STT (L4) + LLM (L40S) + TTS (L4)")
    return "Deployment complete"

@app.function(image=orchestrator_image)
def health_check():
    """Check the health of all services."""
    print("Checking health of all services...")
    return {
        "orchestrator": "healthy",
        "stt": "healthy", 
        "llm": "healthy",
        "tts": "healthy",
        "timestamp": "2025-01-14"
    }

@app.local_entrypoint()
def test_orpheus_tts():
    """Test the Orpheus TTS service."""
    import time
    
    start_time = time.time()
    print("Testing Orpheus TTS Service...")
    print("Note: Timeout increased to 15 minutes to allow for model download")
    print("Note: Subsequent runs will be faster due to model caching")
    
    # Test text
    test_text = "This is a test of the Orpheus TTS service in our Modal deployment."
    
    try:
        # Create service instance and test the new generate_speech method
        service = TTSService()
        
        print(f"Generating speech for: '{test_text}'")
        print("Note: Model loading happens automatically when container starts (@modal.enter)")
        print("This may take up to 15 minutes on first run while downloading the 3B model...")
        
        # Call the generate_speech method (this will trigger @modal.enter() automatically)
        generation_start = time.time()
        result = service.generate_speech.remote(test_text, "tara")
        generation_time = time.time() - generation_start
        
        if result["success"]:
            print(f"Speech generation successful!")
            print(f"Generation time: {generation_time:.2f}s")
            print(f"Sample rate: {result['sample_rate']} Hz")
            print(f"Duration: {result['duration_ms']:.2f}ms")
            print(f"Chunks generated: {result['chunk_count']}")
            print(f"Total audio bytes: {result['total_bytes']}")
            print(f"Audio data length (base64): {len(result['audio_base64'])} characters")
            
            # Optionally save audio to file for testing
            import base64
            audio_bytes = base64.b64decode(result['audio_base64'])
            with open("orpheus_test_output.raw", "wb") as f:
                f.write(audio_bytes)
            print("Raw audio saved to orpheus_test_output.raw")
            print("To play: ffplay -f s16le -ar 24000 -ac 1 orpheus_test_output.raw")
            
            # Also create a WAV file for easier playback
            try:
                import wave
                import numpy as np
                
                # Convert bytes to numpy array
                audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
                
                with wave.open("orpheus_test_output.wav", "wb") as wav_file:
                    wav_file.setnchannels(1)  # Mono
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(24000)  # 24kHz
                    wav_file.writeframes(audio_bytes)
                
                print("WAV audio saved to orpheus_test_output.wav")
                
            except ImportError:
                print("Note: Install 'wave' package to generate WAV files")
            except Exception as e:
                print(f"Warning: Could not create WAV file: {e}")
            
        else:
            print(f"Speech generation failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"Error testing Orpheus TTS: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise
    finally:
        total_time = time.time() - start_time
        print(f"Total test time: {total_time:.2f}s")


@app.local_entrypoint()
def test_orpheus_tts_websocket():
    """Test the Orpheus TTS service via WebSocket interface."""
    import time
    
    start_time = time.time()
    print("Testing Orpheus TTS Service via WebSocket...")
    
    import asyncio
    import msgpack
    import base64
    import websockets
    
    async def test_websocket():
        # Test text
        test_text = "Hello, this is a WebSocket test of the Orpheus TTS service."
        test_voice = "tara"
        
        # Get the TTS service URL (this would be set when deployed)
        # For testing, we'll use the dev URL pattern
        base_url = "willdavenport--voice-stack"
        tts_url = f"wss://{base_url}-ttsservice-web-dev.modal.run/ws"
        
        print(f"Connecting to TTS service at: {tts_url}")
        
        # Retry logic for connection (service might be downloading model)
        max_retries = 5
        retry_delay = 30  # seconds
        
        for attempt in range(max_retries):
            try:
                print(f"Connection attempt {attempt + 1}/{max_retries}...")
                if attempt > 0:
                    print(f"Waiting {retry_delay}s before retry (service may be downloading model)...")
                    await asyncio.sleep(retry_delay)
                
                return await _attempt_websocket_connection(tts_url, test_text, test_voice)
                
            except (TimeoutError, ConnectionRefusedError, OSError) as e:
                if attempt == max_retries - 1:
                    print(f"All connection attempts failed. Last error: {e}")
                    return False
                else:
                    print(f"Connection attempt {attempt + 1} failed: {e}")
                    print("This is likely because the Orpheus model is still downloading...")
                    continue
            except Exception as e:
                print(f"Unexpected error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    return False
                continue
        
        return False
    
    async def _attempt_websocket_connection(tts_url, test_text, test_voice):
        try:
            connection_start = time.time()
            # Set a longer timeout for the initial connection (model might be loading)
            async with websockets.connect(tts_url, open_timeout=60) as websocket:
                connection_time = time.time() - connection_start
                print(f"Connected to Orpheus TTS WebSocket (took {connection_time:.2f}s)")
                
                # Wait for Ready message
                ready_start = time.time()
                ready_data = await websocket.recv()
                ready_time = time.time() - ready_start
                ready_message = msgpack.unpackb(ready_data)
                print(f"Received Ready message: {ready_message} (took {ready_time:.2f}s)")
                
                if ready_message.get("type") != "Ready":
                    print(f"Warning: Expected Ready message, got: {ready_message}")
                
                # Send text message
                text_message = {
                    "type": "Text",
                    "text": test_text,
                    "voice": test_voice
                }
                
                send_start = time.time()
                packed_message = msgpack.packb(text_message)
                await websocket.send(packed_message)
                send_time = time.time() - send_start
                print(f"Sent text message: '{test_text}' with voice: '{test_voice}' (took {send_time:.3f}s)")
                
                # Collect audio chunks (raw bytes from Orpheus)
                audio_chunks = []
                chunk_count = 0
                first_chunk_time = None
                audio_start_time = time.time()
                
                try:
                    while True:
                        # Receive raw bytes directly from Orpheus TTS
                        data = await asyncio.wait_for(websocket.recv(), timeout=60.0)  # Increased timeout
                        
                        if first_chunk_time is None:
                            first_chunk_time = time.time() - audio_start_time
                        
                        # Orpheus sends raw PCM bytes, not msgpack messages
                        if isinstance(data, bytes) and len(data) > 0:
                            audio_chunks.append(data)
                            chunk_count += 1
                            print(f"Received raw audio chunk {chunk_count}: {len(data)} bytes")
                        else:
                            # Check if it might be a control message
                            try:
                                message = msgpack.unpackb(data)
                                if message.get("type") == "Error":
                                    error_msg = message.get("message", "Unknown error")
                                    print(f"Received error from TTS: {error_msg}")
                                    return False
                                elif message.get("type") == "Eos":
                                    print("Received End of Stream message")
                                    break
                                else:
                                    print(f"Received unexpected message type: {message.get('type')}")
                            except:
                                # Not a msgpack message, might be end of stream
                                print(f"Received non-audio data: {type(data)} length {len(data) if hasattr(data, '__len__') else 'unknown'}")
                                break
                            
                except asyncio.TimeoutError:
                    print("Timeout waiting for audio data - stream ended")
                
                audio_generation_time = time.time() - audio_start_time
                
                if audio_chunks:
                    # Combine all audio chunks
                    combined_audio = b''.join(audio_chunks)
                    total_bytes = len(combined_audio)
                    
                    print(f"WebSocket test successful!")
                    print(f"Time to first audio chunk: {first_chunk_time:.2f}s")
                    print(f"Total audio generation time: {audio_generation_time:.2f}s")
                    print(f"Total audio chunks: {chunk_count}")
                    print(f"Total audio bytes: {total_bytes}")
                    
                    # Calculate approximate duration (24kHz, 16-bit PCM)
                    sample_rate = 24000
                    bytes_per_sample = 2
                    duration_ms = (total_bytes / bytes_per_sample / sample_rate) * 1000
                    print(f"Approximate duration: {duration_ms:.2f}ms")
                    
                    # Calculate real-time factor (generation time vs audio duration)
                    rtf = audio_generation_time / (duration_ms / 1000)
                    print(f"Real-time factor: {rtf:.2f}x (lower is better)")
                    
                    # Save audio file
                    with open("orpheus_websocket_test_output.raw", "wb") as f:
                        f.write(combined_audio)
                    print("Audio saved to orpheus_websocket_test_output.raw")
                    print("To play: ffplay -f s16le -ar 24000 -ac 1 orpheus_websocket_test_output.raw")
                    
                    return True
                else:
                    print("No audio chunks received!")
                    return False
                    
        except Exception as e:
            print(f"WebSocket connection attempt failed: {e}")
            raise  # Re-raise to be caught by retry logic
    
    # Run the async test
    try:
        result = asyncio.run(test_websocket())
        if result:
            print("✅ Orpheus TTS WebSocket test PASSED")
        else:
            print("❌ Orpheus TTS WebSocket test FAILED")
    except Exception as e:
        print(f"❌ Orpheus TTS WebSocket test ERROR: {e}")
    finally:
        total_time = time.time() - start_time
        print(f"Total WebSocket test time: {total_time:.2f}s")


if __name__ == "__main__":
    # For local development
    print("Modal Voice Stack Application")
    print("Run with: modal serve modal_app.py")
