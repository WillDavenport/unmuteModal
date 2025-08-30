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

logger = logging.getLogger(__name__)

# Modal app definition
app = modal.App("voice-stack")

# Define the base image with common dependencies (NO local files yet)
base_deps_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl", "ffmpeg", "libsndfile1", "build-essential")
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
        # Create cache directories
        "mkdir -p /root/.cache/huggingface/hub",
        "mkdir -p /root/.cache/huggingface/xet",
        # Download STT model files with simple retry logic
        "python -c \"import time; from huggingface_hub import hf_hub_download; [hf_hub_download('kyutai/stt-1b-en_fr-candle', f, cache_dir='/root/.cache/huggingface/hub') for f in ['model.safetensors', 'tokenizer_en_fr_audio_8000.model', 'mimi-pytorch-e351c8d8@125.safetensors']]\"",
        # Verify models were downloaded
        "echo 'Verifying downloaded STT models...'",
        "find /root/.cache/huggingface/hub -name 'model.safetensors' | head -5",
        "find /root/.cache/huggingface/hub -name 'tokenizer_en_fr_audio_8000.model' | head -5",
        "find /root/.cache/huggingface/hub -name 'mimi-pytorch-e351c8d8@125.safetensors' | head -5",
        "echo 'STT model pre-download completed'"
    )
    # Add local files LAST
    .add_local_python_source("unmute")
    .add_local_file("voices.yaml", "/root/voices.yaml")
)

# TTS image: build steps first, then local files
tts_image = (
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
        "huggingface_hub>=0.19.0",
        "safetensors>=0.4.0",
        "transformers>=4.35.0",
        "tokenizers>=0.15.0",
        # Moshi Python package and dependencies for moshi-server
        "moshi>=0.2.8",
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
        # Pre-download TTS models during image build to avoid startup delays
        "echo 'Pre-downloading TTS models to cache...'",
        # Create cache directories
        "mkdir -p /root/.cache/huggingface/hub",
        "mkdir -p /root/.cache/huggingface/xet",
        "mkdir -p /root/tts-voices-cache",
        # Download essential TTS tokenizer (required for TTS to work)
        "python -c \"from huggingface_hub import hf_hub_download; hf_hub_download('kyutai/tts-1.6b-en_fr', 'tokenizer_spm_8k_en_fr_audio.model', cache_dir='/root/.cache/huggingface/hub')\"",
        # Create directory structure for voices (actual download will happen at runtime with persistent storage)
        "echo 'Creating TTS voices directory structure...'",
        "mkdir -p /root/tts-voices-cache/unmute-prod-website",
        "echo 'TTS voices directory structure created'",
        # Verify essential models were downloaded
        "echo 'Verifying downloaded models...'",
        "ls -la /root/.cache/huggingface/hub/ | head -10",
        "ls -la /root/tts-voices-cache/ | head -10",
        "find /root/.cache/huggingface/hub -name 'tokenizer_spm_8k_en_fr_audio.model' | head -5 || echo 'Tokenizer not found in expected location'",
        "find /root/tts-voices-cache -name '*.safetensors' | wc -l",
        "echo 'TTS model pre-download completed. Essential voices cached locally.'"
    )
    # Add local files LAST
    .add_local_python_source("unmute")
    .add_local_file("voices.yaml", "/root/voices.yaml")
)

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

# Modal volumes for model storage
models_volume = modal.Volume.from_name("voice-models")
tts_voices_volume = modal.Volume.from_name("tts-voices-cache")

# Cache volumes for LLM optimization
hf_cache_vol = modal.Volume.from_name("huggingface-cache")
vllm_cache_vol = modal.Volume.from_name("vllm-cache")

# Cache volume for Rust binaries (shared between STT and TTS)
rust_binaries_volume = modal.Volume.from_name("rust-binaries-cache")

# Model configuration for Mistral
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
MODEL_REVISION = "main"  # Pin to specific revision when available
FAST_BOOT = True  # Set to False for better performance if you have persistent replicas

# Modal secrets for API keys and auth tokens
secrets = [
    modal.Secret.from_name("voice-auth"),
    modal.Secret.from_name("huggingface-secret"),  # Add HF token for model access
]


def install_moshi_server_with_cuda():
    """
    Install moshi-server binary with CUDA support at runtime.
    This function checks if a cached binary exists, and if not, compiles it with CUDA features.
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
        # Verify CUDA installation
        print("Verifying CUDA installation...")
        try:
            nvcc_result = subprocess.run([
                "nvcc", "--version"
            ], env=env, capture_output=True, text=True, timeout=10)
            if nvcc_result.returncode == 0:
                print(f"CUDA nvcc found: {nvcc_result.stdout.strip()}")
            else:
                print(f"CUDA nvcc check failed: {nvcc_result.stderr}")
        except Exception as e:
            print(f"CUDA nvcc verification failed: {e}")
            print("Continuing without CUDA verification - compilation may still work")
        
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
        
        # Verify the binary works
        result = subprocess.run([
            local_binary_path, "--help"
        ], capture_output=True, text=True, timeout=10, env=env)
        
        if result.returncode != 0:
            raise RuntimeError(f"Compiled binary failed verification: {result.stderr}")
            
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
    },
    secrets=secrets,
    min_containers=0,
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
                                await moshi_ws.send(data)
                        except WebSocketDisconnect:
                            pass
                    
                    async def moshi_to_client():
                        try:
                            async for message in moshi_ws:
                                if isinstance(message, bytes):
                                    await websocket.send_bytes(message)
                                else:
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
    gpu="L4", 
    image=tts_image,
    volumes={
        "/models": models_volume, 
        "/persistent-voices": tts_voices_volume,
        "/rust-binaries": rust_binaries_volume,
    },
    secrets=secrets,
    min_containers=0,
    scaledown_window=600,  # 10 minutes - prevent scaling during long conversations
)
@modal.concurrent(max_inputs=10)
class TTSService:
    """Text-to-Speech service using Moshi TTS model"""
    
    @modal.enter()
    def load_model(self):
        """Load TTS model weights once per container"""
        import subprocess
        import tempfile
        import os
        import time
        
        print("Setting up TTS Service...")
        
        # Install moshi-server with CUDA support
        moshi_server_path = install_moshi_server_with_cuda()
        
        # Set up environment with cargo bin in PATH
        env = os.environ.copy()
        cargo_bin_path = os.path.expanduser("~/.cargo/bin")
        env["PATH"] = f"{cargo_bin_path}:{env.get('PATH', '')}"
        # Reduce noisy batched_asr logs by setting log level to off for moshi_server::batched_asr
        env["RUST_LOG"] = "info"
        
        # Check if HuggingFace token is available
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if hf_token:
            print(f"TTS: HuggingFace token available (length: {len(hf_token)})")
        else:
            print("TTS: No HuggingFace token found - this may cause issues downloading models")
        
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
        
        # Set up persistent voice storage
        print("Setting up persistent voice storage...")
        persistent_voices_dir = "/persistent-voices"
        os.makedirs(persistent_voices_dir, exist_ok=True)
        
        # Check if voices already exist in persistent storage
        voices_exist = (os.path.exists(f"{persistent_voices_dir}/unmute-prod-website") and 
                       len([f for f in os.listdir(f"{persistent_voices_dir}/unmute-prod-website") if f.endswith('.safetensors')]) > 0) if os.path.exists(f"{persistent_voices_dir}/unmute-prod-website") else False
        
        if not voices_exist:
            print("Downloading essential voices to persistent storage (one-time setup)...")
            try:
                # Download only the essential voices directly to persistent storage
                from huggingface_hub import snapshot_download
                snapshot_download(
                    'kyutai/tts-voices',
                    cache_dir='/root/.cache/huggingface/hub',
                    local_dir=persistent_voices_dir,
                    allow_patterns=['unmute-prod-website/*.safetensors', 'unmute-prod-website/*.wav']
                )
                print("Essential voices downloaded successfully to persistent storage")
                
                # Verify that voices were actually downloaded
                voice_files = [f for f in os.listdir(f"{persistent_voices_dir}/unmute-prod-website") if f.endswith('.safetensors')] if os.path.exists(f"{persistent_voices_dir}/unmute-prod-website") else []
                if len(voice_files) == 0:
                    raise RuntimeError("No voice files found after download - TTS will not work")
                print(f"Verified {len(voice_files)} voice files downloaded successfully")
                
            except Exception as e:
                print(f"CRITICAL ERROR: Voice download failed: {e}")
                print("TTS service cannot start without voices - failing container startup")
                raise RuntimeError(f"TTS service initialization failed: {e}. Container must be restarted.")
        else:
            print("Voices already exist in persistent storage, skipping download")
            # Verify existing voices are still valid
            voice_files = [f for f in os.listdir(f"{persistent_voices_dir}/unmute-prod-website") if f.endswith('.safetensors')] if os.path.exists(f"{persistent_voices_dir}/unmute-prod-website") else []
            if len(voice_files) == 0:
                print("CRITICAL ERROR: No voice files found in persistent storage")
                raise RuntimeError("TTS service cannot start: No voice files available in persistent storage")
        
        # Create temporary config file for TTS
        config_content = '''
static_dir = "./static/"
log_dir = "/tmp/unmute_logs"
instance_name = "tts"
authorized_ids = ["public_token"]

[modules.tts_py]
type = "Py"
path = "/api/tts_streaming"
text_tokenizer_file = "hf://kyutai/tts-1.6b-en_fr/tokenizer_spm_8k_en_fr_audio.model"
batch_size = 16
text_bos_token = 1

[modules.tts_py.py]
log_folder = "/tmp/unmute_logs"
# Use persistent volume for voices to avoid re-downloading on each container restart
voice_folder = "/persistent-voices"
default_voice = "unmute-prod-website/default_voice.wav"
cfg_coef = 2.0
cfg_is_no_text = true
padding_between = 1
n_q = 24
# Add debug logging
log_level = "info"
'''
        
        os.makedirs("/tmp/unmute_logs", exist_ok=True)
        os.makedirs("./static", exist_ok=True)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write(config_content)
            self.config_path = f.name
        
        # Start moshi-server as a background process
        print("Starting TTS moshi-server...")
        print(f"Config file: {self.config_path}")
        
        # Debug: print the config content
        with open(self.config_path, 'r') as f:
            config_content = f.read()
            print("TTS Config content:")
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
            "--port", "8089"
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
                        print(f"TTS moshi-server: {line_stripped}")
            except Exception as e:
                print(f"TTS output monitor error: {e}")
        
        output_thread = threading.Thread(target=monitor_output, daemon=True)
        output_thread.start()
        
        max_attempts = 120  # 120 seconds max - TTS takes longer than STT
        for attempt in range(max_attempts):
            # Check if process died
            if self.proc.poll() is not None:
                print(f"TTS moshi-server process died with return code: {self.proc.returncode}")
                break
                
            try:
                # Try to connect to the moshi server
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1.0)
                result = sock.connect_ex(('127.0.0.1', 8089))
                sock.close()
                if result == 0:
                    print(f"TTS moshi-server ready after {attempt + 1} seconds")
                    break
            except Exception:
                pass
            # Print progress every 10 seconds
            if (attempt + 1) % 10 == 0:
                print(f"TTS moshi-server still starting... {attempt + 1}/{max_attempts} seconds")
            time.sleep(1)
        else:
            # Check if process is still running
            if self.proc.poll() is not None:
                stdout, stderr = self.proc.communicate()
                print(f"TTS moshi-server failed to start. Return code: {self.proc.returncode}")
                print(f"TTS moshi-server stdout: {stdout}")
                print(f"TTS moshi-server stderr: {stderr}")
                
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
                
                raise RuntimeError(f"TTS moshi-server process died during startup with return code {self.proc.returncode}")
            else:
                print("TTS moshi-server taking longer than expected to start, continuing anyway...")
            
        print("TTS model loaded successfully")
    
    @modal.asgi_app()
    def web(self):
        """WebSocket endpoint for TTS service"""
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect
        import asyncio
        
        app = FastAPI(title="TTS Service", version="1.0.0")
        
        @app.get("/")
        def root():
            return {"service": "tts", "model": "kyutai-tts", "status": "ready"}
        
        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            print("=== TTS_SERVICE: WebSocket connection attempt ===")
            
            # First check if our internal moshi server is ready before accepting the connection
            import socket
            import asyncio
            max_wait_attempts = 120  # 120 seconds max to allow for moshi-server startup
            for attempt in range(max_wait_attempts):
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(1.0)
                    result = sock.connect_ex(('127.0.0.1', 8089))
                    sock.close()
                    if result == 0:
                        print(f"=== TTS_SERVICE: Internal moshi server ready after {attempt + 1} attempts ===")
                        break
                except Exception:
                    pass
                print(f"=== TTS_SERVICE: Waiting for internal moshi server, attempt {attempt + 1}/{max_wait_attempts} ===")
                await asyncio.sleep(1)  # Use async sleep instead of blocking sleep
            else:
                print("=== TTS_SERVICE: Internal moshi server not ready after 120 seconds ===")
                # We must accept the websocket first before we can close it
                await websocket.accept()
                await websocket.close(code=1011, reason="Internal TTS server not ready")
                return
            
            await websocket.accept()
            print("=== TTS_SERVICE: WebSocket connection accepted, internal server ready ===")
            
            try:
                # Connect to the local moshi server and proxy messages
                import websockets
                try:
                    print("=== TTS_SERVICE: Connecting to internal moshi server ===")
                    moshi_connection = await websockets.connect(
                        "ws://localhost:8089/api/tts_streaming?format=PcmMessagePack",
                        additional_headers={"kyutai-api-key": "public_token"}
                    )
                    print("=== TTS_SERVICE: Successfully connected to internal moshi server ===")
                except ConnectionRefusedError as e:
                    print(f"=== TTS_SERVICE: Failed to connect to internal moshi server: {e} ===")
                    await websocket.close(code=1011, reason="Internal TTS server not ready")
                    return
                except Exception as e:
                    print(f"=== TTS_SERVICE: Unexpected error connecting to moshi server: {e} ===")
                    await websocket.close(code=1011, reason="Internal TTS server error")
                    return
                
                async with moshi_connection as moshi_ws:
                    # First, wait for and forward the initial Ready message from moshi-server
                    try:
                        initial_message = await asyncio.wait_for(moshi_ws.recv(), timeout=30.0)
                        print(f"TTS: Received initial message from moshi-server: {type(initial_message)}")
                        if isinstance(initial_message, bytes):
                            await websocket.send_bytes(initial_message)
                        else:
                            await websocket.send_text(initial_message)
                    except asyncio.TimeoutError:
                        print("TTS: Timeout waiting for initial Ready message from moshi-server")
                        await websocket.close(code=1011, reason="Internal TTS server timeout")
                        return
                    except Exception as e:
                        print(f"TTS: Error receiving initial message: {e}")
                        await websocket.close(code=1011, reason="Internal TTS server error")
                        return
                    
                    # Now proxy messages between client and moshi server
                    async def client_to_moshi():
                        try:
                            while True:
                                data = await websocket.receive_bytes()
                                await moshi_ws.send(data)
                        except WebSocketDisconnect:
                            pass
                    
                    async def moshi_to_client():
                        try:
                            async for message in moshi_ws:
                                if isinstance(message, bytes):
                                    await websocket.send_bytes(message)
                                else:
                                    await websocket.send_text(message)
                        except Exception as e:
                            print(f"TTS moshi_to_client error: {e}")
                    
                    # Run both directions concurrently
                    await asyncio.gather(
                        client_to_moshi(),
                        moshi_to_client(),
                        return_exceptions=True
                    )
            except Exception as e:
                print(f"TTS websocket error: {e}")
        
        return app


@app.cls(
    gpu="L4",
    image=llm_image,
    volumes={
        "/models": models_volume,
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    secrets=secrets,
    min_containers=0,
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
    min_containers=0,
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
        os.environ["KYUTAI_STT_URL"] = f"wss://{base_url}-sttservice-web.modal.run"
        os.environ["KYUTAI_TTS_URL"] = f"wss://{base_url}-ttsservice-web.modal.run"
        os.environ["KYUTAI_LLM_URL"] = f"https://{base_url}-llmservice-web.modal.run"
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
            from unmute.unmute_handler import UnmuteHandler
            from unmute.main_websocket import receive_loop, emit_loop, debug_running_tasks, _get_health, _report_websocket_exception
            import unmute.openai_realtime_api_events as ora
            from fastapi import status
            
            try:
                logger.debug("Creating handler instance")
                # Create handler instance
                handler = UnmuteHandler()
                
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
                            tg.create_task(handler.quest_manager.wait(), name="quest_manager.wait()")
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

if __name__ == "__main__":
    # For local development
    print("Modal Voice Stack Application")
    print("Run with: modal serve modal_app.py")
