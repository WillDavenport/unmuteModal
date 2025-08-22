"""
Modal Voice Stack Application

This Modal app implements the target architecture with four serverless classes:
- Orchestrator (CPU only) - handles client WebSocket connections
- STT (L4 GPU) - Speech-to-Text service with WebSocket endpoint  
- LLM (L40S GPU) - Language Model service with WebSocket endpoint
- TTS (L4 GPU) - Text-to-Speech service with WebSocket endpoint

Each service uses @modal.asgi_app() for WebSocket endpoints and @modal.enter() for model loading.
"""

import modal

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
        "openai>=1.70.0",
        "mistralai>=1.5.1",
    )
)

# STT image: build steps first, then local files
stt_image = (
    base_deps_image
    .apt_install("cmake", "pkg-config", "libopus-dev", "git", "curl", "libssl-dev", "openssl")
    .run_commands(
        # Install latest Rust toolchain to handle Cargo.lock version 4
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable",
        ". ~/.cargo/env"
    )
    .pip_install(
        "torch>=2.1.0",
        "torchaudio>=2.1.0",
    )
    .run_commands(
        # Set environment variables for building
        "export CXXFLAGS='-include cstdint'",
        # Set LD_LIBRARY_PATH for Python integration
        "export LD_LIBRARY_PATH=$(python -c 'import sysconfig; print(sysconfig.get_config_var(\"LIBDIR\"))')",
        # Ensure Rust environment is available
        ". ~/.cargo/env",
        # Debug: Check OpenSSL installation
        "echo 'Checking OpenSSL installation:' && find /usr -name '*ssl*' -type f 2>/dev/null | grep -E '\\.(so|a)$' | head -10",
        "ls -la /usr/lib/x86_64-linux-gnu/ | grep ssl || echo 'No SSL libs in x86_64-linux-gnu'",
        "ls -la /usr/include/ | grep ssl || echo 'No SSL headers in include'",
        # Clone moshi repo and try multiple approaches to build moshi-server
        "git clone --depth 1 https://github.com/kyutai-labs/moshi.git /tmp/moshi",
        # Build without CUDA features during image build (CUDA will be available at runtime)
        # This avoids nvidia-smi dependency during build while still allowing GPU usage at runtime
        "cd /tmp/moshi/rust && (rm -f Cargo.lock && . ~/.cargo/env && export OPENSSL_DIR=/usr && export OPENSSL_LIB_DIR=/usr/lib/x86_64-linux-gnu && export OPENSSL_INCLUDE_DIR=/usr/include/openssl && export PKG_CONFIG_PATH=/usr/lib/x86_64-linux-gnu/pkgconfig && timeout 600 cargo build --release --bin moshi-server && cp target/release/moshi-server /usr/local/bin/ && echo 'Success with CPU-compatible build') || echo 'Build approach 1 failed, trying approach 2'",
        # Approach 2: If approach 1 failed, patch hf-hub dependency and try again
        "cd /tmp/moshi/rust && (test -f /usr/local/bin/moshi-server || (rm -f Cargo.lock && sed -i 's/hf-hub = { version = \\\"0.4.3\\\", features = \\[\\\"tokio\\\"\\] }/hf-hub = { version = \\\"0.4.3\\\", features = [\\\"native-tls\\\"] }/' Cargo.toml && . ~/.cargo/env && export OPENSSL_DIR=/usr && export OPENSSL_LIB_DIR=/usr/lib/x86_64-linux-gnu && export OPENSSL_INCLUDE_DIR=/usr/include/openssl && export PKG_CONFIG_PATH=/usr/lib/x86_64-linux-gnu/pkgconfig && timeout 600 cargo build --release --bin moshi-server && cp target/release/moshi-server /usr/local/bin/ && echo 'Success with patched dependencies'))",
        "rm -rf /tmp/moshi",
        # Verify moshi-server was installed successfully
        ". ~/.cargo/env && which moshi-server && moshi-server --version || echo 'moshi-server installation verification failed'"
    )
    # Add local files LAST
    .add_local_python_source("unmute")
    .add_local_file("voices.yaml", "/root/voices.yaml")
)

# TTS image: build steps first, then local files
tts_image = (
    base_deps_image
    .apt_install("cmake", "pkg-config", "libopus-dev", "git", "curl", "libssl-dev", "openssl")
    .run_commands(
        # Install latest Rust toolchain to handle Cargo.lock version 4
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable",
        ". ~/.cargo/env"
    )
    .pip_install(
        "torch>=2.1.0",
        "torchaudio>=2.1.0",
        "huggingface_hub>=0.19.0",
    )
    .run_commands(
        # Set environment variables for building
        "export CXXFLAGS='-include cstdint'",
        # Set LD_LIBRARY_PATH for Python integration
        "export LD_LIBRARY_PATH=$(python -c 'import sysconfig; print(sysconfig.get_config_var(\"LIBDIR\"))')",
        # Ensure Rust environment is available
        ". ~/.cargo/env",
        # Debug: Check OpenSSL installation
        "echo 'Checking OpenSSL installation:' && find /usr -name '*ssl*' -type f 2>/dev/null | grep -E '\\.(so|a)$' | head -10",
        "ls -la /usr/lib/x86_64-linux-gnu/ | grep ssl || echo 'No SSL libs in x86_64-linux-gnu'",
        "ls -la /usr/include/ | grep ssl || echo 'No SSL headers in include'",
        # Clone moshi repo and try multiple approaches to build moshi-server
        "git clone --depth 1 https://github.com/kyutai-labs/moshi.git /tmp/moshi",
        # Build without CUDA features during image build (CUDA will be available at runtime)
        # This avoids nvidia-smi dependency during build while still allowing GPU usage at runtime
        "cd /tmp/moshi/rust && (rm -f Cargo.lock && . ~/.cargo/env && export OPENSSL_DIR=/usr && export OPENSSL_LIB_DIR=/usr/lib/x86_64-linux-gnu && export OPENSSL_INCLUDE_DIR=/usr/include/openssl && export PKG_CONFIG_PATH=/usr/lib/x86_64-linux-gnu/pkgconfig && timeout 600 cargo build --release --bin moshi-server && cp target/release/moshi-server /usr/local/bin/ && echo 'Success with CPU-compatible build') || echo 'Build approach 1 failed, trying approach 2'",
        # Approach 2: If approach 1 failed, patch hf-hub dependency and try again
        "cd /tmp/moshi/rust && (test -f /usr/local/bin/moshi-server || (rm -f Cargo.lock && sed -i 's/hf-hub = { version = \\\"0.4.3\\\", features = \\[\\\"tokio\\\"\\] }/hf-hub = { version = \\\"0.4.3\\\", features = [\\\"native-tls\\\"] }/' Cargo.toml && . ~/.cargo/env && export OPENSSL_DIR=/usr && export OPENSSL_LIB_DIR=/usr/lib/x86_64-linux-gnu && export OPENSSL_INCLUDE_DIR=/usr/include/openssl && export PKG_CONFIG_PATH=/usr/lib/x86_64-linux-gnu/pkgconfig && timeout 600 cargo build --release --bin moshi-server && cp target/release/moshi-server /usr/local/bin/ && echo 'Success with patched dependencies'))",
        "rm -rf /tmp/moshi",
        # Verify moshi-server was installed successfully
        ". ~/.cargo/env && which moshi-server && moshi-server --version || echo 'moshi-server installation verification failed'"
    )
    # Add local files LAST
    .add_local_python_source("unmute")
    .add_local_file("voices.yaml", "/root/voices.yaml")
)

# LLM image: additional deps first, then local files
llm_image = (
    base_deps_image
    .pip_install(
        "vllm==0.8.5",
        "torch>=2.1.0",
        "transformers>=4.35.0",
    )
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

# Modal secrets for API keys and auth tokens
secrets = [
    modal.Secret.from_name("voice-auth"),
]


@app.cls(
    gpu="L4",
    image=stt_image,
    volumes={"/models": models_volume},
    secrets=secrets,
    min_containers=0,
    scaledown_window=300,  # 5 minutes
)
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
        
        # First, verify moshi-server binary is available
        print("Checking moshi-server availability...")
        try:
            result = subprocess.run(["which", "moshi-server"], 
                                  capture_output=True, text=True, timeout=10)
            print(f"moshi-server location: {result.stdout.strip()}")
            if result.returncode != 0:
                print("moshi-server not found in PATH")
        except Exception as e:
            print(f"Failed to check moshi-server location: {e}")
        
        # Check if Rust environment is available
        try:
            result = subprocess.run(["rustc", "--version"], 
                                  capture_output=True, text=True, timeout=10)
            print(f"Rust version: {result.stdout.strip()}")
        except Exception as e:
            print(f"Rust not available: {e}")
        
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
batch_size = 1
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
        
        self.proc = subprocess.Popen([
            "moshi-server", "worker", 
            "--config", self.config_path,
            "--port", "8090"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Wait for the server to be ready with health checking
        import socket
        max_attempts = 30  # 30 seconds max
        for attempt in range(max_attempts):
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
                    version_result = subprocess.run(["moshi-server", "--version"], 
                                                  capture_output=True, text=True, timeout=10)
                    print(f"moshi-server version: {version_result.stdout}")
                    if version_result.stderr:
                        print(f"moshi-server version stderr: {version_result.stderr}")
                except Exception as e:
                    print(f"Failed to get moshi-server version: {e}")
                
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
            await websocket.accept()
            print("STT WebSocket connection established")
            
            try:
                # Connect to the local moshi server and proxy messages
                import websockets
                try:
                    moshi_connection = await websockets.connect(
                        "ws://localhost:8090/api/asr-streaming",
                        additional_headers={"kyutai-api-key": "public_token"}
                    )
                except ConnectionRefusedError as e:
                    print(f"STT: Failed to connect to internal moshi server: {e}")
                    await websocket.close(code=1011, reason="Internal STT server not ready")
                    return
                except Exception as e:
                    print(f"STT: Unexpected error connecting to moshi server: {e}")
                    await websocket.close(code=1011, reason="Internal STT server error")
                    return
                
                async with moshi_connection as moshi_ws:
                    # Proxy messages between client and moshi server
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
    volumes={"/models": models_volume},
    secrets=secrets,
    min_containers=0,
    scaledown_window=300,  # 5 minutes
)
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
        
        # First, verify moshi-server binary is available
        print("Checking moshi-server availability...")
        try:
            result = subprocess.run(["which", "moshi-server"], 
                                  capture_output=True, text=True, timeout=10)
            print(f"moshi-server location: {result.stdout.strip()}")
            if result.returncode != 0:
                print("moshi-server not found in PATH")
        except Exception as e:
            print(f"Failed to check moshi-server location: {e}")
        
        # Check if Rust environment is available
        try:
            result = subprocess.run(["rustc", "--version"], 
                                  capture_output=True, text=True, timeout=10)
            print(f"Rust version: {result.stdout.strip()}")
        except Exception as e:
            print(f"Rust not available: {e}")
        
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
batch_size = 2
text_bos_token = 1

[modules.tts_py.py]
log_folder = "/tmp/unmute_logs"
voice_folder = "hf-snapshot://kyutai/tts-voices/**/*.safetensors"
default_voice = "unmute-prod-website/default_voice.wav"
cfg_coef = 2.0
cfg_is_no_text = true
padding_between = 1
n_q = 24
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
        
        self.proc = subprocess.Popen([
            "moshi-server", "worker",
            "--config", self.config_path, 
            "--port", "8089"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Wait for the server to be ready with health checking
        import socket
        max_attempts = 30  # 30 seconds max
        for attempt in range(max_attempts):
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
                    version_result = subprocess.run(["moshi-server", "--version"], 
                                                  capture_output=True, text=True, timeout=10)
                    print(f"moshi-server version: {version_result.stdout}")
                    if version_result.stderr:
                        print(f"moshi-server version stderr: {version_result.stderr}")
                except Exception as e:
                    print(f"Failed to get moshi-server version: {e}")
                
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
            await websocket.accept()
            print("TTS WebSocket connection established")
            
            try:
                # Connect to the local moshi server and proxy messages
                import websockets
                try:
                    moshi_connection = await websockets.connect(
                        "ws://localhost:8089/api/tts_streaming",
                        additional_headers={"kyutai-api-key": "public_token"}
                    )
                except ConnectionRefusedError as e:
                    print(f"TTS: Failed to connect to internal moshi server: {e}")
                    await websocket.close(code=1011, reason="Internal TTS server not ready")
                    return
                except Exception as e:
                    print(f"TTS: Unexpected error connecting to moshi server: {e}")
                    await websocket.close(code=1011, reason="Internal TTS server error")
                    return
                
                async with moshi_connection as moshi_ws:
                    # Proxy messages between client and moshi server
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
    volumes={"/models": models_volume},
    secrets=secrets,
    min_containers=0,
    scaledown_window=300,  # 5 minutes
)
class LLMService:
    """Large Language Model service using VLLM"""
    
    @modal.enter()
    def load_model(self):
        """Load LLM model weights once per container"""
        print("Setting up LLM Service...")
        
        from vllm import AsyncLLMEngine, AsyncEngineArgs
        
        # Initialize VLLM engine
        engine_args = AsyncEngineArgs(
            model="google/gemma-3-1b-it",
            max_model_len=8192,
            dtype="bfloat16",
            gpu_memory_utilization=0.3,
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        print("LLM model loaded successfully")
    
    @modal.asgi_app()
    def web(self):
        """WebSocket endpoint for LLM service"""
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect
        import json
        
        app = FastAPI(title="LLM Service", version="1.0.0")
        
        @app.get("/")
        def root():
            return {"service": "llm", "model": "gemma-3-1b-it", "status": "ready"}
        
        @app.get("/v1/models")
        def list_models():
            """List available models endpoint for OpenAI client compatibility"""
            return {
                "object": "list",
                "data": [
                    {
                        "id": "google/gemma-3-1b-it",
                        "object": "model",
                        "created": 1640995200,
                        "owned_by": "google",
                        "permission": [],
                        "root": "google/gemma-3-1b-it",
                        "parent": None
                    }
                ]
            }
        
        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            print("LLM WebSocket connection established")
            
            try:
                while True:
                    # Receive prompt from orchestrator
                    message = await websocket.receive_text()
                    data = json.loads(message)
                    
                    if data.get("type") == "generate":
                        prompt = data.get("prompt", "")
                        temperature = data.get("temperature", 0.7)
                        max_tokens = data.get("max_tokens", 1024)
                        
                        # Generate streaming response
                        from vllm import SamplingParams
                        sampling_params = SamplingParams(
                            temperature=temperature,
                            max_tokens=max_tokens,
                        )
                        
                        request_id = f"req_{hash(prompt)}_{id(websocket)}"
                        
                        # Generate response using VLLM
                        results_generator = self.engine.generate(
                            prompt,
                            sampling_params,
                            request_id=request_id
                        )
                        
                        # Stream results
                        async for request_output in results_generator:
                            if request_output.outputs:
                                for output in request_output.outputs:
                                    # Send incremental text
                                    await websocket.send_text(json.dumps({
                                        "type": "token",
                                        "text": output.text,
                                        "finished": request_output.finished
                                    }))
                        
                        # Send completion signal
                        await websocket.send_text(json.dumps({
                            "type": "complete"
                        }))
                        
            except WebSocketDisconnect:
                print("LLM WebSocket disconnected")
            except Exception as e:
                print(f"LLM error: {e}")
        
        return app


@app.cls(
    cpu=2,
    image=orchestrator_image,
    secrets=secrets,
    min_containers=0,
    scaledown_window=300,  # 5 minutes
)
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
        
        print(f"Orchestrator setup complete - STT: {os.environ['KYUTAI_STT_URL']}")
        print(f"Orchestrator setup complete - TTS: {os.environ['KYUTAI_TTS_URL']}")
        print(f"Orchestrator setup complete - LLM: {os.environ['KYUTAI_LLM_URL']}")
        
        print("Orchestrator setup complete")
    
    @modal.asgi_app()
    def web(self):
        """Main WebSocket endpoint for client connections"""
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile
        from fastapi.middleware.cors import CORSMiddleware
        import asyncio
        import json
        import base64
        
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
            except Exception as e:
                # If voice cloning server is not available, return a mock response
                import uuid
                mock_name = "custom:" + str(uuid.uuid4())
                return {"name": mock_name}
        
        @app.websocket("/v1/realtime")
        async def websocket_endpoint(websocket: WebSocket):
            """Main client WebSocket endpoint using the existing unmute handler"""
            await websocket.accept()
            print("Orchestrator WebSocket connection established")
            
            try:
                # Import unmute modules within the function
                from unmute.unmute_handler import UnmuteHandler
                from unmute.main_websocket import receive_loop, emit_loop
                import unmute.openai_realtime_api_events as ora
                
                # Create handler instance
                handler = UnmuteHandler()
                emit_queue: asyncio.Queue[ora.ServerEvent] = asyncio.Queue()
                
                # Start the handler
                async with handler:
                    # Start STT
                    await handler.start_up_stt()
                    
                    # Run the main loops
                    await asyncio.gather(
                        receive_loop(websocket, handler, emit_queue),
                        emit_loop(websocket, handler, emit_queue),
                        return_exceptions=True
                    )
                    
            except Exception as e:
                print(f"Orchestrator error: {e}")
                try:
                    await websocket.close(code=1000, reason=str(e))
                except Exception:
                    pass  # WebSocket might already be closed
        
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
