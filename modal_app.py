import asyncio
import base64
import json
import logging
import os
import tempfile
from typing import Dict, Optional, Any

import modal
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# Modal app definition
app = modal.App("voice-stack")

# Common image with basic dependencies
base_image = modal.Image.debian_slim().pip_install(
    "fastapi",
    "uvicorn", 
    "websockets",
    "numpy",
    "pydantic",
    "msgpack",
    "requests",
    "prometheus-fastapi-instrumentator",
    "sphn",
    "fastrtc",
)

# GPU images for each service
stt_image = base_image.pip_install(
    "torch",
    "torchaudio",
).run_commands(
    # Install Rust
    "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
    "echo 'source ~/.cargo/env' >> ~/.bashrc",
    # Set environment variables for building
    "export CXXFLAGS='-include cstdint'",
    # Install moshi-server
    "source ~/.cargo/env && cargo install --features cuda moshi-server@0.6.3"
)

tts_image = base_image.pip_install(
    "torch",
    "torchaudio",
    "huggingface_hub",
    "uv",  # For Python environment management
).run_commands(
    # Install Rust
    "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
    "echo 'source ~/.cargo/env' >> ~/.bashrc",
    # Install uv for Python package management
    "curl -LsSf https://astral.sh/uv/install.sh | sh",
    # Set environment variables for building
    "export CXXFLAGS='-include cstdint'",
    # Install moshi-server
    "source ~/.cargo/env && cargo install --features cuda moshi-server@0.6.3"
)

llm_image = base_image.pip_install(
    "vllm==0.9.1",
    "torch",
    "transformers",
    "openai",
)

# Add the unmute package to orchestrator image
orchestrator_image = base_image.pip_install(
    "fastapi",
    "uvicorn",
    "websockets", 
    "numpy",
    "pydantic",
    "msgpack",
    "requests",
    "prometheus-fastapi-instrumentator",
    "sphn",
    "fastrtc",
    "openai",
).copy_local_dir("unmute", "/app/unmute")

# Modal volumes for model storage
models_volume = modal.Volume.from_name("voice-models", create_if_missing=True)

# Modal secrets for API keys and auth tokens
auth_secret = modal.Secret.from_name("voice-auth", create_if_missing=True)


@app.cls(
    gpu="L4",
    image=stt_image,
    volumes={"/models": models_volume},
    secrets=[auth_secret],
    concurrency_limit=1,
    keep_warm=1,
    container_idle_timeout=1200,  # 20 minutes
)
class STTService:
    """Speech-to-Text service using Moshi STT model"""
    
    @modal.enter()
    def load_model(self):
        """Load STT model weights once per container"""
        import subprocess
        import os
        
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
        self.proc = subprocess.Popen([
            "moshi-server", "worker", 
            "--config", self.config_path,
            "--port", "8090"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a bit for the server to start
        import time
        time.sleep(5)
        
        print("STT model loaded successfully")
    
    @modal.asgi_app()
    def web(self):
        """WebSocket endpoint for STT service"""
        app = FastAPI()
        
        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            
            try:
                # Connect to the local moshi server and proxy messages
                import websockets
                async with websockets.connect(
                    "ws://localhost:8090/api/asr-streaming",
                    extra_headers={"kyutai-api-key": "public_token"}
                ) as moshi_ws:
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
    secrets=[auth_secret],
    concurrency_limit=1,
    keep_warm=1,
)
class TTSService:
    """Text-to-Speech service using Moshi TTS model"""
    
    @modal.enter()
    def load_model(self):
        """Load TTS model weights once per container"""
        import subprocess
        import tempfile
        import os
        
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
            
        print("TTS model loaded successfully")
    
    @modal.asgi_app()
    def web(self):
        """WebSocket endpoint for TTS service"""
        app = FastAPI()
        
        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            
            # Start moshi-server as subprocess
            import subprocess
            import asyncio
            
            proc = subprocess.Popen([
                "moshi-server", "worker",
                "--config", self.config_path, 
                "--port", "8089"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            try:
                # Connect to the moshi server and proxy messages
                import websockets
                async with websockets.connect("ws://localhost:8089/api/tts_streaming") as moshi_ws:
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
                        except Exception:
                            pass
                    
                    # Run both directions concurrently
                    await asyncio.gather(
                        client_to_moshi(),
                        moshi_to_client(),
                        return_exceptions=True
                    )
            finally:
                proc.terminate()
                proc.wait()
        
        return app


@app.cls(
    gpu="L40S",
    image=llm_image,
    volumes={"/models": models_volume},
    secrets=[auth_secret],
    concurrency_limit=1,
    keep_warm=1,
)
class LLMService:
    """Large Language Model service using VLLM"""
    
    @modal.enter()
    def load_model(self):
        """Load LLM model weights once per container"""
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
        app = FastAPI()
        
        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            
            try:
                while True:
                    # Receive prompt from orchestrator
                    message = await websocket.receive_text()
                    data = json.loads(message)
                    
                    if data.get("type") == "generate":
                        prompt = data.get("prompt", "")
                        
                        # Generate streaming response
                        from vllm import SamplingParams
                        sampling_params = SamplingParams(
                            temperature=0.7,
                            max_tokens=1024,
                        )
                        
                        request_id = "req_" + str(hash(prompt))
                        
                        # Add request to engine
                        await self.engine.add_request(
                            request_id=request_id,
                            prompt=prompt,
                            sampling_params=sampling_params
                        )
                        
                        # Stream results
                        async for output in self.engine.generate_stream(request_id):
                            if output.outputs:
                                text = output.outputs[0].text
                                await websocket.send_text(json.dumps({
                                    "type": "token",
                                    "text": text
                                }))
                        
                        # Send completion signal
                        await websocket.send_text(json.dumps({
                            "type": "complete"
                        }))
                        
            except WebSocketDisconnect:
                pass
        
        return app


@app.cls(
    cpu=2,
    image=orchestrator_image,
    secrets=[auth_secret],
    keep_warm=1,
    container_idle_timeout=1200,  # 20 minutes
)
class OrchestratorService:
    """Orchestrator service that coordinates between STT, LLM, and TTS"""
    
    @modal.enter()
    def setup(self):
        """Set up the orchestrator environment"""
        import sys
        sys.path.append("/app")
        
        # Set environment variables to point to Modal services
        os.environ["KYUTAI_STT_URL"] = "wss://kyutai-labs--voice-stack-sttservice-web.modal.run"
        os.environ["KYUTAI_TTS_URL"] = "wss://kyutai-labs--voice-stack-ttsservice-web.modal.run"
        os.environ["KYUTAI_LLM_URL"] = "https://kyutai-labs--voice-stack-llmservice-web.modal.run"
        
        print("Orchestrator setup complete")
    
    @modal.asgi_app()
    def web(self):
        """Main WebSocket endpoint for client connections"""
        app = FastAPI()
        
        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """Main client WebSocket endpoint using the existing unmute handler"""
            await websocket.accept()
            
            try:
                # Import unmute modules
                from unmute.unmute_handler import UnmuteHandler
                from unmute.main_websocket import receive_loop, emit_loop
                import unmute.openai_realtime_api_events as ora
                from fastrtc import OpusReader
                
                # Create handler instance
                handler = UnmuteHandler()
                emit_queue: asyncio.Queue[ora.ServerEvent] = asyncio.Queue()
                opus_reader = OpusReader()
                
                # Start the handler
                async with handler:
                    # Start STT
                    await handler.start_up_stt()
                    
                    # Run the main loops
                    await asyncio.gather(
                        receive_loop(websocket, handler, emit_queue, opus_reader),
                        emit_loop(websocket, handler, emit_queue),
                        return_exceptions=True
                    )
                    
            except Exception as e:
                print(f"Orchestrator error: {e}")
                                 await websocket.close(code=1000, reason=str(e))
        
        return app


# Additional function to deploy all services
@app.function()
def deploy_all():
    """Deploy all services"""
    print("All services deployed to Modal!")
    return "Deployment complete"


if __name__ == "__main__":
    # For local development
    import modal
    modal.runner.deploy_app(app)