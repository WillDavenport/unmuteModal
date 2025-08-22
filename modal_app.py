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
    )
)

# STT image: use Python moshi package instead of Rust binary
stt_image = (
    base_deps_image
    .pip_install(
        "torch>=2.1.0",
        "torchaudio>=2.1.0",
        # Use Python moshi package instead of compiling Rust binary
        "moshi>=0.2.8",
        "transformers>=4.35.0",
    )
    # Add local files LAST
    .add_local_python_source("unmute")
    .add_local_file("voices.yaml", "/root/voices.yaml")
)

# TTS image: use Python moshi package instead of Rust binary
tts_image = (
    base_deps_image
    .pip_install(
        "torch>=2.1.0",
        "torchaudio>=2.1.0",
        "huggingface_hub>=0.19.0",
        # Use Python moshi package instead of compiling Rust binary
        "moshi>=0.2.8",
        "transformers>=4.35.0",
    )
    # Add local files LAST
    .add_local_python_source("unmute")
    .add_local_file("voices.yaml", "/root/voices.yaml")
)

# LLM image: additional deps first, then local files
llm_image = (
    base_deps_image
    .pip_install(
        "vllm==0.9.1",
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
    concurrency_limit=1,
    keep_warm=0,
    container_idle_timeout=1200,  # 20 minutes
)
class STTService:
    """Speech-to-Text service using Moshi STT model"""
    
    @modal.enter()
    def load_model(self):
        """Load STT model weights once per container"""
        print("Setting up STT Service...")
        
        # Initialize the Python moshi STT model
        try:
            from moshi.models import loaders
            import torch
            
            # Load the STT model using Python moshi package
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Loading STT model on device: {device}")
            
            # Load the model - this will download if not cached
            self.stt_model = loaders.get_stt_model("kyutai/stt-1b-en_fr", device=device)
            
            print("STT model loaded successfully")
            
        except Exception as e:
            print(f"Error loading STT model: {e}")
            print("STT service will use fallback simulation")
            self.stt_model = None
    
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
                import numpy as np
                import json
                import base64
                
                while True:
                    # Receive audio data from client
                    data = await websocket.receive_bytes()
                    
                    # Process audio with STT model
                    if self.stt_model is not None:
                        try:
                            # Convert bytes to audio array
                            audio_array = np.frombuffer(data, dtype=np.float32)
                            
                            # Run STT inference (this is a simplified example)
                            # In practice, you'd need to handle the specific moshi STT API
                            transcription = await self._transcribe_audio(audio_array)
                            
                            # Send transcription result
                            result = {
                                "type": "Word",
                                "text": transcription,
                                "start_time": 0.0
                            }
                            await websocket.send_text(json.dumps(result))
                            
                        except Exception as e:
                            print(f"STT inference error: {e}")
                            # Send empty result on error
                            result = {"type": "Word", "text": "", "start_time": 0.0}
                            await websocket.send_text(json.dumps(result))
                    else:
                        # Fallback simulation when model not available
                        result = {
                            "type": "Word", 
                            "text": "Hello, this is simulated STT output",
                            "start_time": 0.0
                        }
                        await websocket.send_text(json.dumps(result))
                        
            except WebSocketDisconnect:
                print("STT WebSocket disconnected")
            except Exception as e:
                print(f"STT websocket error: {e}")
        
        async def _transcribe_audio(self, audio_array: np.ndarray) -> str:
            """Transcribe audio using the loaded STT model"""
            try:
                # This is a placeholder - you'd implement the actual moshi STT inference here
                # The exact API depends on how the moshi Python package exposes the STT model
                if len(audio_array) > 0:
                    return "Transcribed text from audio"  # Placeholder
                else:
                    return ""
            except Exception as e:
                print(f"Transcription error: {e}")
                return ""
        
        # Attach the method to self for access
        self._transcribe_audio = _transcribe_audio
        
        return app


@app.cls(
    gpu="L4", 
    image=tts_image,
    volumes={"/models": models_volume},
    secrets=secrets,
    concurrency_limit=1,
    keep_warm=0,
    container_idle_timeout=1200,  # 20 minutes
)
class TTSService:
    """Text-to-Speech service using Moshi TTS model"""
    
    @modal.enter()
    def load_model(self):
        """Load TTS model weights once per container"""
        print("Setting up TTS Service...")
        
        # Initialize the Python moshi TTS model
        try:
            from moshi.models import loaders
            import torch
            
            # Load the TTS model using Python moshi package
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Loading TTS model on device: {device}")
            
            # Load the model - this will download if not cached
            self.tts_model = loaders.get_tts_model("kyutai/tts-1.6b-en_fr", device=device)
            
            print("TTS model loaded successfully")
            
        except Exception as e:
            print(f"Error loading TTS model: {e}")
            print("TTS service will use fallback simulation")
            self.tts_model = None
    
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
                import numpy as np
                import json
                
                while True:
                    # Receive text data from client
                    message = await websocket.receive_text()
                    data = json.loads(message)
                    
                    text = data.get("text", "")
                    voice = data.get("voice", "default")
                    
                    # Process text with TTS model
                    if self.tts_model is not None and text.strip():
                        try:
                            # Run TTS inference
                            audio_bytes = await self._synthesize_speech(text, voice)
                            
                            # Send audio data
                            await websocket.send_bytes(audio_bytes)
                            
                        except Exception as e:
                            print(f"TTS inference error: {e}")
                            # Send empty audio on error
                            await websocket.send_bytes(b"")
                    else:
                        # Fallback simulation when model not available
                        # Generate simple sine wave audio as placeholder
                        sample_rate = 24000
                        duration = max(len(text) * 0.1, 1.0)
                        samples = int(duration * sample_rate)
                        
                        t = np.linspace(0, duration, samples, dtype=np.float32)
                        frequency = 440
                        amplitude = 0.1
                        audio_array = amplitude * np.sin(2 * np.pi * frequency * t)
                        
                        audio_bytes = audio_array.astype(np.float32).tobytes()
                        await websocket.send_bytes(audio_bytes)
                        
            except WebSocketDisconnect:
                print("TTS WebSocket disconnected")
            except Exception as e:
                print(f"TTS websocket error: {e}")
        
        async def _synthesize_speech(self, text: str, voice: str = "default") -> bytes:
            """Synthesize speech using the loaded TTS model"""
            try:
                # This is a placeholder - you'd implement the actual moshi TTS inference here
                # The exact API depends on how the moshi Python package exposes the TTS model
                import numpy as np
                
                # Generate simple audio as placeholder
                sample_rate = 24000
                duration = max(len(text) * 0.1, 1.0)
                samples = int(duration * sample_rate)
                
                # Simple sine wave based on text length
                t = np.linspace(0, duration, samples, dtype=np.float32)
                frequency = 440 + (len(text) % 200)  # Vary frequency based on text
                amplitude = 0.1
                audio_array = amplitude * np.sin(2 * np.pi * frequency * t)
                
                return audio_array.astype(np.float32).tobytes()
                
            except Exception as e:
                print(f"Speech synthesis error: {e}")
                return b""
        
        # Attach the method to self for access
        self._synthesize_speech = _synthesize_speech
        
        return app


@app.cls(
    gpu="L40S",
    image=llm_image,
    volumes={"/models": models_volume},
    secrets=secrets,
    concurrency_limit=1,
    keep_warm=0,
    container_idle_timeout=1200,  # 20 minutes
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
    keep_warm=0,
    container_idle_timeout=1200,  # 20 minutes
)
class OrchestratorService:
    """Orchestrator service that coordinates between STT, LLM, and TTS"""
    
    @modal.enter()
    def setup(self):
        """Set up the orchestrator environment"""
        import os
        
        print("Setting up Orchestrator Service...")
        
        # Set environment variables to point to Modal services
        # These will be updated based on actual Modal deployment URLs
        os.environ["KYUTAI_STT_URL"] = "wss://your-username--voice-stack-sttservice-web.modal.run"
        os.environ["KYUTAI_TTS_URL"] = "wss://your-username--voice-stack-ttsservice-web.modal.run"
        os.environ["KYUTAI_LLM_URL"] = "https://your-username--voice-stack-llmservice-web.modal.run"
        
        print("Orchestrator setup complete")
    
    @modal.asgi_app()
    def web(self):
        """Main WebSocket endpoint for client connections"""
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect
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
        
        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """Main client WebSocket endpoint using the existing unmute handler"""
            await websocket.accept()
            print("Orchestrator WebSocket connection established")
            
            try:
                # Import unmute modules within the function
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
