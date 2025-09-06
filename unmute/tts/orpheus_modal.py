"""
Modal entrypoint for Orpheus TTS using the Docker image from orpheus_fast_api
"""

import modal
import os

# Create Modal app
app = modal.App("orpheus-tts")

# Create image from the orpheus_fast_api Dockerfile
orpheus_image = modal.Image.from_dockerfile(
    "/workspace/unmute/orpheus_fast_api/Dockerfile.gpu",
    context_mount=modal.Mount.from_local_dir(
        "/workspace/unmute/orpheus_fast_api",
        remote_path="/app"
    )
)

# Create volume for model storage
model_volume = modal.Volume.from_name("orpheus-models", create_if_missing=True)

@app.cls(
    image=orpheus_image,
    gpu="L4",  # Use L4 GPU for TTS
    timeout=30 * 60,  # 30 minutes
    volumes={
        "/models": model_volume,
    },
    secrets=[
        modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"]),
    ],
    ports=[5005],  # Expose port 5005 for the FastAPI server
)
class OrpheusTTS:
    """Modal service for Orpheus TTS using the Docker image"""
    
    @modal.enter()
    def initialize(self):
        """Initialize the service"""
        import subprocess
        import time
        import os
        
        print("Initializing Orpheus TTS service...")
        
        # Set up environment variables for the service
        os.environ["ORPHEUS_HOST"] = "0.0.0.0"
        os.environ["ORPHEUS_PORT"] = "5005"
        os.environ["USE_GPU"] = "true"
        
        # Set up the API URL for the llama.cpp server
        # This should point to a running llama.cpp server with the Orpheus model
        # For now, we'll use a placeholder - this should be updated to point to your actual llama.cpp server
        os.environ["ORPHEUS_API_URL"] = os.environ.get("ORPHEUS_API_URL", "http://localhost:5006/v1/completions")
        
        # Set up model configuration
        os.environ["ORPHEUS_MODEL_NAME"] = os.environ.get("ORPHEUS_MODEL_NAME", "Orpheus-3b-FT-Q8_0.gguf")
        os.environ["ORPHEUS_MAX_TOKENS"] = os.environ.get("ORPHEUS_MAX_TOKENS", "8192")
        os.environ["ORPHEUS_TEMPERATURE"] = os.environ.get("ORPHEUS_TEMPERATURE", "0.6")
        os.environ["ORPHEUS_TOP_P"] = os.environ.get("ORPHEUS_TOP_P", "0.9")
        os.environ["ORPHEUS_SAMPLE_RATE"] = os.environ.get("ORPHEUS_SAMPLE_RATE", "24000")
        
        print("Environment variables set up successfully")
        print(f"API URL: {os.environ.get('ORPHEUS_API_URL')}")
        print(f"Model: {os.environ.get('ORPHEUS_MODEL_NAME')}")
        
        # The Docker image's CMD will automatically start the FastAPI server
        # We don't need to start it manually since Modal will handle that
        
    @modal.method()
    def generate_speech(self, text: str, voice: str = "tara", response_format: str = "wav") -> bytes:
        """Generate speech from text"""
        import requests
        import time
        
        print(f"Generating speech for text: '{text[:50]}...' with voice: {voice}")
        
        # Call the FastAPI endpoint running inside the container
        try:
            response = requests.post(
                "http://localhost:5005/v1/audio/speech",
                json={
                    "input": text,
                    "voice": voice,
                    "response_format": response_format,
                    "model": "orpheus"
                },
                timeout=120
            )
            
            if response.status_code == 200:
                print(f"Successfully generated {len(response.content)} bytes of audio")
                return response.content
            else:
                print(f"Error: {response.status_code} - {response.text}")
                raise Exception(f"TTS generation failed: {response.status_code}")
                
        except Exception as e:
            print(f"Error generating speech: {e}")
            raise
    
    @modal.asgi_app()
    def asgi_app(self):
        """Proxy ASGI app that forwards to the internal FastAPI server"""
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import Response
        from pydantic import BaseModel
        import requests
        
        app = FastAPI(title="Orpheus TTS Modal Proxy")
        
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
                # Forward to the internal FastAPI server
                response = requests.post(
                    "http://localhost:5005/v1/audio/speech",
                    json=request.dict(),
                    timeout=120
                )
                
                if response.status_code == 200:
                    return Response(
                        content=response.content,
                        media_type="audio/wav"
                    )
                else:
                    raise HTTPException(status_code=response.status_code, detail=response.text)
                    
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/health")
        async def health():
            """Health check endpoint"""
            try:
                # Check if the internal FastAPI server is responding
                response = requests.get("http://localhost:5005/", timeout=10)
                if response.status_code == 200:
                    return {"status": "healthy", "service": "orpheus-tts-modal"}
                else:
                    return {"status": "unhealthy", "error": f"Internal server returned {response.status_code}"}
            except Exception as e:
                return {"status": "unhealthy", "error": str(e)}
        
        return app


# Separate app for the llama.cpp server
llama_app = modal.App("orpheus-llama-server")

# Use the official llama.cpp server image with CUDA support
llama_image = modal.Image.from_registry(
    "ghcr.io/ggml-org/llama.cpp:server-cuda"
).pip_install("huggingface_hub")

@llama_app.cls(
    image=llama_image,
    gpu="L40S",  # Use L40S for better LLM performance
    timeout=60 * 60,  # 1 hour
    volumes={
        "/models": model_volume,
    },
    secrets=[
        modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"]),
    ],
    ports=[5006],  # Expose port 5006 for the llama.cpp server
)
class OrpheusLlamaServer:
    """Modal service for the Orpheus llama.cpp server"""
    
    @modal.enter()
    def initialize(self):
        """Initialize and start the llama.cpp server"""
        import subprocess
        import os
        import time
        from pathlib import Path
        
        print("Initializing Orpheus llama.cpp server...")
        
        # Model configuration
        model_name = os.environ.get("ORPHEUS_MODEL_NAME", "Orpheus-3b-FT-Q8_0.gguf")
        model_path = f"/models/{model_name}"
        
        # Download model if not exists
        if not Path(model_path).exists():
            print(f"Downloading model {model_name}...")
            from huggingface_hub import hf_hub_download
            
            try:
                downloaded_path = hf_hub_download(
                    repo_id=f"lex-au/{model_name}",
                    filename=model_name,
                    cache_dir="/models",
                    local_dir="/models",
                    token=os.environ.get("HF_TOKEN")
                )
                print(f"Model downloaded to {downloaded_path}")
            except Exception as e:
                print(f"Error downloading model: {e}")
                raise
        else:
            print(f"Model already exists at {model_path}")
        
        # Start llama.cpp server
        max_tokens = int(os.environ.get("ORPHEUS_MAX_TOKENS", "8192"))
        
        cmd = [
            "/app/llama-server",
            "-m", model_path,
            "--port", "5006",
            "--host", "0.0.0.0",
            "--n-gpu-layers", "29",
            "--ctx-size", str(max_tokens),
            "--n-predict", str(max_tokens),
            "--rope-scaling", "linear"
        ]
        
        print(f"Starting llama.cpp server with command: {' '.join(cmd)}")
        
        # Start the server process
        self.server_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for server to be ready
        import requests
        for i in range(60):  # Wait up to 60 seconds
            try:
                response = requests.get("http://localhost:5006/health", timeout=5)
                if response.status_code == 200:
                    print("llama.cpp server is ready")
                    break
            except:
                pass
            time.sleep(1)
        else:
            print("Warning: llama.cpp server may not be ready")
    
    @modal.exit()
    def cleanup(self):
        """Clean up the server process"""
        if hasattr(self, 'server_process') and self.server_process:
            print("Stopping llama.cpp server...")
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
    
    @modal.asgi_app()
    def asgi_app(self):
        """ASGI app that proxies to the llama.cpp server"""
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import StreamingResponse
        import requests
        import json
        
        app = FastAPI(title="Orpheus llama.cpp Server")
        
        @app.post("/v1/completions")
        async def create_completion(request: dict):
            """OpenAI-compatible completions endpoint"""
            try:
                # Forward to the internal llama.cpp server
                response = requests.post(
                    "http://localhost:5006/v1/completions",
                    json=request,
                    stream=request.get("stream", False),
                    timeout=120
                )
                
                if request.get("stream", False):
                    def generate():
                        for line in response.iter_lines():
                            if line:
                                yield line + b'\n'
                    
                    return StreamingResponse(
                        generate(),
                        media_type="text/plain"
                    )
                else:
                    return response.json()
                    
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/health")
        async def health():
            """Health check endpoint"""
            try:
                response = requests.get("http://localhost:5006/health", timeout=10)
                return response.json()
            except Exception as e:
                return {"status": "unhealthy", "error": str(e)}
        
        return app


# Export the apps
if __name__ == "__main__":
    pass