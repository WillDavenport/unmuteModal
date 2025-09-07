"""
Modal entrypoint for Orpheus TTS using the official orpheus-speech package
"""

import modal
import os
import asyncio
from typing import AsyncGenerator, Iterator

# Create Modal app
app = modal.App("orpheus-tts")

# Create Modal image with Orpheus TTS package
orpheus_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "orpheus-speech",  # Main Orpheus TTS package
        "vllm==0.7.3",     # Specific vllm version as recommended
        "fastapi",         # For REST API
        "uvicorn",         # ASGI server
        "websockets",      # For streaming support
        "pydantic",        # For request models
    ])
    .env({"HF_HOME": "/cache/huggingface"})
)

# Create volume for model storage and cache
model_volume = modal.Volume.from_name("orpheus-models")

@app.cls(
    image=orpheus_image,
    gpu="H100",
    timeout=10 * 60,  # 10 minutes
    volumes={
        "/cache": model_volume,
    },
    secrets=[
        modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"]),
    ],
    min_containers=int(os.environ.get("MIN_CONTAINERS", "0")),
    scaledown_window=600,  # 10 minutes - prevent scaling during long conversations
)
@modal.concurrent(max_inputs=5)
class OrpheusTTS:
    """Modal service for Orpheus TTS using the official orpheus-speech package"""
    
    @modal.enter()
    def initialize(self):
        """Initialize the Orpheus TTS model"""
        print("Initializing Orpheus TTS service...")
        
        from orpheus_tts import OrpheusModel
        import os
        
        # Set HuggingFace token for authenticated model access
        if "HF_TOKEN" in os.environ:
            os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]
        
        # Initialize the Orpheus model
        model_name = os.environ.get("ORPHEUS_MODEL_NAME", "canopylabs/orpheus-tts-0.1-finetune-prod")
        
        print(f"Loading Orpheus model: {model_name}")
        
        try:
            self.model = OrpheusModel(
                model_name=model_name,
            )
            print("Orpheus TTS model loaded successfully")
        except Exception as e:
            print(f"Error loading Orpheus model: {e}")
            raise
        
    @modal.method()
    def generate_speech(self, text: str, voice: str = "tara", response_format: str = "wav") -> bytes:
        """Generate speech from text using Orpheus TTS"""
        import wave
        import io
        import time
        
        print(f"Generating speech for text: '{text[:50]}...' with voice: {voice}")
        
        try:
            start_time = time.monotonic()
            first_token_time = None
            
            # Generate speech tokens using the Orpheus model
            syn_tokens = self.model.generate_speech(
                prompt=text,
                voice=voice,
            )
            
            # Create WAV file in memory
            audio_buffer = io.BytesIO()
            
            with wave.open(audio_buffer, "wb") as wf:
                wf.setnchannels(1)  # Mono
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(24000)  # 24kHz sample rate
                
                total_frames = 0
                chunk_counter = 0
                
                # Process streaming audio chunks
                for audio_chunk in syn_tokens:
                    chunk_counter += 1
                    
                    # Log time to first token
                    if first_token_time is None:
                        first_token_time = time.monotonic()
                        ttft = first_token_time - start_time
                        print(f"Time to first token: {ttft:.3f}s")
                    
                    frame_count = len(audio_chunk) // (wf.getsampwidth() * wf.getnchannels())
                    total_frames += frame_count
                    wf.writeframes(audio_chunk)
                
                duration = total_frames / wf.getframerate()
            
            end_time = time.monotonic()
            generation_time = end_time - start_time
            
            print(f"Generated {duration:.2f}s of audio in {generation_time:.2f}s ({chunk_counter} chunks)")
            
            # Return the WAV file bytes
            audio_buffer.seek(0)
            return audio_buffer.getvalue()
            
        except Exception as e:
            print(f"Error generating speech: {e}")
            raise
    
    @modal.method()
    def generate_speech_stream(self, text: str, voice: str = "tara") -> Iterator[bytes]:
        """Generate speech from text with streaming output (raw audio data)"""
        import time
        
        print(f"Generating streaming speech for text: '{text[:50]}...' with voice: {voice}")
        
        try:
            start_time = time.monotonic()
            first_token_time = None
            
            # Generate speech tokens using the Orpheus model
            syn_tokens = self.model.generate_speech(
                prompt=text,
                voice=voice,
            )
            
            chunk_counter = 0
            
            # Stream raw audio chunks as they're generated
            for audio_chunk in syn_tokens:
                chunk_counter += 1
                
                # Log time to first token
                if first_token_time is None:
                    first_token_time = time.monotonic()
                    ttft = first_token_time - start_time
                    print(f"Time to first token: {ttft:.3f}s")
                
                # Return raw audio data (no WAV headers per chunk)
                yield audio_chunk
            
            end_time = time.monotonic()
            generation_time = end_time - start_time
            
            print(f"Completed streaming generation in {generation_time:.2f}s ({chunk_counter} chunks)")
            
        except Exception as e:
            print(f"Error generating streaming speech: {e}")
            raise
    
    @modal.asgi_app()
    def asgi_app(self):
        """FastAPI app with OpenAI-compatible TTS endpoints"""
        from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
        from fastapi.responses import Response, StreamingResponse
        from pydantic import BaseModel
        import json
        import asyncio
        
        app = FastAPI(title="Orpheus TTS Modal Service")
        
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
                # Generate speech using the Orpheus model
                audio_bytes = self.generate_speech(
                    text=request.input,
                    voice=request.voice,
                    response_format=request.response_format
                )
                
                return Response(
                    content=audio_bytes,
                    media_type="audio/wav",
                    headers={
                        "Content-Disposition": "attachment; filename=speech.wav",
                        "Content-Length": str(len(audio_bytes))
                    }
                )
                    
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/v1/audio/speech/stream")
        async def create_speech_stream(request: SpeechRequest):
            """Streaming TTS endpoint - returns raw audio chunks"""
            try:
                def generate_audio():
                    # Stream raw audio chunks (client needs to handle WAV formatting)
                    for chunk in self.generate_speech_stream(
                        text=request.input,
                        voice=request.voice
                    ):
                        yield chunk
                
                return StreamingResponse(
                    generate_audio(),
                    media_type="audio/raw",
                    headers={
                        "X-Audio-Format": "raw-pcm",
                        "X-Audio-Sample-Rate": "24000",
                        "X-Audio-Channels": "1",
                        "X-Audio-Sample-Width": "2"
                    }
                )
                    
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time TTS streaming"""
            await websocket.accept()
            
            try:
                while True:
                    # Receive text input from client
                    data = await websocket.receive_text()
                    message = json.loads(data)
                    
                    if message.get("type") == "text":
                        text = message.get("text", "")
                        voice = message.get("voice", "tara")
                        
                        if text.strip():
                            # Send audio format info first
                            try:
                                await websocket.send_text(json.dumps({
                                    "type": "audio_format",
                                    "sample_rate": 24000,
                                    "channels": 1,
                                    "sample_width": 2,
                                    "format": "raw-pcm"
                                }))
                                
                                # Stream raw audio chunks back to client
                                for chunk in self.generate_speech_stream(text=text, voice=voice):
                                    await websocket.send_bytes(chunk)
                                
                                # Send completion message
                                await websocket.send_text(json.dumps({
                                    "type": "complete",
                                    "message": "Audio generation complete"
                                }))
                            except Exception as e:
                                await websocket.send_text(json.dumps({
                                    "type": "error",
                                    "error": str(e)
                                }))
                    
            except WebSocketDisconnect:
                print("WebSocket client disconnected")
            except Exception as e:
                print(f"WebSocket error: {e}")
                try:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "error": str(e)
                    }))
                except:
                    pass
        
        @app.get("/health")
        async def health():
            """Health check endpoint"""
            try:
                # Simple health check - verify model is loaded
                if hasattr(self, 'model') and self.model is not None:
                    return {"status": "healthy", "service": "orpheus-tts-modal"}
                else:
                    return {"status": "unhealthy", "error": "Model not loaded"}
            except Exception as e:
                return {"status": "unhealthy", "error": str(e)}
        
        @app.get("/")
        async def root():
            """Root endpoint"""
            return {
                "service": "Orpheus TTS Modal Service",
                "version": "1.0.0",
                "endpoints": [
                    "/v1/audio/speech",
                    "/v1/audio/speech/stream", 
                    "/ws",
                    "/health"
                ]
            }
        
        return app


@app.local_entrypoint()
def test_orpheus_tts():
    """Test the Orpheus TTS service with streaming generation and performance comparison."""
    print("Testing Orpheus TTS Service...")
    
    # Test texts - similar length for fair comparison
    test_text_1 = "This is a test of the Orpheus TTS service streaming functionality in our Modal deployment."
    test_text_2 = "Here we are testing the second generation to see if subsequent calls are faster than the first."
    
    print(f"Text 1 length: {len(test_text_1)} characters")
    print(f"Text 2 length: {len(test_text_2)} characters")
    
    try:
        # Create service instance
        service = OrpheusTTS()
        
        # Test health check via ASGI app (if available)
        print("Service instance created successfully")
        
        print("\n" + "="*60)
        print("FIRST GENERATION TEST")
        print("="*60)
        
        # Test speech generation (non-streaming) - First generation
        print(f"Generating speech for: '{test_text_1}'")
        import time
        start_time = time.monotonic()
        result = service.generate_speech.remote(
            text=test_text_1,
            voice="tara",
            response_format="wav"
        )
        end_time = time.monotonic()
        first_gen_total_time = end_time - start_time
        
        if result:
            print(f"First generation successful!")
            print(f"Total time (including network): {first_gen_total_time:.3f}s")
            print(f"Audio data length: {len(result)} bytes")
            
            # Save non-streaming audio to file for testing
            with open("orpheus_test_output_1.wav", "wb") as f:
                f.write(result)
            print("First generation audio saved to orpheus_test_output_1.wav")
        else:
            print("First speech generation failed: No audio data returned")
        
        # Test streaming speech generation - First generation
        print(f"\nTesting streaming speech generation for: '{test_text_1}'")
        stream_chunks = []
        chunk_count = 0
        first_chunk_time = None
        
        stream_start = time.monotonic()
        for chunk in service.generate_speech_stream.remote_gen(
            text=test_text_1,
            voice="tara"
        ):
            if first_chunk_time is None:
                first_chunk_time = time.monotonic()
            chunk_count += 1
            stream_chunks.append(chunk)
            print(f"Received chunk {chunk_count}: {len(chunk)} bytes")
        stream_end = time.monotonic()
        first_stream_total_time = stream_end - stream_start
        first_ttft = first_chunk_time - stream_start if first_chunk_time else None
        
        if stream_chunks:
            # Combine all raw audio chunks
            combined_raw_audio = b"".join(stream_chunks)
            print(f"First streaming generation successful!")
            print(f"Total streaming time (including network): {first_stream_total_time:.3f}s")
            if first_ttft:
                print(f"Time to first token (TTFT): {first_ttft:.3f}s")
            print(f"Total chunks received: {chunk_count}")
            print(f"Combined raw audio data length: {len(combined_raw_audio)} bytes")
            
            # Create proper WAV file from raw audio data
            import wave
            import io
            
            audio_buffer = io.BytesIO()
            with wave.open(audio_buffer, "wb") as wf:
                wf.setnchannels(1)  # Mono
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(24000)  # 24kHz sample rate
                wf.writeframes(combined_raw_audio)
            
            audio_buffer.seek(0)
            wav_data = audio_buffer.getvalue()
            
            # Calculate duration for verification
            duration = len(combined_raw_audio) // (2 * 1 * 24000)  # bytes / (sample_width * channels * sample_rate)
            print(f"Expected audio duration: {duration:.2f}s")
            
            # Calculate real-time factor (RTF)
            if duration > 0:
                rtf = first_stream_total_time / duration
                print(f"Real-time factor (RTF): {rtf:.3f}x (lower is better)")
            
            # Save streaming audio to file for testing
            with open("orpheus_streaming_test_output_1.wav", "wb") as f:
                f.write(wav_data)
            print("First streaming audio saved to orpheus_streaming_test_output_1.wav")
        else:
            print("First streaming speech generation failed: No chunks received")
        
        print("\n" + "="*60)
        print("SECOND GENERATION TEST")
        print("="*60)
        
        # Test speech generation (non-streaming) - Second generation
        print(f"Generating speech for: '{test_text_2}'")
        start_time = time.monotonic()
        result_2 = service.generate_speech.remote(
            text=test_text_2,
            voice="tara",
            response_format="wav"
        )
        end_time = time.monotonic()
        second_gen_total_time = end_time - start_time
        
        if result_2:
            print(f"Second generation successful!")
            print(f"Total time (including network): {second_gen_total_time:.3f}s")
            print(f"Audio data length: {len(result_2)} bytes")
            
            # Save non-streaming audio to file for testing
            with open("orpheus_test_output_2.wav", "wb") as f:
                f.write(result_2)
            print("Second generation audio saved to orpheus_test_output_2.wav")
        else:
            print("Second speech generation failed: No audio data returned")
        
        # Test streaming speech generation - Second generation
        print(f"\nTesting streaming speech generation for: '{test_text_2}'")
        stream_chunks_2 = []
        chunk_count_2 = 0
        second_first_chunk_time = None
        
        stream_start_2 = time.monotonic()
        for chunk in service.generate_speech_stream.remote_gen(
            text=test_text_2,
            voice="tara"
        ):
            if second_first_chunk_time is None:
                second_first_chunk_time = time.monotonic()
            chunk_count_2 += 1
            stream_chunks_2.append(chunk)
            print(f"Received chunk {chunk_count_2}: {len(chunk)} bytes")
        stream_end_2 = time.monotonic()
        second_stream_total_time = stream_end_2 - stream_start_2
        second_ttft = second_first_chunk_time - stream_start_2 if second_first_chunk_time else None
        
        if stream_chunks_2:
            # Combine all raw audio chunks
            combined_raw_audio_2 = b"".join(stream_chunks_2)
            print(f"Second streaming generation successful!")
            print(f"Total streaming time (including network): {second_stream_total_time:.3f}s")
            if second_ttft:
                print(f"Time to first token (TTFT): {second_ttft:.3f}s")
            print(f"Total chunks received: {chunk_count_2}")
            print(f"Combined raw audio data length: {len(combined_raw_audio_2)} bytes")
            
            # Create proper WAV file from raw audio data
            audio_buffer_2 = io.BytesIO()
            with wave.open(audio_buffer_2, "wb") as wf:
                wf.setnchannels(1)  # Mono
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(24000)  # 24kHz sample rate
                wf.writeframes(combined_raw_audio_2)
            
            audio_buffer_2.seek(0)
            wav_data_2 = audio_buffer_2.getvalue()
            
            # Calculate duration for verification
            duration_2 = len(combined_raw_audio_2) // (2 * 1 * 24000)  # bytes / (sample_width * channels * sample_rate)
            print(f"Expected audio duration: {duration_2:.2f}s")
            
            # Calculate real-time factor (RTF)
            if duration_2 > 0:
                rtf_2 = second_stream_total_time / duration_2
                print(f"Real-time factor (RTF): {rtf_2:.3f}x (lower is better)")
            
            # Save streaming audio to file for testing
            with open("orpheus_streaming_test_output_2.wav", "wb") as f:
                f.write(wav_data_2)
            print("Second streaming audio saved to orpheus_streaming_test_output_2.wav")
        else:
            print("Second streaming speech generation failed: No chunks received")
        
        print("\n" + "="*60)
        print("PERFORMANCE COMPARISON")
        print("="*60)
        
        if 'first_gen_total_time' in locals() and 'second_gen_total_time' in locals():
            print(f"Non-streaming generation times:")
            print(f"  First generation:  {first_gen_total_time:.3f}s")
            print(f"  Second generation: {second_gen_total_time:.3f}s")
            time_diff = first_gen_total_time - second_gen_total_time
            if time_diff > 0:
                print(f"  Second generation was {time_diff:.3f}s faster ({(time_diff/first_gen_total_time*100):.1f}% improvement)")
            else:
                print(f"  First generation was {abs(time_diff):.3f}s faster ({(abs(time_diff)/second_gen_total_time*100):.1f}% better)")
        
        if 'first_stream_total_time' in locals() and 'second_stream_total_time' in locals():
            print(f"\nStreaming generation times:")
            print(f"  First streaming:  {first_stream_total_time:.3f}s")
            print(f"  Second streaming: {second_stream_total_time:.3f}s")
            stream_time_diff = first_stream_total_time - second_stream_total_time
            if stream_time_diff > 0:
                print(f"  Second streaming was {stream_time_diff:.3f}s faster ({(stream_time_diff/first_stream_total_time*100):.1f}% improvement)")
            else:
                print(f"  First streaming was {abs(stream_time_diff):.3f}s faster ({(abs(stream_time_diff)/second_stream_total_time*100):.1f}% better)")
        
        # Time to First Token comparison
        if 'first_ttft' in locals() and 'second_ttft' in locals() and first_ttft and second_ttft:
            print(f"\nTime to First Token (TTFT) comparison:")
            print(f"  First generation TTFT:  {first_ttft:.3f}s")
            print(f"  Second generation TTFT: {second_ttft:.3f}s")
            ttft_diff = first_ttft - second_ttft
            if ttft_diff > 0:
                print(f"  Second generation TTFT was {ttft_diff:.3f}s faster ({(ttft_diff/first_ttft*100):.1f}% improvement)")
            else:
                print(f"  First generation TTFT was {abs(ttft_diff):.3f}s faster ({(abs(ttft_diff)/second_ttft*100):.1f}% better)")
        
        # Real-time Factor comparison
        if 'rtf' in locals() and 'rtf_2' in locals():
            print(f"\nReal-time Factor (RTF) comparison:")
            print(f"  First generation RTF:  {rtf:.3f}x")
            print(f"  Second generation RTF: {rtf_2:.3f}x")
            rtf_diff = rtf - rtf_2
            if rtf_diff > 0:
                print(f"  Second generation RTF was {rtf_diff:.3f}x better (lower RTF is better)")
            else:
                print(f"  First generation RTF was {abs(rtf_diff):.3f}x better (lower RTF is better)")
            
    except Exception as e:
        print(f"Error testing Orpheus TTS: {e}")
        import traceback
        traceback.print_exc()
        raise


# Export the app
if __name__ == "__main__":
    pass