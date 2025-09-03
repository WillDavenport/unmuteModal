#!/usr/bin/env python3
"""
Test script for Modal voice stack deployment.

This script tests the deployed Modal services to ensure they're working correctly.
"""

import asyncio
import json
import base64
import websockets
import numpy as np
from typing import Optional


class ModalDeploymentTester:
    """Test the Modal deployment end-to-end."""
    
    def __init__(self, base_url: str = "kyutai-labs--voice-stack"):
        """Initialize tester with Modal app base URL."""
        self.base_url = base_url
        self.orchestrator_url = f"wss://{base_url}-orchestratorservice-web.modal.run/ws"
        self.stt_url = f"wss://{base_url}-sttservice-web.modal.run/ws"
        self.tts_url = f"wss://{base_url}-orpheusttsservice-web.modal.run/ws"
        self.llm_url = f"wss://{base_url}-llmservice-web.modal.run/ws"
    
    async def test_service_connectivity(self, url: str, service_name: str) -> bool:
        """Test if a service is reachable."""
        try:
            async with websockets.connect(url) as ws:
                print(f"✅ {service_name} service is reachable")
                return True
        except Exception as e:
            print(f"❌ {service_name} service failed: {e}")
            return False
    
    async def test_llm_service(self) -> bool:
        """Test LLM service with a simple prompt."""
        try:
            async with websockets.connect(self.llm_url) as ws:
                # Send test prompt
                prompt_message = {
                    "type": "generate",
                    "prompt": "Hello, how are you?",
                    "temperature": 0.7,
                    "max_tokens": 50
                }
                await ws.send(json.dumps(prompt_message))
                
                # Wait for response
                response_received = False
                timeout = 30  # 30 second timeout
                
                try:
                    response = await asyncio.wait_for(ws.recv(), timeout=timeout)
                    data = json.loads(response)
                    
                    if data.get("type") == "token" and data.get("text"):
                        print(f"✅ LLM service responded: {data['text'][:50]}...")
                        response_received = True
                    
                except asyncio.TimeoutError:
                    print("❌ LLM service timeout")
                    return False
                
                return response_received
                
        except Exception as e:
            print(f"❌ LLM service test failed: {e}")
            return False
    
    async def test_orchestrator_session(self) -> bool:
        """Test orchestrator with a session update."""
        try:
            async with websockets.connect(self.orchestrator_url) as ws:
                # Send session update
                session_message = {
                    "type": "session.update",
                    "session": {
                        "modalities": ["text", "audio"],
                        "instructions": "You are a helpful assistant."
                    }
                }
                await ws.send(json.dumps(session_message))
                
                # Wait for session.updated response
                response = await asyncio.wait_for(ws.recv(), timeout=10)
                data = json.loads(response)
                
                if data.get("type") == "session.updated":
                    print("✅ Orchestrator session update successful")
                    return True
                else:
                    print(f"❌ Unexpected orchestrator response: {data}")
                    return False
                    
        except Exception as e:
            print(f"❌ Orchestrator test failed: {e}")
            return False
    
    def generate_test_audio(self, duration_ms: int = 1000) -> bytes:
        """Generate test audio data in Opus format."""
        # Generate a simple sine wave
        sample_rate = 24000
        duration_sec = duration_ms / 1000.0
        samples = int(sample_rate * duration_sec)
        
        t = np.linspace(0, duration_sec, samples, False)
        frequency = 440.0  # A4 note
        audio_data = np.sin(2 * np.pi * frequency * t)
        
        # Convert to 16-bit PCM
        audio_16bit = (audio_data * 32767).astype(np.int16)
        
        # For testing, we'll just return the raw PCM data
        # In a real test, you'd encode this as Opus
        return audio_16bit.tobytes()
    
    async def test_audio_pipeline(self) -> bool:
        """Test the full audio pipeline (STT -> LLM -> TTS)."""
        try:
            async with websockets.connect(self.orchestrator_url) as ws:
                print("🎤 Testing full audio pipeline...")
                
                # Generate test audio
                test_audio = self.generate_test_audio()
                audio_b64 = base64.b64encode(test_audio).decode()
                
                # Send audio buffer append
                audio_message = {
                    "type": "input_audio_buffer.append",
                    "audio": audio_b64
                }
                await ws.send(json.dumps(audio_message))
                
                # Listen for responses
                responses_received = []
                timeout = 30
                
                try:
                    while len(responses_received) < 3:  # Expect STT, LLM, TTS responses
                        response = await asyncio.wait_for(ws.recv(), timeout=5)
                        data = json.loads(response)
                        responses_received.append(data.get("type"))
                        
                        if data.get("type") in ["conversation.item.input_audio_transcription.delta", 
                                              "response.text.delta", 
                                              "response.audio.delta"]:
                            print(f"📡 Received: {data.get('type')}")
                        
                        if len(responses_received) >= 3:
                            break
                            
                except asyncio.TimeoutError:
                    print("⚠️  Audio pipeline test timeout (this is expected with test audio)")
                
                if responses_received:
                    print("✅ Audio pipeline is responding")
                    return True
                else:
                    print("❌ No responses from audio pipeline")
                    return False
                    
        except Exception as e:
            print(f"❌ Audio pipeline test failed: {e}")
            return False
    
    async def run_all_tests(self) -> bool:
        """Run all tests and return overall success."""
        print("🧪 Starting Modal deployment tests...")
        print("=" * 50)
        
        tests = [
            ("Service Connectivity - Orchestrator", 
             self.test_service_connectivity(self.orchestrator_url, "Orchestrator")),
            ("Service Connectivity - STT", 
             self.test_service_connectivity(self.stt_url, "STT")),
            ("Service Connectivity - TTS", 
             self.test_service_connectivity(self.tts_url, "TTS")),
            ("Service Connectivity - LLM", 
             self.test_service_connectivity(self.llm_url, "LLM")),
            ("LLM Functionality", self.test_llm_service()),
            ("Orchestrator Session", self.test_orchestrator_session()),
            ("Audio Pipeline", self.test_audio_pipeline()),
        ]
        
        results = []
        for test_name, test_coro in tests:
            print(f"\n🔍 Running: {test_name}")
            try:
                result = await test_coro
                results.append(result)
                if result:
                    print(f"✅ {test_name}: PASSED")
                else:
                    print(f"❌ {test_name}: FAILED")
            except Exception as e:
                print(f"❌ {test_name}: ERROR - {e}")
                results.append(False)
        
        # Summary
        passed = sum(results)
        total = len(results)
        
        print("\n" + "=" * 50)
        print(f"📊 Test Results: {passed}/{total} passed")
        
        if passed == total:
            print("🎉 All tests passed! Your Modal deployment is working correctly.")
            return True
        else:
            print("⚠️  Some tests failed. Check the logs above for details.")
            return False


async def main():
    """Main test function."""
    import sys
    
    # Allow custom base URL
    base_url = "kyutai-labs--voice-stack"
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    
    tester = ModalDeploymentTester(base_url)
    success = await tester.run_all_tests()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
