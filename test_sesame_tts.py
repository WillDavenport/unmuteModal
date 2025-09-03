#!/usr/bin/env python3
"""
Test script for the Sesame_TTS Modal service

This script demonstrates how to use the Sesame_TTS service for text-to-speech generation
using the CSM (Conversational Speech Model) from SesameAI Labs.

Usage:
    python test_sesame_tts.py
"""

import asyncio
import json
import base64
import websockets
from pathlib import Path


async def test_sesame_tts_websocket():
    """Test the Sesame TTS service via WebSocket"""
    
    # Note: Replace with your actual Modal deployment URL
    # Pattern: https://username--appname-classname-web.modal.run
    websocket_url = "wss://your-username--voice-stack-sesamettservice-web.modal.run/ws"
    
    print(f"Connecting to Sesame TTS WebSocket: {websocket_url}")
    
    try:
        async with websockets.connect(websocket_url) as websocket:
            print("Connected to Sesame TTS service")
            
            # Test message
            test_request = {
                "text": "Hello from Sesame TTS! This is a test of the conversational speech model.",
                "speaker": 0,
                "max_audio_length_ms": 10000,
                "temperature": 0.9,
                "topk": 50
            }
            
            # Send request
            await websocket.send(json.dumps(test_request))
            print(f"Sent TTS request: {test_request['text']}")
            
            # Receive audio response
            audio_bytes = await websocket.recv()
            print(f"Received audio: {len(audio_bytes)} bytes")
            
            # Save audio to file
            output_file = Path("sesame_tts_output.wav")
            with open(output_file, "wb") as f:
                f.write(audio_bytes)
            
            print(f"Audio saved to: {output_file}")
            
    except Exception as e:
        print(f"WebSocket test failed: {e}")


async def test_sesame_tts_http():
    """Test the Sesame TTS service via HTTP endpoint"""
    import aiohttp
    
    # Note: Replace with your actual Modal deployment URL
    http_url = "https://your-username--voice-stack-sesamettservice-web.modal.run/generate"
    
    test_request = {
        "text": "This is a test of the HTTP endpoint for Sesame TTS.",
        "speaker": 1,
        "max_audio_length_ms": 8000,
        "temperature": 0.8,
        "topk": 40
    }
    
    print(f"Testing HTTP endpoint: {http_url}")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(http_url, json=test_request) as response:
                if response.status == 200:
                    result = await response.json()
                    
                    # Decode base64 audio
                    audio_bytes = base64.b64decode(result['audio'])
                    
                    # Save audio to file
                    output_file = Path("sesame_tts_http_output.wav")
                    with open(output_file, "wb") as f:
                        f.write(audio_bytes)
                    
                    print(f"HTTP test successful! Audio saved to: {output_file}")
                    print(f"Sample rate: {result['sample_rate']}")
                else:
                    print(f"HTTP test failed with status: {response.status}")
                    print(await response.text())
                    
    except Exception as e:
        print(f"HTTP test failed: {e}")


def test_local_modal_function():
    """Test the Sesame TTS service using Modal function calls (if running locally)"""
    try:
        # This would work if you have modal installed locally and authenticated
        import modal
        
        # Get the app
        app = modal.App.lookup("voice-stack", create_if_missing=False)
        
        # Get the Sesame TTS service
        sesame_tts = app.cls.SesameTTSService()
        
        # Generate speech
        audio_bytes = sesame_tts.generate_speech.remote(
            text="Testing local Modal function call for Sesame TTS",
            speaker=0,
            max_audio_length_ms=10000
        )
        
        # Save audio
        output_file = Path("sesame_tts_local_output.wav")
        with open(output_file, "wb") as f:
            f.write(audio_bytes)
        
        print(f"Local Modal function test successful! Audio saved to: {output_file}")
        
    except Exception as e:
        print(f"Local Modal function test failed: {e}")
        print("Note: This requires modal to be installed and authenticated locally")


async def main():
    """Run all tests"""
    print("=== Sesame TTS Service Tests ===\n")
    
    print("1. Testing WebSocket endpoint...")
    await test_sesame_tts_websocket()
    print()
    
    print("2. Testing HTTP endpoint...")
    await test_sesame_tts_http()
    print()
    
    print("3. Testing local Modal function...")
    test_local_modal_function()
    print()
    
    print("=== Tests completed ===")
    print("\nNote: Update the URLs in this script with your actual Modal deployment URLs")
    print("The URLs follow the pattern: https://username--appname-classname-web.modal.run")


if __name__ == "__main__":
    asyncio.run(main())