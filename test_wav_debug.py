#!/usr/bin/env python3
"""
Test script to demonstrate WAV debug functionality.
"""

import os
import numpy as np
import asyncio
from unmute.wav_debug import write_debug_wav, write_debug_wav_async

def test_wav_debug():
    """Test WAV debug file creation."""
    
    # Enable debug mode for testing
    os.environ["DEBUG_WAV_ENABLED"] = "true"
    os.environ["DEBUG_WAV_DIR"] = "/tmp/wav_debug_test"
    
    # Create test audio data (440 Hz sine wave for 1 second at 24000 Hz)
    sample_rate = 24000
    duration = 1.0
    frequency = 440.0
    
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio_data = np.sin(2 * np.pi * frequency * t).astype(np.float32) * 0.5
    
    print(f"Generated {len(audio_data)} samples of {frequency}Hz sine wave")
    
    # Test synchronous WAV creation
    print("Testing synchronous WAV creation...")
    filepath = write_debug_wav(
        audio_data, sample_rate, "test_sync", "test_conversation"
    )
    if filepath:
        print(f"Created sync WAV file: {filepath}")
    else:
        print("Sync WAV creation disabled (DEBUG_WAV_ENABLED=false)")
    
    return audio_data, sample_rate

async def test_wav_debug_async():
    """Test async WAV debug file creation."""
    
    audio_data, sample_rate = test_wav_debug()
    
    # Test asynchronous WAV creation
    print("Testing asynchronous WAV creation...")
    filepath = await write_debug_wav_async(
        audio_data, sample_rate, "test_async", "test_conversation"
    )
    if filepath:
        print(f"Created async WAV file: {filepath}")
    else:
        print("Async WAV creation disabled (DEBUG_WAV_ENABLED=false)")

if __name__ == "__main__":
    print("WAV Debug Test Script")
    print("====================")
    
    # Test with debug disabled
    print("\n1. Testing with DEBUG_WAV_ENABLED=false:")
    os.environ["DEBUG_WAV_ENABLED"] = "false"
    test_wav_debug()
    
    # Test with debug enabled
    print("\n2. Testing with DEBUG_WAV_ENABLED=true:")
    os.environ["DEBUG_WAV_ENABLED"] = "true"
    asyncio.run(test_wav_debug_async())
    
    print("\nTest completed!")
    print("Check /tmp/wav_debug_test/ for generated WAV files")
