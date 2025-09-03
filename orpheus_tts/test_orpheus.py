#!/usr/bin/env python3
"""
Standalone test script for Orpheus TTS Modal integration.

This script can be used to verify that the Orpheus TTS service
works correctly without running the full Modal application.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add the orpheus_tts directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from orpheus_modal import OrpheusTTSModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_orpheus_local():
    """Test the Orpheus TTS model locally."""
    print("=== Testing Orpheus TTS Model Locally ===")
    
    try:
        # Initialize the model
        print("Initializing Orpheus TTS model...")
        model = OrpheusTTSModel()
        model.load()
        print("Model loaded successfully!")
        
        # Test text
        test_text = "Hello, this is a test of the Orpheus TTS system."
        print(f"Test text: {test_text}")
        
        # Generate speech
        print("Generating speech...")
        audio_bytes = await model.generate_speech(test_text, voice="tara")
        print(f"Generated {len(audio_bytes)} bytes of audio")
        
        # Save to file
        output_dir = Path("test_output")
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / "orpheus_local_test.wav"
        with open(output_file, "wb") as f:
            f.write(audio_bytes)
        
        print(f"Audio saved to: {output_file}")
        print("Test completed successfully!")
        
        # Test prompt formatting
        print("\n=== Testing Prompt Formatting ===")
        formatted = model.format_prompt("Hello world", "tara")
        print(f"Formatted prompt: {formatted[:100]}...")
        
        # Test token extraction
        test_tokens = "<custom_token_1234><custom_token_5678>"
        extracted = model.split_custom_tokens(test_tokens)
        print(f"Extracted tokens from '{test_tokens}': {extracted}")
        
        print("\nAll tests passed!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


async def test_snac_only():
    """Test just the SNAC model functionality."""
    print("\n=== Testing SNAC Model Only ===")
    
    try:
        from orpheus_modal import SnacModelBatched
        import torch
        
        print("Initializing SNAC model...")
        snac_model = SnacModelBatched()
        print("SNAC model initialized!")
        
        # Create dummy codes for testing
        dummy_codes = [
            torch.randint(1, 4096, (1, 4)).cuda(),
            torch.randint(1, 4096, (1, 8)).cuda(), 
            torch.randint(1, 4096, (1, 16)).cuda(),
        ]
        
        print("Testing SNAC decoding with dummy codes...")
        result = await snac_model.batch_snac_model.acall({"codes": dummy_codes})
        print(f"SNAC result shape: {result.shape if hasattr(result, 'shape') else type(result)}")
        
        print("SNAC test completed!")
        
    except Exception as e:
        print(f"Error testing SNAC: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def main():
    """Main test function."""
    print("Orpheus TTS Integration Test")
    print("=" * 40)
    
    # Check if CUDA is available
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name()}")
    except ImportError:
        print("PyTorch not available")
    
    # Run tests
    success = True
    
    # Test 1: SNAC model only
    if torch.cuda.is_available():
        success &= asyncio.run(test_snac_only())
    else:
        print("Skipping SNAC test (CUDA not available)")
    
    # Test 2: Full model
    success &= asyncio.run(test_orpheus_local())
    
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()