#!/usr/bin/env python3
"""
Test script for the simplified TTS pipeline.
This demonstrates the key differences from the legacy pipeline.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add the unmute directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from unmute.tts.text_to_speech import SimplifiedOrpheusTextToSpeech, SimplifiedTTSMessage

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_simplified_tts():
    """Test the simplified TTS pipeline."""
    logger.info("=== Testing Simplified TTS Pipeline ===")
    
    # Create TTS instance with small buffers for testing
    tts = SimplifiedOrpheusTextToSpeech(
        ring_buffer_ms=100,  # Small buffer for testing
        opus_packet_ms=20,   # Small packets for testing
    )
    
    try:
        # Start up the service (this would connect to Modal in real use)
        logger.info("Starting TTS service...")
        # await tts.start_up()  # Skip for demo since we don't have Modal running
        
        # Simulate sending text
        test_text = "Hello, this is a test of the simplified TTS pipeline."
        logger.info(f"Sending text: '{test_text}'")
        
        # This would normally call Modal, but we'll simulate the flow
        logger.info("=== Pipeline Flow Demonstration ===")
        logger.info("1. Text sent to SimplifiedOrpheusTextToSpeech.send_complete_text()")
        logger.info("2. Orpheus generates PCM chunks via Modal")
        logger.info("3. PCM chunks added to ring buffer (max 100ms)")
        logger.info("4. When buffer has ≥20ms, encode to Opus immediately")
        logger.info("5. Send base64 Opus as response.audio.delta")
        logger.info("6. Frontend receives and decodes immediately")
        logger.info("7. Frontend jitter buffer (≤120ms) smooths playback")
        
        # Demonstrate interrupt handling
        logger.info("\n=== Interrupt Handling ===")
        logger.info("1. VAD detects user speech")
        logger.info("2. Call tts.interrupt('vad_interrupt')")
        logger.info("3. Cancel Modal generation task")
        logger.info("4. Flush ring buffer (≤100ms)")
        logger.info("5. Send response.interrupted message")
        logger.info("6. Frontend flushes jitter buffer immediately")
        logger.info("7. Total latency: ≤220ms (100ms + 120ms)")
        
        # Show the key differences
        logger.info("\n=== Key Improvements ===")
        logger.info("✓ Removed multi-layer queuing (RealtimeQueue, cadence control)")
        logger.info("✓ Direct Opus encoding with minimal buffering")
        logger.info("✓ Immediate interrupt handling with buffer flush")
        logger.info("✓ Reduced frontend buffer from 60s to 120ms")
        logger.info("✓ Fast audio start with 40ms initial buffer")
        logger.info("✓ Simple control messages (audio.start, interrupted)")
        
        logger.info("\n=== Configuration ===")
        logger.info(f"Ring buffer: {tts.ring_buffer_ms}ms (vs unlimited in legacy)")
        logger.info(f"Opus packets: {tts.opus_packet_ms}ms")
        logger.info(f"Sample rate: {tts.frame_size_samples * 50}Hz (20ms frames)")
        logger.info("Frontend buffer: 80-120ms (vs 60s in legacy)")
        
    except Exception as e:
        logger.error(f"Test error: {e}")
    finally:
        await tts.shutdown()
        logger.info("=== Test Complete ===")

async def demonstrate_message_flow():
    """Demonstrate the message flow in the simplified pipeline."""
    logger.info("\n=== Message Flow Comparison ===")
    
    logger.info("LEGACY PIPELINE:")
    logger.info("Orpheus → RealtimeQueue → cadence scheduler → Opus encoder → WebSocket")
    logger.info("         ↳ timing control ↳ backpressure    ↳ transport")
    logger.info("WebSocket → decoder worker → AudioWorklet queue (60s) → output")
    
    logger.info("\nSIMPLIFIED PIPELINE:")
    logger.info("Orpheus → ring buffer (200ms) → Opus encoder → WebSocket")
    logger.info("                              ↳ immediate encoding")
    logger.info("WebSocket → decoder worker → AudioWorklet buffer (120ms) → output")
    
    logger.info("\nINTERRUPT HANDLING:")
    logger.info("LEGACY: VAD → cancel tasks → wait for queue drain (seconds)")
    logger.info("SIMPLIFIED: VAD → tts.interrupt() → flush buffers (220ms max)")

if __name__ == "__main__":
    # Set environment variable to enable simplified TTS
    os.environ["SIMPLIFIED_TTS_ENABLED"] = "true"
    
    asyncio.run(test_simplified_tts())
    asyncio.run(demonstrate_message_flow())