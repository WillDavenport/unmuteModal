#!/usr/bin/env python3
"""
Debug script to test the audio flow from Orpheus TTS through to frontend emission.

This script helps identify where audio cutoffs are occurring by testing each stage:
1. Orpheus Modal TTS generation 
2. Backend audio queue processing
3. Conversation output queue handling
4. WebSocket emission to frontend

Usage:
    python debug_audio_flow.py --test-text "This is a test message to debug audio cutoffs"
    python debug_audio_flow.py --test-orpheus-only  # Test just Orpheus generation
    python debug_audio_flow.py --test-full-pipeline  # Test end-to-end pipeline
"""

import asyncio
import argparse
import logging
import time
import json
import base64
from typing import AsyncIterator
import numpy as np

# Configure logging for maximum visibility
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the audio pipeline components
from unmute.tts.text_to_speech import OrpheusTextToSpeech, TTSAudioMessage
from unmute.tts.orpheus_modal import OrpheusTTS
from unmute.conversation import Conversation
from unmute.kyutai_constants import SAMPLE_RATE
import unmute.openai_realtime_api_events as ora

class AudioFlowDebugger:
    """Debug tool for tracing audio flow through the TTS pipeline."""
    
    def __init__(self):
        self.test_results = {}
        
    async def test_orpheus_modal_direct(self, text: str) -> dict:
        """Test Orpheus Modal TTS generation directly."""
        logger.info("=== TESTING ORPHEUS MODAL DIRECT ===")
        
        try:
            # Create Orpheus TTS instance
            orpheus_service = OrpheusTTS()
            
            # Test streaming generation
            chunks_received = 0
            total_bytes = 0
            start_time = time.monotonic()
            first_chunk_time = None
            
            logger.info(f"Starting direct Orpheus Modal test with text: '{text[:50]}...'")
            
            for audio_chunk in orpheus_service.generate_speech_stream.remote_gen(
                text=text,
                voice="tara"
            ):
                chunks_received += 1
                chunk_size = len(audio_chunk)
                total_bytes += chunk_size
                
                if first_chunk_time is None:
                    first_chunk_time = time.monotonic()
                    ttft = first_chunk_time - start_time
                    logger.info(f"Time to first chunk: {ttft:.3f}s")
                
                logger.info(f"Received chunk {chunks_received}: {chunk_size} bytes")
                
                # Verify chunk is valid PCM data
                if chunk_size % 2 != 0:
                    logger.warning(f"Chunk {chunks_received} has odd byte count: {chunk_size}")
                    
            end_time = time.monotonic()
            total_time = end_time - start_time
            
            # Calculate audio duration
            duration_seconds = total_bytes / (2 * 24000) if total_bytes > 0 else 0
            rtf = total_time / duration_seconds if duration_seconds > 0 else 0
            
            result = {
                "success": True,
                "chunks_received": chunks_received,
                "total_bytes": total_bytes,
                "total_time": total_time,
                "audio_duration": duration_seconds,
                "real_time_factor": rtf,
                "ttft": first_chunk_time - start_time if first_chunk_time else None,
            }
            
            logger.info(f"Orpheus Modal test completed: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Orpheus Modal test failed: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}

    async def test_orpheus_tts_adapter(self, text: str) -> dict:
        """Test OrpheusTextToSpeech adapter (backend integration)."""
        logger.info("=== TESTING ORPHEUS TTS ADAPTER ===")
        
        try:
            # Create OrpheusTextToSpeech instance
            orpheus_tts = OrpheusTextToSpeech()
            await orpheus_tts.start_up()
            
            # Send text for generation
            await orpheus_tts.send_complete_text(text)
            
            # Collect messages from the adapter
            messages_received = 0
            total_samples = 0
            start_time = time.monotonic()
            first_message_time = None
            
            logger.info(f"Starting OrpheusTextToSpeech adapter test with text: '{text[:50]}...'")
            
            async for message in orpheus_tts:
                messages_received += 1
                
                if first_message_time is None:
                    first_message_time = time.monotonic()
                    ttft = first_message_time - start_time
                    logger.info(f"Time to first message: {ttft:.3f}s")
                
                if isinstance(message, TTSAudioMessage):
                    samples = len(message.pcm)
                    total_samples += samples
                    logger.info(f"Received TTSAudioMessage {messages_received}: {samples} samples")
                else:
                    logger.info(f"Received message {messages_received}: {type(message).__name__}")
                    
                # Break after reasonable timeout to avoid hanging
                if time.monotonic() - start_time > 30:
                    logger.warning("Test timeout reached, breaking")
                    break
                    
            end_time = time.monotonic()
            total_time = end_time - start_time
            
            # Calculate audio duration
            duration_seconds = total_samples / SAMPLE_RATE if total_samples > 0 else 0
            
            result = {
                "success": True,
                "messages_received": messages_received,
                "total_samples": total_samples,
                "total_time": total_time,
                "audio_duration": duration_seconds,
                "ttft": first_message_time - start_time if first_message_time else None,
            }
            
            logger.info(f"OrpheusTextToSpeech adapter test completed: {result}")
            
            # Clean up
            await orpheus_tts.shutdown(full_shutdown=True)
            
            return result
            
        except Exception as e:
            logger.error(f"OrpheusTextToSpeech adapter test failed: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}

    async def test_conversation_pipeline(self, text: str) -> dict:
        """Test the full conversation pipeline (without WebSocket)."""
        logger.info("=== TESTING CONVERSATION PIPELINE ===")
        
        try:
            # Create conversation instance
            conversation = Conversation()
            await conversation.start()
            
            # Simulate LLM response generation
            logger.info(f"Starting conversation pipeline test with text: '{text[:50]}...'")
            
            # Initialize TTS for this test
            await conversation._init_tts(generating_message_i=1)
            
            # Send text to TTS
            if conversation.tts:
                await conversation.tts.send_complete_text(text)
            
            # Collect outputs from conversation
            outputs_received = 0
            audio_deltas_received = 0
            total_samples = 0
            start_time = time.monotonic()
            first_audio_time = None
            
            # Monitor conversation outputs
            timeout_start = time.monotonic()
            while time.monotonic() - timeout_start < 30:  # 30 second timeout
                try:
                    output = await asyncio.wait_for(conversation.get_output(), timeout=1.0)
                    if output is None:
                        continue
                        
                    outputs_received += 1
                    
                    # Check if it's an audio tuple
                    if isinstance(output, tuple) and len(output) == 2:
                        sample_rate, audio_data = output
                        if isinstance(audio_data, np.ndarray):
                            samples = len(audio_data)
                            total_samples += samples
                            audio_deltas_received += 1
                            
                            if first_audio_time is None:
                                first_audio_time = time.monotonic()
                                ttft = first_audio_time - start_time
                                logger.info(f"Time to first audio output: {ttft:.3f}s")
                            
                            logger.info(f"Received audio output {audio_deltas_received}: {samples} samples at {sample_rate}Hz")
                    
                    elif isinstance(output, ora.ResponseAudioDelta):
                        audio_deltas_received += 1
                        logger.info(f"Received ResponseAudioDelta {audio_deltas_received}: {len(output.delta)} base64 chars")
                    
                    else:
                        logger.info(f"Received output {outputs_received}: {type(output).__name__}")
                        
                except asyncio.TimeoutError:
                    # Check if TTS is still active
                    if conversation.tts_task and not conversation.tts_task.done():
                        continue
                    else:
                        logger.info("No more outputs and TTS task completed")
                        break
                        
            end_time = time.monotonic()
            total_time = end_time - start_time
            
            # Calculate audio duration
            duration_seconds = total_samples / SAMPLE_RATE if total_samples > 0 else 0
            
            result = {
                "success": True,
                "outputs_received": outputs_received,
                "audio_deltas_received": audio_deltas_received,
                "total_samples": total_samples,
                "total_time": total_time,
                "audio_duration": duration_seconds,
                "ttft": first_audio_time - start_time if first_audio_time else None,
            }
            
            logger.info(f"Conversation pipeline test completed: {result}")
            
            # Clean up
            await conversation.cleanup()
            
            return result
            
        except Exception as e:
            logger.error(f"Conversation pipeline test failed: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}

    async def test_websocket_emission_simulation(self, text: str) -> dict:
        """Simulate WebSocket emission by processing audio through the emission pipeline."""
        logger.info("=== TESTING WEBSOCKET EMISSION SIMULATION ===")
        
        try:
            # Import WebSocket emission components
            import sphn
            from fastrtc import audio_to_float32
            
            # Create Opus encoder (similar to main_websocket.py)
            opus_writer = sphn.OpusStreamWriter(SAMPLE_RATE)
            
            # Generate some test audio data
            logger.info("Generating test audio data for WebSocket emission test")
            test_duration = 3.0  # 3 seconds of test audio
            test_samples = int(SAMPLE_RATE * test_duration)
            
            # Generate a simple sine wave for testing
            frequency = 440  # A4 note
            t = np.linspace(0, test_duration, test_samples, dtype=np.float32)
            test_audio = 0.1 * np.sin(2 * np.pi * frequency * t).astype(np.float32)
            
            # Process audio through WebSocket emission pipeline
            emissions_created = 0
            total_opus_bytes = 0
            total_base64_chars = 0
            start_time = time.monotonic()
            
            # Split audio into chunks (similar to how it comes from TTS)
            chunk_size = 1024  # samples per chunk
            for i in range(0, len(test_audio), chunk_size):
                chunk = test_audio[i:i + chunk_size]
                
                # Convert to float32 (as done in main_websocket.py)
                audio = audio_to_float32(chunk)
                
                # Encode to Opus
                opus_bytes = opus_writer.append_pcm(audio)
                
                if opus_bytes:
                    emissions_created += 1
                    opus_byte_count = len(opus_bytes)
                    total_opus_bytes += opus_byte_count
                    
                    # Create ResponseAudioDelta (as done in main_websocket.py)
                    response_audio_delta = ora.ResponseAudioDelta(
                        delta=base64.b64encode(opus_bytes).decode("utf-8")
                    )
                    
                    base64_chars = len(response_audio_delta.delta)
                    total_base64_chars += base64_chars
                    
                    logger.info(f"Created ResponseAudioDelta {emissions_created}: {opus_byte_count} opus bytes -> {base64_chars} base64 chars")
                    
                    # Simulate JSON serialization (as done in WebSocket send)
                    json_message = response_audio_delta.model_dump_json()
                    logger.debug(f"JSON message length: {len(json_message)} chars")
                    
            end_time = time.monotonic()
            total_time = end_time - start_time
            
            result = {
                "success": True,
                "emissions_created": emissions_created,
                "total_opus_bytes": total_opus_bytes,
                "total_base64_chars": total_base64_chars,
                "total_time": total_time,
                "test_audio_duration": test_duration,
                "test_audio_samples": test_samples,
            }
            
            logger.info(f"WebSocket emission simulation completed: {result}")
            return result
            
        except Exception as e:
            logger.error(f"WebSocket emission simulation failed: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}

    async def run_full_test_suite(self, test_text: str):
        """Run all audio flow tests."""
        logger.info("=== STARTING FULL AUDIO FLOW DEBUG TEST SUITE ===")
        logger.info(f"Test text: '{test_text}'")
        
        # Test 1: Orpheus Modal Direct
        self.test_results["orpheus_modal_direct"] = await self.test_orpheus_modal_direct(test_text)
        
        # Test 2: OrpheusTextToSpeech Adapter
        self.test_results["orpheus_tts_adapter"] = await self.test_orpheus_tts_adapter(test_text)
        
        # Test 3: Conversation Pipeline
        self.test_results["conversation_pipeline"] = await self.test_conversation_pipeline(test_text)
        
        # Test 4: WebSocket Emission Simulation
        self.test_results["websocket_emission"] = await self.test_websocket_emission_simulation(test_text)
        
        # Print summary
        self.print_test_summary()

    def print_test_summary(self):
        """Print a summary of all test results."""
        logger.info("=== AUDIO FLOW DEBUG TEST SUMMARY ===")
        
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result.get("success", False) else "❌ FAIL"
            logger.info(f"{test_name}: {status}")
            
            if result.get("success", False):
                # Print key metrics
                if "chunks_received" in result:
                    logger.info(f"  Chunks/Messages: {result['chunks_received']}")
                if "messages_received" in result:
                    logger.info(f"  Messages: {result['messages_received']}")
                if "audio_deltas_received" in result:
                    logger.info(f"  Audio Deltas: {result['audio_deltas_received']}")
                if "emissions_created" in result:
                    logger.info(f"  Emissions: {result['emissions_created']}")
                    
                if "total_time" in result:
                    logger.info(f"  Total Time: {result['total_time']:.3f}s")
                if "ttft" in result and result["ttft"]:
                    logger.info(f"  Time to First Token: {result['ttft']:.3f}s")
                if "audio_duration" in result:
                    logger.info(f"  Audio Duration: {result['audio_duration']:.3f}s")
                if "real_time_factor" in result:
                    logger.info(f"  Real-time Factor: {result['real_time_factor']:.3f}x")
            else:
                logger.error(f"  Error: {result.get('error', 'Unknown error')}")
            
            logger.info("")

async def main():
    parser = argparse.ArgumentParser(description="Debug audio flow from TTS to frontend")
    parser.add_argument("--test-text", default="This is a test message to debug audio cutoffs in the TTS pipeline.", help="Text to use for testing")
    parser.add_argument("--test-orpheus-only", action="store_true", help="Test only Orpheus Modal generation")
    parser.add_argument("--test-adapter-only", action="store_true", help="Test only OrpheusTextToSpeech adapter")
    parser.add_argument("--test-conversation-only", action="store_true", help="Test only conversation pipeline")
    parser.add_argument("--test-websocket-only", action="store_true", help="Test only WebSocket emission simulation")
    parser.add_argument("--test-full-pipeline", action="store_true", help="Test the complete pipeline")
    
    args = parser.parse_args()
    
    debugger = AudioFlowDebugger()
    
    if args.test_orpheus_only:
        result = await debugger.test_orpheus_modal_direct(args.test_text)
        logger.info(f"Orpheus Modal test result: {result}")
    elif args.test_adapter_only:
        result = await debugger.test_orpheus_tts_adapter(args.test_text)
        logger.info(f"OrpheusTextToSpeech adapter test result: {result}")
    elif args.test_conversation_only:
        result = await debugger.test_conversation_pipeline(args.test_text)
        logger.info(f"Conversation pipeline test result: {result}")
    elif args.test_websocket_only:
        result = await debugger.test_websocket_emission_simulation(args.test_text)
        logger.info(f"WebSocket emission test result: {result}")
    else:
        # Run full test suite by default
        await debugger.run_full_test_suite(args.test_text)

if __name__ == "__main__":
    asyncio.run(main())