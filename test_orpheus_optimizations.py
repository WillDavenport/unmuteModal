#!/usr/bin/env python3
"""
Test script to verify Orpheus TTS optimizations are working correctly.
Run this to test the performance improvements.
"""

import asyncio
import time
import logging
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from unmute.tts.orpheus_tts import OrpheusModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def test_orpheus_performance():
    """Test the optimized Orpheus TTS performance"""
    print("🚀 Testing Optimized Orpheus TTS Performance")
    print("=" * 60)
    
    # Initialize the model
    print("📦 Loading Orpheus model with H100 optimizations...")
    model = OrpheusModel()
    
    try:
        model.load()
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return
    
    # Test texts of varying lengths
    test_cases = [
        ("Short", "Hello world!"),
        ("Medium", "This is a medium length sentence to test the Orpheus TTS performance with optimizations."),
        ("Long", "This is a much longer text that should demonstrate the streaming capabilities and real-time performance of the optimized Orpheus TTS system running on NVIDIA H100 GPU with all the latest optimizations including torch.compile, mixed precision, and efficient streaming generation."),
    ]
    
    print("\n🧪 Running Performance Tests:")
    print("-" * 60)
    
    for test_name, text in test_cases:
        print(f"\n📝 Testing {test_name} text ({len(text)} chars): '{text[:50]}{'...' if len(text) > 50 else ''}'")
        
        start_time = time.time()
        first_chunk_time = None
        chunk_count = 0
        total_bytes = 0
        
        try:
            async for audio_chunk in model.generate_speech_stream(text, voice="tara"):
                if first_chunk_time is None:
                    first_chunk_time = time.time() - start_time
                    print(f"⚡ Time to First Byte (TTFB): {first_chunk_time:.3f}s")
                
                chunk_count += 1
                total_bytes += len(audio_chunk)
                
                # Print progress for longer generations
                if chunk_count % 10 == 0:
                    elapsed = time.time() - start_time
                    print(f"   📊 Chunk {chunk_count}: {total_bytes:,} bytes in {elapsed:.2f}s")
            
            total_time = time.time() - start_time
            
            # Calculate performance metrics
            audio_duration = total_bytes / (2 * 24000)  # 24kHz, 16-bit samples
            rtf = total_time / audio_duration if audio_duration > 0 else float('inf')
            
            # Results
            print(f"📈 Results for {test_name} text:")
            print(f"   ⏱️  Total time: {total_time:.3f}s")
            print(f"   🎵 Audio duration: {audio_duration:.3f}s")
            print(f"   🚀 Real-time factor: {rtf:.2f}x")
            print(f"   📦 Total chunks: {chunk_count}")
            print(f"   💾 Total bytes: {total_bytes:,}")
            
            # Performance assessment
            if first_chunk_time and first_chunk_time < 0.15:
                ttfb_status = "🟢 EXCELLENT"
            elif first_chunk_time and first_chunk_time < 1.0:
                ttfb_status = "🟡 GOOD"
            elif first_chunk_time and first_chunk_time < 2.0:
                ttfb_status = "🟠 ACCEPTABLE"
            else:
                ttfb_status = "🔴 NEEDS IMPROVEMENT"
            
            if rtf < 1.0:
                rtf_status = "🟢 EXCELLENT (faster than real-time)"
            elif rtf < 2.0:
                rtf_status = "🟡 GOOD"
            elif rtf < 5.0:
                rtf_status = "🟠 ACCEPTABLE"
            else:
                rtf_status = "🔴 NEEDS IMPROVEMENT"
            
            print(f"   📊 TTFB Assessment: {ttfb_status}")
            print(f"   📊 RTF Assessment: {rtf_status}")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
        
        print("-" * 60)
    
    print("\n✅ Performance testing completed!")
    
    # Run the built-in benchmark
    print("\n🔬 Running comprehensive benchmark...")
    try:
        await model.benchmark_performance()
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_orpheus_performance())