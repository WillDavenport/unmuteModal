#!/usr/bin/env python3
"""
Test script for Modal Audio Debugging System

This demonstrates how the Modal-compatible debugging system works.
"""

import json
import time
from modal_audio_debug import (
    AudioDebugSession, 
    log_orpheus_event, 
    log_backend_event,
    log_conversation_event,
    log_websocket_event,
    log_frontend_event,
    audio_debug
)

def simulate_perfect_audio_flow():
    """Simulate a perfect audio flow through all stages."""
    print("🎵 Simulating perfect audio flow...")
    
    with AudioDebugSession("test_perfect_flow") as session:
        # Orpheus generation
        log_orpheus_event("generation_start", {
            "text_length": 50,
            "text_preview": "Hello, this is a test message",
            "voice": "tara"
        })
        
        for i in range(1, 4):  # Generate 3 chunks
            log_orpheus_event("chunk_generated", {
                "chunk_number": i,
                "chunk_bytes": 4800,
                "chunk_samples": 2400,
                "total_bytes": i * 4800,
                "total_samples": i * 2400
            })
            time.sleep(0.1)  # Simulate processing time
        
        log_orpheus_event("generation_complete", {
            "total_chunks": 3,
            "total_bytes": 14400,
            "total_samples": 7200,
            "audio_duration": 0.3,
            "generation_time": 0.5,
            "rtf": 0.6
        })
        
        # Backend processing
        log_backend_event("stream_start", {
            "text_length": 50,
            "text_preview": "Hello, this is a test message"
        })
        
        for i in range(1, 4):  # Process 3 chunks
            log_backend_event("chunk_received", {
                "chunk_number": i,
                "chunk_bytes": 4800
            })
            
            log_backend_event("chunk_queued", {
                "chunk_number": i,
                "chunk_samples": 2400,
                "queue_size": 1,
                "total_samples_queued": i * 2400
            })
            
            log_backend_event("message_yielded", {
                "message_number": i,
                "message_samples": 2400,
                "queue_size": 0,
                "total_samples_yielded": i * 2400
            })
            time.sleep(0.05)
        
        log_backend_event("modal_complete", {"total_chunks": 3})
        
        # Conversation processing
        for i in range(1, 4):
            log_conversation_event("message_received", {
                "message_number": i,
                "message_type": "TTSAudioMessage"
            })
            
            log_conversation_event("audio_processing", {
                "message_samples": 2400,
                "total_samples_processed": i * 2400
            })
            
            log_conversation_event("to_output_queue", {
                "audio_samples": 2400,
                "total_samples_to_output": i * 2400
            })
            time.sleep(0.03)
        
        log_conversation_event("loop_complete", {
            "total_messages": 3,
            "total_samples_processed": 7200
        })
        
        # WebSocket processing
        for i in range(1, 4):
            log_websocket_event("audio_received", {
                "audio_samples": 2400
            })
            
            log_websocket_event("opus_encoded", {
                "opus_bytes": 120,
                "b64_size": 160
            })
            
            log_websocket_event("message_sent", {
                "json_size": 200,
                "b64_delta_size": 160
            })
            time.sleep(0.02)
        
        # Frontend processing
        for i in range(1, 4):
            log_frontend_event("message_received", {
                "opus_bytes": 120,
                "message_timestamp": time.time()
            })
            
            log_frontend_event("sent_to_decoder", {
                "chunk_number": i
            })
            
            log_frontend_event("decoder_frames", {
                "frame_length": 960
            })
            
            log_frontend_event("worklet_frames", {
                "frame_number": i
            })
            time.sleep(0.01)

def simulate_audio_flow_with_losses():
    """Simulate an audio flow with losses at various stages."""
    print("⚠️  Simulating audio flow with losses...")
    
    with AudioDebugSession("test_flow_with_losses") as session:
        # Orpheus generates 4 chunks
        log_orpheus_event("generation_start", {
            "text_length": 80,
            "text_preview": "This test shows audio loss in the pipeline",
            "voice": "tara"
        })
        
        for i in range(1, 5):  # Generate 4 chunks
            log_orpheus_event("chunk_generated", {
                "chunk_number": i,
                "chunk_bytes": 4800,
                "chunk_samples": 2400
            })
        
        log_orpheus_event("generation_complete", {
            "total_chunks": 4,
            "total_bytes": 19200,
            "total_samples": 9600
        })
        
        # Backend only receives 3 chunks (1 lost)
        log_backend_event("stream_start", {"text_length": 80})
        
        for i in range(1, 4):  # Only 3 chunks received
            log_backend_event("chunk_received", {
                "chunk_number": i,
                "chunk_bytes": 4800
            })
            
            log_backend_event("chunk_queued", {
                "chunk_number": i,
                "chunk_samples": 2400
            })
            
            log_backend_event("message_yielded", {
                "message_number": i,
                "message_samples": 2400
            })
        
        # Conversation only receives 2 messages (1 lost)
        for i in range(1, 3):  # Only 2 messages
            log_conversation_event("message_received", {
                "message_number": i
            })
            
            log_conversation_event("to_output_queue", {
                "audio_samples": 2400
            })
        
        # WebSocket receives 2 but only encodes 1 (Opus buffer issue)
        for i in range(1, 3):
            log_websocket_event("audio_received", {
                "audio_samples": 2400
            })
            
            if i == 1:  # Only first one encodes successfully
                log_websocket_event("opus_encoded", {"opus_bytes": 120})
                log_websocket_event("message_sent", {"json_size": 200})
            else:
                log_websocket_event("no_opus_output", {
                    "input_samples": 2400,
                    "reason": "buffering"
                })
        
        # Frontend receives only 1 message
        log_frontend_event("message_received", {
            "opus_bytes": 120
        })
        
        log_frontend_event("sent_to_decoder", {"chunk_number": 1})
        log_frontend_event("decoder_frames", {"frame_length": 960})
        
        # Simulate an error
        log_backend_event("stream_error", {
            "error": "Connection timeout",
            "error_type": "TimeoutError"
        })

def print_debug_stats():
    """Print current debug statistics."""
    stats = audio_debug.get_stats()
    
    print("\n" + "="*60)
    print("MODAL AUDIO DEBUG STATISTICS")
    print("="*60)
    
    print(f"Total events logged: {stats['total_events']}")
    print(f"Recent events: {len(stats['recent_logs'])}")
    
    # Calculate pipeline efficiency
    orpheus_chunks = stats["stats"].get("orpheus_chunk_generated", 0)
    frontend_messages = stats["stats"].get("frontend_message_received", 0)
    
    if orpheus_chunks > 0:
        efficiency = (frontend_messages / orpheus_chunks) * 100
        print(f"Pipeline efficiency: {efficiency:.1f}%")
        
        if efficiency < 50:
            print("🚨 CRITICAL: Severe audio loss detected!")
        elif efficiency < 90:
            print("⚠️  WARNING: Significant audio loss detected")
        else:
            print("✅ Pipeline working efficiently")
    
    print("\nStage Statistics:")
    print("-" * 20)
    
    stages = {
        "Orpheus": ["orpheus_generation_start", "orpheus_chunk_generated", "orpheus_generation_complete"],
        "Backend": ["backend_chunk_received", "backend_chunk_queued", "backend_message_yielded"],
        "Conversation": ["conversation_message_received", "conversation_to_output_queue"],
        "WebSocket": ["websocket_audio_received", "websocket_opus_encoded", "websocket_message_sent"],
        "Frontend": ["frontend_message_received", "frontend_sent_to_decoder", "frontend_decoder_frames"]
    }
    
    for stage_name, events in stages.items():
        print(f"\n{stage_name}:")
        for event in events:
            count = stats["stats"].get(event, 0)
            print(f"  {event.replace('_', ' ').title()}: {count}")
    
    # Show recent events
    print(f"\nRecent Events (last 10):")
    print("-" * 30)
    for event in stats["recent_logs"][-10:]:
        timestamp = event["timestamp"][-12:-4]  # Extract time part
        stage = event["stage"].upper()
        event_name = event["event"].replace("_", " ").title()
        print(f"{timestamp} [{stage}] {event_name}")
    
    print("\n" + "="*60)

def main():
    """Run the Modal debugging test."""
    print("🎯 Testing Modal Audio Debugging System")
    print("=" * 50)
    
    # Reset stats
    audio_debug.reset_stats()
    print("Reset debug statistics")
    
    # Test perfect flow
    simulate_perfect_audio_flow()
    print("\n✅ Perfect flow simulation completed")
    print_debug_stats()
    
    # Reset and test with losses
    print("\n" + "="*50)
    audio_debug.reset_stats()
    
    simulate_audio_flow_with_losses()
    print("\n⚠️  Flow with losses simulation completed")
    print_debug_stats()
    
    print("\n🎯 Modal debugging test completed!")
    print("\nTo use in Modal:")
    print("1. Deploy your app with the debug endpoints")
    print("2. Access https://your-app-url/audio_debug_stats")
    print("3. Monitor logs in Modal dashboard")

if __name__ == "__main__":
    main()