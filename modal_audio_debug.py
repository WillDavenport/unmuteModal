#!/usr/bin/env python3
"""
Modal-compatible Audio Flow Debugging System

This module provides debugging capabilities that work within Modal's environment
by using Modal's built-in logging and creating debug endpoints.
"""

import json
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
import threading

# Make modal import optional for testing
try:
    import modal
    MODAL_AVAILABLE = True
except ImportError:
    MODAL_AVAILABLE = False
    # Create a mock modal for testing
    class MockModal:
        class App:
            def __init__(self, name):
                self.name = name
            def function(self):
                def decorator(func):
                    return func
                return decorator
            def web_endpoint(self, method="GET"):
                def decorator(func):
                    return func
                return decorator
    modal = MockModal()

# Global debug state that persists across Modal function calls
class AudioDebugState:
    def __init__(self):
        self.stats = defaultdict(int)
        self.recent_logs = deque(maxlen=1000)  # Keep last 1000 log entries
        self.session_stats = {}
        self.lock = threading.Lock()
        
    def log_event(self, stage: str, event: str, data: Dict[str, Any] = None):
        """Log an audio flow event."""
        timestamp = datetime.now().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "stage": stage,
            "event": event,
            "data": data or {}
        }
        
        with self.lock:
            self.recent_logs.append(log_entry)
            self.stats[f"{stage}_{event}"] += 1
            
        # Also print to Modal logs for visibility
        print(f"=== AUDIO_DEBUG: {stage.upper()}_{event.upper()} ===")
        if data:
            for key, value in data.items():
                print(f"[AUDIO_DEBUG] {key}: {value}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current debugging statistics."""
        with self.lock:
            recent_logs = list(self.recent_logs)
            return {
                "stats": dict(self.stats),
                "recent_logs": recent_logs[-50:] if recent_logs else [],  # Last 50 events
                "total_events": len(recent_logs)
            }
    
    def reset_stats(self):
        """Reset all statistics."""
        with self.lock:
            self.stats.clear()
            self.recent_logs.clear()
            self.session_stats.clear()

# Global debug instance
audio_debug = AudioDebugState()


def debug_orpheus_generation(func):
    """Decorator to add debugging to Orpheus generation functions."""
    def wrapper(*args, **kwargs):
        text = kwargs.get('text', args[0] if args else 'unknown')
        audio_debug.log_event("orpheus", "generation_start", {
            "text_length": len(text),
            "text_preview": text[:100] + "..." if len(text) > 100 else text
        })
        
        try:
            chunk_count = 0
            total_bytes = 0
            total_samples = 0
            
            # Call the original function
            for chunk in func(*args, **kwargs):
                chunk_count += 1
                chunk_bytes = len(chunk)
                chunk_samples = chunk_bytes // 2  # 16-bit PCM
                total_bytes += chunk_bytes
                total_samples += chunk_samples
                
                audio_debug.log_event("orpheus", "chunk_generated", {
                    "chunk_number": chunk_count,
                    "chunk_bytes": chunk_bytes,
                    "chunk_samples": chunk_samples,
                    "total_bytes": total_bytes,
                    "total_samples": total_samples
                })
                
                yield chunk
            
            audio_debug.log_event("orpheus", "generation_complete", {
                "total_chunks": chunk_count,
                "total_bytes": total_bytes,
                "total_samples": total_samples,
                "audio_duration": total_samples / 24000.0
            })
            
        except Exception as e:
            audio_debug.log_event("orpheus", "generation_error", {
                "error": str(e),
                "error_type": type(e).__name__
            })
            raise
    
    return wrapper


class ModalAudioDebugger:
    """Modal-specific audio debugging utilities."""
    
    @staticmethod
    def create_debug_endpoint(app):
        """Create a debug endpoint for the Modal app."""
        
        if not MODAL_AVAILABLE:
            # Return mock functions for testing
            def mock_debug_stats():
                return audio_debug.get_stats()
            def mock_reset_debug():
                audio_debug.reset_stats()
                return {"status": "reset"}
            return mock_debug_stats, mock_reset_debug
        
        @app.function()
        @modal.web_endpoint(method="GET")
        def audio_debug_stats():
            """Get current audio debugging statistics."""
            stats = audio_debug.get_stats()
            
            # Calculate pipeline efficiency
            orpheus_chunks = stats["stats"].get("orpheus_chunk_generated", 0)
            frontend_messages = stats["stats"].get("frontend_message_received", 0)
            
            efficiency = 0
            if orpheus_chunks > 0:
                efficiency = (frontend_messages / orpheus_chunks) * 100
            
            # Analyze losses
            losses = []
            stages = [
                ("orpheus_chunk_generated", "backend_chunk_queued", "Orpheus → Backend"),
                ("backend_message_yielded", "conversation_message_received", "Backend → Conversation"),
                ("websocket_message_sent", "frontend_message_received", "WebSocket → Frontend")
            ]
            
            for source_key, dest_key, stage_name in stages:
                source_count = stats["stats"].get(source_key, 0)
                dest_count = stats["stats"].get(dest_key, 0)
                if source_count > dest_count:
                    loss = source_count - dest_count
                    loss_pct = (loss / source_count) * 100 if source_count > 0 else 0
                    losses.append({
                        "stage": stage_name,
                        "loss_count": loss,
                        "loss_percentage": loss_pct
                    })
            
            return {
                "status": "ok",
                "timestamp": datetime.now().isoformat(),
                "pipeline_efficiency": efficiency,
                "total_events": stats["total_events"],
                "stage_stats": {
                    "orpheus": {
                        "chunks_generated": stats["stats"].get("orpheus_chunk_generated", 0),
                        "generation_complete": stats["stats"].get("orpheus_generation_complete", 0),
                        "errors": stats["stats"].get("orpheus_generation_error", 0)
                    },
                    "backend": {
                        "chunks_received": stats["stats"].get("backend_chunk_received", 0),
                        "chunks_queued": stats["stats"].get("backend_chunk_queued", 0),
                        "messages_yielded": stats["stats"].get("backend_message_yielded", 0),
                        "stream_complete": stats["stats"].get("backend_stream_complete", 0)
                    },
                    "conversation": {
                        "messages_received": stats["stats"].get("conversation_message_received", 0),
                        "loop_complete": stats["stats"].get("conversation_loop_complete", 0)
                    },
                    "websocket": {
                        "audio_received": stats["stats"].get("websocket_audio_received", 0),
                        "opus_encoded": stats["stats"].get("websocket_opus_encoded", 0),
                        "messages_sent": stats["stats"].get("websocket_message_sent", 0),
                        "no_opus_output": stats["stats"].get("websocket_no_opus_output", 0)
                    },
                    "frontend": {
                        "messages_received": stats["stats"].get("frontend_message_received", 0),
                        "sent_to_decoder": stats["stats"].get("frontend_sent_to_decoder", 0),
                        "decoder_frames": stats["stats"].get("frontend_decoder_frames", 0)
                    }
                },
                "losses": losses,
                "recent_events": stats["recent_logs"]
            }
        
        @app.function()
        @modal.web_endpoint(method="POST")
        def reset_audio_debug():
            """Reset audio debugging statistics."""
            audio_debug.reset_stats()
            return {"status": "reset", "timestamp": datetime.now().isoformat()}
        
        return audio_debug_stats, reset_audio_debug
    
    @staticmethod
    def log_backend_event(event: str, data: Dict[str, Any] = None):
        """Log a backend TTS event."""
        audio_debug.log_event("backend", event, data)
    
    @staticmethod
    def log_conversation_event(event: str, data: Dict[str, Any] = None):
        """Log a conversation TTS event."""
        audio_debug.log_event("conversation", event, data)
    
    @staticmethod
    def log_websocket_event(event: str, data: Dict[str, Any] = None):
        """Log a WebSocket event."""
        audio_debug.log_event("websocket", event, data)
    
    @staticmethod
    def log_frontend_event(event: str, data: Dict[str, Any] = None):
        """Log a frontend event (this would be called via API from frontend)."""
        audio_debug.log_event("frontend", event, data)


# Convenience functions for easy integration
def log_orpheus_event(event: str, data: Dict[str, Any] = None):
    """Log an Orpheus generation event."""
    audio_debug.log_event("orpheus", event, data)

def log_backend_event(event: str, data: Dict[str, Any] = None):
    """Log a backend TTS event."""
    ModalAudioDebugger.log_backend_event(event, data)

def log_conversation_event(event: str, data: Dict[str, Any] = None):
    """Log a conversation TTS event."""
    ModalAudioDebugger.log_conversation_event(event, data)

def log_websocket_event(event: str, data: Dict[str, Any] = None):
    """Log a WebSocket event."""
    ModalAudioDebugger.log_websocket_event(event, data)

def log_frontend_event(event: str, data: Dict[str, Any] = None):
    """Log a frontend event."""
    ModalAudioDebugger.log_frontend_event(event, data)


# Context manager for session tracking
class AudioDebugSession:
    """Context manager for tracking a complete audio generation session."""
    
    def __init__(self, session_id: str = None):
        self.session_id = session_id or f"session_{int(time.time())}"
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        audio_debug.log_event("session", "start", {
            "session_id": self.session_id,
            "start_time": self.start_time
        })
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        
        audio_debug.log_event("session", "end", {
            "session_id": self.session_id,
            "duration": duration,
            "success": exc_type is None,
            "error": str(exc_val) if exc_val else None
        })


if __name__ == "__main__":
    # Example usage
    print("Modal Audio Debug System")
    print("This module provides debugging for Modal-based audio pipelines")
    
    # Example of how to use the debugging
    with AudioDebugSession("test_session") as session:
        log_orpheus_event("generation_start", {"text": "Test message"})
        log_orpheus_event("chunk_generated", {"chunk_number": 1, "bytes": 4800})
        log_backend_event("chunk_queued", {"samples": 2400})
    
    # Print current stats
    stats = audio_debug.get_stats()
    print(f"Total events logged: {stats['total_events']}")
    print(f"Recent events: {len(stats['recent_logs'])}")