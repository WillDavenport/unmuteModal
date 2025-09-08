"""
Audio flow debugging utilities for tracking audio messages through the pipeline.
"""

import logging
import time
from typing import Dict, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class AudioFlowDebugTracker:
    """Tracks audio message flow through different stages of the pipeline."""
    
    # Stage counters
    modal_chunks_received: int = 0
    audio_messages_queued: int = 0
    audio_messages_yielded: int = 0
    output_queue_messages: int = 0
    websocket_messages_sent: int = 0
    frontend_messages_received: int = 0
    
    # Timing tracking
    first_chunk_time: float = 0
    last_chunk_time: float = 0
    
    # Data volume tracking
    total_bytes_from_modal: int = 0
    total_samples_processed: int = 0
    total_opus_bytes_sent: int = 0
    
    # Error tracking
    conversion_errors: int = 0
    queue_timeouts: int = 0
    websocket_errors: int = 0
    
    # State tracking
    generation_active: bool = False
    websocket_connected: bool = False
    
    def reset(self):
        """Reset all counters for a new generation."""
        self.modal_chunks_received = 0
        self.audio_messages_queued = 0
        self.audio_messages_yielded = 0
        self.output_queue_messages = 0
        self.websocket_messages_sent = 0
        self.frontend_messages_received = 0
        self.first_chunk_time = 0
        self.last_chunk_time = 0
        self.total_bytes_from_modal = 0
        self.total_samples_processed = 0
        self.total_opus_bytes_sent = 0
        self.conversion_errors = 0
        self.queue_timeouts = 0
        self.websocket_errors = 0
        
    def record_modal_chunk(self, chunk_size: int):
        """Record a chunk received from Modal."""
        self.modal_chunks_received += 1
        self.total_bytes_from_modal += chunk_size
        current_time = time.time()
        
        if self.first_chunk_time == 0:
            self.first_chunk_time = current_time
        self.last_chunk_time = current_time
        
    def record_audio_message_queued(self, sample_count: int):
        """Record an audio message queued in TTS service."""
        self.audio_messages_queued += 1
        self.total_samples_processed += sample_count
        
    def record_audio_message_yielded(self, sample_count: int):
        """Record an audio message yielded from TTS service."""
        self.audio_messages_yielded += 1
        
    def record_output_queue_message(self):
        """Record a message added to conversation output queue."""
        self.output_queue_messages += 1
        
    def record_websocket_message_sent(self, opus_bytes: int):
        """Record a message sent via WebSocket."""
        self.websocket_messages_sent += 1
        self.total_opus_bytes_sent += opus_bytes
        
    def record_conversion_error(self):
        """Record an audio conversion error."""
        self.conversion_errors += 1
        
    def record_queue_timeout(self):
        """Record a queue timeout."""
        self.queue_timeouts += 1
        
    def record_websocket_error(self):
        """Record a WebSocket error."""
        self.websocket_errors += 1
        
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all tracking data."""
        duration = self.last_chunk_time - self.first_chunk_time if self.first_chunk_time > 0 else 0
        
        return {
            "stage_counts": {
                "modal_chunks_received": self.modal_chunks_received,
                "audio_messages_queued": self.audio_messages_queued,
                "audio_messages_yielded": self.audio_messages_yielded,
                "output_queue_messages": self.output_queue_messages,
                "websocket_messages_sent": self.websocket_messages_sent,
                "frontend_messages_received": self.frontend_messages_received,
            },
            "data_volume": {
                "total_bytes_from_modal": self.total_bytes_from_modal,
                "total_samples_processed": self.total_samples_processed,
                "total_opus_bytes_sent": self.total_opus_bytes_sent,
            },
            "timing": {
                "generation_duration_sec": duration,
                "first_chunk_time": self.first_chunk_time,
                "last_chunk_time": self.last_chunk_time,
            },
            "errors": {
                "conversion_errors": self.conversion_errors,
                "queue_timeouts": self.queue_timeouts,
                "websocket_errors": self.websocket_errors,
            },
            "state": {
                "generation_active": self.generation_active,
                "websocket_connected": self.websocket_connected,
            },
            "dropoff_analysis": self._analyze_dropoff(),
        }
        
    def _analyze_dropoff(self) -> Dict[str, Any]:
        """Analyze where audio messages are being dropped."""
        stages = [
            ("Modal → TTS Queue", self.modal_chunks_received, self.audio_messages_queued),
            ("TTS Queue → TTS Yield", self.audio_messages_queued, self.audio_messages_yielded),
            ("TTS Yield → Output Queue", self.audio_messages_yielded, self.output_queue_messages),
            ("Output Queue → WebSocket", self.output_queue_messages, self.websocket_messages_sent),
            ("WebSocket → Frontend", self.websocket_messages_sent, self.frontend_messages_received),
        ]
        
        dropoffs = []
        for stage_name, input_count, output_count in stages:
            if input_count > 0:  # Only analyze if there was input
                drop_count = input_count - output_count
                drop_percentage = (drop_count / input_count) * 100 if input_count > 0 else 0
                
                dropoffs.append({
                    "stage": stage_name,
                    "input_count": input_count,
                    "output_count": output_count,
                    "dropped_count": drop_count,
                    "drop_percentage": drop_percentage,
                    "is_significant_drop": drop_percentage > 10,  # More than 10% drop
                })
        
        return dropoffs
        
    def log_summary(self):
        """Log a comprehensive summary of the audio flow."""
        summary = self.get_summary()
        
        logger.info("=== COMPREHENSIVE AUDIO FLOW DEBUG SUMMARY ===")
        logger.info("Stage Counts:")
        for stage, count in summary["stage_counts"].items():
            logger.info(f"  {stage}: {count}")
            
        logger.info("Data Volume:")
        for metric, value in summary["data_volume"].items():
            logger.info(f"  {metric}: {value}")
            
        logger.info("Timing:")
        for metric, value in summary["timing"].items():
            logger.info(f"  {metric}: {value}")
            
        logger.info("Errors:")
        for error_type, count in summary["errors"].items():
            logger.info(f"  {error_type}: {count}")
            
        logger.info("Dropoff Analysis:")
        for dropoff in summary["dropoff_analysis"]:
            status = "⚠️  SIGNIFICANT DROP" if dropoff["is_significant_drop"] else "✅ Normal"
            logger.info(f"  {dropoff['stage']}: {dropoff['input_count']} → {dropoff['output_count']} "
                       f"({dropoff['drop_percentage']:.1f}% drop) {status}")
                       
        logger.info("=== END AUDIO FLOW DEBUG SUMMARY ===")


# Global tracker instance
global_audio_debug_tracker = AudioFlowDebugTracker()


def get_audio_debug_tracker() -> AudioFlowDebugTracker:
    """Get the global audio debug tracker."""
    return global_audio_debug_tracker


def log_audio_debug_summary():
    """Log the current audio debug summary."""
    global_audio_debug_tracker.log_summary()


def reset_audio_debug_tracker():
    """Reset the global audio debug tracker."""
    global_audio_debug_tracker.reset()