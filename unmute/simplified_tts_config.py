"""
Configuration for the simplified TTS pipeline.
"""

import os
from typing import Literal

# Environment variable to enable/disable simplified TTS pipeline
SIMPLIFIED_TTS_ENABLED = os.getenv("SIMPLIFIED_TTS_ENABLED", "false").lower() == "true"

# Pipeline configuration
SIMPLIFIED_TTS_CONFIG = {
    # Ring buffer settings
    "ring_buffer_ms": float(os.getenv("SIMPLIFIED_TTS_RING_BUFFER_MS", "200")),  # 200ms max buffer
    "opus_packet_ms": float(os.getenv("SIMPLIFIED_TTS_OPUS_PACKET_MS", "40")),  # 40ms packets (2x20ms frames)
    
    # Frontend buffer settings
    "frontend_initial_buffer_ms": float(os.getenv("SIMPLIFIED_TTS_FRONTEND_INITIAL_MS", "80")),  # 80ms initial buffer
    "frontend_target_buffer_ms": float(os.getenv("SIMPLIFIED_TTS_FRONTEND_TARGET_MS", "100")),  # 100ms target buffer
    "frontend_max_buffer_ms": float(os.getenv("SIMPLIFIED_TTS_FRONTEND_MAX_MS", "120")),      # 120ms max buffer
}

def get_tts_pipeline_type() -> Literal["simplified", "legacy"]:
    """Get the TTS pipeline type based on configuration."""
    return "simplified" if SIMPLIFIED_TTS_ENABLED else "legacy"

def should_use_simplified_tts() -> bool:
    """Check if simplified TTS pipeline should be used."""
    return SIMPLIFIED_TTS_ENABLED