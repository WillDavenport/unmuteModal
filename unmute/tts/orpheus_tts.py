"""
Orpheus Text-to-Speech Service
Replaces the Kyutai TTS implementation with Orpheus for better performance.
"""

import asyncio
import logging
import time
from typing import AsyncIterator, Optional
import numpy as np
import msgpack
from pydantic import BaseModel
from unmute.tts.orpheus_model import OrpheusModel

logger = logging.getLogger(__name__)


class OrpheusTextToSpeech:
    """Orpheus-based Text-to-Speech service."""
    
    def __init__(self, voice: str = "tara"):
        self.voice = voice
        self.model: Optional[OrpheusModel] = None
        self.is_loaded = False
        
    async def load_model(self):
        """Load the Orpheus model."""
        if self.is_loaded:
            return
            
        logger.info("Loading Orpheus TTS model...")
        self.model = OrpheusModel()
        self.model.load()
        self.is_loaded = True
        logger.info("Orpheus TTS model loaded successfully")
    
    async def synthesize(self, text: str, voice: Optional[str] = None) -> AsyncIterator[bytes]:
        """
        Synthesize speech from text using Orpheus.
        
        Args:
            text: Text to synthesize
            voice: Voice to use (defaults to instance voice)
            
        Yields:
            Audio bytes in WAV format
        """
        if not self.is_loaded or not self.model:
            await self.load_model()
            
        voice = voice or self.voice
        logger.info(f"Synthesizing speech with Orpheus: '{text[:50]}...' using voice '{voice}'")
        
        start_time = time.time()
        total_bytes = 0
        
        try:
            async for audio_chunk in self.model.generate_speech(text, voice):
                total_bytes += len(audio_chunk)
                yield audio_chunk
                
            elapsed_time = time.time() - start_time
            logger.info(f"Orpheus TTS completed: {total_bytes} bytes in {elapsed_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Orpheus TTS synthesis failed: {e}")
            raise


class OrpheusTTSMessage(BaseModel):
    """Message format for Orpheus TTS communication."""
    type: str
    text: Optional[str] = None
    voice: Optional[str] = None
    audio_data: Optional[bytes] = None
    error: Optional[str] = None


async def create_orpheus_tts_service(voice: str = "tara") -> OrpheusTextToSpeech:
    """Create and initialize an Orpheus TTS service."""
    service = OrpheusTextToSpeech(voice=voice)
    await service.load_model()
    return service