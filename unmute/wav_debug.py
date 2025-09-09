"""
Debug utility for creating wav files from PCM audio data.
Supports both local filesystem and Modal volumes for storage.
"""
import os
import wave
import struct
import numpy as np
from typing import Union, List, Optional
from logging import getLogger
import asyncio
from datetime import datetime

logger = getLogger(__name__)

# Debug configuration
DEBUG_WAV_ENABLED = os.getenv("DEBUG_WAV_ENABLED", "true").lower() == "true"
DEBUG_WAV_DIR = os.getenv("DEBUG_WAV_DIR", "/tmp/unmute_audio_debug")

# Modal volume configuration
USE_MODAL_VOLUME = os.getenv("USE_MODAL_VOLUME", "true").lower() == "true"
MODAL_VOLUME_PATH = "/debug-audio"  # Mount point for Modal volume

class WavDebugWriter:
    """Utility class for writing PCM audio data to wav files for debugging."""
    
    def __init__(self, base_dir: str = DEBUG_WAV_DIR):
        self.base_dir = base_dir
        self.enabled = DEBUG_WAV_ENABLED
        self.use_modal_volume = USE_MODAL_VOLUME
        self.modal_volume = None
        
        # Audio accumulation for consolidated WAV files
        self.audio_buffers: dict[str, list[np.ndarray]] = {}  # stage -> list of audio chunks
        self.buffer_sample_rates: dict[str, int] = {}  # stage -> sample rate
        self.buffer_conversation_ids: dict[str, str] = {}  # stage -> conversation_id
        self.buffer_start_times: dict[str, datetime] = {}  # stage -> start time
        
        self._ensure_debug_dir()
        
    def _ensure_debug_dir(self):
        """Ensure the debug directory exists."""
        if not self.enabled:
            return
            
        if self.use_modal_volume:
            # Check if we're running in Modal and volume is mounted
            if os.path.exists(MODAL_VOLUME_PATH):
                self.base_dir = MODAL_VOLUME_PATH
                logger.info(f"WAV debug files will be written to Modal volume: {self.base_dir}")
                # Try to import Modal volume for commit functionality
                try:
                    import modal
                    self.modal_volume = modal.Volume.from_name("debug-audio-volume", create_if_missing=True)
                    logger.info("Modal volume configured for WAV debug files")
                except ImportError:
                    logger.warning("Modal not available, falling back to local filesystem")
                    self.use_modal_volume = False
                except Exception as e:
                    logger.warning(f"Failed to configure Modal volume: {e}, falling back to local filesystem")
                    self.use_modal_volume = False
            else:
                logger.warning(f"Modal volume path {MODAL_VOLUME_PATH} not found, falling back to local filesystem")
                self.use_modal_volume = False
        
        if not self.use_modal_volume:
            # Use local filesystem
            os.makedirs(self.base_dir, exist_ok=True)
            logger.info(f"WAV debug files will be written to local filesystem: {self.base_dir}")
    
    def _get_debug_filename(self, stage: str, conversation_id: str = "unknown", start_time: Optional[datetime] = None) -> str:
        """Generate a debug filename with timestamp."""
        if start_time:
            timestamp = start_time.strftime("%Y%m%d_%H%M%S_%f")[:-3]  # milliseconds
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # milliseconds
        return f"{conversation_id}_{stage}_{timestamp}.wav"
    
    def accumulate_audio(
        self, 
        pcm_data: Union[List[float], np.ndarray], 
        sample_rate: int,
        stage: str,
        conversation_id: str = "unknown"
    ) -> None:
        """
        Accumulate audio data for later consolidated WAV file creation.
        
        Args:
            pcm_data: PCM audio data as list of floats or numpy array (range -1.0 to 1.0)
            sample_rate: Sample rate in Hz
            stage: Debug stage identifier
            conversation_id: Conversation ID for filename
        """
        if not self.enabled:
            return
            
        try:
            # Convert to numpy array if needed
            if isinstance(pcm_data, list):
                pcm_data = np.array(pcm_data, dtype=np.float32)
            elif not isinstance(pcm_data, np.ndarray):
                logger.warning(f"Invalid PCM data type: {type(pcm_data)}")
                return
                
            # Ensure it's float32
            if pcm_data.dtype != np.float32:
                pcm_data = pcm_data.astype(np.float32)
            
            # Create stage key
            stage_key = f"{conversation_id}_{stage}"
            
            # Initialize buffer for this stage if needed
            if stage_key not in self.audio_buffers:
                self.audio_buffers[stage_key] = []
                self.buffer_sample_rates[stage_key] = sample_rate
                self.buffer_conversation_ids[stage_key] = conversation_id
                self.buffer_start_times[stage_key] = datetime.now()
                logger.info(f"=== WAV_DEBUG: Started accumulating audio for {stage_key} ===")
            
            # Add audio chunk to buffer
            self.audio_buffers[stage_key].append(pcm_data)
            
            # Log accumulation progress
            total_samples = sum(len(chunk) for chunk in self.audio_buffers[stage_key])
            duration_sec = total_samples / sample_rate
            logger.debug(f"=== WAV_DEBUG: Accumulated {len(pcm_data)} samples for {stage_key}, total: {total_samples} samples ({duration_sec:.2f}s) ===")
            
        except Exception as e:
            logger.error(f"Failed to accumulate audio data for stage {stage}: {e}")
    
    def finalize_stage(self, stage: str, conversation_id: str = "unknown") -> Optional[str]:
        """
        Finalize accumulated audio for a stage and create consolidated WAV file.
        
        Args:
            stage: Debug stage identifier
            conversation_id: Conversation ID
            
        Returns:
            Path to created WAV file, or None if no data or disabled
        """
        if not self.enabled:
            return None
            
        stage_key = f"{conversation_id}_{stage}"
        
        if stage_key not in self.audio_buffers or not self.audio_buffers[stage_key]:
            logger.debug(f"No audio data accumulated for {stage_key}")
            return None
        
        try:
            # Concatenate all audio chunks
            consolidated_audio = np.concatenate(self.audio_buffers[stage_key])
            sample_rate = self.buffer_sample_rates[stage_key]
            start_time = self.buffer_start_times[stage_key]
            
            # Create consolidated WAV file
            filepath = self.write_pcm_to_wav(
                consolidated_audio, sample_rate, stage, conversation_id, start_time
            )
            
            # Clear the buffer
            del self.audio_buffers[stage_key]
            del self.buffer_sample_rates[stage_key] 
            del self.buffer_conversation_ids[stage_key]
            del self.buffer_start_times[stage_key]
            
            total_samples = len(consolidated_audio)
            duration_sec = total_samples / sample_rate
            logger.info(f"=== WAV_DEBUG: Finalized {stage_key} with {total_samples} samples ({duration_sec:.2f}s) -> {filepath} ===")
            
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to finalize audio for stage {stage}: {e}")
            return None
    
    def write_pcm_to_wav(
        self, 
        pcm_data: Union[List[float], np.ndarray], 
        sample_rate: int,
        stage: str,
        conversation_id: str = "unknown",
        start_time: Optional[datetime] = None
    ) -> str | None:
        """
        Write PCM float32 data to a wav file for debugging.
        
        Args:
            pcm_data: PCM audio data as list of floats or numpy array (range -1.0 to 1.0)
            sample_rate: Sample rate in Hz (typically 24000)
            stage: Debug stage identifier (e.g., "tts_audio_queue", "output_queue", "audio_processor")
            conversation_id: Conversation ID for filename
            
        Returns:
            Path to created wav file, or None if disabled
        """
        if not self.enabled:
            return None
            
        try:
            # Convert to numpy array if needed
            if isinstance(pcm_data, list):
                pcm_data = np.array(pcm_data, dtype=np.float32)
            elif not isinstance(pcm_data, np.ndarray):
                logger.warning(f"Invalid PCM data type: {type(pcm_data)}")
                return None
                
            # Ensure it's float32
            if pcm_data.dtype != np.float32:
                pcm_data = pcm_data.astype(np.float32)
            
            # Clamp values to valid range
            pcm_data = np.clip(pcm_data, -1.0, 1.0)
            
            # Convert to int16 for wav file
            pcm_int16 = (pcm_data * 32767).astype(np.int16)
            
            # Generate filename
            filename = self._get_debug_filename(stage, conversation_id, start_time)
            filepath = os.path.join(self.base_dir, filename)
            
            # Write wav file
            with wave.open(filepath, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 2 bytes per sample (int16)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(pcm_int16.tobytes())
            
            # If using Modal volume, commit the changes
            if self.use_modal_volume and self.modal_volume:
                try:
                    self.modal_volume.commit()
                    logger.info(f"=== WAV_DEBUG: Committed {filename} to Modal volume ===")
                except Exception as e:
                    logger.warning(f"Failed to commit WAV file to Modal volume: {e}")
            
            logger.info(f"=== WAV_DEBUG: Wrote {len(pcm_data)} samples to {filepath} (stage: {stage}) ===")
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to write debug wav file for stage {stage}: {e}")
            return None
    
    async def write_pcm_to_wav_async(
        self,
        pcm_data: Union[List[float], np.ndarray],
        sample_rate: int,
        stage: str,
        conversation_id: str = "unknown"
    ) -> str | None:
        """Async wrapper for write_pcm_to_wav."""
        return await asyncio.to_thread(
            self.write_pcm_to_wav, pcm_data, sample_rate, stage, conversation_id
        )


# Global instance
_wav_debug_writer = WavDebugWriter()

def get_wav_debug_writer() -> WavDebugWriter:
    """Get the global wav debug writer instance."""
    return _wav_debug_writer

def write_debug_wav(
    pcm_data: Union[List[float], np.ndarray],
    sample_rate: int,
    stage: str,
    conversation_id: str = "unknown"
) -> str | None:
    """Convenience function to write debug wav file."""
    return _wav_debug_writer.write_pcm_to_wav(pcm_data, sample_rate, stage, conversation_id)

async def write_debug_wav_async(
    pcm_data: Union[List[float], np.ndarray],
    sample_rate: int,
    stage: str,
    conversation_id: str = "unknown"
) -> str | None:
    """Async convenience function to write debug wav file."""
    return await _wav_debug_writer.write_pcm_to_wav_async(pcm_data, sample_rate, stage, conversation_id)

def accumulate_debug_audio(
    pcm_data: Union[List[float], np.ndarray],
    sample_rate: int,
    stage: str,
    conversation_id: str = "unknown"
) -> None:
    """Convenience function to accumulate audio data for consolidated WAV file."""
    _wav_debug_writer.accumulate_audio(pcm_data, sample_rate, stage, conversation_id)

async def accumulate_debug_audio_async(
    pcm_data: Union[List[float], np.ndarray],
    sample_rate: int,
    stage: str,
    conversation_id: str = "unknown"
) -> None:
    """Async convenience function to accumulate audio data."""
    await asyncio.to_thread(
        _wav_debug_writer.accumulate_audio, pcm_data, sample_rate, stage, conversation_id
    )

def finalize_debug_stage(stage: str, conversation_id: str = "unknown") -> str | None:
    """Convenience function to finalize accumulated audio and create consolidated WAV file."""
    return _wav_debug_writer.finalize_stage(stage, conversation_id)

async def finalize_debug_stage_async(stage: str, conversation_id: str = "unknown") -> str | None:
    """Async convenience function to finalize accumulated audio."""
    return await asyncio.to_thread(_wav_debug_writer.finalize_stage, stage, conversation_id)
