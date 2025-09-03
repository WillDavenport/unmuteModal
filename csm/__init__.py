"""
CSM (Conversational Speech Model) integration for Modal.

This package contains the core files and logic from the Sesame CSM repository
for generating conversational speech from text and audio inputs.
"""

from .generator import Generator, Segment, load_csm_1b
from .models import Model, ModelArgs
from .watermarking import load_watermarker, watermark, verify, CSM_1B_GH_WATERMARK

__all__ = [
    "Generator",
    "Segment", 
    "load_csm_1b",
    "Model",
    "ModelArgs",
    "load_watermarker",
    "watermark",
    "verify",
    "CSM_1B_GH_WATERMARK"
]
