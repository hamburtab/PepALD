"""
Autoregressive Latent Diffusion (ALD) for HELM Peptide Generation

This package implements an autoregressive latent diffusion architecture that generates
peptide sequences token-by-token, where each token is generated via a complete
diffusion denoising process conditioned on previously generated tokens.

Architecture Overview:
    1. Macro-Step (Autoregressive Loop): Generate sequence token-by-token
    2. Context Encoder: Causal Transformer processing history [x_0, ..., x_{t-1}]
    3. Diffusion Process: Complete diffusion denoising at EACH step
    4. Token Mapper: Nearest neighbor (cosine similarity) to HELM monomers
    5. Ring Bond Prediction: Predict connections between tokens
"""

from .models.ald_model import AutoregressiveLatentDiffusion
from .models.context_encoder import CausalContextEncoder
from .models.token_mapper import TokenMapper
from .models.ring_predictor import RingBondPredictor
from .diffusion.engine import DiffusionEngine
from .utils.data import HELMDataset

__version__ = "1.0.0"
__all__ = [
    "AutoregressiveLatentDiffusion",
    "CausalContextEncoder", 
    "TokenMapper",
    "RingBondPredictor",
    "DiffusionEngine",
    "HELMDataset",
]
