"""
Diffusion DPO (Direct Preference Optimization) for ALD model.

Implements the standard Diffusion-DPO loss (Wallace et al. 2023):
    loss = -log σ( β · [(mse_ref_w - mse_θ_w) - (mse_ref_l - mse_θ_l)] )
"""

from .loss import compute_diffusion_dpo_loss, scatter_mean
from .dataset import PreferencePairDataset, PreferencePairCollator
from .trainer import DPOTrainer

__all__ = [
    "compute_diffusion_dpo_loss",
    "scatter_mean",
    "PreferencePairDataset",
    "PreferencePairCollator",
    "DPOTrainer",
]
