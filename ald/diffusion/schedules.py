"""
Variance schedules for the diffusion process.

Implements the beta schedules and derived quantities (alphas, etc.)
exactly as specified in the original DDPM paper.

CRITICAL: The mathematical formulas here are preserved from helm_diffusion.py
"""

import torch
import torch.nn.functional as F
from typing import Dict, Literal
from abc import ABC, abstractmethod


class VarianceSchedule(ABC):
    """
    Abstract base class for variance schedules.
    
    Defines the interface for computing betas and derived quantities
    used in the diffusion forward and reverse processes.
    """
    
    @abstractmethod
    def get_betas(self, num_steps: int) -> torch.Tensor:
        """
        Get the beta schedule.
        
        Args:
            num_steps: Number of diffusion steps
            
        Returns:
            betas: Tensor of shape [num_steps] containing β_t values
        """
        pass
    
    def compute_schedule_quantities(
        self,
        num_steps: int,
        device: torch.device = torch.device('cpu')
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all quantities derived from the beta schedule.
        
        This implements the exact same formulas as the original helm_diffusion.py:
            - alphas = 1 - betas
            - alphas_cumprod = cumulative product of alphas
            - sqrt_alphas_cumprod = sqrt(alphas_cumprod)
            - sqrt_one_minus_alphas_cumprod = sqrt(1 - alphas_cumprod)
            - sigmas for DDPM sampling
        
        Args:
            num_steps: Number of diffusion steps
            device: Device for tensors
            
        Returns:
            Dictionary containing all schedule quantities
        """
        betas = self.get_betas(num_steps).to(device)
        
        # α_t = 1 - β_t
        alphas = 1.0 - betas
        
        # ᾱ_t = ∏_{s=1}^{t} α_s (cumulative product)
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        # ᾱ_{t-1} (shifted with padding)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # σ_t for DDPM sampling (exact formula from original code)
        sigmas = torch.zeros_like(betas)
        for i in range(1, len(betas)):
            sigmas[i] = ((1 - alphas_cumprod_prev[i]) / (1 - alphas_cumprod[i])) * betas[i]
        sigmas = torch.sqrt(sigmas)
        
        return {
            'betas': betas,
            'alphas': alphas,
            'alphas_cumprod': alphas_cumprod,
            'alphas_cumprod_prev': alphas_cumprod_prev,
            'sqrt_alphas_cumprod': torch.sqrt(alphas_cumprod),
            'sqrt_one_minus_alphas_cumprod': torch.sqrt(1.0 - alphas_cumprod),
            'sigmas': sigmas,
        }


class LinearSchedule(VarianceSchedule):
    """
    Linear variance schedule as used in the original DDPM paper.
    
    β_t increases linearly from beta_start to beta_end.
    
    Args:
        beta_start: Starting value of beta (default: 1e-4)
        beta_end: Ending value of beta (default: 0.02)
    """
    
    def __init__(self, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.beta_start = beta_start
        self.beta_end = beta_end
        
    def get_betas(self, num_steps: int) -> torch.Tensor:
        """
        Generate linear beta schedule.
        
        Formula: β_t = beta_start + t * (beta_end - beta_start) / (T - 1)
        """
        return torch.linspace(self.beta_start, self.beta_end, num_steps)


class CosineSchedule(VarianceSchedule):
    """
    Cosine variance schedule as proposed in "Improved DDPM".
    
    Provides smoother noise scheduling that can improve sample quality,
    especially for high-resolution images/long sequences.
    
    Args:
        s: Offset to prevent β_t from being too small at t=0 (default: 0.008)
    """
    
    def __init__(self, s: float = 0.008):
        self.s = s
        
    def get_betas(self, num_steps: int) -> torch.Tensor:
        """
        Generate cosine beta schedule.
        
        Formula (exact from original code):
            f(t) = cos((t/T + s) / (1 + s) * π/2)^2
            ᾱ_t = f(t) / f(0)
            β_t = 1 - ᾱ_t / ᾱ_{t-1}
        """
        steps = num_steps + 1
        x = torch.linspace(0, num_steps, steps)
        
        # Compute ᾱ_t using cosine schedule
        alphas_cumprod = torch.cos(((x / num_steps) + self.s) / (1 + self.s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        
        # Derive betas from alphas_cumprod
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        
        # Clip betas to prevent instability
        return torch.clip(betas, 0.0001, 0.9999)


class SigmoidSchedule(VarianceSchedule):
    """
    Sigmoid variance schedule.
    
    An alternative schedule that provides smooth transitions.
    
    Args:
        beta_start: Starting value
        beta_end: Ending value
    """
    
    def __init__(self, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.beta_start = beta_start
        self.beta_end = beta_end
        
    def get_betas(self, num_steps: int) -> torch.Tensor:
        """Generate sigmoid beta schedule."""
        x = torch.linspace(-6, 6, num_steps)
        betas = torch.sigmoid(x)
        betas = betas * (self.beta_end - self.beta_start) + self.beta_start
        return betas


def create_variance_schedule(
    schedule_type: Literal['linear', 'cosine', 'sigmoid'] = 'cosine',
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
    **kwargs
) -> VarianceSchedule:
    """
    Factory function to create a variance schedule.
    
    Args:
        schedule_type: Type of schedule ('linear', 'cosine', 'sigmoid')
        beta_start: Starting beta value (for linear/sigmoid)
        beta_end: Ending beta value (for linear/sigmoid)
        **kwargs: Additional arguments for specific schedules
        
    Returns:
        VarianceSchedule instance
    """
    if schedule_type == 'linear':
        return LinearSchedule(beta_start, beta_end)
    elif schedule_type == 'cosine':
        s = kwargs.get('s', 0.008)
        return CosineSchedule(s)
    elif schedule_type == 'sigmoid':
        return SigmoidSchedule(beta_start, beta_end)
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")
