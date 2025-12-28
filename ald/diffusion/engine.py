"""
Diffusion Engine for the ALD architecture.

Implements the core diffusion process:
    - Forward process (adding noise)
    - Reverse process (denoising/sampling)
    
CRITICAL: The mathematical formulas are preserved exactly from helm_diffusion.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, Literal

from .schedules import create_variance_schedule, VarianceSchedule
from .denoiser import DiffusionDenoiser


class DiffusionEngine(nn.Module):
    """
    Diffusion engine for single-token generation.
    
    At each autoregressive step t, this engine performs a complete
    diffusion process to generate one token embedding:
        1. Start from pure noise z_K ~ N(0, I)
        2. Iteratively denoise: z_{k-1} = denoise(z_k, k, context)
        3. Output clean embedding z_0
    
    Args:
        embedding_dim: Dimension of Uni-Mol embeddings
        d_model: Model dimension for denoiser
        n_heads: Number of attention heads
        n_layers: Number of denoiser layers
        d_ff: Feed-forward dimension
        dropout: Dropout probability
        num_diffusion_steps: Number of diffusion steps (K)
        variance_schedule: Type of variance schedule
        beta_start: Starting beta (for linear schedule)
        beta_end: Ending beta (for linear schedule)
    """
    
    def __init__(
        self,
        embedding_dim: int = 512,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 2048,
        dropout: float = 0.1,
        num_diffusion_steps: int = 100,
        variance_schedule: Literal['linear', 'cosine'] = 'cosine',
        beta_start: float = 1e-4,
        beta_end: float = 0.02
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_steps = num_diffusion_steps
        
        # Create denoiser network
        self.denoiser = DiffusionDenoiser(
            embedding_dim=embedding_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            predict_noise=True  # Predict noise ε
        )
        
        # Setup variance schedule (exact formulas from original)
        self._setup_variance_schedule(variance_schedule, beta_start, beta_end)
        
    def _setup_variance_schedule(
        self,
        schedule_type: str,
        beta_start: float,
        beta_end: float
    ) -> None:
        """
        Setup variance schedule and register all derived quantities as buffers.
        
        IMPORTANT: This implements the EXACT same formulas as helm_diffusion.py
        """
        schedule = create_variance_schedule(schedule_type, beta_start, beta_end)
        quantities = schedule.compute_schedule_quantities(self.num_steps)
        
        # Register all quantities as buffers
        self.register_buffer('betas', quantities['betas'])
        self.register_buffer('alphas', quantities['alphas'])
        self.register_buffer('alphas_cumprod', quantities['alphas_cumprod'])
        self.register_buffer('alphas_cumprod_prev', quantities['alphas_cumprod_prev'])
        self.register_buffer('sqrt_alphas_cumprod', quantities['sqrt_alphas_cumprod'])
        self.register_buffer('sqrt_one_minus_alphas_cumprod', quantities['sqrt_one_minus_alphas_cumprod'])
        self.register_buffer('sigmas', quantities['sigmas'])
        
    def add_noise(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add noise to clean embedding (forward diffusion process).
        
        Formula (exact from original):
            x_t = √(ᾱ_t) * x_0 + √(1 - ᾱ_t) * ε
            
        where ε ~ N(0, I)
        
        Args:
            x_0: Clean embedding [batch_size, embedding_dim]
            t: Timesteps [batch_size] (1-indexed: t ∈ [1, num_steps])
            noise: Optional pre-generated noise
            
        Returns:
            x_t: Noisy embedding [batch_size, embedding_dim]
            noise: The noise that was added
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        # Get schedule coefficients (t is 1-indexed, arrays are 0-indexed)
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t - 1]
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t - 1]
        
        # Handle dimensions
        if x_0.dim() == 2:
            sqrt_alpha_cumprod_t = sqrt_alpha_cumprod_t.view(-1, 1)
            sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alpha_cumprod_t.view(-1, 1)
        elif x_0.dim() == 3:
            sqrt_alpha_cumprod_t = sqrt_alpha_cumprod_t.view(-1, 1, 1)
            sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alpha_cumprod_t.view(-1, 1, 1)
        
        # Forward diffusion
        x_t = sqrt_alpha_cumprod_t * x_0 + sqrt_one_minus_alpha_cumprod_t * noise
        
        return x_t, noise
    
    def predict_noise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Predict the noise component in x_t.
        
        Args:
            x_t: Noisy embedding [batch_size, embedding_dim]
            t: Timesteps [batch_size]
            context: Context from encoder [batch_size, ctx_len, d_model]
            context_mask: Mask for context
            
        Returns:
            Predicted noise [batch_size, embedding_dim]
        """
        # Normalize timestep to [0, 1] for the denoiser
        t_normalized = t.float() / self.num_steps
        
        return self.denoiser(x_t, t_normalized, context, context_mask)
    
    def training_step(
        self,
        x_0: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training loss for diffusion model.
        
        Samples random timesteps, adds noise, predicts noise, computes MSE loss.
        
        Args:
            x_0: Clean embeddings [batch_size, embedding_dim]
            context: Context from encoder
            context_mask: Mask for context
            
        Returns:
            Dictionary with 'loss', 'predicted_noise', 'target_noise', 't'
        """
        batch_size = x_0.size(0)
        device = x_0.device
        
        # Sample random timesteps (1-indexed: t ∈ [1, num_steps])
        t = torch.randint(1, self.num_steps + 1, (batch_size,), device=device)
        
        # Add noise
        x_t, noise = self.add_noise(x_0, t)
        
        # Predict noise
        predicted_noise = self.predict_noise(x_t, t, context, context_mask)
        
        # Compute MSE loss
        loss = F.mse_loss(predicted_noise, noise)
        
        return {
            'loss': loss,
            'predicted_noise': predicted_noise,
            'target_noise': noise,
            't': t,
            'x_t': x_t
        }
    
    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        return_trajectory: bool = False
    ) -> torch.Tensor:
        """
        Sample clean embeddings via reverse diffusion process.
        
        DDPM Sampling Formula (exact from original):
            x_{t-1} = (1/√α_t) * (x_t - (1-α_t)/√(1-ᾱ_t) * ε_θ(x_t, t)) + σ_t * z
            
        where z ~ N(0, I) for t > 1, z = 0 for t = 1
        
        Args:
            batch_size: Number of samples to generate
            context: Context from encoder [batch_size, ctx_len, d_model]
            context_mask: Mask for context
            device: Device for computation
            return_trajectory: Whether to return full denoising trajectory
            
        Returns:
            Clean embeddings [batch_size, embedding_dim]
            (Optional) Trajectory list if return_trajectory=True
        """
        if device is None:
            device = next(self.parameters()).device
        
        # Start from pure noise
        x_t = torch.randn(batch_size, self.embedding_dim, device=device)
        
        trajectory = [x_t] if return_trajectory else None
        
        # Reverse diffusion: t from num_steps down to 1
        for i in reversed(range(1, self.num_steps + 1)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            
            # Predict noise
            predicted_noise = self.predict_noise(x_t, t, context, context_mask)
            
            # Get schedule coefficients (exact from original)
            alpha_t = self.alphas[i - 1]
            alpha_bar_t = self.alphas_cumprod[i - 1]
            
            # Compute denoising coefficients
            c0 = 1.0 / torch.sqrt(alpha_t + 1e-8)
            c1 = (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t + 1e-8)
            
            # Add noise for t > 1
            if i > 1:
                z = torch.randn_like(x_t)
                sigma = self.sigmas[i - 1]
            else:
                z = torch.zeros_like(x_t)
                sigma = 0.0
            
            # DDPM denoising step
            x_t = c0 * (x_t - c1 * predicted_noise) + sigma * z
            
            if return_trajectory:
                trajectory.append(x_t)
        
        if return_trajectory:
            return x_t, trajectory
        return x_t
    
    @torch.no_grad()
    def sample_ddim(
        self,
        batch_size: int,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        num_inference_steps: int = 50,
        eta: float = 0.0
    ) -> torch.Tensor:
        """
        Sample using DDIM (Denoising Diffusion Implicit Models).
        
        DDIM allows for faster sampling with fewer steps while maintaining quality.
        
        Args:
            batch_size: Number of samples
            context: Context from encoder
            context_mask: Mask for context
            device: Device for computation
            num_inference_steps: Number of DDIM steps (can be < num_steps)
            eta: Controls stochasticity (0 = deterministic, 1 = DDPM)
            
        Returns:
            Clean embeddings [batch_size, embedding_dim]
        """
        if device is None:
            device = next(self.parameters()).device
        
        # Compute timestep schedule for DDIM
        step_ratio = self.num_steps // num_inference_steps
        timesteps = torch.arange(0, self.num_steps, step_ratio, device=device)
        timesteps = torch.flip(timesteps, dims=[0]) + 1  # Reverse and 1-index
        
        # Start from noise
        x_t = torch.randn(batch_size, self.embedding_dim, device=device)
        
        for i in range(len(timesteps)):
            t = timesteps[i]
            t_batch = t.expand(batch_size)
            
            # Predict noise
            predicted_noise = self.predict_noise(x_t, t_batch, context, context_mask)
            
            # Get alphas
            alpha_bar_t = self.alphas_cumprod[t - 1]
            
            if i + 1 < len(timesteps):
                t_prev = timesteps[i + 1]
                alpha_bar_t_prev = self.alphas_cumprod[t_prev - 1]
            else:
                alpha_bar_t_prev = torch.tensor(1.0, device=device)
            
            # Predict x_0
            x_0_pred = (x_t - torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)
            
            # Compute direction pointing to x_t
            sigma_t = eta * torch.sqrt((1 - alpha_bar_t_prev) / (1 - alpha_bar_t)) * \
                      torch.sqrt(1 - alpha_bar_t / alpha_bar_t_prev)
            
            # DDIM step
            pred_dir = torch.sqrt(1 - alpha_bar_t_prev - sigma_t ** 2) * predicted_noise
            x_t = torch.sqrt(alpha_bar_t_prev) * x_0_pred + pred_dir
            
            if eta > 0 and i + 1 < len(timesteps):
                x_t = x_t + sigma_t * torch.randn_like(x_t)
        
        return x_t
