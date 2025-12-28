"""
Diffusion Denoiser network for the ALD architecture.

This network takes:
    - Noisy embedding z_k at diffusion step k
    - Diffusion timestep embedding t_diff
    - Context h_t from the Context Encoder
    
And predicts the clean Uni-Mol embedding for the current token.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from ..core.attention import MultiHeadAttention, CrossAttention
from ..core.layers import FeedForward, DenoiserBlock, AdaptiveLayerNorm
from ..core.embeddings import DiffusionTimeEmbedding


class DiffusionDenoiser(nn.Module):
    """
    Denoising network for single-token diffusion.
    
    Architecture:
        1. Project noisy embedding to model dimension
        2. Add diffusion time embedding
        3. Process through denoiser blocks with cross-attention to context
        4. Project back to embedding dimension
        
    The network predicts the noise ε (or equivalently, the clean embedding x_0).
    
    Args:
        embedding_dim: Dimension of Uni-Mol embeddings
        d_model: Internal model dimension
        n_heads: Number of attention heads
        n_layers: Number of denoiser blocks
        d_ff: Feed-forward hidden dimension
        dropout: Dropout probability
        time_embed_dim: Dimension of time embedding
        predict_noise: If True, predict noise; if False, predict clean embedding
    """
    
    def __init__(
        self,
        embedding_dim: int = 512,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 2048,
        dropout: float = 0.1,
        time_embed_dim: int = 128,
        predict_noise: bool = True
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.d_model = d_model
        self.predict_noise = predict_noise
        
        # Project embedding to model dimension
        self.input_projection = nn.Linear(embedding_dim, d_model)
        
        # Time embedding
        self.time_embedding = DiffusionTimeEmbedding(time_embed_dim, d_model)
        
        # Denoiser blocks with cross-attention
        self.denoiser_blocks = nn.ModuleList([
            DenoiserBlock(d_model, n_heads, d_ff, dropout, activation='gelu')
            for _ in range(n_layers)
        ])
        
        # Adaptive layer norms conditioned on time
        self.adaptive_norms = nn.ModuleList([
            AdaptiveLayerNorm(d_model, d_model)
            for _ in range(n_layers)
        ])
        
        # Output projection
        self.output_norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, embedding_dim)
        
        # Initialize output projection to small values for stable training
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        
    def forward(
        self,
        z_noisy: torch.Tensor,
        t_diff: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Denoise the noisy embedding.
        
        Args:
            z_noisy: Noisy embedding [batch_size, embedding_dim] or [batch_size, 1, embedding_dim]
            t_diff: Diffusion timestep [batch_size] (normalized to [0, 1] or integer)
            context: Context from encoder [batch_size, ctx_len, d_model] (optional)
            context_mask: Mask for context [batch_size, ctx_len] (optional)
            
        Returns:
            Predicted noise or clean embedding [batch_size, embedding_dim]
        """
        # Handle input dimensions
        if z_noisy.dim() == 2:
            z_noisy = z_noisy.unsqueeze(1)  # [batch_size, 1, embedding_dim]
        
        batch_size = z_noisy.size(0)
        
        # Project to model dimension
        x = self.input_projection(z_noisy)  # [batch_size, 1, d_model]
        
        # Get time embedding and add to input
        t_emb = self.time_embedding(t_diff)  # [batch_size, d_model]
        x = x + t_emb.unsqueeze(1)  # [batch_size, 1, d_model]
        
        # Process through denoiser blocks
        for block, ada_norm in zip(self.denoiser_blocks, self.adaptive_norms):
            # Apply adaptive normalization conditioned on time
            x = ada_norm(x, t_emb)
            # Process through block (includes cross-attention with context)
            x = block(x, context, context_mask)
        
        # Output projection
        x = self.output_norm(x)
        output = self.output_projection(x)  # [batch_size, 1, embedding_dim]
        
        # Remove sequence dimension
        output = output.squeeze(1)  # [batch_size, embedding_dim]
        
        return output


class LightweightDenoiser(nn.Module):
    """
    Lightweight denoiser for faster inference.
    
    A simpler architecture with fewer parameters, suitable for
    scenarios where speed is more important than quality.
    
    Args:
        embedding_dim: Dimension of embeddings
        hidden_dim: Hidden dimension
        n_layers: Number of MLP layers
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        embedding_dim: int = 512,
        hidden_dim: int = 1024,
        n_layers: int = 3,
        dropout: float = 0.1,
        time_embed_dim: int = 128
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # Time embedding
        self.time_embedding = DiffusionTimeEmbedding(time_embed_dim, hidden_dim)
        
        # Context projection (if provided)
        self.context_projection = nn.Linear(embedding_dim, hidden_dim)
        
        # Build MLP layers
        layers = []
        in_dim = embedding_dim + hidden_dim + hidden_dim  # noisy + time + context
        
        for i in range(n_layers):
            out_dim = hidden_dim if i < n_layers - 1 else embedding_dim
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim) if i < n_layers - 1 else nn.Identity(),
                nn.GELU() if i < n_layers - 1 else nn.Identity(),
                nn.Dropout(dropout) if i < n_layers - 1 else nn.Identity(),
            ])
            in_dim = out_dim
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(
        self,
        z_noisy: torch.Tensor,
        t_diff: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Lightweight denoising.
        
        Args:
            z_noisy: [batch_size, embedding_dim]
            t_diff: [batch_size]
            context: [batch_size, ctx_len, embedding_dim] (optional)
            context_mask: [batch_size, ctx_len] (optional)
        """
        batch_size = z_noisy.size(0)
        
        # Time embedding
        t_emb = self.time_embedding(t_diff)  # [batch_size, hidden_dim]
        
        # Context aggregation (mean pooling if available)
        if context is not None and context.size(1) > 0:
            if context_mask is not None:
                # Masked mean pooling
                mask = context_mask.unsqueeze(-1).float()
                ctx_sum = (context * mask).sum(dim=1)
                ctx_count = mask.sum(dim=1).clamp(min=1)
                ctx_agg = ctx_sum / ctx_count
            else:
                ctx_agg = context.mean(dim=1)
            ctx_emb = self.context_projection(ctx_agg)  # [batch_size, hidden_dim]
        else:
            ctx_emb = torch.zeros(batch_size, t_emb.size(-1), device=z_noisy.device)
        
        # Concatenate all inputs
        combined = torch.cat([z_noisy, t_emb, ctx_emb], dim=-1)
        
        # Forward through MLP
        output = self.mlp(combined)
        
        return output
