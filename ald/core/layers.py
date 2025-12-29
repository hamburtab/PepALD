"""
Transformer layer implementations for the ALD architecture.

Contains:
    - FeedForward: Position-wise feed-forward network

    - CausalTransformerLayer: Transformer layer with causal masking
    - DenoiserBlock: Transformer block for the diffusion denoiser
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .attention import MultiHeadAttention, CausalMultiHeadAttention, CrossAttention


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    
    Two-layer MLP with ReLU activation, as used in the original Transformer.
    
    Args:
        d_model: Model dimension (input and output)
        d_ff: Hidden dimension (typically 4 * d_model)
        dropout: Dropout probability
        activation: Activation function ('relu' or 'gelu')
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = 'relu'
    ):
        super().__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'silu':
            self.activation = F.silu
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply feed-forward transformation.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class CausalTransformerLayer(nn.Module):
    """
    Causal Transformer Layer for autoregressive modeling.
    
    Uses causal (masked) self-attention to ensure that each position
    can only attend to previous positions.
    
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        d_ff: Feed-forward hidden dimension
        dropout: Dropout probability
        max_seq_len: Maximum sequence length for causal mask
        activation: Activation function for FFN
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        max_seq_len: int = 256,
        activation: str = 'relu'
    ):
        super().__init__()
        
        self.self_attn = CausalMultiHeadAttention(
            d_model, n_heads, dropout, max_seq_len
        )
        self.feed_forward = FeedForward(d_model, d_ff, dropout, activation)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with causal masking.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            key_padding_mask: Optional padding mask [batch_size, seq_len]
            return_attention: Whether to return attention weights
            
        Returns:
            output: Transformed tensor
            attn_weights: Optional attention weights
        """
        # Causal self-attention with residual
        attn_output, attn_weights = self.self_attn(
            x, x, x, key_padding_mask, return_attention=True
        )
        x = self.norm1(x + self.dropout1(attn_output))
        
        # Feed-forward with residual
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        
        if return_attention:
            return x, attn_weights
        return x, None


class DenoiserBlock(nn.Module):
    """
    Transformer block for the Diffusion Denoiser.
    
    Contains:
        1. Self-attention on the noisy embedding
        2. Cross-attention with context from the Context Encoder
        3. Feed-forward network
        
    All with residual connections and layer normalization.
    
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        d_ff: Feed-forward hidden dimension
        dropout: Dropout probability
        activation: Activation function
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = 'gelu'
    ):
        super().__init__()
        
        # Self-attention
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        # Cross-attention with context
        self.cross_attn = CrossAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        
        # Feed-forward
        self.feed_forward = FeedForward(d_model, d_ff, dropout, activation)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through denoiser block.
        
        Args:
            x: Noisy embedding [batch_size, 1, d_model]
            context: Context from encoder [batch_size, ctx_len, d_model] (optional)
            context_mask: Mask for context [batch_size, ctx_len] (optional)
            
        Returns:
            Processed embedding [batch_size, 1, d_model]
        """
        # Self-attention
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout1(attn_out))
        
        # Cross-attention with context (if available)
        if context is not None and context.size(1) > 0:
            cross_out = self.cross_attn(x, context, context_mask)
            x = self.norm2(x + self.dropout2(cross_out))
        
        # Feed-forward
        ff_out = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_out))
        
        return x


class AdaptiveLayerNorm(nn.Module):
    """
    Adaptive Layer Normalization conditioned on timestep embedding.
    
    Modulates the normalized features using scale and shift parameters
    predicted from the timestep embedding.
    
    Args:
        d_model: Model dimension
        time_embed_dim: Dimension of time embedding
    """
    
    def __init__(self, d_model: int, time_embed_dim: int):
        super().__init__()
        
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        
        # Predict scale and shift from time embedding
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, 2 * d_model)
        )
        
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        Apply adaptive layer normalization.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            time_emb: Time embedding [batch_size, time_embed_dim]
            
        Returns:
            Normalized and modulated tensor
        """
        # Get scale and shift
        scale_shift = self.adaLN_modulation(time_emb)
        scale, shift = scale_shift.chunk(2, dim=-1)
        
        # Handle dimensions
        if x.dim() == 3:
            scale = scale.unsqueeze(1)
            shift = shift.unsqueeze(1)
        
        # Normalize and modulate
        x = self.norm(x)
        x = x * (1 + scale) + shift
        
        return x
