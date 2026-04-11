"""
Attention mechanisms for the ALD architecture.

Contains:
    - MultiHeadAttention: Standard multi-head attention (bidirectional)
    - CausalMultiHeadAttention: Masked attention for autoregressive generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class MultiHeadAttention(nn.Module):
    """
    Standard Multi-Head Attention mechanism.
    
    This implementation follows the original Transformer architecture,
    computing scaled dot-product attention across multiple heads.
    
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads  
        dropout: Dropout probability
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scale = math.sqrt(self.d_k)
        
        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        # Output projection
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for multi-head attention.
        
        Args:
            query: Query tensor [batch_size, seq_len_q, d_model]
            key: Key tensor [batch_size, seq_len_k, d_model]  
            value: Value tensor [batch_size, seq_len_v, d_model]
            mask: Optional attention mask [batch_size, 1, seq_len_q, seq_len_k]
            return_attention: Whether to return attention weights
            
        Returns:
            output: Attention output [batch_size, seq_len_q, d_model]
            attn_weights: Optional attention weights [batch_size, n_heads, seq_len_q, seq_len_k]
        """
        batch_size = query.size(0)
        
        # Linear projections and reshape for multi-head attention
        # [batch_size, seq_len, d_model] -> [batch_size, n_heads, seq_len, d_k]
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        # scores: [batch_size, n_heads, seq_len_q, seq_len_k]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        # context: [batch_size, n_heads, seq_len_q, d_k]
        context = torch.matmul(attn_weights, V)
        
        # Reshape and project output
        # [batch_size, n_heads, seq_len_q, d_k] -> [batch_size, seq_len_q, d_model]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.w_o(context)
        
        if return_attention:
            return output, attn_weights
        return output, None


class CausalMultiHeadAttention(nn.Module):
    """
    Causal (Masked) Multi-Head Attention for autoregressive generation.
    
    This attention mechanism ensures that each position can only attend to
    previous positions (and itself), enabling autoregressive generation.
    
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        dropout: Dropout probability
        max_seq_len: Maximum sequence length for causal mask caching
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        max_seq_len: int = 256
    ):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scale = math.sqrt(self.d_k)
        self.max_seq_len = max_seq_len
        
        # Linear projections
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # Register causal mask as buffer (not a parameter)
        # Lower triangular matrix: allows attending to current and previous positions
        causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer('causal_mask', causal_mask)
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with causal masking.
        
        Args:
            query: Query tensor [batch_size, seq_len_q, d_model]
            key: Key tensor [batch_size, seq_len_k, d_model]
            value: Value tensor [batch_size, seq_len_v, d_model]
            key_padding_mask: Optional padding mask [batch_size, seq_len_k]
            return_attention: Whether to return attention weights
            
        Returns:
            output: Attention output [batch_size, seq_len_q, d_model]
            attn_weights: Optional attention weights
        """
        batch_size, seq_len_q, _ = query.size()
        seq_len_k = key.size(1)
        
        # Linear projections
        Q = self.w_q(query).view(batch_size, seq_len_q, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len_k, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len_k, self.n_heads, self.d_k).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply causal mask
        causal_mask = self.causal_mask[:seq_len_q, :seq_len_k]
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len_q, seq_len_k]
        scores = scores.masked_fill(causal_mask == 0, float('-inf'))
        
        # Apply key padding mask if provided
        if key_padding_mask is not None:
            # key_padding_mask: [batch_size, seq_len_k] -> [batch_size, 1, 1, seq_len_k]
            padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(padding_mask == 0, float('-inf'))
        
        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        # Handle NaN from all-masked positions
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)
        
        # Reshape and project
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.d_model)
        output = self.w_o(context)
        
        if return_attention:
            return output, attn_weights
        return output, None


class CrossAttention(nn.Module):
    """
    Cross-Attention mechanism for conditioning the denoiser on context.
    
    Used to inject context information from the Context Encoder into
    the Diffusion Denoiser.
    
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        dropout: Dropout probability
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scale = math.sqrt(self.d_k)
        
        # Query comes from the denoiser
        self.w_q = nn.Linear(d_model, d_model)
        # Key and Value come from the context encoder
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply cross-attention between query and context.
        
        Args:
            query: Query from denoiser [batch_size, 1, d_model] (single token)
            context: Context from encoder [batch_size, ctx_len, d_model]
            context_mask: Optional mask for context [batch_size, ctx_len]
            
        Returns:
            output: Contextualized representation [batch_size, 1, d_model]
        """
        batch_size = query.size(0)
        seq_len_q = query.size(1)
        seq_len_k = context.size(1)
        
        Q = self.w_q(query).view(batch_size, seq_len_q, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(context).view(batch_size, seq_len_k, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(context).view(batch_size, seq_len_k, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if context_mask is not None:
            mask = context_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.dropout(attn_weights)
        
        context_out = torch.matmul(attn_weights, V)
        context_out = context_out.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.d_model)
        
        return self.w_o(context_out)
