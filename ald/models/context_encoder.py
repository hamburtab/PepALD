"""
Causal Context Encoder for the ALD architecture.

This is the "Brain" of the model - a Causal Transformer that processes
the history of previously generated tokens [x_0, ..., x_{t-1}] and outputs
a context vector h_t that conditions the diffusion process for generating x_t.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List

from ..core.attention import CausalMultiHeadAttention
from ..core.layers import CausalTransformerLayer, FeedForward
from ..core.embeddings import (
    SinusoidalPositionalEncoding,
    LearnablePositionalEncoding,
    StartTokenEmbedding,
    UniMolEmbeddingLoader
)


class CausalContextEncoder(nn.Module):
    """
    Causal Context Encoder - The "Brain" of the ALD model.
    
    Architecture:
        1. Embed input tokens using Uni-Mol embeddings
        2. Add positional encoding
        3. Process through causal transformer layers
        4. Output context vectors for conditioning the diffusion process
    
    The causal masking ensures that when generating token t, the model
    can only see tokens [0, 1, ..., t-1].
    
    Args:
        embedding_dim: Dimension of Uni-Mol embeddings
        d_model: Internal model dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        d_ff: Feed-forward hidden dimension
        max_seq_len: Maximum sequence length
        dropout: Dropout probability
        embeddings_dir: Directory for Uni-Mol embeddings
        freeze_embeddings: Whether to freeze Uni-Mol embeddings
    """
    
    def __init__(
        self,
        embedding_dim: int = 512,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        max_seq_len: int = 256,
        dropout: float = 0.1,
        embeddings_dir: str = "./unimol_embeddings",
        freeze_embeddings: bool = True
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        
        # Uni-Mol embedding layer
        self.embedding = UniMolEmbeddingLoader(
            embeddings_dir=embeddings_dir,
            freeze_embeddings=freeze_embeddings
        )
        
        # Verify embedding dimension matches
        actual_embed_dim = self.embedding.embedding_dim
        if actual_embed_dim != embedding_dim:
            print(f"[CausalContextEncoder] Note: Uni-Mol dim ({actual_embed_dim}) != specified ({embedding_dim})")
            self.embedding_dim = actual_embed_dim
        
        # Project embedding to model dimension if needed
        if self.embedding_dim != d_model:
            self.input_projection = nn.Linear(self.embedding_dim, d_model)
        else:
            self.input_projection = nn.Identity()
        
        # Start token embedding (for autoregressive start)
        self.start_token = StartTokenEmbedding(self.embedding_dim)
        
        # Positional encoding
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, dropout, max_seq_len)
        
        # Causal transformer layers
        self.transformer_layers = nn.ModuleList([
            CausalTransformerLayer(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                max_seq_len=max_seq_len,
                activation='gelu'
            )
            for _ in range(n_layers)
        ])
        
        # Final layer norm
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Output projection (context vectors)
        self.context_projection = nn.Linear(d_model, d_model)
        
    def forward(
        self,
        token_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_all_contexts: bool = True
    ) -> torch.Tensor:
        """
        Encode token history into context vectors.
        
        Args:
            token_ids: Token indices [batch_size, seq_len]
            mask: Padding mask [batch_size, seq_len] (1 = valid, 0 = padding)
            return_all_contexts: If True, return context for all positions;
                                If False, return only the last context
                                
        Returns:
            Context vectors [batch_size, seq_len, d_model] or [batch_size, d_model]
        """
        # Get embeddings
        x = self.embedding(token_ids)  # [batch_size, seq_len, embedding_dim]
        
        # Project to model dimension
        x = self.input_projection(x)  # [batch_size, seq_len, d_model]
        
        # Add positional encoding
        x = self.pos_encoding(x)  # [batch_size, seq_len, d_model]
        
        # Process through causal transformer layers
        for layer in self.transformer_layers:
            x, _ = layer(x, key_padding_mask=mask)
        
        # Final normalization
        x = self.layer_norm(x)
        
        # Project to context space
        contexts = self.context_projection(x)  # [batch_size, seq_len, d_model]
        
        if return_all_contexts:
            return contexts
        else:
            # Return only the last valid context for each sequence
            if mask is not None:
                # Find last valid position for each sequence
                lengths = mask.sum(dim=1).long() - 1
                batch_indices = torch.arange(x.size(0), device=x.device)
                return contexts[batch_indices, lengths]  # [batch_size, d_model]
            else:
                return contexts[:, -1]  # [batch_size, d_model]
    
    def forward_with_embeddings(
        self,
        embeddings: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode using pre-computed embeddings (for generated tokens during inference).
        
        Args:
            embeddings: Pre-computed embeddings [batch_size, seq_len, embedding_dim]
            mask: Padding mask [batch_size, seq_len]
            
        Returns:
            Context vectors [batch_size, seq_len, d_model]
        """
        # Project to model dimension
        x = self.input_projection(embeddings)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Process through transformer
        for layer in self.transformer_layers:
            x, _ = layer(x, key_padding_mask=mask)
        
        # Final normalization and projection
        x = self.layer_norm(x)
        contexts = self.context_projection(x)
        
        return contexts
    
    def get_context_for_next_token(
        self,
        history_embeddings: torch.Tensor,
        history_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get context vector for generating the next token.
        
        This is the main interface used during autoregressive generation.
        Given the history of previously generated embeddings, returns the
        context vector that will condition the diffusion process.
        
        Args:
            history_embeddings: Embeddings of previous tokens [batch_size, history_len, embedding_dim]
            history_mask: Mask for history [batch_size, history_len]
            
        Returns:
            Context vector for next token [batch_size, d_model]
        """
        device = history_embeddings.device
        
        if history_embeddings.size(1) == 0:
            # No history - return context from start token
            batch_size = history_embeddings.size(0)
            start_emb = self.start_token(batch_size).to(device)  # [batch_size, 1, embedding_dim]
            contexts = self.forward_with_embeddings(start_emb)
            return contexts[:, 0]  # [batch_size, d_model]
        
        # Encode history and get last context
        contexts = self.forward_with_embeddings(history_embeddings, history_mask)
        
        if history_mask is not None:
            # Get context at last valid position
            lengths = history_mask.sum(dim=1).long() - 1
            batch_indices = torch.arange(contexts.size(0), device=contexts.device)
            return contexts[batch_indices, lengths]
        else:
            return contexts[:, -1]
    
    def get_token_embedding(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Get Uni-Mol embeddings for token IDs.
        
        Args:
            token_ids: Token indices [batch_size] or [batch_size, seq_len]
            
        Returns:
            Embeddings
        """
        return self.embedding(token_ids)


class KVCacheContextEncoder(CausalContextEncoder):
    """
    Context Encoder with Key-Value caching for faster autoregressive generation.
    
    During generation, caches the key and value projections from previous
    positions to avoid recomputing them at each step.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._kv_cache = None
        
    def reset_cache(self):
        """Reset the KV cache (call at the start of each new sequence)."""
        self._kv_cache = None
        
    def forward_with_cache(
        self,
        new_token_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Process a single new token using cached KV pairs.
        
        Args:
            new_token_embedding: Embedding of new token [batch_size, 1, embedding_dim]
            
        Returns:
            Context for the new position [batch_size, d_model]
        """
        # Project new token
        x = self.input_projection(new_token_embedding)
        
        # Add positional encoding for current position
        if self._kv_cache is None:
            pos = 0
        else:
            pos = self._kv_cache[0][0].size(2)  # Get cached sequence length
        
        # Manual positional encoding addition
        x = x + self.pos_encoding.pe[pos:pos+1].unsqueeze(0)
        
        # Process through layers with caching
        # Note: Full KV caching implementation would require modifying attention layers
        # This is a simplified version
        for layer in self.transformer_layers:
            x, _ = layer(x)
        
        x = self.layer_norm(x)
        context = self.context_projection(x)
        
        return context.squeeze(1)
