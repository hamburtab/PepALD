"""
Causal Context Encoder - processes token history and outputs context vectors
for conditioning the diffusion process.
"""

import torch
import torch.nn as nn
from typing import Optional

from ..core.layers import CausalTransformerLayer
from ..core.embeddings import (
    SinusoidalPositionalEncoding,
    StartTokenEmbedding,
    UniMolEmbeddingLoader
)


class CausalContextEncoder(nn.Module):
    """
    Encodes token history [x_0, ..., x_{t-1}] into context vectors h_t
    that condition the diffusion process for generating x_t.
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
        embeddings_dir: str = "./data/processed/unimol_embeddings",
        freeze_embeddings: bool = True
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Uni-Mol embedding layer
        self.embedding = UniMolEmbeddingLoader(
            embeddings_dir=embeddings_dir,
            freeze_embeddings=freeze_embeddings
        )
        self.embedding_dim = self.embedding.embedding_dim
        
        # Project embedding to model dimension if needed
        self.input_projection = (
            nn.Linear(self.embedding_dim, d_model)
            if self.embedding_dim != d_model else nn.Identity()
        )
        
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
        
        # Final layer norm and context projection
        self.layer_norm = nn.LayerNorm(d_model)
        self.context_projection = nn.Linear(d_model, d_model)
    
    def _encode(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Core encoding: projection -> pos_enc -> transformer -> norm -> projection."""
        x = self.input_projection(x)
        x = self.pos_encoding(x)
        for layer in self.transformer_layers:
            x, _ = layer(x, key_padding_mask=mask)
        x = self.layer_norm(x)
        return self.context_projection(x)
    
    def _get_last_context(self, contexts: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Get context at last valid position for each sequence."""
        if mask is not None:
            lengths = mask.sum(dim=1).long() - 1
            batch_indices = torch.arange(contexts.size(0), device=contexts.device)
            return contexts[batch_indices, lengths]
        return contexts[:, -1]
        
    def forward(
        self,
        token_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_all_contexts: bool = True
    ) -> torch.Tensor:
        """
        Encode token history into context vectors.
        
        Args:
            token_ids: [batch_size, seq_len]
            mask: [batch_size, seq_len] (1=valid, 0=pad)
            return_all_contexts: If False, return only last context
        """
        x = self.embedding(token_ids)
        contexts = self._encode(x, mask)
        
        if return_all_contexts:
            return contexts
        return self._get_last_context(contexts, mask)
    
    def forward_with_embeddings(
        self,
        embeddings: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Encode using pre-computed embeddings (for inference)."""
        return self._encode(embeddings, mask)
    
    def get_context_for_next_token(
        self,
        history_embeddings: torch.Tensor,
        history_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get context vector for generating the next token."""
        device = history_embeddings.device
        
        if history_embeddings.size(1) == 0:
            # No history - use start token
            batch_size = history_embeddings.size(0)
            start_emb = self.start_token(batch_size).to(device)
            return self._encode(start_emb)[:, 0]
        
        contexts = self._encode(history_embeddings, history_mask)
        return self._get_last_context(contexts, history_mask)
    
    def get_token_embedding(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Get Uni-Mol embeddings for token IDs."""
        return self.embedding(token_ids)
