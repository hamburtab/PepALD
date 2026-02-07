"""
Embedding modules for the ALD architecture.

Contains:
    - SinusoidalPositionalEncoding: Fixed sinusoidal position embeddings
    - LearnablePositionalEncoding: Learnable position embeddings
    - DiffusionTimeEmbedding: Embedding for diffusion timesteps
    - UniMolEmbeddingLoader: Loader for pre-trained Uni-Mol monomer embeddings
"""

import torch
import torch.nn as nn
import numpy as np
import json
import math
from pathlib import Path
from typing import Optional, Dict, Tuple


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding as described in "Attention Is All You Need".
    
    Uses sine and cosine functions of different frequencies to encode positions.
    This encoding is fixed (not learned) and allows the model to extrapolate
    to longer sequences than seen during training.
    
    Args:
        d_model: Model dimension
        dropout: Dropout probability
        max_len: Maximum sequence length
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Compute the div term: exp(2i * -log(10000) / d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        # Apply sin to even indices, cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter, but should be saved with model)
        # Shape: [max_len, d_model]
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            Output with positional encoding added [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        # Add positional encoding and apply dropout
        x = x + self.pe[:seq_len].unsqueeze(0)
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    """
    Learnable positional encoding.
    
    Uses learned embeddings for each position, allowing the model to learn
    optimal position representations for the specific task.
    
    Args:
        d_model: Model dimension
        max_len: Maximum sequence length
        dropout: Dropout probability
    """
    
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        
        # Learnable position embeddings
        self.pe = nn.Embedding(max_len, d_model)
        
        # Initialize with small values
        nn.init.normal_(self.pe.weight, mean=0.0, std=0.02)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add learnable positional encoding to input.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            Output with positional encoding added
        """
        batch_size, seq_len, _ = x.size()
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        x = x + self.pe(positions)
        return self.dropout(x)


class DiffusionTimeEmbedding(nn.Module):
    """
    Embedding for diffusion timesteps.
    
    Converts scalar timestep values into high-dimensional embeddings
    that can be used to condition the denoising network.
    
    Uses sinusoidal encoding followed by MLP projection, similar to
    the approach in DDPM and subsequent diffusion models.
    
    Args:
        time_dim: Dimension of the sinusoidal encoding
        embed_dim: Output embedding dimension
    """
    
    def __init__(self, time_dim: int = 128, embed_dim: int = 512):
        super().__init__()
        self.time_dim = time_dim
        self.embed_dim = embed_dim
        
        # MLP to project sinusoidal features to embedding dimension
        self.mlp = nn.Sequential(
            nn.Linear(time_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        
    def _sinusoidal_embedding(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Create sinusoidal embeddings for timesteps.
        
        Args:
            timesteps: Timestep values [batch_size] or [batch_size, 1]
            
        Returns:
            Sinusoidal embeddings [batch_size, time_dim]
        """
        if timesteps.dim() == 2:
            timesteps = timesteps.squeeze(-1)
            
        half_dim = self.time_dim // 2
        
        # Compute frequencies
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half_dim, device=timesteps.device) / half_dim
        )
        
        # Compute angles
        args = timesteps.unsqueeze(-1).float() * freqs.unsqueeze(0)
        
        # Concatenate sin and cos
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        
        return embedding
        
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Embed diffusion timesteps.
        
        Args:
            timesteps: Timestep values [batch_size] (normalized to [0, 1] or integer steps)
            
        Returns:
            Time embeddings [batch_size, embed_dim]
        """
        sinusoidal_emb = self._sinusoidal_embedding(timesteps)
        return self.mlp(sinusoidal_emb)


class UniMolEmbeddingLoader(nn.Module):
    """
    Loader and wrapper for pre-trained Uni-Mol monomer embeddings.
    
    Loads pre-computed Uni-Mol embeddings for HELM monomers and provides
    an interface compatible with standard PyTorch embedding layers.
    
    Fusion formula: CLS + r_weight * (R1 + R2 + R3)
    
    Args:
        embeddings_dir: Directory containing full_embeddings.npy and metadata.json
        freeze_embeddings: Whether to freeze the embeddings (not update during training)
        r_weight: Weight for R-group embeddings (default 0.3, adjust as needed)
    """
    
    def __init__(
        self,
        embeddings_dir: str = "./unimol_embeddings",
        freeze_embeddings: bool = True,
        r_weight: float = 0.0  # 可调整的权重
    ):
        super().__init__()
        
        self.embeddings_dir = Path(embeddings_dir)
        self.freeze_embeddings = freeze_embeddings
        self.r_weight = r_weight
        
        self._load_embeddings()
        
    def _load_embeddings(self) -> None:
        """Load pre-trained embedding matrix and metadata."""
        # Load metadata
        metadata_path = self.embeddings_dir / "metadata.json"
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Load full_embeddings: (N, 4, 512) = [CLS, R1, R2, R3]
        embeddings_path = self.embeddings_dir / "full_embeddings.npy"
        embeddings_matrix = np.load(embeddings_path, allow_pickle=True)  # (N, 4, 512)
        
        self.num_monomers = embeddings_matrix.shape[0]
        self.embedding_dim = embeddings_matrix.shape[2]  # 512
        
        # Convert to PyTorch tensor
        embeddings_tensor = torch.from_numpy(embeddings_matrix).float()  # (N, 4, 512)
        
        # Add PAD token embedding (zero vector)
        pad_embedding = torch.zeros(1, 4, self.embedding_dim)  # (1, 4, 512)
        full_embeddings = torch.cat([embeddings_tensor, pad_embedding], dim=0)  # (N+1, 4, 512)
        
        # Register as buffer (not nn.Embedding since shape is 3D)
        self.register_buffer('_embeddings', full_embeddings)
        
        self.vocab_size = self.num_monomers + 1
        self.pad_idx = self.num_monomers
        
        print(f"[UniMolEmbeddingLoader] Loaded embeddings:")
        print(f"  - Embedding dim: {self.embedding_dim}")
        print(f"  - Num monomers: {self.num_monomers}")
        print(f"  - Vocab size: {self.vocab_size} (including <PAD>)")
        print(f"  - Frozen: {self.freeze_embeddings}")
        print(f"  - Fusion: CLS + {self.r_weight} * (R1 + R2 + R3)")
    
    def _fuse(self, full: torch.Tensor) -> torch.Tensor:
        """加权融合: CLS + r_weight * (R1 + R2 + R3)"""
        cls_vec = full[..., 0, :]   # CLS
        r_sum = full[..., 1:, :].sum(dim=-2)  # R1 + R2 + R3
        return cls_vec + self.r_weight * r_sum
        
    def forward(
        self, 
        input_ids: torch.Tensor, 
        return_r_groups: bool = False
    ) -> torch.Tensor:
        """
        Get embeddings for input token IDs.
        
        Args:
            input_ids: Token indices [batch_size, seq_len] or [batch_size]
            return_r_groups: If True, also return R1/R2/R3 embeddings separately
            
        Returns:
            If return_r_groups=False:
                Embeddings [batch_size, seq_len, embedding_dim]
            If return_r_groups=True:
                Tuple of:
                    - cls_emb: [batch_size, seq_len, embedding_dim] (fused embedding)
                    - r_emb: [batch_size, seq_len, 3, embedding_dim] (R1, R2, R3)
        """
        full = self._embeddings[input_ids]  # (B, L, 4, 512) or (B, 4, 512)
        cls_emb = self._fuse(full)
        
        if return_r_groups:
            r_emb = full[..., 1:, :]  # (B, L, 3, 512) or (B, 3, 512)
            return cls_emb, r_emb
        
        return cls_emb
    
    def get_r_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Get only R1/R2/R3 embeddings for ring bond prediction.
        
        Args:
            input_ids: Token indices [batch_size, seq_len]
            
        Returns:
            R-group embeddings [batch_size, seq_len, 3, embedding_dim]
        """
        full = self._embeddings[input_ids]  # (B, L, 4, 512)
        return full[..., 1:, :]  # (B, L, 3, 512)
    
    def get_all_embeddings(self) -> torch.Tensor:
        """
        Get the full embedding matrix (excluding PAD), fused.
        
        Returns:
            Embedding matrix [num_monomers, embedding_dim]
        """
        return self._fuse(self._embeddings[:self.num_monomers])
    
    def get_embedding_for_token(self, token_id: int) -> torch.Tensor:
        """
        Get embedding for a single token, fused.
        
        Args:
            token_id: Token index
            
        Returns:
            Embedding vector [embedding_dim]
        """
        return self._fuse(self._embeddings[token_id])


class StartTokenEmbedding(nn.Module):
    """
    Learnable embedding for the start-of-sequence token.
    
    Used to initialize the autoregressive generation process.
    
    Args:
        embedding_dim: Dimension of the embedding
    """
    
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Learnable start token embedding
        self.start_embedding = nn.Parameter(torch.randn(1, embedding_dim) * 0.02)
        
    def forward(self, batch_size: int) -> torch.Tensor:
        """
        Get start token embeddings for a batch.
        
        Args:
            batch_size: Number of sequences in batch
            
        Returns:
            Start embeddings [batch_size, 1, embedding_dim]
        """
        return self.start_embedding.unsqueeze(0).expand(batch_size, 1, -1)
