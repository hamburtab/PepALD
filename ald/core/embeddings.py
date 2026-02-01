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
    Loader for pre-trained Uni-Mol monomer embeddings with R-group fusion.
    
    Loads (N, 4, 512) embeddings: [CLS, R1, R2, R3] and fuses them via MLP.
    Final output = CLS + MLP(concat[R1, R2, R3])
    
    Args:
        embeddings_dir: Directory containing full_embeddings.npy and metadata.json
        freeze_embeddings: Whether to freeze the base embeddings
    """
    
    def __init__(
        self,
        embeddings_dir: str = "./unimol_embeddings",
        freeze_embeddings: bool = True
    ):
        super().__init__()
        
        self.embeddings_dir = Path(embeddings_dir)
        self.freeze_embeddings = freeze_embeddings
        
        self._load_embeddings()
        self._build_fusion_mlp()
        
    def _load_embeddings(self) -> None:
        """Load pre-trained embedding matrix [N, 4, 512]."""
        metadata_path = self.embeddings_dir / "metadata.json"
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # 优先加载 full_embeddings.npy，回退到 embeddings_matrix.npy
        full_path = self.embeddings_dir / "full_embeddings.npy"
        legacy_path = self.embeddings_dir / "embeddings_matrix.npy"
        
        if full_path.exists():
            embeddings_matrix = np.load(full_path)  # (N, 4, 512)
            self.use_fusion = True
        else:
            embeddings_matrix = np.load(legacy_path)  # (N, 512)
            self.use_fusion = False
        
        self.num_monomers = embeddings_matrix.shape[0]
        self.embedding_dim = embeddings_matrix.shape[-1]  # 最后一维是 512
        
        # 添加 PAD token (全零)
        if self.use_fusion:
            pad_embedding = np.zeros((1, 4, self.embedding_dim))
            embeddings_matrix = np.concatenate([embeddings_matrix, pad_embedding], axis=0)
        else:
            pad_embedding = np.zeros((1, self.embedding_dim))
            embeddings_matrix = np.concatenate([embeddings_matrix, pad_embedding], axis=0)
        
        # 注册为 buffer（不参与梯度，但会保存到 checkpoint）
        self.register_buffer('embeddings', torch.from_numpy(embeddings_matrix).float())
        
        self.vocab_size = self.num_monomers + 1
        self.pad_idx = self.num_monomers
        
        print(f"[UniMolEmbeddingLoader] Loaded embeddings:")
        print(f"  - Shape: {self.embeddings.shape}")
        print(f"  - Fusion mode: {self.use_fusion}")
        print(f"  - Vocab size: {self.vocab_size} (including <PAD>)")
        
    def _build_fusion_mlp(self) -> None:
        """Build MLP for fusing R1, R2, R3 features."""
        if self.use_fusion:
            # MLP: concat(R1, R2, R3) -> embedding_dim
            self.r_group_mlp = nn.Sequential(
                nn.Linear(self.embedding_dim * 3, self.embedding_dim),
                nn.GELU(),
                nn.Linear(self.embedding_dim, self.embedding_dim),
            )
        else:
            self.r_group_mlp = None
        
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Get fused embeddings for input token IDs.
        
        Args:
            input_ids: [batch_size, seq_len] or [batch_size]
            
        Returns:
            Fused embeddings with same shape + embedding_dim
        """
        # 查表获取 embedding
        raw_emb = self.embeddings[input_ids]  # (..., 4, 512) or (..., 512)
        
        if not self.use_fusion:
            return raw_emb
        
        # 分离 CLS 和 R 基团
        cls_vec = raw_emb[..., 0, :]   # (..., 512)
        r1_vec = raw_emb[..., 1, :]
        r2_vec = raw_emb[..., 2, :]
        r3_vec = raw_emb[..., 3, :]
        
        # 融合: CLS + MLP(concat[R1, R2, R3])
        r_concat = torch.cat([r1_vec, r2_vec, r3_vec], dim=-1)  # (..., 1536)
        r_features = self.r_group_mlp(r_concat)                  # (..., 512)
        
        return cls_vec + r_features
    
    def get_all_embeddings(self) -> torch.Tensor:
        """Get fused embedding matrix (excluding PAD)."""
        all_ids = torch.arange(self.num_monomers, device=self.embeddings.device)
        return self.forward(all_ids)
    
    def get_embedding_for_token(self, token_id: int) -> torch.Tensor:
        """Get fused embedding for a single token."""
        return self.forward(torch.tensor([token_id], device=self.embeddings.device)).squeeze(0)


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
