"""
Token Mapper - maps continuous embeddings to discrete HELM monomers
using nearest neighbor search with position-aware constraints.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List


class TokenMapper(nn.Module):
    """Maps diffusion embeddings to discrete monomer tokens via nearest neighbor search."""
    
    def __init__(
        self,
        vocab: Dict[str, int],
        embeddings_dir: str = "./unimol_embeddings",
        data_dir: str = "./data",
        use_embedding_norm: bool = True
    ):
        super().__init__()
        
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.idx_to_token = {v: k for k, v in vocab.items()}
        self.use_embedding_norm = use_embedding_norm
        
        self.embeddings_dir = Path(embeddings_dir)
        self.data_dir = Path(data_dir)
        
        self._load_embeddings()
        self._classify_monomers()
        
    def _load_embeddings(self) -> None:
        """Load the Uni-Mol embedding matrix."""
        embeddings_path = self.embeddings_dir / "embeddings_matrix.npy"
        embeddings_matrix = np.load(embeddings_path, allow_pickle=True)
        self.embedding_dim = embeddings_matrix.shape[1]
        
        # Add PAD embedding (zeros)
        full_embeddings = np.vstack([embeddings_matrix, np.zeros((1, self.embedding_dim))])
        self.register_buffer('reference_embeddings', torch.from_numpy(full_embeddings).float())
        print(f"[TokenMapper] Loaded embeddings: {self.reference_embeddings.shape}")
            
    def _classify_monomers(self) -> None:
        """Classify monomers by R1/R2 connectivity for position constraints."""
        self.class1_tokens: List[int] = []  # Has R2 (first)
        self.class2_tokens: List[int] = []  # Has R1 and R2 (middle)
        self.class3_tokens: List[int] = []  # Has R1 (last)
        
        try:
            monomer_path = self.data_dir / "monomer_library.csv"
            if monomer_path.exists():
                df = pd.read_csv(monomer_path)
                for _, row in df.iterrows():
                    symbol = row['Symbol']
                    if symbol not in self.vocab:
                        continue
                    token_id = self.vocab[symbol]
                    r1, r2 = str(row.get('R1', '-')).strip(), str(row.get('R2', '-')).strip()
                    has_r1 = r1 not in ('-', 'nan', '')
                    has_r2 = r2 not in ('-', 'nan', '')
                    
                    if has_r2: self.class1_tokens.append(token_id)
                    if has_r1 and has_r2: self.class2_tokens.append(token_id)
                    if has_r1: self.class3_tokens.append(token_id)
                
                print(f"[TokenMapper] Monomer classification:")
                print(f"  Class 1 (first): {len(self.class1_tokens)}")
                print(f"  Class 2 (middle): {len(self.class2_tokens)}")
                print(f"  Class 3 (last): {len(self.class3_tokens)}")
        except Exception as e:
            print(f"[TokenMapper] Warning: Could not classify monomers: {e}")
            all_tokens = list(range(self.vocab_size))
            self.class1_tokens = self.class2_tokens = self.class3_tokens = all_tokens
    
    def _compute_distances(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute distances between embeddings and reference embeddings."""
        ref_embs = self.reference_embeddings.to(embeddings.device)
        if self.use_embedding_norm:
            ref_norm = ref_embs / ref_embs.norm(dim=1, keepdim=True).clamp(min=1e-8)
            emb_norm = embeddings / embeddings.norm(dim=1, keepdim=True).clamp(min=1e-8)
            return 1.0 - torch.mm(emb_norm, ref_norm.t())
        return torch.cdist(embeddings, ref_embs)
    
    def _get_allowed_tokens(self, position: int, seq_len: int) -> List[int]:
        """Get allowed tokens based on position constraints."""
        if position == 0:
            return self.class1_tokens or list(range(self.vocab_size))
        elif position == seq_len - 1:
            return self.class3_tokens or list(range(self.vocab_size))
        return self.class2_tokens or list(range(self.vocab_size))
    
    def _select_token(self, distances: torch.Tensor, allowed: List[int]) -> int:
        """Select nearest token from allowed tokens."""
        idx = torch.argmin(distances[allowed]).item()
        return allowed[idx]
    
    def forward(
        self,
        embeddings: torch.Tensor,
        position: int,
        seq_len: int,
        allow_all: bool = False
    ) -> torch.Tensor:
        """Map embeddings to token IDs. [batch_size, embedding_dim] -> [batch_size]"""
        batch_size = embeddings.size(0)
        device = embeddings.device
        distances = self._compute_distances(embeddings)
        
        allowed = list(range(self.vocab_size)) if allow_all else self._get_allowed_tokens(position, seq_len)
        tokens = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        for b in range(batch_size):
            tokens[b] = self._select_token(distances[b], allowed)
        return tokens
    
    def batch_map(
        self,
        embeddings: torch.Tensor,
        positions: int,
        seq_lens: torch.Tensor
    ) -> torch.Tensor:
        """Batch map at same position with different target lengths. [batch_size, embedding_dim] -> [batch_size]"""
        batch_size = embeddings.size(0)
        device = embeddings.device
        distances = self._compute_distances(embeddings)
        tokens = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        for b in range(batch_size):
            allowed = self._get_allowed_tokens(positions, seq_lens[b].item())
            tokens[b] = self._select_token(distances[b], allowed)
        return tokens
    
    def get_embedding(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Get embeddings for token IDs (reverse lookup)."""
        return self.reference_embeddings[token_ids]
