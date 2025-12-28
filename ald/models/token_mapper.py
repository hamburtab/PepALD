"""
Token Mapper for the ALD architecture.

Maps continuous embeddings from the diffusion process to discrete HELM monomers
using nearest neighbor search with cosine similarity.

The logic is preserved from helm_diffusion.py's _embedding_to_tokens method.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List, Set, Tuple


class TokenMapper(nn.Module):
    """
    Maps continuous embeddings to discrete HELM monomer tokens.
    
    Uses cosine similarity (or L2 distance) to find the nearest monomer
    embedding from the Uni-Mol embedding matrix.
    
    Features:
        - Position-aware constraints (R1/R2 based monomer classification)
        - Frequency-based weighting for low-frequency monomers
        - Monomer blacklist filtering
        - Temperature-based sampling
    
    Args:
        vocab: Dictionary mapping monomer symbols to indices
        embeddings_dir: Directory containing Uni-Mol embeddings
        data_dir: Directory containing vocab and frequency data
        use_embedding_norm: Use L2 normalization (cosine similarity)
        use_freq_weight: Apply frequency-based penalties
        use_temperature_sampling: Enable probabilistic sampling
        temperature: Sampling temperature (higher = more random)
        freq_weight_scale: Scale for frequency penalty
        use_blacklist: Filter out blacklisted monomers
    """
    
    def __init__(
        self,
        vocab: Dict[str, int],
        embeddings_dir: str = "./unimol_embeddings",
        data_dir: str = "./data",
        use_embedding_norm: bool = True,
        use_freq_weight: bool = False,
        use_temperature_sampling: bool = False,
        temperature: float = 0.0,
        freq_weight_scale: float = 0.1,
        use_blacklist: bool = False
    ):
        super().__init__()
        
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.idx_to_token = {v: k for k, v in vocab.items()}
        
        self.use_embedding_norm = use_embedding_norm
        self.use_freq_weight = use_freq_weight
        self.use_temperature_sampling = use_temperature_sampling
        self.temperature = temperature
        self.freq_weight_scale = freq_weight_scale
        self.use_blacklist = use_blacklist
        
        self.embeddings_dir = Path(embeddings_dir)
        self.data_dir = Path(data_dir)
        
        # Load embedding matrix
        self._load_embeddings()
        
        # Load frequency scores
        self._load_frequency_scores()
        
        # Load blacklist
        self._load_blacklist()
        
        # Classify monomers by R1/R2 connectivity
        self._classify_monomers()
        
        # Statistics tracking
        self.monomer_min_distances: Dict[int, List[float]] = {}
        self.global_min_distances: List[float] = []
        
    def _load_embeddings(self) -> None:
        """Load the Uni-Mol embedding matrix."""
        embeddings_path = self.embeddings_dir / "embeddings_matrix.npy"
        embeddings_matrix = np.load(embeddings_path, allow_pickle=True)
        
        self.embedding_dim = embeddings_matrix.shape[1]
        
        # Add PAD embedding (zeros)
        pad_embedding = np.zeros((1, self.embedding_dim))
        full_embeddings = np.vstack([embeddings_matrix, pad_embedding])
        
        # Register as buffer (not a parameter)
        self.register_buffer(
            'reference_embeddings',
            torch.from_numpy(full_embeddings).float()
        )
        
        print(f"[TokenMapper] Loaded embeddings: {self.reference_embeddings.shape}")
        
    def _load_frequency_scores(self) -> None:
        """Load monomer frequency scores for weighting."""
        self._freq_score_array = None
        
        try:
            freq_path = self.data_dir / "monomer_frequency.json"
            if freq_path.exists():
                with open(freq_path, 'r') as f:
                    freq = json.load(f)
                
                total = sum(freq.values()) if freq else 1
                
                arr = []
                for i in range(self.vocab_size):
                    token = self.idx_to_token.get(i)
                    if token is None:
                        arr.append(0.0)
                    else:
                        c = float(freq.get(token, 0))
                        arr.append(c / total)
                
                probs = np.array(arr, dtype=float)
                if probs.sum() > 0:
                    probs = probs / probs.max()
                
                self._freq_score_array = probs
                print(f"[TokenMapper] Loaded frequency scores for {len(arr)} monomers")
        except Exception as e:
            print(f"[TokenMapper] Warning: Could not load frequency scores: {e}")
            
    def _load_blacklist(self) -> None:
        """Load monomer blacklist."""
        self._blacklist_ids: Set[int] = set()
        
        try:
            blacklist_path = self.data_dir / "monomer_blacklist.json"
            if blacklist_path.exists():
                with open(blacklist_path, 'r') as f:
                    blacklist_data = json.load(f)
                
                for symbol in blacklist_data.get('blacklist', []):
                    if symbol in self.vocab:
                        self._blacklist_ids.add(self.vocab[symbol])
                
                print(f"[TokenMapper] Loaded {len(self._blacklist_ids)} blacklisted monomers")
        except Exception as e:
            print(f"[TokenMapper] Warning: Could not load blacklist: {e}")
            
    def _classify_monomers(self) -> None:
        """
        Classify monomers by R1/R2 connectivity for position constraints.
        
        Classes:
            - Class 1: Has R2 (can be first position)
            - Class 2: Has R1 AND R2 (can be middle positions)
            - Class 3: Has R1 (can be last position)
        """
        self.class1_tokens: List[int] = []  # Has R2 (first position)
        self.class2_tokens: List[int] = []  # Has R1 and R2 (middle positions)
        self.class3_tokens: List[int] = []  # Has R1 (last position)
        
        try:
            monomer_path = self.data_dir / "monomer_library.csv"
            if monomer_path.exists():
                df = pd.read_csv(monomer_path)
                
                for _, row in df.iterrows():
                    symbol = row['Symbol']
                    if symbol not in self.vocab:
                        continue
                    
                    token_id = self.vocab[symbol]
                    r1 = str(row.get('R1', '-')).strip()
                    r2 = str(row.get('R2', '-')).strip()
                    
                    has_r1 = (r1 != '-' and r1 != 'nan' and r1 != '')
                    has_r2 = (r2 != '-' and r2 != 'nan' and r2 != '')
                    
                    if has_r2:
                        self.class1_tokens.append(token_id)
                    if has_r1 and has_r2:
                        self.class2_tokens.append(token_id)
                    if has_r1:
                        self.class3_tokens.append(token_id)
                
                self.class1_tokens = sorted(self.class1_tokens)
                self.class2_tokens = sorted(self.class2_tokens)
                self.class3_tokens = sorted(self.class3_tokens)
                
                print(f"[TokenMapper] Monomer classification:")
                print(f"  Class 1 (first): {len(self.class1_tokens)}")
                print(f"  Class 2 (middle): {len(self.class2_tokens)}")
                print(f"  Class 3 (last): {len(self.class3_tokens)}")
        except Exception as e:
            print(f"[TokenMapper] Warning: Could not classify monomers: {e}")
            # Fallback: all tokens allowed at all positions
            all_tokens = list(range(self.vocab_size))
            self.class1_tokens = all_tokens
            self.class2_tokens = all_tokens
            self.class3_tokens = all_tokens
    
    def forward(
        self,
        embeddings: torch.Tensor,
        position: int,
        seq_len: int,
        allow_all: bool = False
    ) -> torch.Tensor:
        """
        Map a single embedding to a token ID.
        
        Args:
            embeddings: Embedding vector [batch_size, embedding_dim]
            position: Current position in sequence (0-indexed)
            seq_len: Target sequence length
            allow_all: If True, ignore position constraints
            
        Returns:
            Token IDs [batch_size]
        """
        batch_size = embeddings.size(0)
        device = embeddings.device
        
        # Get reference embeddings
        ref_embs = self.reference_embeddings.to(device)
        
        # Compute distances
        if self.use_embedding_norm:
            # Cosine similarity via L2 normalized embeddings
            ref_norms = ref_embs.norm(dim=1, keepdim=True).clamp(min=1e-8)
            ref_normalized = ref_embs / ref_norms
            
            emb_norms = embeddings.norm(dim=1, keepdim=True).clamp(min=1e-8)
            emb_normalized = embeddings / emb_norms
            
            cosine_sim = torch.mm(emb_normalized, ref_normalized.t())
            distances = 1.0 - cosine_sim  # [batch_size, vocab_size+1]
        else:
            # L2 distance
            distances = torch.cdist(embeddings, ref_embs)
        
        # Determine allowed tokens based on position
        if allow_all:
            allowed_tokens = list(range(self.vocab_size))
        elif position == 0:
            allowed_tokens = self.class1_tokens
        elif position == seq_len - 1:
            allowed_tokens = self.class3_tokens
        else:
            allowed_tokens = self.class2_tokens
        
        # Apply blacklist
        if self.use_blacklist:
            allowed_tokens = [t for t in allowed_tokens if t not in self._blacklist_ids]
        
        # Fallback if all filtered out
        if not allowed_tokens:
            allowed_tokens = list(range(self.vocab_size))
        
        # Get distances for allowed tokens
        allowed_distances = distances[:, allowed_tokens]  # [batch_size, num_allowed]
        
        # Apply frequency weighting
        if self.use_freq_weight and self._freq_score_array is not None:
            freq_tensor = torch.from_numpy(self._freq_score_array).to(device).float()
            freq_vals = freq_tensor[allowed_tokens]
            penalties = self.freq_weight_scale * (1.0 - freq_vals)
            allowed_distances = allowed_distances + penalties.unsqueeze(0)
        
        # Select tokens
        tokens = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        for b in range(batch_size):
            if self.use_temperature_sampling and self.temperature > 0:
                # Temperature-based sampling
                scores = -allowed_distances[b] / self.temperature
                scores = scores - scores.max()
                probs = F.softmax(scores, dim=0)
                try:
                    idx = torch.multinomial(probs, num_samples=1).item()
                except RuntimeError:
                    idx = torch.argmin(allowed_distances[b]).item()
                tokens[b] = allowed_tokens[idx]
            else:
                # Argmin selection
                idx = torch.argmin(allowed_distances[b]).item()
                tokens[b] = allowed_tokens[idx]
        
        return tokens
    
    def map_sequence(
        self,
        embeddings: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Map a sequence of embeddings to token IDs.
        
        Args:
            embeddings: [batch_size, seq_len, embedding_dim]
            lengths: Target lengths [batch_size]
            
        Returns:
            Token IDs [batch_size, seq_len]
        """
        batch_size, seq_len, _ = embeddings.shape
        device = embeddings.device
        
        tokens = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
        pad_id = self.vocab.get('<PAD>', self.vocab_size - 1)
        
        for b in range(batch_size):
            target_len = lengths[b].item() if lengths is not None else seq_len
            
            for pos in range(seq_len):
                if pos >= target_len:
                    tokens[b, pos] = pad_id
                else:
                    emb = embeddings[b, pos:pos+1]  # [1, embedding_dim]
                    token = self.forward(emb, position=pos, seq_len=target_len)
                    tokens[b, pos] = token[0]
        
        return tokens
    
    def get_embedding(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Get embeddings for token IDs (reverse lookup).
        
        Args:
            token_ids: Token indices [batch_size] or [batch_size, seq_len]
            
        Returns:
            Embeddings
        """
        return self.reference_embeddings[token_ids]


class FastTokenMapper(nn.Module):
    """
    Simplified fast token mapper without position constraints.
    
    For scenarios where speed is more important than constraint satisfaction.
    """
    
    def __init__(
        self,
        vocab: Dict[str, int],
        embeddings_dir: str = "./unimol_embeddings"
    ):
        super().__init__()
        
        self.vocab = vocab
        self.vocab_size = len(vocab)
        
        # Load embeddings
        embeddings_path = Path(embeddings_dir) / "embeddings_matrix.npy"
        embeddings_matrix = np.load(embeddings_path, allow_pickle=True)
        
        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
        normalized = embeddings_matrix / np.clip(norms, 1e-8, None)
        
        self.register_buffer(
            'normalized_embeddings',
            torch.from_numpy(normalized).float()
        )
        
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Fast nearest neighbor lookup.
        
        Args:
            embeddings: [batch_size, embedding_dim]
            
        Returns:
            Token IDs [batch_size]
        """
        # Normalize query
        norms = embeddings.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        normalized = embeddings / norms
        
        # Cosine similarity
        similarities = torch.mm(normalized, self.normalized_embeddings.t())
        
        # Argmax
        return similarities.argmax(dim=-1)
