"""
Token Mapper - maps continuous embeddings to discrete HELM monomers
using nearest neighbor search with position-aware constraints.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional


class TokenMapper(nn.Module):
    """Maps diffusion embeddings to discrete monomer tokens via nearest neighbor search."""
    
    def __init__(
        self,
        vocab: Dict[str, int],
        embeddings_dir: str = "./data/processed/unimol_embeddings",
        data_dir: str = "./data/processed",
        use_embedding_norm: bool = True,
        reference_embeddings: Optional[torch.Tensor] = None,
        allowed_token_ids: Optional[List[int]] = None,
    ):
        super().__init__()
        
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.idx_to_token = {v: k for k, v in vocab.items()}
        self.use_embedding_norm = use_embedding_norm
        if allowed_token_ids is None:
            allowed_token_ids = [
                token_id
                for token, token_id in vocab.items()
                if not self._is_special_token(token)
            ]
        self.unconstrained_tokens = list(dict.fromkeys(int(x) for x in allowed_token_ids))
        
        self.embeddings_dir = Path(embeddings_dir)
        self.data_dir = Path(data_dir)
        
        self._load_embeddings(reference_embeddings)
        self._classify_monomers()

    @staticmethod
    def _is_special_token(token: str) -> bool:
        """Return whether a vocabulary entry is a generation control token."""
        normalized = token.strip().upper()
        named_specials = {
            'PAD', 'BOS', 'EOS', 'UNK', 'MASK',
            '<PAD>', '<BOS>', '<EOS>', '<UNK>', '<MASK>',
            '[PAD]', '[BOS]', '[EOS]', '[UNK]', '[MASK]',
        }
        return (
            normalized in named_specials
            or (normalized.startswith('<') and normalized.endswith('>'))
        )
        
    def _load_embeddings(
        self, reference_embeddings: Optional[torch.Tensor] = None
    ) -> None:
        """Load the Uni-Mol embedding matrix."""
        if reference_embeddings is not None:
            if reference_embeddings.ndim != 2:
                raise ValueError(
                    "reference_embeddings must have shape [vocab_size, embedding_dim]"
                )
            if reference_embeddings.shape[0] != self.vocab_size:
                raise ValueError(
                    "Reference codebook/vocabulary size mismatch: "
                    f"{reference_embeddings.shape[0]} vs {self.vocab_size}"
                )
            self.embedding_dim = reference_embeddings.shape[1]
            self.register_buffer(
                'reference_embeddings',
                reference_embeddings.detach().clone(),
            )
            print(
                "[TokenMapper] Using shared ChemEmb codebook: "
                f"{self.reference_embeddings.shape}"
            )
            return

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
            all_tokens = self.unconstrained_tokens
            self.class1_tokens = self.class2_tokens = self.class3_tokens = all_tokens
    
    def _compute_distances(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute distances between embeddings and reference embeddings."""
        ref_embs = self.reference_embeddings.to(embeddings.device)
        if self.use_embedding_norm:
            ref_norm = ref_embs / ref_embs.norm(dim=1, keepdim=True).clamp(min=1e-8)
            emb_norm = embeddings / embeddings.norm(dim=1, keepdim=True).clamp(min=1e-8)
            return 1.0 - torch.mm(emb_norm, ref_norm.t())
        return torch.cdist(embeddings, ref_embs)
    
    def _get_allowed_tokens(
        self,
        position: int,
        seq_len: int,
        enforce_r1r2_constraints: bool = True,
    ) -> List[int]:
        """Get ordinary monomers, optionally masked by positional R1/R2 rules."""
        if not enforce_r1r2_constraints:
            return self.unconstrained_tokens
        if position == 0:
            return self.class1_tokens or self.unconstrained_tokens
        elif position == seq_len - 1:
            return self.class3_tokens or self.unconstrained_tokens
        return self.class2_tokens or self.unconstrained_tokens
    
    def _select_token(self, distances: torch.Tensor, allowed: List[int]) -> int:
        """Select nearest token from allowed tokens."""
        idx = torch.argmin(distances[allowed]).item()
        return allowed[idx]

    def _apply_frequency_penalty(
        self,
        scores: torch.Tensor,
        allowed_tensor: torch.Tensor,
        history_tokens: torch.Tensor,
        frequency_penalty: float
    ) -> torch.Tensor:
        """Discourage repeatedly sampling the same monomers within one peptide."""
        if history_tokens is None or frequency_penalty <= 0:
            return scores

        history_tokens = history_tokens[history_tokens >= 0]
        if history_tokens.numel() == 0:
            return scores

        counts = torch.bincount(history_tokens, minlength=self.vocab_size).float()
        penalties = counts[allowed_tensor].to(scores.device)
        return scores - frequency_penalty * penalties

    def sample_from_scores(
        self,
        scores: torch.Tensor,
        allowed: List[int],
        history_tokens: torch.Tensor = None,
        top_k: int = 8,
        top_p: float = 1.0,
        temperature: float = 1.0,
        frequency_penalty: float = 0.0,
    ) -> int:
        """
        Sample a token from precomputed per-vocab scores, where higher is better.
        """
        if len(allowed) == 0:
            return int(torch.argmax(scores).item())

        device = scores.device
        allowed_tensor = torch.tensor(allowed, dtype=torch.long, device=device)
        candidate_scores = scores[allowed_tensor]
        candidate_scores = self._apply_frequency_penalty(
            candidate_scores,
            allowed_tensor,
            history_tokens,
            frequency_penalty,
        )

        if top_k is not None and top_k > 0 and candidate_scores.numel() > top_k:
            top_scores, top_indices = torch.topk(candidate_scores, top_k)
            allowed_tensor = allowed_tensor[top_indices]
            candidate_scores = top_scores

        if top_p is not None and 0 < top_p < 1.0 and candidate_scores.numel() > 1:
            sorted_scores, sorted_indices = torch.sort(candidate_scores, descending=True)
            logits = sorted_scores / max(temperature, 1e-6)
            probs = torch.softmax(logits, dim=0)
            cumulative = torch.cumsum(probs, dim=0)
            keep_mask = cumulative <= top_p
            keep_mask[0] = True
            allowed_tensor = allowed_tensor[sorted_indices[keep_mask]]
            candidate_scores = sorted_scores[keep_mask]

        if candidate_scores.numel() == 1:
            return int(allowed_tensor[0].item())

        if temperature is None or temperature <= 1e-6:
            return int(allowed_tensor[torch.argmax(candidate_scores)].item())

        probs = torch.softmax(candidate_scores / temperature, dim=0)
        sampled_idx = torch.multinomial(probs, num_samples=1)
        return int(allowed_tensor[sampled_idx].item())
    
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
        
        allowed = (
            self.unconstrained_tokens
            if allow_all
            else self._get_allowed_tokens(position, seq_len)
        )
        tokens = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        for b in range(batch_size):
            tokens[b] = self._select_token(distances[b], allowed)
        return tokens
    
    def batch_map(
        self,
        embeddings: torch.Tensor,
        positions: int,
        seq_lens: torch.Tensor,
        enforce_r1r2_constraints: bool = True,
    ) -> torch.Tensor:
        """Batch map at same position with different target lengths. [batch_size, embedding_dim] -> [batch_size]"""
        batch_size = embeddings.size(0)
        device = embeddings.device
        distances = self._compute_distances(embeddings)
        tokens = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        for b in range(batch_size):
            allowed = self._get_allowed_tokens(
                positions,
                seq_lens[b].item(),
                enforce_r1r2_constraints=enforce_r1r2_constraints,
            )
            tokens[b] = self._select_token(distances[b], allowed)
        return tokens

    def batch_sample(
        self,
        embeddings: torch.Tensor,
        positions: int,
        seq_lens: torch.Tensor,
        token_histories: torch.Tensor = None,
        top_k: int = 8,
        top_p: float = 1.0,
        temperature: float = 1.0,
        frequency_penalty: float = 0.0,
        enforce_r1r2_constraints: bool = True,
    ) -> torch.Tensor:
        """Sample tokens from top-ranked neighbors instead of greedy nearest-neighbor decode."""
        batch_size = embeddings.size(0)
        device = embeddings.device
        distances = self._compute_distances(embeddings)
        scores = -distances
        tokens = torch.zeros(batch_size, dtype=torch.long, device=device)

        for b in range(batch_size):
            allowed = self._get_allowed_tokens(
                positions,
                seq_lens[b].item(),
                enforce_r1r2_constraints=enforce_r1r2_constraints,
            )
            history = None if token_histories is None else token_histories[b]
            tokens[b] = self.sample_from_scores(
                scores[b],
                allowed,
                history_tokens=history,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                frequency_penalty=frequency_penalty,
            )
        return tokens
    
    def get_embedding(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Get embeddings for token IDs (reverse lookup)."""
        return self.reference_embeddings[token_ids]
