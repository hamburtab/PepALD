"""Constraint-only monomer selection for the LM-only ablation."""

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
import torch.nn as nn


class TokenConstraintSampler(nn.Module):
    """Apply the existing positional monomer constraints to LM logits.

    Unlike :class:`TokenMapper`, this module has no reference embedding
    codebook and exposes no nearest-neighbor/distance mapping path.
    """

    def __init__(self, vocab: Dict[str, int], data_dir: str = "./data/processed"):
        super().__init__()
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.idx_to_token = {v: k for k, v in vocab.items()}
        self.data_dir = Path(data_dir)
        self._classify_monomers()

    def _classify_monomers(self) -> None:
        """Classify monomers using the same R1/R2 rules as TokenMapper."""
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
                    r1 = str(row.get('R1', '-')).strip()
                    r2 = str(row.get('R2', '-')).strip()
                    has_r1 = r1 not in ('-', 'nan', '')
                    has_r2 = r2 not in ('-', 'nan', '')

                    if has_r2:
                        self.class1_tokens.append(token_id)
                    if has_r1 and has_r2:
                        self.class2_tokens.append(token_id)
                    if has_r1:
                        self.class3_tokens.append(token_id)

                print("[TokenConstraintSampler] Monomer classification:")
                print(f"  Class 1 (first): {len(self.class1_tokens)}")
                print(f"  Class 2 (middle): {len(self.class2_tokens)}")
                print(f"  Class 3 (last): {len(self.class3_tokens)}")
        except Exception as exc:
            print(f"[TokenConstraintSampler] Warning: Could not classify monomers: {exc}")
            all_tokens = list(range(self.vocab_size))
            self.class1_tokens = all_tokens
            self.class2_tokens = all_tokens
            self.class3_tokens = all_tokens

    def _get_allowed_tokens(self, position: int, seq_len: int) -> List[int]:
        """Get the same admissible token set used by TokenMapper."""
        if position == 0:
            return self.class1_tokens or list(range(self.vocab_size))
        if position == seq_len - 1:
            return self.class3_tokens or list(range(self.vocab_size))
        return self.class2_tokens or list(range(self.vocab_size))

    def _apply_frequency_penalty(
        self,
        scores: torch.Tensor,
        allowed_tensor: torch.Tensor,
        history_tokens: Optional[torch.Tensor],
        frequency_penalty: float,
    ) -> torch.Tensor:
        """Apply the same repeated-token penalty used by TokenMapper."""
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
        history_tokens: Optional[torch.Tensor] = None,
        top_k: int = 8,
        top_p: float = 1.0,
        temperature: float = 1.0,
        frequency_penalty: float = 0.0,
    ) -> int:
        """Sample an admissible token directly from LM scores."""
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
