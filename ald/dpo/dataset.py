"""
Preference Pair Dataset for Diffusion-DPO training.

数据构造流程:
    1. 用 pretrained model 生成 N 条 HELM 序列
    2. 对每条计算 reward = w1 * Vina_score + w2 * Perm_score
    3. 取 top-25% 为 winner, bottom-25% 为 loser
    4. 随机配对构成 (winner, loser) pairs
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Any, Optional
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ald.utils.topology import HELMTopologyAnalyzer


class PreferencePairDataset(Dataset):
    """
    Dataset of (winner, loser) preference pairs for DPO training.

    Each item returns:
        token_ids_w: [L]    好样本的 token id (padded)
        mask_w:      [L]    好样本的 valid mask
        token_ids_l: [L]    差样本的 token id (padded)
        mask_l:      [L]    差样本的 valid mask
    """

    def __init__(
        self,
        winner_helms: List[str],
        loser_helms: List[str],
        vocab_file: str = "./data/helm_vocab.json",
        max_seq_len: int = 45,
    ):
        """
        Args:
            winner_helms: 好样本 HELM 序列列表 (top-25% by reward)
            loser_helms:  差样本 HELM 序列列表 (bottom-25% by reward)
            vocab_file:   词表路径
            max_seq_len:  序列最大长度 (pad 到此长度)
        """
        with open(vocab_file, 'r') as f:
            self.vocab = json.load(f)
        self.pad_id = self.vocab.get('<PAD>', 0)
        self.max_seq_len = max_seq_len
        self.topology_analyzer = HELMTopologyAnalyzer()

        # 解析并过滤有效序列
        self.winners = self._parse_sequences(winner_helms)
        self.losers = self._parse_sequences(loser_helms)

        # 配对: 取两组中较小的长度, 随机配对
        n_pairs = min(len(self.winners), len(self.losers))
        if n_pairs == 0:
            raise ValueError("No valid preference pairs. Check HELM sequences and vocab.")

        # 随机打乱后截断到相同长度
        rng = np.random.default_rng(42)
        w_idx = rng.permutation(len(self.winners))[:n_pairs]
        l_idx = rng.permutation(len(self.losers))[:n_pairs]
        self.winners = [self.winners[i] for i in w_idx]
        self.losers = [self.losers[i] for i in l_idx]

        print(f"[PreferencePairDataset] {n_pairs} pairs "
              f"(from {len(winner_helms)} winners, {len(loser_helms)} losers)")

    def _parse_sequences(self, helms: List[str]) -> List[List[int]]:
        """Parse HELM sequences to token id lists, filtering invalid ones."""
        results = []
        for helm in helms:
            try:
                parsed = self.topology_analyzer.parse_helm_sequence(helm)
                monomers = parsed['monomers']
                if len(monomers) > self.max_seq_len or len(monomers) == 0:
                    continue
                token_ids = [self.vocab.get(m, self.pad_id) for m in monomers]
                results.append(token_ids)
            except Exception:
                continue
        return results

    def _pad_and_mask(self, token_ids: List[int]):
        """Pad token_ids to max_seq_len and create mask."""
        actual_len = len(token_ids)
        padded = token_ids + [self.pad_id] * (self.max_seq_len - actual_len)
        mask = [1.0] * actual_len + [0.0] * (self.max_seq_len - actual_len)
        return (
            torch.tensor(padded, dtype=torch.long),
            torch.tensor(mask, dtype=torch.float),
        )

    def __len__(self) -> int:
        return len(self.winners)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        token_ids_w, mask_w = self._pad_and_mask(self.winners[idx])
        token_ids_l, mask_l = self._pad_and_mask(self.losers[idx])
        return {
            'token_ids_w': token_ids_w,  # [L]
            'mask_w': mask_w,            # [L]
            'token_ids_l': token_ids_l,  # [L]
            'mask_l': mask_l,            # [L]
        }


class PreferencePairCollator:
    """Collate preference pairs into batched tensors."""

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        return {
            'token_ids_w': torch.stack([s['token_ids_w'] for s in batch]),  # [Bz, L]
            'mask_w': torch.stack([s['mask_w'] for s in batch]),            # [Bz, L]
            'token_ids_l': torch.stack([s['token_ids_l'] for s in batch]),  # [Bz, L]
            'mask_l': torch.stack([s['mask_l'] for s in batch]),            # [Bz, L]
        }


def build_preference_pairs(
    all_helms: List[str],
    all_rewards: np.ndarray,
    top_ratio: float = 0.25,
    bottom_ratio: float = 0.25,
):
    """
    从生成结果中构造 winner/loser 列表.

    Args:
        all_helms:   生成的 HELM 序列列表
        all_rewards:  对应的 reward 数组 (= w1*vina + w2*perm)
        top_ratio:    取 reward 最高的比例作为 winner
        bottom_ratio: 取 reward 最低的比例作为 loser

    Returns:
        winner_helms: List[str]
        loser_helms:  List[str]
    """
    n = len(all_helms)
    sorted_idx = np.argsort(all_rewards)

    n_top = max(1, int(n * top_ratio))
    n_bottom = max(1, int(n * bottom_ratio))

    winner_idx = sorted_idx[-n_top:]
    loser_idx = sorted_idx[:n_bottom]

    winner_helms = [all_helms[i] for i in winner_idx]
    loser_helms = [all_helms[i] for i in loser_idx]

    print(f"[build_preference_pairs] N={n}, "
          f"winners={len(winner_helms)} (reward >= {all_rewards[winner_idx[0]]:.4f}), "
          f"losers={len(loser_helms)} (reward <= {all_rewards[loser_idx[-1]]:.4f})")

    return winner_helms, loser_helms
