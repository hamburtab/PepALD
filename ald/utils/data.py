"""Dataset classes for HELM sequences."""

import torch
from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path
from typing import Dict, List, Any

from .topology import HELMTopologyAnalyzer


class HELMDataset(Dataset):
    """PyTorch Dataset for HELM peptide sequences."""
    
    def __init__(
        self,
        data_file: str,
        vocab_file: str = "./data/helm_vocab.json",
        max_seq_len: int = 45,
        include_ring_bonds: bool = True
    ):
        self.max_seq_len = max_seq_len
        self.include_ring_bonds = include_ring_bonds
        
        with open(vocab_file, 'r') as f:
            self.vocab = json.load(f)
        self.idx_to_token = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        self.pad_id = self.vocab.get('<PAD>', 0)
        
        self.topology_analyzer = HELMTopologyAnalyzer()
        
        self.sequences = []
        with open(data_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Filter by length
                parsed = self.topology_analyzer.parse_helm_sequence(line)
                if len(parsed['monomers']) <= self.max_seq_len:
                    self.sequences.append(line)
        
        print(f"[HELMDataset] Loaded {len(self.sequences)} sequences (<= {self.max_seq_len}), vocab: {self.vocab_size}")
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        helm_seq = self.sequences[idx]
        parsed = self.topology_analyzer.parse_helm_sequence(helm_seq)
        monomers = parsed['monomers']
        
        # Convert to token IDs
        token_ids = [self.vocab.get(m, self.pad_id) for m in monomers]
        actual_len = len(token_ids)
        
        # Pad to max_seq_len
        token_ids = token_ids + [self.pad_id] * (self.max_seq_len - actual_len)
        
        # Create mask (1 for valid, 0 for padding)
        mask = [1.0] * actual_len + [0.0] * (self.max_seq_len - actual_len)
        
        result = {
            'token_ids': torch.tensor(token_ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.float),
            'length': actual_len,
            'helm_sequence': helm_seq,
            'peptide_type': parsed['peptide_type'],
        }
        
        if self.include_ring_bonds:
            ring_info = self.topology_analyzer.extract_ring_info(helm_seq)
            if ring_info is not None:
                result['ring_bond_array'] = torch.tensor(ring_info['bond_array'], dtype=torch.long)
                result['has_ring_bonds'] = len(parsed['connections']) > 0
            else:
                result['ring_bond_array'] = torch.tensor([], dtype=torch.long)
                result['has_ring_bonds'] = False
        
        return result


class HELMCollator:
    """Collate function for batching HELM samples."""
    
    def __init__(self, pad_id: int = 0):
        self.pad_id = pad_id
    
    def __call__(self, batch: List[Dict]) -> Dict[str, Any]:
        result = {
            'token_ids': torch.stack([s['token_ids'] for s in batch]),
            'mask': torch.stack([s['mask'] for s in batch]),
            'lengths': torch.tensor([s['length'] for s in batch]),
            'helm_sequences': [s['helm_sequence'] for s in batch],
            'peptide_types': [s['peptide_type'] for s in batch],
        }
        
        if 'ring_bond_array' in batch[0]:
            result['has_ring_bonds'] = torch.tensor([s.get('has_ring_bonds', False) for s in batch])
            result['ring_bond_arrays'] = [s.get('ring_bond_array', torch.tensor([])) for s in batch]
        
        return result


def create_dataloader(
    data_file: str,
    vocab_file: str = "./data/helm_vocab.json",
    batch_size: int = 32,
    max_seq_len: int = 45,
    shuffle: bool = True,
    num_workers: int = 4
) -> DataLoader:
    """Create a DataLoader for HELM data."""
    dataset = HELMDataset(data_file, vocab_file, max_seq_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=HELMCollator(dataset.pad_id),
        pin_memory=True
    )
