"""Dataset classes for HELM sequences."""

import torch
from torch.utils.data import Dataset, DataLoader
import json
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional

from .topology import HELMTopologyAnalyzer


class HELMDataset(Dataset):
    """PyTorch Dataset for HELM peptide sequences."""
    
    def __init__(
        self,
        data_file: str,
        vocab_file: str = "./data/processed/helm_vocab.json",
        max_seq_len: int = 45,
        include_ring_bonds: bool = True,
        cyclic_only: bool = False
    ):
        self.max_seq_len = max_seq_len
        self.include_ring_bonds = include_ring_bonds
        self.cyclic_only = cyclic_only
        
        with open(vocab_file, 'r') as f:
            self.vocab = json.load(f)
        self.idx_to_token = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        self.pad_id = self.vocab.get('<PAD>', 0)
        
        self.topology_analyzer = HELMTopologyAnalyzer()
        
        self.sequences = []
        for sequence in self._iterate_helm_sequences(data_file):
            parsed = self.topology_analyzer.parse_helm_sequence(sequence)
            if len(parsed['monomers']) <= self.max_seq_len:
                # Filter for cyclic only if requested
                if self.cyclic_only:
                    if parsed['peptide_type'] in ['cyclic', 'q_type']:
                        self.sequences.append(sequence)
                else:
                    self.sequences.append(sequence)
        
        cyclic_str = " (cyclic only)" if self.cyclic_only else ""
        print(f"[HELMDataset] Loaded {len(self.sequences)} sequences (<= {self.max_seq_len}){cyclic_str}, vocab: {self.vocab_size}")

    def _iterate_helm_sequences(self, data_file: str):
        """Yield HELM sequences from txt (one-sequence-per-line) or csv files."""
        data_path = Path(data_file)

        if data_path.suffix.lower() == '.csv':
            with open(data_file, 'r', newline='') as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames:
                    raise ValueError(f"CSV file has no header: {data_file}")

                # Prefer canonical uppercase naming, but tolerate lowercase.
                helm_key = 'HELM' if 'HELM' in reader.fieldnames else 'helm'
                if helm_key not in reader.fieldnames:
                    raise ValueError(
                        f"CSV file {data_file} must contain a HELM column, got: {reader.fieldnames}"
                    )

                for row in reader:
                    sequence = (row.get(helm_key) or '').strip()
                    if sequence:
                        yield sequence
            return

        with open(data_file, 'r') as f:
            for line in f:
                sequence = line.strip()
                if sequence:
                    yield sequence
    
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
            # Legacy format (flattened upper triangular)
            ring_info = self.topology_analyzer.extract_ring_info(helm_seq)
            if ring_info is not None:
                result['ring_bond_array'] = torch.tensor(ring_info['bond_array'], dtype=torch.long)
                result['has_ring_bonds'] = len(parsed['connections']) > 0
            else:
                result['ring_bond_array'] = torch.tensor([], dtype=torch.long)
                result['has_ring_bonds'] = False
            
            # New format for autoregressive training
            # ring_bonds: [{'i': int, 'j': int, 'type': int}, ...]
            result['ring_bonds'] = self.topology_analyzer.extract_ring_bonds_for_ar(helm_seq)
        
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
        
        if 'ring_bonds' in batch[0]:
            # List of lists: [[{'i':..., 'j':..., 'type':...}, ...], ...]
            result['ring_bonds'] = [s.get('ring_bonds', []) for s in batch]
        
        return result


class CyclicHELMDataset(HELMDataset):
    """Dataset that only loads cyclic peptides for fine-tuning."""
    
    def __init__(
        self,
        data_file: str,
        vocab_file: str = "./data/processed/helm_vocab.json",
        max_seq_len: int = 45
    ):
        super().__init__(
            data_file=data_file,
            vocab_file=vocab_file,
            max_seq_len=max_seq_len,
            include_ring_bonds=True,
            cyclic_only=True
        )


def create_dataloader(
    data_file: str,
    vocab_file: str = "./data/processed/helm_vocab.json",
    batch_size: int = 32,
    max_seq_len: int = 45,
    shuffle: bool = True,
    num_workers: int = 4,
    cyclic_only: bool = False
) -> DataLoader:
    """Create a DataLoader for HELM data."""
    dataset = HELMDataset(
        data_file, vocab_file, max_seq_len,
        include_ring_bonds=True,
        cyclic_only=cyclic_only
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=HELMCollator(dataset.pad_id),
        pin_memory=True
    )
