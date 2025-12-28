"""
Dataset classes for HELM sequences.

Provides PyTorch Dataset implementations for loading and processing
HELM sequence data for training the ALD model.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from .topology import HELMTopologyAnalyzer


class HELMDataset(Dataset):
    """
    PyTorch Dataset for HELM peptide sequences.
    
    Loads HELM sequences from a text file and converts them to
    token IDs suitable for training the ALD model.
    
    Args:
        data_file: Path to file containing HELM sequences (one per line)
        vocab_file: Path to vocabulary JSON file
        max_seq_len: Maximum sequence length (sequences longer are truncated)
        include_ring_bonds: Whether to include ring bond labels
    """
    
    def __init__(
        self,
        data_file: str,
        vocab_file: str = "./data/helm_vocab.json",
        max_seq_len: int = 45,
        include_ring_bonds: bool = True
    ):
        self.data_file = Path(data_file)
        self.vocab_file = Path(vocab_file)
        self.max_seq_len = max_seq_len
        self.include_ring_bonds = include_ring_bonds
        
        # Load vocabulary
        with open(vocab_file, 'r') as f:
            self.vocab = json.load(f)
        
        self.idx_to_token = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        self.pad_id = self.vocab.get('<PAD>', 0)
        
        # Topology analyzer
        self.topology_analyzer = HELMTopologyAnalyzer()
        
        # Load sequences
        self.sequences = []
        with open(data_file, 'r') as f:
            for line in f:
                helm_seq = line.strip()
                if helm_seq:
                    self.sequences.append(helm_seq)
        
        print(f"[HELMDataset] Loaded {len(self.sequences)} sequences from {data_file}")
        print(f"[HELMDataset] Vocab size: {self.vocab_size}, Max length: {max_seq_len}")
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample.
        
        Returns:
            Dictionary with:
                - token_ids: Tensor of token indices [max_seq_len]
                - mask: Attention mask [max_seq_len]
                - length: Actual sequence length
                - helm_sequence: Original HELM string
                - peptide_type: Type of peptide (linear/cyclic/q_type)
                - ring_bonds: Ring bond information (if include_ring_bonds)
        """
        helm_seq = self.sequences[idx]
        
        # Parse HELM sequence
        parsed = self.topology_analyzer.parse_helm_sequence(helm_seq)
        monomers = parsed['monomers']
        
        # Convert to token IDs
        token_ids = self._monomers_to_ids(monomers)
        actual_len = len(token_ids)
        
        # Truncate if necessary
        if len(token_ids) > self.max_seq_len:
            token_ids = token_ids[:self.max_seq_len]
            actual_len = self.max_seq_len
        
        # Pad to max length
        padding_length = self.max_seq_len - len(token_ids)
        token_ids = token_ids + [self.pad_id] * padding_length
        
        # Create mask (1 for valid tokens, 0 for padding)
        # Mask should ONLY cover actual tokens, NOT padding positions
        # This ensures the model only learns to predict real tokens
        mask = [1.0] * actual_len
        mask = mask + [0.0] * (self.max_seq_len - len(mask))
        
        result = {
            'token_ids': torch.tensor(token_ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.float),
            'length': actual_len,
            'helm_sequence': helm_seq,
            'peptide_type': parsed['peptide_type'],
        }
        
        # Add ring bond information if requested
        if self.include_ring_bonds:
            ring_info = self.topology_analyzer.extract_ring_info(helm_seq)
            if ring_info is not None:
                result['ring_bond_array'] = torch.tensor(ring_info['bond_array'], dtype=torch.long)
                result['has_ring_bonds'] = len(parsed['connections']) > 0
            else:
                result['ring_bond_array'] = torch.tensor([], dtype=torch.long)
                result['has_ring_bonds'] = False
        
        return result
    
    def _monomers_to_ids(self, monomers: List[str]) -> List[int]:
        """Convert monomer symbols to token IDs."""
        return [self.vocab.get(m, self.pad_id) for m in monomers]
    
    def decode_tokens(
        self,
        token_ids: torch.Tensor,
        ring_connections: Optional[List[Dict]] = None
    ) -> str:
        """
        Decode token IDs back to HELM string.
        
        Args:
            token_ids: Tensor of token indices
            ring_connections: Optional ring bond information
            
        Returns:
            HELM string
        """
        monomers = []
        for token_id in token_ids.tolist():
            if token_id == self.pad_id:
                break
            monomer = self.idx_to_token.get(token_id, '?')
            monomers.append(monomer)
        
        if not monomers:
            return "PEPTIDE1{}$$$$"
        
        # Convert ring_connections to HELM format
        connections = []
        if ring_connections:
            for conn in ring_connections:
                bond_type = conn.get('bond_type', 'R3R3')
                if bond_type == 'R3R3':
                    r1, r2 = 3, 3
                elif bond_type == 'R1R2':
                    r1, r2 = 1, 2
                elif bond_type == 'R1R3':
                    r1, r2 = 1, 3
                elif bond_type == 'R3R2':
                    r1, r2 = 3, 2
                else:
                    continue
                
                connections.append({
                    'pos1': conn['res1'],
                    'r1': r1,
                    'pos2': conn['res2'],
                    'r2': r2
                })
        
        return self.topology_analyzer.build_helm_string(monomers, connections)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        lengths = []
        type_counts = {'linear': 0, 'cyclic': 0, 'q_type': 0}
        
        for helm_seq in self.sequences:
            parsed = self.topology_analyzer.parse_helm_sequence(helm_seq)
            lengths.append(len(parsed['monomers']))
            type_counts[parsed['peptide_type']] = type_counts.get(parsed['peptide_type'], 0) + 1
        
        return {
            'num_sequences': len(self.sequences),
            'length_mean': sum(lengths) / len(lengths) if lengths else 0,
            'length_min': min(lengths) if lengths else 0,
            'length_max': max(lengths) if lengths else 0,
            'type_distribution': type_counts
        }


class HELMCollator:
    """
    Collate function for HELM dataset.
    
    Handles variable-length sequences and batching.
    """
    
    def __init__(self, pad_id: int = 0):
        self.pad_id = pad_id
    
    def __call__(self, batch: List[Dict]) -> Dict[str, Any]:
        """
        Collate a batch of samples.
        
        Args:
            batch: List of sample dictionaries
            
        Returns:
            Batched dictionary
        """
        # Stack tensors
        token_ids = torch.stack([sample['token_ids'] for sample in batch])
        mask = torch.stack([sample['mask'] for sample in batch])
        lengths = torch.tensor([sample['length'] for sample in batch])
        
        result = {
            'token_ids': token_ids,
            'mask': mask,
            'lengths': lengths,
            'helm_sequences': [sample['helm_sequence'] for sample in batch],
            'peptide_types': [sample['peptide_type'] for sample in batch],
        }
        
        # Handle ring bonds if present
        if 'ring_bond_array' in batch[0]:
            result['has_ring_bonds'] = torch.tensor([
                sample.get('has_ring_bonds', False) for sample in batch
            ])
            # Ring bond arrays have variable lengths, keep as list
            result['ring_bond_arrays'] = [
                sample.get('ring_bond_array', torch.tensor([])) 
                for sample in batch
            ]
        
        return result


def create_dataloader(
    dataset_or_path,
    vocab_file: str = "./data/helm_vocab.json",
    batch_size: int = 32,
    max_seq_len: int = 45,
    shuffle: bool = True,
    num_workers: int = 4,
    **kwargs
) -> DataLoader:
    """
    Convenience function to create a DataLoader for HELM data.
    
    Args:
        dataset_or_path: Either a HELMDataset instance or path to HELM sequence file
        vocab_file: Path to vocabulary file (only used if dataset_or_path is a path)
        batch_size: Batch size
        max_seq_len: Maximum sequence length (only used if dataset_or_path is a path)
        shuffle: Whether to shuffle
        num_workers: Number of data loading workers
        **kwargs: Additional arguments for Dataset (only used if dataset_or_path is a path)
        
    Returns:
        DataLoader instance
    """
    if isinstance(dataset_or_path, HELMDataset):
        dataset = dataset_or_path
    else:
        dataset = HELMDataset(
            data_file=dataset_or_path,
            vocab_file=vocab_file,
            max_seq_len=max_seq_len,
            **kwargs
        )
    
    collator = HELMCollator(pad_id=dataset.pad_id)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True
    )
