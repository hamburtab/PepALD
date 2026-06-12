import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
from typing import Optional


class UniMolEmbedding(nn.Module):
    """
    The vocabulary index matches the embedding-matrix index exactly.
    - 0-3103: monomers, in monomer_mapping.csv order
    - 3104: <PAD>
    """
    def __init__(self, 
                 embeddings_dir: str = "./data/processed/unimol_embeddings",
                 freeze_embeddings: bool = False):
        super().__init__()
        
        self.embeddings_dir = Path(embeddings_dir)
        self.freeze_embeddings = freeze_embeddings
        
        self._load_embeddings()
        
        print("   Uni-Mol embedding layer initialized:")
        print(f"   Embedding dim: {self.embedding_dim}")
        print(f"   Num monomers: {self.num_monomers}")
        print(f"   Vocab size: {self.vocab_size} (including <PAD>)")
        print(f"   Frozen: {self.freeze_embeddings}")
    
    def _load_embeddings(self):
        """Load embedding matrix and metadata."""
        with open(self.embeddings_dir / "metadata.json", 'r') as f:
            self.metadata = json.load(f)
        
        # Load embedding matrix [num_monomers, embedding_dim].
        embeddings_matrix = np.load(
            self.embeddings_dir / "embeddings_matrix.npy", 
            allow_pickle=True
        )
        self.num_monomers = embeddings_matrix.shape[0]  # 3104
        self.embedding_dim = embeddings_matrix.shape[1]
        
        self.embedding_matrix = torch.from_numpy(embeddings_matrix).float()
        
        pad_embedding = torch.zeros(1, self.embedding_dim)
        full_embeddings = torch.cat([self.embedding_matrix, pad_embedding], dim=0)  # [3105, embedding_dim]
        
        self.embeddings = nn.Embedding.from_pretrained(
            full_embeddings,
            freeze=self.freeze_embeddings,
            padding_idx=self.num_monomers
        )
        
        self.vocab_size = self.num_monomers + 1
        
        print("   Loaded embedding stats:")
        print(f"   Model: {self.metadata['model_name']}")
        print(f"   Successful monomers: {self.metadata['successful_count']}")
        print(f"   Failed monomers: {self.metadata['failed_count']}")
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: token indices [batch_size, seq_len].
            
        Returns:
            Embeddings [batch_size, seq_len, embedding_dim].
        """
        return self.embeddings(input_ids)
