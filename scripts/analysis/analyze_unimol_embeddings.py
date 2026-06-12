import torch
import numpy as np

# Load the embedding matrix.
try:
    emb = np.load("./data/processed/unimol_embeddings/embeddings_matrix.npy", allow_pickle=True)
    emb_tensor = torch.from_numpy(emb).float()
    
    print(f"Embedding Shape: {emb_tensor.shape}")
    print(f"Mean: {emb_tensor.mean().item():.4f}")
    print(f"Std: {emb_tensor.std().item():.4f}")
    print(f"Max Value: {emb_tensor.max().item():.4f}")
    print(f"Min Value: {emb_tensor.min().item():.4f}")
    
    norms = torch.norm(emb_tensor, dim=1)
    print(f"Average L2 norm: {norms.mean().item():.4f}")
    
except Exception as e:
    print(f"Failed to load embeddings: {e}")
