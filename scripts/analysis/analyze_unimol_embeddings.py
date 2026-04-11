import torch
import numpy as np

# 加载Embedding 矩阵
try:
    emb = np.load("./data/processed/unimol_embeddings/embeddings_matrix.npy", allow_pickle=True)
    emb_tensor = torch.from_numpy(emb).float()
    
    print(f"Embedding Shape: {emb_tensor.shape}")
    print(f"Mean: {emb_tensor.mean().item():.4f}")
    print(f"Std (标准差): {emb_tensor.std().item():.4f}")
    print(f"Max Value: {emb_tensor.max().item():.4f}")
    print(f"Min Value: {emb_tensor.min().item():.4f}")
    
    # 计算平均范数 (Norm)
    norms = torch.norm(emb_tensor, dim=1)
    print(f"Average Norm (L2长度): {norms.mean().item():.4f}")
    
except Exception as e:
    print(f"无法加载: {e}")
