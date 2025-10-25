import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
from typing import Optional


class MolFormerEmbedding(nn.Module):
    """
    vocab索引与embedding矩阵索引完全一致
    - 索引 0-3103: 单体 (与 monomer_mapping.csv 顺序一致)
    - 索引 3104: <PAD> 特殊token
    """
    def __init__(self, 
                 embeddings_dir: str = "./molformer_embeddings",
                 freeze_embeddings: bool = False):
        super().__init__()
        
        self.embeddings_dir = Path(embeddings_dir)
        self.freeze_embeddings = freeze_embeddings
        
        self._load_embeddings()
        
        print(f"   MolFormer嵌入层初始化完成:")
        print(f"   嵌入维度: {self.embedding_dim}")
        print(f"   单体数量: {self.num_monomers}")
        print(f"   总词汇量: {self.vocab_size} (包括 <PAD>)")
        print(f"   冻结参数: {self.freeze_embeddings}")
    
    def _load_embeddings(self):
        """加载预训练的embedding矩阵和元数据"""
        with open(self.embeddings_dir / "metadata.json", 'r') as f:
            self.metadata = json.load(f)
        
        # 加载embeddings矩阵 [num_monomers, embedding_dim]
        embeddings_matrix = np.load(
            self.embeddings_dir / "embeddings_matrix.npy", 
            allow_pickle=True
        )
        self.num_monomers = embeddings_matrix.shape[0]  # 3104
        self.embedding_dim = embeddings_matrix.shape[1]  # 768
        
        # 转换为PyTorch embedding层
        self.embedding_matrix = torch.from_numpy(embeddings_matrix).float()
        
        # 添加PAD的embedding (全0向量)
        pad_embedding = torch.zeros(1, self.embedding_dim)
        full_embeddings = torch.cat([self.embedding_matrix, pad_embedding], dim=0)  # [3105, 768]
        
        # 创建embedding层
        self.embeddings = nn.Embedding.from_pretrained(
            full_embeddings,
            freeze=self.freeze_embeddings,
            padding_idx=self.num_monomers  # PAD的索引是3104
        )
        
        self.vocab_size = self.num_monomers + 1  # 3105 (3104个单体 + 1个PAD)
        
        print(f"   加载的嵌入统计:")
        print(f"   模型: {self.metadata['model_name']}")
        print(f"   成功单体: {self.metadata['successful_count']}")
        print(f"   失败单体: {self.metadata['failed_count']}")
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            input_ids: token索引 [batch_size, seq_len]
                      范围: 0-3103 (单体), 3104 (PAD)
            
        Returns:
            embeddings: 嵌入向量 [batch_size, seq_len, embedding_dim]
        """
        return self.embeddings(input_ids)
        