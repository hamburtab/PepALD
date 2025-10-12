import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Optional


class MolFormerEmbedding(nn.Module):
    def __init__(self, 
                 embeddings_dir: str = "./molformer_embeddings",
                 vocab: Optional[Dict[str, int]] = None,
                 freeze_embeddings: bool = False):
        super().__init__()
        
        self.embeddings_dir = Path(embeddings_dir)
        self.freeze_embeddings = freeze_embeddings
        
        self._load_embeddings()
        
        if vocab is not None:
            self._setup_vocab_mapping(vocab)
        
        print(f"   MolFormer嵌入层初始化完成:")
        print(f"   嵌入维度: {self.embedding_dim}")
        print(f"   可用单体数: {self.num_monomers}")
        print(f"   冻结参数: {self.freeze_embeddings}")
    
    def _load_embeddings(self):
        with open(self.embeddings_dir / "metadata.json", 'r') as f:
            self.metadata = json.load(f)
        
        # 加载embeddings矩阵
        embeddings_matrix = np.load(self.embeddings_dir / "complete_embeddings_matrix.npy", allow_pickle=True)
        self.embedding_dim = embeddings_matrix.shape[1]
        self.num_monomers = embeddings_matrix.shape[0]
        
        # 加载单体映射
        self.monomer_mapping = pd.read_csv(self.embeddings_dir / "（3104*768）complete_monomer_mapping.csv")
        
        # 创建symbol到索引的映射
        self.symbol_to_idx = {symbol: idx for idx, symbol in enumerate(self.monomer_mapping['symbol'])}
        self.idx_to_symbol = {idx: symbol for symbol, idx in self.symbol_to_idx.items()}
        
        # 转换为PyTorch embedding层
        self.embedding_matrix = torch.from_numpy(embeddings_matrix).float()
        self.embeddings = nn.Embedding.from_pretrained(
            self.embedding_matrix,
            freeze=self.freeze_embeddings
        )
        
        # 创建特殊token的embedding
        special_embeddings = torch.randn(4, self.embedding_dim) * 0.1
        special_embeddings[0, :] = 0.0  # <PAD>设为全0向量
        self.special_embeddings = nn.Parameter(special_embeddings, requires_grad=False)  # 冻结特殊token
        
        print(f"   加载的嵌入统计:")
        print(f"   模型: {self.metadata['model_name']}")
        print(f"   成功单体: {self.metadata['successful_count']}")
        print(f"   失败单体: {self.metadata['failed_count']}")
    
    def _setup_vocab_mapping(self, vocab: Dict[str, int]):
        self.vocab = vocab
        self.vocab_size = len(vocab)
        
        self.idx_to_token = {idx: token for token, idx in vocab.items()}
        
        self.vocab_to_embedding = torch.full((self.vocab_size,), -1, dtype=torch.long)
        
        matched_count = 0
        special_tokens = ['<PAD>', '<UNK>', '<START>', '<END>']
        
        for token, vocab_idx in vocab.items():
            if token in special_tokens:
                special_idx = special_tokens.index(token)
                self.vocab_to_embedding[vocab_idx] = self.num_monomers + special_idx
            elif token in self.symbol_to_idx:
                embedding_idx = self.symbol_to_idx[token]
                self.vocab_to_embedding[vocab_idx] = embedding_idx
                matched_count += 1

            # 认为存在vocab中的token可能带有括号且symbol_to_idx中没有括号的情况
            else:
                cleaned_token = token.strip('[]')
                if cleaned_token in self.symbol_to_idx:
                    embedding_idx = self.symbol_to_idx[cleaned_token]
                    self.vocab_to_embedding[vocab_idx] = embedding_idx
                    matched_count += 1
                else:
                    unk_idx = special_tokens.index('<UNK>')
                    self.vocab_to_embedding[vocab_idx] = self.num_monomers + unk_idx
        
        print(f"   词汇表映射统计:")
        print(f"   词汇表大小: {self.vocab_size}")
        print(f"   匹配单体: {matched_count}")
        print(f"   匹配率: {matched_count/self.vocab_size*100:.1f}%")
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            input_ids: token索引 [batch_size, seq_len]
            
        Returns:
            embeddings: 嵌入向量 [batch_size, seq_len, embedding_dim]
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        if hasattr(self, 'vocab_to_embedding'):
            # 使用vocab映射
            embedding_ids = self.vocab_to_embedding[input_ids.cpu()].to(device)
            
            # 分离常规embeddings和特殊embeddings
            regular_mask = embedding_ids < self.num_monomers
            special_mask = embedding_ids >= self.num_monomers
            
            output = torch.zeros(batch_size, seq_len, self.embedding_dim, device=device)
            
            if regular_mask.any():
                regular_ids = embedding_ids[regular_mask]
                output[regular_mask] = self.embeddings(regular_ids)
            
            if special_mask.any():
                special_ids = embedding_ids[special_mask] - self.num_monomers
                output[special_mask] = self.special_embeddings[special_ids]
            
            return output
        else:
            # 直接使用input_ids作为embedding索引
            return self.embeddings(input_ids)