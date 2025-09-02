import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple


class HybridPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 256):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        self.register_buffer('linear_pe_buffer', self._create_linear_encoding(max_len, d_model))
        self.ring_type_set = ['R3R3', 'R1R2', 'R1R3', 'R3R2']
    
    def _create_linear_encoding(self, max_len: int, d_model: int):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0).transpose(0, 1)
    
    def forward(self, x, peptide_type=None, connection_info=None):
        seq_len, batch_size, d_model = x.shape
        pe = self.linear_pe_buffer[:seq_len, 0, :].unsqueeze(1).expand(-1, batch_size, -1)
        x = x + pe
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, V)
        
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        output = self.w_o(context)
        return output, attn_weights


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        attn_output, attn_weights = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x, attn_weights


class HELMTransformer(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 768,  # MolFormer embedding维度
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        max_seq_len: int = 256,
        dropout: float = 0.1,
        time_embed_dim: int = 128,
        ring_bond_type: int = 4
    ):
        super().__init__()
        
        self.d_model = d_model
        self.embedding_dim = embedding_dim
        
        self.embedding_projection = nn.Linear(embedding_dim, d_model)
        
        self.time_embedding = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, d_model)
        )
        
        self.pos_encoding = HybridPositionalEncoding(d_model, dropout=dropout, max_len=max_seq_len)
        
        #共循环TransformerEncoderLayer层6次
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.output_projection = nn.Linear(d_model, embedding_dim)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout_layer = nn.Dropout(dropout)
        
        # 兼容性属性
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        
        # 环键类型定义
        self.bond_types = ['R3R3', 'R1R2', 'R1R3', 'R3R2']
        
        self.ring_bond_embedding = nn.Sequential(
            nn.Linear(1, ring_bond_type),
            nn.SiLU(),
            nn.Linear(ring_bond_type, ring_bond_type)
        )
        
    def forward(
        self,
        x: torch.Tensor,  # [batch_size, seq_len, embedding_dim]
        t: torch.Tensor,  # [batch_size] 时间步
        mask: Optional[torch.Tensor] = None,  # [batch_size, seq_len]
        peptide_type: Optional[list] = None,
        connection_info: Optional[list] = None,
        helm_sequences: Optional[list] = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        x = self.embedding_projection(x)  # [batch_size, seq_len, d_model]
        
        t_embed = self.time_embedding(t.float().unsqueeze(-1))  # [batch_size, d_model] - 确保float类型
        t_embed = t_embed.unsqueeze(1).expand(-1, seq_len, -1)  # [batch_size, seq_len, d_model]
        x = x + t_embed
        
        x = x.transpose(0, 1)  # [seq_len, batch_size, d_model]
        x = self.pos_encoding(x, peptide_type=peptide_type, connection_info=connection_info)
        x = x.transpose(0, 1)  # [batch_size, seq_len, d_model]
        
        if mask is not None:
            attn_mask = mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
            attn_mask = attn_mask.expand(-1, 1, seq_len, -1)  # [batch_size, 1, seq_len, seq_len]
        else:
            attn_mask = None
        
        for layer in self.transformer_layers:
            x, attn_weights = layer(x, attn_mask)
        
        x = self.layer_norm(x)
        x = self.output_projection(x)
        
        # 环键嵌入处理 - 仅在低噪声场景（t < 100）计算
        ring_bond_loss = None
        if hasattr(self, 'ring_bond_embedding') and helm_sequences is not None:
            low_noise_mask = t < 100
            if low_noise_mask.any():
                try:
                    ring_bond_targets = []
                    valid_indices = []
                    
                    for idx, helm_seq in enumerate(helm_sequences):
                        if low_noise_mask[idx].item():
                            ring_info = self._extract_ring_info(helm_seq)
                            if ring_info and ring_info.get('ring_connections'):
                                # 构建环键类型矩阵
                                bond_matrix = torch.zeros(seq_len, seq_len, dtype=torch.long, device=x.device)
                                
                                for connection in ring_info['ring_connections']:
                                    pos1, r1, pos2, r2 = connection
                                    if pos1 < seq_len and pos2 < seq_len:
                                        bond_type = f"{r1}{r2}"
                                        if bond_type in self.bond_types:
                                            bond_idx = self.bond_types.index(bond_type)
                                            bond_matrix[pos1, pos2] = bond_idx + 1  # 使用1-based索引，0表示无连接
                                            bond_matrix[pos2, pos1] = bond_idx + 1
                                
                                ring_bond_targets.append(bond_matrix)
                                valid_indices.append(idx)
                    
                    if ring_bond_targets and valid_indices:
                        # 提取对应的注意力权重
                        valid_attn_weights = attn_weights[valid_indices]  # [valid_batch, num_heads, seq_len, seq_len]
                        
                        # 平均所有注意力头
                        avg_attn = valid_attn_weights.mean(dim=1).float()  # [valid_batch, seq_len, seq_len]，确保float类型
                        
                        # 构建上三角掩码
                        triu_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
                        
                        # 提取上三角部分用于环键预测
                        triu_attn = avg_attn[:, triu_mask]  # [valid_batch, num_pairs]
                        ring_bond_pred = self.ring_bond_embedding(triu_attn.unsqueeze(-1))  # [valid_batch, num_pairs, num_bond_types]
                        
                        # 构建目标标签
                        ring_bond_targets_tensor = torch.stack(ring_bond_targets)  # [valid_batch, seq_len, seq_len]
                        target_labels = ring_bond_targets_tensor[:, triu_mask]  # [valid_batch, num_pairs]
                        
                        # 计算环键分类损失（注意要调整标签）
                        # 将1-based标签转为0-based，并只考虑有连接的位置
                        valid_positions = target_labels > 0
                        if valid_positions.any():
                            valid_labels = target_labels[valid_positions] - 1  # 转为0-based
                            valid_preds = ring_bond_pred.view(-1, len(self.bond_types))[valid_positions.view(-1)]
                            
                            ring_bond_loss = F.cross_entropy(valid_preds, valid_labels)
                        
                except Exception as e:
                    print(f"环键损失计算中的错误: {e}")
                    ring_bond_loss = None
        
        # 对于简单的环键嵌入，当没有特定损失时使用均值嵌入
        if ring_bond_loss is None:
            # 确保数据类型一致
            attn_mean = attn_weights.mean(dim=1).float()  # 转换为float
            ring_bond_embedding = self.ring_bond_embedding(attn_mean.unsqueeze(-1))
            return x, ring_bond_embedding
        else:
            return x, ring_bond_loss
    
    def get_attention_weights(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> list:
        batch_size, seq_len, _ = x.shape
        
        x = self.embedding_projection(x)
        
        t_embed = self.time_embedding(t.float().unsqueeze(-1))
        t_embed = t_embed.unsqueeze(1).expand(-1, seq_len, -1)
        x = x + t_embed
        
        x = x.transpose(0, 1)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)
        
        if mask is not None:
            attn_mask = mask.unsqueeze(1).unsqueeze(2)
            attn_mask = attn_mask.expand(-1, 1, seq_len, -1)
        else:
            attn_mask = None
        
        attention_weights = []
        for layer in self.transformer_layers:
            _, weights = layer.self_attn(x, x, x, attn_mask)
            attention_weights.append(weights)
            x = layer(x, attn_mask)
        
        return attention_weights
    
    def _extract_ring_info(self, helm_seq: str):
        """从HELM序列中提取环键信息
        基于prepare_chembl32_data.py的正确实现
        格式: PEPTIDE1{[dY].[dC].F.W.K.[meT].C.T.[am]}$PEPTIDE1,PEPTIDE1,7:R3-2:R3$$$
        """
        # 检查是否是环肽（不以}$$$$结尾）
        if helm_seq.split('}')[-1] == '$$$$':
            return None
        
        try:
            # 提取序列部分
            seq_part = helm_seq.split('{')[1].split('}')[0]
            res_num = len(seq_part.split('.'))
            
            # 提取环键信息部分（在}之后）
            ring_info = helm_seq.split('}')[-1]
            if not ring_info:
                return None
            
            ring_connections = []
            
            # 按|分割不同的环键信息
            for ring in ring_info.split('|'):
                if not ring or ':' not in ring:
                    continue
                    
                try:
                    # 按prepare_chembl32_data.py的解析逻辑
                    # 格式：某些前缀,某些内容,位置:R基团-位置:R基团
                    parts = ring.split(':')
                    if len(parts) < 3:
                        continue
                    
                    # 提取起始信息
                    r_st = parts[1].split('-')[0]  # 'R3-3' -> 'R3'
                    res_st = int(parts[0].split(',')[-1]) - 1  # '位置' -> 0-based
                    
                    # 提取结束信息  
                    r_ed = parts[2].split('$')[0]  # 'R3$$$' -> 'R3'
                    res_ed = int(parts[1].split('-')[1]) - 1  # 'R3-3' -> '3' -> 2 (0-based)
                    
                    # 验证位置有效性
                    if 0 <= res_st < res_num and 0 <= res_ed < res_num:
                        ring_connections.append((res_st, r_st, res_ed, r_ed))
                    
                except (ValueError, IndexError):
                    continue
            
            if ring_connections:
                return {
                    'sequence_length': res_num,
                    'ring_connections': ring_connections
                }
            else:
                return None
            
        except (IndexError, ValueError):
            return None


def create_helm_transformer_for_chembl32(vocab_size, d_model=512, nhead=8, num_layers=10, 
                                        max_seq_len=150, dropout=0.15):
    """为ChEMBL32创建兼容的HELMTransformer"""
    return HELMTransformer(
        embedding_dim=768,  # 保持MolFormer embedding维度
        d_model=d_model,
        n_heads=nhead,
        n_layers=num_layers,
        d_ff=d_model * 4,
        max_seq_len=max_seq_len,
        dropout=dropout
    )


if __name__ == "__main__":
    model = HELMTransformer()
    
    batch_size = 2
    seq_len = 10
    embedding_dim = 768
    
    x = torch.randn(batch_size, seq_len, embedding_dim)
    t = torch.randn(batch_size)
    mask = torch.ones(batch_size, seq_len)
    
    output = model(x, t, mask)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("Model test passed")
