import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class HybridPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 100):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        
        # 标准位置编码
        self.register_buffer('linear_pe_buffer', self._create_linear_encoding(max_len, d_model))
        
        # 圆环投影层
        self.ring_projection = nn.Linear(d_model, d_model)
        
        # 连接投影层 (考虑R基团类型)
        self.connection_projection = nn.Linear(d_model, d_model)
        
        # R基团类型嵌入 (R1, R2, R3等)
        self.r_group_embedding = nn.Embedding(10, d_model // 4)  # 支持R1-R9
        
        # 尾部投影层
        self.tail_projection = nn.Linear(d_model, d_model)
    
    def _create_linear_encoding(self, max_len: int, d_model: int):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0).transpose(0, 1)
    
    def _create_circular_encoding(self, seq_len: int, d_model: int, start_pos: int = 0, end_pos: int = None):
        if end_pos is None:
            end_pos = seq_len
        ring_len = end_pos - start_pos
        
        pe = torch.zeros(seq_len, d_model)
        for i in range(start_pos, end_pos):
            # 圆环相对位置
            ring_pos = (i - start_pos) / ring_len * 2 * math.pi
            
            for j in range(0, d_model, 2):
                freq = 1.0 / (10000.0 ** (j / d_model))
                pe[i, j] = math.sin(ring_pos * freq)
                if j + 1 < d_model:
                    pe[i, j + 1] = math.cos(ring_pos * freq)
        
        return pe
    
    def forward(self, x, peptide_type=None, connection_info=None):
        seq_len, batch_size, d_model = x.shape
        
        if peptide_type is None:
            peptide_type = ['linear'] * batch_size
        
        pe = torch.zeros_like(x)
        
        for b in range(batch_size):
            if peptide_type[b] == 'linear':
                # 标准线性肽：使用线性位置编码
                pe[:, b, :] = self.linear_pe_buffer[:seq_len, 0, :]
                
            elif peptide_type[b] == 'cyclic':
                # 标准环肽：纯圆环位置编码
                circular_pe = self._create_circular_encoding(seq_len, d_model)
                pe[:, b, :] = self.ring_projection(circular_pe)
                
            elif peptide_type[b] == 'q_type' and connection_info is not None and connection_info[b] is not None:
                # Q型环肽：混合编码
                connections = connection_info[b]
                if connections:
                    conn_dict = connections[0]  # 使用第一个连接
                    conn_start = conn_dict['pos1']
                    conn_end = conn_dict['pos2']
                    r1 = conn_dict['r1']
                    r2 = conn_dict['r2']
                    
                    # 环部分：圆环编码
                    if conn_start < conn_end:
                        ring_pe = self._create_circular_encoding(seq_len, d_model, conn_start, conn_end + 1)
                        pe[conn_start:conn_end + 1, b, :] = self.ring_projection(ring_pe[conn_start:conn_end + 1])
                    
                    # 连接点特殊标记（包含R基团类型）
                    conn_encoding = torch.zeros(seq_len, d_model)
                    
                    # R基团类型嵌入
                    r1_embed = self.r_group_embedding(torch.tensor(r1 - 1))  # R1->0, R2->1, etc.
                    r2_embed = self.r_group_embedding(torch.tensor(r2 - 1))
                    
                    # 为连接位置添加R基团特定编码
                    conn_encoding[conn_start, :d_model//4] = r1_embed
                    conn_encoding[conn_end, :d_model//4] = r2_embed
                    
                    # 添加位置标记
                    conn_encoding[conn_start, d_model//4:d_model//2] = 1.0
                    conn_encoding[conn_end, d_model//2:3*d_model//4] = 1.0
                pe[:, b, :] += self.connection_projection(conn_encoding)
                
                # 尾部：线性编码 + 衰减
                tail_start = max(conn_start, conn_end) + 1
                if tail_start < seq_len:
                    tail_pe = self.linear_pe_buffer[tail_start:seq_len, 0, :]
                    for i, pos in enumerate(range(tail_start, seq_len)):
                        decay = 1.0 - i / (seq_len - tail_start)
                        pe[pos, b, :] = self.tail_projection(tail_pe[i] * decay)
            else:
                # 默认使用线性编码
                pe[:, b, :] = self.linear_pe_buffer[:seq_len, 0, :]
        
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
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


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
        time_embed_dim: int = 128
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
        
    def forward(
        self,
        x: torch.Tensor,  # [batch_size, seq_len, embedding_dim]
        t: torch.Tensor,  # [batch_size] 时间步
        mask: Optional[torch.Tensor] = None,  # [batch_size, seq_len]
        peptide_type: Optional[list] = None,
        connection_info: Optional[list] = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        x = self.embedding_projection(x)  # [batch_size, seq_len, d_model]
        
        t_embed = self.time_embedding(t.unsqueeze(-1))  # [batch_size, d_model]
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
            x = layer(x, attn_mask)
        
        x = self.layer_norm(x)
        x = self.output_projection(x)  # [batch_size, seq_len, embedding_dim]
        
        return x
    
    def get_attention_weights(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> list:
        batch_size, seq_len, _ = x.shape
        
        x = self.embedding_projection(x)
        
        t_embed = self.time_embedding(t.unsqueeze(-1))
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
