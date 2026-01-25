"""
使用化学知识和多模态信息预测环肽成环位置：
1. 筛选化学上可配对的单体
2. 融合全局上下文、化学特征和attention信息(context_embedding+unimol_embeddings+attention_weights)
3. 使用门控机制调制化学特征
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
import pandas as pd
import numpy as np
from pathlib import Path


class ChemicallyInformedRingPredictor(nn.Module):
    """
    1. 降维投影（512→256）
    2. 化学配对网络（融合两个单体特征）
    3. Attention门控（1→256，调制化学特征）
    4. 跨模态融合（上下文+化学→128）
    5. 分类头（预测5种bond types）
    """
    
    BOND_TYPES = ['none', 'R3R3', 'R1R2', 'R1R3', 'R3R2']
    
    def __init__(
        self,
        d_model: int = 512,
        unimol_dim: int = 512,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        monomer_library_path: str = "./data/monomer_library.csv",
        embeddings_dir: str = "./unimol_embeddings"
    ):
        """
        Args:
            d_model: Context encoder的输出维度
            unimol_dim: UniMol embedding维度
            hidden_dim: 隐藏层维度
            dropout: Dropout概率
            monomer_library_path: 单体库CSV路径
            embeddings_dir: UniMol embeddings目录
        """
        super().__init__()
        
        self.d_model = d_model
        self.unimol_dim = unimol_dim
        self.hidden_dim = hidden_dim
        
        # 加载单体化学信息和UniMol embeddings
        self.monomer_info = self._load_monomer_library(monomer_library_path)
        self.unimol_embeddings = self._load_unimol_embeddings(embeddings_dir)
        
        # === 第1层：降维投影 ===
        self.context_proj = nn.Linear(d_model, hidden_dim)
        self.unimol_proj = nn.Linear(unimol_dim, hidden_dim)
        
        # === 第2层：化学配对网络 ===
        self.chemical_pair_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # === 第3层：Attention门控网络 ===
        self.attention_gate = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Sigmoid()  # 输出[0,1]的门控值
        )
        
        # === 第4层：跨模态融合网络 ===
        self.cross_modal_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU()
        )
        
        # === 第5层：分类头 ===
        self.bond_classifier = nn.Linear(hidden_dim // 2, 5)
    
    def _load_monomer_library(self, path: str) -> Dict[str, Dict[str, bool]]:
        """加载单体库的R基团信息"""
        try:
            df = pd.read_csv(path)
            monomer_info = {}
            
            for _, row in df.iterrows():
                symbol = row['Symbol']
                monomer_info[symbol] = {
                    'R1': row['R1'] != '-',
                    'R2': row['R2'] != '-',
                    'R3': row['R3'] != '-'
                }
            
            print(f"[RingPredictor] Loaded {len(monomer_info)} monomers with R-group info")
            return monomer_info
            
        except Exception as e:
            print(f"[RingPredictor] Warning: Failed to load monomer library: {e}")
            return {}
    
    def _load_unimol_embeddings(self, embeddings_dir: str) -> Optional[torch.Tensor]:
        """加载UniMol embeddings矩阵"""
        try:
            emb_path = Path(embeddings_dir) / "embeddings_matrix.npy"
            embeddings = np.load(emb_path)
            embeddings_tensor = torch.from_numpy(embeddings).float()
            
            print(f"[RingPredictor] Loaded UniMol embeddings: {embeddings_tensor.shape}")
            return embeddings_tensor
            
        except Exception as e:
            print(f"[RingPredictor] Warning: Failed to load UniMol embeddings: {e}")
            return None
    
    def _can_pair(self, r_i: Dict[str, bool], r_j: Dict[str, bool]) -> bool:
        """
        判断两个单体是否可以化学配对
        
        配对规则：
        - R1-R2: 主链连接（头尾成环）
        - R3-R3: 侧链桥接
        - R1-R3: 混合连接
        - R3-R2: 混合连接
        """
        # R1-R2配对（最常见：头尾成环）
        if (r_i['R1'] and r_j['R2']) or (r_i['R2'] and r_j['R1']):
            return True
        
        # R3-R3配对（侧链交联）
        if r_i['R3'] and r_j['R3']:
            return True
        
        # R1-R3配对
        if (r_i['R1'] and r_j['R3']) or (r_i['R3'] and r_j['R1']):
            return True
        
        # R3-R2配对
        if (r_i['R3'] and r_j['R2']) or (r_i['R2'] and r_j['R3']):
            return True
        
        return False
    
    def filter_candidate_pairs(
        self,
        sequence_symbols: List[str],
        min_distance: int = 2
    ) -> List[Tuple[int, int]]:
        """
        筛选化学上可配对的单体对
        
        Args:
            sequence_symbols: 单体符号列表，如 ['A', 'L', 'Y', 'K']
            min_distance: 最小间隔距离（至少间隔几个残基）
            
        Returns:
            候选配对列表 [(i, j), ...]，其中 i < j
        """
        candidates = []
        n = len(sequence_symbols)
        
        for i in range(n):
            for j in range(i + min_distance, n):
                mono_i = sequence_symbols[i]
                mono_j = sequence_symbols[j]
                
                # 检查是否在单体库中
                if mono_i not in self.monomer_info or mono_j not in self.monomer_info:
                    continue
                
                r_i = self.monomer_info[mono_i]
                r_j = self.monomer_info[mono_j]
                
                # 化学配对可行性检查
                if self._can_pair(r_i, r_j):
                    candidates.append((i, j))
        
        return candidates
    
    def forward(
        self,
        global_context: torch.Tensor,
        attention_weights: torch.Tensor,
        token_ids: torch.Tensor,
        sequence_symbols: List[List[str]],
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        预测成环位置
        
        Args:
            global_context: 全局上下文向量 [batch_size, d_model]
            attention_weights: Attention权重 [batch_size, n_heads, seq_len, seq_len]
            token_ids: Token IDs [batch_size, seq_len]
            sequence_symbols: 单体符号列表 [batch_size, seq_len]
            mask: 序列mask [batch_size, seq_len]
            
        Returns:
            {
                'candidate_pairs': List[List[Tuple]], 每个样本的候选配对
                'bond_logits': [batch_size, max_candidates, 5]
                'bond_probs': [batch_size, max_candidates, 5]
            }
        """
        batch_size = global_context.size(0)
        device = global_context.device
        
        # 检查UniMol embeddings是否加载
        if self.unimol_embeddings is None:
            return self._empty_result(batch_size, device)
        
        # 将UniMol embeddings移到正确的设备
        if self.unimol_embeddings.device != device:
            self.unimol_embeddings = self.unimol_embeddings.to(device)
        
        # 平均所有attention heads
        avg_attn = attention_weights.mean(dim=1)  # [batch_size, seq_len, seq_len]
        
        # 对每个样本处理
        all_logits = []
        all_pairs = []
        
        for b in range(batch_size):
            # 获取实际序列长度
            if mask is not None:
                actual_len = int(mask[b].sum().item())
            else:
                actual_len = len(sequence_symbols[b])
            
            # 筛选候选配对
            candidates = self.filter_candidate_pairs(
                sequence_symbols[b][:actual_len]
            )
            
            if not candidates:
                all_pairs.append([])
                continue
            
            # 为每个候选配对计算logits
            batch_logits = []
            
            for i, j in candidates:
                # 1. 全局上下文（已经是压缩的表示）
                ctx = global_context[b]  # [d_model]
                
                # 2. Attention权重
                attn_score = avg_attn[b, i, j].unsqueeze(0)  # [1]
                
                # 3. UniMol embeddings
                token_i = token_ids[b, i].item()
                token_j = token_ids[b, j].item()
                
                # 确保token_id在范围内
                if token_i >= self.unimol_embeddings.size(0) or token_j >= self.unimol_embeddings.size(0):
                    continue
                
                unimol_i = self.unimol_embeddings[token_i]  # [unimol_dim]
                unimol_j = self.unimol_embeddings[token_j]  # [unimol_dim]
                
                # === 方案A的5层处理 ===
                
                # 第1层：降维投影
                ctx_proj = self.context_proj(ctx)          # [hidden_dim]
                chem_i = self.unimol_proj(unimol_i)       # [hidden_dim]
                chem_j = self.unimol_proj(unimol_j)       # [hidden_dim]
                
                # 第2层：化学配对
                chemical_pair = torch.cat([chem_i, chem_j], dim=0)  # [hidden_dim * 2]
                chemical_feat = self.chemical_pair_net(chemical_pair)  # [hidden_dim]
                
                # 第3层：Attention门控
                attn_gate = self.attention_gate(attn_score)  # [hidden_dim]
                chemical_feat_gated = chemical_feat * attn_gate  # 逐元素相乘
                
                # 第4层：跨模态融合
                fused = torch.cat([ctx_proj, chemical_feat_gated], dim=0)  # [hidden_dim * 2]
                fused = self.cross_modal_fusion(fused)  # [hidden_dim // 2]
                
                # 第5层：分类
                logits = self.bond_classifier(fused)  # [5]
                batch_logits.append(logits)
            
            if batch_logits:
                all_logits.append(torch.stack(batch_logits))
                all_pairs.append(candidates)
        
        # 处理结果
        if not all_logits:
            return self._empty_result(batch_size, device)
        
        # Padding到统一长度
        max_candidates = max(logits.size(0) for logits in all_logits)
        
        padded_logits = []
        for logits in all_logits:
            if logits.size(0) < max_candidates:
                padding = torch.full(
                    (max_candidates - logits.size(0), 5),
                    float('-inf'),
                    device=device
                )
                logits = torch.cat([logits, padding], dim=0)
            padded_logits.append(logits)
        
        bond_logits = torch.stack(padded_logits)  # [batch_size, max_candidates, 5]
        bond_probs = F.softmax(bond_logits, dim=-1)
        
        return {
            'candidate_pairs': all_pairs,
            'bond_logits': bond_logits,
            'bond_probs': bond_probs
        }
    
    def _empty_result(self, batch_size: int, device: torch.device) -> Dict:
        """返回空结果"""
        return {
            'candidate_pairs': [[] for _ in range(batch_size)],
            'bond_logits': torch.zeros(batch_size, 0, 5, device=device),
            'bond_probs': torch.zeros(batch_size, 0, 5, device=device)
        }
    
    def predict_bonds(
        self,
        global_context: torch.Tensor,
        attention_weights: torch.Tensor,
        token_ids: torch.Tensor,
        sequence_symbols: List[str],
        threshold: float = 0.5,
        max_bonds: int = 1
    ) -> List[Dict]:
        """
        生成阶段的成环预测
        
        Args:
            global_context: [1, d_model]
            attention_weights: [1, n_heads, seq_len, seq_len]
            token_ids: [1, seq_len]
            sequence_symbols: 单体符号列表
            threshold: 置信度阈值
            max_bonds: 最多预测几个环
            
        Returns:
            预测的环连接列表
        """
        result = self.forward(
            global_context,
            attention_weights,
            token_ids,
            [sequence_symbols]
        )
        
        candidates = result['candidate_pairs'][0]
        probs = result['bond_probs'][0]  # [num_candidates, 5]
        
        if len(candidates) == 0:
            return []
        
        bonds = []
        
        for idx, (i, j) in enumerate(candidates):
            # 获取最大概率的bond type（排除no-bond）
            bond_probs = probs[idx, 1:]  # 排除class 0
            max_prob, max_type = bond_probs.max(dim=0)
            
            max_prob = max_prob.item()
            max_type = max_type.item() + 1  # 加回偏移
            
            if max_prob > threshold:
                bonds.append({
                    'res1': i + 1,  # 1-indexed
                    'res2': j + 1,
                    'bond_type': self.BOND_TYPES[max_type],
                    'confidence': max_prob
                })
        
        # 按置信度排序，取top-k
        bonds = sorted(bonds, key=lambda x: -x['confidence'])[:max_bonds]
        
        return bonds
