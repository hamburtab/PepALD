import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
import json

from helm_transformer import HELMTransformer
from molformer_embedding import MolFormerEmbedding
from helm_topology_analyzer import HELMTopologyAnalyzer


class HELMDiffusionModel(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        max_seq_len: int,
        dropout: float,
        num_diffusion_steps: int,
        variance_schedule: str,
        beta_start: float,
        beta_end: float
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_steps = num_diffusion_steps
        
        self.denoising_network = HELMTransformer(
            embedding_dim=embedding_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            dropout=dropout
        )
        
        self._setup_variance_schedule(variance_schedule, beta_start, beta_end)
        
    def _setup_variance_schedule(self, schedule_type: str, beta_start: float, beta_end: float):
        if schedule_type == "linear":
            betas = torch.linspace(beta_start, beta_end, self.num_steps) # The length of betas is num_steps
        elif schedule_type == "cosine":
            def cosine_beta_schedule(timesteps, s=0.008):
                steps = timesteps + 1
                x = torch.linspace(0, timesteps, steps)
                alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
                alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
                betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
                return torch.clip(betas, 0.0001, 0.9999)
            betas = cosine_beta_schedule(self.num_steps) 
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
            
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        
    def add_noise(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x_0) # [batch_size, seq_len, embedding_dim]
            
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t] # [batch_size]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
        
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.view(-1, 1, 1) # [batch_size, 1, 1]
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.view(-1, 1, 1)
        
        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
        
        return x_t, noise
    
    def predict_noise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        peptide_type: Optional[list] = None,
        connection_info: Optional[list] = None,
        helm_sequences: Optional[list] = None
    ) -> torch.Tensor:
        t_normalized = t.float() / self.num_steps
        transformer_output = self.denoising_network(x_t, t_normalized, mask=mask, 
                                                   peptide_type=peptide_type, 
                                                   connection_info=connection_info,
                                                   helm_sequences=helm_sequences)
        
        if isinstance(transformer_output, tuple):
            predicted_noise = transformer_output[0]
            ring_bond_loss = transformer_output[1] if len(transformer_output) > 1 else None
            if isinstance(ring_bond_loss, torch.Tensor) and ring_bond_loss.dim() == 0:
                return predicted_noise, ring_bond_loss
        return predicted_noise, None

    def forward(
        self,
        x_0: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        peptide_type: Optional[list] = None,
        connection_info: Optional[list] = None,
        helm_sequences: Optional[list] = None
    ) -> Dict[str, torch.Tensor]:
        batch_size = x_0.shape[0]
        device = x_0.device
        
        t = torch.randint(0, self.num_steps, (batch_size,), device=device)
        x_t, noise = self.add_noise(x_0, t)
        predicted_noise, ring_bond_loss = self.predict_noise(x_t, t, mask, 
                                           peptide_type=peptide_type, 
                                           connection_info=connection_info,
                                           helm_sequences=helm_sequences)
        
        if mask is not None:
            diffusion_loss = F.mse_loss(predicted_noise, noise, reduction='none')
            mask_expanded = mask.unsqueeze(-1).expand_as(diffusion_loss)
            diffusion_loss = (diffusion_loss * mask_expanded).sum() / mask_expanded.sum()
        else:
            diffusion_loss = F.mse_loss(predicted_noise, noise)
        total_loss = diffusion_loss
        if ring_bond_loss is not None:
            total_loss = diffusion_loss + 0.3 * ring_bond_loss

        return {
            'loss': total_loss,
            'diffusion_loss': diffusion_loss,
            'ring_bond_loss': ring_bond_loss if ring_bond_loss is not None else torch.tensor(0.0, device=device),
            'predicted_noise': predicted_noise,
            'target_noise': noise,
            't': t,
            'x_t': x_t
        }
    
    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int, int, int],  # (batch_size, seq_len, embedding_dim)
        mask: Optional[torch.Tensor] = None,
        device: str = 'cuda',
        guidance_scale: float = 1.0,
        eta: float = 0.0,
        peptide_type: Optional[list] = None,
        connection_info: Optional[list] = None
    ) -> torch.Tensor:
        batch_size, seq_len, embedding_dim = shape
        
        x_t = torch.randn(shape, device=device)
        
        for i in reversed(range(self.num_steps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            
            result = self.predict_noise(x_t, t, mask, 
                                      peptide_type=peptide_type, 
                                      connection_info=connection_info)
            
            if isinstance(result, tuple):
                predicted_noise, _ = result
            else:
                predicted_noise = result
            
            if i > 0:
                alpha_t = self.alphas[i]
                alpha_cumprod_t = self.alphas_cumprod[i]
                alpha_cumprod_t_prev = self.alphas_cumprod[i-1]
                beta_t = self.betas[i]
                
                # 预测x_0
                x_0_pred = (x_t - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)
                
                # 计算均值
                mean = (torch.sqrt(alpha_cumprod_t_prev) * beta_t) / (1 - alpha_cumprod_t) * x_0_pred + \
                       (torch.sqrt(alpha_t) * (1 - alpha_cumprod_t_prev)) / (1 - alpha_cumprod_t) * x_t
                
                # 添加噪声
                if eta > 0:
                    variance = eta * beta_t
                    noise = torch.randn_like(x_t)
                    x_t = mean + torch.sqrt(variance) * noise
                else:
                    x_t = mean
            else:
                alpha_cumprod_t = self.alphas_cumprod[i]
                x_t = (x_t - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)
                
        return x_t
    
class HELMDiffusion(nn.Module):
    def __init__(self, transformer, vocab_size: int, T: int = 1000, beta_schedule: str = "linear", 
                 vocab: Optional[Dict[str, int]] = None, use_molformer: bool = True,
                 beta_start: float = 1e-4, beta_end: float = 0.02, d_ff: int = 2048):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.T = T
        self.use_molformer = use_molformer
        self.vocab = vocab  # 保存词汇表引用
        
        if use_molformer and vocab is not None:
            self.embedding = MolFormerEmbedding(
                embeddings_dir="./molformer_embeddings",
                vocab=vocab,
                freeze_embeddings=False
            )
            embedding_dim = self.embedding.embedding_dim
        else:
            embedding_dim = transformer.embedding_dim
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.diffusion_model = HELMDiffusionModel(
            embedding_dim=embedding_dim,
            d_model=transformer.d_model,
            n_heads=transformer.n_heads,
            n_layers=transformer.n_layers,
            d_ff=d_ff,
            max_seq_len=transformer.max_seq_len,
            dropout=transformer.dropout,
            num_diffusion_steps=T,
            variance_schedule=beta_schedule,
            beta_start=beta_start,
            beta_end=beta_end
        )
        
        # 不再需要输出投影层，直接使用embedding距离
        # self.output_projection = nn.Linear(embedding_dim, vocab_size)
        
    def get_ground_truth_embeddings(self, x):
        if self.use_molformer:
            with torch.no_grad():
                ground_truth = self.embedding(x)
        else:
            ground_truth = self.embedding(x)
        return ground_truth
        
    def forward(self, x, mask=None, helm_sequences=None):
        target_embeddings = self.get_ground_truth_embeddings(x)
        
        if mask is None:
            mask = torch.ones(target_embeddings.shape[:2], device=target_embeddings.device)
        
        result = self.diffusion_model(target_embeddings, mask, helm_sequences=helm_sequences)
        return result
    
    def compute_loss(self, x, mask=None, helm_sequences=None):
        result = self.forward(x, mask, helm_sequences=helm_sequences)
        return result['loss']
    
    def sample(self, num_samples: int, max_seq_len: int, device=None, predict_ring_bonds=True):
        if device is None:
            device = next(self.parameters()).device
            
        shape = (num_samples, max_seq_len, self.diffusion_model.embedding_dim)
        
        samples = self.diffusion_model.sample(shape=shape, device=device)
        
        token_samples = self._embedding_to_tokens(samples)

        # 利用模型学到的结束位置：若某序列在某处预测为 <PAD>，则将其后的所有位置强制设为 <PAD>
        try:
            pad_id = self.vocab.get('<PAD>', 0) if isinstance(self.vocab, dict) else 0
            if isinstance(token_samples, torch.Tensor):
                # 对每个样本逐一处理（batch 通常不大，这里用简单循环保证清晰）
                for i in range(token_samples.shape[0]):
                    row = token_samples[i]
                    pad_positions = (row == pad_id).nonzero(as_tuple=False)
                    if pad_positions.numel() > 0:
                        first_pad = pad_positions[0].item()
                        if first_pad + 1 < row.shape[0]:
                            row[first_pad + 1:] = pad_id
        except Exception:
            # 若 vocab 或张量设备/形状异常，不影响后续流程
            pass
        
        if predict_ring_bonds:
            # 预测环键连接
            return self._predict_ring_bonds_for_samples(token_samples, samples, device)
        else:
            return token_samples
    
    def _embedding_to_tokens(self, embeddings):
        """通过计算与MolFormer embedding的距离来选择最近的token"""
        if self.use_molformer and hasattr(self.embedding, 'embedding_matrix'):
            # 获取MolFormer embedding矩阵，包括特殊token
            reference_embeddings = self.embedding.embedding_matrix.to(embeddings.device)
            num_regular = reference_embeddings.shape[0]
            if hasattr(self.embedding, 'special_embeddings'):
                reference_embeddings = torch.cat([
                    reference_embeddings,
                    self.embedding.special_embeddings.to(embeddings.device)
                ], dim=0)
                num_total = reference_embeddings.shape[0]
            else:
                num_total = num_regular
            
            # 计算距离: embeddings [batch, seq, 768], reference [vocab, 768]
            batch_size, seq_len, embed_dim = embeddings.shape
            embeddings_flat = embeddings.view(-1, embed_dim)  # [batch*seq, 768]
            
            # 计算距离矩阵 [batch*seq, vocab]
            distances = torch.cdist(embeddings_flat, reference_embeddings)
            
            # 选择最近的候选索引（在 reference_embeddings 上的行号）
            nn_indices = torch.argmin(distances, dim=1)  # [batch*seq]

            # 将 reference 索引映射回 vocab id（优先使用已有的 vocab_to_embedding 映射）
            if hasattr(self.embedding, 'vocab_to_embedding') and isinstance(self.vocab, dict):
                # 构建 embedding_idx -> vocab_id 的反向映射
                vocab_to_emb = self.embedding.vocab_to_embedding
                # 放到 CPU 便于遍历，最终再转回设备
                emb_to_vocab = torch.full((num_total,), -1, dtype=torch.long)
                # 先根据已有映射填充（第一次出现即记录）
                for vocab_id in range(vocab_to_emb.shape[0]):
                    emb_id = int(vocab_to_emb[vocab_id].item())
                    if 0 <= emb_id < num_total and emb_to_vocab[emb_id].item() == -1:
                        emb_to_vocab[emb_id] = vocab_id
                # 覆盖特殊token，确保特殊候选能映射到正确的 vocab id
                special_tokens = ['<PAD>', '<UNK>', '<START>', '<END>']
                for s_idx, s_tok in enumerate(special_tokens):
                    emb_id = num_regular + s_idx
                    if emb_id < num_total and s_tok in self.vocab:
                        emb_to_vocab[emb_id] = self.vocab[s_tok]
                # 回到正确设备
                emb_to_vocab = emb_to_vocab.to(embeddings.device)
                # 映射最近邻结果到 vocab id
                tokens = emb_to_vocab[nn_indices]
                # 对于仍未映射成功的位置，回退为最近邻索引（尽量维持旧行为）
                fallback_mask = tokens.lt(0)
                if fallback_mask.any():
                    # 简单回退：裁剪到 vocab 范围（可能与真实 vocab 不完全一致，但保持旧行为的鲁棒性）
                    tokens[fallback_mask] = torch.remainder(nn_indices[fallback_mask], self.vocab_size)
            else:
                tokens = nn_indices

            tokens = tokens.view(batch_size, seq_len)  # [batch, seq]
            return tokens
        else:
            # 简化的随机采样方法（如果没有MolFormer）
            batch_size, seq_len = embeddings.shape[:2]
            tokens = torch.randint(0, self.vocab_size, (batch_size, seq_len), device=embeddings.device)
            return tokens
    
    def _predict_ring_bonds_for_samples(self, token_samples, embeddings, device):
        """为生成的样本预测环键连接"""
        results = []
        
        with torch.no_grad():
            # 使用低噪声时间步来获取环键预测
            batch_size = token_samples.shape[0]
            t = torch.zeros(batch_size, device=device)  # 使用t=0（无噪声）
            
            # 直接使用已去噪的embeddings获取环键预测
            t_normalized = t.float() / self.diffusion_model.num_steps
            with torch.no_grad():
                # 只获取环键预测，不进行去噪计算
                _, ring_bond_predictions = self.diffusion_model.denoising_network(
                    embeddings, t_normalized, helm_sequences=None
                )
            
            for i in range(batch_size):
                tokens = token_samples[i]
                
                # 找到实际序列长度（到PAD为止）
                pad_id = self.vocab.get('<PAD>', 0)
                actual_len = (tokens != pad_id).sum().item()
                
                # 线性肽或无环键预测
                if actual_len <= 1 or ring_bond_predictions is None:
                    results.append({
                        'tokens': tokens,
                        'ring_connections': []
                    })
                    continue
                
                # 解析环键预测
                ring_connections = self._parse_ring_bond_predictions(
                    ring_bond_predictions[i], actual_len
                )
                
                results.append({
                    'tokens': tokens,
                    'ring_connections': ring_connections
                })
        
        return results
    
    def _parse_ring_bond_predictions(self, ring_predictions, seq_len):
        """解析环键预测结果为连接列表"""
        connections = []
        
        if ring_predictions is None:
            return connections
            
        try:
            # ring_predictions的形状为[num_pairs, 5]
            if ring_predictions.dim() == 2 and ring_predictions.shape[1] == 5:
                bond_probs = torch.softmax(ring_predictions, dim=1)
                
                # 找到所有pairs中概率最大的那个键
                max_probs, max_indices = torch.max(bond_probs.flatten(), dim=0)
                max_prob = max_probs.item()
                max_idx = max_indices.item()
                # 只有当最大概率 > 0.25 且不是无键类别(0)时才生成环肽
                if max_prob > 0.25:
                    pair_idx = max_idx // 5  # 哪个pair
                    bond_type_idx = max_idx % 5  # 键类型
                    
                    if bond_type_idx > 0:  # 不是无键
                        # 重构pair位置
                        count = 0
                        for i in range(seq_len):
                            for j in range(i + 1, seq_len):
                                if count == pair_idx:
                                    bond_types = ['R3R3', 'R1R2', 'R1R3', 'R3R2']
                                    connections.append({
                                        'res1': i + 1,
                                        'res2': j + 1,
                                        'bond_type': bond_types[bond_type_idx - 1]
                                    })
                                    return connections
                                count += 1
        
        except Exception as e:
            print(f"环键预测解析错误: {e}")
        
        return connections


class HELMSequenceDataset(torch.utils.data.Dataset):

    def __init__(self, data_file: str, max_seq_len: int = 45, vocab_file: str = "./data/helm_vocab.json"):
        self.data_file = data_file
        self.max_seq_len = max_seq_len
        self.topology_analyzer = HELMTopologyAnalyzer()
        
        with open(vocab_file, 'r') as f:
            self.vocab = json.load(f)
        
        self.idx_to_token = {v: k for k, v in self.vocab.items()}
        
        self.sequences = []
        with open(data_file, 'r') as f:
            for line in f:
                helm_seq = line.strip()
                if helm_seq:
                    self.sequences.append(helm_seq)
        
        print(f"加载 {len(self.sequences)} 个HELM序列")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        helm_seq = self.sequences[idx]
        
        tokens = self._parse_helm_sequence(helm_seq)

        # 去掉tokens中单体的[]
        tokens = [token[1:-1] if token.startswith('[') and token.endswith(']') else token for token in tokens]
        
        token_ids = self._tokens_to_ids(tokens)
        
        # 记录真实序列长度
        actual_len = len(token_ids)
        
        if len(token_ids) > self.max_seq_len:
            token_ids = token_ids[:self.max_seq_len]
            actual_len = self.max_seq_len
        else:
            pad_id = self.vocab.get('<PAD>', 0)
            token_ids.extend([pad_id] * (self.max_seq_len - len(token_ids)))
        
        # 创建mask: 真实token + 第一个PAD(作为END) = 1，后续PAD = 0
        mask = torch.zeros(self.max_seq_len, dtype=torch.float)
        if actual_len < self.max_seq_len:
            mask[:actual_len + 1] = 1.0
        else:
            # 序列已满，全部为真实token（没有结束符）
            mask[:actual_len] = 1.0
        
        return {
            'token_ids': torch.tensor(token_ids, dtype=torch.long),
            'mask': mask,  # 修正的mask
            'helm_sequence': helm_seq
        }

    def _parse_helm_sequence(self, helm_seq: str):
        parsed_result = self.topology_analyzer.parse_helm_sequence(helm_seq)
        sequence = parsed_result['sequence']
        return sequence.split('.') if sequence else []
    
    def _tokens_to_ids(self, tokens):
        unk_id = self.vocab.get('<UNK>', 1)
        return [self.vocab.get(token, unk_id) for token in tokens]
    
    def decode_sequence(self, token_ids, ring_connections=None):
        """解码token序列为HELM字符串
        遇到PAD就停止解码
        如果提供ring_connections，则生成包含环键的HELM序列
        """
        tokens = []
        for token_id in token_ids:
            if isinstance(token_id, torch.Tensor):
                token_id = token_id.item()
            
            if token_id in self.idx_to_token:
                token = self.idx_to_token[token_id]
                if token == '<PAD>':
                    break
                elif token not in ['<UNK>', '<START>', '<END>']:
                    tokens.append(token)
        
        if not tokens:
            return "PEPTIDE1{}$$$$"
        
        # 构建基本序列部分
        sequence_part = f"PEPTIDE1{{{'.'.join(tokens)}}}"
        
        # 如果有环键连接，添加连接信息
        if ring_connections and len(ring_connections) > 0:
            connection_parts = []
            for conn in ring_connections:
                res1, res2 = conn['res1'], conn['res2']
                bond_type = conn['bond_type']
                
                # 构建连接字符串格式: PEPTIDE1,PEPTIDE1,res1:bond1-res2:bond2
                if bond_type == 'R3R3':
                    conn_str = f"PEPTIDE1,PEPTIDE1,{res1}:R3-{res2}:R3"
                elif bond_type == 'R1R2':
                    conn_str = f"PEPTIDE1,PEPTIDE1,{res1}:R1-{res2}:R2"
                elif bond_type == 'R1R3':
                    conn_str = f"PEPTIDE1,PEPTIDE1,{res1}:R1-{res2}:R3"
                elif bond_type == 'R3R2':
                    conn_str = f"PEPTIDE1,PEPTIDE1,{res1}:R3-{res2}:R2"
                else:
                    continue
                    
                connection_parts.append(conn_str)
            
            if connection_parts:
                connection_str = '|'.join(connection_parts)
                return f"{sequence_part}${connection_str}$$$"
        
        # 无环键的情况
        return f"{sequence_part}$$$$"
