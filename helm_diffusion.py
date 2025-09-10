import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Tuple
import pickle
import json

from helm_transformer import HELMTransformer
from molformer_embedding import MolFormerEmbedding
from helm_topology_analyzer import HELMTopologyAnalyzer


class HELMDiffusionModel(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 768,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        max_seq_len: int = 256,
        dropout: float = 0.1,
        num_diffusion_steps: int = 1000,
        variance_schedule: str = "linear",
        beta_start: float = 0.0001,
        beta_end: float = 0.02
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
            betas = torch.linspace(beta_start, beta_end, self.num_steps)
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
            noise = torch.randn_like(x_0)
            
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
        
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.view(-1, 1, 1)
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
        
        # HELMTransformer 返回 (predicted_noise, ring_bond_loss/embedding)
        if isinstance(transformer_output, tuple):
            predicted_noise = transformer_output[0]
        else:
            predicted_noise = transformer_output
            
        return predicted_noise

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
        predicted_noise = self.predict_noise(x_t, t, mask, 
                                           peptide_type=peptide_type, 
                                           connection_info=connection_info,
                                           helm_sequences=helm_sequences)
        
        if mask is not None:
            loss = F.mse_loss(predicted_noise, noise, reduction='none')
            mask_expanded = mask.unsqueeze(-1).expand_as(loss)
            loss = (loss * mask_expanded).sum() / mask_expanded.sum()
        else:
            loss = F.mse_loss(predicted_noise, noise)
            
        return {
            'loss': loss,
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
            
            predicted_noise = self.predict_noise(x_t, t, mask, 
                                                peptide_type=peptide_type, 
                                                connection_info=connection_info)
            
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
    
    @torch.no_grad()
    def ddim_sample(
        self,
        shape: Tuple[int, int, int],
        mask: Optional[torch.Tensor] = None,
        # device: str = 'cuda',
        device: str = 'cpu',
        num_steps: int = 50,
        eta: float = 0.0
    ) -> torch.Tensor:
        batch_size, seq_len, embedding_dim = shape
        
        step_size = self.num_steps // num_steps
        timesteps = list(range(0, self.num_steps, step_size))[:num_steps]
        timesteps = timesteps[::-1]
        
        x_t = torch.randn(shape, device=device)
        
        for i, t in enumerate(timesteps):
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            predicted_noise = self.predict_noise(x_t, t_tensor, mask)
            
            alpha_cumprod_t = self.alphas_cumprod[t]
            
            if i < len(timesteps) - 1:
                alpha_cumprod_t_prev = self.alphas_cumprod[timesteps[i+1]]
            else:
                alpha_cumprod_t_prev = torch.tensor(1.0, device=device)
            
            x_0_pred = (x_t - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)
            
            direction_to_x_t = torch.sqrt(1 - alpha_cumprod_t_prev - eta**2 * (1 - alpha_cumprod_t_prev)) * predicted_noise
            
            x_t = torch.sqrt(alpha_cumprod_t_prev) * x_0_pred + direction_to_x_t
            
            if eta > 0 and i < len(timesteps) - 1:
                noise = torch.randn_like(x_t)
                x_t += eta * torch.sqrt(1 - alpha_cumprod_t_prev) * noise
                
        return x_t
    
class HELMDiffusion(nn.Module):
    def __init__(self, transformer, vocab_size: int, T: int = 1000, beta_schedule: str = "linear", 
                 vocab: Optional[Dict[str, int]] = None, use_molformer: bool = True):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.T = T
        self.use_molformer = use_molformer
        
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
            max_seq_len=transformer.max_seq_len,
            dropout=transformer.dropout,
            num_diffusion_steps=T,
            variance_schedule=beta_schedule
        )
        
        self.output_projection = nn.Linear(embedding_dim, vocab_size)
        
    def get_ground_truth_embeddings(self, x):
        if self.use_molformer:
            with torch.no_grad():
                ground_truth = self.embedding(x)
        else:
            ground_truth = self.embedding(x)
        return ground_truth
        
    def forward(self, x, mask=None):
        target_embeddings = self.get_ground_truth_embeddings(x)
        
        if mask is None:
            mask = torch.ones(target_embeddings.shape[:2], device=target_embeddings.device)
        
        result = self.diffusion_model(target_embeddings, mask)
        return result
    
    def compute_loss(self, x, mask=None):
        result = self.forward(x, mask)
        return result['loss']
    
    def sample(self, num_samples: int, max_seq_len: int, device=None):
        if device is None:
            device = next(self.parameters()).device
            
        shape = (num_samples, max_seq_len, self.diffusion_model.embedding_dim)
        
        samples = self.diffusion_model.sample(shape=shape, device=device)
        
        token_samples = self._embedding_to_tokens(samples)
        return token_samples
    
    def sample_ddim(self, num_samples: int, max_seq_len: int, ddim_steps: int = 50, eta: float = 0.0):
        device = next(self.parameters()).device
        shape = (num_samples, max_seq_len, self.diffusion_model.embedding_dim)
        
        samples = self.diffusion_model.ddim_sample(
            shape=shape, 
            num_steps=ddim_steps,
            device=device
        )
        
        token_samples = self._embedding_to_tokens(samples)
        return token_samples
    
    def _embedding_to_tokens(self, embeddings):
        logits = self.output_projection(embeddings)
        tokens = torch.argmax(logits, dim=-1)
        return tokens


class HELMSequenceDataset(torch.utils.data.Dataset):

    def __init__(self, data_file: str, max_seq_len: int = 128, vocab_file: str = "./data/helm_vocab.json"):
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
        
        token_ids = self._tokens_to_ids(tokens)
        
        if len(token_ids) > self.max_seq_len:
            token_ids = token_ids[:self.max_seq_len]
        else:
            pad_id = self.vocab.get('<PAD>', 0)
            token_ids.extend([pad_id] * (self.max_seq_len - len(token_ids)))
        
        return torch.tensor(token_ids, dtype=torch.long)
    
    def _parse_helm_sequence(self, helm_seq: str):
        parsed_result = self.topology_analyzer.parse_helm_sequence(helm_seq)
        sequence = parsed_result['sequence']
        return sequence.split('.') if sequence else []
    
    def _tokens_to_ids(self, tokens):
        unk_id = self.vocab.get('<UNK>', 1)
        return [self.vocab.get(token, unk_id) for token in tokens]
    
    def decode_sequence(self, token_ids):
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
        
        if tokens:
            return f"PEPTIDE1{{{'.'.join(tokens)}}}$$$$"
        else:
            return "PEPTIDE1{}$$$$"
