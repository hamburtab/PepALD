"""
Autoregressive Latent Diffusion (ALD) Model for HELM Peptide Generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List

from .context_encoder import CausalContextEncoder
from .token_mapper import TokenMapper
from .ring_predictor import RingBondPredictor, AutoregressiveRingPredictor
from ..diffusion.engine import DiffusionEngine
from ..config import ALDConfig


class AutoregressiveLatentDiffusion(nn.Module):
    """
    Token-by-token peptide generation via diffusion in latent space.
    
    For each position t:
        1. h_t = ContextEncoder([x_0, ..., x_{t-1}])
        2. z_t = DiffusionEngine.sample(context=h_t)
        3. x_t = TokenMapper(z_t)
    """
    
    def __init__(
        self,
        vocab: Dict[str, int],
        config: Optional[ALDConfig] = None,
        embeddings_dir: Optional[str] = None,
        data_dir: Optional[str] = None,
        verbose: bool = True
    ):
        super().__init__()
        
        if config is None:
            config = ALDConfig()
        
        self.config = config
        model_cfg = config.model
        gen_cfg = config.generation
        train_cfg = config.training
        
        embeddings_dir = embeddings_dir or train_cfg.embeddings_dir
        data_dir = data_dir or train_cfg.data_dir
        
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.embedding_dim = model_cfg.embedding_dim
        self.d_model = model_cfg.d_model
        self.max_seq_len = model_cfg.max_seq_len
        self.num_diffusion_steps = model_cfg.num_diffusion_steps
        
        self.idx_to_token = {v: k for k, v in vocab.items()}
        self.pad_id = vocab.get('<PAD>', self.vocab_size - 1)
        
        # 1. Context Encoder
        self.context_encoder = CausalContextEncoder(
            embedding_dim=model_cfg.embedding_dim,
            d_model=model_cfg.d_model,
            n_heads=model_cfg.n_heads,
            n_layers=model_cfg.context_layers,
            d_ff=model_cfg.d_ff,
            max_seq_len=model_cfg.max_seq_len,
            dropout=model_cfg.dropout,
            embeddings_dir=embeddings_dir,
            freeze_embeddings=True
        )
        
        # Update embedding_dim from actual loaded embeddings
        actual_embed_dim = self.context_encoder.embedding.embedding_dim
        if actual_embed_dim != model_cfg.embedding_dim:
            self.embedding_dim = actual_embed_dim
        
        # 2. Diffusion Engine
        self.diffusion_engine = DiffusionEngine(
            embedding_dim=self.embedding_dim,
            d_model=model_cfg.d_model,
            n_heads=model_cfg.n_heads,
            n_layers=model_cfg.denoiser_layers,
            d_ff=model_cfg.d_ff,
            dropout=model_cfg.dropout,
            num_diffusion_steps=model_cfg.num_diffusion_steps,
            variance_schedule=model_cfg.variance_schedule,
            beta_start=model_cfg.beta_start,
            beta_end=model_cfg.beta_end
        )
        
        # 3. Token Mapper
        self.token_mapper = TokenMapper(
            vocab=vocab,
            embeddings_dir=embeddings_dir,
            data_dir=data_dir,
            use_embedding_norm=gen_cfg.use_embedding_norm
        )
        
        # 4. Ring Bond Predictors
        self.ring_predictor = RingBondPredictor(
            d_model=model_cfg.d_model,
            hidden_dim=model_cfg.d_model // 2,
            num_bond_types=5
        )
        self.ar_ring_predictor = AutoregressiveRingPredictor(
            d_model=model_cfg.d_model,
            hidden_dim=model_cfg.d_model // 2,
            num_bond_types=5
        )
        
        # 5. LM Head for Hybrid Modeling (Next Token Prediction)
        self.lm_head = nn.Linear(self.d_model, self.vocab_size)
        
        if verbose:
            self._print_model_info(model_cfg)
    
    def _print_model_info(self, model_cfg):
        """Print model configuration summary."""
        print(f"[ALD] Model: d={model_cfg.d_model}, layers={model_cfg.context_layers}+{model_cfg.denoiser_layers}, T={model_cfg.num_diffusion_steps}")
    
    def _empty_loss(self, device: torch.device) -> Dict[str, torch.Tensor]:
        """Return zero loss dictionary for edge cases."""
        return {
            'loss': torch.tensor(0.0, device=device, requires_grad=True),
            'diffusion_loss': torch.tensor(0.0, device=device),
            'ring_bond_loss': torch.tensor(0.0, device=device),
            'ce_loss': torch.tensor(0.0, device=device)
        }
    
    def _prepare_contexts(
        self,
        token_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> tuple:
        """
        Prepare ground truth embeddings and shifted contexts for training.
        
        Returns:
            gt_embeddings: [batch_size, seq_len, embedding_dim]
            contexts_for_pred: [batch_size, seq_len, d_model] (shifted)
        """
        gt_embeddings = self.context_encoder.get_token_embedding(token_ids)
        full_contexts = self.context_encoder(token_ids, mask)
        
        # Start context for position 0
        start_context = self.context_encoder.get_context_for_next_token(
            gt_embeddings[:, :0, :]
        )
        
        # Shift contexts: context[t] predicts position t
        contexts_for_pred = torch.cat([
            start_context.unsqueeze(1),
            full_contexts[:, :-1, :]
        ], dim=1)
        
        return gt_embeddings, contexts_for_pred
        
    def forward(
        self,
        token_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Training forward pass with teacher forcing (all positions)."""
        batch_size, seq_len = token_ids.shape
        device = token_ids.device
        
        if seq_len <= 1:
            return self._empty_loss(device)
        
        gt_embeddings, contexts_for_pred = self._prepare_contexts(token_ids, mask)
        
        # Create mask for valid positions
        if mask is not None:
            valid_mask = mask.bool()
        else:
            valid_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
        
        # Flatten valid positions
        target_flat = gt_embeddings[valid_mask]
        context_flat = contexts_for_pred[valid_mask].unsqueeze(1)
        
        # Diffusion loss
        diff_result = self.diffusion_engine.training_step(target_flat, context_flat)
        diffusion_loss = diff_result['loss']
        
        # Auxiliary Next Token Prediction Loss (Hybrid Modeling)
        token_logits = self.lm_head(contexts_for_pred)  # [Batch, Seq, Vocab]
        active_logits = token_logits[valid_mask]
        active_labels = token_ids[valid_mask]
        ce_loss = F.cross_entropy(active_logits, active_labels)
        
        # TODO: Add ring bond supervision
        ring_bond_loss = torch.tensor(0.0, device=device)
        
        return {
            'loss': diffusion_loss + 0.1 * ring_bond_loss + 0.5 * ce_loss,
            'diffusion_loss': diffusion_loss,
            'ring_bond_loss': ring_bond_loss,
            'ce_loss': ce_loss
        }
    
    def forward_efficient(
        self,
        token_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        sample_positions: int = 5
    ) -> Dict[str, torch.Tensor]:
        """Efficient training by sampling random positions per sequence."""
        batch_size, seq_len = token_ids.shape
        device = token_ids.device
        
        if seq_len <= 1:
            return self._empty_loss(device)
        
        gt_embeddings, contexts_for_pred = self._prepare_contexts(token_ids, mask)
        
        # Get sequence lengths
        if mask is not None:
            lengths = mask.sum(dim=1).long()
        else:
            lengths = torch.full((batch_size,), seq_len, dtype=torch.long, device=device)
        
        # Sample random positions
        sampled_targets = []
        sampled_contexts = []
        sampled_token_ids = []
        
        for b in range(batch_size):
            seq_length = lengths[b].item()
            if seq_length <= 0:
                continue
            
            num_samples = min(sample_positions, seq_length)
            positions = torch.randperm(seq_length, device=device)[:num_samples]
            
            sampled_targets.append(gt_embeddings[b, positions, :])
            sampled_contexts.append(contexts_for_pred[b, positions, :])
            sampled_token_ids.append(token_ids[b, positions])
        
        if len(sampled_targets) == 0:
            return self._empty_loss(device)
        
        target_flat = torch.cat(sampled_targets, dim=0)
        context_flat = torch.cat(sampled_contexts, dim=0).unsqueeze(1)
        token_ids_flat = torch.cat(sampled_token_ids, dim=0)
        
        diff_result = self.diffusion_engine.training_step(target_flat, context_flat)
        diffusion_loss = diff_result['loss']
        
        # Auxiliary Next Token Prediction Loss (Hybrid Modeling)
        token_logits = self.lm_head(context_flat.squeeze(1))
        ce_loss = F.cross_entropy(token_logits, token_ids_flat)
        
        # TODO: Add ring bond supervision
        ring_bond_loss = torch.tensor(0.0, device=device)
        
        return {
            'loss': diffusion_loss + 0.1 * ring_bond_loss + 0.5 * ce_loss,
            'diffusion_loss': diffusion_loss,
            'ring_bond_loss': ring_bond_loss,
            'ce_loss': ce_loss
        }
    
    @torch.no_grad()
    def sample(
        self,
        num_samples: Optional[int] = None,
        max_seq_len: Optional[int] = None,
        min_seq_len: Optional[int] = None,
        device: Optional[torch.device] = None,
        use_ddim: Optional[bool] = None,
        ddim_steps: Optional[int] = None,
        lambda_gpt: float = 0.8,
        predict_ring_bonds: Optional[bool] = None,
        verbose: bool = False
    ) -> List[Dict]:
        """
        Batch-parallel generation with optional Hybrid Sampling (Diffusion + GPT).
        """
        gen_cfg = self.config.generation
        
        # Use config defaults if not specified
        num_samples = num_samples if num_samples is not None else gen_cfg.num_samples
        max_seq_len = max_seq_len if max_seq_len is not None else gen_cfg.max_length
        min_seq_len = min_seq_len if min_seq_len is not None else gen_cfg.min_length
        use_ddim = use_ddim if use_ddim is not None else gen_cfg.use_ddim
        ddim_steps = ddim_steps if ddim_steps is not None else gen_cfg.ddim_steps
        predict_ring_bonds = predict_ring_bonds if predict_ring_bonds is not None else gen_cfg.predict_ring_bonds
        
        if device is None:
            device = next(self.parameters()).device
        
        self.eval()
        
        # Determine target lengths for each sample
        if min_seq_len is not None:
            lengths = torch.randint(min_seq_len, max_seq_len + 1, (num_samples,), device=device)
        else:
            lengths = torch.full((num_samples,), max_seq_len, device=device)
        
        # Storage: [num_samples, max_seq_len, embedding_dim]
        all_embeddings = torch.zeros(num_samples, max_seq_len, self.embedding_dim, device=device)
        all_tokens = torch.full((num_samples, max_seq_len), self.pad_id, dtype=torch.long, device=device)
        active_mask = torch.ones(num_samples, dtype=torch.bool, device=device)
        
        # Generate token by token (parallel across samples)
        for t in range(max_seq_len):
            # Check which samples are still active (not yet reached their target length)
            active_mask = t < lengths
            num_active = active_mask.sum().item()
            if num_active == 0:
                break
            
            # Get active indices
            active_idx = active_mask.nonzero(as_tuple=True)[0]
            
            # 1. Get context for active samples
            if t == 0:
                history = torch.zeros(num_active, 0, self.embedding_dim, device=device)
            else:
                history = all_embeddings[active_idx, :t, :]  # [num_active, t, embedding_dim]
            
            context = self.context_encoder.get_context_for_next_token(history)  # [num_active, d_model]
            
            # 2. Diffusion Generation
            context_cond = context.unsqueeze(1)  # [num_active, 1, d_model]
            if use_ddim:
                embeddings = self.diffusion_engine.sample_ddim(
                    batch_size=num_active, context=context_cond, device=device,
                    num_inference_steps=ddim_steps
                )
            else:
                embeddings = self.diffusion_engine.sample(
                    batch_size=num_active, context=context_cond, device=device
                )
            embeddings = embeddings.squeeze(1)  # [num_active, embedding_dim]
            
            # 3. Joint Decision (Hybrid Sampling)
            if lambda_gpt > 0.0:
                # A. Diffusion Distance Score (lower is better)
                dists = self.token_mapper._compute_distances(embeddings)
                
                # B. GPT Probability Score (higher is better -> -log_prob lower is better)
                gpt_logits = self.lm_head(context)
                gpt_log_probs = F.log_softmax(gpt_logits, dim=-1)
                
                # C. Fused Score: Score = Dist - lambda * LogProb
                final_scores = dists - lambda_gpt * gpt_log_probs
                
                # D. Apply Mask & Select (with chemical constraints)
                token_ids = torch.zeros(num_active, dtype=torch.long, device=device)
                for i in range(num_active):
                    # Get allowed tokens for this specific sample based on position t and its total length
                    current_seq_len = lengths[active_idx[i]].item()
                    allowed = self.token_mapper._get_allowed_tokens(t, current_seq_len)
                    
                    # Select best token ONLY from allowed list
                    # final_scores[i, allowed] extracts scores for allowed tokens
                    best_idx_in_allowed = torch.argmin(final_scores[i, allowed]).item()
                    token_ids[i] = allowed[best_idx_in_allowed]
            else:
                # Pure Diffusion
                token_ids = self.token_mapper.batch_map(
                    embeddings, positions=t, seq_lens=lengths[active_idx]
                )
            
            # Store results
            all_embeddings[active_idx, t, :] = embeddings
            all_tokens[active_idx, t] = token_ids
            
            if verbose and (t + 1) % 5 == 0:
                print(f"  Step {t+1}/{max_seq_len}, active samples: {num_active}")
        
        # Build results
        results = []
        for i in range(num_samples):
            seq_len = lengths[i].item()
            results.append({
                'tokens': all_tokens[i, :seq_len],
                'embeddings': all_embeddings[i, :seq_len, :],
                'length': seq_len
            })
        
        return results
    
    def decode_to_helm(
        self,
        tokens: torch.Tensor,
        ring_connections: Optional[List[Dict]] = None
    ) -> str:
        """Decode token IDs to HELM string."""
        symbols = []
        for token_id in tokens.tolist():
            if token_id == self.pad_id:
                break
            symbols.append(self.idx_to_token.get(token_id, '?'))
        
        if not symbols:
            return "PEPTIDE1{}$$$$"
        
        sequence_part = f"PEPTIDE1{{{'.'.join(symbols)}}}"
        
        if ring_connections:
            conn_strings = []
            for conn in ring_connections:
                res1, res2 = conn['res1'], conn['res2']
                bond_type = conn['bond_type']
                
                bond_map = {
                    'R3R3': f"{res1}:R3-{res2}:R3",
                    'R1R2': f"{res1}:R1-{res2}:R2",
                    'R1R3': f"{res1}:R1-{res2}:R3",
                    'R3R2': f"{res1}:R3-{res2}:R2"
                }
                if bond_type in bond_map:
                    conn_strings.append(f"PEPTIDE1,PEPTIDE1,{bond_map[bond_type]}")
            
            if conn_strings:
                return f"{sequence_part}${'|'.join(conn_strings)}$$$"
        
        return f"{sequence_part}$$$$"
    
    def generate_helm_sequences(
        self,
        num_samples: int,
        max_length: int,
        **kwargs
    ) -> List[str]:
        """Generate HELM strings directly."""
        results = self.sample(num_samples, max_length, **kwargs)
        return [
            self.decode_to_helm(r['tokens'], r.get('ring_connections', []))
            for r in results
        ]
