"""
Autoregressive Latent Diffusion (ALD) Model for HELM Peptide Generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple

from .context_encoder import CausalContextEncoder
from .token_mapper import TokenMapper
from .ring_predictor import RingBondPredictor, AutoregressiveRingPredictor, compute_ring_loss_ar
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
        # New autoregressive ring predictor with R-group awareness
        self.ar_ring_predictor = AutoregressiveRingPredictor(
            d_model=model_cfg.d_model,
            r_dim=self.embedding_dim,
            hidden_dim=model_cfg.d_model // 2,
            dropout=model_cfg.dropout
        )
        
        # 5. LM Head for Hybrid Modeling (Next Token Prediction)
        self.lm_head = nn.Linear(self.d_model, self.vocab_size)
        
        # 6. R-group lookup table for generation-time validation
        self.token_rgroups = self._build_rgroup_table(data_dir)
        
        if verbose:
            self._print_model_info(model_cfg)
    
    def _print_model_info(self, model_cfg):
        """Print model configuration summary."""
        print(f"[ALD] Model: d={model_cfg.d_model}, layers={model_cfg.context_layers}+{model_cfg.denoiser_layers}, T={model_cfg.num_diffusion_steps}")

    def _build_rgroup_table(self, data_dir) -> Dict[int, set]:
        """Build lookup: token_id -> set of available R-groups ('R1','R2','R3')."""
        table = {}
        try:
            import pandas as pd
            from pathlib import Path
            csv_path = Path(data_dir) / "monomer_library.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                for _, row in df.iterrows():
                    symbol = row['Symbol']
                    if symbol not in self.vocab:
                        continue
                    tid = self.vocab[symbol]
                    rgroups = set()
                    for rname in ('R1', 'R2', 'R3'):
                        val = str(row.get(rname, '-')).strip()
                        if val not in ('-', 'nan', ''):
                            rgroups.add(rname)
                    table[tid] = rgroups
                print(f"[ALD] R-group table: {len(table)} monomers, "
                      f"{sum(1 for v in table.values() if 'R3' in v)} with R3")
        except Exception as e:
            print(f"[ALD] Warning: Could not build R-group table: {e}")
        return table

    @property
    def embedding_loader(self):
        """Backward-compatible access to the Uni-Mol embedding loader."""
        return self.context_encoder.embedding
    
    def _empty_loss(self, device: torch.device) -> Dict[str, torch.Tensor]:
        """Return zero loss dictionary for edge cases."""
        return {
            'loss': torch.tensor(0.0, device=device, requires_grad=True),
            'diffusion_loss': torch.tensor(0.0, device=device),
            'ring_bond_loss': torch.tensor(0.0, device=device),
            'ring_position_loss': torch.tensor(0.0, device=device),
            'ring_type_loss': torch.tensor(0.0, device=device),
            'ce_loss': torch.tensor(0.0, device=device)
        }
    
    def _prepare_contexts(
        self,
        token_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_r_groups: bool = False
    ) -> tuple:
        """
        Prepare ground truth embeddings and shifted contexts for training.
        
        Args:
            token_ids: [batch_size, seq_len]
            mask: [batch_size, seq_len]
            return_r_groups: If True, also return R-group embeddings
        
        Returns:
            gt_embeddings: [batch_size, seq_len, embedding_dim]
            contexts_for_pred: [batch_size, seq_len, d_model] (shifted)
            r_emb (optional): [batch_size, seq_len, 3, embedding_dim]
        """
        # Get embeddings with optional R-groups
        if return_r_groups:
            gt_embeddings, r_emb = self.context_encoder.embedding(token_ids, return_r_groups=True)
        else:
            gt_embeddings = self.context_encoder.get_token_embedding(token_ids)
            r_emb = None
        
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
        
        if return_r_groups:
            return gt_embeddings, contexts_for_pred, r_emb
        return gt_embeddings, contexts_for_pred
        
    def forward(
        self,
        token_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        ring_bonds: Optional[List[List[Dict]]] = None,
        compute_ring_loss: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Training forward pass with teacher forcing (all positions).
        
        Args:
            token_ids: [batch_size, seq_len]
            mask: [batch_size, seq_len]
            ring_bonds: List of ring bond info per sample for ring loss
                Each element: [{'i': int, 'j': int, 'type': int}, ...]
            compute_ring_loss: Whether to compute ring bond loss
        """
        batch_size, seq_len = token_ids.shape
        device = token_ids.device
        
        if seq_len <= 1:
            return self._empty_loss(device)
        
        # Prepare contexts (with R-groups if computing ring loss)
        if compute_ring_loss and ring_bonds is not None:
            gt_embeddings, contexts_for_pred, r_emb = self._prepare_contexts(
                token_ids, mask, return_r_groups=True
            )
        else:
            gt_embeddings, contexts_for_pred = self._prepare_contexts(token_ids, mask)
            r_emb = None
        
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
        
        # Ring bond loss (autoregressive)
        ring_bond_loss = torch.tensor(0.0, device=device)
        ring_position_loss = torch.tensor(0.0, device=device)
        ring_type_loss = torch.tensor(0.0, device=device)
        
        if compute_ring_loss and ring_bonds is not None and r_emb is not None:
            ring_loss_result = self._compute_ring_loss_ar(
                contexts_for_pred, r_emb, ring_bonds, mask
            )
            ring_bond_loss = ring_loss_result['total_loss']
            ring_position_loss = ring_loss_result['position_loss']
            ring_type_loss = ring_loss_result['type_loss']
        
        return {
            'loss': diffusion_loss + 0.5 * ring_bond_loss + 0.5 * ce_loss,
            'diffusion_loss': diffusion_loss,
            'ring_bond_loss': ring_bond_loss,
            'ring_position_loss': ring_position_loss,
            'ring_type_loss': ring_type_loss,
            'ce_loss': ce_loss
        }
    
    def _compute_ring_loss_ar(
        self,
        context: torch.Tensor,
        r_emb: torch.Tensor,
        ring_bonds: List[List[Dict]],
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute ring bond loss in autoregressive manner.
        
        For each position t (from 1 to L-1), predict if it bonds with any position < t.
        
        Args:
            context: [B, L, d_model] - context vectors (shifted for prediction)
            r_emb: [B, L, 3, r_dim] - R-group embeddings
            ring_bonds: List of ring bond info per sample
            mask: [B, L] - valid position mask
        """
        device = context.device
        batch_size, seq_len, _ = context.shape
        
        total_position_loss = torch.tensor(0.0, device=device)
        total_type_loss = torch.tensor(0.0, device=device)
        pos_count = 0
        type_count = 0
        
        # Get sequence lengths
        if mask is not None:
            lengths = mask.sum(dim=1).long()
        else:
            lengths = torch.full((batch_size,), seq_len, dtype=torch.long, device=device)
        
        # Process each position t (from 1 to L-1)
        for t in range(1, seq_len):
            # Skip if this position is padding for all samples
            active_mask = t < lengths
            if not active_mask.any():
                continue
            
            # Current position context and R-groups
            current_ctx = context[:, t, :]  # [B, d_model]
            current_r = r_emb[:, t, :, :]   # [B, 3, r_dim]
            
            # History context and R-groups (positions 0 to t-1)
            history_ctx = context[:, :t, :]  # [B, t, d_model]
            history_r = r_emb[:, :t, :, :]   # [B, t, 3, r_dim]
            
            # Predict
            position_scores, type_logits = self.ar_ring_predictor(
                current_ctx, current_r, history_ctx, history_r
            )
            # position_scores: [B, t], type_logits: [B, t, 4]
            
            # Compute loss for each sample
            for b in range(batch_size):
                if not active_mask[b]:
                    continue
                
                # Find if position t has a bond with any history position
                bonds = ring_bonds[b]
                positive_hist_pos = -1
                bond_type = -1
                
                for bond in bonds:
                    i, j = bond['i'], bond['j']
                    # Check if j == t and i < t (bond from history to current)
                    if j == t and i < t:
                        positive_hist_pos = i
                        bond_type = bond['type']
                        break
                    # Also check if i == t and j < t
                    elif i == t and j < t:
                        positive_hist_pos = j
                        bond_type = bond['type']
                        break
                
                if positive_hist_pos >= 0:
                    # === Position Loss (Contrastive) ===
                    pos_score = position_scores[b, positive_hist_pos]
                    
                    # Create mask for negative samples
                    neg_mask = torch.ones(t, dtype=torch.bool, device=device)
                    neg_mask[positive_hist_pos] = False
                    neg_scores = position_scores[b][neg_mask]
                    
                    if neg_scores.numel() > 0:
                        margin = 1.0
                        loss_pos = F.relu(margin - pos_score + neg_scores.max())
                        total_position_loss = total_position_loss + loss_pos
                        pos_count += 1
                    
                    # === Type Loss (CrossEntropy) ===
                    logits = type_logits[b, positive_hist_pos]  # [4]
                    label = torch.tensor(bond_type, device=device)
                    loss_type = F.cross_entropy(logits.unsqueeze(0), label.unsqueeze(0))
                    total_type_loss = total_type_loss + loss_type
                    type_count += 1
                else:
                    # === Negative Sample Loss ===
                    # This position should NOT have any ring bond.
                    # Penalize the highest score to push all scores below -0.5
                    scores_t = position_scores[b, :t]
                    if scores_t.numel() > 0:
                        max_score = scores_t.max()
                        loss_neg = F.relu(max_score + 0.5)
                        total_position_loss = total_position_loss + loss_neg
                        pos_count += 1
        
        # Average losses
        if pos_count > 0:
            total_position_loss = total_position_loss / pos_count
        if type_count > 0:
            total_type_loss = total_type_loss / type_count
        
        total_loss = total_position_loss + total_type_loss
        
        return {
            'total_loss': total_loss,
            'position_loss': total_position_loss,
            'type_loss': total_type_loss
        }
    
    def forward_efficient(
        self,
        token_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        sample_positions: int = 5,
        ring_bonds: Optional[List[List[Dict]]] = None,
        compute_ring_loss: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Efficient training by sampling random positions per sequence."""
        batch_size, seq_len = token_ids.shape
        device = token_ids.device
        
        if seq_len <= 1:
            return self._empty_loss(device)
        
        # Prepare contexts (with R-groups if computing ring loss)
        if compute_ring_loss and ring_bonds is not None:
            gt_embeddings, contexts_for_pred, r_emb = self._prepare_contexts(
                token_ids, mask, return_r_groups=True
            )
        else:
            gt_embeddings, contexts_for_pred = self._prepare_contexts(token_ids, mask)
            r_emb = None
        
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
        
        # Ring bond loss (autoregressive) - compute on full sequence
        ring_bond_loss = torch.tensor(0.0, device=device)
        ring_position_loss = torch.tensor(0.0, device=device)
        ring_type_loss = torch.tensor(0.0, device=device)
        
        if compute_ring_loss and ring_bonds is not None and r_emb is not None:
            ring_loss_result = self._compute_ring_loss_ar(
                contexts_for_pred, r_emb, ring_bonds, mask
            )
            ring_bond_loss = ring_loss_result['total_loss']
            ring_position_loss = ring_loss_result['position_loss']
            ring_type_loss = ring_loss_result['type_loss']
        
        return {
            'loss': diffusion_loss + 0.5 * ring_bond_loss + 0.5 * ce_loss,
            'diffusion_loss': diffusion_loss,
            'ring_bond_loss': ring_bond_loss,
            'ring_position_loss': ring_position_loss,
            'ring_type_loss': ring_type_loss,
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
        lambda_gpt: float = 0.0,
        predict_ring_bonds: Optional[bool] = None,
        ring_threshold: float = 0.5,
        ring_top_k: int = 1,
        verbose: bool = False
    ) -> List[Dict]:
        """
        Batch-parallel generation with optional Hybrid Sampling (Diffusion + GPT).
        
        Args:
            predict_ring_bonds: Whether to predict ring bonds during generation
            ring_threshold: Score threshold for ring bond prediction
            ring_top_k: Number of top candidates to consider per position
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
        
        # R-group embeddings storage for ring prediction
        all_r_embeddings = None
        if predict_ring_bonds:
            r_dim = self.config.model.r_group_dim  # 512
            all_r_embeddings = torch.zeros(num_samples, max_seq_len, 3, r_dim, device=device)
        
        # Ring bond predictions: List[List[Dict]] for each sample
        predicted_ring_bonds = [[] for _ in range(num_samples)]
        # Track which samples already have a ring bond predicted
        has_ring_bond = [False for _ in range(num_samples)]
        
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
            
            # 4. Ring Bond Prediction (if enabled)
            if predict_ring_bonds and t > 0:
                # Always store R-group embeddings for later use
                _, r_emb_current = self.embedding_loader(token_ids, return_r_groups=True)
                # r_emb_current: [num_active, 3, r_dim]
                
                # Store R-group embeddings
                all_r_embeddings[active_idx, t, :, :] = r_emb_current
                
                # For first token (t==0), also need to store
                if t == 1:
                    first_tokens = all_tokens[active_idx, 0]
                    _, r_emb_first = self.embedding_loader(first_tokens, return_r_groups=True)
                    all_r_embeddings[active_idx, 0, :, :] = r_emb_first
            
            # Only start predicting ring bonds after enough tokens are generated
            min_ring_start = 3
            min_ring_distance = 3
            if predict_ring_bonds and t >= min_ring_start:
                # Predict ring bonds for each active sample
                for i in range(num_active):
                    sample_idx = active_idx[i].item()
                    
                    # Skip if this sample already has a ring bond (only one per cyclic peptide)
                    if has_ring_bond[sample_idx]:
                        continue
                    
                    # Get current context and R-groups
                    current_ctx = context[i:i+1]  # [1, d_model]
                    current_r = r_emb_current[i:i+1]  # [1, 3, r_dim]
                    
                    # Get history context and R-groups
                    history_ctx = self.context_encoder.get_context_for_next_token(
                        all_embeddings[sample_idx:sample_idx+1, :t, :]
                    )  # Need all positions 0 to t-1
                    # Actually, we need the context at each position
                    history_ctx_all = []
                    for j in range(t):
                        if j == 0:
                            h = torch.zeros(1, 0, self.embedding_dim, device=device)
                        else:
                            h = all_embeddings[sample_idx:sample_idx+1, :j, :]
                        ctx_j = self.context_encoder.get_context_for_next_token(h)
                        history_ctx_all.append(ctx_j)
                    history_ctx = torch.stack(history_ctx_all, dim=1)  # [1, t, d_model]
                    
                    history_r = all_r_embeddings[sample_idx:sample_idx+1, :t, :, :]  # [1, t, 3, r_dim]
                    
                    # Predict
                    position_scores, type_logits = self.ar_ring_predictor(
                        current_ctx, current_r,
                        history_ctx, history_r
                    )
                    # position_scores: [1, t], type_logits: [1, t, 4]
                    
                    # Get top-K candidates with minimum distance constraint
                    scores = position_scores[0]  # [t]
                    if scores.numel() > 0:
                        # Mask out candidates too close to current position
                        valid_scores = scores.clone()
                        valid_scores[max(0, t - min_ring_distance + 1):] = float('-inf')
                        
                        num_valid = (valid_scores > float('-inf')).sum().item()
                        if num_valid > 0:
                            top_k = min(ring_top_k, num_valid)
                            top_scores, top_positions = torch.topk(valid_scores, top_k)
                        else:
                            top_scores = torch.tensor([], device=device)
                            top_positions = torch.tensor([], dtype=torch.long, device=device)
                        
                        # Check if top score exceeds threshold
                        if top_scores.numel() > 0 and top_scores[0].item() > ring_threshold:
                            best_pos = top_positions[0].item()
                            
                            bond_type_map = {0: 'R3R3', 1: 'R1R2', 2: 'R1R3', 3: 'R3R2'}
                            # R-group requirements: (R-group for pos_i, R-group for pos_j)
                            bond_rgroup_req = {
                                0: ('R3', 'R3'), 1: ('R1', 'R2'),
                                2: ('R1', 'R3'), 3: ('R3', 'R2')
                            }
                            
                            # Get R-groups available on both monomers
                            tid_i = all_tokens[sample_idx, best_pos].item()
                            tid_j = all_tokens[sample_idx, t].item()
                            rg_i = self.token_rgroups.get(tid_i, set())
                            rg_j = self.token_rgroups.get(tid_j, set())
                            
                            # Try bond types in order of probability
                            type_probs = F.softmax(type_logits[0, best_pos], dim=-1)
                            sorted_types = torch.argsort(type_probs, descending=True)
                            
                            chosen = None
                            for bt in sorted_types.tolist():
                                req_i, req_j = bond_rgroup_req[bt]
                                if req_i not in rg_i or req_j not in rg_j:
                                    continue
                                # Position i: R1 is occupied by main chain when i > 0
                                if req_i == 'R1' and best_pos > 0:
                                    continue
                                chosen = bt
                                break
                            
                            if chosen is not None:
                                req_i, req_j = bond_rgroup_req[chosen]
                                predicted_ring_bonds[sample_idx].append({
                                    'i': best_pos,
                                    'j': t,
                                    'type': chosen,
                                    'type_name': bond_type_map[chosen],
                                    'score': top_scores[0].item()
                                })
                                has_ring_bond[sample_idx] = True
                                # Position j: R2 occupied by ring → main chain cannot continue
                                if req_j == 'R2':
                                    lengths[sample_idx] = t + 1
            
            if verbose and (t + 1) % 5 == 0:
                print(f"  Step {t+1}/{max_seq_len}, active samples: {num_active}")
        
        # Build results
        results = []
        for i in range(num_samples):
            seq_len = lengths[i].item()
            result = {
                'tokens': all_tokens[i, :seq_len],
                'embeddings': all_embeddings[i, :seq_len, :],
                'length': seq_len
            }
            
            # Add ring bond predictions
            if predict_ring_bonds:
                ring_bonds = predicted_ring_bonds[i]
                result['ring_bonds'] = ring_bonds
                
                # Convert to ring_connections format for decode_to_helm
                ring_connections = []
                for bond in ring_bonds:
                    ring_connections.append({
                        'res1': bond['i'] + 1,  # 1-indexed for HELM
                        'res2': bond['j'] + 1,
                        'bond_type': bond['type_name']
                    })
                result['ring_connections'] = ring_connections
            
            results.append(result)
        
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
