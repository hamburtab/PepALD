"""
DPO Trainer for Autoregressive Latent Diffusion model.

Training flow per step:
    1. Compute causal contexts for model and reference model.
    2. Add shared timestep/noise to winner and loser samples.
    3. Predict noise for model/ref on winner/loser samples.
    4. per-position MSE → scatter_mean → per-sample MSE [Bz]
    5. Backpropagate DPO loss plus optional winner protection.
"""

import math
import time
from copy import deepcopy
from pathlib import Path
from dataclasses import asdict
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from .loss import compute_diffusion_dpo_loss, scatter_mean


class DPOTrainer:
    """
    Diffusion-DPO trainer for AutoregressiveLatentDiffusion.

    Args:
        model:       pretrained ALD model (will be partially unfrozen for training)
        config:      ALDConfig
        train_loader: DataLoader of PreferencePairDataset
        beta_dpo:    DPO temperature (controls how aggressively model deviates from ref)
        lr:          learning rate
        freeze_mode: 'denoiser_only' or 'all' (which parts of model to train)
    """

    def __init__(
        self,
        model: nn.Module,
        config,
        train_loader: DataLoader,
        beta_dpo: float = 0.1,
        lr: float = 1e-5,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        freeze_mode: str = 'denoiser_only',
        dpop_winner_reg_alpha: float = 0.0,
        dpop_winner_reg_mode: str = 'external_reg',
        checkpoint_dir: str = './checkpoints/ald_dpo',
        device: str = 'cuda',
    ):
        self.config = config
        self.train_loader = train_loader
        self.beta_dpo = beta_dpo
        self.max_grad_norm = max_grad_norm
        self.dpop_winner_reg_alpha = float(dpop_winner_reg_alpha)
        self.dpop_winner_reg_mode = str(dpop_winner_reg_mode or 'external_reg')
        valid_dpop_modes = {'none', 'external_reg', 'margin_max'}
        if self.dpop_winner_reg_mode not in valid_dpop_modes:
            raise ValueError(
                f"Unknown dpop_winner_reg_mode: {self.dpop_winner_reg_mode}. "
                f"Expected one of {sorted(valid_dpop_modes)}"
            )
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Frozen reference model.
        self.ref_model = deepcopy(model)
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False
        self.ref_model = self.ref_model.to(self.device)

        # Trainable policy model.
        self.model = model.to(self.device)
        self._setup_freeze(freeze_mode)

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        n_trainable = sum(p.numel() for p in trainable_params)
        n_total = sum(p.numel() for p in self.model.parameters())
        print(f"[DPOTrainer] Trainable: {n_trainable/1e6:.2f}M / {n_total/1e6:.2f}M params")

        self.optimizer = optim.AdamW(
            trainable_params,
            lr=lr,
            weight_decay=weight_decay,
        )

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.global_step = 0
        self.epoch = 0

        self.T = self.model.diffusion_engine.num_steps
        self.dm = self.model.embedding_dim

    def _setup_freeze(self, freeze_mode: str):
        """Setup parameter freezing."""
        if freeze_mode == 'denoiser_only':
            # Train only the denoiser.
            for name, param in self.model.named_parameters():
                if 'diffusion_engine.denoiser' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            print("[DPOTrainer] Freeze mode: denoiser_only")
        elif freeze_mode == 'all':
            for param in self.model.parameters():
                param.requires_grad = True
            print("[DPOTrainer] Freeze mode: all trainable")
        else:
            raise ValueError(f"Unknown freeze_mode: {freeze_mode}")

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Single DPO training step.

        Args:
            batch: dict with token_ids_w, mask_w, token_ids_l, mask_l (all [Bz, L])

        Returns:
            dict with loss, margin, mse_w, mse_l
        """
        self.model.train()

        token_ids_w = batch['token_ids_w'].to(self.device)  # [Bz, L]
        mask_w = batch['mask_w'].to(self.device)             # [Bz, L]
        token_ids_l = batch['token_ids_l'].to(self.device)   # [Bz, L]
        mask_l = batch['mask_l'].to(self.device)             # [Bz, L]

        Bz, L = token_ids_w.shape

        # Step 1: context encoding with teacher forcing.
        # _prepare_contexts:
        #   token_ids → UniMolEmbedding → gt_embeddings [Bz, L, dm]
        #   token_ids → CausalTransformer → contexts [Bz, L, d_model]
        #   shifted so ctx[i] only sees tokens 0..i-1

        gt_w, ctx_w = self.model._prepare_contexts(token_ids_w, mask_w)
        gt_l, ctx_l = self.model._prepare_contexts(token_ids_l, mask_l)
        # gt_w/gt_l:  [Bz, L, dm]
        # ctx_w/ctx_l: [Bz, L, d_model]

        with torch.no_grad():
            _, ctx_w_ref = self.ref_model._prepare_contexts(token_ids_w, mask_w)
            _, ctx_l_ref = self.ref_model._prepare_contexts(token_ids_l, mask_l)
            # ctx_w_ref/ctx_l_ref: [Bz, L, d_model]

        # Step 2: shared timestep and noise for paired variance reduction.
        t = torch.randint(1, self.T + 1, (Bz,), device=self.device)

        noise = torch.randn(Bz, L, self.dm, device=self.device)

        # Step 3: forward diffusion.
        # x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        x_w_noisy, _ = self.model.diffusion_engine.add_noise(gt_w, t, noise)
        x_l_noisy, _ = self.model.diffusion_engine.add_noise(gt_l, t, noise)
        # x_w_noisy, x_l_noisy: [Bz, L, dm]

        # Step 4: flatten valid positions for the denoiser.
        # denoiser expects 2D input: [batch, dm], not 3D [Bz, L, dm]
        valid_w = mask_w.bool()   # [Bz, L]
        valid_l = mask_l.bool()   # [Bz, L]

        # Flatten winner
        target_w_flat = x_w_noisy[valid_w]              # [N_w, dm]
        noise_w_flat = noise[valid_w]                    # [N_w, dm]
        ctx_w_flat = ctx_w[valid_w].unsqueeze(1)         # [N_w, 1, d_model]
        ctx_w_ref_flat = ctx_w_ref[valid_w].unsqueeze(1) # [N_w, 1, d_model]

        # Flatten loser
        target_l_flat = x_l_noisy[valid_l]               # [N_l, dm]
        noise_l_flat = noise[valid_l]                     # [N_l, dm]
        ctx_l_flat = ctx_l[valid_l].unsqueeze(1)          # [N_l, 1, d_model]
        ctx_l_ref_flat = ctx_l_ref[valid_l].unsqueeze(1)  # [N_l, 1, d_model]

        # Expand t to each valid position
        t_expanded = t.unsqueeze(1).expand(-1, L)         # [Bz, L]
        t_w_flat = t_expanded[valid_w]                    # [N_w]
        t_l_flat = t_expanded[valid_l]                    # [N_l]

        # ── 4 noise predictions ──
        # model (with grad)
        pred_w_flat = self.model.diffusion_engine.predict_noise(
            target_w_flat, t_w_flat, ctx_w_flat
        )  # [N_w, dm]
        pred_l_flat = self.model.diffusion_engine.predict_noise(
            target_l_flat, t_l_flat, ctx_l_flat
        )  # [N_l, dm]

        # ref_model (no grad)
        with torch.no_grad():
            pred_w_ref_flat = self.ref_model.diffusion_engine.predict_noise(
                target_w_flat, t_w_flat, ctx_w_ref_flat
            )  # [N_w, dm]
            pred_l_ref_flat = self.ref_model.diffusion_engine.predict_noise(
                target_l_flat, t_l_flat, ctx_l_ref_flat
            )  # [N_l, dm]

        # ── Step 5: Per-sample MSE via scatter_mean ──
        mse_per_pos_theta_w = (pred_w_flat - noise_w_flat).pow(2).mean(dim=-1)  # [N_w]
        mse_per_pos_theta_l = (pred_l_flat - noise_l_flat).pow(2).mean(dim=-1)  # [N_l]
        mse_per_pos_ref_w = (pred_w_ref_flat - noise_w_flat).pow(2).mean(dim=-1)  # [N_w]
        mse_per_pos_ref_l = (pred_l_ref_flat - noise_l_flat).pow(2).mean(dim=-1)  # [N_l]

        # Sample index for scatter
        sample_idx_w = torch.arange(Bz, device=self.device).unsqueeze(1).expand(-1, L)[valid_w]  # [N_w]
        sample_idx_l = torch.arange(Bz, device=self.device).unsqueeze(1).expand(-1, L)[valid_l]  # [N_l]

        mse_theta_w = scatter_mean(mse_per_pos_theta_w, sample_idx_w, Bz, self.device)  # [Bz]
        mse_theta_l = scatter_mean(mse_per_pos_theta_l, sample_idx_l, Bz, self.device)  # [Bz]
        mse_ref_w = scatter_mean(mse_per_pos_ref_w, sample_idx_w, Bz, self.device)      # [Bz]
        mse_ref_l = scatter_mean(mse_per_pos_ref_l, sample_idx_l, Bz, self.device)      # [Bz]

        # ── Step 6: DPO / DPOP loss ──
        progress_w = mse_ref_w - mse_theta_w       # A ~= log pi_theta(w) / pi_ref(w)
        progress_l = mse_ref_l - mse_theta_l       # B ~= log pi_theta(l) / pi_ref(l)
        margin_per_sample = progress_w - progress_l
        dpo_loss, margin = compute_diffusion_dpo_loss(
            mse_theta_w, mse_ref_w,
            mse_theta_l, mse_ref_l,
            beta=self.beta_dpo,
        )

        # DPOP winner protection.
        # In diffusion-MSE form, max(0, -A) becomes max(0, mse_theta_w - mse_ref_w).
        winner_mse_delta = mse_theta_w - mse_ref_w
        dpop_winner_reg_per_sample = torch.relu(winner_mse_delta)
        dpop_winner_reg = dpop_winner_reg_per_sample.mean()

        if self.dpop_winner_reg_alpha <= 0.0 or self.dpop_winner_reg_mode == 'none':
            adjusted_margin = margin_per_sample
            loss = dpo_loss
        elif self.dpop_winner_reg_mode == 'external_reg':
            adjusted_margin = margin_per_sample
            loss = dpo_loss + self.dpop_winner_reg_alpha * dpop_winner_reg
        else:
            # Classical DPOP-style max protection inside the preference margin:
            # -log sigmoid(beta * (A - B - lambda * max(0, -A))).
            adjusted_margin = margin_per_sample - self.dpop_winner_reg_alpha * dpop_winner_reg_per_sample
            loss = -F.logsigmoid(self.beta_dpo * adjusted_margin).mean()

        # ── Backward ──
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
        self.optimizer.step()

        self.global_step += 1

        return {
            'loss': loss.item(),
            'dpo_loss': dpo_loss.item(),
            'margin': margin.item(),
            'mse_w': mse_theta_w.mean().item(),
            'mse_l': mse_theta_l.mean().item(),
            'mse_ref_w': mse_ref_w.mean().item(),
            'mse_ref_l': mse_ref_l.mean().item(),
            'winner_mse_delta': winner_mse_delta.mean().item(),
            'dpop_winner_reg': dpop_winner_reg.item(),
            'adjusted_margin': adjusted_margin.mean().item(),
        }

    def train_epoch(self, log_interval: int = 10) -> Dict[str, float]:
        """Run one epoch of DPO training."""
        self.epoch += 1
        epoch_metrics = {
            'loss': 0.0,
            'dpo_loss': 0.0,
            'margin': 0.0,
            'mse_w': 0.0,
            'mse_l': 0.0,
            'mse_ref_w': 0.0,
            'mse_ref_l': 0.0,
            'winner_mse_delta': 0.0,
            'dpop_winner_reg': 0.0,
            'adjusted_margin': 0.0,
        }
        n_steps = 0
        t_start = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            metrics = self.train_step(batch)
            n_steps += 1

            for k in epoch_metrics:
                epoch_metrics[k] += metrics[k]

            if log_interval > 0 and (batch_idx + 1) % log_interval == 0:
                avg_loss = epoch_metrics['loss'] / n_steps
                avg_margin = epoch_metrics['margin'] / n_steps
                elapsed = time.time() - t_start
                print(
                    f"  [Epoch {self.epoch}] Step {batch_idx+1}/{len(self.train_loader)} | "
                    f"loss={metrics['loss']:.4f} dpo={metrics['dpo_loss']:.4f} "
                    f"margin={metrics['margin']:.4f} adj_margin={metrics['adjusted_margin']:.4f} "
                    f"mse_w={metrics['mse_w']:.4f} mse_l={metrics['mse_l']:.4f} "
                    f"dpop={metrics['dpop_winner_reg']:.4f} "
                    f"({elapsed:.1f}s)"
                )

        # Average metrics
        for k in epoch_metrics:
            epoch_metrics[k] /= max(n_steps, 1)

        elapsed = time.time() - t_start
        print(
            f"[Epoch {self.epoch}] Done in {elapsed:.1f}s | "
            f"avg_loss={epoch_metrics['loss']:.4f} avg_dpo={epoch_metrics['dpo_loss']:.4f} "
            f"avg_margin={epoch_metrics['margin']:.4f} avg_adj_margin={epoch_metrics['adjusted_margin']:.4f} "
            f"avg_mse_w={epoch_metrics['mse_w']:.4f} avg_mse_l={epoch_metrics['mse_l']:.4f} "
            f"avg_dpop={epoch_metrics['dpop_winner_reg']:.4f}"
        )

        return epoch_metrics

    def save_checkpoint(self, filename: Optional[str] = None):
        """Save model checkpoint."""
        if filename is None:
            filename = f"dpo_epoch_{self.epoch}.pt"

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'ref_model_state_dict': self.ref_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'beta_dpo': self.beta_dpo,
            'dpop_winner_reg_alpha': self.dpop_winner_reg_alpha,
            'dpop_winner_reg_mode': self.dpop_winner_reg_mode,
            'vocab': self.model.vocab,
        }

        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")

        # Also save as latest
        torch.save(checkpoint, self.checkpoint_dir / 'dpo_latest.pt')

    def load_checkpoint(self, path: str):
        """Load DPO checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'ref_model_state_dict' in checkpoint:
            self.ref_model.load_state_dict(checkpoint['ref_model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception:
                print("Warning: Could not load optimizer state, starting fresh")
        self.global_step = checkpoint.get('global_step', 0)
        self.epoch = checkpoint.get('epoch', 0)
        print(f"Loaded DPO checkpoint from {path} (epoch={self.epoch}, step={self.global_step})")
