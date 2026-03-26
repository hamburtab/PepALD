"""
DPO Training Script for Autoregressive Latent Diffusion.

Diffusion-DPO (Wallace et al. 2023) applied to cyclic peptide generation:
    1. Generate sequences with pretrained model
    2. Evaluate reward = w1 * Vina_docking_score + w2 * Permeability_score
    3. Build preference pairs (top-25% vs bottom-25%)
    4. Train with DPO loss: -log σ(β · [progress_w - progress_l])

Usage:
    python train_dpo.py
    python train_dpo.py --config configs/dpo.json
    python train_dpo.py --config configs/dpo.json --skip_generate --winner_file w.txt --loser_file l.txt
"""

import os
import sys
import json
import argparse
import time
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from ald import AutoregressiveLatentDiffusion
from ald.config import ALDConfig
from ald.dpo.dataset import PreferencePairDataset, PreferencePairCollator, build_preference_pairs
from ald.dpo.trainer import DPOTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="DPO training for ALD model")
    parser.add_argument(
        "--config", type=str, default="configs/dpo.json",
        help="Path to DPO config file"
    )
    parser.add_argument(
        "--skip_generate", action="store_true",
        help="Skip generation step, use pre-computed winner/loser files"
    )
    parser.add_argument(
        "--winner_file", type=str, default=None,
        help="Pre-computed winner HELM sequences (one per line)"
    )
    parser.add_argument(
        "--loser_file", type=str, default=None,
        help="Pre-computed loser HELM sequences (one per line)"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Resume DPO training from checkpoint"
    )
    return parser.parse_args()


def load_pretrained_model(config: ALDConfig, device: torch.device):
    """Load pretrained model from checkpoint."""
    checkpoint_path = config.training.pretrained_checkpoint
    print(f"Loading pretrained model from: {checkpoint_path}")

    if not Path(checkpoint_path).exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get vocab
    if 'vocab' in checkpoint:
        vocab = checkpoint['vocab']
    else:
        with open(config.training.vocab_file, 'r') as f:
            vocab = json.load(f)

    # Create and load model
    model = AutoregressiveLatentDiffusion(vocab=vocab, config=config)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded ({total_params/1e6:.1f}M parameters)")

    return model, vocab


def generate_and_evaluate(
    model: AutoregressiveLatentDiffusion,
    config: ALDConfig,
    dpo_cfg: dict,
    device: torch.device,
):
    """
    Step 1: Generate sequences with pretrained model.
    Step 2: Evaluate reward for each sequence.

    Returns:
        all_helms:   List[str]
        all_rewards: np.ndarray
    """
    gen_cfg = config.generation
    n_generate = dpo_cfg.get('num_generate', 2000)

    print(f"\n{'='*60}")
    print(f"Step 1: Generating {n_generate} sequences...")
    print(f"{'='*60}")

    model.eval()
    with torch.no_grad():
        samples = model.sample(
            num_samples=n_generate,
            max_seq_len=gen_cfg.max_length,
            min_seq_len=gen_cfg.min_length,
            device=device,
            use_ddim=gen_cfg.use_ddim,
            ddim_steps=gen_cfg.ddim_steps if gen_cfg.use_ddim else None,
            predict_ring_bonds=gen_cfg.predict_ring_bonds,
            ring_threshold=gen_cfg.ring_bond_threshold,
        )

    # Decode to HELM
    all_helms = []
    for sample in samples:
        if isinstance(sample, dict) and 'tokens' in sample:
            tokens = sample['tokens']
            ring_connections = sample.get('ring_connections', [])
        else:
            tokens = sample
            ring_connections = []
        helm_seq = model.decode_to_helm(tokens, ring_connections)
        all_helms.append(helm_seq)

    print(f"Generated {len(all_helms)} HELM sequences")

    # ── Evaluate rewards ──
    print(f"\n{'='*60}")
    print(f"Step 2: Evaluating rewards...")
    print(f"{'='*60}")

    w_vina = dpo_cfg.get('reward_w_vina', 1.0)
    w_perm = dpo_cfg.get('reward_w_perm', 0.5)

    # Permeability prediction
    perm_scores = np.zeros(len(all_helms))
    try:
        from eval.eval_permeability import Permeability
        perm_predictor = Permeability()
        perm_scores = perm_predictor(all_helms)
        valid_perm = perm_scores[perm_scores > -10]
        print(f"Permeability: {len(valid_perm)}/{len(all_helms)} valid, "
              f"mean={valid_perm.mean():.4f}" if len(valid_perm) > 0 else "No valid permeability scores")
    except Exception as e:
        print(f"Warning: Permeability evaluation failed: {e}")
        print("Using zero permeability scores")

    # Vina docking score
    # TODO: 接入实际的 Vina docking 代码
    # 目前使用占位符, 需要你提供 Vina docking 函数:
    #   vina_scores = dock(all_helms)  # 返回 np.ndarray, 越负越好(结合越强)
    vina_scores = np.zeros(len(all_helms))
    try:
        # 尝试导入 Vina docking 模块 (你需要实现这个)
        from Vina.dock import dock_helms
        vina_scores = dock_helms(all_helms)
        valid_vina = vina_scores[vina_scores != 0]
        print(f"Vina docking: {len(valid_vina)}/{len(all_helms)} valid, "
              f"mean={valid_vina.mean():.4f}" if len(valid_vina) > 0 else "No valid Vina scores")
    except ImportError:
        print("Warning: Vina docking module not found (Vina/dock.py)")
        print("Using zero Vina scores. Only permeability will drive DPO.")
    except Exception as e:
        print(f"Warning: Vina docking failed: {e}")
        print("Using zero Vina scores")

    # Combined reward
    # Vina score 越负越好, 取负号使得越大越好 (与 DPO 的 "higher is better" 一致)
    all_rewards = w_vina * (-vina_scores) + w_perm * perm_scores
    print(f"\nReward stats: mean={all_rewards.mean():.4f}, "
          f"std={all_rewards.std():.4f}, "
          f"min={all_rewards.min():.4f}, max={all_rewards.max():.4f}")

    return all_helms, all_rewards


def save_helm_list(helms: list, path: str):
    """Save HELM sequences to file, one per line."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        for h in helms:
            f.write(h + '\n')


def load_helm_list(path: str) -> list:
    """Load HELM sequences from file."""
    with open(path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def main():
    args = parse_args()

    # ── Load config ──
    print(f"Loading config from: {args.config}")
    config = ALDConfig.load(args.config)

    # Load DPO-specific config
    with open(args.config, 'r') as f:
        full_config = json.load(f)
    dpo_cfg = full_config.get('dpo', {})

    # ── Device ──
    device = torch.device(config.training.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ── Load pretrained model ──
    model, vocab = load_pretrained_model(config, device)

    # ── Build preference pairs ──
    if args.skip_generate and args.winner_file and args.loser_file:
        print(f"\nLoading pre-computed preference pairs...")
        winner_helms = load_helm_list(args.winner_file)
        loser_helms = load_helm_list(args.loser_file)
        print(f"Loaded {len(winner_helms)} winners, {len(loser_helms)} losers")
    else:
        # Generate and evaluate
        all_helms, all_rewards = generate_and_evaluate(model, config, dpo_cfg, device)

        # Build pairs
        top_ratio = dpo_cfg.get('top_ratio', 0.25)
        bottom_ratio = dpo_cfg.get('bottom_ratio', 0.25)
        winner_helms, loser_helms = build_preference_pairs(
            all_helms, all_rewards, top_ratio, bottom_ratio
        )

        # Save for reproducibility
        save_dir = Path(config.training.checkpoint_dir) / 'dpo_data'
        save_helm_list(winner_helms, str(save_dir / 'winners.txt'))
        save_helm_list(loser_helms, str(save_dir / 'losers.txt'))
        np.save(str(save_dir / 'rewards.npy'), all_rewards)
        print(f"Saved preference data to {save_dir}")

    # ── Create dataset & dataloader ──
    dataset = PreferencePairDataset(
        winner_helms=winner_helms,
        loser_helms=loser_helms,
        vocab_file=config.training.vocab_file,
        max_seq_len=config.model.max_seq_len,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        collate_fn=PreferencePairCollator(),
        drop_last=True,
    )

    # ── Create DPO trainer ──
    trainer = DPOTrainer(
        model=model,
        config=config,
        train_loader=dataloader,
        beta_dpo=dpo_cfg.get('beta_dpo', 0.1),
        lr=dpo_cfg.get('lr', 1e-5),
        weight_decay=config.training.weight_decay,
        max_grad_norm=config.training.max_grad_norm,
        freeze_mode=dpo_cfg.get('freeze_mode', 'denoiser_only'),
        checkpoint_dir=config.training.checkpoint_dir,
        device=str(device),
    )

    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # ── Train ──
    num_epochs = dpo_cfg.get('num_epochs', 10)
    log_interval = dpo_cfg.get('log_interval', 10)
    save_interval = dpo_cfg.get('save_interval_epochs', 1)

    print(f"\n{'='*60}")
    print(f"DPO Training")
    print(f"{'='*60}")
    print(f"  beta_dpo:       {trainer.beta_dpo}")
    print(f"  num_epochs:     {num_epochs}")
    print(f"  batch_size:     {config.training.batch_size}")
    print(f"  lr:             {dpo_cfg.get('lr', 1e-5)}")
    print(f"  freeze_mode:    {dpo_cfg.get('freeze_mode', 'denoiser_only')}")
    print(f"  dataset_size:   {len(dataset)}")
    print(f"  steps/epoch:    {len(dataloader)}")
    print(f"{'='*60}\n")

    t_start = time.time()

    for epoch in range(1, num_epochs + 1):
        metrics = trainer.train_epoch(log_interval=log_interval)

        # Save checkpoint
        if epoch % save_interval == 0:
            trainer.save_checkpoint()

        # Health check
        if metrics['margin'] < -0.5:
            print(f"WARNING: margin is negative ({metrics['margin']:.4f}), model may be learning backwards!")
        if metrics['loss'] > 5.0:
            print(f"WARNING: loss is very high ({metrics['loss']:.4f}), consider reducing lr or increasing beta")

    total_time = time.time() - t_start
    print(f"\nDPO training complete in {total_time/60:.1f} minutes")

    # Final save
    trainer.save_checkpoint(f"dpo_final_epoch{num_epochs}.pt")
    print("Done!")


if __name__ == "__main__":
    main()
