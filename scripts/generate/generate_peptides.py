"""
Generation script for Autoregressive Latent Diffusion (ALD) model.

Usage:
    # 生成线性肽（预训练模型）
    python scripts/generate/generate_peptides.py
    python scripts/generate/generate_peptides.py --mode linear
    
    # 生成环肽（微调模型）
    python scripts/generate/generate_peptides.py --mode cyclic

    # 使用 Vina 配置生成样本
    python scripts/generate/generate_peptides.py --mode vina

    # 使用 DPO 配置生成样本
    python scripts/generate/generate_peptides.py --mode dpo
    
    # 自定义配置
    python scripts/generate/generate_peptides.py --config configs/training/finetune_cyclic.json
"""

import os
import sys
import json
import time
import argparse
import re
from pathlib import Path

import torch
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from pepar_diff import AutoregressiveLatentDiffusion
from pepar_diff.config import ALDConfig
from pepar_diff.postprocess import is_head_tail_single_cycle

# ============================================================
# 默认配置路径
# ============================================================
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "training" / "pretrain.json"
CYCLIC_CONFIG = PROJECT_ROOT / "configs" / "training" / "finetune_cyclic.json"
CPP_CONFIG = PROJECT_ROOT / "configs" / "training" / "finetune_permeability_top1000.json"
VINA_CONFIG = PROJECT_ROOT / "configs" / "inference" / "vina.json"
DPO_CONFIG = PROJECT_ROOT / "configs" / "training" / "dpo.json"
# ============================================================


def parse_mode(value: str) -> str:
    """Normalize mode and validate supported values."""
    mode = value.lower()
    valid_modes = {"linear", "cyclic", "cpp", "vina", "dpo"}
    if mode not in valid_modes:
        raise argparse.ArgumentTypeError(
            f"invalid mode '{value}'. Choose from: linear, cyclic, cpp, vina, dpo"
        )
    return mode


def parse_args():
    parser = argparse.ArgumentParser(description="Generate peptides with ALD model")
    parser.add_argument(
        "--mode", type=parse_mode, choices=["linear", "cyclic", "cpp", "vina", "dpo"], default="linear",
        help="Generation mode: 'linear', 'cyclic', 'cpp', 'vina', or 'dpo'"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config file (overrides --mode)"
    )
    parser.add_argument(
        "--num_samples", type=int, default=None,
        help="Number of samples to generate"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output file path"
    )
    parser.add_argument(
        "--lambda_gpt", "--lambda", dest="lambda_gpt", type=float, default=None,
        help="Override generation.lambda_gpt for hybrid diffusion/GPT token selection"
    )
    return parser.parse_args()


def is_explicit_dpo_config(config_file: Path, config_arg: str | None) -> bool:
    """Special-case DPO generation requests selected by --mode or --config."""
    return config_file.stem == "dpo" or config_file.stem.startswith("dpo_")


def force_head_tail_cyclize_helm(helm_seq: str) -> str:
    """Rewrite a generated peptide as a head-tail single-cycle HELM."""
    match = re.search(r"PEPTIDE1\{(.+?)\}", helm_seq)
    if not match:
        return helm_seq

    tokens = [tok for tok in match.group(1).split(".") if tok]
    if len(tokens) < 2:
        return helm_seq

    sequence_part = f"PEPTIDE1{{{'.'.join(tokens)}}}"
    return f"{sequence_part}$PEPTIDE1,PEPTIDE1,1:R1-{len(tokens)}:R2$$$"


def load_model(config: ALDConfig, device: torch.device):
    """Load model from checkpoint."""
    
    checkpoint_path = config.generation.checkpoint_path
    print(f" Loading checkpoint from: {checkpoint_path}")
    
    if not Path(checkpoint_path).exists():
        print(f" Checkpoint not found: {checkpoint_path}")
        print("   Please update 'generation.checkpoint_path' in config file")
        sys.exit(1)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get vocab from checkpoint
    if 'vocab' in checkpoint:
        vocab = checkpoint['vocab']
        print(f"   Loaded vocab from checkpoint ({len(vocab)} tokens)")
    else:
        # Load from file
        vocab_path = config.training.vocab_file
        print(f"   Loading vocab from: {vocab_path}")
        with open(vocab_path, 'r') as f:
            vocab = json.load(f)
    
    # Create model
    model = AutoregressiveLatentDiffusion(vocab=vocab, config=config)
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f" Model loaded successfully ({total_params/1e6:.1f}M parameters)")
    
    return model, vocab


def decode_sequence(token_ids: torch.Tensor, vocab: dict, idx_to_token: dict = None) -> str:
    """Decode token IDs to HELM sequence string."""
    
    if idx_to_token is None:
        idx_to_token = {v: k for k, v in vocab.items()}
    
    tokens = []
    pad_id = vocab.get('<PAD>', 0)
    eos_id = vocab.get('<EOS>', 2)
    bos_id = vocab.get('<BOS>', 1)
    
    for idx in token_ids:
        idx = idx.item() if hasattr(idx, 'item') else idx
        if idx == pad_id:
            continue
        if idx == eos_id:
            break
        if idx == bos_id:
            continue
        
        token = idx_to_token.get(idx, f'<UNK:{idx}>')
        tokens.append(token)
    
    # Build HELM sequence
    if not tokens:
        return "PEPTIDE1{}"
    
    return "PEPTIDE1{" + ".".join(tokens) + "}$$$$"


def generate_samples(
    model,
    config: ALDConfig,
    vocab: dict,
    device: torch.device,
    force_head_tail_cycle: bool = False,
):
    """Generate samples using the model."""
    
    gen_cfg = config.generation
    idx_to_token = {v: k for k, v in vocab.items()}
    
    verbose = gen_cfg.verbose
    
    if verbose:
        print("\n" + "="*60)
        print("Generation Configuration")
        print("="*60)
        print(f"  num_samples:              {gen_cfg.num_samples}")
        print(f"  max_length:               {gen_cfg.max_length}")
        print(f"  min_length:               {gen_cfg.min_length}")
        print(f"  use_ddim:                 {gen_cfg.use_ddim}")
        if gen_cfg.use_ddim:
            print(f"  ddim_steps:               {gen_cfg.ddim_steps}")
        print(f"  lambda_gpt:               {gen_cfg.lambda_gpt}")
        print(f"  predict_ring_bonds:       {gen_cfg.predict_ring_bonds}")
        print(f"  cyclization_mode:         {getattr(gen_cfg, 'cyclization_mode', 'predict_ring')}")
        print(f"  use_embedding_norm:       {gen_cfg.use_embedding_norm}")
        print(f"  mapping_sample:           {gen_cfg.mapping_sample}")
        if gen_cfg.mapping_sample:
            print(f"  mapping_top_k:            {gen_cfg.mapping_top_k}")
            print(f"  mapping_top_p:            {gen_cfg.mapping_top_p}")
            print(f"  mapping_temperature:      {gen_cfg.mapping_temperature}")
            print(f"  mapping_frequency_penalty:{gen_cfg.mapping_frequency_penalty}")
        print("="*60)
    
    # Generate samples
    print(f"\n Generating {gen_cfg.num_samples} samples...")
    start_time = time.time()
    
    with torch.no_grad():
        samples = model.sample(
            num_samples=gen_cfg.num_samples,
            max_seq_len=gen_cfg.max_length,
            min_seq_len=gen_cfg.min_length,
            device=device,
            use_ddim=gen_cfg.use_ddim,
            ddim_steps=gen_cfg.ddim_steps if gen_cfg.use_ddim else None,
            lambda_gpt=gen_cfg.lambda_gpt,
            predict_ring_bonds=gen_cfg.predict_ring_bonds,
            ring_threshold=gen_cfg.ring_bond_threshold,
            ring_top_k=gen_cfg.ring_top_k,
            verbose=gen_cfg.verbose,
        )
    
    elapsed_time = time.time() - start_time
    
    print(f"   Generated {len(samples)} samples in {elapsed_time:.2f}s")
    print(f"   ({elapsed_time/len(samples)*1000:.1f}ms per sample)")
    
    # Decode samples
    helm_sequences = []
    ring_bond_counts = []
    sequence_lengths = []
    
    for sample in samples:
        if isinstance(sample, dict) and 'tokens' in sample:
            tokens = sample['tokens']
            ring_connections = sample.get('ring_connections', [])
        else:
            tokens = sample
            ring_connections = []

        # Decode to HELM. For DPO config we force head-tail single cyclization.
        if force_head_tail_cycle:
            linear_helm = model.decode_to_helm(tokens, [])
            helm_seq = force_head_tail_cyclize_helm(linear_helm)
            ring_count = 1 if "$PEPTIDE1,PEPTIDE1," in helm_seq else 0
        else:
            helm_seq = model.decode_to_helm(tokens, ring_connections)
            ring_count = len(ring_connections)

        helm_sequences.append(helm_seq)

        # Statistics
        pad_id = vocab.get('<PAD>', 0)
        if hasattr(tokens, 'cpu'):
            tokens_np = tokens.cpu().numpy()
        else:
            tokens_np = np.array(tokens)
        seq_len = np.sum(tokens_np != pad_id)
        sequence_lengths.append(seq_len)
        ring_bond_counts.append(ring_count)
    
    # Print samples if verbose
    if verbose:
        print("\n--- Generated HELM Sequences ---")
        for i, (helm_seq, ring_count, seq_len) in enumerate(zip(helm_sequences, ring_bond_counts, sequence_lengths)):
            ring_info = f"[环键数: {ring_count}]" if ring_count > 0 else "[线性]"
            print(f"样本 {i+1:3d} {ring_info} (长度: {seq_len:2d}): {helm_seq}")
        print("--- 生成完毕 ---\n")
    
    # Print statistics
    print("\n Generation Statistics:")
    print(f"   Total samples:     {len(helm_sequences)}")
    print(f"   Avg sequence len:  {np.mean(sequence_lengths):.1f}")
    print(f"   Min sequence len:  {np.min(sequence_lengths)}")
    print(f"   Max sequence len:  {np.max(sequence_lengths)}")
    print(f"   Cyclic sequences:  {sum(1 for c in ring_bond_counts if c > 0)}")
    print(f"   Linear sequences:  {sum(1 for c in ring_bond_counts if c == 0)}")
    
    return helm_sequences, ring_bond_counts, sequence_lengths


def save_samples(helm_sequences: list, output_file: str, append: bool = False):
    """Save generated samples to file."""
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    mode = 'a' if append else 'w'
    with open(output_path, mode) as f:
        for seq in helm_sequences:
            f.write(f"{seq}\n")
    
    print(f"\n Saved {len(helm_sequences)} sequences to {output_path}")


def filter_head_tail_single_cycle_sequences(helm_sequences: list[str]) -> list[str]:
    """Keep only head-tail single cyclic HELM sequences."""
    return [seq for seq in helm_sequences if is_head_tail_single_cycle(seq)]


def main():
    """Main function."""
    args = parse_args()
    
    # Determine config file
    if args.config:
        config_file = Path(args.config)
    elif args.mode == "cyclic":
        config_file = CYCLIC_CONFIG
    elif args.mode == "cpp":
        config_file = CPP_CONFIG
    elif args.mode == "vina":
        config_file = VINA_CONFIG
    elif args.mode == "dpo":
        config_file = DPO_CONFIG
    else:
        config_file = DEFAULT_CONFIG
    
    # Load config
    print(f"📄 Loading config from: {config_file}")
    print(f"🎯 Mode: {args.mode}")
    
    if not config_file.exists():
        print(f"❌ Config file not found: {config_file}")
        sys.exit(1)
    
    config = ALDConfig.load(str(config_file))
    gen_cfg = config.generation
    is_dpo_config = is_explicit_dpo_config(config_file, args.config)
    cyclization_mode = getattr(gen_cfg, "cyclization_mode", "predict_ring")
    force_head_tail_cycle = is_dpo_config and cyclization_mode == "force_head_tail"
    filter_to_head_tail_cycle = is_dpo_config and cyclization_mode == "predict_ring"
    
    # Override from command line args
    if args.num_samples:
        gen_cfg.num_samples = args.num_samples
    if args.output:
        gen_cfg.output_file = args.output
    if args.lambda_gpt is not None:
        gen_cfg.lambda_gpt = args.lambda_gpt
    
    # For explicit dpo.json generation, cyclization is controlled by generation.cyclization_mode.
    if is_dpo_config and cyclization_mode == "force_head_tail":
        gen_cfg.predict_ring_bonds = False
        print("🔒 dpo.json cyclization_mode=force_head_tail: forcing head-tail single-cycle HELM output")
    elif is_dpo_config and cyclization_mode == "predict_ring":
        gen_cfg.predict_ring_bonds = True
        print("🔁 dpo.json cyclization_mode=predict_ring: using automatic ring-bond prediction")
    elif args.mode == "linear":
        gen_cfg.predict_ring_bonds = False
    elif args.mode in {"cyclic", "vina"}:
        gen_cfg.predict_ring_bonds = True
    
    # Set seed
    if gen_cfg.seed is not None:
        torch.manual_seed(gen_cfg.seed)
        np.random.seed(gen_cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(gen_cfg.seed)
        print(f"🎲 Random seed set to {gen_cfg.seed}")
    
    # Device
    device = torch.device(config.training.device if torch.cuda.is_available() else 'cpu')
    print(f"💻 Using device: {device}")
    
    # Load model
    model, vocab = load_model(config, device)
    
    # Generate samples
    helm_sequences, ring_bond_counts, sequence_lengths = generate_samples(
        model, config, vocab, device,
        force_head_tail_cycle=force_head_tail_cycle,
    )

    if filter_to_head_tail_cycle:
        filtered_sequences = filter_head_tail_single_cycle_sequences(helm_sequences)
        removed = len(helm_sequences) - len(filtered_sequences)
        print("\n Head-tail single-cycle filtering (dpo.json + predict_ring):")
        print(f"   Kept head-tail single cycles: {len(filtered_sequences)}")
        print(f"   Removed linear / non-head-tail samples: {removed}")
        helm_sequences = filtered_sequences

    # Save if output file specified
    if gen_cfg.output_file:
        save_samples(helm_sequences, gen_cfg.output_file)
    else:
        print("\n💡 Tip: Use --output <file> to save sequences")

    print("\n✅ Generation complete!")


if __name__ == "__main__":
    main()
