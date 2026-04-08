"""
DPO Training Script for Autoregressive Latent Diffusion.

Diffusion-DPO (Wallace et al. 2023) applied to cyclic peptide generation:
    1. Load candidate sequences from file (or generate with pretrained model)
    2. Evaluate reward = w1 * Vina_docking_score + w2 * Permeability_score
    3. Build preference pairs (e.g., top-20% vs bottom-20%)
    4. Train with DPO loss: -log σ(β · [progress_w - progress_l])

Usage:
    python train_dpo.py
    python train_dpo.py --config configs/dpo.json
    python train_dpo.py --sample_file chembl32_samples/head_tail_single_cycle_samples.txt
    python train_dpo.py --config configs/dpo.json --skip_generate --winner_file w.txt --loser_file l.txt
"""

import sys
import json
import argparse
import time
import os
import csv
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from ald import AutoregressiveLatentDiffusion
from ald.config import ALDConfig
from ald.dpo.candidate_utils import compute_chemistry_scores, robust_normalize
from ald.dpo.dataset import PreferencePairDataset, PreferencePairCollator, build_preference_pairs
from ald.dpo.trainer import DPOTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="DPO training for ALD model")
    parser.add_argument(
        "--config", type=str, default="configs/dpo.json",
        help="Path to DPO config file"
    )
    parser.add_argument(
        "--sample_file", type=str, default=None,
        help="Candidate HELM sequences file for ranking (one per line). "
             "Overrides dpo.sample_file in config."
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
    parser.add_argument(
        "--perm_score_file", type=str, default=None,
        help="Optional CSV/TSV file with precomputed permeability scores. "
             "Overrides dpo.perm_score_file in config."
    )
    return parser.parse_args()


def resolve_path(path_str: str) -> Path:
    """Resolve path; relative paths are interpreted from project root."""
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


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


def _print_component_stats(name: str, raw_values: np.ndarray, normalized_values: np.ndarray):
    print(
        f"{name:>12s}: raw_mean={raw_values.mean():.4f}, raw_std={raw_values.std():.4f}, "
        f"norm_mean={normalized_values.mean():.4f}, norm_std={normalized_values.std():.4f}"
    )


def evaluate_rewards(all_helms: list, dpo_cfg: dict):
    """Step 2: Evaluate reward for a list of HELM sequences."""
    if len(all_helms) == 0:
        raise ValueError("No HELM sequences provided for reward evaluation.")
    print(f"\n{'='*60}")
    print(f"Step 2: Evaluating rewards...")
    print(f"{'='*60}")

    w_vina = dpo_cfg.get('reward_w_vina', 1.0)
    w_perm = dpo_cfg.get('reward_w_perm', 0.5)
    vina_device = str(dpo_cfg.get('vina_device', 'cpu')).lower()
    vina_cpu = resolve_vina_cpu(dpo_cfg.get('vina_cpu', None))
    vina_cpu_per_worker = int(dpo_cfg.get('vina_cpu_per_worker', 2))
    vina_num_workers = int(dpo_cfg.get('vina_num_workers', 0))
    vina_exhaustiveness = int(dpo_cfg.get('vina_exhaustiveness', 8))
    vina_n_poses = int(dpo_cfg.get('vina_n_poses', 2))
    vina_show_progress = bool(dpo_cfg.get('vina_show_progress', True))
    docking_backend = str(dpo_cfg.get('docking_backend', 'vina')).lower()
    dock_box_size = dpo_cfg.get('dock_box_size', 30.0)
    dock_seed = int(dpo_cfg.get('dock_seed', 42))
    unidock_binary = str(dpo_cfg.get('unidock_binary', 'unidock'))
    unidock_batch_size = int(dpo_cfg.get('unidock_batch_size', 64))
    unidock_search_mode = str(dpo_cfg.get('unidock_search_mode', 'fast'))
    unidock_scoring = str(dpo_cfg.get('unidock_scoring', 'vina'))
    unidock_refine_step = int(dpo_cfg.get('unidock_refine_step', 3))
    unidock_max_step = int(dpo_cfg.get('unidock_max_step', 20))
    unidock_max_gpu_memory = int(dpo_cfg.get('unidock_max_gpu_memory', 0))
    unidock_keep_workdir = bool(dpo_cfg.get('unidock_keep_workdir', False))
    unidock_verbosity = int(dpo_cfg.get('unidock_verbosity', 0))
    perm_score_file = dpo_cfg.get('perm_score_file')

    # Permeability prediction
    perm_scores = np.zeros(len(all_helms))
    if w_perm == 0:
        print("Permeability reward disabled (reward_w_perm=0); skipping permeability scoring")
    elif perm_score_file:
        perm_scores = load_precomputed_perm_scores(all_helms, perm_score_file)
        valid_perm = perm_scores[perm_scores > -10]
        print(
            f"Permeability (precomputed): {len(valid_perm)}/{len(all_helms)} valid, "
            f"mean={valid_perm.mean():.4f}"
            if len(valid_perm) > 0 else
            "No valid precomputed permeability scores"
        )
    else:
        try:
            from eval.eval_permeability import Permeability
            perm_predictor = Permeability()
            perm_scores = perm_predictor(all_helms)
            valid_perm = perm_scores[perm_scores > -10]
            print(f"Permeability: {len(valid_perm)}/{len(all_helms)} valid, "
                  f"mean={valid_perm.mean():.4f}" if len(valid_perm) > 0 else "No valid permeability scores")
        except Exception as e:
            raise RuntimeError(
                "Permeability evaluation failed while reward_w_perm > 0. "
                "Use the isolated perm_env to export a score file and pass "
                "--perm_score_file (or set dpo.perm_score_file), or set reward_w_perm=0 "
                f"if you intentionally want Vina-only DPO. Original error: {e}"
            ) from e

    # Vina docking score (required).
    try:
        from Vina.dock import dock_helms
    except Exception as e:
        raise RuntimeError(
            f"Vina docking import failed, cannot continue DPO training: {e}"
        ) from e

    try:
        print(
            f"Docking runtime: backend={docking_backend}, device={vina_device}, cpu={vina_cpu}, "
            f"workers={vina_num_workers}, cpu_per_worker={vina_cpu_per_worker}, "
            f"exhaustiveness={vina_exhaustiveness}, n_poses={vina_n_poses}"
        )
        vina_scores = np.asarray(
            dock_helms(
                all_helms,
                device=vina_device,
                cpu=vina_cpu,
                cpu_per_worker=vina_cpu_per_worker,
                num_workers=vina_num_workers,
                exhaustiveness=vina_exhaustiveness,
                n_poses=vina_n_poses,
                show_progress=vina_show_progress,
                backend=docking_backend,
                box_size=dock_box_size,
                seed=dock_seed,
                unidock_binary=unidock_binary,
                unidock_batch_size=unidock_batch_size,
                unidock_search_mode=unidock_search_mode,
                unidock_scoring=unidock_scoring,
                unidock_refine_step=unidock_refine_step,
                unidock_max_step=unidock_max_step,
                unidock_max_gpu_memory=unidock_max_gpu_memory,
                unidock_keep_workdir=unidock_keep_workdir,
                unidock_verbosity=unidock_verbosity,
            ),
            dtype=np.float64,
        )
    except Exception as e:
        raise RuntimeError(
            f"Vina docking execution failed, cannot continue DPO training: {e}"
        ) from e

    if vina_scores.shape[0] != len(all_helms):
        raise RuntimeError(
            f"Vina returned {vina_scores.shape[0]} scores for {len(all_helms)} sequences."
        )

    valid_vina = vina_scores[vina_scores != 0]
    if len(valid_vina) == 0:
        raise RuntimeError(
            "Vina produced zero valid scores (all scores are INVALID_SCORE=0.0); aborting DPO."
        )

    print(f"Vina docking: {len(valid_vina)}/{len(all_helms)} valid, "
          f"mean={valid_vina.mean():.4f}")

    # Combined reward
    # Vina score 越负越好, 取负号使得越大越好 (与 DPO 的 "higher is better" 一致)
    reward_w_chem = float(dpo_cfg.get('reward_w_chemistry', 0.0))
    normalize_rewards = bool(dpo_cfg.get('reward_normalize', True))
    chemistry_target_len = dpo_cfg.get('chemistry_target_length', None)

    reward_vina = -vina_scores
    reward_perm = perm_scores
    chemistry_scores = compute_chemistry_scores(all_helms, target_len=chemistry_target_len)

    if normalize_rewards:
        norm_vina = robust_normalize(reward_vina)
        norm_perm = robust_normalize(reward_perm)
        norm_chem = robust_normalize(chemistry_scores)
    else:
        norm_vina = reward_vina
        norm_perm = reward_perm
        norm_chem = chemistry_scores

    _print_component_stats("Vina", reward_vina, norm_vina)
    _print_component_stats("Perm", reward_perm, norm_perm)
    if reward_w_chem > 0:
        _print_component_stats("Chemistry", chemistry_scores, norm_chem)

    all_rewards = w_vina * norm_vina + w_perm * norm_perm + reward_w_chem * norm_chem
    print(f"\nReward stats: mean={all_rewards.mean():.4f}, "
          f"std={all_rewards.std():.4f}, "
          f"min={all_rewards.min():.4f}, max={all_rewards.max():.4f}")

    reward_info = {
        'reward_vina': reward_vina,
        'reward_perm': reward_perm,
        'chemistry_scores': chemistry_scores,
        'norm_vina': norm_vina,
        'norm_perm': norm_perm,
        'norm_chemistry': norm_chem,
    }
    return all_rewards, vina_scores, perm_scores, reward_info


def _detect_delimiter(header_line: str, path: Path) -> str:
    if path.suffix.lower() == '.tsv':
        return '\t'
    if '\t' in header_line and ',' not in header_line:
        return '\t'
    return ','


def load_precomputed_perm_scores(all_helms: list, score_file: str) -> np.ndarray:
    """Load precomputed permeability scores aligned by HELM sequence."""
    score_path = resolve_path(score_file)
    if not score_path.exists():
        raise FileNotFoundError(f"Permeability score file not found: {score_path}")

    with open(score_path, 'r', newline='') as f:
        header_line = f.readline()
        if not header_line:
            raise ValueError(f"Permeability score file is empty: {score_path}")
        delimiter = _detect_delimiter(header_line, score_path)
        f.seek(0)
        reader = csv.DictReader(f, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(
                "Permeability score file must contain a header row with at least "
                "'helm' and 'permeability' columns."
            )

        field_map = {name.strip().lower(): name for name in reader.fieldnames if name}
        helm_key = field_map.get('helm')
        score_key = (
            field_map.get('permeability')
            or field_map.get('perm_score')
            or field_map.get('score')
        )
        if helm_key is None or score_key is None:
            raise ValueError(
                "Permeability score file must contain 'helm' and 'permeability' "
                "(or 'perm_score' / 'score') columns."
            )

        score_by_helm = {}
        duplicate_rows = 0
        for row in reader:
            helm = (row.get(helm_key) or '').strip()
            if not helm:
                continue
            raw_score = (row.get(score_key) or '').strip()
            try:
                score = float(raw_score)
            except ValueError as e:
                raise ValueError(
                    f"Invalid permeability score '{raw_score}' for HELM: {helm}"
                ) from e

            if helm in score_by_helm:
                duplicate_rows += 1
            score_by_helm[helm] = score

    scores = np.zeros(len(all_helms), dtype=np.float64)
    missing = []
    invalid = []
    for idx, helm in enumerate(all_helms):
        if helm not in score_by_helm:
            missing.append(helm)
            continue
        score = float(score_by_helm[helm])
        if not np.isfinite(score):
            invalid.append((helm, score))
            continue
        scores[idx] = score

    if missing or invalid:
        examples = []
        if missing:
            examples.extend(missing[:3])
        if invalid:
            examples.extend([f"{helm} -> {score}" for helm, score in invalid[:3]])
        details = "; ".join(examples)
        raise ValueError(
            f"Permeability score file {score_path} does not cover the candidate set "
            f"(missing={len(missing)}, invalid={len(invalid)}). Examples: {details}"
        )

    print(
        f"Loaded precomputed permeability scores for {len(scores)} HELM sequences "
        f"from {score_path}"
        + (f" ({duplicate_rows} duplicate rows overwritten by last occurrence)" if duplicate_rows else "")
    )
    return scores


def resolve_vina_cpu(vina_cpu_value) -> int:
    """Resolve Vina CPU threads from config; default is all available CPU cores."""
    available = os.cpu_count() or 1

    if vina_cpu_value is None:
        return available

    if isinstance(vina_cpu_value, str):
        value = vina_cpu_value.strip().lower()
        if value in {"auto", "max", "all"}:
            return available
        try:
            vina_cpu_value = int(value)
        except ValueError as e:
            raise ValueError(
                "Invalid dpo.vina_cpu value. Use integer, 0/-1, or one of: auto/max/all."
            ) from e

    try:
        cpu = int(vina_cpu_value)
    except (TypeError, ValueError) as e:
        raise ValueError(
            "Invalid dpo.vina_cpu value. Use integer, 0/-1, or one of: auto/max/all."
        ) from e

    if cpu <= 0:
        return available

    if cpu > available:
        print(f"Warning: vina_cpu={cpu} exceeds available CPU cores ({available}); capping to {available}")
        return available

    return cpu


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
    all_rewards, vina_scores, perm_scores, reward_info = evaluate_rewards(all_helms, dpo_cfg)
    return all_helms, all_rewards, vina_scores, perm_scores, reward_info


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


def deduplicate_helms(helms: list, labels: list = None, stage: str = "candidates"):
    """Remove duplicate HELM sequences while preserving order."""
    n = len(helms)
    seen = set()
    unique_helms = []
    unique_labels = []
    duplicate_count = 0

    for idx, helm in enumerate(helms):
        if helm in seen:
            duplicate_count += 1
            continue
        seen.add(helm)
        unique_helms.append(helm)
        if labels is not None:
            unique_labels.append(labels[idx])

    if duplicate_count == 0:
        print(f"{stage} dedup: no duplicates found")
        return (helms, labels) if labels is not None else helms

    print(
        f"{stage} dedup: {n} -> {len(unique_helms)} unique "
        f"({duplicate_count} duplicates removed)"
    )
    return (unique_helms, unique_labels) if labels is not None else unique_helms


def load_candidate_helms(sample_files: list):
    """Load candidate HELM sequences from one or more files."""
    all_helms = []
    source_labels = []
    resolved_paths = []

    for sample_file in sample_files:
        sample_path = resolve_path(sample_file)
        if not sample_path.exists():
            print(f"Sample file not found: {sample_path}")
            sys.exit(1)

        helms = load_helm_list(str(sample_path))
        if len(helms) == 0:
            print(f"No valid HELM sequences found in sample file: {sample_path}")
            sys.exit(1)

        source_name = sample_path.stem
        all_helms.extend(helms)
        source_labels.extend([source_name] * len(helms))
        resolved_paths.append(sample_path)
        print(f"Loaded {len(helms)} HELM sequences from {sample_path}")

    all_helms, source_labels = deduplicate_helms(
        all_helms,
        labels=source_labels,
        stage="candidate files",
    )
    return all_helms, source_labels, resolved_paths


def load_and_evaluate_from_file(sample_files: list, dpo_cfg: dict):
    """
    Step 1: Load candidate sequences from file(s).
    Step 2: Evaluate reward for each sequence.
    """
    print(f"\n{'='*60}")
    print(f"Step 1: Loading candidate sequences from file(s)...")
    print(f"{'='*60}")
    all_helms, source_labels, sample_paths = load_candidate_helms(sample_files)
    print(f"Loaded {len(all_helms)} unique HELM sequences from {len(sample_paths)} file(s)")
    all_rewards, vina_scores, perm_scores, reward_info = evaluate_rewards(all_helms, dpo_cfg)
    return all_helms, all_rewards, vina_scores, perm_scores, reward_info, source_labels, sample_paths


def main():
    args = parse_args()

    # ── Load config ──
    print(f"Loading config from: {args.config}")
    config = ALDConfig.load(args.config)

    # Load DPO-specific config
    with open(args.config, 'r') as f:
        full_config = json.load(f)
    dpo_cfg = full_config.get('dpo', {})
    if args.perm_score_file:
        dpo_cfg['perm_score_file'] = args.perm_score_file

    # ── Device ──
    device = torch.device(config.training.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ── Load pretrained model ──
    model, vocab = load_pretrained_model(config, device)

    # ── Build preference pairs ──
    if args.skip_generate:
        if not (args.winner_file and args.loser_file):
            print("When --skip_generate is set, both --winner_file and --loser_file must be provided.")
            sys.exit(1)
        print(f"\nLoading pre-computed preference pairs...")
        winner_helms = load_helm_list(args.winner_file)
        loser_helms = load_helm_list(args.loser_file)
        print(f"Loaded {len(winner_helms)} winners, {len(loser_helms)} losers")
    else:
        sample_file = args.sample_file
        sample_files = []
        if sample_file:
            sample_files = [sample_file]
        elif dpo_cfg.get('sample_files'):
            sample_files = list(dpo_cfg.get('sample_files', []))
        elif dpo_cfg.get('sample_file'):
            sample_files = [dpo_cfg.get('sample_file')]

        if dpo_cfg.get('perm_score_file') and not sample_files:
            print(
                "Precomputed permeability scores require a fixed candidate file. "
                "Set --sample_file (or dpo.sample_file / dpo.sample_files) when using --perm_score_file."
            )
            sys.exit(1)

        if sample_files:
            (
                all_helms,
                all_rewards,
                vina_scores,
                perm_scores,
                reward_info,
                source_labels,
                sample_paths,
            ) = load_and_evaluate_from_file(sample_files, dpo_cfg)
            print(f"Using candidate set from: {', '.join(str(p) for p in sample_paths)}")
        else:
            # Fallback: generate and evaluate with pretrained model
            all_helms, all_rewards, vina_scores, perm_scores, reward_info = generate_and_evaluate(model, config, dpo_cfg, device)
            source_labels = ["generated"] * len(all_helms)

        # Build pairs
        top_ratio = dpo_cfg.get('top_ratio', 0.2)
        bottom_ratio = dpo_cfg.get('bottom_ratio', 0.2)
        w_vina = dpo_cfg.get('reward_w_vina', 1.0)
        w_perm = dpo_cfg.get('reward_w_perm', 0.5)
        winner_helms, loser_helms = build_preference_pairs(
            all_helms, all_rewards, top_ratio, bottom_ratio,
            vina_scores=vina_scores, perm_scores=perm_scores,
            chemistry_scores=reward_info.get('chemistry_scores'),
            reward_w_vina=w_vina, reward_w_perm=w_perm,
            source_labels=source_labels,
            winner_pool_ratio=dpo_cfg.get('winner_pool_ratio'),
            loser_pool_ratio=dpo_cfg.get('loser_pool_ratio'),
            winner_diversity_lambda=dpo_cfg.get('winner_diversity_lambda', 0.35),
            loser_diversity_lambda=dpo_cfg.get('loser_diversity_lambda', 0.20),
            pair_strategy=dpo_cfg.get('pair_strategy', 'nearest_hard_negative'),
            min_reward_gap=float(dpo_cfg.get('min_reward_gap', 0.0)),
        )

        # Save for reproducibility
        save_dir = Path(config.training.checkpoint_dir) / 'dpo_data'
        save_helm_list(all_helms, str(save_dir / 'candidates.txt'))
        save_helm_list(winner_helms, str(save_dir / 'winners.txt'))
        save_helm_list(loser_helms, str(save_dir / 'losers.txt'))
        np.save(str(save_dir / 'rewards.npy'), all_rewards)
        if reward_info:
            for key, value in reward_info.items():
                np.save(str(save_dir / f"{key}.npy"), np.asarray(value))
        with open(save_dir / 'sources.json', 'w') as f:
            json.dump(source_labels, f, ensure_ascii=False, indent=2)
        print(f"Saved preference data to {save_dir}")

    # ── Create dataset & dataloader ──
    dataset = PreferencePairDataset(
        winner_helms=winner_helms,
        loser_helms=loser_helms,
        vocab_file=config.training.vocab_file,
        max_seq_len=config.model.max_seq_len,
        preserve_pairing=bool(dpo_cfg.get('preserve_pairing', True)),
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
