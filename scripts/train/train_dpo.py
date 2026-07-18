"""
DPO Training Script for Autoregressive Latent Diffusion.

Diffusion-DPO (Wallace et al. 2023) applied to cyclic peptide generation:
    1. Load candidate sequences from file (or generate with pretrained model)
    2. Evaluate reward = w1 * Vina_docking_score + w2 * Permeability_score
    3. Build preference pairs (e.g., top-20% vs bottom-20%)
    4. Train with DPO loss: -log σ(β · [progress_w - progress_l])

Usage:
    python scripts/train/train_dpo.py
    python scripts/train/train_dpo.py --config configs/training/dpo.json
    python scripts/train/train_dpo.py --sample_file outputs/samples/case1/train_candidates/candidates.txt
    python scripts/train/train_dpo.py --config configs/training/dpo.json --skip_generate --winner_file w.txt --loser_file l.txt
"""

import sys
import json
import argparse
import os
import random
import time
import csv
from datetime import datetime
from collections import Counter
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from pepar_diff import AutoregressiveLatentDiffusion
from pepar_diff.config import ALDConfig
from pepar_diff.dpo.candidate_utils import compute_chemistry_scores, robust_normalize
from pepar_diff.dpo.dataset import PreferencePairDataset, PreferencePairCollator, build_preference_pairs
from pepar_diff.dpo.epoch_sampling import generate_epoch_samples
from pepar_diff.dpo.pair_io import save_preference_pair_snapshot
from pepar_diff.dpo.trainer import DPOTrainer
from pepar_diff.vina.constants import INVALID_SCORE


class TeeStream:
    """Mirror writes to both the original stream and a log file."""

    def __init__(self, primary, secondary):
        self.primary = primary
        self.secondary = secondary

    def write(self, data):
        self.primary.write(data)
        self.secondary.write(data)
        return len(data)

    def flush(self):
        self.primary.flush()
        self.secondary.flush()

    def isatty(self):
        return bool(getattr(self.primary, "isatty", lambda: False)())

    def __getattr__(self, name):
        return getattr(self.primary, name)


def parse_args():
    parser = argparse.ArgumentParser(description="DPO training for ALD model")
    parser.add_argument(
        "--config", type=str, default="configs/training/dpo.json",
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
        "--prepare_pairs_only", action="store_true",
        help="Build, validate, and save preference pairs, then exit before DPO training"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Override dpo.seed for reproducible data order, timestep, and noise sampling"
    )
    parser.add_argument(
        "--perm_score_file", type=str, default=None,
        help="Optional CSV/TSV file with precomputed permeability scores. "
             "Overrides dpo.perm_score_file in config."
    )
    parser.add_argument(
        "--vina_score_file", type=str, default=None,
        help="Optional CSV/TSV file used as an incremental cache for docking scores. "
             "Existing rows are reused; missing HELMs are docked and appended."
    )
    return parser.parse_args()


def resolve_path(path_str: str) -> Path:
    """Resolve path; relative paths are interpreted from project root."""
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def setup_output_logging(checkpoint_dir: str) -> Path:
    """Mirror stdout/stderr to a timestamped log file under checkpoint_dir/logs."""
    log_dir = resolve_path(str(Path(checkpoint_dir) / "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"dpo_train_{timestamp}.log"
    log_file = open(log_path, "a", encoding="utf-8", buffering=1)

    sys.stdout = TeeStream(sys.stdout, log_file)
    sys.stderr = TeeStream(sys.stderr, log_file)
    return log_path


def configure_reproducibility(seed: int | None, deterministic: bool = False) -> None:
    """Configure the RNGs and CUDA behavior used by DPO training."""
    if seed is None:
        print("DPO seed: unset (run is not guaranteed to be reproducible)")
        return

    seed = int(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    if deterministic:
        # This must be present before the first CUDA BLAS operation.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)

    print(f"DPO seed: {seed}")
    print(f"Deterministic algorithms: {'enabled' if deterministic else 'disabled'}")


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
    vina_exhaustiveness = int(dpo_cfg.get('vina_exhaustiveness', 8))
    vina_n_poses = int(dpo_cfg.get('vina_n_poses', 2))
    vina_show_progress = bool(dpo_cfg.get('vina_show_progress', True))
    dock_box_size = dpo_cfg.get('dock_box_size', 30.0)
    dock_center = dpo_cfg.get('dock_center')
    protein_pdbqt_path = dpo_cfg.get('protein_pdbqt_path')
    ref_sdf_path = dpo_cfg.get('ref_sdf_path')
    protein_pdbqt_path = str(resolve_path(protein_pdbqt_path)) if protein_pdbqt_path else None
    ref_sdf_path = str(resolve_path(ref_sdf_path)) if ref_sdf_path else None
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
    unidock_prep_workers = int(dpo_cfg.get('unidock_prep_workers', 1))
    docking_mode = str(dpo_cfg.get('docking_mode', 'flexible')).lower()
    vina_score_file = dpo_cfg.get('vina_score_file')
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
            from pepar_diff.evaluation.permeability import Permeability
            perm_predictor = Permeability()
            perm_scores = perm_predictor(all_helms)
            valid_perm = perm_scores[perm_scores > -10]
            print(f"Permeability: {len(valid_perm)}/{len(all_helms)} valid, "
                  f"mean={valid_perm.mean():.4f}" if len(valid_perm) > 0 else "No valid permeability scores")
        except Exception as e:
            raise RuntimeError(
                "Permeability evaluation failed while reward_w_perm > 0. "
                "Use the isolated pepardiff-perm environment to export a score file and pass "
                "--perm_score_file (or set dpo.perm_score_file), or set reward_w_perm=0 "
                f"if you intentionally want Vina-only DPO. Original error: {e}"
            ) from e

    # Vina docking score (required).
    try:
        from pepar_diff.vina.dock import dock_helms
    except Exception as e:
        raise RuntimeError(
            f"Vina docking import failed, cannot continue DPO training: {e}"
        ) from e

    vina_scores = np.full(len(all_helms), INVALID_SCORE, dtype=np.float64)
    missing_vina_indices = list(range(len(all_helms)))
    if vina_score_file:
        vina_scores, missing_vina_indices, _ = load_cached_vina_scores(all_helms, vina_score_file)

    if missing_vina_indices:
        missing_helms = [all_helms[i] for i in missing_vina_indices]
        try:
            print(
                f"Docking runtime: Uni-Dock GPU, batch_size={unidock_batch_size}, "
                f"prep_workers={unidock_prep_workers}, "
                f"search_mode={unidock_search_mode}, exhaustiveness={vina_exhaustiveness}, "
                f"n_poses={vina_n_poses}, docking_mode={docking_mode}"
            )
            if protein_pdbqt_path or ref_sdf_path or dock_center:
                print(f"Docking receptor: {protein_pdbqt_path or 'default'}")
                print(f"Docking reference SDF: {ref_sdf_path or 'default'}")
                print(f"Docking center: {dock_center if dock_center is not None else 'reference SDF centroid'}")
                print(f"Docking box size: {dock_box_size}")
            if vina_score_file:
                print(
                    f"Docking {len(missing_helms)} missing HELM sequences "
                    f"(incremental cache: {resolve_path(vina_score_file)})"
                )
            docked_scores = np.asarray(
                dock_helms(
                    missing_helms,
                    protein_pdbqt_path=protein_pdbqt_path,
                    ref_sdf_path=ref_sdf_path,
                    dock_center=dock_center,
                    exhaustiveness=vina_exhaustiveness,
                    n_poses=vina_n_poses,
                    show_progress=vina_show_progress,
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
                    unidock_prep_workers=unidock_prep_workers,
                    score_log_path=vina_score_file,
                    docking_mode=docking_mode,
                ),
                dtype=np.float64,
            )
        except Exception as e:
            raise RuntimeError(
                f"Vina docking execution failed, cannot continue DPO training: {e}"
            ) from e

        if docked_scores.shape[0] != len(missing_vina_indices):
            raise RuntimeError(
                f"Vina returned {docked_scores.shape[0]} scores for {len(missing_vina_indices)} missing sequences."
            )
        for local_idx, global_idx in enumerate(missing_vina_indices):
            vina_scores[global_idx] = docked_scores[local_idx]
        if vina_score_file:
            print(f"Updated Vina score cache: {resolve_path(vina_score_file)}")
    else:
        print("Vina docking: all candidate scores loaded from cache, skipping docking")

    if vina_scores.shape[0] != len(all_helms):
        raise RuntimeError(
            f"Vina returned {vina_scores.shape[0]} scores for {len(all_helms)} sequences."
        )

    valid_vina_mask = vina_scores != INVALID_SCORE
    vina_invalid_score_cutoff = dpo_cfg.get('vina_invalid_score_cutoff')
    if vina_invalid_score_cutoff is not None:
        cutoff = float(vina_invalid_score_cutoff)
        cutoff_invalid_mask = vina_scores > cutoff
        newly_invalid = int((valid_vina_mask & cutoff_invalid_mask).sum())
        valid_vina_mask &= ~cutoff_invalid_mask
        if newly_invalid > 0:
            print(
                f"Vina cutoff: treating {newly_invalid} samples with "
                f"vina_score > {cutoff:.4f} as invalid before DPO."
            )
    valid_vina = vina_scores[valid_vina_mask]
    if len(valid_vina) == 0:
        raise RuntimeError(
            "Vina produced zero valid scores (all scores are INVALID_SCORE=0.0); aborting DPO."
        )

    print(f"Vina docking: {len(valid_vina)}/{len(all_helms)} valid, "
          f"mean={valid_vina.mean():.4f}")
    invalid_vina = int((~valid_vina_mask).sum())
    if invalid_vina > 0:
        print(
            f"Vina docking: {invalid_vina} samples are INVALID_SCORE and will be "
            "excluded before preference pair construction."
        )

    # Combined reward
    # Lower Vina is better; negate it so higher reward is better.
    reward_w_chem = float(dpo_cfg.get('reward_w_chemistry', 0.0))
    normalize_rewards = bool(dpo_cfg.get('reward_normalize', True))
    chemistry_target_len = dpo_cfg.get('chemistry_target_length', None)

    reward_vina = -vina_scores
    reward_perm = perm_scores
    chemistry_scores = compute_chemistry_scores(all_helms, target_len=chemistry_target_len)

    reward_vina_for_norm = reward_vina.copy()
    reward_perm_for_norm = reward_perm.copy()
    chemistry_for_norm = chemistry_scores.copy()
    reward_vina_for_norm[~valid_vina_mask] = np.nan
    reward_perm_for_norm[~valid_vina_mask] = np.nan
    chemistry_for_norm[~valid_vina_mask] = np.nan

    if normalize_rewards:
        norm_vina = robust_normalize(reward_vina_for_norm)
        norm_perm = robust_normalize(reward_perm_for_norm)
        norm_chem = robust_normalize(chemistry_for_norm)
    else:
        norm_vina = reward_vina_for_norm
        norm_perm = reward_perm_for_norm
        norm_chem = chemistry_for_norm

    _print_component_stats("Vina", reward_vina[valid_vina_mask], norm_vina[valid_vina_mask])
    _print_component_stats("Perm", reward_perm[valid_vina_mask], norm_perm[valid_vina_mask])
    if reward_w_chem > 0:
        _print_component_stats("Chemistry", chemistry_scores[valid_vina_mask], norm_chem[valid_vina_mask])

    all_rewards = w_vina * norm_vina + w_perm * norm_perm + reward_w_chem * norm_chem
    all_rewards[~valid_vina_mask] = -np.inf
    valid_rewards = all_rewards[valid_vina_mask]
    print(f"\nReward stats (valid docking only): mean={valid_rewards.mean():.4f}, "
          f"std={valid_rewards.std():.4f}, "
          f"min={valid_rewards.min():.4f}, max={valid_rewards.max():.4f}")

    reward_info = {
        'reward_vina': reward_vina,
        'reward_perm': reward_perm,
        'chemistry_scores': chemistry_scores,
        'norm_vina': norm_vina,
        'norm_perm': norm_perm,
        'norm_chemistry': norm_chem,
        'valid_vina_mask': valid_vina_mask,
    }
    return all_rewards, vina_scores, perm_scores, reward_info


def _detect_delimiter(header_line: str, path: Path) -> str:
    if path.suffix.lower() == '.tsv':
        return '\t'
    if '\t' in header_line and ',' not in header_line:
        return '\t'
    return ','


def load_cached_vina_scores(all_helms: list, score_file: str):
    """Load cached docking scores and report which HELMs still need docking."""
    score_path = resolve_path(score_file)
    scores = np.full(len(all_helms), INVALID_SCORE, dtype=np.float64)
    missing_indices = list(range(len(all_helms)))
    status_counter = Counter()

    if not score_path.exists():
        print(f"Vina cache not found yet: {score_path}")
        return scores, missing_indices, status_counter

    with open(score_path, 'r', newline='') as f:
        header_line = f.readline()
        if not header_line:
            print(f"Vina cache exists but is empty: {score_path}")
            return scores, missing_indices, status_counter
        delimiter = _detect_delimiter(header_line, score_path)
        f.seek(0)
        reader = csv.DictReader(f, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(
                "Vina score file must contain a header row with at least "
                "'helm' and 'vina_score' columns."
            )

        field_map = {name.strip().lower(): name for name in reader.fieldnames if name}
        helm_key = field_map.get('helm')
        score_key = (
            field_map.get('vina_score')
            or field_map.get('score')
        )
        status_key = field_map.get('status')
        if helm_key is None or score_key is None:
            raise ValueError(
                "Vina score file must contain 'helm' and 'vina_score' "
                "(or 'score') columns."
            )

        score_by_helm = {}
        status_by_helm = {}
        duplicate_rows = 0
        for row in reader:
            helm = (row.get(helm_key) or '').strip()
            if not helm:
                continue
            raw_score = (row.get(score_key) or '').strip()
            try:
                score = float(raw_score)
            except ValueError:
                score = INVALID_SCORE

            if helm in score_by_helm:
                duplicate_rows += 1
            score_by_helm[helm] = score
            status_by_helm[helm] = (row.get(status_key) or 'unknown').strip() if status_key else 'unknown'

    missing_indices = []
    for idx, helm in enumerate(all_helms):
        if helm not in score_by_helm:
            missing_indices.append(idx)
            continue
        scores[idx] = score_by_helm[helm]
        status_counter[status_by_helm.get(helm, 'unknown')] += 1

    cached = len(all_helms) - len(missing_indices)
    print(
        f"Loaded cached Vina scores for {cached}/{len(all_helms)} HELM sequences from {score_path}"
        + (f" ({duplicate_rows} duplicate rows overwritten by last occurrence)" if duplicate_rows else "")
    )
    if status_counter:
        status_text = ", ".join(f"{k}={v}" for k, v in sorted(status_counter.items()))
        print(f"  Cached docking status breakdown: {status_text}")

    return scores, missing_indices, status_counter


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
            lambda_gpt=gen_cfg.lambda_gpt,
            predict_ring_bonds=gen_cfg.predict_ring_bonds,
            ring_threshold=gen_cfg.ring_bond_threshold,
            ring_top_k=gen_cfg.ring_top_k,
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


def filter_invalid_docking_candidates(
    all_helms: list,
    all_rewards: np.ndarray,
    vina_scores: np.ndarray,
    perm_scores: np.ndarray,
    reward_info: dict,
    source_labels: list | None = None,
):
    """Drop candidates with invalid docking scores before pair construction."""
    valid_mask = None if reward_info is None else reward_info.get('valid_vina_mask')
    if valid_mask is None:
        return all_helms, all_rewards, vina_scores, perm_scores, reward_info, source_labels

    valid_mask = np.asarray(valid_mask, dtype=bool)
    if valid_mask.shape[0] != len(all_helms):
        raise ValueError("valid_vina_mask does not align with candidate HELM sequences.")

    dropped = int((~valid_mask).sum())
    if dropped == 0:
        return all_helms, all_rewards, vina_scores, perm_scores, reward_info, source_labels

    kept = int(valid_mask.sum())
    print(f"Filtering out {dropped} invalid docking candidates before DPO ({kept} remain)")

    filtered_helms = [helm for helm, keep in zip(all_helms, valid_mask) if keep]
    filtered_sources = None
    if source_labels is not None:
        filtered_sources = [label for label, keep in zip(source_labels, valid_mask) if keep]

    filtered_reward_info = {}
    for key, value in (reward_info or {}).items():
        if isinstance(value, np.ndarray) and value.shape[0] == valid_mask.shape[0]:
            filtered_reward_info[key] = value[valid_mask]
        else:
            filtered_reward_info[key] = value
    filtered_reward_info['valid_vina_mask'] = np.ones(kept, dtype=bool)

    return (
        filtered_helms,
        np.asarray(all_rewards)[valid_mask],
        np.asarray(vina_scores)[valid_mask],
        np.asarray(perm_scores)[valid_mask],
        filtered_reward_info,
        filtered_sources,
    )


def main():
    args = parse_args()

    with open(args.config, 'r') as f:
        raw_config = json.load(f)
    checkpoint_dir = raw_config.get('training', {}).get('checkpoint_dir', './checkpoints/ald_dpo')
    log_path = setup_output_logging(checkpoint_dir)

    dpo_cfg = raw_config.setdefault('dpo', {})
    if args.seed is not None:
        dpo_cfg['seed'] = int(args.seed)
    training_seed = dpo_cfg.get('seed')
    training_seed = None if training_seed is None else int(training_seed)
    deterministic = bool(dpo_cfg.get('deterministic', False))
    configure_reproducibility(training_seed, deterministic=deterministic)

    config = ALDConfig.load(args.config)
    print(f"Loading config from: {args.config}")
    print(f"Training log will be written to: {log_path}")

    # Load DPO-specific config
    if args.perm_score_file:
        dpo_cfg['perm_score_file'] = args.perm_score_file
    if args.vina_score_file:
        dpo_cfg['vina_score_file'] = args.vina_score_file
    elif not dpo_cfg.get('vina_score_file'):
        dpo_cfg['vina_score_file'] = str(Path(config.training.checkpoint_dir) / 'dpo_data' / 'vina_scores.csv')

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

        (
            all_helms,
            all_rewards,
            vina_scores,
            perm_scores,
            reward_info,
            source_labels,
        ) = filter_invalid_docking_candidates(
            all_helms,
            all_rewards,
            vina_scores,
            perm_scores,
            reward_info,
            source_labels=source_labels,
        )

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
            loser_vina_score_min=dpo_cfg.get('loser_vina_score_min'),
            loser_vina_score_max=dpo_cfg.get('loser_vina_score_max'),
            allow_loser_pool_fallback=bool(
                dpo_cfg.get('allow_loser_pool_fallback', False)
            ),
        )

    # Snapshot aligned pairs for both newly constructed and pre-computed inputs.
    save_dir = Path(config.training.checkpoint_dir) / 'dpo_data'
    pair_manifest_path = save_preference_pair_snapshot(
        winner_helms,
        loser_helms,
        save_dir,
        preserve_pairing=bool(dpo_cfg.get('preserve_pairing', True)),
        source_winner_file=args.winner_file if args.skip_generate else None,
        source_loser_file=args.loser_file if args.skip_generate else None,
    )
    if not args.skip_generate:
        save_helm_list(all_helms, str(save_dir / 'candidates.txt'))
        np.save(str(save_dir / 'rewards.npy'), all_rewards)
        if reward_info:
            for key, value in reward_info.items():
                np.save(str(save_dir / f"{key}.npy"), np.asarray(value))
        with open(save_dir / 'sources.json', 'w') as f:
            json.dump(source_labels, f, ensure_ascii=False, indent=2)
    print(f"Saved preference data to {save_dir}")
    print(f"Preference manifest: {pair_manifest_path}")

    if args.prepare_pairs_only:
        print("Preference-pair preparation complete; skipping DPO training.")
        return

    # ── Create dataset & dataloader ──
    dataset = PreferencePairDataset(
        winner_helms=winner_helms,
        loser_helms=loser_helms,
        vocab_file=config.training.vocab_file,
        max_seq_len=config.model.max_seq_len,
        preserve_pairing=bool(dpo_cfg.get('preserve_pairing', True)),
        shuffle_seed=training_seed if training_seed is not None else 42,
    )

    dataloader_generator = None
    if training_seed is not None:
        dataloader_generator = torch.Generator()
        dataloader_generator.manual_seed(training_seed)

    dataloader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        collate_fn=PreferencePairCollator(),
        drop_last=True,
        generator=dataloader_generator,
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
        dpop_winner_reg_alpha=dpo_cfg.get('dpop_winner_reg_alpha', 0.0),
        dpop_winner_reg_mode=dpo_cfg.get('dpop_winner_reg_mode', 'external_reg'),
        checkpoint_dir=config.training.checkpoint_dir,
        device=str(device),
        sampling_seed=training_seed,
        deterministic=deterministic,
        audit_sampling_trace=bool(dpo_cfg.get('audit_sampling_trace', False)),
    )

    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # ── Train ──
    num_epochs = dpo_cfg.get('num_epochs', 10)
    log_interval = dpo_cfg.get('log_interval', 10)
    save_interval = dpo_cfg.get('save_interval_epochs', 1)
    epoch_sample_count = int(dpo_cfg.get('epoch_sample_count', 0))
    epoch_sample_seed = int(
        dpo_cfg.get(
            'epoch_sample_seed',
            (training_seed if training_seed is not None else 42) + 1_000_000,
        )
    )
    epoch_sample_dir = Path(config.training.checkpoint_dir) / 'epoch_samples'
    if epoch_sample_count > 0 and not args.resume:
        epoch_sample_dir.mkdir(parents=True, exist_ok=True)
        with open(epoch_sample_dir / 'manifest.jsonl', 'w', encoding='utf-8') as f:
            f.write('')

    print(f"\n{'='*60}")
    print(f"DPO Training")
    print(f"{'='*60}")
    print(f"  beta_dpo:       {trainer.beta_dpo}")
    print(f"  num_epochs:     {num_epochs}")
    print(f"  batch_size:     {config.training.batch_size}")
    print(f"  lr:             {dpo_cfg.get('lr', 1e-5)}")
    print(f"  freeze_mode:    {dpo_cfg.get('freeze_mode', 'denoiser_only')}")
    print(f"  dpop_w_reg_alpha:{dpo_cfg.get('dpop_winner_reg_alpha', 0.0)}")
    print(f"  dpop_w_reg_mode: {dpo_cfg.get('dpop_winner_reg_mode', 'external_reg')}")
    print(f"  seed:             {training_seed}")
    print(f"  deterministic:    {deterministic}")
    print(f"  samples/epoch:    {epoch_sample_count}")
    if epoch_sample_count > 0:
        print(f"  epoch_sample_seed:{epoch_sample_seed} + epoch")
    print(f"  dataset_size:   {len(dataset)}")
    print(f"  steps/epoch:    {len(dataloader)}")
    print(f"{'='*60}\n")

    t_start = time.time()
    epoch_metrics_path = Path(config.training.checkpoint_dir) / "epoch_metrics.jsonl"
    epoch_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(epoch_metrics_path, "w", encoding="utf-8") as f:
        f.write("")
    print(f"Epoch metrics JSONL: {epoch_metrics_path}")

    for epoch in range(1, num_epochs + 1):
        metrics = trainer.train_epoch(log_interval=log_interval)
        metrics_record = {
            "epoch": epoch,
            "global_step": trainer.global_step,
            **{key: float(value) for key, value in metrics.items()},
        }
        with open(epoch_metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics_record, ensure_ascii=False) + "\n")

        # Save checkpoint
        if epoch % save_interval == 0:
            trainer.save_checkpoint()

        if epoch_sample_count > 0:
            sample_path = generate_epoch_samples(
                model=trainer.model,
                config=config,
                device=device,
                epoch=trainer.epoch,
                num_samples=epoch_sample_count,
                base_seed=epoch_sample_seed,
                output_dir=epoch_sample_dir,
                alpha_win=trainer.dpop_winner_reg_alpha,
            )
            print(
                f"Generated {epoch_sample_count} epoch-{trainer.epoch} samples: "
                f"{sample_path}"
            )

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
