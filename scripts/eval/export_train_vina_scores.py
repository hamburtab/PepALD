"""
Export or resume cached Vina docking scores for HELM candidate files.

This script reuses the same Uni-Dock scoring settings as DPO training, but it
stops after writing / updating the Vina score cache file.

Usage:
    python scripts/eval/export_train_vina_scores.py
    python scripts/eval/export_train_vina_scores.py --config configs/training/dpo.json
    python scripts/eval/export_train_vina_scores.py --sample_file outputs/samples/dpo_train_data/combined_candidates.txt
    python scripts/eval/export_train_vina_scores.py --vina_score_file outputs/samples/dpo_train_data/combined_candidates.vina.csv
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train.train_dpo import load_cached_vina_scores, load_candidate_helms, resolve_path
from pepar_diff.vina.constants import INVALID_SCORE
from pepar_diff.vina.dock import dock_helms


def parse_args():
    parser = argparse.ArgumentParser(description="Export cached Vina docking scores only")
    parser.add_argument(
        "--config", type=str, default="configs/training/dpo.json",
        help="Path to DPO config file (reads dpo section for sample files and docking settings)"
    )
    parser.add_argument(
        "--sample_file", type=str, default=None,
        help="Optional single HELM file override. If omitted, uses dpo.sample_files / dpo.sample_file."
    )
    parser.add_argument(
        "--vina_score_file", type=str, default=None,
        help="Optional Vina score cache CSV override."
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Only score the first N deduplicated HELM sequences."
    )
    return parser.parse_args()


def print_vina_summary(helms: list[str], scores: np.ndarray) -> None:
    valid_mask = scores != INVALID_SCORE
    valid_scores = scores[valid_mask]
    invalid_count = int(np.sum(~valid_mask))

    print("\n=== Vina Summary ===")
    print(f"  Unique samples scored:   {len(helms)}")
    print(f"  Valid docking scores:    {len(valid_scores)}")
    print(f"  Invalid / failed scores: {invalid_count}")

    if len(valid_scores) == 0:
        print("  No valid Vina scores were produced.")
        return

    valid_indices = np.flatnonzero(valid_mask)
    best_idx = int(valid_indices[np.argmin(valid_scores)])
    top_k = min(10, len(valid_scores))
    top_mean = float(np.mean(np.sort(valid_scores)[:top_k]))

    print(f"  Mean Vina:               {valid_scores.mean():.4f}")
    print(f"  Median Vina:             {np.median(valid_scores):.4f}")
    print(f"  Std Vina:                {valid_scores.std():.4f}")
    print(f"  Best Vina:               {scores[best_idx]:.4f}")
    print(f"  Best HELM:               {helms[best_idx]}")
    print(f"  Top-{top_k} mean Vina:        {top_mean:.4f}")


def main():
    args = parse_args()

    with open(args.config, "r") as f:
        full_config = json.load(f)
    dpo_cfg = full_config.get("dpo", {})

    sample_files = []
    if args.sample_file:
        sample_files = [args.sample_file]
    elif dpo_cfg.get("sample_files"):
        sample_files = list(dpo_cfg.get("sample_files", []))
    elif dpo_cfg.get("sample_file"):
        sample_files = [dpo_cfg.get("sample_file")]

    if not sample_files:
        raise ValueError("No candidate file specified. Use --sample_file or set dpo.sample_files / dpo.sample_file.")

    vina_score_file = (
        args.vina_score_file
        or dpo_cfg.get("vina_score_file")
        or "outputs/samples/dpo_train_data/combined_candidates.vina.csv"
    )

    print(f"Loading config from: {args.config}")
    all_helms, source_labels, sample_paths = load_candidate_helms(sample_files)
    print(f"Using candidate set from: {', '.join(str(p) for p in sample_paths)}")
    print(f"Loaded {len(all_helms)} unique HELM sequences from {len(sample_paths)} file(s)")

    if args.max_samples is not None and args.max_samples < len(all_helms):
        all_helms = all_helms[:args.max_samples]
        source_labels = source_labels[:args.max_samples]
        print(f"Using first {args.max_samples} unique HELM sequences")

    vina_exhaustiveness = int(dpo_cfg.get("vina_exhaustiveness", 8))
    vina_n_poses = int(dpo_cfg.get("vina_n_poses", 2))
    vina_show_progress = bool(dpo_cfg.get("vina_show_progress", True))
    dock_box_size = dpo_cfg.get("dock_box_size", 30.0)
    dock_center = dpo_cfg.get("dock_center")
    protein_pdbqt_path = dpo_cfg.get("protein_pdbqt_path")
    ref_sdf_path = dpo_cfg.get("ref_sdf_path")
    protein_pdbqt_path = str(resolve_path(protein_pdbqt_path)) if protein_pdbqt_path else None
    ref_sdf_path = str(resolve_path(ref_sdf_path)) if ref_sdf_path else None
    dock_seed = int(dpo_cfg.get("dock_seed", 42))
    unidock_binary = str(dpo_cfg.get("unidock_binary", "unidock"))
    unidock_batch_size = int(dpo_cfg.get("unidock_batch_size", 64))
    unidock_search_mode = str(dpo_cfg.get("unidock_search_mode", "fast"))
    unidock_scoring = str(dpo_cfg.get("unidock_scoring", "vina"))
    unidock_refine_step = int(dpo_cfg.get("unidock_refine_step", 3))
    unidock_max_step = int(dpo_cfg.get("unidock_max_step", 20))
    unidock_max_gpu_memory = int(dpo_cfg.get("unidock_max_gpu_memory", 0))
    unidock_keep_workdir = bool(dpo_cfg.get("unidock_keep_workdir", False))
    unidock_verbosity = int(dpo_cfg.get("unidock_verbosity", 0))
    unidock_prep_workers = int(dpo_cfg.get("unidock_prep_workers", 1))

    cache_path = resolve_path(vina_score_file)
    scores = np.full(len(all_helms), INVALID_SCORE, dtype=np.float64)
    missing_indices = list(range(len(all_helms)))
    if cache_path.exists():
        scores, missing_indices, _ = load_cached_vina_scores(all_helms, vina_score_file)
    else:
        print(f"Vina cache will be created at: {cache_path}")

    if missing_indices:
        missing_helms = [all_helms[i] for i in missing_indices]
        print(
            f"Docking {len(missing_helms)} missing HELM sequences with Uni-Dock "
            f"(batch_size={unidock_batch_size}, prep_workers={unidock_prep_workers}, "
            f"search_mode={unidock_search_mode}, n_poses={vina_n_poses})"
        )
        if protein_pdbqt_path or ref_sdf_path or dock_center:
            print(f"Docking receptor: {protein_pdbqt_path or 'default'}")
            print(f"Docking reference SDF: {ref_sdf_path or 'default'}")
            print(f"Docking center: {dock_center if dock_center is not None else 'reference SDF centroid'}")
            print(f"Docking box size: {dock_box_size}")
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
            ),
            dtype=np.float64,
        )
        if docked_scores.shape[0] != len(missing_indices):
            raise RuntimeError(
                f"Docking returned {docked_scores.shape[0]} scores for {len(missing_indices)} missing sequences."
            )
        for local_idx, global_idx in enumerate(missing_indices):
            scores[global_idx] = docked_scores[local_idx]
        print(f"Updated Vina cache: {cache_path}")
    else:
        print("All candidate HELM sequences already have cached Vina scores; nothing to dock.")

    print(f"\nVina cache summary: {cache_path}")
    print(f"  Total sequences: {len(all_helms)}")
    print_vina_summary(all_helms, scores)
    print("\nDone. (This script only exports / resumes Vina scoring, no DPO training started.)")


if __name__ == "__main__":
    main()
