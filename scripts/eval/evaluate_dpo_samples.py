"""
Evaluate generated DPO samples with:
  1. Vina docking summary (same Uni-Dock settings as DPO)
  2. Full structural metrics from pepar_diff.utils.metrics.Metrics

Default input:
    outputs/samples/helm_dpo_samples.txt

Default Vina cache/output:
    outputs/samples/dpo_generated_samples_score.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train.train_dpo import load_cached_vina_scores, resolve_path
from pepar_diff.utils.metrics import Metrics
from pepar_diff.vina.constants import INVALID_SCORE
from pepar_diff.vina.dock import dock_helms


DEFAULT_INPUT = "outputs/samples/helm_dpo_samples.txt"
DEFAULT_VINA_OUTPUT = "outputs/samples/dpo_generated_samples_score.csv"
DEFAULT_PRIOR = "data/processed/prior_data.csv"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DPO generated samples")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training/dpo.json",
        help="Path to config file; reads Uni-Dock settings from its dpo section.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_INPUT,
        help="Input HELM sample file (one sequence per line).",
    )
    parser.add_argument(
        "--vina_score_file",
        type=str,
        default=DEFAULT_VINA_OUTPUT,
        help="CSV file used to cache/write Vina scores.",
    )
    parser.add_argument(
        "--prior_path",
        type=str,
        default=DEFAULT_PRIOR,
        help="Prior CSV for eval_full_metrics-style comparison.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional limit on the number of raw generated HELM sequences to evaluate.",
    )
    return parser.parse_args()


def load_helm_list(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def deduplicate_preserve_order(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def summarize_vina(helms: list[str], scores: np.ndarray):
    valid_mask = scores != INVALID_SCORE
    valid_scores = scores[valid_mask]
    invalid_count = int((~valid_mask).sum())

    print("\n=== Vina Summary ===")
    print(f"  Unique samples scored:   {len(helms)}")
    print(f"  Valid docking scores:    {len(valid_scores)}")
    print(f"  Invalid / failed scores: {invalid_count}")

    if len(valid_scores) == 0:
        print("  No valid Vina scores were produced.")
        return

    best_idx = int(np.argmin(scores))
    top_k = min(10, len(valid_scores))
    top_mean = float(np.mean(np.sort(valid_scores)[:top_k]))

    print(f"  Mean Vina:               {valid_scores.mean():.4f}")
    print(f"  Median Vina:             {np.median(valid_scores):.4f}")
    print(f"  Std Vina:                {valid_scores.std():.4f}")
    print(f"  Best Vina:               {scores[best_idx]:.4f}")
    print(f"  Best HELM:               {helms[best_idx]}")
    print(f"  Top-{top_k} mean Vina:        {top_mean:.4f}")


def compact_score_csv(score_file: Path, helms: list[str], scores: np.ndarray):
    score_file.parent.mkdir(parents=True, exist_ok=True)
    with open(score_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["helm", "vina_score", "status", "detail"])
        writer.writeheader()
        for helm, score in zip(helms, scores):
            if score != INVALID_SCORE:
                status = "ok"
                detail = ""
            else:
                status = "invalid"
                detail = "INVALID_SCORE"
            writer.writerow(
                {
                    "helm": helm,
                    "vina_score": f"{float(score):.8f}",
                    "status": status,
                    "detail": detail,
                }
            )


def main():
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        full_config = json.load(f)
    dpo_cfg = full_config.get("dpo", {})

    input_path = resolve_path(args.input)
    vina_score_path = resolve_path(args.vina_score_file)
    prior_path = resolve_path(args.prior_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input sample file not found: {input_path}")
    if not prior_path.exists():
        raise FileNotFoundError(f"Prior CSV not found: {prior_path}")

    raw_helms = load_helm_list(input_path)
    if not raw_helms:
        raise ValueError(f"No valid HELM sequences found in: {input_path}")

    if args.max_samples is not None and args.max_samples < len(raw_helms):
        raw_helms = raw_helms[:args.max_samples]

    unique_helms = deduplicate_preserve_order(raw_helms)

    print(f"Loading config from: {args.config}")
    print(f"Input samples: {input_path}")
    print(f"Raw generated samples: {len(raw_helms)}")
    print(f"Unique generated samples: {len(unique_helms)}")
    print(f"Duplicate samples removed for docking: {len(raw_helms) - len(unique_helms)}")
    print(f"Vina score CSV: {vina_score_path}")

    print("\n=== Full Metrics (eval_full_metrics style) ===")
    metrics = Metrics(
        prior_path=str(prior_path),
        n_jobs=1,
        input_type="helm",
    )
    metric_results = metrics.get_metrics(raw_helms)
    for key, value in metric_results.items():
        print(f"  {key}: {value:.4f}")

    scores = np.full(len(unique_helms), INVALID_SCORE, dtype=np.float64)
    missing_indices = list(range(len(unique_helms)))
    if vina_score_path.exists():
        scores, missing_indices, _ = load_cached_vina_scores(unique_helms, str(vina_score_path))
    else:
        print(f"\nVina cache will be created at: {vina_score_path}")

    if missing_indices:
        missing_helms = [unique_helms[i] for i in missing_indices]
        print(
            f"\nDocking {len(missing_helms)} missing unique HELM sequences with Uni-Dock "
            f"(batch_size={int(dpo_cfg.get('unidock_batch_size', 64))}, "
            f"search_mode={str(dpo_cfg.get('unidock_search_mode', 'fast'))}, "
            f"n_poses={int(dpo_cfg.get('vina_n_poses', 2))})"
        )
        docked_scores = np.asarray(
            dock_helms(
                missing_helms,
                exhaustiveness=int(dpo_cfg.get("vina_exhaustiveness", 8)),
                n_poses=int(dpo_cfg.get("vina_n_poses", 2)),
                show_progress=bool(dpo_cfg.get("vina_show_progress", True)),
                box_size=dpo_cfg.get("dock_box_size", 30.0),
                seed=int(dpo_cfg.get("dock_seed", 42)),
                unidock_binary=str(dpo_cfg.get("unidock_binary", "unidock")),
                unidock_batch_size=int(dpo_cfg.get("unidock_batch_size", 64)),
                unidock_search_mode=str(dpo_cfg.get("unidock_search_mode", "fast")),
                unidock_scoring=str(dpo_cfg.get("unidock_scoring", "vina")),
                unidock_refine_step=int(dpo_cfg.get("unidock_refine_step", 3)),
                unidock_max_step=int(dpo_cfg.get("unidock_max_step", 20)),
                unidock_max_gpu_memory=int(dpo_cfg.get("unidock_max_gpu_memory", 0)),
                unidock_keep_workdir=bool(dpo_cfg.get("unidock_keep_workdir", False)),
                unidock_verbosity=int(dpo_cfg.get("unidock_verbosity", 0)),
                score_log_path=str(vina_score_path),
            ),
            dtype=np.float64,
        )
        if docked_scores.shape[0] != len(missing_indices):
            raise RuntimeError(
                f"Docking returned {docked_scores.shape[0]} scores for {len(missing_indices)} missing sequences."
            )
        for local_idx, global_idx in enumerate(missing_indices):
            scores[global_idx] = docked_scores[local_idx]
    else:
        print("\nAll unique generated HELM sequences already have cached Vina scores.")

    compact_score_csv(vina_score_path, unique_helms, scores)
    print(f"Rewrote compact Vina score CSV to: {vina_score_path}")

    summarize_vina(unique_helms, scores)
    print("\nDone.")


if __name__ == "__main__":
    main()
