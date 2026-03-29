"""
Standalone reward evaluation script.

Only evaluates rewards and prints preference pair statistics.
Does NOT start DPO training.

Usage:
    python eval_rewards.py
    python eval_rewards.py --sample_file chembl32_samples/head_tail_single_cycle_samples.txt
    python eval_rewards.py --sample_file samples.txt --top_ratio 0.25 --bottom_ratio 0.25
    python eval_rewards.py --sample_file samples.txt --max_samples 100
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from train_dpo import evaluate_rewards, load_helm_list, resolve_path
from ald.dpo.dataset import build_preference_pairs


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate rewards for HELM sequences (no training)")
    parser.add_argument(
        "--config", type=str, default="configs/dpo.json",
        help="Path to DPO config file (reads dpo section for weights/ratios)"
    )
    parser.add_argument(
        "--sample_file", type=str, default=None,
        help="HELM sequences file (one per line). Defaults to dpo.sample_file in config."
    )
    parser.add_argument(
        "--top_ratio", type=float, default=None,
        help="Top ratio for winners (overrides config)"
    )
    parser.add_argument(
        "--bottom_ratio", type=float, default=None,
        help="Bottom ratio for losers (overrides config)"
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Only evaluate the first N samples (for quick testing)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    with open(args.config, 'r') as f:
        full_config = json.load(f)
    dpo_cfg = full_config.get('dpo', {})

    # Resolve sample file
    sample_file = args.sample_file or dpo_cfg.get('sample_file')
    if not sample_file:
        print("No sample file specified. Use --sample_file or set dpo.sample_file in config.")
        sys.exit(1)

    sample_path = resolve_path(sample_file)
    if not sample_path.exists():
        print(f"Sample file not found: {sample_path}")
        sys.exit(1)

    # Load sequences
    all_helms = load_helm_list(str(sample_path))
    print(f"Loaded {len(all_helms)} HELM sequences from {sample_path}")

    if args.max_samples and args.max_samples < len(all_helms):
        all_helms = all_helms[:args.max_samples]
        print(f"Using first {args.max_samples} samples for evaluation")

    # Evaluate rewards
    all_rewards, vina_scores, perm_scores = evaluate_rewards(all_helms, dpo_cfg)

    # Build preference pairs (with detailed stats)
    top_ratio = args.top_ratio or dpo_cfg.get('top_ratio', 0.2)
    bottom_ratio = args.bottom_ratio or dpo_cfg.get('bottom_ratio', 0.2)
    w_vina = dpo_cfg.get('reward_w_vina', 1.0)
    w_perm = dpo_cfg.get('reward_w_perm', 0.5)

    winner_helms, loser_helms = build_preference_pairs(
        all_helms, all_rewards, top_ratio, bottom_ratio,
        vina_scores=vina_scores, perm_scores=perm_scores,
        reward_w_vina=w_vina, reward_w_perm=w_perm,
    )

    # Print some example sequences
    print(f"\n--- Example Winners (top 5) ---")
    sorted_idx = np.argsort(all_rewards)
    for i in sorted_idx[-5:][::-1]:
        print(f"  reward={all_rewards[i]:+.4f}  vina={vina_scores[i]:+.4f}  perm={perm_scores[i]:+.4f}  {all_helms[i]}")

    print(f"\n--- Example Losers (bottom 5) ---")
    for i in sorted_idx[:5]:
        print(f"  reward={all_rewards[i]:+.4f}  vina={vina_scores[i]:+.4f}  perm={perm_scores[i]:+.4f}  {all_helms[i]}")

    print(f"\nDone. (This script only evaluates, no training started.)")


if __name__ == "__main__":
    main()
