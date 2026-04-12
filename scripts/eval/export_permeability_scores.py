"""
Export permeability scores for a HELM candidate file.

Run this script inside the isolated `pepardiff-perm` environment, then pass the
resulting score file to train_dpo.py / eval_rewards.py via --perm_score_file.

Usage:
    python scripts/eval/export_permeability_scores.py
    python scripts/eval/export_permeability_scores.py \
        --input outputs/samples/helm_chembl32only_r1r2_cyclized.txt \
        --output outputs/samples/helm_chembl32only_r1r2_cyclized.perm.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from pepar_diff.evaluation.permeability import Permeability


DEFAULT_INPUT = "outputs/samples/helm_dpo_samples.txt"
DEFAULT_OUTPUT = "outputs/samples/helm_dpo_samples.perm.csv"


def parse_args():
    parser = argparse.ArgumentParser(description="Export precomputed permeability scores")
    parser.add_argument(
        "--input", default=DEFAULT_INPUT, type=str,
        help="Candidate HELM file (one sequence per line)"
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT, type=str,
        help="Output CSV file with helm,permeability"
    )
    return parser.parse_args()


def resolve_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def load_helm_list(path: Path):
    with open(path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def main():
    args = parse_args()
    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input HELM file not found: {input_path}")

    helms = load_helm_list(input_path)
    if not helms:
        raise ValueError(f"No valid HELM sequences found in: {input_path}")

    print(f"Using input:  {input_path}")
    print(f"Using output: {output_path}")
    print(f"Loaded {len(helms)} HELM sequences from {input_path}")
    predictor = Permeability()
    scores = predictor(helms)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['helm', 'permeability'])
        for helm, score in zip(helms, scores):
            writer.writerow([helm, f"{float(score):.8f}"])

    valid_scores = scores[scores > -10]
    print(f"Saved permeability scores to {output_path}")
    print(f"  Total sequences: {len(helms)}")
    print(f"  Valid predictions: {len(valid_scores)}")
    if len(valid_scores) > 0:
        print(f"  Mean permeability: {valid_scores.mean():.4f}")
        print(f"  Std permeability:  {valid_scores.std():.4f}")
        print(f"  Min permeability:  {valid_scores.min():.4f}")
        print(f"  Max permeability:  {valid_scores.max():.4f}")
        print(f"  High permeability (> -6): {(valid_scores > -6).sum()} ({(valid_scores > -6).mean() * 100:.1f}%)")

    invalid_count = int(np.sum(scores <= -10))
    if invalid_count > 0:
        print(f"  Invalid predictions: {invalid_count}")


if __name__ == "__main__":
    main()
