"""
Export PepTune solubility scores for a HELM or SMILES candidate file.

The score is PepTune's XGBoost positive-class probability for solubility.

Usage:
    python scripts/eval/evaluate_solubility_scores.py
    python scripts/eval/evaluate_solubility_scores.py \
        --input outputs/samples/case1/generated/helm_dpo_samples.txt \
        --output outputs/samples/case1/generated/helm_dpo_samples.solubility.csv
"""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from pepar_diff.evaluation.solubility import PepTuneSolubility


DEFAULT_INPUT = "outputs/samples/case1/generated/helm_dpo_samples.txt"
# DEFAULT_INPUT = "outputs/samples/finetune/permeability1000_samples.txt"
DEFAULT_OUTPUT = "outputs/samples/case1/generated/helm_dpo_samples.solubility.csv"
# DEFAULT_OUTPUT = "outputs/samples/finetune/permeability1000_samples.solubility.csv"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export PepTune solubility positive-class probabilities"
    )
    parser.add_argument(
        "--input", default=DEFAULT_INPUT, type=str,
        help="Candidate file, one HELM or SMILES sequence per line"
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT, type=str,
        help="Output CSV file"
    )
    parser.add_argument(
        "--input_type", choices=["helm", "smiles"], default="helm",
        help="Interpret input lines as HELM or SMILES"
    )
    parser.add_argument(
        "--peptune_dir", default=None, type=str,
        help="Optional local PepTune repo or downloaded Hugging Face snapshot"
    )
    parser.add_argument(
        "--model_path", default=None, type=str,
        help="Optional explicit PepTune solubility XGBoost model path"
    )
    parser.add_argument(
        "--vocab_path", default=None, type=str,
        help="Optional explicit PepTune tokenizer new_vocab.txt path"
    )
    parser.add_argument(
        "--splits_path", default=None, type=str,
        help="Optional explicit PepTune tokenizer new_splits.txt path"
    )
    parser.add_argument(
        "--tokenizer_module_path", default=None, type=str,
        help="Optional explicit PepTune tokenizer/my_tokenizers.py path"
    )
    parser.add_argument(
        "--batch_size", default=1, type=int,
        help="Embedding batch size. Default 1 matches PepTune's original scorer."
    )
    parser.add_argument(
        "--device", default=None, type=str,
        help="Torch device, for example cuda, cpu, or mps. Defaults to cuda if available."
    )
    parser.add_argument(
        "--hf_endpoint", default=None, type=str,
        help="Optional Hugging Face endpoint, for example https://hf-mirror.com"
    )
    return parser.parse_args()


def resolve_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def load_sequence_list(path: Path):
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def format_score(score: float) -> str:
    if np.isfinite(score):
        return f"{float(score):.8f}"
    return "nan"


def main():
    args = parse_args()
    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    sequences = load_sequence_list(input_path)
    if not sequences:
        raise ValueError(f"No valid sequences found in: {input_path}")

    print(f"Using input:  {input_path}")
    print(f"Using output: {output_path}")
    print(f"Loaded {len(sequences)} {args.input_type.upper()} sequences")

    scorer = PepTuneSolubility(
        peptune_dir=args.peptune_dir,
        model_path=args.model_path,
        vocab_path=args.vocab_path,
        splits_path=args.splits_path,
        tokenizer_module_path=args.tokenizer_module_path,
        batch_size=args.batch_size,
        input_type=args.input_type,
        device=args.device,
        hf_endpoint=args.hf_endpoint,
    )
    scores, smiles_list, statuses = scorer.predict_with_details(sequences)

    input_column = "helm" if args.input_type == "helm" else "input_smiles"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([input_column, "smiles", "solubility", "status"])
        for seq, smiles, score, status in zip(sequences, smiles_list, scores, statuses):
            writer.writerow([seq, smiles, format_score(score), status])

    valid_mask = np.isfinite(scores)
    valid_scores = scores[valid_mask]
    status_counts = Counter(statuses)

    print(f"Saved solubility scores to {output_path}")
    print(f"  Total sequences: {len(sequences)}")
    print(f"  Valid predictions: {len(valid_scores)}")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")
    if len(valid_scores) > 0:
        print(f"  Mean solubility: {valid_scores.mean():.4f}")
        print(f"  Std solubility:  {valid_scores.std():.4f}")
        print(f"  Min solubility:  {valid_scores.min():.4f}")
        print(f"  Max solubility:  {valid_scores.max():.4f}")
        print(
            "  Predicted soluble (> 0.500): "
            f"{(valid_scores > 0.5).sum()} ({(valid_scores > 0.5).mean() * 100:.1f}%)"
        )


if __name__ == "__main__":
    main()
