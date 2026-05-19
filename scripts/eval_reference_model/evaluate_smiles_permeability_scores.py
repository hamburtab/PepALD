"""
Export permeability scores for a CSV file with a SMILES column.

This script reuses pepar_diff.evaluation.permeability.Permeability with
input_type="smiles".

Usage:
    python scripts/eval_reference_model/evaluate_smiles_permeability_scores.py \
        --input samples.csv \
        --output samples.permeability.csv

Input CSV example:
    smiles
    NCN1c2ccc(C(=O)N22CCC(CC(=O)O)cc1)C(NC(=O)[C@@H](CSCCN=C2ccccN)NC(=O)N2(N2CC3OCCCNC(3N)C(=O)N12)NC(=O)N[O-]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from pepar_diff.evaluation.permeability import Permeability

rdBase.DisableLog("rdApp.error")


SMILES_COLUMN_CANDIDATES = (
    "smiles",
    "SMILES",
    "Smiles",
    "canonical_smiles",
    "cano_smi",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export permeability scores for a CSV file containing SMILES."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=str,
        help="Input CSV file with one SMILES column.",
    )
    parser.add_argument(
        "--output",
        default=None,
        type=str,
        help=(
            "Output CSV with input_smiles, canonical_smiles, permeability, and "
            "status. Defaults to '<input>.permeability.csv'."
        ),
    )
    parser.add_argument(
        "--smiles_column",
        default=None,
        type=str,
        help="Name of the SMILES column. If omitted, common names are auto-detected.",
    )
    parser.add_argument(
        "--model_path",
        default=None,
        type=str,
        help="Optional explicit permeability RandomForest model path.",
    )
    parser.add_argument(
        "--batch_size",
        default=1000,
        type=int,
        help="Prediction batch size passed to the Permeability wrapper.",
    )
    return parser.parse_args()


def resolve_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}.permeability.csv")


def find_column(df: pd.DataFrame, explicit: str | None) -> str:
    if explicit is not None:
        if explicit not in df.columns:
            raise ValueError(
                f"Column '{explicit}' not found in input CSV. "
                f"Available columns: {', '.join(map(str, df.columns))}"
            )
        return explicit

    for name in SMILES_COLUMN_CANDIDATES:
        if name in df.columns:
            return name

    lower_to_original = {str(col).lower(): col for col in df.columns}
    for name in SMILES_COLUMN_CANDIDATES:
        col = lower_to_original.get(name.lower())
        if col is not None:
            return str(col)

    if len(df.columns) == 1:
        return str(df.columns[0])

    raise ValueError(
        "Could not auto-detect a SMILES column. Pass --smiles_column. "
        f"Available columns: {', '.join(map(str, df.columns))}"
    )


def load_smiles_csv(path: Path, smiles_column: str | None) -> tuple[list[str], str]:
    df = pd.read_csv(path)
    column = find_column(df, smiles_column)
    smiles = df[column].dropna().astype(str).str.strip().tolist()
    smiles = [smi for smi in smiles if smi]
    if not smiles:
        raise ValueError(f"No non-empty SMILES found in column '{column}': {path}")
    return smiles, column


def canonicalize_smiles(smiles: list[str]) -> tuple[list[str], list[bool]]:
    canonical = []
    valid_mask = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is None:
            canonical.append("")
            valid_mask.append(False)
        else:
            canonical.append(Chem.MolToSmiles(mol))
            valid_mask.append(True)
    return canonical, valid_mask


def format_score(score: float) -> str:
    if np.isfinite(score):
        return f"{float(score):.8f}"
    return "nan"


def main() -> None:
    args = parse_args()
    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output) if args.output else default_output_path(input_path)
    model_path = str(resolve_path(args.model_path)) if args.model_path else None

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    smiles, smiles_column = load_smiles_csv(input_path, args.smiles_column)
    canonical_smiles, valid_mask = canonicalize_smiles(smiles)

    print(f"Using input:  {input_path}")
    print(f"Using output: {output_path}")
    print(f"SMILES column: {smiles_column}")
    print(f"Loaded {len(smiles)} SMILES")

    predictor = Permeability(
        model_path=model_path,
        batch_size=args.batch_size,
        input_type="smiles",
    )
    scores = predictor(smiles)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["input_smiles", "canonical_smiles", "permeability", "status"])
        for smi, cano, is_valid, score in zip(smiles, canonical_smiles, valid_mask, scores):
            status = "ok" if is_valid else "invalid_smiles"
            writer.writerow([smi, cano, format_score(float(score)), status])

    valid_scores = np.asarray([score for score, is_valid in zip(scores, valid_mask) if is_valid])
    print(f"Saved permeability scores to {output_path}")
    print(f"  Total sequences: {len(smiles)}")
    print(f"  Valid SMILES: {int(sum(valid_mask))}")
    print(f"  Invalid SMILES: {len(smiles) - int(sum(valid_mask))}")
    if len(valid_scores) > 0:
        print(f"  Mean permeability: {valid_scores.mean():.4f}")
        print(f"  Std permeability:  {valid_scores.std():.4f}")
        print(f"  Min permeability:  {valid_scores.min():.4f}")
        print(f"  Max permeability:  {valid_scores.max():.4f}")
        print(
            "  High permeability (> -6): "
            f"{(valid_scores > -6).sum()} ({(valid_scores > -6).mean() * 100:.1f}%)"
        )


if __name__ == "__main__":
    main()
