"""
Evaluate generation metrics for a CSV file with a SMILES column.

This script mirrors the core metrics used by scripts/eval/evaluate_full_metrics.py
but accepts direct SMILES input. The reference dataset is optional; when omitted,
reference-dependent metrics (SNN and novelty) are reported as NaN.

Usage:
    python scripts/eval_reference_model/evaluate_smiles_full_metrics.py \
        --input samples.csv \
        --output samples.full_metrics.csv

Input CSV example:
    smiles
    NCN1c2ccc(C(=O)N22CCC(CC(=O)O)cc1)C(NC(=O)[C@@H](CSCCN=C2ccccN)NC(=O)N2(N2CC3OCCCNC(3N)C(=O)N12)NC(=O)N[O-]
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from pepar_diff.utils.metrics import batch_tanimoto, fingerprint, load_reference_smiles
from pepar_diff.utils.sascore import sascorer

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
        description="Evaluate full metrics for a CSV file containing SMILES."
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
            "Output summary CSV. Defaults to '<input>.full_metrics.csv' next to "
            "the input file."
        ),
    )
    parser.add_argument(
        "--smiles_column",
        default=None,
        type=str,
        help="Name of the SMILES column. If omitted, common names are auto-detected.",
    )
    parser.add_argument(
        "--prior_path",
        default=None,
        type=str,
        help=(
            "Optional reference CSV for SNN/novelty. Supports the same reference "
            "schemas as pepar_diff.utils.metrics.load_reference_smiles. Omit this "
            "when the reference dataset should be left blank."
        ),
    )
    parser.add_argument(
        "--details_output",
        default=None,
        type=str,
        help=(
            "Optional per-row CSV with input SMILES, canonical SMILES, and validity "
            "status."
        ),
    )
    parser.add_argument(
        "--reference_cache",
        default=None,
        type=str,
        help=(
            "Optional .npz cache for reference canonical SMILES and fingerprints. "
            "Defaults to '<prior_path>.fingerprints.npz' when --prior_path is set."
        ),
    )
    return parser.parse_args()


def resolve_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def default_output_path(input_path: Path, suffix: str) -> Path:
    return input_path.with_name(f"{input_path.stem}{suffix}")


def find_column(df: pd.DataFrame, explicit: str | None, candidates: tuple[str, ...]) -> str:
    if explicit is not None:
        if explicit not in df.columns:
            raise ValueError(
                f"Column '{explicit}' not found in input CSV. "
                f"Available columns: {', '.join(map(str, df.columns))}"
            )
        return explicit

    for name in candidates:
        if name in df.columns:
            return name

    lower_to_original = {str(col).lower(): col for col in df.columns}
    for name in candidates:
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
    column = find_column(df, smiles_column, SMILES_COLUMN_CANDIDATES)
    smiles = df[column].dropna().astype(str).str.strip().tolist()
    smiles = [smi for smi in smiles if smi]
    if not smiles:
        raise ValueError(f"No non-empty SMILES found in column '{column}': {path}")
    return smiles, column


def canonicalize_smiles(smiles: list[str]) -> tuple[list[str], list[str], list[Chem.Mol | None]]:
    canonical = []
    statuses = []
    mols: list[Chem.Mol | None] = []

    for smi in smiles:
        mol = Chem.MolFromSmiles(smi) if smi else None
        mols.append(mol)
        if mol is None:
            canonical.append("")
            statuses.append("invalid_smiles")
        else:
            canonical.append(Chem.MolToSmiles(mol))
            statuses.append("ok")

    return canonical, statuses, mols


def default_reference_cache_path(prior_path: Path) -> Path:
    return prior_path.with_suffix(prior_path.suffix + ".fingerprints.npz")


def load_cached_reference(
    prior_path: Path, cache_path: Path
) -> tuple[set[str], np.ndarray, str, int] | None:
    if not cache_path.exists():
        return None

    source_stat = prior_path.stat()
    try:
        with np.load(cache_path, allow_pickle=False) as cache:
            if (
                int(cache["source_mtime_ns"]) != source_stat.st_mtime_ns
                or int(cache["source_size"]) != source_stat.st_size
            ):
                return None

            ref_smiles = set(cache["ref_smiles"].astype(str).tolist())
            ref_fps = cache["ref_fps"].astype(np.float32, copy=False)
            ref_column = str(cache["ref_column"])
            ref_size = int(cache["ref_size"])
            return ref_smiles, ref_fps, ref_column, ref_size
    except (KeyError, OSError, ValueError):
        return None


def save_reference_cache(
    prior_path: Path,
    cache_path: Path,
    ref_smiles: set[str],
    ref_fps: np.ndarray,
    ref_column: str,
    ref_size: int,
) -> None:
    source_stat = prior_path.stat()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        ref_smiles=np.array(sorted(ref_smiles), dtype=str),
        ref_fps=ref_fps,
        ref_column=np.array(ref_column),
        ref_size=np.array(ref_size),
        source_mtime_ns=np.array(source_stat.st_mtime_ns),
        source_size=np.array(source_stat.st_size),
    )


def load_reference(
    prior_path: Path | None, cache_path: Path | None = None
) -> tuple[set[str], np.ndarray, str, int]:
    if prior_path is None:
        return set(), np.zeros((0, 2048), dtype=np.float32), "", 0

    if not prior_path.exists():
        raise FileNotFoundError(f"Reference CSV not found: {prior_path}")

    if cache_path is not None:
        cached = load_cached_reference(prior_path, cache_path)
        if cached is not None:
            print(f"Loaded reference cache: {cache_path}")
            return cached

    ref_smiles, ref_column = load_reference_smiles(prior_path)
    ref_canonical, _, ref_mols = canonicalize_smiles(ref_smiles)
    valid_pairs = [
        (cano, mol)
        for cano, mol in zip(ref_canonical, ref_mols)
        if cano and mol is not None
    ]
    if not valid_pairs:
        return set(), np.zeros((0, 2048), dtype=np.float32), ref_column, 0

    ref_set = {cano for cano, _ in valid_pairs}
    ref_fps = np.vstack([fingerprint(mol) for _, mol in valid_pairs])
    if cache_path is not None:
        save_reference_cache(
            prior_path, cache_path, ref_set, ref_fps, ref_column, len(valid_pairs)
        )
        print(f"Saved reference cache: {cache_path}")
    return ref_set, ref_fps, ref_column, len(valid_pairs)


def nan_if_needed(value: float) -> float:
    return value if math.isfinite(value) else float("nan")


def format_metric(value: float) -> str:
    if value is None or not math.isfinite(float(value)):
        return "nan"
    return f"{float(value):.8f}"


def save_details(
    input_smiles: list[str],
    canonical_smiles: list[str],
    statuses: list[str],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["input_smiles", "canonical_smiles", "status"])
        for row in zip(input_smiles, canonical_smiles, statuses):
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    input_path = resolve_path(args.input)
    output_path = (
        resolve_path(args.output)
        if args.output
        else default_output_path(input_path, ".full_metrics.csv")
    )
    details_path = resolve_path(args.details_output) if args.details_output else None
    prior_path = resolve_path(args.prior_path) if args.prior_path else None
    reference_cache_path = (
        resolve_path(args.reference_cache)
        if args.reference_cache
        else default_reference_cache_path(prior_path)
        if prior_path is not None
        else None
    )

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    smiles, smiles_column = load_smiles_csv(input_path, args.smiles_column)
    canonical_smiles, statuses, mols = canonicalize_smiles(smiles)

    total = len(smiles)
    valid_indices = [idx for idx, mol in enumerate(mols) if mol is not None]
    valid_mols = [mols[idx] for idx in valid_indices]
    valid_smiles = [canonical_smiles[idx] for idx in valid_indices]
    validity = len(valid_mols) / total if total else 0.0

    unique_smiles = list(dict.fromkeys(valid_smiles))
    unique_mols = [Chem.MolFromSmiles(smi) for smi in unique_smiles]
    unique_mols = [mol for mol in unique_mols if mol is not None]
    uniqueness = len(unique_smiles) / len(valid_smiles) if valid_smiles else 0.0

    gen_fps = (
        np.vstack([fingerprint(mol) for mol in unique_mols])
        if unique_mols
        else np.zeros((0, 2048), dtype=np.float32)
    )
    diversity = (
        1.0 - float(batch_tanimoto(gen_fps, gen_fps, agg="mean"))
        if len(gen_fps) > 1
        else 0.0
    )

    ref_smiles, ref_fps, ref_column, ref_size = load_reference(
        prior_path, reference_cache_path
    )
    if len(gen_fps) > 0 and len(ref_fps) > 0:
        snn = float(batch_tanimoto(ref_fps, gen_fps, agg="max"))
        novelty = (
            sum(1 for smi in unique_smiles if smi not in ref_smiles) / len(unique_smiles)
            if unique_smiles
            else float("nan")
        )
    else:
        snn = float("nan")
        novelty = float("nan")

    sa_scores = [sascorer.calculateScore(mol) for mol in unique_mols]
    mean_sa = float(np.mean(sa_scores)) if sa_scores else float("nan")

    results = {
        "input": str(input_path),
        "smiles_column": smiles_column,
        "prior_path": "" if prior_path is None else str(prior_path),
        "reference_column": ref_column,
        "reference_size": ref_size,
        "total": total,
        "valid": len(valid_mols),
        "unique": len(unique_smiles),
        "validity": validity,
        "uniqueness": uniqueness,
        "diversity": diversity,
        "snn": nan_if_needed(snn),
        "novelty": nan_if_needed(novelty),
        "SA": nan_if_needed(mean_sa),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(results.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                key: format_metric(value) if isinstance(value, float) else value
                for key, value in results.items()
            }
        )

    if details_path is not None:
        save_details(smiles, canonical_smiles, statuses, details_path)

    print(f"Using input:  {input_path}")
    print(f"Using output: {output_path}")
    print(f"SMILES column: {smiles_column}")
    if prior_path is None:
        print("Reference: blank (SNN and novelty are reported as nan)")
    else:
        print(f"Reference: {prior_path} ({ref_size} valid molecules)")
    print("\n=== SMILES Full Metrics ===")
    for key in ("validity", "uniqueness", "diversity", "snn", "novelty", "SA"):
        print(f"  {key}: {format_metric(float(results[key]))}")
    print(f"  total: {total}")
    print(f"  valid: {len(valid_mols)}")
    print(f"  unique: {len(unique_smiles)}")
    if details_path is not None:
        print(f"Details saved to: {details_path}")


if __name__ == "__main__":
    main()
