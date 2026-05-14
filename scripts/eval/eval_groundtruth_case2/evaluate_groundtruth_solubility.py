#!/usr/bin/env python3
"""Score the case2 groundtruth ligand with PepTune solubility.

Default input:
    data/docking2/2axi_cyclicpep.sdf

Default output:
    outputs/samples/case2/eval_groundtruth/case2_groundtruth.solubility.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from pepar_diff.evaluation.solubility import PepTuneSolubility


DEFAULT_SDF = PROJECT_ROOT / "data" / "docking2" / "2axi_cyclicpep.sdf"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "outputs"
    / "samples"
    / "case2"
    / "eval_groundtruth"
    / "case2_groundtruth.solubility.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score the case2 groundtruth ligand with PepTune solubility"
    )
    parser.add_argument(
        "--sdf",
        type=str,
        default=str(DEFAULT_SDF),
        help="Groundtruth ligand SDF path.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--peptune_dir",
        default=None,
        type=str,
        help="Optional local PepTune repo or downloaded snapshot.",
    )
    parser.add_argument(
        "--model_path",
        default=None,
        type=str,
        help="Optional explicit PepTune solubility XGBoost model path.",
    )
    parser.add_argument(
        "--vocab_path",
        default=None,
        type=str,
        help="Optional explicit PepTune tokenizer new_vocab.txt path.",
    )
    parser.add_argument(
        "--splits_path",
        default=None,
        type=str,
        help="Optional explicit PepTune tokenizer new_splits.txt path.",
    )
    parser.add_argument(
        "--tokenizer_module_path",
        default=None,
        type=str,
        help="Optional explicit PepTune tokenizer/my_tokenizers.py path.",
    )
    parser.add_argument(
        "--batch_size",
        default=1,
        type=int,
        help="Embedding batch size. Default 1 matches PepTune's original scorer.",
    )
    parser.add_argument(
        "--device",
        default=None,
        type=str,
        help="Torch device, for example cuda, cpu, or mps.",
    )
    parser.add_argument(
        "--hf_endpoint",
        default=None,
        type=str,
        help="Optional Hugging Face endpoint, for example https://hf-mirror.com",
    )
    return parser.parse_args()


def resolve_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def load_first_valid_smiles(sdf_path: Path) -> tuple[int, str]:
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    for idx, mol in enumerate(supplier):
        if mol is None:
            continue
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        if smiles:
            return idx, smiles
    raise RuntimeError(f"Failed to read a valid molecule from {sdf_path}")


def format_score(score: float) -> str:
    return f"{float(score):.8f}" if np.isfinite(score) else "nan"


def main() -> None:
    args = parse_args()
    sdf_path = resolve_path(args.sdf)
    output_path = resolve_path(args.output)

    if not sdf_path.exists():
        raise FileNotFoundError(f"Input SDF not found: {sdf_path}")

    mol_index, smiles = load_first_valid_smiles(sdf_path)
    print(f"Using input SDF: {sdf_path}")
    print(f"Loaded molecule index: {mol_index}")
    print(f"Derived SMILES: {smiles}")

    scorer = PepTuneSolubility(
        peptune_dir=args.peptune_dir,
        model_path=args.model_path,
        vocab_path=args.vocab_path,
        splits_path=args.splits_path,
        tokenizer_module_path=args.tokenizer_module_path,
        batch_size=args.batch_size,
        input_type="smiles",
        device=args.device,
        hf_endpoint=args.hf_endpoint,
    )
    scores, smiles_list, statuses = scorer.predict_with_details([smiles])
    score = float(scores[0])
    status = statuses[0]
    smiles_out = smiles_list[0]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source_sdf", "mol_index", "smiles", "solubility", "status"])
        writer.writerow([str(sdf_path), mol_index, smiles_out, format_score(score), status])

    print(f"Saved solubility score to {output_path}")
    print(f"  Solubility: {score:.4f}")
    print(f"  Status:     {status}")


if __name__ == "__main__":
    main()
