#!/usr/bin/env python3
"""Score the case2 groundtruth ligand with the permeability predictor.

Run this script inside the isolated `perm_env` conda environment.

Default input:
    data/docking_9bt3/9bt3_ligand.sdf

Default output:
    outputs/samples/case2/eval_groundtruth/case2_groundtruth.perm.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from pepar_diff.evaluation.permeability import Permeability


DEFAULT_SDF = PROJECT_ROOT / "data" / "docking_9bt3" / "9bt3_ligand.sdf"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "outputs"
    / "samples"
    / "case2"
    / "eval_groundtruth"
    / "case2_groundtruth.perm.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score the case2 groundtruth ligand with permeability"
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
        mol = Chem.RemoveHs(mol)
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

    current_env = os.environ.get("CONDA_DEFAULT_ENV")
    if current_env and current_env != "perm_env":
        print(f"Warning: expected conda env 'perm_env', current env is '{current_env}'")

    mol_index, smiles = load_first_valid_smiles(sdf_path)
    print(f"Using input SDF: {sdf_path}")
    print(f"Loaded molecule index: {mol_index}")
    print(f"Derived SMILES: {smiles}")

    scorer = Permeability(input_type="smiles")
    score = float(scorer([smiles])[0])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source_sdf", "mol_index", "smiles", "permeability", "status"])
        writer.writerow([str(sdf_path), mol_index, smiles, format_score(score), "ok"])

    print(f"Saved permeability score to {output_path}")
    print(f"  Permeability: {score:.4f}")
    print("  Status:       ok")


if __name__ == "__main__":
    main()
