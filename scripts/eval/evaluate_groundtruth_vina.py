"""
Score a reference/groundtruth cyclic peptide SDF with the same Uni-Dock
settings used by DPO training.

Typical usage:
    python scripts/eval/evaluate_groundtruth_vina.py
    python scripts/eval/evaluate_groundtruth_vina.py --config configs/training/dpo.json
    python scripts/eval/evaluate_groundtruth_vina.py --sdf data/docking/raw_cyclic_pep.sdf

This script is meant to establish the baseline Vina score that generated
samples should beat. It prints the baseline plus the target thresholds for
"better by 0.5" and "better by 1.0".
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from rdkit import Chem

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from pepar_diff.vina.constants import DEFAULT_RECEPTOR, DEFAULT_REF_SDF, INVALID_SCORE
from pepar_diff.vina.unidock_backend import (
    MAX_UNIDOCK_ATOMS,
    _build_unidock_cmd,
    _get_reference_center,
    _read_first_score_from_result_sdf,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score the groundtruth/reference cyclic peptide SDF with Uni-Dock"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training/dpo.json",
        help="Path to config file; reads docking settings from its dpo section.",
    )
    parser.add_argument(
        "--sdf",
        type=str,
        default=str(DEFAULT_REF_SDF),
        help="Ligand SDF to score. Defaults to data/docking/raw_cyclic_pep.sdf",
    )
    parser.add_argument(
        "--ref_sdf",
        type=str,
        default=None,
        help="Reference SDF used to define docking box center. Defaults to --sdf.",
    )
    parser.add_argument(
        "--receptor",
        type=str,
        default=str(DEFAULT_RECEPTOR),
        help="Receptor PDBQT path. Defaults to data/docking/6dn5_receptor.pdbqt",
    )
    parser.add_argument(
        "--keep_workdir",
        action="store_true",
        help="Keep the temporary Uni-Dock workdir for debugging.",
    )
    return parser.parse_args()


def resolve_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _load_first_mol(sdf_path: Path):
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    for mol in supplier:
        if mol is not None:
            return mol
    return None


def main():
    args = parse_args()

    with open(args.config, "r") as f:
        full_config = json.load(f)
    dpo_cfg = full_config.get("dpo", {})

    sdf_path = resolve_path(args.sdf)
    ref_sdf_path = resolve_path(args.ref_sdf) if args.ref_sdf else sdf_path
    receptor_path = resolve_path(args.receptor)

    if not sdf_path.exists():
        raise FileNotFoundError(f"Ligand SDF not found: {sdf_path}")
    if not ref_sdf_path.exists():
        raise FileNotFoundError(f"Reference SDF not found: {ref_sdf_path}")
    if not receptor_path.exists():
        raise FileNotFoundError(f"Receptor PDBQT not found: {receptor_path}")

    current_platform = platform.system()
    if current_platform.lower() != "linux":
        raise RuntimeError(
            "Uni-Dock groundtruth scoring requires Linux + NVIDIA GPU. "
            f"Current platform: {current_platform}"
        )

    unidock_binary = str(dpo_cfg.get("unidock_binary", "unidock"))
    binary_path = shutil.which(unidock_binary)
    if binary_path is None:
        raise RuntimeError(
            f"Uni-Dock binary '{unidock_binary}' was not found in PATH."
        )

    ligand = _load_first_mol(sdf_path)
    if ligand is None:
        raise RuntimeError(f"Failed to read a valid molecule from {sdf_path}")
    num_atoms = int(ligand.GetNumAtoms())
    if num_atoms > MAX_UNIDOCK_ATOMS:
        raise RuntimeError(
            f"Groundtruth ligand has {num_atoms} atoms, exceeding Uni-Dock limit "
            f"{MAX_UNIDOCK_ATOMS}."
        )

    vina_exhaustiveness = int(dpo_cfg.get("vina_exhaustiveness", 8))
    vina_n_poses = int(dpo_cfg.get("vina_n_poses", 2))
    dock_box_size = dpo_cfg.get("dock_box_size", 30.0)
    dock_seed = int(dpo_cfg.get("dock_seed", 42))
    unidock_search_mode = str(dpo_cfg.get("unidock_search_mode", "fast"))
    unidock_scoring = str(dpo_cfg.get("unidock_scoring", "vina"))
    unidock_refine_step = int(dpo_cfg.get("unidock_refine_step", 3))
    unidock_max_step = int(dpo_cfg.get("unidock_max_step", 20))
    unidock_max_gpu_memory = int(dpo_cfg.get("unidock_max_gpu_memory", 0))
    unidock_verbosity = int(dpo_cfg.get("unidock_verbosity", 0))

    if isinstance(dock_box_size, (int, float)):
        box_size = [float(dock_box_size)] * 3
    else:
        box_size = [float(v) for v in dock_box_size]
        if len(box_size) != 3:
            raise ValueError("dock_box_size must be a float or a length-3 sequence")

    center = _get_reference_center(str(ref_sdf_path))

    if args.keep_workdir:
        workdir = Path(tempfile.mkdtemp(prefix="unidock_gt_", dir="/tmp"))
        temp_ctx = None
    else:
        temp_ctx = tempfile.TemporaryDirectory(prefix="unidock_gt_", dir="/tmp")
        workdir = Path(temp_ctx.name)

    try:
        inputs_dir = workdir / "inputs"
        outputs_dir = workdir / "outputs"
        inputs_dir.mkdir(parents=True, exist_ok=True)
        outputs_dir.mkdir(parents=True, exist_ok=True)

        ligand_path = inputs_dir / "groundtruth_00000.sdf"
        writer = Chem.SDWriter(str(ligand_path))
        writer.write(ligand)
        writer.close()

        ligand_index_path = workdir / "ligand_index_0000.txt"
        with open(ligand_index_path, "w") as f:
            f.write(str(ligand_path))

        cmd = _build_unidock_cmd(
            binary=binary_path,
            receptor_path=str(receptor_path),
            ligand_index_path=ligand_index_path,
            output_dir=outputs_dir,
            center=center,
            box_size=box_size,
            scoring=unidock_scoring,
            search_mode=unidock_search_mode,
            exhaustiveness=vina_exhaustiveness,
            max_step=unidock_max_step,
            num_modes=vina_n_poses,
            refine_step=unidock_refine_step,
            seed=dock_seed,
            verbosity=unidock_verbosity,
            max_gpu_memory=unidock_max_gpu_memory,
        )

        print(f"Loading config from: {args.config}")
        print(f"Groundtruth ligand: {sdf_path}")
        print(f"Reference center from: {ref_sdf_path}")
        print(f"Receptor: {receptor_path}")
        print(
            f"Uni-Dock settings: binary={unidock_binary}, "
            f"search_mode={unidock_search_mode or 'manual'}, num_modes={vina_n_poses}"
        )
        if not unidock_search_mode:
            print(
                f"Manual search params: exhaustiveness={vina_exhaustiveness}, "
                f"max_step={unidock_max_step}"
            )

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            stderr_tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-10:]
            detail = "\n".join(stderr_tail)
            raise RuntimeError(
                f"Uni-Dock failed while scoring {sdf_path} (returncode={proc.returncode}).\n{detail}"
            )

        result_path = outputs_dir / f"{ligand_path.stem}_out.sdf"
        if not result_path.exists():
            raise RuntimeError(f"Uni-Dock completed but no output SDF was found: {result_path}")

        score = float(_read_first_score_from_result_sdf(result_path))
        if score == INVALID_SCORE:
            raise RuntimeError(
                f"Uni-Dock output exists but no valid ENERGY was parsed from {result_path}"
            )

        print(f"\nGroundtruth Vina score: {score:.4f}")
        print("Reminder: more negative is better.")
        print(f"To beat it by 0.5, target <= {score - 0.5:.4f}")
        print(f"To beat it by 1.0, target <= {score - 1.0:.4f}")

        if args.keep_workdir:
            print(f"Kept workdir at: {workdir}")
    finally:
        if temp_ctx is not None:
            temp_ctx.cleanup()


if __name__ == "__main__":
    main()
