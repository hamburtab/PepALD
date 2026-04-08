"""Uni-Dock GPU backend for HELM docking."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from utils.helm2smiles import get_cycpep_smi_from_helm
from Vina.constants import DEFAULT_RECEPTOR, DEFAULT_REF_SDF, INVALID_SCORE

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


def _chunked(items: Sequence, chunk_size: int) -> Iterable[Sequence]:
    for start in range(0, len(items), chunk_size):
        yield items[start:start + chunk_size]


def _get_reference_center(ref_sdf_path: str) -> List[float]:
    supplier = Chem.SDMolSupplier(str(ref_sdf_path), removeHs=False)
    reference_mol = next(supplier)
    if reference_mol is None or reference_mol.GetNumConformers() == 0:
        raise RuntimeError(f"Failed to load reference ligand from {ref_sdf_path}")
    center = reference_mol.GetConformer().GetPositions().mean(axis=0)
    return center.tolist()


def _write_sdf_from_smiles(smiles: str, output_path: Path, name: str) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False

    mol = Chem.AddHs(mol)
    embed_status = AllChem.EmbedMolecule(mol, randomSeed=42)
    if embed_status == -1:
        embed_status = AllChem.EmbedMolecule(
            mol,
            useRandomCoords=True,
            randomSeed=42,
        )
        if embed_status == -1:
            return False

    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        pass

    mol.SetProp("_Name", name)
    writer = Chem.SDWriter(str(output_path))
    writer.write(mol)
    writer.close()
    return True


def _read_first_score_from_result_sdf(result_path: Path) -> float:
    score_line = ""
    with open(result_path, "r") as f:
        for line in f:
            if line.startswith("> <Uni-Dock RESULT>") or line.startswith(">  <Uni-Dock RESULT>"):
                score_line = next(f, "").strip()
                break

    if not score_line:
        return INVALID_SCORE

    try:
        return float([x for x in score_line[len("ENERGY="):].split(" ") if x][0])
    except Exception:
        return INVALID_SCORE


def _build_unidock_cmd(
    binary: str,
    receptor_path: str,
    ligand_index_path: Path,
    output_dir: Path,
    center: Sequence[float],
    box_size: Sequence[float],
    scoring: str,
    search_mode: str,
    exhaustiveness: int,
    max_step: int,
    num_modes: int,
    refine_step: int,
    seed: int,
    verbosity: int,
    max_gpu_memory: int,
) -> List[str]:
    cmd = [
        binary,
        "--receptor", str(receptor_path),
        "--ligand_index", str(ligand_index_path),
        "--dir", str(output_dir),
        "--center_x", str(center[0]),
        "--center_y", str(center[1]),
        "--center_z", str(center[2]),
        "--size_x", str(box_size[0]),
        "--size_y", str(box_size[1]),
        "--size_z", str(box_size[2]),
        "--scoring", scoring,
        "--num_modes", str(num_modes),
        "--refine_step", str(refine_step),
        "--seed", str(seed),
        "--verbosity", str(verbosity),
    ]

    if search_mode:
        cmd += ["--search_mode", search_mode]
    else:
        cmd += [
            "--exhaustiveness", str(exhaustiveness),
            "--max_step", str(max_step),
        ]

    if max_gpu_memory > 0:
        cmd += ["--max_gpu_memory", str(max_gpu_memory)]

    return cmd


def dock_helms_unidock(
    helm_list: List[str],
    protein_pdbqt_path: str | None = None,
    ref_sdf_path: str | None = None,
    unidock_binary: str = "unidock",
    batch_size: int = 64,
    scoring: str = "vina",
    search_mode: str = "fast",
    exhaustiveness: int = 128,
    max_step: int = 20,
    num_modes: int = 1,
    refine_step: int = 3,
    box_size: float | Sequence[float] = 30.0,
    seed: int = 42,
    verbosity: int = 0,
    max_gpu_memory: int = 0,
    show_progress: bool = True,
    keep_workdir: bool = False,
) -> np.ndarray:
    """
    Dock HELM ligands with Uni-Dock GPU backend.

    Uni-Dock officially supports Linux + NVIDIA GPU. This function prepares one
    SDF per HELM ligand, batches them through `unidock --ligand_index`, and
    returns the first pose energy per ligand.
    """
    if protein_pdbqt_path is None:
        protein_pdbqt_path = DEFAULT_RECEPTOR
    if ref_sdf_path is None:
        ref_sdf_path = DEFAULT_REF_SDF

    current_platform = platform.system()
    if current_platform.lower() != "linux":
        raise RuntimeError(
            "Uni-Dock backend requires Linux + NVIDIA GPU according to the upstream README. "
            f"Current platform: {current_platform}"
        )

    binary_path = shutil.which(unidock_binary)
    if binary_path is None:
        raise RuntimeError(
            f"Uni-Dock binary '{unidock_binary}' was not found in PATH. "
            "Install it in the target conda env, e.g. `conda install -n molformer_env -c conda-forge unidock`."
        )

    if isinstance(box_size, (int, float)):
        box_size = [float(box_size)] * 3
    else:
        box_size = [float(v) for v in box_size]
        if len(box_size) != 3:
            raise ValueError("box_size must be a float or a length-3 sequence")

    center = _get_reference_center(str(ref_sdf_path))
    scores = np.full(len(helm_list), INVALID_SCORE, dtype=np.float64)

    temp_ctx = None
    if keep_workdir:
        workdir = Path(tempfile.mkdtemp(prefix="unidock_", dir="/tmp"))
    else:
        temp_ctx = tempfile.TemporaryDirectory(prefix="unidock_", dir="/tmp")
        workdir = Path(temp_ctx.name)

    try:
        inputs_dir = workdir / "inputs"
        outputs_dir = workdir / "outputs"
        inputs_dir.mkdir(parents=True, exist_ok=True)
        outputs_dir.mkdir(parents=True, exist_ok=True)

        valid_entries: List[Tuple[int, Path]] = []
        prep_iter = enumerate(helm_list)
        if show_progress and tqdm is not None:
            prep_iter = tqdm(prep_iter, total=len(helm_list), desc="Uni-Dock prep", unit="ligand")

        for idx, helm in prep_iter:
            smiles = get_cycpep_smi_from_helm(helm)
            if not smiles:
                continue

            ligand_path = inputs_dir / f"ligand_{idx:05d}.sdf"
            if _write_sdf_from_smiles(smiles, ligand_path, ligand_path.stem):
                valid_entries.append((idx, ligand_path))

        if len(valid_entries) == 0:
            return scores

        batch_iter = list(_chunked(valid_entries, max(1, int(batch_size))))
        if show_progress and tqdm is not None:
            batch_iter = tqdm(batch_iter, desc="Uni-Dock batches", unit="batch")

        for batch_id, batch in enumerate(batch_iter):
            batch_out_dir = outputs_dir / f"batch_{batch_id:04d}"
            batch_out_dir.mkdir(parents=True, exist_ok=True)

            ligand_index_path = workdir / f"ligand_index_{batch_id:04d}.txt"
            with open(ligand_index_path, "w") as f:
                f.write("\n".join(str(path) for _, path in batch))

            cmd = _build_unidock_cmd(
                binary=binary_path,
                receptor_path=str(protein_pdbqt_path),
                ligand_index_path=ligand_index_path,
                output_dir=batch_out_dir,
                center=center,
                box_size=box_size,
                scoring=scoring,
                search_mode=search_mode,
                exhaustiveness=exhaustiveness,
                max_step=max_step,
                num_modes=num_modes,
                refine_step=refine_step,
                seed=seed,
                verbosity=verbosity,
                max_gpu_memory=max_gpu_memory,
            )

            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    "Uni-Dock execution failed.\n"
                    f"Command: {' '.join(cmd)}\n"
                    f"stdout:\n{proc.stdout}\n"
                    f"stderr:\n{proc.stderr}"
                )

            for idx, ligand_path in batch:
                result_path = batch_out_dir / f"{ligand_path.stem}_out.sdf"
                if result_path.exists():
                    scores[idx] = _read_first_score_from_result_sdf(result_path)

    finally:
        if temp_ctx is not None:
            temp_ctx.cleanup()

    valid_mask = scores != INVALID_SCORE
    print(
        f"  Uni-Dock done: {valid_mask.sum()}/{len(helm_list)} valid, avg={scores[valid_mask].mean():.2f}"
        if valid_mask.any()
        else f"  Uni-Dock done: 0/{len(helm_list)} valid"
    )
    if keep_workdir:
        print(f"  Uni-Dock workdir kept at: {workdir}")
    return scores
