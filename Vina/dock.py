"""
HELM -> docking score backend bridge used by DPO training.

Supports:
  - AutoDock Vina Python API (CPU / optional custom CUDA build)
  - Uni-Dock CLI (GPU, Linux + NVIDIA)
"""

import os
import time
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import List, Sequence

import numpy as np
from rdkit import Chem

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.helm2smiles import get_cycpep_smi_from_helm
from Vina.unidock_backend import dock_helms_unidock
from Vina.vina_score import DEFAULT_RECEPTOR, DEFAULT_REF_SDF, INVALID_SCORE, vina_score


def _dock_single_vina(args):
    """Dock a single HELM with the Vina Python backend."""
    idx, helm, protein_pdbqt_path, ref_sdf_path, device, cpu_per_worker, exhaustiveness, n_poses = args

    from rdkit import RDLogger  # noqa: E402
    RDLogger.DisableLog('rdApp.*')

    ref_supplier = Chem.SDMolSupplier(ref_sdf_path, removeHs=False)
    reference_mol = next(ref_supplier)
    if reference_mol is None:
        return idx, INVALID_SCORE

    smi = get_cycpep_smi_from_helm(helm)
    if smi is None:
        return idx, INVALID_SCORE

    score, _, _ = vina_score(
        ligand_mol_smi=smi,
        protein_pdbqt_path=protein_pdbqt_path,
        reference_mol=reference_mol,
        device=device,
        cpu=cpu_per_worker,
        exhaustiveness=exhaustiveness,
        n_poses=n_poses,
    )
    return idx, score


def _dock_helms_vina(
    helm_list: List[str],
    protein_pdbqt_path: str,
    ref_sdf_path: str,
    device: str,
    cpu: int,
    cpu_per_worker: int,
    num_workers: int,
    exhaustiveness: int,
    n_poses: int,
    show_progress: bool,
) -> np.ndarray:
    total_cpu = cpu if cpu > 0 else (os.cpu_count() or 1)
    if num_workers <= 0:
        num_workers = max(1, total_cpu // cpu_per_worker)
    actual_cpu_per_worker = max(1, total_cpu // num_workers)

    print(
        f"  Vina parallel: {num_workers} workers x {actual_cpu_per_worker} cores/worker "
        f"(total {total_cpu} cores), exhaustiveness={exhaustiveness}, n_poses={n_poses}"
    )

    tasks = [
        (i, helm, protein_pdbqt_path, ref_sdf_path, device,
         actual_cpu_per_worker, exhaustiveness, n_poses)
        for i, helm in enumerate(helm_list)
    ]

    scores = np.full(len(helm_list), INVALID_SCORE, dtype=np.float64)
    valid_count = 0
    valid_sum = 0.0
    use_tqdm = bool(show_progress and tqdm is not None)

    if num_workers == 1:
        iterator = tasks
        if use_tqdm:
            iterator = tqdm(tasks, desc=f"Vina docking ({device})", unit="ligand")

        for task in iterator:
            idx, score = _dock_single_vina(task)
            scores[idx] = score
            if score != INVALID_SCORE:
                valid_count += 1
                valid_sum += score
            if use_tqdm and (idx + 1) % 20 == 0:
                avg = (valid_sum / valid_count) if valid_count > 0 else 0.0
                iterator.set_postfix(valid=f"{valid_count}/{idx + 1}", avg=f"{avg:.2f}")
    else:
        progress = None
        if use_tqdm:
            progress = tqdm(
                total=len(tasks),
                desc=f"Vina docking ({device}, {num_workers}w)",
                unit="ligand",
                smoothing=0.05,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}, {postfix}]",
            )

        print(f"  Warming up {num_workers} workers, first results in ~1-3 min...")
        t_start = time.time()
        first_result = True

        with Pool(processes=num_workers) as pool:
            for idx, score in pool.imap_unordered(_dock_single_vina, tasks):
                scores[idx] = score
                if score != INVALID_SCORE:
                    valid_count += 1
                    valid_sum += score

                if progress is not None:
                    progress.update(1)
                    if first_result:
                        warmup_sec = time.time() - t_start
                        progress.write(
                            f"  First result after {warmup_sec:.1f}s, ETA is now estimating..."
                        )
                        first_result = False
                    if progress.n % 20 == 0:
                        avg = (valid_sum / valid_count) if valid_count > 0 else 0.0
                        progress.set_postfix(valid=f"{valid_count}/{progress.n}", avg=f"{avg:.2f}")

        if progress is not None:
            progress.close()

    valid_mask = scores != INVALID_SCORE
    print(
        f"  Vina done: {valid_mask.sum()}/{len(helm_list)} valid, avg={scores[valid_mask].mean():.2f}"
        if valid_mask.any() else f"  Vina done: 0/{len(helm_list)} valid"
    )
    return scores


def dock_helms(
    helm_list: List[str],
    protein_pdbqt_path: str = None,
    ref_sdf_path: str = None,
    device: str = "cpu",
    cpu: int = 0,
    cpu_per_worker: int = 2,
    num_workers: int = 0,
    exhaustiveness: int = 8,
    n_poses: int = 2,
    show_progress: bool = True,
    backend: str = "vina",
    box_size: float | Sequence[float] = 30.0,
    seed: int = 42,
    unidock_binary: str = "unidock",
    unidock_batch_size: int = 64,
    unidock_search_mode: str = "fast",
    unidock_scoring: str = "vina",
    unidock_refine_step: int = 3,
    unidock_max_step: int = 20,
    unidock_max_gpu_memory: int = 0,
    unidock_keep_workdir: bool = False,
    unidock_verbosity: int = 0,
) -> np.ndarray:
    """
    Dock a HELM list with the selected backend.

    Args:
        backend:
            - "vina": current Python Vina backend
            - "unidock": GPU Uni-Dock CLI backend
    """
    if protein_pdbqt_path is None:
        protein_pdbqt_path = DEFAULT_RECEPTOR
    if ref_sdf_path is None:
        ref_sdf_path = DEFAULT_REF_SDF

    backend = str(backend).lower()
    if backend in {"vina", "vina_python"}:
        return _dock_helms_vina(
            helm_list=helm_list,
            protein_pdbqt_path=protein_pdbqt_path,
            ref_sdf_path=ref_sdf_path,
            device=device,
            cpu=cpu,
            cpu_per_worker=cpu_per_worker,
            num_workers=num_workers,
            exhaustiveness=exhaustiveness,
            n_poses=n_poses,
            show_progress=show_progress,
        )

    if backend in {"unidock", "unidock_gpu", "gpu"}:
        print(
            f"  Uni-Dock backend: binary={unidock_binary}, batch_size={unidock_batch_size}, "
            f"search_mode={unidock_search_mode or 'manual'}, num_modes={n_poses}"
        )
        return dock_helms_unidock(
            helm_list=helm_list,
            protein_pdbqt_path=protein_pdbqt_path,
            ref_sdf_path=ref_sdf_path,
            unidock_binary=unidock_binary,
            batch_size=unidock_batch_size,
            scoring=unidock_scoring,
            search_mode=unidock_search_mode,
            exhaustiveness=max(exhaustiveness, 1),
            max_step=max(unidock_max_step, 1),
            num_modes=max(n_poses, 1),
            refine_step=max(unidock_refine_step, 1),
            box_size=box_size,
            seed=seed,
            verbosity=unidock_verbosity,
            max_gpu_memory=unidock_max_gpu_memory,
            show_progress=show_progress,
            keep_workdir=unidock_keep_workdir,
        )

    raise ValueError(f"Unknown docking backend: {backend}")
