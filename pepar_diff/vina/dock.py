"""
HELM -> GPU Vina scoring bridge used by DPO training.

This module intentionally keeps a single docking path:
    Uni-Dock (GPU, Linux + NVIDIA)

Environment/setup failures still raise directly. Individual ligand docking
failures are marked invalid and can be cached for later reuse.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from pepar_diff.vina.constants import DEFAULT_RECEPTOR, DEFAULT_REF_SDF
from pepar_diff.vina.unidock_backend import dock_helms_unidock


def dock_helms(
    helm_list: List[str],
    protein_pdbqt_path: str | None = None,
    ref_sdf_path: str | None = None,
    dock_center: Sequence[float] | None = None,
    exhaustiveness: int = 8,
    n_poses: int = 2,
    show_progress: bool = True,
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
    unidock_prep_workers: int = 1,
    score_log_path: str | None = None,
) -> np.ndarray:
    """Dock HELM ligands with the GPU Uni-Dock backend and return Vina scores."""
    if protein_pdbqt_path is None:
        protein_pdbqt_path = DEFAULT_RECEPTOR
    if ref_sdf_path is None:
        ref_sdf_path = DEFAULT_REF_SDF

    print(
        f"  Uni-Dock GPU scoring: binary={unidock_binary}, batch_size={unidock_batch_size}, "
        f"prep_workers={unidock_prep_workers}, search_mode={unidock_search_mode or 'manual'}, "
        f"num_modes={n_poses}"
    )

    return dock_helms_unidock(
        helm_list=helm_list,
        protein_pdbqt_path=protein_pdbqt_path,
        ref_sdf_path=ref_sdf_path,
        dock_center=dock_center,
        unidock_binary=unidock_binary,
        batch_size=unidock_batch_size,
        scoring=unidock_scoring,
        search_mode=unidock_search_mode,
        exhaustiveness=max(int(exhaustiveness), 1),
        max_step=max(int(unidock_max_step), 1),
        num_modes=max(int(n_poses), 1),
        refine_step=max(int(unidock_refine_step), 1),
        box_size=box_size,
        seed=int(seed),
        verbosity=int(unidock_verbosity),
        max_gpu_memory=int(unidock_max_gpu_memory),
        show_progress=bool(show_progress),
        keep_workdir=bool(unidock_keep_workdir),
        prep_workers=max(int(unidock_prep_workers), 1),
        score_log_path=score_log_path,
    )
