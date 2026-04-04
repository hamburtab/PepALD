"""
HELM → SMILES → Vina docking score 的衔接模块。
供 train_dpo.py 调用。

支持多进程并行 docking：自动根据总 CPU 核数和每个 worker 分配的核数计算并行度。
"""

import os
import numpy as np
from typing import List
from rdkit import Chem
from multiprocessing import Pool, cpu_count
from functools import partial

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.helm2smiles import get_cycpep_smi_from_helm
from Vina.vina_score import vina_score, INVALID_SCORE, DEFAULT_RECEPTOR, DEFAULT_REF_SDF


def _dock_single(args):
    """
    Worker 函数：对单条 HELM 做 docking。
    用 tuple 传参以兼容 multiprocessing.Pool.imap_unordered。
    """
    idx, helm, protein_pdbqt_path, ref_sdf_path, device, cpu_per_worker, exhaustiveness, n_poses = args

    # 每个 worker 进程内加载参考分子
    # (Chem.Mol 不能 pickle 跨进程，所以每个 worker 自己加载)
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
) -> np.ndarray:
    """
    批量对 HELM 序列做 Vina docking（多进程并行）。

    Args:
        helm_list:           HELM 序列列表
        protein_pdbqt_path:  受体文件, 默认 6dn5_receptor.pdbqt
        ref_sdf_path:        参考配体, 默认 raw_cyclic_pep.sdf
        device:              打分设备标记
        cpu:                 总 CPU 核数 (0 = auto-detect)
        cpu_per_worker:      每个 docking worker 分配的核数
        num_workers:         并行 worker 数 (0 = auto: total_cpu // cpu_per_worker)
        exhaustiveness:      Vina exhaustiveness 参数
        n_poses:             Vina 生成 pose 数
        show_progress:       是否显示进度条

    Returns:
        scores: np.ndarray [N], 每条 HELM 的 docking score (越负越好, 无效=0.0)
    """
    if protein_pdbqt_path is None:
        protein_pdbqt_path = DEFAULT_RECEPTOR
    if ref_sdf_path is None:
        ref_sdf_path = DEFAULT_REF_SDF

    # 自动检测 CPU 核数并计算并行度
    total_cpu = cpu if cpu > 0 else (os.cpu_count() or 1)
    if num_workers <= 0:
        num_workers = max(1, total_cpu // cpu_per_worker)
    # 确保不超过总核数
    actual_cpu_per_worker = max(1, total_cpu // num_workers)

    print(f"  Vina parallel: {num_workers} workers × {actual_cpu_per_worker} cores/worker "
          f"(total {total_cpu} cores), exhaustiveness={exhaustiveness}, n_poses={n_poses}")

    # 构建任务列表
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
        # 单进程模式（调试友好）
        iterator = tasks
        if use_tqdm:
            iterator = tqdm(tasks, desc=f"Vina docking ({device})", unit="ligand")

        for task in iterator:
            idx, score = _dock_single(task)
            scores[idx] = score
            if score != INVALID_SCORE:
                valid_count += 1
                valid_sum += score
            if use_tqdm and (idx + 1) % 20 == 0:
                avg = (valid_sum / valid_count) if valid_count > 0 else 0.0
                iterator.set_postfix(valid=f"{valid_count}/{idx+1}", avg=f"{avg:.2f}")
    else:
        # 多进程模式
        progress = None
        if use_tqdm:
            progress = tqdm(total=len(tasks), desc=f"Vina docking ({device}, {num_workers}w)", unit="ligand")

        with Pool(processes=num_workers) as pool:
            for idx, score in pool.imap_unordered(_dock_single, tasks):
                scores[idx] = score
                if score != INVALID_SCORE:
                    valid_count += 1
                    valid_sum += score
                if progress is not None:
                    progress.update(1)
                    if progress.n % 20 == 0:
                        avg = (valid_sum / valid_count) if valid_count > 0 else 0.0
                        progress.set_postfix(valid=f"{valid_count}/{progress.n}", avg=f"{avg:.2f}")

        if progress is not None:
            progress.close()

    valid_mask = scores != INVALID_SCORE
    print(f"  Vina done: {valid_mask.sum()}/{len(helm_list)} valid, "
          f"avg={scores[valid_mask].mean():.2f}" if valid_mask.any() else
          f"  Vina done: 0/{len(helm_list)} valid")

    return scores
