"""
HELM → SMILES → Vina docking score 的衔接模块。
供 train_dpo.py 调用。
"""

import numpy as np
from typing import List
from rdkit import Chem

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.helm2smiles import get_cycpep_smi_from_helm
from Vina.vina_score import vina_score, INVALID_SCORE, DEFAULT_RECEPTOR, DEFAULT_REF_SDF


def dock_helms(
    helm_list: List[str],
    protein_pdbqt_path: str = None,
    ref_sdf_path: str = None,
    device: str = "cuda",
    cpu: int = 1,
    show_progress: bool = True,
) -> np.ndarray:
    """
    批量对 HELM 序列做 Vina docking。

    流程: HELM → SMILES (via get_cycpep_smi_from_helm) → vina_score

    Args:
        helm_list:           HELM 序列列表
        protein_pdbqt_path:  受体文件, 默认 6dn5_receptor.pdbqt
        ref_sdf_path:        参考配体, 默认 raw_cyclic_pep.sdf
        device:              打分设备标记, 默认 cuda
        cpu:                 Vina CPU 线程数
        show_progress:       是否显示进度条

    Returns:
        scores: np.ndarray [N], 每条 HELM 的 docking score (越负越好, 无效=0.0)
    """
    if protein_pdbqt_path is None:
        protein_pdbqt_path = DEFAULT_RECEPTOR
    if ref_sdf_path is None:
        ref_sdf_path = DEFAULT_REF_SDF

    # 加载参考分子（只加载一次）
    ref_supplier = Chem.SDMolSupplier(ref_sdf_path, removeHs=False)
    reference_mol = next(ref_supplier)
    if reference_mol is None:
        raise ValueError(f"无法从 {ref_sdf_path} 加载参考分子")

    scores = np.full(len(helm_list), INVALID_SCORE, dtype=np.float64)
    valid_count = 0
    valid_sum = 0.0

    use_tqdm = bool(show_progress and tqdm is not None)
    if show_progress and tqdm is None:
        print("Warning: tqdm is not available, fallback to periodic logging.")

    iterator = range(len(helm_list))
    progress = None
    if use_tqdm:
        progress = tqdm(iterator, desc=f"Vina docking ({device})", unit="ligand")
        iterator = progress

    def _log_timing(msg: str):
        if use_tqdm and progress is not None:
            progress.write(msg)
        else:
            print(msg)

    for i in iterator:
        helm = helm_list[i]
        # HELM → SMILES
        smi = get_cycpep_smi_from_helm(helm)
        if smi is None:
            _log_timing(
                f"[timing] ligand={i+1}/{len(helm_list)} mmff=0.000s "
                f"vina_dock=0.000s total=0.000s status=invalid_smiles"
            )
            if use_tqdm and progress is not None and (i + 1) % 20 == 0:
                avg = (valid_sum / valid_count) if valid_count > 0 else 0.0
                progress.set_postfix(valid=f"{valid_count}/{i+1}", avg=f"{avg:.2f}")
            continue

        # SMILES → Vina score
        score, mmff_sec, dock_sec = vina_score(
            ligand_mol_smi=smi,
            protein_pdbqt_path=protein_pdbqt_path,
            reference_mol=reference_mol,
            device=device,
            cpu=cpu,
        )
        scores[i] = score

        status = "valid" if score != INVALID_SCORE else "invalid"
        _log_timing(
            f"[timing] ligand={i+1}/{len(helm_list)} mmff={mmff_sec:.3f}s "
            f"vina_dock={dock_sec:.3f}s total={mmff_sec + dock_sec:.3f}s status={status}"
        )

        if score != INVALID_SCORE:
            valid_count += 1
            valid_sum += score

        if use_tqdm and progress is not None:
            if (i + 1) % 20 == 0:
                avg = (valid_sum / valid_count) if valid_count > 0 else 0.0
                progress.set_postfix(valid=f"{valid_count}/{i+1}", avg=f"{avg:.2f}")
        elif (i + 1) % 50 == 0:
            valid_rate = valid_count / (i + 1)
            avg = (valid_sum / valid_count) if valid_count > 0 else 0.0
            print(f"  Vina: {i+1}/{len(helm_list)}, "
                  f"valid={valid_rate:.1%}, avg={avg:.2f}")

    valid_mask = scores != INVALID_SCORE
    print(f"  Vina done: {valid_mask.sum()}/{len(helm_list)} valid, "
          f"avg={scores[valid_mask].mean():.2f}" if valid_mask.any() else
          f"  Vina done: 0/{len(helm_list)} valid")

    return scores
