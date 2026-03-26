"""
HELM → SMILES → Vina docking score 的衔接模块。
供 train_dpo.py 调用。
"""

import numpy as np
from typing import List
from rdkit import Chem

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.helm2smiles import get_cycpep_smi_from_helm
from Vina.vina_score import vina_score, INVALID_SCORE, DEFAULT_RECEPTOR, DEFAULT_REF_SDF


def dock_helms(
    helm_list: List[str],
    protein_pdbqt_path: str = None,
    ref_sdf_path: str = None,
) -> np.ndarray:
    """
    批量对 HELM 序列做 Vina docking。

    流程: HELM → SMILES (via get_cycpep_smi_from_helm) → vina_score

    Args:
        helm_list:           HELM 序列列表
        protein_pdbqt_path:  受体文件, 默认 6dn5_receptor.pdbqt
        ref_sdf_path:        参考配体, 默认 raw_cyclic_pep.sdf

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

    for i, helm in enumerate(helm_list):
        # HELM → SMILES
        smi = get_cycpep_smi_from_helm(helm)
        if smi is None:
            continue

        # SMILES → Vina score
        scores[i] = vina_score(
            ligand_mol_smi=smi,
            protein_pdbqt_path=protein_pdbqt_path,
            reference_mol=reference_mol,
        )

        if (i + 1) % 50 == 0:
            valid_mask = scores[:i+1] != INVALID_SCORE
            valid_rate = valid_mask.sum() / (i + 1)
            avg = scores[:i+1][valid_mask].mean() if valid_mask.any() else 0
            print(f"  Vina: {i+1}/{len(helm_list)}, "
                  f"valid={valid_rate:.1%}, avg={avg:.2f}")

    valid_mask = scores != INVALID_SCORE
    print(f"  Vina done: {valid_mask.sum()}/{len(helm_list)} valid, "
          f"avg={scores[valid_mask].mean():.2f}" if valid_mask.any() else
          f"  Vina done: 0/{len(helm_list)} valid")

    return scores
