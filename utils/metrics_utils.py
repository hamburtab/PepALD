"""
HELM生成样本评估指标：validity, uniqueness, diversity, snn, novelty
"""
import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem, DataStructs

rdBase.DisableLog('rdApp.error')

# 导入HELM转SMILES函数
try:
    from .helm2smiles import get_cycpep_smi_from_helm
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.helm2smiles import get_cycpep_smi_from_helm

# 导入SAscore计算模块
try:
    from .sascore import sascorer
except ImportError:
    from utils.sascore import sascorer


def fingerprint(mol, radius=3, size=2048):
    """计算分子的Morgan指纹"""
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=size)
    arr = np.zeros((size,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def batch_tanimoto(ref_fps, gen_fps, agg='max'):
    """
    批量计算Tanimoto相似度
    ref_fps: 参考集指纹 (N, D)
    gen_fps: 生成集指纹 (M, D)
    agg: 'max'返回每个生成分子与参考集最大相似度的平均, 'mean'返回整体平均
    """
    # Tanimoto = (A·B) / (|A| + |B| - A·B)
    dot = np.dot(gen_fps, ref_fps.T)  # (M, N)
    gen_sum = gen_fps.sum(axis=1, keepdims=True)  # (M, 1)
    ref_sum = ref_fps.sum(axis=1, keepdims=True).T  # (1, N)
    tanimoto = dot / (gen_sum + ref_sum - dot + 1e-10)
    
    if agg == 'max':
        return tanimoto.max(axis=1).mean()  # 每个生成分子的最大相似度，取平均
    else:
        return tanimoto.mean()


class Metrics:
    """HELM生成样本评估器"""
    
    def __init__(self, prior_path, n_jobs=1, input_type='helm'):
        """
        prior_path: 训练集CSV路径，需包含cano_smi列或HELM列
        """
        df = pd.read_csv(prior_path)
        if 'cano_smi' in df.columns:
            train_smiles = df['cano_smi'].dropna().tolist()
        else:
            train_smiles = [get_cycpep_smi_from_helm(h) for h in df.iloc[:, 0].dropna()]
        
        # 计算训练集指纹
        self.train_smiles = set()
        ref_fps = []
        for smi in train_smiles:
            mol = Chem.MolFromSmiles(smi) if smi else None
            if mol:
                self.train_smiles.add(Chem.MolToSmiles(mol))
                ref_fps.append(fingerprint(mol))
        self.ref_fps = np.vstack(ref_fps) if ref_fps else np.zeros((0, 2048))
        self.input_type = input_type
    
    def get_metrics(self, inputs):
        """
        计算五个核心指标
        inputs: HELM序列列表或SMILES列表
        """
        # 1. 转换为SMILES并解析分子
        if self.input_type == 'helm':
            smiles = [get_cycpep_smi_from_helm(h) for h in inputs]
        else:
            smiles = inputs
        
        mols = [Chem.MolFromSmiles(s) if s else None for s in smiles]
        valid_mols = [m for m in mols if m is not None]
        
        # === Validity: 有效分子比例 ===
        validity = len(valid_mols) / len(mols)
        
        # 获取有效分子的canonical SMILES
        valid_smiles = [Chem.MolToSmiles(m) for m in valid_mols]
        
        # === Uniqueness: 唯一分子比例 ===
        unique_smiles = list(set(valid_smiles))
        uniqueness = len(unique_smiles) / len(valid_smiles) if valid_smiles else 0
        
        # 计算唯一分子的指纹
        unique_mols = [Chem.MolFromSmiles(s) for s in unique_smiles]
        gen_fps = np.vstack([fingerprint(m) for m in unique_mols]) if unique_mols else np.zeros((0, 2048))
        
        # === Diversity: 生成集内部多样性 = 1 - 平均内部相似度 ===
        if len(gen_fps) > 1:
            diversity = 1 - batch_tanimoto(gen_fps, gen_fps, agg='mean')
        else:
            diversity = 0.0
        
        # === SNN: 与训练集的结构最近邻相似度 ===
        if len(gen_fps) > 0 and len(self.ref_fps) > 0:
            snn = batch_tanimoto(self.ref_fps, gen_fps, agg='max')
        else:
            snn = 0.0
        
        # === Novelty: 不在训练集中的比例 ===
        novel_count = sum(1 for s in unique_smiles if s not in self.train_smiles)
        novelty = novel_count / len(unique_smiles) if unique_smiles else 0
        
        # === SAscore: 合成可及性分数 (越低越好) ===
        sa_scores = [sascorer.calculateScore(m) for m in unique_mols]
        mean_sa = np.mean(sa_scores) if sa_scores else 0

        print(f"validity\tuniqueness\tdiversity\tsnn\tnovelty\tSA")
        print(f"{validity:.3f}\t{uniqueness:.3f}\t{diversity:.3f}\t{snn:.3f}\t{novelty:.3f}\t{mean_sa:.3f}")
        
        return {
            "validity": validity,
            "uniqueness": uniqueness, 
            "diversity": diversity,
            "snn": snn,
            "novelty": novelty,
            "SA": mean_sa
        }
