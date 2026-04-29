"""
训练阶段
提取环肽数据：从 ChEMBL32 中筛选环肽 + 合并 CycPeptMPDB（全是环肽）
输出: data/processed/helm_sequences_cyclic.txt
用于训练环肽专用模型（finetune_cyclic.json）
"""

import re
from pathlib import Path

# 路径配置
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
CHEMBL_FILE = DATA_DIR / "helm_sequences_chembl32.txt"
CYCPEPT_FILE = DATA_DIR / "helm_sequences_cycpeptmpdb.txt"
OUTPUT_FILE = DATA_DIR / "helm_sequences_cyclic.txt"


def is_cyclic(helm_seq: str) -> bool:
    """判断 HELM 序列是否为环肽（有任何连接即为环肽）"""
    return bool(re.search(r'\d+:R\d+-\d+:R\d+', helm_seq))


def extract_cyclic_peptides():
    cyclic_seqs = set()  # 用 set 去重
    
    # 1. 从 ChEMBL32 提取环肽
    chembl_total, chembl_cyclic = 0, 0
    with open(CHEMBL_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chembl_total += 1
            if is_cyclic(line):
                cyclic_seqs.add(line)
                chembl_cyclic += 1
    
    # 2. 合并 CycPeptMPDB（全是环肽）
    cycpept_count = 0
    with open(CYCPEPT_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                cyclic_seqs.add(line)
                cycpept_count += 1
    
    # 3. 保存
    with open(OUTPUT_FILE, 'w') as f:
        for seq in sorted(cyclic_seqs):
            f.write(f"{seq}\n")
    
    # 统计
    print(f"ChEMBL32:     {chembl_cyclic}/{chembl_total} cyclic")
    print(f"CycPeptMPDB:  {cycpept_count} (all cyclic)")
    print(f"Total unique: {len(cyclic_seqs)}")
    print(f"Saved to:     {OUTPUT_FILE}")


if __name__ == "__main__":
    extract_cyclic_peptides()
