"""
环肽样本质量预评估工具 —— 在 Vina docking / DPO 训练之前快速判断生成样本的质量。
所有样本均视为 7-mer 头尾环化环肽 (1:R1-N:R2)。

评估维度 (5 大项, 满分 100):
  1. Monomer 多样性       (20分): 种类数 + Shannon 熵 + 头部集中度
  2. 功能团均衡性          (20分): 芳香/极性/N-甲基化/D-氨基酸
  3. 单肽药理学特征分布     (25分): 每条肽的 NMe 数、HBD、极性/疏水比
  4. 位置多样性            (15分): 每个位置的 Shannon 熵
  5. 序列间距离            (20分): Jaccard 距离衡量整体探索广度

用法:
    python scripts/eval/eval_add/evaluate_sample_quality.py <样本文件.txt> [--compare <文件2.txt> ...]
"""

import re
import csv
import math
import argparse
import itertools
import random
from pathlib import Path
from collections import Counter
from typing import Dict, List, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MONOMER_LIBRARY = PROJECT_ROOT / "data" / "processed" / "monomer_library.csv"

# ============================================================
# 常量
# ============================================================
TERMINAL_MODS = {"ac", "am", "am_G", "HOCOCH2_Bal"}

STANDARD_AA = {
    "G", "A", "V", "L", "I", "P", "F", "W", "M",
    "S", "T", "C", "Y", "H", "K", "R", "D", "E", "N", "Q",
}

# --- 功能团分类 ---
AROMATIC_TOKENS = {
    "F", "Y", "W", "H",
    "dF", "dY", "dW", "dH",
    "meF", "meY", "meW",
    "Me_dF", "Me_dW",
    "Cha", "Phe", "Trp", "Tyr",
    "dNal", "bHph", "Bn_Gly",
    "PhPr_Gly", "PhEt_Gly",
}

POLAR_CHARGED_TOKENS = {
    "R", "K", "H", "D", "E", "Q", "N", "S", "T",
    "dR", "dK", "dH", "dD", "dE", "dQ", "dN", "dS", "dT",
}

HYDROPHOBIC_TOKENS = {
    "L", "I", "V", "A", "M", "F", "W", "P",
    "dL", "dI", "dV", "dA",
    "meL", "meA", "meF", "meW", "meY",
    "Me_dA", "Me_dL", "Me_dF",
    "Nle", "Tle", "dTle",
}

# 脯氨酸族: 影响环肽骨架 cis/trans 构象
PROLINE_LIKE_TOKENS = {
    "P", "dP", "Aze", "Hpr", "dHpr", "lalloHyp", "Hyp",
}


# ============================================================
# 辅助函数: monomer 属性判断
# ============================================================
def is_nmethylated(token: str) -> bool:
    """N-甲基化残基: 减少骨架 NH, 降低 HBD, 有利透膜。"""
    return (token.startswith("me") and len(token) > 2 and token[2:3].isupper()) \
        or token.startswith("Me_d") \
        or token == "Sar"


def is_d_amino(token: str) -> bool:
    """D-构型残基: 影响骨架构象, 有利代谢稳定性。"""
    return (token.startswith("d") and len(token) > 1 and token[1:2].isupper()) \
        or token.startswith("Me_d")


def is_aromatic(token: str) -> bool:
    if token in AROMATIC_TOKENS:
        return True
    lower = token.lower()
    return any(sub in lower for sub in ["phe", "nal", "hph", "trp", "tyr", "bn_"])


def is_proline_like(token: str) -> bool:
    return token in PROLINE_LIKE_TOKENS


def is_polar(token: str) -> bool:
    return token in POLAR_CHARGED_TOKENS


def is_hydrophobic(token: str) -> bool:
    return token in HYDROPHOBIC_TOKENS


# ============================================================
# 数据加载
# ============================================================
def load_rgroup_table(csv_path: Path) -> Dict[str, Set[str]]:
    rgroups = {}
    if not csv_path.exists():
        return rgroups
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            symbol = row["Symbol"]
            available = set()
            for rg in ("R1", "R2", "R3"):
                val = row.get(rg, "-").strip()
                if val and val != "-" and val.lower() != "nan":
                    available.add(rg)
            rgroups[symbol] = available
    return rgroups


def parse_helm_file(filepath: str) -> List[dict]:
    samples = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = re.search(r"PEPTIDE1\{(.+?)\}", line)
            if not m:
                continue
            # 过滤末端修饰, 只保留真正的 monomer
            raw_tokens = m.group(1).split(".")
            tokens = [t for t in raw_tokens if t not in TERMINAL_MODS]
            if not tokens:
                continue
            samples.append({"raw": line, "tokens": tokens})
    return samples


# ============================================================
# 维度 1: Monomer 多样性
# ============================================================
def eval_diversity(samples: List[dict]) -> dict:
    all_monomers = []
    for s in samples:
        all_monomers.extend(s["tokens"])

    mc = Counter(all_monomers)
    total = sum(mc.values())
    num_types = len(mc)

    # Shannon 熵
    entropy = 0.0
    for count in mc.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    max_entropy = math.log2(num_types) if num_types > 1 else 1.0
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

    top3 = sum(c for _, c in mc.most_common(3))
    top10 = sum(c for _, c in mc.most_common(10))
    rare = [(t, c) for t, c in mc.items() if c <= 2]

    return {
        "num_monomer_types": num_types,
        "total_monomer_count": total,
        "shannon_entropy": round(entropy, 3),
        "normalized_entropy": round(normalized_entropy, 3),
        "top3_concentration": round(top3 / total, 3) if total > 0 else 0,
        "top10_concentration": round(top10 / total, 3) if total > 0 else 0,
        "rare_monomer_count": len(rare),
        "top20_monomers": mc.most_common(20),
    }


# ============================================================
# 维度 2: 功能团均衡性 (全局)
# ============================================================
def eval_functional_groups(samples: List[dict]) -> dict:
    all_monomers = []
    for s in samples:
        all_monomers.extend(s["tokens"])

    total = len(all_monomers)
    mc = Counter(all_monomers)

    aromatic_count = sum(c for t, c in mc.items() if is_aromatic(t))
    polar_count = sum(c for t, c in mc.items() if is_polar(t))
    hydrophobic_count = sum(c for t, c in mc.items() if is_hydrophobic(t))
    nmethyl_count = sum(c for t, c in mc.items() if is_nmethylated(t))
    d_amino_count = sum(c for t, c in mc.items() if is_d_amino(t))
    proline_count = sum(c for t, c in mc.items() if is_proline_like(t))

    std_count = sum(mc.get(t, 0) for t in STANDARD_AA)

    return {
        "aromatic_ratio": round(aromatic_count / total, 3) if total else 0,
        "aromatic_count": aromatic_count,
        "polar_charged_ratio": round(polar_count / total, 3) if total else 0,
        "polar_charged_count": polar_count,
        "hydrophobic_ratio": round(hydrophobic_count / total, 3) if total else 0,
        "hydrophobic_count": hydrophobic_count,
        "nmethylated_ratio": round(nmethyl_count / total, 3) if total else 0,
        "nmethylated_count": nmethyl_count,
        "d_amino_ratio": round(d_amino_count / total, 3) if total else 0,
        "d_amino_count": d_amino_count,
        "proline_like_ratio": round(proline_count / total, 3) if total else 0,
        "proline_like_count": proline_count,
        "standard_aa_ratio": round(std_count / total, 3) if total else 0,
        "non_standard_ratio": round((total - std_count) / total, 3) if total else 0,
    }


# ============================================================
# 维度 3: 单肽药理学特征分布
# ============================================================
def eval_per_peptide_pharmacology(samples: List[dict]) -> dict:
    """
    对每条环肽计算药理学相关特征, 分析其分布:
      - NMe 数:   N-甲基化残基个数 (含 Sar)
      - HBD 估计: 骨架 NH 数 ≈ 序列长度 - NMe数 - Pro数
                   (环肽无末端 NH3+/COO-, 每个非NMe/非Pro残基贡献 1 个骨架 NH)
      - 疏水比:   疏水残基 / 总长度
      - 极性比:   极性残基 / 总长度
      - Pro 数:   脯氨酸族个数 (影响 cis/trans 骨架构象)
      - D-aa 数:  D-氨基酸个数 (影响骨架手性)
    """
    nme_counts = []
    hbd_estimates = []
    hydrophobic_ratios = []
    polar_ratios = []
    proline_counts = []
    d_amino_counts = []
    aromatic_counts = []

    for s in samples:
        tokens = s["tokens"]
        n = len(tokens)
        if n == 0:
            continue

        n_nme = sum(1 for t in tokens if is_nmethylated(t))
        n_pro = sum(1 for t in tokens if is_proline_like(t))
        n_hydro = sum(1 for t in tokens if is_hydrophobic(t))
        n_polar = sum(1 for t in tokens if is_polar(t))
        n_d = sum(1 for t in tokens if is_d_amino(t))
        n_aro = sum(1 for t in tokens if is_aromatic(t))

        # 环肽骨架 HBD ≈ N - NMe - Pro
        # (每个残基有一个酰胺 NH, 但 N-甲基化和 Pro 没有)
        hbd = max(0, n - n_nme - n_pro)

        nme_counts.append(n_nme)
        hbd_estimates.append(hbd)
        hydrophobic_ratios.append(n_hydro / n)
        polar_ratios.append(n_polar / n)
        proline_counts.append(n_pro)
        d_amino_counts.append(n_d)
        aromatic_counts.append(n_aro)

    def dist_stats(values):
        if not values:
            return {"mean": 0, "std": 0, "min": 0, "max": 0, "dist": {}}
        mean = sum(values) / len(values)
        std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
        return {
            "mean": round(mean, 2),
            "std": round(std, 2),
            "min": min(values),
            "max": max(values),
            "dist": dict(sorted(Counter(values).items())),
        }

    def ratio_stats(values):
        if not values:
            return {"mean": 0, "std": 0}
        mean = sum(values) / len(values)
        std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
        return {"mean": round(mean, 3), "std": round(std, 3)}

    # 药物化学理想值 (7-mer 头尾环肽):
    #   NMe: 2-4 (Cyclosporine A 比例外推; 降低 HBD)
    #   HBD: ≤4  (Veber rule for oral: HBD ≤ 5; 环肽更严格)
    #   疏水: 0.3-0.6  (太高溶解性差, 太低透膜差)

    # NMe ≥ 2 的肽占比 (有利于透膜)
    nme_ge2_ratio = sum(1 for x in nme_counts if x >= 2) / max(len(nme_counts), 1)
    # HBD ≤ 4 的肽占比 (有利于透膜)
    hbd_le4_ratio = sum(1 for x in hbd_estimates if x <= 4) / max(len(hbd_estimates), 1)
    # 含至少 1 个芳香残基的肽占比 (有利于蛋白结合)
    aro_ge1_ratio = sum(1 for x in aromatic_counts if x >= 1) / max(len(aromatic_counts), 1)

    return {
        "nme_per_peptide": dist_stats(nme_counts),
        "hbd_per_peptide": dist_stats(hbd_estimates),
        "proline_per_peptide": dist_stats(proline_counts),
        "d_amino_per_peptide": dist_stats(d_amino_counts),
        "aromatic_per_peptide": dist_stats(aromatic_counts),
        "hydrophobic_ratio_per_peptide": ratio_stats(hydrophobic_ratios),
        "polar_ratio_per_peptide": ratio_stats(polar_ratios),
        "nme_ge2_ratio": round(nme_ge2_ratio, 3),
        "hbd_le4_ratio": round(hbd_le4_ratio, 3),
        "aro_ge1_ratio": round(aro_ge1_ratio, 3),
    }


# ============================================================
# 维度 4: 位置多样性
# ============================================================
def eval_positional_diversity(samples: List[dict]) -> dict:
    """
    每个位置的 monomer Shannon 熵。
    位置多样性高 → DPO 在各位置都有信号。
    位置多样性低 → 某些位置被锁死, DPO 学习空间受限。
    """
    seq_len = max((len(s["tokens"]) for s in samples), default=0)
    position_counters = [Counter() for _ in range(seq_len)]

    for s in samples:
        for i, tok in enumerate(s["tokens"]):
            if i < seq_len:
                position_counters[i][tok] += 1

    n = len(samples)
    position_entropies = []
    position_num_types = []
    for i, pc in enumerate(position_counters):
        ent = 0.0
        for count in pc.values():
            p = count / n
            if p > 0:
                ent -= p * math.log2(p)
        position_entropies.append(round(ent, 3))
        position_num_types.append(len(pc))

    mean_ent = sum(position_entropies) / max(len(position_entropies), 1)
    min_ent = min(position_entropies) if position_entropies else 0
    # 最薄弱位置 = entropy 最低的
    weakest_pos = position_entropies.index(min_ent) if position_entropies else -1

    return {
        "position_entropies": position_entropies,
        "position_num_types": position_num_types,
        "mean_positional_entropy": round(mean_ent, 3),
        "min_positional_entropy": round(min_ent, 3),
        "weakest_position": weakest_pos,
    }


# ============================================================
# 维度 5: 序列间距离 (化学空间覆盖)
# ============================================================
def eval_pairwise_distance(samples: List[dict], max_pairs: int = 5000) -> dict:
    """
    采样序列对, 计算 Jaccard 距离 (基于 bigram 集合)。
    衡量样本集覆盖的化学空间广度。
    Jaccard 距离越高 → 样本越分散 → DPO winner/loser 差异越大。
    """
    # 每条肽转为 bigram 集合 (捕获相邻残基交互)
    bigram_sets = []
    for s in samples:
        tokens = s["tokens"]
        bigrams = set()
        for i in range(len(tokens) - 1):
            bigrams.add((tokens[i], tokens[i + 1]))
        # 头尾环化: 最后一个 → 第一个
        if len(tokens) >= 2:
            bigrams.add((tokens[-1], tokens[0]))
        bigram_sets.append(bigrams)

    # 采样计算 pairwise Jaccard distance
    n = len(bigram_sets)
    if n < 2:
        return {"mean_jaccard": 0, "std_jaccard": 0, "min_jaccard": 0, "max_jaccard": 0}

    all_pairs = list(itertools.combinations(range(n), 2))
    if len(all_pairs) > max_pairs:
        random.seed(42)
        all_pairs = random.sample(all_pairs, max_pairs)

    distances = []
    for i, j in all_pairs:
        a, b = bigram_sets[i], bigram_sets[j]
        intersection = len(a & b)
        union = len(a | b)
        jaccard_dist = 1.0 - (intersection / union) if union > 0 else 0
        distances.append(jaccard_dist)

    mean_d = sum(distances) / len(distances)
    std_d = (sum((d - mean_d) ** 2 for d in distances) / len(distances)) ** 0.5

    return {
        "mean_jaccard": round(mean_d, 3),
        "std_jaccard": round(std_d, 3),
        "min_jaccard": round(min(distances), 3),
        "max_jaccard": round(max(distances), 3),
        "num_pairs_sampled": len(all_pairs),
    }


# ============================================================
# 综合评分
# ============================================================
def compute_overall_score(diversity: dict, functional: dict,
                          pharmacology: dict, positional: dict,
                          pairwise: dict) -> Tuple[float, dict]:
    """
    综合评分 (0-100):
      1. Monomer 多样性       (20分)
      2. 功能团均衡性          (20分)
      3. 单肽药理学特征        (25分)
      4. 位置多样性            (15分)
      5. 序列间距离            (20分)
    """
    scores = {}

    def range_score(val, low, high, max_pts):
        """值在 [low, high] 内满分, 超出线性衰减。"""
        if low <= val <= high:
            return max_pts
        if val < low:
            return max_pts * (val / low) if low > 0 else 0
        return max_pts * max(0, 1 - (val - high) / high)

    # 1. Monomer 多样性 (20分)
    type_score = min(diversity["num_monomer_types"] / 200, 1.0) * 10
    entropy_score = diversity["normalized_entropy"] * 10
    scores["monomer_diversity"] = type_score + entropy_score

    # 2. 功能团均衡性 (20分)
    aro_s = range_score(functional["aromatic_ratio"], 0.08, 0.25, 5)
    pol_s = range_score(functional["polar_charged_ratio"], 0.08, 0.25, 5)
    nme_s = range_score(functional["nmethylated_ratio"], 0.03, 0.20, 5)
    dam_s = min(functional["d_amino_ratio"] / 0.03, 1.0) * 5
    scores["functional_balance"] = aro_s + pol_s + nme_s + dam_s

    # 3. 单肽药理学特征 (25分)
    #    NMe≥2 占比 (10分) — 环肽透膜的关键
    #    HBD≤4 占比  (8分) — Veber rule
    #    含芳香≥1 占比 (7分) — 蛋白结合潜力
    scores["pharmacology"] = (
        pharmacology["nme_ge2_ratio"] * 10
        + pharmacology["hbd_le4_ratio"] * 8
        + pharmacology["aro_ge1_ratio"] * 7
    )

    # 4. 位置多样性 (15分)
    #    平均位置熵 (10分) — 以 log2(50)≈5.64 为参考上限
    #    最弱位置熵 (5分) — 木桶效应
    ref_ent = 5.0  # 约 32 种 monomer 均匀分布时的熵
    mean_pos = min(positional["mean_positional_entropy"] / ref_ent, 1.0) * 10
    min_pos = min(positional["min_positional_entropy"] / ref_ent, 1.0) * 5
    scores["positional_diversity"] = mean_pos + min_pos

    # 5. 序列间距离 (20分)
    #    mean Jaccard ≥ 0.8 满分 (完全不同的 bigram)
    scores["pairwise_distance"] = min(pairwise["mean_jaccard"] / 0.8, 1.0) * 20

    total = sum(scores.values())
    return round(total, 1), {k: round(v, 1) for k, v in scores.items()}


# ============================================================
# 打印报告
# ============================================================
def print_report(filepath: str, diversity: dict, functional: dict,
                 pharmacology: dict, positional: dict, pairwise: dict,
                 total_score: float, sub_scores: dict, num_samples: int):
    name = Path(filepath).name
    w = 64

    print(f"\n{'=' * w}")
    print(f" 环肽样本质量评估: {name}  ({num_samples} 条)")
    print(f"{'=' * w}")

    grade = ("A+" if total_score >= 85 else "A" if total_score >= 75
             else "B" if total_score >= 60 else "C" if total_score >= 45 else "D")
    print(f"\n  综合评分: {total_score}/100  ({grade})")
    print(f"{'─' * w}")

    max_pts = {
        "monomer_diversity": 20, "functional_balance": 20,
        "pharmacology": 25, "positional_diversity": 15,
        "pairwise_distance": 20,
    }
    label = {
        "monomer_diversity": "Monomer 多样性",
        "functional_balance": "功能团均衡性",
        "pharmacology": "单肽药理学特征",
        "positional_diversity": "位置多样性",
        "pairwise_distance": "序列间距离",
    }
    for k in max_pts:
        v = sub_scores[k]
        mp = max_pts[k]
        bar_len = int(v / mp * 20) if mp > 0 else 0
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"  {label[k]:14s}  {bar} {v:5.1f}/{mp}")

    # --- Monomer 多样性 ---
    print(f"\n{'─' * w}")
    print(f" [1] Monomer 多样性")
    print(f"  种类数:          {diversity['num_monomer_types']}")
    print(f"  Shannon 熵:      {diversity['shannon_entropy']}  (归一化: {diversity['normalized_entropy']})")
    print(f"  Top-3 集中度:    {diversity['top3_concentration']:.1%}")
    print(f"  Top-10 集中度:   {diversity['top10_concentration']:.1%}")
    print(f"  稀有 monomer:    {diversity['rare_monomer_count']} 种 (≤2次)")
    print(f"  Top-20:")
    for tok, cnt in diversity["top20_monomers"]:
        pct = cnt / diversity["total_monomer_count"] * 100
        print(f"    {tok:20s}: {cnt:4d} ({pct:.1f}%)")

    # --- 功能团 ---
    print(f"\n{'─' * w}")
    print(f" [2] 功能团均衡性")
    for name_key, ratio_key, ideal in [
        ("芳香族",     "aromatic_ratio",       "8-25%"),
        ("极性/带电",  "polar_charged_ratio",  "8-25%"),
        ("N-甲基化",   "nmethylated_ratio",    "3-20%"),
        ("D-氨基酸",   "d_amino_ratio",        "≥3%"),
        ("脯氨酸族",   "proline_like_ratio",   "参考"),
        ("疏水",       "hydrophobic_ratio",    "参考"),
    ]:
        v = functional[ratio_key]
        cnt_key = ratio_key.replace("_ratio", "_count")
        cnt = functional.get(cnt_key, "")
        ok = "✓" if ratio_key == "proline_like_ratio" or ratio_key == "hydrophobic_ratio" else ""
        if not ok:
            if ratio_key == "d_amino_ratio":
                ok = "✓" if v >= 0.03 else "⚠ 偏低"
            elif ratio_key in ("aromatic_ratio", "polar_charged_ratio"):
                ok = "✓" if 0.08 <= v <= 0.25 else ("⚠ 偏低" if v < 0.08 else "⚠ 偏高")
            elif ratio_key == "nmethylated_ratio":
                ok = "✓" if 0.03 <= v <= 0.20 else ("⚠ 偏低" if v < 0.03 else "⚠ 偏高")
        print(f"  {name_key:10s}: {cnt:5} ({v:5.1%})  理想: {ideal:6s}  {ok}")

    # --- 单肽药理学 ---
    print(f"\n{'─' * w}")
    print(f" [3] 单肽药理学特征分布 (每条肽)")
    ph = pharmacology
    print(f"  NMe/肽:  {ph['nme_per_peptide']['mean']:.1f} ± {ph['nme_per_peptide']['std']:.1f}  "
          f"分布: {ph['nme_per_peptide']['dist']}")
    print(f"    NMe ≥ 2 占比: {ph['nme_ge2_ratio']:.1%}  "
          f"{'✓' if ph['nme_ge2_ratio'] >= 0.4 else '⚠ 偏低 (透膜性受限)'}")
    print(f"  HBD/肽:  {ph['hbd_per_peptide']['mean']:.1f} ± {ph['hbd_per_peptide']['std']:.1f}  "
          f"分布: {ph['hbd_per_peptide']['dist']}")
    print(f"    HBD ≤ 4 占比:  {ph['hbd_le4_ratio']:.1%}  "
          f"{'✓' if ph['hbd_le4_ratio'] >= 0.5 else '⚠ 偏低 (透膜性受限)'}")
    print(f"  Pro/肽:  {ph['proline_per_peptide']['mean']:.1f} ± {ph['proline_per_peptide']['std']:.1f}  "
          f"分布: {ph['proline_per_peptide']['dist']}")
    print(f"  D-aa/肽: {ph['d_amino_per_peptide']['mean']:.1f} ± {ph['d_amino_per_peptide']['std']:.1f}  "
          f"分布: {ph['d_amino_per_peptide']['dist']}")
    print(f"  芳香/肽: {ph['aromatic_per_peptide']['mean']:.1f} ± {ph['aromatic_per_peptide']['std']:.1f}  "
          f"分布: {ph['aromatic_per_peptide']['dist']}")
    print(f"    含芳香 ≥ 1 占比: {ph['aro_ge1_ratio']:.1%}  "
          f"{'✓' if ph['aro_ge1_ratio'] >= 0.5 else '⚠ 偏低 (蛋白结合潜力有限)'}")
    print(f"  疏水比/肽: {ph['hydrophobic_ratio_per_peptide']['mean']:.1%} ± {ph['hydrophobic_ratio_per_peptide']['std']:.1%}")
    print(f"  极性比/肽: {ph['polar_ratio_per_peptide']['mean']:.1%} ± {ph['polar_ratio_per_peptide']['std']:.1%}")

    # --- 位置多样性 ---
    print(f"\n{'─' * w}")
    print(f" [4] 位置多样性 (每个位置的 Shannon 熵)")
    pos = positional
    for i, (ent, nt) in enumerate(zip(pos["position_entropies"], pos["position_num_types"])):
        bar_len = int(ent / 6.0 * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        weak = " ← 最薄弱" if i == pos["weakest_position"] else ""
        print(f"  位置 {i+1}: {bar} {ent:.2f} ({nt:3d} 种){weak}")
    print(f"  平均: {pos['mean_positional_entropy']:.3f}   最低: {pos['min_positional_entropy']:.3f}")

    # --- 序列间距离 ---
    print(f"\n{'─' * w}")
    print(f" [5] 序列间 Jaccard 距离 (基于环 bigram)")
    pw = pairwise
    print(f"  平均距离: {pw['mean_jaccard']:.3f} ± {pw['std_jaccard']:.3f}")
    print(f"  范围:     [{pw['min_jaccard']:.3f}, {pw['max_jaccard']:.3f}]")
    print(f"  采样对数: {pw['num_pairs_sampled']}")
    if pw["mean_jaccard"] >= 0.8:
        print(f"  ✓ 样本间差异大, 化学空间覆盖良好")
    elif pw["mean_jaccard"] >= 0.6:
        print(f"  ~ 样本间差异中等")
    else:
        print(f"  ⚠ 样本间差异小, 化学空间覆盖不足")

    print(f"\n{'=' * w}\n")


# ============================================================
# 评估入口
# ============================================================
def evaluate_file(filepath: str, rgroups: Dict[str, Set[str]], verbose: bool = True) -> dict:
    samples = parse_helm_file(filepath)
    if not samples:
        print(f"⚠ 文件为空或无法解析: {filepath}")
        return {}

    diversity = eval_diversity(samples)
    functional = eval_functional_groups(samples)
    pharmacology = eval_per_peptide_pharmacology(samples)
    positional = eval_positional_diversity(samples)
    pairwise = eval_pairwise_distance(samples)
    total_score, sub_scores = compute_overall_score(
        diversity, functional, pharmacology, positional, pairwise
    )

    if verbose:
        print_report(filepath, diversity, functional, pharmacology,
                     positional, pairwise, total_score, sub_scores, len(samples))

    return {
        "filepath": filepath,
        "num_samples": len(samples),
        "total_score": total_score,
        "sub_scores": sub_scores,
        "diversity": diversity,
        "functional": functional,
        "pharmacology": pharmacology,
        "positional": positional,
        "pairwise": pairwise,
    }


def main():
    parser = argparse.ArgumentParser(description="环肽样本质量预评估 (Vina + 透膜性)")
    parser.add_argument("input", type=str, help="待评估的 HELM 样本文件")
    parser.add_argument("--compare", type=str, nargs="+", default=[], help="对比文件")
    args = parser.parse_args()

    rgroups = load_rgroup_table(MONOMER_LIBRARY)
    print(f"已加载 {len(rgroups)} 个 monomer 的 R-group 信息")

    files = [args.input] + args.compare
    results = []
    for f in files:
        r = evaluate_file(f, rgroups)
        if r:
            results.append(r)

    if len(results) > 1:
        print("=" * 64)
        print(" 对比总结")
        print("=" * 64)
        print(f"  {'文件':32s} {'总分':>5s} {'种类':>5s} {'芳香':>6s} {'NMe≥2':>6s} {'HBD≤4':>6s} {'Jaccard':>7s}")
        print("  " + "─" * 62)
        for r in sorted(results, key=lambda x: x["total_score"], reverse=True):
            nm = Path(r["filepath"]).name[:32]
            print(f"  {nm:32s} {r['total_score']:5.1f} "
                  f"{r['diversity']['num_monomer_types']:5d} "
                  f"{r['functional']['aromatic_ratio']:6.1%} "
                  f"{r['pharmacology']['nme_ge2_ratio']:6.1%} "
                  f"{r['pharmacology']['hbd_le4_ratio']:6.1%} "
                  f"{r['pairwise']['mean_jaccard']:7.3f}")
        print()


if __name__ == "__main__":
    main()
