"""
从 helm_chembl32only_samples.txt 中筛选出
第一个 token 和最后一个 token 都同时拥有 R1、R2 的序列。
这些序列可以通过添加 1:R1-N:R2 头尾环化键变成环肽。

用法:
    python chembl32_samples/filter_head_tail_r1r2.py
"""

import re
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_FILE = Path(__file__).resolve().parent / "helm_chembl32only_samples.txt"
OUTPUT_FILE = Path(__file__).resolve().parent / "helm_chembl32only_r1r2_filtered.txt"


def load_rgroup_table(csv_path: str) -> dict:
    """从 monomer_library.csv 加载每个 monomer 的可用 R-group 集合。"""
    rgroups = {}
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


def extract_tokens(helm_line: str) -> list:
    """从 HELM 字符串中提取 monomer token 列表。"""
    m = re.search(r"PEPTIDE1\{(.+?)\}", helm_line)
    if not m:
        return []
    return m.group(1).split(".")


def main():
    rgroups = load_rgroup_table(DATA_DIR / "monomer_library.csv")
    print(f"已加载 {len(rgroups)} 个 monomer 的 R-group 信息")

    with open(SAMPLE_FILE) as f:
        lines = [l.strip() for l in f if l.strip()]
    print(f"总样本数: {len(lines)}")

    kept = []
    skipped_reasons = {"no_tokens": 0, "first_missing": 0, "last_missing": 0, "first_no_r1r2": 0, "last_no_r1r2": 0}

    for line in lines:
        tokens = extract_tokens(line)
        if not tokens:
            skipped_reasons["no_tokens"] += 1
            continue

        first_token = tokens[0]
        last_token = tokens[-1]

        # 跳过末端修饰（ac, am 等不是真正的 monomer）
        # 如果首/尾是修饰符，取下一个/前一个
        non_monomer = {"ac", "am", "am_G", "HOCOCH2_Bal"}
        if first_token in non_monomer:
            if len(tokens) < 2:
                skipped_reasons["no_tokens"] += 1
                continue
            first_token = tokens[1]
        if last_token in non_monomer:
            if len(tokens) < 2:
                skipped_reasons["no_tokens"] += 1
                continue
            last_token = tokens[-2]

        rg_first = rgroups.get(first_token)
        rg_last = rgroups.get(last_token)

        if rg_first is None:
            skipped_reasons["first_missing"] += 1
            continue
        if rg_last is None:
            skipped_reasons["last_missing"] += 1
            continue

        if "R1" not in rg_first or "R2" not in rg_first:
            skipped_reasons["first_no_r1r2"] += 1
            continue
        if "R1" not in rg_last or "R2" not in rg_last:
            skipped_reasons["last_no_r1r2"] += 1
            continue

        kept.append(line)

    # 保存
    with open(OUTPUT_FILE, "w") as f:
        for line in kept:
            f.write(line + "\n")

    print(f"\n筛选结果:")
    print(f"  保留: {len(kept)} 条 ({len(kept)/len(lines)*100:.1f}%)")
    print(f"  丢弃: {len(lines) - len(kept)} 条")
    print(f"\n丢弃原因:")
    for reason, count in skipped_reasons.items():
        if count > 0:
            print(f"  {reason}: {count}")
    print(f"\n已保存到: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
