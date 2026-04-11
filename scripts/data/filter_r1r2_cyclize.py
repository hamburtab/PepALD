"""
从 helm_chembl32only_samples.txt 中筛选出严格满足以下条件的序列：
1. 第一个 token 同时拥有 R1、R2
2. 最后一个 token 同时拥有 R1、R2

满足条件的线性肽会被强制转换成首尾成环的 HELM：
PEPTIDE1{...}$PEPTIDE1,PEPTIDE1,1:R1-N:R2$$$

其余样本全部丢弃。
"""

import csv
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
SAMPLE_FILE = PROJECT_ROOT / "outputs" / "samples" / "helm_chembl32only_samples.txt"
OUTPUT_FILE = PROJECT_ROOT / "outputs" / "samples" / "helm_chembl32only_r1r2_cyclized.txt"


def load_rgroup_table(csv_path: Path) -> dict[str, set[str]]:
    """加载 monomer_library.csv 中每个 monomer 的可用 R-group。"""
    rgroups: dict[str, set[str]] = {}
    with csv_path.open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            available = set()
            for rg in ("R1", "R2", "R3"):
                value = row.get(rg, "-").strip()
                if value and value != "-" and value.lower() != "nan":
                    available.add(rg)
            rgroups[row["Symbol"]] = available
    return rgroups


def extract_tokens(helm_line: str) -> list[str]:
    """提取 PEPTIDE1{...} 中的 monomer token。"""
    match = re.search(r"PEPTIDE1\{([^}]*)\}", helm_line)
    if not match:
        return []
    return [token.strip() for token in match.group(1).split(".") if token.strip()]


def normalize_token(token: str) -> str:
    """将 [meA] 这类 token 转成 monomer_library.csv 中的 Symbol。"""
    if token.startswith("[") and token.endswith("]"):
        return token[1:-1].strip()
    return token


def has_r1_r2(token: str, rgroups: dict[str, set[str]]) -> bool:
    """判断 token 是否同时具有 R1 和 R2。"""
    available = rgroups.get(normalize_token(token))
    return available is not None and {"R1", "R2"}.issubset(available)


def cyclize_if_valid(helm_line: str, rgroups: dict[str, set[str]]) -> str | None:
    """若首尾 token 都有 R1/R2，则强制转换为 1:R1-N:R2 成环 HELM。"""
    tokens = extract_tokens(helm_line)
    if not tokens:
        return None

    if not has_r1_r2(tokens[0], rgroups):
        return None
    if not has_r1_r2(tokens[-1], rgroups):
        return None

    peptide = f"PEPTIDE1{{{'.'.join(tokens)}}}"
    return f"{peptide}$PEPTIDE1,PEPTIDE1,1:R1-{len(tokens)}:R2$$$"


def main() -> None:
    rgroups = load_rgroup_table(DATA_DIR / "monomer_library.csv")

    with SAMPLE_FILE.open("r") as f:
        lines = [line.strip() for line in f if line.strip()]

    cyclized = []
    for line in lines:
        converted = cyclize_if_valid(line, rgroups)
        if converted is not None:
            cyclized.append(converted)

    with OUTPUT_FILE.open("w") as f:
        for line in cyclized:
            f.write(line + "\n")

    print(f"总样本数: {len(lines)}")
    print(f"保留并成环: {len(cyclized)}")
    print(f"输出文件: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
