import csv
import re
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
SAMPLE_FILE = PROJECT_ROOT / "outputs" / "samples" / "pretrain" / "helm_chembl32only_samples.txt"
OUTPUT_FILE = PROJECT_ROOT / "outputs" / "samples" / "case1" / "train_candidates" / "helm_chembl32only_r1r2_cyclized.txt"


def load_rgroup_table(csv_path: Path) -> dict[str, set[str]]:
    """Load available R-groups for each monomer."""
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
    """Extract monomer tokens from PEPTIDE1{...}."""
    match = re.search(r"PEPTIDE1\{([^}]*)\}", helm_line)
    if not match:
        return []
    return [token.strip() for token in match.group(1).split(".") if token.strip()]


def normalize_token(token: str) -> str:
    """Convert bracketed HELM tokens to monomer-library symbols."""
    if token.startswith("[") and token.endswith("]"):
        return token[1:-1].strip()
    return token


def has_r1_r2(token: str, rgroups: dict[str, set[str]]) -> bool:
    """Return whether a token has both R1 and R2."""
    available = rgroups.get(normalize_token(token))
    return available is not None and {"R1", "R2"}.issubset(available)


def cyclize_if_valid(helm_line: str, rgroups: dict[str, set[str]]) -> str | None:
    """Force a 1:R1-N:R2 head-tail cycle when both ends support R1/R2."""
    tokens = extract_tokens(helm_line)
    if not tokens:
        return None

    if not has_r1_r2(tokens[0], rgroups):
        return None
    if not has_r1_r2(tokens[-1], rgroups):
        return None

    peptide = f"PEPTIDE1{{{'.'.join(tokens)}}}"
    return f"{peptide}$PEPTIDE1,PEPTIDE1,1:R1-{len(tokens)}:R2$$$"


def cyclize_file(input_file: Path, output_file: Path, monomer_library: Path) -> tuple[int, int]:
    rgroups = load_rgroup_table(monomer_library)

    with input_file.open("r") as f:
        lines = [line.strip() for line in f if line.strip()]

    cyclized = []
    for line in lines:
        converted = cyclize_if_valid(line, rgroups)
        if converted is not None:
            cyclized.append(converted)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w") as f:
        for line in cyclized:
            f.write(line + "\n")

    return len(lines), len(cyclized)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Force valid HELM peptides into 1:R1-N:R2 head-tail cycles."
    )
    parser.add_argument("--input", type=Path, default=SAMPLE_FILE, help="Input HELM txt file")
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE, help="Output HELM txt file")
    parser.add_argument(
        "--monomer_library",
        type=Path,
        default=DATA_DIR / "monomer_library.csv",
        help="Monomer library CSV with R-group availability",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    total, kept = cyclize_file(args.input, args.output, args.monomer_library)
    print(f"Total samples: {total}")
    print(f"Kept and cyclized: {kept}")
    print(f"Output file: {args.output}")


if __name__ == "__main__":
    main()
