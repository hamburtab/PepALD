#!/usr/bin/env python3
"""Export cached high-quality DPO-generated cyclic peptide samples.

The script does not recompute Vina or permeability scores. It reads the cached
files under outputs/samples/case1/generated, filters head-to-tail cyclic HELM
sequences, and writes a compact report with permeability tiers.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from pepar_diff.postprocess import is_head_tail_single_cycle


DEFAULT_SOURCE_DIR = PROJECT_ROOT / "outputs" / "samples" / "case1" / "generated"
DEFAULT_SAMPLE_FILE = DEFAULT_SOURCE_DIR / "helm_dpo_samples.txt"
DEFAULT_VINA_FILE = DEFAULT_SOURCE_DIR / "helm_dpo_samples.case1.vina.csv"
DEFAULT_PERM_FILE = DEFAULT_SOURCE_DIR / "helm_dpo_samples.perm.csv"
DEFAULT_OUTPUT_FILE = Path(__file__).resolve().parent / "high_quality_samples.txt"

VINA_CUTOFF = -8.1270
PERMEABILITY_CUTOFFS = (-6.0, -5.5, -5.0, -4.5)


def resolve_path(path_str: str | Path) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def load_helm_samples(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_score_map(path: Path, score_column: str) -> dict[str, float]:
    scores: dict[str, float] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Score file has no header: {path}")

        field_map = {name.strip().lower(): name for name in reader.fieldnames if name}
        helm_key = field_map.get("helm")
        score_key = field_map.get(score_column.lower())
        if helm_key is None or score_key is None:
            raise ValueError(
                f"{path} must contain 'helm' and '{score_column}' columns; "
                f"got {reader.fieldnames}"
            )

        for row in reader:
            helm = (row.get(helm_key) or "").strip()
            if not helm:
                continue
            scores[helm] = float(row.get(score_key) or "nan")

    return scores


def filter_tier(
    helms: list[str],
    vina_scores: dict[str, float],
    permeability_scores: dict[str, float],
    permeability_cutoff: float,
) -> list[tuple[str, float, float]]:
    rows: list[tuple[str, float, float]] = []
    for helm in helms:
        if not is_head_tail_single_cycle(helm):
            continue

        vina_score = vina_scores.get(helm)
        permeability = permeability_scores.get(helm)
        if vina_score is None or permeability is None:
            continue

        if vina_score < VINA_CUTOFF and permeability > permeability_cutoff:
            rows.append((helm, vina_score, permeability))

    return rows


def write_report(
    output_file: Path,
    helms: list[str],
    tier_rows: dict[float, list[tuple[str, float, float]]],
    sample_file: Path,
    vina_file: Path,
    permeability_file: Path,
) -> None:
    denominator = len(helms)
    head_tail_count = sum(1 for helm in helms if is_head_tail_single_cycle(helm))

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        f.write("# PepAR-Diff high-quality sample export\n")
        f.write(f"# sample_file: {sample_file}\n")
        f.write(f"# vina_file: {vina_file}\n")
        f.write(f"# permeability_file: {permeability_file}\n")
        f.write("# criteria: head-to-tail single cycle, "
                f"vina_score < {VINA_CUTOFF:.4f}, "
                "permeability above tier cutoff\n")
        f.write(f"# denominator: {denominator} generated samples\n")
        f.write(f"# head_tail_single_cycle_samples: {head_tail_count}\n")
        f.write("# permeability is treated as higher-is-better\n")
        f.write("#\n")
        f.write("# Summary\n")
        f.write("# permeability_cutoff\tcount\tratio\tratio_percent\n")

        for cutoff in PERMEABILITY_CUTOFFS:
            count = len(tier_rows[cutoff])
            ratio = count / denominator if denominator else 0.0
            f.write(f"# > {cutoff:.1f}\t{count}\t{ratio:.8f}\t{ratio * 100:.4f}%\n")

        for cutoff in PERMEABILITY_CUTOFFS:
            f.write("\n")
            f.write(f"## permeability > {cutoff:.1f}\n")
            f.write("helm\tvina_score\tpermeability\n")
            for helm, vina_score, permeability in tier_rows[cutoff]:
                f.write(f"{helm}\t{vina_score:.8f}\t{permeability:.8f}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export high-quality cached DPO generated samples."
    )
    parser.add_argument("--sample_file", default=str(DEFAULT_SAMPLE_FILE))
    parser.add_argument("--vina_file", default=str(DEFAULT_VINA_FILE))
    parser.add_argument("--permeability_file", default=str(DEFAULT_PERM_FILE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_FILE))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_file = resolve_path(args.sample_file)
    vina_file = resolve_path(args.vina_file)
    permeability_file = resolve_path(args.permeability_file)
    output_file = resolve_path(args.output)

    for path in (sample_file, vina_file, permeability_file):
        if not path.exists():
            raise FileNotFoundError(path)

    helms = load_helm_samples(sample_file)
    vina_scores = load_score_map(vina_file, "vina_score")
    permeability_scores = load_score_map(permeability_file, "permeability")

    tier_rows = {
        cutoff: filter_tier(helms, vina_scores, permeability_scores, cutoff)
        for cutoff in PERMEABILITY_CUTOFFS
    }
    write_report(
        output_file,
        helms,
        tier_rows,
        sample_file,
        vina_file,
        permeability_file,
    )

    print(f"Source samples: {len(helms)}")
    print(f"Head-tail single cycles: {sum(is_head_tail_single_cycle(h) for h in helms)}")
    print(f"Vina cutoff: < {VINA_CUTOFF:.4f}")
    print("Permeability tiers:")
    for cutoff in PERMEABILITY_CUTOFFS:
        count = len(tier_rows[cutoff])
        ratio = count / len(helms) if helms else 0.0
        print(f"  > {cutoff:.1f}: {count} / {len(helms)} = {ratio:.4%}")
    print(f"Exported report: {output_file}")


if __name__ == "__main__":
    main()
