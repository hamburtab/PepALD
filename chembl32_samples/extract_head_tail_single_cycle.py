#!/usr/bin/env python3
"""Extract head-to-tail single cyclic peptides from a HELM sample file.

Filtering rules (all must be satisfied):
0) Sequence ends with "$$$" and not "$$$$".
1) Connection suffix must be exactly:
   "$PEPTIDE1,PEPTIDE1,1:R1-<length>:R2$$$"
   where <length> is the monomer count in PEPTIDE1{...}.
2) Sequence must not contain "|".
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


DEFAULT_INPUT = Path(__file__).parent / "helm_chembl32only_samples.txt"
DEFAULT_OUTPUT = Path(__file__).parent / "head_tail_single_cycle_samples.txt"


def get_peptide_length(helm_line: str) -> int | None:
    """Return monomer length from PEPTIDE1{...}, or None if parsing fails."""
    match = re.search(r"PEPTIDE1\{([^}]*)\}", helm_line)
    if not match:
        return None

    monomer_str = match.group(1).strip()
    if not monomer_str:
        return None

    monomers = [m for m in monomer_str.split(".") if m]
    return len(monomers) if monomers else None


def is_head_tail_single_cycle(helm_line: str) -> bool:
    """Check whether a HELM line is a single head-to-tail cyclic peptide."""
    seq = helm_line.strip()
    if not seq:
        return False

    # Rule (0)
    if not seq.endswith("$$$") or seq.endswith("$$$$"):
        return False

    # Rule (2)
    if "|" in seq:
        return False

    # Rule (1)
    length = get_peptide_length(seq)
    if length is None:
        return False

    expected_suffix = f"$PEPTIDE1,PEPTIDE1,1:R1-{length}:R2$$$"
    return seq.endswith(expected_suffix)


def extract_sequences(input_file: Path, output_file: Path) -> tuple[int, int]:
    """Extract valid sequences and write to output file.

    Returns:
        (total_count, kept_count)
    """
    total = 0
    kept = []

    with input_file.open("r") as f:
        for line in f:
            seq = line.strip()
            if not seq:
                continue
            total += 1
            if is_head_tail_single_cycle(seq):
                kept.append(seq)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w") as f:
        for seq in kept:
            f.write(seq + "\n")

    return total, len(kept)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract single head-to-tail cyclic peptides from HELM samples."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input HELM txt file (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output txt file (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    total, kept = extract_sequences(args.input, args.output)
    print(f"Input file:  {args.input}")
    print(f"Output file: {args.output}")
    print(f"Total lines: {total}")
    print(f"Kept lines:  {kept}")


if __name__ == "__main__":
    main()
