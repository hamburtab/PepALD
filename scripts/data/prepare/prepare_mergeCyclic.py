"""
Build the cyclic-peptide training corpus.

The script extracts cyclic HELM sequences from ChEMBL32 and merges them with
CycPeptMPDB sequences, writing data/processed/helm_sequences_cyclic.txt.
"""

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
CHEMBL_FILE = DATA_DIR / "helm_sequences_chembl32.txt"
CYCPEPT_FILE = DATA_DIR / "helm_sequences_cycpeptmpdb.txt"
OUTPUT_FILE = DATA_DIR / "helm_sequences_cyclic.txt"


def is_cyclic(helm_seq: str) -> bool:
    """Return whether a HELM sequence contains any explicit connection."""
    return bool(re.search(r'\d+:R\d+-\d+:R\d+', helm_seq))


def extract_cyclic_peptides():
    cyclic_seqs = set()
    
    # Extract cyclic entries from ChEMBL32.
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
    
    # Merge CycPeptMPDB entries, which are already cyclic.
    cycpept_count = 0
    with open(CYCPEPT_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                cyclic_seqs.add(line)
                cycpept_count += 1
    
    with open(OUTPUT_FILE, 'w') as f:
        for seq in sorted(cyclic_seqs):
            f.write(f"{seq}\n")

    print(f"ChEMBL32:     {chembl_cyclic}/{chembl_total} cyclic")
    print(f"CycPeptMPDB:  {cycpept_count} (all cyclic)")
    print(f"Total unique: {len(cyclic_seqs)}")
    print(f"Saved to:     {OUTPUT_FILE}")


if __name__ == "__main__":
    extract_cyclic_peptides()
