"""Extract cyclic samples from pretrained-generation outputs."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_ROOT / "outputs" / "samples" / "pretrain" / "helm_chembl32only_samples.txt"
OUTPUT_FILE = PROJECT_ROOT / "outputs" / "samples" / "pretrain" / "cyclic_samples.txt"

with open(INPUT_FILE) as f:
    cyclic = [l for l in f if l.strip().endswith("$$$") and not l.strip().endswith("$$$$")]

with open(OUTPUT_FILE, "w") as f:
    f.writelines(cyclic)

print(f"Extracted {len(cyclic)} cyclic peptides")
