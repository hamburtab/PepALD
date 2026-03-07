"""Extract cyclic peptides (sequences ending with $$$ instead of $$$$)."""

with open("chembl32_samples/helm_chembl32only_samples.txt") as f:
    cyclic = [l for l in f if l.strip().endswith("$$$") and not l.strip().endswith("$$$$")]

with open("chembl32_samples/cyclic_samples.txt", "w") as f:
    f.writelines(cyclic)

print(f"Extracted {len(cyclic)} cyclic peptides")
