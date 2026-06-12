"""
Pre-screen cyclic-peptide samples before Vina docking or DPO training.
Samples are treated as 7-mer head-tail cyclic peptides (1:R1-N:R2).

Score components (100 total):
  1. Monomer diversity (20): type count, Shannon entropy, head concentration
  2. Functional balance (20): aromatic, polar, N-methylated, D-amino residues
  3. Per-peptide pharmacology (25): NMe, HBD, polar/hydrophobic ratios
  4. Positional diversity (15): Shannon entropy at each position
  5. Pairwise distance (20): Jaccard distance over cyclic bigrams

Usage:
    python scripts/eval/eval_add/evaluate_sample_quality.py <samples.txt> [--compare <file2.txt> ...]
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
# Constants
# ============================================================
TERMINAL_MODS = {"ac", "am", "am_G", "HOCOCH2_Bal"}

STANDARD_AA = {
    "G", "A", "V", "L", "I", "P", "F", "W", "M",
    "S", "T", "C", "Y", "H", "K", "R", "D", "E", "N", "Q",
}

# --- Functional-group classes ---
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

# Proline-like residues can affect cyclic-peptide cis/trans backbone states.
PROLINE_LIKE_TOKENS = {
    "P", "dP", "Aze", "Hpr", "dHpr", "lalloHyp", "Hyp",
}


# ============================================================
# Monomer property helpers
# ============================================================
def is_nmethylated(token: str) -> bool:
    """Return whether a residue is N-methylated."""
    return (token.startswith("me") and len(token) > 2 and token[2:3].isupper()) \
        or token.startswith("Me_d") \
        or token == "Sar"


def is_d_amino(token: str) -> bool:
    """Return whether a residue has D-amino-acid notation."""
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
# Data loading
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
            # Drop terminal modifiers and keep real monomers.
            raw_tokens = m.group(1).split(".")
            tokens = [t for t in raw_tokens if t not in TERMINAL_MODS]
            if not tokens:
                continue
            samples.append({"raw": line, "tokens": tokens})
    return samples


# ============================================================
# Component 1: monomer diversity
# ============================================================
def eval_diversity(samples: List[dict]) -> dict:
    all_monomers = []
    for s in samples:
        all_monomers.extend(s["tokens"])

    mc = Counter(all_monomers)
    total = sum(mc.values())
    num_types = len(mc)

    # Shannon entropy.
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
# Component 2: global functional-group balance
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
# Component 3: per-peptide pharmacology
# ============================================================
def eval_per_peptide_pharmacology(samples: List[dict]) -> dict:
    """
    Compute per-peptide pharmacology-related features:
      - NMe count: N-methylated residues, including Sar
      - HBD estimate: backbone NH count ~= length - NMe - Pro
      - Hydrophobic ratio: hydrophobic residues / length
      - Polar ratio: polar residues / length
      - Pro count: proline-like residues
      - D-aa count: D-amino-acid residues
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

        # Cyclic-peptide backbone HBD ~= N - NMe - Pro.
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

    # Medicinal-chemistry targets for 7-mer head-tail cyclic peptides:
    #   NMe: 2-4, to reduce HBD
    #   HBD: <=4, stricter than the oral Veber rule
    #   Hydrophobic ratio: 0.3-0.6

    # Fraction with NMe >= 2.
    nme_ge2_ratio = sum(1 for x in nme_counts if x >= 2) / max(len(nme_counts), 1)
    # Fraction with HBD <= 4.
    hbd_le4_ratio = sum(1 for x in hbd_estimates if x <= 4) / max(len(hbd_estimates), 1)
    # Fraction with at least one aromatic residue.
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
# Component 4: positional diversity
# ============================================================
def eval_positional_diversity(samples: List[dict]) -> dict:
    """
    Compute monomer Shannon entropy at each position.
    Low position entropy means that some positions are nearly fixed.
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
    # Weakest position = lowest entropy.
    weakest_pos = position_entropies.index(min_ent) if position_entropies else -1

    return {
        "position_entropies": position_entropies,
        "position_num_types": position_num_types,
        "mean_positional_entropy": round(mean_ent, 3),
        "min_positional_entropy": round(min_ent, 3),
        "weakest_position": weakest_pos,
    }


# ============================================================
# Component 5: pairwise distance as chemical-space coverage
# ============================================================
def eval_pairwise_distance(samples: List[dict], max_pairs: int = 5000) -> dict:
    """
    Sample sequence pairs and compute Jaccard distance over cyclic bigrams.
    Higher distance indicates broader sample-set coverage.
    """
    # Convert each peptide into a cyclic bigram set.
    bigram_sets = []
    for s in samples:
        tokens = s["tokens"]
        bigrams = set()
        for i in range(len(tokens) - 1):
            bigrams.add((tokens[i], tokens[i + 1]))
        # Head-tail cyclization: last -> first.
        if len(tokens) >= 2:
            bigrams.add((tokens[-1], tokens[0]))
        bigram_sets.append(bigrams)

    # Sample pairwise Jaccard distances.
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
# Overall score
# ============================================================
def compute_overall_score(diversity: dict, functional: dict,
                          pharmacology: dict, positional: dict,
                          pairwise: dict) -> Tuple[float, dict]:
    """
    Overall score (0-100):
      1. Monomer diversity (20)
      2. Functional balance (20)
      3. Per-peptide pharmacology (25)
      4. Positional diversity (15)
      5. Pairwise distance (20)
    """
    scores = {}

    def range_score(val, low, high, max_pts):
        """Full score inside [low, high], linear decay outside."""
        if low <= val <= high:
            return max_pts
        if val < low:
            return max_pts * (val / low) if low > 0 else 0
        return max_pts * max(0, 1 - (val - high) / high)

    # 1. Monomer diversity (20).
    type_score = min(diversity["num_monomer_types"] / 200, 1.0) * 10
    entropy_score = diversity["normalized_entropy"] * 10
    scores["monomer_diversity"] = type_score + entropy_score

    # 2. Functional balance (20).
    aro_s = range_score(functional["aromatic_ratio"], 0.08, 0.25, 5)
    pol_s = range_score(functional["polar_charged_ratio"], 0.08, 0.25, 5)
    nme_s = range_score(functional["nmethylated_ratio"], 0.03, 0.20, 5)
    dam_s = min(functional["d_amino_ratio"] / 0.03, 1.0) * 5
    scores["functional_balance"] = aro_s + pol_s + nme_s + dam_s

    # 3. Per-peptide pharmacology (25).
    #    NMe>=2 fraction (10), HBD<=4 fraction (8), aromatic>=1 fraction (7).
    scores["pharmacology"] = (
        pharmacology["nme_ge2_ratio"] * 10
        + pharmacology["hbd_le4_ratio"] * 8
        + pharmacology["aro_ge1_ratio"] * 7
    )

    # 4. Positional diversity (15).
    ref_ent = 5.0  # Entropy for roughly 32 uniformly distributed monomer types.
    mean_pos = min(positional["mean_positional_entropy"] / ref_ent, 1.0) * 10
    min_pos = min(positional["min_positional_entropy"] / ref_ent, 1.0) * 5
    scores["positional_diversity"] = mean_pos + min_pos

    # 5. Pairwise distance (20); mean Jaccard >= 0.8 gets full credit.
    scores["pairwise_distance"] = min(pairwise["mean_jaccard"] / 0.8, 1.0) * 20

    total = sum(scores.values())
    return round(total, 1), {k: round(v, 1) for k, v in scores.items()}


# ============================================================
# Report printing
# ============================================================
def print_report(filepath: str, diversity: dict, functional: dict,
                 pharmacology: dict, positional: dict, pairwise: dict,
                 total_score: float, sub_scores: dict, num_samples: int):
    name = Path(filepath).name
    w = 64

    print(f"\n{'=' * w}")
    print(f" Cyclic Peptide Sample Quality: {name}  ({num_samples} samples)")
    print(f"{'=' * w}")

    grade = ("A+" if total_score >= 85 else "A" if total_score >= 75
             else "B" if total_score >= 60 else "C" if total_score >= 45 else "D")
    print(f"\n  Overall score: {total_score}/100  ({grade})")
    print(f"{'─' * w}")

    max_pts = {
        "monomer_diversity": 20, "functional_balance": 20,
        "pharmacology": 25, "positional_diversity": 15,
        "pairwise_distance": 20,
    }
    label = {
        "monomer_diversity": "Monomer diversity",
        "functional_balance": "Functional balance",
        "pharmacology": "Pharmacology",
        "positional_diversity": "Positional diversity",
        "pairwise_distance": "Pairwise distance",
    }
    for k in max_pts:
        v = sub_scores[k]
        mp = max_pts[k]
        bar_len = int(v / mp * 20) if mp > 0 else 0
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"  {label[k]:14s}  {bar} {v:5.1f}/{mp}")

    # --- Monomer diversity ---
    print(f"\n{'─' * w}")
    print(" [1] Monomer diversity")
    print(f"  Types:           {diversity['num_monomer_types']}")
    print(f"  Shannon entropy: {diversity['shannon_entropy']}  (normalized: {diversity['normalized_entropy']})")
    print(f"  Top-3 share:     {diversity['top3_concentration']:.1%}")
    print(f"  Top-10 share:    {diversity['top10_concentration']:.1%}")
    print(f"  Rare monomers:   {diversity['rare_monomer_count']} types (<=2 occurrences)")
    print(f"  Top-20:")
    for tok, cnt in diversity["top20_monomers"]:
        pct = cnt / diversity["total_monomer_count"] * 100
        print(f"    {tok:20s}: {cnt:4d} ({pct:.1f}%)")

    # --- Functional groups ---
    print(f"\n{'─' * w}")
    print(" [2] Functional balance")
    for name_key, ratio_key, ideal in [
        ("Aromatic",       "aromatic_ratio",       "8-25%"),
        ("Polar/charged",  "polar_charged_ratio",  "8-25%"),
        ("N-methylated",   "nmethylated_ratio",    "3-20%"),
        ("D-amino",        "d_amino_ratio",        ">=3%"),
        ("Proline-like",   "proline_like_ratio",   "reference"),
        ("Hydrophobic",    "hydrophobic_ratio",    "reference"),
    ]:
        v = functional[ratio_key]
        cnt_key = ratio_key.replace("_ratio", "_count")
        cnt = functional.get(cnt_key, "")
        ok = "✓" if ratio_key == "proline_like_ratio" or ratio_key == "hydrophobic_ratio" else ""
        if not ok:
            if ratio_key == "d_amino_ratio":
                ok = "✓" if v >= 0.03 else "⚠ low"
            elif ratio_key in ("aromatic_ratio", "polar_charged_ratio"):
                ok = "✓" if 0.08 <= v <= 0.25 else ("⚠ low" if v < 0.08 else "⚠ high")
            elif ratio_key == "nmethylated_ratio":
                ok = "✓" if 0.03 <= v <= 0.20 else ("⚠ low" if v < 0.03 else "⚠ high")
        print(f"  {name_key:14s}: {cnt:5} ({v:5.1%})  target: {ideal:9s}  {ok}")

    # --- Per-peptide pharmacology ---
    print(f"\n{'─' * w}")
    print(" [3] Per-peptide pharmacology")
    ph = pharmacology
    print(f"  NMe/peptide:      {ph['nme_per_peptide']['mean']:.1f} ± {ph['nme_per_peptide']['std']:.1f}  "
          f"dist: {ph['nme_per_peptide']['dist']}")
    print(f"    NMe >= 2 ratio: {ph['nme_ge2_ratio']:.1%}  "
          f"{'✓' if ph['nme_ge2_ratio'] >= 0.4 else '⚠ low permeability signal'}")
    print(f"  HBD/peptide:      {ph['hbd_per_peptide']['mean']:.1f} ± {ph['hbd_per_peptide']['std']:.1f}  "
          f"dist: {ph['hbd_per_peptide']['dist']}")
    print(f"    HBD <= 4 ratio: {ph['hbd_le4_ratio']:.1%}  "
          f"{'✓' if ph['hbd_le4_ratio'] >= 0.5 else '⚠ low permeability signal'}")
    print(f"  Pro/peptide:      {ph['proline_per_peptide']['mean']:.1f} ± {ph['proline_per_peptide']['std']:.1f}  "
          f"dist: {ph['proline_per_peptide']['dist']}")
    print(f"  D-aa/peptide:     {ph['d_amino_per_peptide']['mean']:.1f} ± {ph['d_amino_per_peptide']['std']:.1f}  "
          f"dist: {ph['d_amino_per_peptide']['dist']}")
    print(f"  Aromatic/peptide: {ph['aromatic_per_peptide']['mean']:.1f} ± {ph['aromatic_per_peptide']['std']:.1f}  "
          f"dist: {ph['aromatic_per_peptide']['dist']}")
    print(f"    Aromatic >= 1 ratio: {ph['aro_ge1_ratio']:.1%}  "
          f"{'✓' if ph['aro_ge1_ratio'] >= 0.5 else '⚠ limited binding signal'}")
    print(f"  Hydrophobic ratio/peptide: {ph['hydrophobic_ratio_per_peptide']['mean']:.1%} ± {ph['hydrophobic_ratio_per_peptide']['std']:.1%}")
    print(f"  Polar ratio/peptide:       {ph['polar_ratio_per_peptide']['mean']:.1%} ± {ph['polar_ratio_per_peptide']['std']:.1%}")

    # --- Positional diversity ---
    print(f"\n{'─' * w}")
    print(" [4] Positional diversity")
    pos = positional
    for i, (ent, nt) in enumerate(zip(pos["position_entropies"], pos["position_num_types"])):
        bar_len = int(ent / 6.0 * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        weak = " ← weakest" if i == pos["weakest_position"] else ""
        print(f"  Position {i+1}: {bar} {ent:.2f} ({nt:3d} types){weak}")
    print(f"  Mean: {pos['mean_positional_entropy']:.3f}   Min: {pos['min_positional_entropy']:.3f}")

    # --- Pairwise distance ---
    print(f"\n{'─' * w}")
    print(" [5] Pairwise Jaccard distance over cyclic bigrams")
    pw = pairwise
    print(f"  Mean distance: {pw['mean_jaccard']:.3f} ± {pw['std_jaccard']:.3f}")
    print(f"  Range:         [{pw['min_jaccard']:.3f}, {pw['max_jaccard']:.3f}]")
    print(f"  Sampled pairs: {pw['num_pairs_sampled']}")
    if pw["mean_jaccard"] >= 0.8:
        print("  ✓ High sample diversity and broad chemical-space coverage")
    elif pw["mean_jaccard"] >= 0.6:
        print("  ~ Moderate sample diversity")
    else:
        print("  ⚠ Low sample diversity and limited chemical-space coverage")

    print(f"\n{'=' * w}\n")


# ============================================================
# Evaluation entry point
# ============================================================
def evaluate_file(filepath: str, rgroups: Dict[str, Set[str]], verbose: bool = True) -> dict:
    samples = parse_helm_file(filepath)
    if not samples:
        print(f"⚠ Empty or unparsable file: {filepath}")
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
    parser = argparse.ArgumentParser(description="Cyclic-peptide sample quality pre-screening")
    parser.add_argument("input", type=str, help="HELM sample file to evaluate")
    parser.add_argument("--compare", type=str, nargs="+", default=[], help="Comparison files")
    args = parser.parse_args()

    rgroups = load_rgroup_table(MONOMER_LIBRARY)
    print(f"Loaded R-group info for {len(rgroups)} monomers")

    files = [args.input] + args.compare
    results = []
    for f in files:
        r = evaluate_file(f, rgroups)
        if r:
            results.append(r)

    if len(results) > 1:
        print("=" * 64)
        print(" Comparison Summary")
        print("=" * 64)
        print(f"  {'File':32s} {'Score':>5s} {'Types':>5s} {'Aro':>6s} {'NMe>=2':>7s} {'HBD<=4':>7s} {'Jaccard':>7s}")
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
