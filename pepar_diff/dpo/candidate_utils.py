"""Utilities for diversity-aware DPO candidate scoring and pairing."""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from pepar_diff.data.topology import HELMTopologyAnalyzer


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
    "meL", "meA", "meF", "meW", "meY", "meV",
    "Me_dA", "Me_dL", "Me_dF",
    "Nle", "Tle", "dTle",
}

PROLINE_LIKE_TOKENS = {
    "P", "dP", "Aze", "Hpr", "dHpr", "lalloHyp", "Hyp", "-pip",
}


def is_nmethylated(token: str) -> bool:
    return (
        (token.startswith("me") and len(token) > 2 and token[2:3].isupper())
        or token.startswith("Me_d")
        or token == "Sar"
    )


def is_d_amino(token: str) -> bool:
    return (
        (token.startswith("d") and len(token) > 1 and token[1:2].isupper())
        or token.startswith("Me_d")
    )


def is_aromatic(token: str) -> bool:
    if token in AROMATIC_TOKENS:
        return True
    lower = token.lower()
    return any(sub in lower for sub in ["phe", "nal", "hph", "trp", "tyr", "bn_"])


def is_polar(token: str) -> bool:
    return token in POLAR_CHARGED_TOKENS


def is_hydrophobic(token: str) -> bool:
    return token in HYDROPHOBIC_TOKENS


def is_proline_like(token: str) -> bool:
    return token in PROLINE_LIKE_TOKENS


def window_score(value: float, low: float, high: float) -> float:
    """Score in [0, 1] with a flat optimum window."""
    if high < low:
        low, high = high, low
    if low <= value <= high:
        return 1.0
    if value < low:
        return max(0.0, value / low) if low > 0 else 0.0
    span = max(high, 1e-6)
    return max(0.0, 1.0 - (value - high) / span)


def robust_normalize(values: np.ndarray) -> np.ndarray:
    """Median/MAD normalization, with std fallback."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr

    finite_mask = np.isfinite(arr)
    if not finite_mask.any():
        return np.zeros_like(arr)

    valid = arr[finite_mask]
    median = np.median(valid)
    mad = np.median(np.abs(valid - median))
    scale = 1.4826 * mad

    if scale < 1e-6:
        scale = float(valid.std())
    if scale < 1e-6:
        scale = 1.0

    normalized = (arr - median) / scale
    normalized[~finite_mask] = 0.0
    return normalized


_ANALYZER = HELMTopologyAnalyzer()


def extract_tokens_from_helm(helm: str) -> List[str]:
    return _ANALYZER.parse_helm_sequence(helm).get("monomers", [])


def build_ring_bigrams(tokens: Sequence[str]) -> Set[Tuple[str, str]]:
    if len(tokens) < 2:
        return set()

    bigrams = {(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)}
    bigrams.add((tokens[-1], tokens[0]))
    return bigrams


def jaccard_distance(a: Set[Tuple[str, str]], b: Set[Tuple[str, str]]) -> float:
    if not a and not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return 1.0 - (len(a & b) / len(union))


def compute_chemistry_scores(
    helms: Sequence[str],
    target_len: Optional[int] = None,
) -> np.ndarray:
    """
    Lightweight medicinal-chemistry prior for cyclic permeability/docking DPO.

    The score rewards 7-mer-like cyclic peptides that balance:
      - moderate N-methylation
      - lower H-bond donor burden
      - 1-3 aromatic residues
      - mixed hydrophobic/polar composition
      - some stereochemical/conformational diversity
    """
    scores = np.zeros(len(helms), dtype=np.float64)

    for idx, helm in enumerate(helms):
        tokens = extract_tokens_from_helm(helm)
        n = len(tokens)
        if n == 0:
            continue

        expected_len = target_len or n
        n_nme = sum(is_nmethylated(t) for t in tokens)
        n_pro = sum(is_proline_like(t) for t in tokens)
        n_aro = sum(is_aromatic(t) for t in tokens)
        n_hydro = sum(is_hydrophobic(t) for t in tokens)
        n_polar = sum(is_polar(t) for t in tokens)
        n_d = sum(is_d_amino(t) for t in tokens)
        hbd = max(0, n - n_nme - n_pro)

        hydrophobic_ratio = n_hydro / n
        polar_ratio = n_polar / n

        if expected_len <= 8:
            nme_score = window_score(float(n_nme), 2.0, 4.0)
            aromatic_score = window_score(float(n_aro), 1.0, 3.0)
        else:
            nme_low = max(2.0, round(0.20 * expected_len))
            nme_high = max(nme_low + 1.0, round(0.45 * expected_len))
            nme_score = window_score(float(n_nme), nme_low, nme_high)

            aro_low = 1.0
            aro_high = max(2.0, round(0.25 * expected_len))
            aromatic_score = window_score(float(n_aro), aro_low, aro_high)

        hbd_score = window_score(float(hbd), 0.0, 4.0 if expected_len <= 8 else 6.0)
        hydrophobic_score = window_score(hydrophobic_ratio, 0.35, 0.72)
        polar_score = window_score(polar_ratio, 0.05, 0.30)
        d_score = min(1.0, n_d / max(1.0, 0.15 * expected_len))

        score = (
            0.28 * nme_score
            + 0.24 * hbd_score
            + 0.18 * aromatic_score
            + 0.15 * hydrophobic_score
            + 0.10 * polar_score
            + 0.05 * d_score
        )
        scores[idx] = score

    return scores


@dataclass
class CandidateRecord:
    helm: str
    reward: float
    source: str
    tokens: Tuple[str, ...]
    bigrams: Set[Tuple[str, str]]
    vina_score: Optional[float] = None
    perm_score: Optional[float] = None
    chemistry_score: Optional[float] = None


def build_candidate_records(
    helms: Sequence[str],
    rewards: Sequence[float],
    source_labels: Optional[Sequence[str]] = None,
    vina_scores: Optional[Sequence[float]] = None,
    perm_scores: Optional[Sequence[float]] = None,
    chemistry_scores: Optional[Sequence[float]] = None,
) -> List[CandidateRecord]:
    records: List[CandidateRecord] = []
    default_sources = ["generated"] * len(helms)
    for idx, (helm, reward) in enumerate(zip(helms, rewards)):
        tokens = tuple(extract_tokens_from_helm(helm))
        records.append(
            CandidateRecord(
                helm=helm,
                reward=float(reward),
                source=(source_labels or default_sources)[idx],
                tokens=tokens,
                bigrams=build_ring_bigrams(tokens),
                vina_score=None if vina_scores is None else float(vina_scores[idx]),
                perm_score=None if perm_scores is None else float(perm_scores[idx]),
                chemistry_score=None if chemistry_scores is None else float(chemistry_scores[idx]),
            )
        )
    return records


def select_diverse_subset(
    records: Sequence[CandidateRecord],
    candidate_indices: Sequence[int],
    target_count: int,
    base_values: Sequence[float],
    diversity_lambda: float = 0.25,
) -> List[int]:
    """
    Greedy maximal marginal relevance selection.

    `base_values` should already encode whether high or low raw reward is preferred.
    """
    indices = list(candidate_indices)
    if target_count <= 0 or not indices:
        return []
    if len(indices) <= target_count:
        return indices

    base_arr = np.asarray(base_values, dtype=np.float64)
    if base_arr.shape[0] != len(indices):
        raise ValueError("base_values must align with candidate_indices")

    base_norm = robust_normalize(base_arr)
    local_score = {idx: float(score) for idx, score in zip(indices, base_norm)}

    if diversity_lambda <= 0.0:
        return sorted(indices, key=lambda idx: local_score[idx], reverse=True)[:target_count]

    selected: List[int] = []
    remaining = set(indices)

    first = max(indices, key=lambda idx: local_score[idx])
    selected.append(first)
    remaining.remove(first)
    min_distance_to_selected = {
        idx: jaccard_distance(records[idx].bigrams, records[first].bigrams)
        for idx in remaining
    }

    while remaining and len(selected) < target_count:
        best_idx = None
        best_score = None
        for idx in remaining:
            score = local_score[idx] + diversity_lambda * min_distance_to_selected[idx]
            if best_score is None or score > best_score:
                best_score = score
                best_idx = idx

        selected.append(best_idx)
        remaining.remove(best_idx)
        min_distance_to_selected.pop(best_idx, None)

        selected_bigrams = records[best_idx].bigrams
        for idx in remaining:
            dist = jaccard_distance(records[idx].bigrams, selected_bigrams)
            if dist < min_distance_to_selected[idx]:
                min_distance_to_selected[idx] = dist

    return selected


def pair_winners_with_losers(
    records: Sequence[CandidateRecord],
    winner_indices: Sequence[int],
    loser_indices: Sequence[int],
    strategy: str = "nearest_hard_negative",
    min_reward_gap: float = 0.0,
    allow_reuse: bool = False,
) -> List[Tuple[int, int]]:
    """Pair winners with structurally informative losers."""
    if len(winner_indices) == 0 or len(loser_indices) == 0:
        return []

    if strategy == "random":
        pair_count = min(len(winner_indices), len(loser_indices))
        rng = np.random.default_rng(42)
        winners = list(winner_indices)
        losers = list(loser_indices)
        rng.shuffle(winners)
        rng.shuffle(losers)
        return list(zip(winners[:pair_count], losers[:pair_count]))

    winners = sorted(winner_indices, key=lambda idx: records[idx].reward, reverse=True)
    available_losers = list(loser_indices)
    pairs: List[Tuple[int, int]] = []

    for w_idx in winners:
        if not available_losers:
            break

        eligible = [
            l_idx for l_idx in available_losers
            if records[w_idx].reward - records[l_idx].reward >= min_reward_gap
        ]
        candidates = eligible or available_losers

        if strategy == "nearest_hard_negative":
            chosen = min(
                candidates,
                key=lambda l_idx: (
                    jaccard_distance(records[w_idx].bigrams, records[l_idx].bigrams),
                    -records[l_idx].reward,
                ),
            )
        else:
            raise ValueError(f"Unknown pair strategy: {strategy}")

        pairs.append((w_idx, chosen))
        if not allow_reuse:
            available_losers.remove(chosen)

    return pairs


def summarize_sources(indices: Iterable[int], records: Sequence[CandidateRecord]) -> Dict[str, int]:
    summary: Dict[str, int] = {}
    for idx in indices:
        source = records[idx].source
        summary[source] = summary.get(source, 0) + 1
    return dict(sorted(summary.items()))
