"""Teacher-forced evaluation utilities for the autoregressive ring predictor.

This module deliberately evaluates the ring heads directly.  It does not run
the diffusion denoiser, token mapper, chemical constraints, or free generation.
Candidate pairs follow the training convention: every ``(j, t)`` satisfying
``0 <= j < t < sequence_length`` is valid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
)


BOND_TYPES: Tuple[str, ...] = ("R3R3", "R1R2", "R1R3", "R3R2")


@dataclass(frozen=True)
class PeptideRingPredictions:
    """Predictions and labels for all candidate pairs of one peptide."""

    pairs: np.ndarray
    position_logits: np.ndarray
    position_labels: np.ndarray
    type_logits: np.ndarray
    type_labels: np.ndarray
    helm_sequence: str = ""

    def __post_init__(self) -> None:
        num_pairs = int(self.position_logits.shape[0])
        expected = {
            "pairs": (num_pairs, 2),
            "position_labels": (num_pairs,),
            "type_logits": (num_pairs, len(BOND_TYPES)),
            "type_labels": (num_pairs,),
        }
        for name, shape in expected.items():
            actual = tuple(getattr(self, name).shape)
            if actual != shape:
                raise ValueError(
                    f"Invalid {name} shape {actual}; expected {shape}"
                )

    @property
    def ground_truth_topology(self) -> set:
        positive = self.position_labels.astype(bool)
        return {
            (int(j), int(t), int(ring_type))
            for (j, t), ring_type in zip(
                self.pairs[positive], self.type_labels[positive]
            )
        }

    @property
    def position_probabilities(self) -> np.ndarray:
        """Sigmoid probabilities corresponding to the retained raw logits."""
        return _sigmoid(self.position_logits)


@dataclass(frozen=True)
class RingPredictionRecords:
    """Teacher-forced ring predictions for a complete dataset."""

    peptides: Sequence[PeptideRingPredictions]

    def concatenate(self, field: str) -> np.ndarray:
        arrays = [getattr(peptide, field) for peptide in self.peptides]
        if arrays:
            return np.concatenate(arrays, axis=0)
        if field == "pairs":
            return np.empty((0, 2), dtype=np.int64)
        if field == "type_logits":
            return np.empty((0, len(BOND_TYPES)), dtype=np.float32)
        if field in {"position_logits"}:
            return np.empty((0,), dtype=np.float32)
        return np.empty((0,), dtype=np.int64)

    @property
    def position_logits(self) -> np.ndarray:
        return self.concatenate("position_logits")

    @property
    def position_labels(self) -> np.ndarray:
        return self.concatenate("position_labels")

    @property
    def position_probabilities(self) -> np.ndarray:
        return _sigmoid(self.position_logits)

    @property
    def positive_type_logits(self) -> np.ndarray:
        logits = []
        for peptide in self.peptides:
            positive = peptide.position_labels.astype(bool)
            logits.append(peptide.type_logits[positive])
        if not logits:
            return np.empty((0, len(BOND_TYPES)), dtype=np.float32)
        return np.concatenate(logits, axis=0)

    @property
    def positive_type_labels(self) -> np.ndarray:
        labels = []
        for peptide in self.peptides:
            positive = peptide.position_labels.astype(bool)
            labels.append(peptide.type_labels[positive])
        if not labels:
            return np.empty((0,), dtype=np.int64)
        return np.concatenate(labels, axis=0)


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid returning float64 scores."""
    logits = np.asarray(logits, dtype=np.float64)
    scores = np.empty_like(logits)
    positive = logits >= 0
    scores[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp_logits = np.exp(logits[~positive])
    scores[~positive] = exp_logits / (1.0 + exp_logits)
    return scores


def _normalise_ground_truth(
    bonds: Sequence[Mapping[str, Any]], sequence_length: int
) -> Dict[Tuple[int, int], int]:
    """Validate and normalize ring labels to ``(j, t) -> type``."""
    ground_truth: Dict[Tuple[int, int], int] = {}
    for bond in bonds:
        i = int(bond["i"])
        j = int(bond["j"])
        ring_type = int(bond["type"])
        if i > j:
            i, j = j, i
        if not (0 <= i < j < sequence_length):
            raise ValueError(
                "Ground-truth ring bond is not a valid candidate pair: "
                f"(i={i}, j={j}, length={sequence_length})"
            )
        if not (0 <= ring_type < len(BOND_TYPES)):
            raise ValueError(f"Invalid ring type label: {ring_type}")
        pair = (i, j)
        previous = ground_truth.get(pair)
        if previous is not None and previous != ring_type:
            raise ValueError(
                f"Candidate pair {pair} has conflicting type labels "
                f"{previous} and {ring_type}"
            )
        ground_truth[pair] = ring_type
    return ground_truth


@torch.no_grad()
def collect_teacher_forced_predictions(
    model: torch.nn.Module,
    data_loader: Iterable[Mapping[str, Any]],
    device: torch.device,
) -> RingPredictionRecords:
    """Collect raw ring-head outputs on ground-truth monomer sequences.

    The model's own ``_prepare_contexts`` method is used so that the resulting
    ``h_t`` and ``h_j`` are identical to those used by cyclic fine-tuning.
    Padding is excluded using the supplied sequence mask.  Type outputs are
    retained for every pair so topology predictions can be formed after a
    shared position threshold has been selected.
    """
    model.eval()
    peptide_records: List[PeptideRingPredictions] = []

    for batch in data_loader:
        token_ids = batch["token_ids"].to(device)
        mask = batch["mask"].to(device).bool()
        ring_bonds = batch["ring_bonds"]
        helm_sequences = batch.get("helm_sequences", [""] * token_ids.size(0))
        lengths = mask.sum(dim=1).long()

        use_r_sites = bool(getattr(model, "use_r_site_embeddings", True))
        if use_r_sites:
            _, contexts, r_embeddings = model._prepare_contexts(
                token_ids, mask, return_r_groups=True
            )
        else:
            _, contexts = model._prepare_contexts(token_ids, mask)
            r_embeddings = None

        batch_pairs: List[List[Tuple[int, int]]] = [
            [] for _ in range(token_ids.size(0))
        ]
        batch_position_logits: List[List[np.ndarray]] = [
            [] for _ in range(token_ids.size(0))
        ]
        batch_type_logits: List[List[np.ndarray]] = [
            [] for _ in range(token_ids.size(0))
        ]

        max_length = int(lengths.max().item()) if lengths.numel() else 0
        for t in range(1, max_length):
            active = t < lengths
            if not bool(active.any()):
                continue

            current_context = contexts[:, t, :]
            history_context = contexts[:, :t, :]
            if use_r_sites:
                if r_embeddings is None:
                    raise RuntimeError(
                        "Full ring predictor evaluation requires R-site embeddings"
                    )
                position_logits, type_logits = model.ar_ring_predictor(
                    current_context,
                    r_embeddings[:, t, :, :],
                    history_context,
                    r_embeddings[:, :t, :, :],
                )
            else:
                position_logits, type_logits = model.ar_ring_predictor(
                    current_context, history_context
                )

            for batch_index in active.nonzero(as_tuple=False).flatten().tolist():
                batch_pairs[batch_index].extend((j, t) for j in range(t))
                batch_position_logits[batch_index].append(
                    position_logits[batch_index, :t]
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                )
                batch_type_logits[batch_index].append(
                    type_logits[batch_index, :t]
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                )

        for batch_index in range(token_ids.size(0)):
            sequence_length = int(lengths[batch_index].item())
            ground_truth = _normalise_ground_truth(
                ring_bonds[batch_index], sequence_length
            )
            pairs = np.asarray(batch_pairs[batch_index], dtype=np.int64).reshape(-1, 2)
            if batch_position_logits[batch_index]:
                position_logits_np = np.concatenate(
                    batch_position_logits[batch_index], axis=0
                ).astype(np.float32, copy=False)
                type_logits_np = np.concatenate(
                    batch_type_logits[batch_index], axis=0
                ).astype(np.float32, copy=False)
            else:
                position_logits_np = np.empty((0,), dtype=np.float32)
                type_logits_np = np.empty(
                    (0, len(BOND_TYPES)), dtype=np.float32
                )

            position_labels = np.fromiter(
                (int(tuple(pair) in ground_truth) for pair in pairs),
                dtype=np.int64,
                count=len(pairs),
            )
            # -1 is a sentinel for negative pairs.  Metrics only read type
            # labels where position_labels == 1.
            type_labels = np.fromiter(
                (ground_truth.get(tuple(pair), -1) for pair in pairs),
                dtype=np.int64,
                count=len(pairs),
            )
            if int(position_labels.sum()) != len(ground_truth):
                raise RuntimeError(
                    "Not every ground-truth ring bond was represented by a "
                    "valid candidate pair"
                )

            peptide_records.append(
                PeptideRingPredictions(
                    pairs=pairs,
                    position_logits=position_logits_np,
                    position_labels=position_labels,
                    type_logits=type_logits_np,
                    type_labels=type_labels,
                    helm_sequence=str(helm_sequences[batch_index]),
                )
            )

    return RingPredictionRecords(peptides=peptide_records)


def search_position_threshold(
    records: RingPredictionRecords,
) -> Dict[str, Optional[float]]:
    """Find the validation threshold that maximizes global position F1."""
    labels = records.position_labels.astype(np.int64, copy=False)
    scores = _sigmoid(records.position_logits)
    if labels.size == 0 or int(labels.sum()) == 0:
        return {
            "threshold": None,
            "position_precision": None,
            "position_recall": None,
            "position_f1": None,
        }

    precision, recall, thresholds = precision_recall_curve(labels, scores)
    if thresholds.size == 0:
        return {
            "threshold": 0.5,
            "position_precision": float(precision[0]),
            "position_recall": float(recall[0]),
            "position_f1": float(
                2 * precision[0] * recall[0]
                / max(precision[0] + recall[0], np.finfo(float).eps)
            ),
        }

    denominator = precision[:-1] + recall[:-1]
    f1 = np.divide(
        2 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    best_f1 = float(f1.max())
    # Prefer the largest threshold among exact F1 ties.  This is deterministic
    # and avoids unnecessarily adding predicted bonds.
    tied = np.flatnonzero(np.isclose(f1, best_f1, rtol=1e-12, atol=1e-15))
    best_index = int(tied[-1])
    return {
        "threshold": float(thresholds[best_index]),
        "position_precision": float(precision[best_index]),
        "position_recall": float(recall[best_index]),
        "position_f1": best_f1,
    }


def evaluate_ring_predictions(
    records: RingPredictionRecords, position_threshold: float
) -> Dict[str, Any]:
    """Compute all requested global, type, count, and topology metrics."""
    if not 0.0 <= position_threshold <= 1.0:
        raise ValueError("position_threshold must be in [0, 1]")

    labels = records.position_labels.astype(np.int64, copy=False)
    scores = _sigmoid(records.position_logits)
    predictions = (scores >= position_threshold).astype(np.int64)
    num_positive = int(labels.sum())
    num_negative = int(labels.size - num_positive)

    if labels.size == 0 or num_positive == 0:
        auprc: Optional[float] = None
    else:
        # This is intentionally computed once over every candidate pair in the
        # dataset, never as an average of per-batch AUPRC values.
        auprc = float(average_precision_score(labels, scores))

    if labels.size:
        pos_precision, pos_recall, pos_f1, _ = precision_recall_fscore_support(
            labels,
            predictions,
            labels=[1],
            average=None,
            zero_division=0,
        )
        position_metrics = {
            "precision": float(pos_precision[0]),
            "recall": float(pos_recall[0]),
            "f1": float(pos_f1[0]),
        }
    else:
        position_metrics = {"precision": None, "recall": None, "f1": None}

    true_types = records.positive_type_labels
    positive_type_logits = records.positive_type_logits
    predicted_types = (
        positive_type_logits.argmax(axis=1).astype(np.int64)
        if positive_type_logits.shape[0]
        else np.empty((0,), dtype=np.int64)
    )
    type_indices = np.arange(len(BOND_TYPES), dtype=np.int64)
    if true_types.size:
        type_precision, type_recall, type_f1, type_support = (
            precision_recall_fscore_support(
                true_types,
                predicted_types,
                labels=type_indices,
                average=None,
                zero_division=0,
            )
        )
        type_accuracy: Optional[float] = float(
            accuracy_score(true_types, predicted_types)
        )
        type_macro_f1: Optional[float] = float(type_f1.mean())
        type_confusion = confusion_matrix(
            true_types, predicted_types, labels=type_indices
        ).astype(int)
    else:
        type_precision = np.zeros(len(BOND_TYPES), dtype=float)
        type_recall = np.zeros(len(BOND_TYPES), dtype=float)
        type_f1 = np.zeros(len(BOND_TYPES), dtype=float)
        type_support = np.zeros(len(BOND_TYPES), dtype=int)
        type_accuracy = None
        type_macro_f1 = None
        type_confusion = np.zeros(
            (len(BOND_TYPES), len(BOND_TYPES)), dtype=int
        )

    per_class = {}
    missing_classes = []
    for index, name in enumerate(BOND_TYPES):
        support = int(type_support[index])
        per_class[name] = {
            "precision": float(type_precision[index]),
            "recall": float(type_recall[index]),
            "f1": float(type_f1[index]),
            "support": support,
        }
        if support == 0:
            missing_classes.append(name)

    topology_correct = 0
    predicted_bond_total = 0
    ground_truth_bond_total = 0
    for peptide in records.peptides:
        peptide_scores = _sigmoid(peptide.position_logits)
        selected = peptide_scores >= position_threshold
        selected_types = peptide.type_logits.argmax(axis=1)
        predicted_topology = {
            (int(j), int(t), int(ring_type))
            for (j, t), ring_type in zip(
                peptide.pairs[selected], selected_types[selected]
            )
        }
        ground_truth_topology = peptide.ground_truth_topology
        predicted_bond_total += len(predicted_topology)
        ground_truth_bond_total += len(ground_truth_topology)
        topology_correct += int(predicted_topology == ground_truth_topology)

    num_peptides = len(records.peptides)
    topology_exact_match = (
        float(topology_correct / num_peptides) if num_peptides else None
    )

    return {
        "position_threshold": float(position_threshold),
        "bond_position_auprc": auprc,
        "position": {
            **position_metrics,
            "positive_pairs": num_positive,
            "negative_pairs": num_negative,
            "candidate_pairs": int(labels.size),
        },
        "bond_type_macro_f1": type_macro_f1,
        "bond_type_accuracy": type_accuracy,
        "bond_type_per_class": per_class,
        "bond_type_missing_ground_truth_classes": missing_classes,
        "bond_type_confusion_matrix": {
            "labels": list(BOND_TYPES),
            "matrix": type_confusion.tolist(),
        },
        "topology_exact_match": topology_exact_match,
        "topology_exact_match_correct_peptides": topology_correct,
        "test_peptides": num_peptides,
        "ground_truth_ring_bonds": ground_truth_bond_total,
        "predicted_ring_bonds": predicted_bond_total,
    }
