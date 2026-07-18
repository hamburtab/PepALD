#!/usr/bin/env python3
"""Small deterministic tests for teacher-forced Ring Predictor evaluation."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from pepar_diff.evaluation.ring_predictor import (
    PeptideRingPredictions,
    RingPredictionRecords,
    collect_teacher_forced_predictions,
    evaluate_ring_predictions,
    search_position_threshold,
)


def _peptide(pairs, position_logits, labels, predicted_types, true_types):
    type_logits = np.full((len(pairs), 4), -5.0, dtype=np.float32)
    for row, ring_type in enumerate(predicted_types):
        type_logits[row, ring_type] = 5.0
    return PeptideRingPredictions(
        pairs=np.asarray(pairs, dtype=np.int64).reshape(-1, 2),
        position_logits=np.asarray(position_logits, dtype=np.float32),
        position_labels=np.asarray(labels, dtype=np.int64),
        type_logits=type_logits,
        type_labels=np.asarray(true_types, dtype=np.int64),
    )


class _DummyPredictor(nn.Module):
    def __init__(self, with_r_sites: bool):
        super().__init__()
        self.with_r_sites = with_r_sites

    def forward(self, current_context, *args):
        if self.with_r_sites:
            _, history_context, _ = args
        else:
            (history_context,) = args
        batch_size, history_length, _ = history_context.shape
        # Deterministic raw outputs whose exact values are immaterial here.
        position = torch.arange(
            history_length, device=current_context.device, dtype=torch.float32
        ).expand(batch_size, -1)
        types = torch.zeros(
            batch_size, history_length, 4, device=current_context.device
        )
        types[..., 1] = 1.0
        return position, types


class _DummyModel(nn.Module):
    def __init__(self, with_r_sites: bool):
        super().__init__()
        self.use_r_site_embeddings = with_r_sites
        self.ar_ring_predictor = _DummyPredictor(with_r_sites)

    def _prepare_contexts(self, token_ids, mask, return_r_groups=False):
        batch, length = token_ids.shape
        contexts = torch.zeros(batch, length, 3, device=token_ids.device)
        embeddings = torch.zeros(batch, length, 2, device=token_ids.device)
        if return_r_groups:
            r_groups = torch.zeros(
                batch, length, 3, 2, device=token_ids.device
            )
            return embeddings, contexts, r_groups
        return embeddings, contexts


class RingPredictorEvaluationTest(unittest.TestCase):
    def test_collection_excludes_padding_and_builds_all_pairs(self):
        batch = {
            "token_ids": torch.tensor([[1, 2, 3], [4, 5, 0]]),
            "mask": torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool),
            "ring_bonds": [
                [{"i": 0, "j": 2, "type": 2}],
                [{"i": 0, "j": 1, "type": 3}],
            ],
            "helm_sequences": ["first", "second"],
        }
        for with_r_sites in (True, False):
            records = collect_teacher_forced_predictions(
                _DummyModel(with_r_sites), [batch], torch.device("cpu")
            )
            self.assertEqual(len(records.peptides), 2)
            self.assertEqual(records.position_labels.tolist(), [0, 1, 0, 1])
            self.assertEqual(records.positive_type_labels.tolist(), [2, 3])
            self.assertEqual(
                records.peptides[0].pairs.tolist(), [[0, 1], [0, 2], [1, 2]]
            )
            self.assertEqual(records.peptides[1].pairs.tolist(), [[0, 1]])

    def test_metrics_topology_and_missing_class_are_finite(self):
        first = _peptide(
            pairs=[(0, 1), (0, 2), (1, 2)],
            position_logits=[-5, 5, -5],
            labels=[0, 1, 0],
            predicted_types=[0, 2, 0],
            true_types=[-1, 2, -1],
        )
        # Extra predicted pair makes this peptide fail topology exact match.
        second = _peptide(
            pairs=[(0, 1)],
            position_logits=[5],
            labels=[0],
            predicted_types=[1],
            true_types=[-1],
        )
        metrics = evaluate_ring_predictions(
            RingPredictionRecords([first, second]), position_threshold=0.5
        )
        self.assertAlmostEqual(metrics["topology_exact_match"], 0.5)
        self.assertAlmostEqual(metrics["bond_type_accuracy"], 1.0)
        self.assertAlmostEqual(metrics["bond_type_macro_f1"], 0.25)
        self.assertEqual(
            metrics["bond_type_missing_ground_truth_classes"],
            ["R3R3", "R1R2", "R3R2"],
        )
        self.assertFalse(
            any(
                math.isnan(item["f1"])
                for item in metrics["bond_type_per_class"].values()
            )
        )

    def test_threshold_search_uses_global_pairs(self):
        peptide = _peptide(
            pairs=[(0, 1), (0, 2), (1, 2)],
            position_logits=[-4, 2, 1],
            labels=[0, 1, 0],
            predicted_types=[0, 1, 0],
            true_types=[-1, 1, -1],
        )
        result = search_position_threshold(RingPredictionRecords([peptide]))
        self.assertIsNotNone(result["threshold"])
        self.assertAlmostEqual(result["position_f1"], 1.0)

    def test_auprc_is_computed_globally_not_per_peptide(self):
        # Each peptide has perfect AP in isolation, while the globally ranked
        # candidates have AP = (1 + 2/3) / 2 = 5/6.
        first = _peptide(
            pairs=[(0, 1), (0, 2)],
            position_logits=[math.log(9.0), math.log(4.0)],  # scores .9, .8
            labels=[1, 0],
            predicted_types=[1, 0],
            true_types=[1, -1],
        )
        second = _peptide(
            pairs=[(0, 1), (0, 2)],
            position_logits=[math.log(0.25), math.log(1.0 / 9.0)],  # .2, .1
            labels=[1, 0],
            predicted_types=[1, 0],
            true_types=[1, -1],
        )
        metrics = evaluate_ring_predictions(
            RingPredictionRecords([first, second]), position_threshold=0.5
        )
        self.assertAlmostEqual(metrics["bond_position_auprc"], 5.0 / 6.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
