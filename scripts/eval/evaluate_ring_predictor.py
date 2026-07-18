#!/usr/bin/env python3
"""Compare Full and w/o R-site ring predictors with teacher forcing.

Example (fixed shared threshold)::

    python scripts/eval/evaluate_ring_predictor.py \
      --full_checkpoint /path/to/full/checkpoint_epoch_40.pt \
      --wo_rsite_checkpoint /path/to/wo_rsite/checkpoint_epoch_40.pt \
      --test_data /path/to/cycpept_test.txt \
      --position_threshold 0.5

For validation threshold selection, add ``--search_threshold_on_validation``
and ``--validation_data``.  One threshold is selected from the requested
reference model and then used unchanged for both test models.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from pepar_diff import AutoregressiveLatentDiffusion
from pepar_diff.config import ALDConfig
from pepar_diff.data import CyclicHELMDataset, HELMCollator
from pepar_diff.evaluation.ring_predictor import (
    BOND_TYPES,
    RingPredictionRecords,
    collect_teacher_forced_predictions,
    evaluate_ring_predictions,
    search_position_threshold,
)


DEFAULT_TEST_DATA = "data/processed/helm_sequences_cycpeptmpdb.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Teacher-forced Ring Predictor comparison: Full vs w/o R-site"
        )
    )
    parser.add_argument(
        "--full_config",
        default="configs/training/finetune_cyclic.json",
        help="Full PepALD cyclic fine-tuning config",
    )
    parser.add_argument(
        "--full_checkpoint",
        default=None,
        help="Full model checkpoint (defaults to config generation checkpoint)",
    )
    parser.add_argument(
        "--wo_rsite_config",
        default="configs/ablations/rsite/finetune_cyclic_context_only.json",
        help="w/o R-site cyclic fine-tuning config",
    )
    parser.add_argument(
        "--wo_rsite_checkpoint",
        default=None,
        help="w/o R-site checkpoint (defaults to config generation checkpoint)",
    )
    parser.add_argument(
        "--test_data",
        default=DEFAULT_TEST_DATA,
        help="Held-out CycPeptMPDB HELM txt/csv file",
    )
    parser.add_argument(
        "--position_threshold",
        type=float,
        default=0.5,
        help="One fixed position probability threshold shared by both models",
    )
    parser.add_argument(
        "--search_threshold_on_validation",
        action="store_true",
        help="Search a shared threshold using validation position F1",
    )
    parser.add_argument(
        "--validation_data",
        default=None,
        help="Validation HELM file required for threshold search",
    )
    parser.add_argument(
        "--threshold_reference",
        choices=("full", "wo_rsite", "pooled"),
        default="full",
        help=(
            "Validation predictions used to choose the one shared threshold; "
            "'pooled' concatenates both models"
        ),
    )
    parser.add_argument(
        "--threshold_output",
        default="outputs/evaluation/ring_predictor_position_threshold.json",
        help="JSON file used to save validation threshold selection",
    )
    parser.add_argument(
        "--output_json",
        default="outputs/evaluation/ring_predictor_full_vs_wo_rsite.json",
        help="Comparison report path",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device (default: cuda when available, otherwise cpu)",
    )
    return parser.parse_args()


def _require_file(path: str, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def _checkpoint_path(config: ALDConfig, override: Optional[str]) -> Path:
    path = override or config.generation.checkpoint_path
    return _require_file(path, "checkpoint")


def _extract_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if isinstance(checkpoint, dict) and checkpoint and all(
        isinstance(value, torch.Tensor) for value in checkpoint.values()
    ):
        return checkpoint
    raise ValueError("Checkpoint does not contain a recognized model state dict")


def _load_checkpoint_vocab(
    checkpoint: Any, config: ALDConfig
) -> Dict[str, int]:
    if isinstance(checkpoint, dict) and "vocab" in checkpoint:
        return checkpoint["vocab"]
    with open(config.training.vocab_file, "r") as handle:
        return json.load(handle)


def load_model(
    config_path: str,
    checkpoint_override: Optional[str],
    expected_ring_feature_mode: str,
    device: torch.device,
) -> Tuple[AutoregressiveLatentDiffusion, Dict[str, int], ALDConfig, Path]:
    config_file = _require_file(config_path, "config")
    config = ALDConfig.load(str(config_file))
    actual_mode = getattr(config.model, "ring_feature_mode", "context_rsite")
    if actual_mode != expected_ring_feature_mode:
        raise ValueError(
            f"{config_file} has ring_feature_mode={actual_mode!r}; expected "
            f"{expected_ring_feature_mode!r}"
        )
    checkpoint_path = _checkpoint_path(config, checkpoint_override)
    print(f"Loading {actual_mode} checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    vocab = _load_checkpoint_vocab(checkpoint, config)
    model = AutoregressiveLatentDiffusion(vocab=vocab, config=config)
    model.load_state_dict(_extract_state_dict(checkpoint), strict=True)
    model.to(device).eval()
    return model, vocab, config, checkpoint_path


def make_loader(
    data_file: Path,
    config: ALDConfig,
    vocab: Dict[str, int],
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    with open(config.training.vocab_file, "r") as handle:
        file_vocab = json.load(handle)
    if file_vocab != vocab:
        raise ValueError(
            "Checkpoint vocabulary differs from config.training.vocab_file; "
            "refusing to evaluate token IDs with a mismatched vocabulary"
        )
    dataset = CyclicHELMDataset(
        data_file=str(data_file),
        vocab_file=config.training.vocab_file,
        max_seq_len=config.model.max_seq_len,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=HELMCollator(pad_id=vocab["<PAD>"]),
        pin_memory=torch.cuda.is_available(),
    )


def _warn_if_training_data(
    evaluation_file: Path, config: ALDConfig, model_name: str
) -> None:
    training_file = Path(config.training.train_data_file).expanduser().resolve()
    if evaluation_file == training_file:
        print(
            "\nWARNING: data leakage risk: "
            f"{model_name} was configured to train on the same file being "
            f"evaluated ({evaluation_file}). These numbers are in-sample and "
            "must not be reported as held-out test performance.\n"
        )


def _collect_for_model(
    model: AutoregressiveLatentDiffusion,
    config: ALDConfig,
    vocab: Dict[str, int],
    data_file: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> RingPredictionRecords:
    loader = make_loader(
        data_file,
        config,
        vocab,
        args.batch_size,
        args.num_workers,
    )
    return collect_teacher_forced_predictions(model, loader, device)


def _pooled_records(
    first: RingPredictionRecords, second: RingPredictionRecords
) -> RingPredictionRecords:
    return RingPredictionRecords(peptides=[*first.peptides, *second.peptides])


def _assert_same_evaluation_targets(
    full: RingPredictionRecords, wo_rsite: RingPredictionRecords
) -> None:
    """Ensure both models were evaluated on identical peptides and labels."""
    if len(full.peptides) != len(wo_rsite.peptides):
        raise RuntimeError(
            "Full and w/o R-site evaluation produced different peptide counts"
        )
    for index, (full_item, wo_item) in enumerate(
        zip(full.peptides, wo_rsite.peptides)
    ):
        same = (
            full_item.helm_sequence == wo_item.helm_sequence
            and np.array_equal(full_item.pairs, wo_item.pairs)
            and np.array_equal(
                full_item.position_labels, wo_item.position_labels
            )
            and np.array_equal(full_item.type_labels, wo_item.type_labels)
        )
        if not same:
            raise RuntimeError(
                "Full and w/o R-site evaluation targets differ at peptide "
                f"index {index}"
            )


def _format_metric(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def print_report(report: Dict[str, Any]) -> None:
    full = report["models"]["full"]
    ablated = report["models"]["wo_rsite"]
    rows = (
        ("Bond-position AUPRC", "bond_position_auprc"),
        ("Position precision", ("position", "precision")),
        ("Position recall", ("position", "recall")),
        ("Position F1", ("position", "f1")),
        ("Bond-type Macro-F1", "bond_type_macro_f1"),
        ("Bond-type accuracy", "bond_type_accuracy"),
        ("Topology exact match", "topology_exact_match"),
        ("Predicted ring bonds", "predicted_ring_bonds"),
    )
    print("\nTeacher-forced Ring Predictor comparison")
    print(f"Shared position threshold: {report['position_threshold']:.8f}")
    print(f"{'Metric':<28} {'Full':>14} {'w/o R-site':>14}")
    print("-" * 58)
    for label, key in rows:
        if isinstance(key, tuple):
            full_value = full[key[0]][key[1]]
            ablated_value = ablated[key[0]][key[1]]
        else:
            full_value = full[key]
            ablated_value = ablated[key]
        print(
            f"{label:<28} {_format_metric(full_value):>14} "
            f"{_format_metric(ablated_value):>14}"
        )

    for model_name, metrics in (("Full", full), ("w/o R-site", ablated)):
        print(f"\n{model_name} bond-type metrics")
        print(f"{'Type':<8} {'Precision':>11} {'Recall':>11} {'F1':>11} {'Support':>9}")
        for ring_type in BOND_TYPES:
            item = metrics["bond_type_per_class"][ring_type]
            print(
                f"{ring_type:<8} {item['precision']:>11.6f} "
                f"{item['recall']:>11.6f} {item['f1']:>11.6f} "
                f"{item['support']:>9d}"
            )
        missing = metrics["bond_type_missing_ground_truth_classes"]
        if missing:
            print(
                "NOTE: no ground-truth test samples for: "
                + ", ".join(missing)
                + "; precision/recall/F1 are set to 0 and included in the "
                "four-class Macro-F1."
            )
        confusion = metrics["bond_type_confusion_matrix"]
        print("Type confusion matrix (rows=true, columns=predicted)")
        print("labels:", confusion["labels"])
        for row in confusion["matrix"]:
            print(row)
        print(
            "Counts: "
            f"peptides={metrics['test_peptides']}, "
            f"position_pos={metrics['position']['positive_pairs']}, "
            f"position_neg={metrics['position']['negative_pairs']}, "
            f"gt_bonds={metrics['ground_truth_ring_bonds']}, "
            f"pred_bonds={metrics['predicted_ring_bonds']}"
        )


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if not 0.0 <= args.position_threshold <= 1.0:
        raise ValueError("position_threshold must be in [0, 1]")
    if args.search_threshold_on_validation and not args.validation_data:
        raise ValueError(
            "--validation_data is required with "
            "--search_threshold_on_validation"
        )

    device = torch.device(
        args.device
        or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    test_data = _require_file(args.test_data, "test data")
    validation_data = (
        _require_file(args.validation_data, "validation data")
        if args.validation_data
        else None
    )
    if validation_data is not None and validation_data == test_data:
        raise ValueError(
            "validation_data and test_data resolve to the same file; refusing "
            "to tune the threshold on test labels"
        )

    print(f"Evaluation device: {device}")
    full_model, full_vocab, full_config, full_checkpoint = load_model(
        args.full_config,
        args.full_checkpoint,
        expected_ring_feature_mode="context_rsite",
        device=device,
    )
    _warn_if_training_data(test_data, full_config, "Full")
    full_test = _collect_for_model(
        full_model, full_config, full_vocab, test_data, args, device
    )
    full_validation = None
    if args.search_threshold_on_validation:
        _warn_if_training_data(validation_data, full_config, "Full validation")
        full_validation = _collect_for_model(
            full_model,
            full_config,
            full_vocab,
            validation_data,
            args,
            device,
        )
    del full_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    wo_model, wo_vocab, wo_config, wo_checkpoint = load_model(
        args.wo_rsite_config,
        args.wo_rsite_checkpoint,
        expected_ring_feature_mode="context_only",
        device=device,
    )
    if wo_vocab != full_vocab:
        raise ValueError("Full and w/o R-site checkpoints use different vocabularies")
    if wo_config.model.max_seq_len != full_config.model.max_seq_len:
        raise ValueError("Full and w/o R-site configs use different max_seq_len")
    _warn_if_training_data(test_data, wo_config, "w/o R-site")
    wo_test = _collect_for_model(
        wo_model, wo_config, wo_vocab, test_data, args, device
    )
    wo_validation = None
    if (
        args.search_threshold_on_validation
        and args.threshold_reference in {"wo_rsite", "pooled"}
    ):
        _warn_if_training_data(
            validation_data, wo_config, "w/o R-site validation"
        )
        wo_validation = _collect_for_model(
            wo_model,
            wo_config,
            wo_vocab,
            validation_data,
            args,
            device,
        )
    del wo_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    _assert_same_evaluation_targets(full_test, wo_test)

    threshold = float(args.position_threshold)
    threshold_selection: Dict[str, Any] = {
        "mode": "fixed",
        "threshold": threshold,
    }
    if args.search_threshold_on_validation:
        if args.threshold_reference == "full":
            threshold_records = full_validation
        elif args.threshold_reference == "wo_rsite":
            threshold_records = wo_validation
        else:
            if full_validation is None or wo_validation is None:
                raise RuntimeError("Pooled threshold search records are incomplete")
            threshold_records = _pooled_records(full_validation, wo_validation)
        search_result = search_position_threshold(threshold_records)
        if search_result["threshold"] is None:
            raise ValueError(
                "Validation set has no positive ring pairs; a position F1 "
                "threshold cannot be selected"
            )
        threshold = float(search_result["threshold"])
        threshold_selection = {
            "mode": "validation_position_f1",
            "reference": args.threshold_reference,
            "validation_data": str(validation_data),
            **search_result,
        }
        threshold_path = Path(args.threshold_output)
        threshold_path.parent.mkdir(parents=True, exist_ok=True)
        with open(threshold_path, "w") as handle:
            json.dump(threshold_selection, handle, indent=2)
        print(f"Saved validation-selected threshold to {threshold_path}")

    report = {
        "evaluation_mode": "teacher_forced",
        "bond_types": list(BOND_TYPES),
        "position_threshold": threshold,
        "threshold_selection": threshold_selection,
        "test_data": str(test_data),
        "checkpoints": {
            "full": str(full_checkpoint),
            "wo_rsite": str(wo_checkpoint),
        },
        "models": {
            "full": evaluate_ring_predictions(full_test, threshold),
            "wo_rsite": evaluate_ring_predictions(wo_test, threshold),
        },
    }
    print_report(report)

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as handle:
        json.dump(report, handle, indent=2)
    print(f"\nSaved comparison report to {output_path}")


if __name__ == "__main__":
    main()
