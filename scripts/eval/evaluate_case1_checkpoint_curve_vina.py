"""Score a 2,400-line case1 checkpoint curve and aggregate 24 Vina values.

The combined input is docked through the existing multi-GPU Vina cache once.
Scores are then expanded back to the original line order and summarized in
consecutive 100-sample groups. The primary result file contains exactly one
number per group, while a companion CSV preserves round/stage metadata and
additional statistics for plotting and auditing.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUTODL_ABLATION_ROOT = Path("/root/autodl-tmp/wp_dpo_ablation")

if AUTODL_ABLATION_ROOT.parent.is_dir():
    DEFAULT_OUTPUT_ROOT = AUTODL_ABLATION_ROOT / "outputs"
    DEFAULT_EVALUATION_ROOT = AUTODL_ABLATION_ROOT / "evaluations"
else:
    DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "ablations" / "wp_dpo"
    DEFAULT_EVALUATION_ROOT = (
        PROJECT_ROOT / "outputs" / "ablations" / "wp_dpo_evaluations"
    )

DEFAULT_CASE1_CONFIG = PROJECT_ROOT / "configs" / "training" / "dpo.json"
INVALID_VINA_SCORE = 0.0
METRIC_CHOICES = ("mean_vina", "median_vina", "best_vina", "top10_mean_vina")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the combined case1 24-checkpoint sample file with Vina "
            "and write one aggregate value for every consecutive 100 samples."
        )
    )
    parser.add_argument("--run_name", required=True, help="Ablation run name.")
    parser.add_argument("--case", choices=["case1"], default="case1")
    parser.add_argument("--arm", choices=["standard_dpo"], default="standard_dpo")
    parser.add_argument(
        "--sample_file",
        default=None,
        help="Defaults to checkpoint_curve_samples/all_24epochs_2400.txt.",
    )
    parser.add_argument(
        "--group_manifest",
        default=None,
        help="Defaults to sample_groups.jsonl next to --sample_file.",
    )
    parser.add_argument("--group_size", type=int, default=100)
    parser.add_argument("--expected_groups", type=int, default=24)
    parser.add_argument(
        "--metric",
        choices=METRIC_CHOICES,
        default="mean_vina",
        help="Statistic written to the 24-line values file (default: mean_vina).",
    )
    parser.add_argument(
        "--gpu_ids",
        default="0,1,2",
        help="Comma-separated Uni-Dock GPU IDs (default: 0,1,2).",
    )
    parser.add_argument(
        "--docking_mode",
        choices=["flexible", "rigid"],
        default=None,
        help="Defaults to dpo.docking_mode in the ablation config.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional Vina config override; normally inferred from the run manifest.",
    )
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--evaluation_root", default=str(DEFAULT_EVALUATION_ROOT))
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate inputs and print the multi-GPU Vina command only.",
    )
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def load_nonempty_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def parse_gpu_ids(value: str) -> list[str]:
    gpu_ids = [part.strip() for part in value.split(",") if part.strip()]
    if not gpu_ids:
        raise ValueError("--gpu_ids must contain at least one GPU ID")
    return gpu_ids


def infer_vina_config(
    explicit_config: str | None,
    output_root: Path,
    run_name: str,
    case_name: str,
    arm: str,
) -> Path:
    if explicit_config:
        config_path = resolve_path(explicit_config)
    else:
        manifest_path = output_root / run_name / case_name / "ablation_manifest.json"
        config_path = DEFAULT_CASE1_CONFIG
        if manifest_path.exists():
            manifest = load_json(manifest_path)
            config_key = "standard_config" if arm == "standard_dpo" else "wp_config"
            if manifest.get(config_key):
                config_path = resolve_path(manifest[config_key])
    if not config_path.exists():
        raise FileNotFoundError(f"Vina config not found: {config_path}")
    return config_path


def load_group_manifest(
    path: Path,
    expected_groups: int,
    group_size: int,
) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Sample group manifest not found: {path}. Run the checkpoint sample "
            "generation script first."
        )
    with path.open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if len(records) != expected_groups:
        raise RuntimeError(
            f"Expected {expected_groups} group records, found {len(records)} in {path}"
        )

    for position, record in enumerate(records, start=1):
        expected_start = (position - 1) * group_size + 1
        expected_end = position * group_size
        if int(record.get("group_index", -1)) != position:
            raise RuntimeError(
                f"Group manifest position {position} has group_index="
                f"{record.get('group_index')!r}"
            )
        if int(record.get("num_samples", -1)) != group_size:
            raise RuntimeError(
                f"Group {position} expected {group_size} samples, "
                f"found {record.get('num_samples')!r}"
            )
        if (
            int(record.get("line_start", -1)) != expected_start
            or int(record.get("line_end", -1)) != expected_end
        ):
            raise RuntimeError(
                f"Group {position} expected lines {expected_start}-{expected_end}, "
                f"found {record.get('line_start')}-{record.get('line_end')}"
            )
    return records


def detect_delimiter(header_line: str, path: Path) -> str:
    if path.suffix.lower() == ".tsv":
        return "\t"
    if "\t" in header_line and "," not in header_line:
        return "\t"
    return ","


def load_score_cache(
    path: Path,
    docking_mode: str,
) -> tuple[list[str], dict[str, dict[str, str]]]:
    if not path.exists():
        raise FileNotFoundError(f"Vina cache not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        header_line = handle.readline()
        if not header_line:
            raise ValueError(f"Vina cache is empty: {path}")
        delimiter = detect_delimiter(header_line, path)
        handle.seek(0)
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"Vina cache has no header: {path}")
        fieldnames = list(reader.fieldnames)
        field_map = {name.strip().lower(): name for name in fieldnames if name}
        helm_key = field_map.get("helm")
        score_key = field_map.get("vina_score") or field_map.get("score")
        mode_key = field_map.get("docking_mode")
        if helm_key is None or score_key is None:
            raise ValueError(f"Vina cache must contain helm and vina_score: {path}")

        rows: dict[str, dict[str, str]] = {}
        for row in reader:
            helm = (row.get(helm_key) or "").strip()
            if not helm:
                continue
            if mode_key is not None:
                row_mode = (row.get(mode_key) or "").strip().lower()
                if row_mode != docking_mode:
                    continue
            normalized = dict(row)
            normalized["helm"] = helm
            normalized["vina_score"] = (row.get(score_key) or "").strip()
            rows[helm] = normalized
    return fieldnames, rows


def parse_score(row: dict[str, str]) -> float:
    try:
        return float(row.get("vina_score", ""))
    except (TypeError, ValueError):
        return INVALID_VINA_SCORE


def summarize_group(scores: Sequence[float]) -> dict[str, float | int]:
    valid_scores = [score for score in scores if score != INVALID_VINA_SCORE]
    invalid_count = len(scores) - len(valid_scores)
    if not valid_scores:
        return {
            "num_valid": 0,
            "num_invalid": invalid_count,
            "valid_fraction": 0.0,
            "mean_vina": math.nan,
            "median_vina": math.nan,
            "std_vina": math.nan,
            "best_vina": math.nan,
            "top10_mean_vina": math.nan,
        }

    top_k = min(10, len(valid_scores))
    return {
        "num_valid": len(valid_scores),
        "num_invalid": invalid_count,
        "valid_fraction": len(valid_scores) / len(scores),
        "mean_vina": statistics.fmean(valid_scores),
        "median_vina": statistics.median(valid_scores),
        "std_vina": statistics.pstdev(valid_scores),
        "best_vina": min(valid_scores),
        "top10_mean_vina": statistics.fmean(sorted(valid_scores)[:top_k]),
    }


def write_csv(rows: Sequence[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_results(
    helms: Sequence[str],
    group_records: Sequence[dict[str, Any]],
    score_cache: Path,
    docking_mode: str,
    group_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_fields, scores_by_helm = load_score_cache(score_cache, docking_mode)
    missing = [helm for helm in dict.fromkeys(helms) if helm not in scores_by_helm]
    if missing:
        raise RuntimeError(
            f"Vina cache misses {len(missing)} sampled HELMs; first missing: {missing[0]}"
        )

    status_key = next(
        (field for field in source_fields if field.strip().lower() == "status"),
        None,
    )
    detail_key = next(
        (field for field in source_fields if field.strip().lower() == "detail"),
        None,
    )
    sample_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []

    for group_record in group_records:
        group_index = int(group_record["group_index"])
        start = (group_index - 1) * group_size
        end = start + group_size
        group_helms = list(helms[start:end])
        group_scores = [parse_score(scores_by_helm[helm]) for helm in group_helms]
        summary = summarize_group(group_scores)

        for local_index, (helm, vina_score) in enumerate(
            zip(group_helms, group_scores), start=1
        ):
            cached_row = scores_by_helm[helm]
            sample_rows.append(
                {
                    "global_sample_index": start + local_index,
                    "group_index": group_index,
                    "round": group_record.get("round", ""),
                    "stage": group_record.get("stage", ""),
                    "stage_epoch": group_record.get("stage_epoch", ""),
                    "sample_index_in_group": local_index,
                    "helm": helm,
                    "vina_score": vina_score,
                    "valid": vina_score != INVALID_VINA_SCORE,
                    "status": cached_row.get(status_key, "") if status_key else "",
                    "detail": cached_row.get(detail_key, "") if detail_key else "",
                    "checkpoint": group_record.get("checkpoint", ""),
                }
            )

        result_rows.append(
            {
                "group_index": group_index,
                "round": group_record.get("round", ""),
                "stage": group_record.get("stage", ""),
                "stage_epoch": group_record.get("stage_epoch", ""),
                "line_start": start + 1,
                "line_end": end,
                "num_samples": group_size,
                **summary,
                "checkpoint": group_record.get("checkpoint", ""),
            }
        )
    return sample_rows, result_rows


def main() -> None:
    args = parse_args()
    if args.group_size < 1 or args.expected_groups < 1:
        raise ValueError("--group_size and --expected_groups must both be >= 1")

    output_root = resolve_path(args.output_root)
    evaluation_root = resolve_path(args.evaluation_root)
    sample_root = (
        evaluation_root
        / args.run_name
        / args.case
        / args.arm
        / "checkpoint_curve_samples"
    )
    sample_file = (
        resolve_path(args.sample_file)
        if args.sample_file
        else sample_root / "all_24epochs_2400.txt"
    )
    group_manifest = (
        resolve_path(args.group_manifest)
        if args.group_manifest
        else sample_file.parent / "sample_groups.jsonl"
    )
    if not sample_file.exists():
        raise FileNotFoundError(f"Combined sample file not found: {sample_file}")

    expected_total = args.group_size * args.expected_groups
    helms = load_nonempty_lines(sample_file)
    if len(helms) != expected_total:
        raise RuntimeError(
            f"Expected exactly {expected_total} non-empty sample lines "
            f"({args.expected_groups} x {args.group_size}), found {len(helms)} in "
            f"{sample_file}"
        )
    group_records = load_group_manifest(
        group_manifest,
        args.expected_groups,
        args.group_size,
    )

    config_path = infer_vina_config(
        args.config,
        output_root,
        args.run_name,
        args.case,
        args.arm,
    )
    config = load_json(config_path)
    docking_mode = (
        args.docking_mode
        or str(config.get("dpo", {}).get("docking_mode", "flexible")).lower()
    )
    if docking_mode not in {"flexible", "rigid"}:
        raise ValueError(f"Unsupported docking mode: {docking_mode}")
    gpu_ids = parse_gpu_ids(args.gpu_ids)

    vina_dir = sample_file.parent / "vina"
    score_cache = vina_dir / f"all_24epochs_2400.{docking_mode}.vina.csv"
    per_sample_csv = vina_dir / "all_2400_sample_vina.csv"
    results_csv = vina_dir / "vina_24_results.csv"
    values_file = vina_dir / f"vina_24_{args.metric}.txt"
    evaluation_manifest = vina_dir / "evaluation_manifest.json"
    vina_command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/eval/export_train_vina_scores_multigpu.py"),
        "--config",
        str(config_path),
        "--sample_file",
        str(sample_file),
        "--vina_score_file",
        str(score_cache),
        "--gpu_ids",
        ",".join(gpu_ids),
        "--docking_mode",
        docking_mode,
    ]

    print(f"Run:                 {args.run_name}")
    print(f"Combined samples:    {sample_file}")
    print(f"Groups:              {args.expected_groups} x {args.group_size}")
    print(f"Total sample lines:  {len(helms)}")
    print(f"Unique HELMs:        {len(set(helms))}")
    print(f"Vina config:         {config_path}")
    print(f"Docking mode:        {docking_mode}")
    print(f"Uni-Dock GPUs:       {', '.join(gpu_ids)}")
    print(f"Primary metric:      {args.metric}")
    print(f"Vina output:         {vina_dir}")

    if args.dry_run:
        print("Dry run; Vina command:")
        print(" ".join(vina_command))
        return

    vina_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(vina_command, cwd=PROJECT_ROOT, check=True)
    sample_rows, result_rows = aggregate_results(
        helms,
        group_records,
        score_cache,
        docking_mode,
        args.group_size,
    )
    if len(sample_rows) != expected_total or len(result_rows) != args.expected_groups:
        raise RuntimeError(
            f"Aggregation verification failed: sample rows={len(sample_rows)}, "
            f"result rows={len(result_rows)}"
        )
    write_csv(sample_rows, per_sample_csv)
    write_csv(result_rows, results_csv)
    with values_file.open("w", encoding="utf-8") as handle:
        for row in result_rows:
            handle.write(f"{float(row[args.metric]):.8f}\n")

    save_json(
        {
            "status": "complete",
            "run_name": args.run_name,
            "case": args.case,
            "arm": args.arm,
            "sample_file": str(sample_file),
            "group_manifest": str(group_manifest),
            "group_size": args.group_size,
            "num_groups": len(result_rows),
            "num_samples": len(sample_rows),
            "num_unique_helms": len(set(helms)),
            "docking_mode": docking_mode,
            "gpu_ids": gpu_ids,
            "vina_config": str(config_path),
            "vina_cache": str(score_cache),
            "per_sample_results": str(per_sample_csv),
            "group_results": str(results_csv),
            "primary_metric": args.metric,
            "primary_values": str(values_file),
        },
        evaluation_manifest,
    )

    print("\nAll 24 Vina group results completed and verified.")
    print(f"24 primary values: {values_file}")
    print(f"Plotting CSV:      {results_csv}")
    print(f"Per-sample CSV:    {per_sample_csv}")


if __name__ == "__main__":
    main()
