"""
Export or resume cached Vina docking scores using one Uni-Dock worker per GPU.

This wrapper keeps the existing single-GPU scorer untouched. It splits only the
missing HELM sequences into shard files, launches one child scorer per GPU with
CUDA_VISIBLE_DEVICES set, then merges the shard CSVs back into the requested
cache file.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_FIELDNAMES = ["helm", "vina_score", "status", "detail", "docking_mode"]
INVALID_SCORE = 0.0


def resolve_path(path_str: str | Path) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def parse_args():
    parser = argparse.ArgumentParser(description="Export cached Vina docking scores on multiple GPUs")
    parser.add_argument(
        "--config", type=str, default="configs/training/dpo.json",
        help="Path to DPO config file."
    )
    parser.add_argument(
        "--sample_file", type=str, default=None,
        help="Optional single HELM file override."
    )
    parser.add_argument(
        "--vina_score_file", type=str, default=None,
        help="Optional merged Vina score cache CSV override."
    )
    parser.add_argument(
        "--gpu_ids", type=str, default=None,
        help="Comma-separated GPU IDs. If omitted, reads dpo.unidock_gpu_ids from config."
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Only score the first N deduplicated HELM sequences."
    )
    parser.add_argument(
        "--docking_mode",
        choices=["flexible", "rigid"],
        default=None,
        help=(
            "Ligand docking mode passed to each shard. Defaults to dpo.docking_mode or flexible."
        )
    )
    return parser.parse_args()


def parse_gpu_ids(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [part.strip() for part in str(value).split(",") if part.strip()]


def detect_delimiter(header_line: str, path: Path) -> str:
    if path.suffix.lower() == ".tsv":
        return "\t"
    if "\t" in header_line and "," not in header_line:
        return "\t"
    return ","


def load_helm_list(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def deduplicate_preserve_order(items: Sequence[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def load_candidate_helms(sample_files: Sequence[str]) -> tuple[list[str], list[str], list[Path]]:
    all_helms = []
    source_labels = []
    resolved_paths = []

    for sample_file in sample_files:
        sample_path = resolve_path(sample_file)
        if not sample_path.exists():
            raise FileNotFoundError(f"Sample file not found: {sample_path}")

        helms = load_helm_list(sample_path)
        if not helms:
            raise ValueError(f"No valid HELM sequences found in sample file: {sample_path}")

        all_helms.extend(helms)
        source_labels.extend([sample_path.stem] * len(helms))
        resolved_paths.append(sample_path)
        print(f"Loaded {len(helms)} HELM sequences from {sample_path}")

    deduped_helms = deduplicate_preserve_order(all_helms)
    if len(deduped_helms) == len(all_helms):
        print("candidate files dedup: no duplicates found")
        return all_helms, source_labels, resolved_paths

    source_by_helm = {}
    for helm, source in zip(all_helms, source_labels):
        source_by_helm.setdefault(helm, source)
    deduped_sources = [source_by_helm[helm] for helm in deduped_helms]
    print(
        f"candidate files dedup: {len(all_helms)} -> {len(deduped_helms)} unique "
        f"({len(all_helms) - len(deduped_helms)} duplicates removed)"
    )
    return deduped_helms, deduped_sources, resolved_paths


def read_score_rows(path: Path) -> tuple[list[str], dict[str, dict]]:
    if not path.exists():
        return [], {}
    with open(path, "r", newline="", encoding="utf-8") as f:
        header_line = f.readline()
        if not header_line:
            return [], {}
        delimiter = detect_delimiter(header_line, path)
        f.seek(0)
        reader = csv.DictReader(f, delimiter=delimiter)
        if reader.fieldnames is None:
            return [], {}
        fieldnames = list(reader.fieldnames)
        field_map = {name.strip().lower(): name for name in fieldnames if name}
        helm_key = field_map.get("helm")
        if helm_key is None:
            return fieldnames, {}

        rows = {}
        for row in reader:
            helm = (row.get(helm_key) or "").strip()
            if helm:
                rows[helm] = row
    return fieldnames, rows


def load_cached_vina_scores(
    all_helms: Sequence[str],
    score_file: str | Path,
    docking_mode: str | None = None,
):
    score_path = resolve_path(score_file)
    missing_indices = list(range(len(all_helms)))
    status_counter = Counter()

    if not score_path.exists():
        print(f"Vina cache not found yet: {score_path}")
        return None, missing_indices, status_counter

    fieldnames, rows = read_score_rows(score_path)
    if not fieldnames:
        print(f"Vina cache exists but is empty or unreadable: {score_path}")
        return None, missing_indices, status_counter

    field_map = {name.strip().lower(): name for name in fieldnames if name}
    status_key = field_map.get("status")
    mode_key = field_map.get("docking_mode")
    if docking_mode is not None and mode_key is None:
        print(
            f"Vina cache has no docking_mode column, so it will not be reused for "
            f"{docking_mode} scoring: {score_path}"
        )
        return None, missing_indices, status_counter

    missing_indices = []
    mismatched_rows = 0
    for idx, helm in enumerate(all_helms):
        row = rows.get(helm)
        if row is None:
            missing_indices.append(idx)
            continue
        if docking_mode is not None:
            row_mode = (row.get(mode_key) or "").strip().lower()
            if row_mode != docking_mode:
                mismatched_rows += 1
                missing_indices.append(idx)
                continue
        status_counter[(row.get(status_key) or "unknown").strip() if status_key else "unknown"] += 1

    cached = len(all_helms) - len(missing_indices)
    mode_text = f" {docking_mode}" if docking_mode else ""
    extra = f" ({mismatched_rows} rows from other docking_mode ignored)" if mismatched_rows else ""
    print(f"Loaded cached{mode_text} Vina scores for {cached}/{len(all_helms)} HELM sequences from {score_path}{extra}")
    if status_counter:
        status_text = ", ".join(f"{k}={v}" for k, v in sorted(status_counter.items()))
        print(f"  Cached docking status breakdown: {status_text}")

    return None, missing_indices, status_counter


def write_helm_file(helms: Sequence[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for helm in helms:
            f.write(f"{helm}\n")


def write_merged_cache(cache_path: Path, all_helms: Sequence[str], source_csvs: Sequence[Path]) -> None:
    fieldnames = list(DEFAULT_FIELDNAMES)
    rows_by_helm: dict[str, dict] = {}

    for source_csv in source_csvs:
        source_fields, rows = read_score_rows(source_csv)
        for field in source_fields:
            if field not in fieldnames:
                fieldnames.append(field)
        rows_by_helm.update(rows)

    missing = [helm for helm in all_helms if helm not in rows_by_helm]
    if missing:
        raise RuntimeError(
            f"Merged Vina cache is still missing {len(missing)} HELM sequences. "
            f"First missing: {missing[0]}"
        )

    tmp_path = cache_path.with_name(cache_path.name + ".tmp")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for helm in all_helms:
            row = dict(rows_by_helm[helm])
            for field in fieldnames:
                row.setdefault(field, "")
            writer.writerow({field: row.get(field, "") for field in fieldnames})

    os.replace(tmp_path, cache_path)


def print_vina_summary_from_cache(
    helms: Sequence[str],
    cache_path: Path,
    docking_mode: str | None = None,
) -> None:
    fieldnames, rows = read_score_rows(cache_path)
    if not fieldnames:
        print("\n=== Vina Summary ===")
        print(f"  Unique samples scored:   {len(helms)}")
        print("  Valid docking scores:    0")
        print(f"  Invalid / failed scores: {len(helms)}")
        print("  No valid Vina scores were produced.")
        return

    field_map = {name.strip().lower(): name for name in fieldnames if name}
    score_key = field_map.get("vina_score") or field_map.get("score")
    if score_key is None:
        raise ValueError(f"Vina cache lacks a vina_score/score column: {cache_path}")
    mode_key = field_map.get("docking_mode")

    scores = []
    for helm in helms:
        row = rows.get(helm, {})
        if docking_mode is not None:
            row_mode = (row.get(mode_key) or "").strip().lower() if mode_key else ""
            if row_mode != docking_mode:
                scores.append(INVALID_SCORE)
                continue
        try:
            score = float((row.get(score_key) or "").strip())
        except ValueError:
            score = INVALID_SCORE
        scores.append(score)

    valid_pairs = [(idx, score) for idx, score in enumerate(scores) if score != INVALID_SCORE]
    invalid_count = len(scores) - len(valid_pairs)

    title_suffix = f" ({docking_mode})" if docking_mode else ""
    print(f"\n=== Vina Summary{title_suffix} ===")
    print(f"  Unique samples scored:   {len(helms)}")
    print(f"  Valid docking scores:    {len(valid_pairs)}")
    print(f"  Invalid / failed scores: {invalid_count}")

    if not valid_pairs:
        print("  No valid Vina scores were produced.")
        return

    valid_scores = [score for _, score in valid_pairs]
    best_idx, best_score = min(valid_pairs, key=lambda item: item[1])
    top_k = min(10, len(valid_scores))
    top_mean = sum(sorted(valid_scores)[:top_k]) / top_k

    mean_score = sum(valid_scores) / len(valid_scores)
    sorted_scores = sorted(valid_scores)
    mid = len(sorted_scores) // 2
    if len(sorted_scores) % 2:
        median_score = sorted_scores[mid]
    else:
        median_score = (sorted_scores[mid - 1] + sorted_scores[mid]) / 2.0
    std_score = (sum((score - mean_score) ** 2 for score in valid_scores) / len(valid_scores)) ** 0.5

    print(f"  Mean Vina:               {mean_score:.4f}")
    print(f"  Median Vina:             {median_score:.4f}")
    print(f"  Std Vina:                {std_score:.4f}")
    print(f"  Best Vina:               {best_score:.4f}")
    print(f"  Best HELM:               {helms[best_idx]}")
    print(f"  Top-{top_k} mean Vina:        {top_mean:.4f}")


def main():
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        full_config = json.load(f)
    dpo_cfg = full_config.get("dpo", {})

    gpu_ids = parse_gpu_ids(args.gpu_ids)
    if not gpu_ids:
        gpu_ids = parse_gpu_ids(dpo_cfg.get("unidock_gpu_ids"))
    if not gpu_ids:
        raise ValueError("No GPU IDs configured. Set --gpu_ids or dpo.unidock_gpu_ids.")

    sample_files = []
    if args.sample_file:
        sample_files = [args.sample_file]
    elif dpo_cfg.get("sample_files"):
        sample_files = list(dpo_cfg.get("sample_files", []))
    elif dpo_cfg.get("sample_file"):
        sample_files = [dpo_cfg.get("sample_file")]
    if not sample_files:
        raise ValueError("No candidate file specified. Use --sample_file or set dpo.sample_files / dpo.sample_file.")

    vina_score_file = (
        args.vina_score_file
        or dpo_cfg.get("vina_score_file")
        or "outputs/samples/case1/train_candidates/candidates.case1.vina.csv"
    )
    docking_mode = args.docking_mode or str(dpo_cfg.get("docking_mode", "flexible")).lower()
    if docking_mode not in {"flexible", "rigid"}:
        raise ValueError(f"docking_mode must be flexible or rigid, got: {docking_mode}")
    cache_path = resolve_path(vina_score_file)

    print(f"Loading config from: {args.config}")
    print(f"Multi-GPU Uni-Dock GPUs: {', '.join(gpu_ids)}")
    print(f"Docking mode: {docking_mode}")
    all_helms, source_labels, sample_paths = load_candidate_helms(sample_files)
    print(f"Using candidate set from: {', '.join(str(p) for p in sample_paths)}")
    print(f"Loaded {len(all_helms)} unique HELM sequences from {len(sample_paths)} file(s)")

    if args.max_samples is not None and args.max_samples < len(all_helms):
        all_helms = all_helms[:args.max_samples]
        source_labels = source_labels[:args.max_samples]
        print(f"Using first {args.max_samples} unique HELM sequences")

    _, missing_indices, _ = load_cached_vina_scores(
        all_helms,
        str(cache_path),
        docking_mode=docking_mode,
    )
    if not missing_indices:
        print("All candidate HELM sequences already have cached Vina scores; nothing to dock.")
        print_vina_summary_from_cache(all_helms, cache_path, docking_mode=docking_mode)
        return

    missing_helms = [all_helms[i] for i in missing_indices]
    shard_root = cache_path.parent / f"{cache_path.stem}.multigpu_shards"
    shard_root.mkdir(parents=True, exist_ok=True)

    shards: list[tuple[str, Path, Path, Path, int]] = []
    for shard_idx, gpu_id in enumerate(gpu_ids):
        shard_helms = missing_helms[shard_idx::len(gpu_ids)]
        if not shard_helms:
            continue
        sample_path = shard_root / f"shard_{shard_idx:02d}_gpu{gpu_id}.txt"
        score_path = shard_root / f"shard_{shard_idx:02d}_gpu{gpu_id}.vina.csv"
        log_path = shard_root / f"shard_{shard_idx:02d}_gpu{gpu_id}.log"
        write_helm_file(shard_helms, sample_path)
        shards.append((gpu_id, sample_path, score_path, log_path, len(shard_helms)))

    print(
        f"Docking {len(missing_helms)} missing HELM sequences across "
        f"{len(shards)} GPU shard(s). Shard files: {shard_root}"
    )

    processes = []
    log_handles = []
    try:
        for gpu_id, sample_path, score_path, log_path, shard_count in shards:
            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "scripts/eval/export_train_vina_scores.py"),
                "--config", str(args.config),
                "--sample_file", str(sample_path),
                "--vina_score_file", str(score_path),
                "--docking_mode", docking_mode,
            ]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            env["PYTHONUNBUFFERED"] = "1"

            log_handle = open(log_path, "a", encoding="utf-8")
            log_handles.append(log_handle)
            log_handle.write(
                f"\nCUDA_VISIBLE_DEVICES={gpu_id} {' '.join(cmd)}\n"
                f"Shard HELM count: {shard_count}\n"
            )
            log_handle.flush()

            print(f"Launching GPU {gpu_id}: {shard_count} HELM -> {score_path}")
            proc = subprocess.Popen(
                cmd,
                cwd=PROJECT_ROOT,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            processes.append((gpu_id, proc, log_path))

        failures = []
        for gpu_id, proc, log_path in processes:
            returncode = proc.wait()
            if returncode != 0:
                failures.append((gpu_id, returncode, log_path))

        if failures:
            details = "; ".join(
                f"GPU {gpu_id} exit={returncode}, log={log_path}"
                for gpu_id, returncode, log_path in failures
            )
            raise RuntimeError(f"One or more Uni-Dock shards failed: {details}")

    except KeyboardInterrupt:
        print("Interrupted; terminating running Uni-Dock shard processes...")
        for _, proc, _ in processes:
            if proc.poll() is None:
                proc.terminate()
        raise
    finally:
        for handle in log_handles:
            handle.close()

    source_csvs = [cache_path] + [score_path for _, _, score_path, _, _ in shards]
    write_merged_cache(cache_path, all_helms, source_csvs)
    print(f"Merged multi-GPU Vina cache: {cache_path}")

    load_cached_vina_scores(all_helms, str(cache_path), docking_mode=docking_mode)
    print_vina_summary_from_cache(all_helms, cache_path, docking_mode=docking_mode)
    print("\nDone. (Multi-GPU Vina scoring cache exported / resumed.)")


if __name__ == "__main__":
    main()
