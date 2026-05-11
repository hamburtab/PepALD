"""
Run multi-round DPO optimization for a configured target.

Round 0 is a bootstrap round:
  1. Use only the candidate pool in outputs/samples/dpo_train_data.
  2. Reuse or export permeability and Vina score CSVs for that pool.
  3. Train DPO from the initial checkpoint.

Rounds 1..N do:
  1. Generate HELM samples from the previous round checkpoint.
  2. Enforce the configured generated-sample post-processing.
  3. Merge samples into the round candidate pool.
  4. Export enabled reward score CSVs.
  5. Train DPO with reward/pair construction handled by train_dpo.py.

The round settings are read from the top-level `dpo_rounds` block in the config.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import shutil
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from pepar_diff.postprocess import is_head_tail_single_cycle


def parse_args():
    parser = argparse.ArgumentParser(description="Run automated multi-round DPO")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training/dpo_2axi.json",
        help="Path to DPO config with a top-level dpo_rounds block.",
    )
    parser.add_argument(
        "--start_round",
        type=int,
        default=0,
        help="First round index to run. Default: 0 (bootstrap round).",
    )
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=None,
        help=(
            "Override dpo_rounds.num_rounds. This is the last generated round index; "
            "with --start_round 0, the script runs bootstrap round 0 plus rounds 1..N."
        ),
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands and write round configs without executing them.",
    )
    return parser.parse_args()


def resolve_path(path_str: str | Path) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def command_prefix(value) -> list[str]:
    if value is None:
        return [sys.executable]
    if isinstance(value, list):
        return [str(v) for v in value]
    return shlex.split(str(value))


def format_cmd(cmd: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in cmd)


def run_command(cmd: Sequence[str], log_path: Path, dry_run: bool = False) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    text = format_cmd(cmd)
    print(f"\n$ {text}")
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] $ {text}\n")

    if dry_run:
        return

    proc = subprocess.run(list(cmd), cwd=PROJECT_ROOT)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}: {text}")


def load_helm_list(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def deduplicate_preserve_order(items: Iterable[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def write_helm_list(helms: Sequence[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for helm in helms:
            f.write(f"{helm}\n")


def filter_head_tail_samples(raw_path: Path, filtered_path: Path) -> list[str]:
    raw_helms = load_helm_list(raw_path)
    filtered = [helm for helm in raw_helms if is_head_tail_single_cycle(helm)]
    filtered = deduplicate_preserve_order(filtered)
    write_helm_list(filtered, filtered_path)
    print(
        f"Head-tail filter: {len(raw_helms)} raw -> "
        f"{len(filtered)} unique head-tail single cycles"
    )
    if not filtered:
        raise RuntimeError(f"No head-tail single-cycle samples survived filtering: {raw_path}")
    return filtered


def force_r1r2_cyclize_samples(
    raw_path: Path,
    filtered_path: Path,
    python_cmd: Sequence[str],
    log_path: Path,
    dry_run: bool = False,
) -> list[str]:
    run_command(
        list(python_cmd) + [
            "scripts/data/vina_filter_r1r2_cyclize.py",
            "--input", str(raw_path),
            "--output", str(filtered_path),
        ],
        log_path=log_path,
        dry_run=dry_run,
    )
    if dry_run:
        return []

    raw_count = len(load_helm_list(raw_path))
    cyclized = deduplicate_preserve_order(load_helm_list(filtered_path))
    write_helm_list(cyclized, filtered_path)
    print(
        f"R1/R2 cyclizer: {raw_count} raw -> "
        f"{len(cyclized)} unique forced head-tail cycles"
    )
    if not cyclized:
        raise RuntimeError(f"No samples survived R1/R2 head-tail cyclization: {raw_path}")
    return cyclized


def postprocess_generated_samples(
    mode: str,
    raw_path: Path,
    filtered_path: Path,
    python_cmd: Sequence[str],
    log_path: Path,
    dry_run: bool = False,
) -> list[str]:
    if mode == "force_r1r2_cyclize":
        return force_r1r2_cyclize_samples(
            raw_path,
            filtered_path,
            python_cmd=python_cmd,
            log_path=log_path,
            dry_run=dry_run,
        )
    if mode == "filter_head_tail":
        if dry_run:
            return []
        return filter_head_tail_samples(raw_path, filtered_path)
    raise ValueError(f"Unknown generated_postprocess mode: {mode}")


def merge_candidate_files(paths: Sequence[Path], output_path: Path) -> list[str]:
    helms = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Candidate source not found: {path}")
        helms.extend(load_helm_list(path))
    merged = deduplicate_preserve_order(helms)
    write_helm_list(merged, output_path)
    print(f"Merged candidates: {len(merged)} unique -> {output_path}")
    return merged


def detect_delimiter(header_line: str, path: Path) -> str:
    if path.suffix.lower() == ".tsv":
        return "\t"
    if "\t" in header_line and "," not in header_line:
        return "\t"
    return ","


def read_score_rows(path: Path, helm_key: str = "helm") -> tuple[list[str], dict[str, dict]]:
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
        actual_helm_key = field_map.get(helm_key.lower())
        if actual_helm_key is None:
            return fieldnames, {}
        rows = {}
        for row in reader:
            helm = (row.get(actual_helm_key) or "").strip()
            if helm:
                rows[helm] = row
    return fieldnames, rows


def score_file_covers(path: Path, helms: Sequence[str], score_columns: Sequence[str]) -> bool:
    fieldnames, rows = read_score_rows(path)
    if not fieldnames or not rows:
        return False
    field_map = {name.strip().lower(): name for name in fieldnames if name}
    score_key = None
    for col in score_columns:
        if col.lower() in field_map:
            score_key = field_map[col.lower()]
            break
    if score_key is None:
        return False

    for helm in helms:
        row = rows.get(helm)
        if row is None:
            return False
        try:
            score = float((row.get(score_key) or "").strip())
        except ValueError:
            return False
        if not math.isfinite(score):
            return False
    return True


def write_subset_csv(source_csv: Path, target_csv: Path, helms: Sequence[str]) -> None:
    fieldnames, rows = read_score_rows(source_csv)
    if not fieldnames:
        raise ValueError(f"Cannot read score CSV: {source_csv}")
    target_csv.parent.mkdir(parents=True, exist_ok=True)

    unique_helms = deduplicate_preserve_order(helms)
    missing = [helm for helm in unique_helms if helm not in rows]
    if missing:
        raise ValueError(
            f"{source_csv} does not cover {len(missing)} requested HELM sequences. "
            f"First missing: {missing[0]}"
        )

    with open(target_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for helm in unique_helms:
            writer.writerow(rows[helm])
    print(f"Wrote subset score CSV: {target_csv}")


def merge_score_csvs(source_csvs: Sequence[Path], target_csv: Path, helms: Sequence[str]) -> bool:
    fieldnames: list[str] | None = None
    merged_rows: dict[str, dict] = {}
    for source_csv in source_csvs:
        if source_csv is None or not source_csv.exists():
            continue
        source_fieldnames, rows = read_score_rows(source_csv)
        if not source_fieldnames:
            continue
        if fieldnames is None:
            fieldnames = source_fieldnames
        merged_rows.update(rows)

    if fieldnames is None:
        return False

    unique_helms = deduplicate_preserve_order(helms)
    missing = [helm for helm in unique_helms if helm not in merged_rows]
    if missing:
        return False

    target_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(target_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for helm in unique_helms:
            writer.writerow(merged_rows[helm])
    print(f"Merged score CSV: {target_csv}")
    return True


def copy_seed_cache(seed_path: Path | None, target_path: Path, dry_run: bool = False) -> None:
    if target_path.exists():
        print(f"Keeping existing score cache: {target_path}")
        return
    if seed_path is None or not seed_path.exists():
        print(f"No seed cache available for: {target_path}")
        return
    if dry_run:
        print(f"Would seed score cache: {seed_path} -> {target_path}")
        return
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(seed_path, target_path)
    print(f"Seeded score cache: {seed_path} -> {target_path}")


def get_round_epochs(rounds_cfg: dict, round_idx: int) -> int:
    if round_idx == 0:
        bootstrap_epochs = rounds_cfg.get("bootstrap_num_epochs")
        if bootstrap_epochs is not None:
            return int(bootstrap_epochs)
        schedule = rounds_cfg.get("epochs_per_round")
        if isinstance(schedule, list) and schedule:
            return int(schedule[0])
        return int(rounds_cfg.get("max_epochs", 5))

    schedule = rounds_cfg.get("epochs_per_round")
    if isinstance(schedule, list) and schedule:
        index = min(round_idx - 1, len(schedule) - 1)
        return int(schedule[index])

    max_epochs = int(rounds_cfg.get("max_epochs", 5))
    min_epochs = int(rounds_cfg.get("min_epochs", 3))
    decay_start = int(rounds_cfg.get("epoch_decay_start_round", 4))
    if round_idx < decay_start:
        return max_epochs
    return max(min_epochs, max_epochs - (round_idx - decay_start + 1))


def build_round_config(
    base_config: dict,
    round_idx: int,
    round_dir: Path,
    checkpoint_dir: Path,
    previous_checkpoint: Path,
    generated_path: Path,
    candidates_path: Path,
    perm_score_file: Path | None,
    vina_score_file: Path,
    epochs: int,
    num_samples: int | None,
) -> Path:
    cfg = deepcopy(base_config)
    cfg.setdefault("training", {})
    cfg.setdefault("generation", {})
    cfg.setdefault("dpo", {})

    cfg["training"]["pretrained_checkpoint"] = str(previous_checkpoint)
    cfg["training"]["checkpoint_dir"] = str(checkpoint_dir)

    cfg["generation"]["checkpoint_path"] = str(previous_checkpoint)
    cfg["generation"]["output_file"] = str(generated_path)
    if num_samples is not None:
        cfg["generation"]["num_samples"] = int(num_samples)

    cfg["dpo"]["sample_files"] = [str(candidates_path)]
    cfg["dpo"]["perm_score_file"] = None if perm_score_file is None else str(perm_score_file)
    cfg["dpo"]["vina_score_file"] = str(vina_score_file)
    cfg["dpo"]["num_epochs"] = int(epochs)

    config_path = round_dir / f"dpo_2axi_round{round_idx}.json"
    save_json(cfg, config_path)
    return config_path


def main():
    args = parse_args()
    config_path = resolve_path(args.config)
    base_config = load_json(config_path)
    dpo_cfg = base_config.get("dpo", {})
    rounds_cfg = base_config.get("dpo_rounds", {})
    if not rounds_cfg:
        raise ValueError(
            f"{config_path} does not define a top-level dpo_rounds block."
        )

    num_rounds = int(
        args.num_rounds if args.num_rounds is not None else rounds_cfg.get("num_rounds", 1)
    )
    if num_rounds < args.start_round:
        raise ValueError("num_rounds must be >= start_round")

    python_cmd = command_prefix(rounds_cfg.get("python"))
    perm_python_cmd = command_prefix(rounds_cfg.get("permeability_python", rounds_cfg.get("python")))
    use_permeability = float(dpo_cfg.get("reward_w_perm", 0.5)) > 0.0
    generated_postprocess = str(rounds_cfg.get("generated_postprocess", "filter_head_tail"))

    output_root = resolve_path(rounds_cfg.get("output_root", "outputs/samples/dpo_rounds"))
    run_name = str(rounds_cfg.get("run_name", "2axi"))
    checkpoint_root = resolve_path(
        rounds_cfg.get(
            "checkpoint_root",
            str(Path(base_config.get("training", {}).get("checkpoint_dir", "checkpoints"))),
        )
    )

    base_candidate_file = resolve_path(
        rounds_cfg.get(
            "base_candidate_file",
            (dpo_cfg.get("sample_files") or [dpo_cfg.get("sample_file")])[0],
        )
    )
    base_perm_raw = rounds_cfg.get("base_perm_score_file", dpo_cfg.get("perm_score_file"))
    base_perm_file = resolve_path(base_perm_raw) if base_perm_raw else None
    base_vina_raw = rounds_cfg.get("base_vina_score_file", dpo_cfg.get("vina_score_file"))
    base_vina_file = resolve_path(base_vina_raw) if base_vina_raw else None
    previous_checkpoint = resolve_path(
        rounds_cfg.get(
            "initial_checkpoint",
            base_config.get("generation", {}).get("checkpoint_path"),
        )
    )

    base_merge_rounds = int(rounds_cfg.get("base_merge_rounds", 3))
    carry_forward = bool(rounds_cfg.get("carry_forward_previous_candidates", True))
    num_samples_per_round = rounds_cfg.get("num_samples_per_round")
    if num_samples_per_round is not None:
        num_samples_per_round = int(num_samples_per_round)

    previous_candidates: Path | None = None
    previous_vina: Path | None = None
    previous_perm: Path | None = None

    # Resuming from a later round needs the previous round artifacts.
    if args.start_round > 0:
        prev_idx = args.start_round - 1
        prev_dir = output_root / f"{run_name}_r{prev_idx}"
        previous_candidates = prev_dir / "candidates.txt"
        previous_vina = prev_dir / "candidates.2axi.vina.csv"
        previous_perm = prev_dir / "candidates.perm.csv" if use_permeability else None
        previous_checkpoint = checkpoint_root / f"{run_name}_r{prev_idx}" / "dpo_latest.pt"
        if not args.dry_run:
            required_paths = [previous_candidates, previous_vina, previous_checkpoint]
            if use_permeability:
                required_paths.append(previous_perm)
            for path in required_paths:
                if path is not None and not path.exists():
                    raise FileNotFoundError(f"Cannot resume; previous artifact missing: {path}")
    elif not args.dry_run and not previous_checkpoint.exists():
        raise FileNotFoundError(
            f"Initial checkpoint not found: {previous_checkpoint}. "
            "Set dpo_rounds.initial_checkpoint to an existing CE/pretrained checkpoint "
            "before launching bootstrap round 0."
        )

    for round_idx in range(args.start_round, num_rounds + 1):
        round_dir = output_root / f"{run_name}_r{round_idx}"
        round_dir.mkdir(parents=True, exist_ok=True)
        log_path = round_dir / "round_commands.log"

        checkpoint_dir = checkpoint_root / f"{run_name}_r{round_idx}"
        generated_raw = round_dir / "generated.txt"
        generated_filtered = (
            round_dir / "generated_r1r2_cyclized.txt"
            if generated_postprocess == "force_r1r2_cyclize"
            else round_dir / "generated_head_tail.txt"
        )
        candidates_path = round_dir / "candidates.txt"
        candidates_perm = round_dir / "candidates.perm.csv" if use_permeability else None
        candidates_vina = round_dir / "candidates.2axi.vina.csv"
        generated_perm = round_dir / "generated.perm.csv" if use_permeability else None
        generated_vina = round_dir / "generated.2axi.vina.csv"
        epochs = get_round_epochs(rounds_cfg, round_idx)

        print("\n" + "=" * 80)
        if round_idx == 0:
            if num_rounds > 0:
                print(f"DPO bootstrap round 0 (then generated rounds 1..{num_rounds})")
            else:
                print("DPO bootstrap round 0 (no generated rounds requested)")
        else:
            print(f"DPO generated round {round_idx}/{num_rounds}")
        print("=" * 80)
        print(f"Previous checkpoint: {previous_checkpoint}")
        print(f"Round directory:     {round_dir}")
        print(f"Checkpoint dir:      {checkpoint_dir}")
        print(f"Epochs this round:   {epochs}")
        print(f"Permeability reward: {'enabled' if use_permeability else 'disabled'}")
        print(f"Generated postproc:  {generated_postprocess}")
        input_checkpoint = previous_checkpoint

        round_config = build_round_config(
            base_config=base_config,
            round_idx=round_idx,
            round_dir=round_dir,
            checkpoint_dir=checkpoint_dir,
            previous_checkpoint=previous_checkpoint,
            generated_path=generated_raw,
            candidates_path=candidates_path,
            perm_score_file=candidates_perm,
            vina_score_file=candidates_vina,
            epochs=epochs,
            num_samples=num_samples_per_round,
        )
        print(f"Round config:        {round_config}")

        is_bootstrap_round = round_idx == 0
        if is_bootstrap_round:
            candidate_sources = [base_candidate_file]
            generated_helms: list[str] = []
            generated_filtered = None
            generated_perm = None
            generated_vina = None
            print("Bootstrap round 0: using only the base candidate pool, no generation step.")
        else:
            run_command(
                python_cmd + [
                    "scripts/generate/generate_peptides.py",
                    "--config", str(round_config),
                    "--output", str(generated_raw),
                ],
                log_path=log_path,
                dry_run=args.dry_run,
            )

            if args.dry_run:
                generated_helms = []
            else:
                generated_helms = postprocess_generated_samples(
                    generated_postprocess,
                    generated_raw,
                    generated_filtered,
                    python_cmd=python_cmd,
                    log_path=log_path,
                    dry_run=False,
                )

            candidate_sources = []
            if carry_forward and previous_candidates is not None:
                candidate_sources.append(previous_candidates)
            if round_idx <= base_merge_rounds:
                candidate_sources.append(base_candidate_file)
            candidate_sources.append(generated_filtered)

        if not args.dry_run:
            print("Candidate sources:")
            for source in candidate_sources:
                print(f"  - {source}")
            candidate_helms = merge_candidate_files(candidate_sources, candidates_path)
        else:
            print("Candidate sources (dry-run):")
            for source in candidate_sources:
                print(f"  - {source}")
            candidate_helms = []

        if not use_permeability:
            print("Permeability reward disabled (reward_w_perm=0); skipping permeability scoring.")
        elif args.dry_run:
            run_command(
                perm_python_cmd + [
                    "scripts/eval/evaluate_permeability_scores.py",
                    "--input", str(candidates_path),
                    "--output", str(candidates_perm),
                ],
                log_path=log_path,
                dry_run=args.dry_run,
            )
        elif score_file_covers(candidates_perm, candidate_helms, ["permeability", "perm_score", "score"]):
            print(f"Permeability cache already covers candidates: {candidates_perm}")
        else:
            if is_bootstrap_round:
                merged = merge_score_csvs(
                    [base_perm_file],
                    candidates_perm,
                    candidate_helms,
                )
                if not merged or not score_file_covers(candidates_perm, candidate_helms, ["permeability", "perm_score", "score"]):
                    print("Bootstrap permeability cache incomplete; scoring base candidate pool.")
                    run_command(
                        perm_python_cmd + [
                            "scripts/eval/evaluate_permeability_scores.py",
                            "--input", str(candidates_path),
                            "--output", str(candidates_perm),
                        ],
                        log_path=log_path,
                        dry_run=False,
                    )
            else:
                if not score_file_covers(generated_perm, generated_helms, ["permeability", "perm_score", "score"]):
                    run_command(
                        perm_python_cmd + [
                            "scripts/eval/evaluate_permeability_scores.py",
                            "--input", str(generated_filtered),
                            "--output", str(generated_perm),
                        ],
                        log_path=log_path,
                        dry_run=False,
                    )

                seed_perm = previous_perm if previous_perm is not None else base_perm_file
                merged = merge_score_csvs(
                    [seed_perm, generated_perm],
                    candidates_perm,
                    candidate_helms,
                )
                if not merged or not score_file_covers(candidates_perm, candidate_helms, ["permeability", "perm_score", "score"]):
                    print("Incremental permeability merge incomplete; scoring full candidate pool.")
                    run_command(
                        perm_python_cmd + [
                            "scripts/eval/evaluate_permeability_scores.py",
                            "--input", str(candidates_path),
                            "--output", str(candidates_perm),
                        ],
                        log_path=log_path,
                        dry_run=False,
                    )

        seed_vina = previous_vina if previous_vina is not None else base_vina_file
        copy_seed_cache(seed_vina, candidates_vina, dry_run=args.dry_run)
        run_command(
            python_cmd + [
                "scripts/eval/export_train_vina_scores.py",
                "--config", str(round_config),
                "--sample_file", str(candidates_path),
                "--vina_score_file", str(candidates_vina),
            ],
            log_path=log_path,
            dry_run=args.dry_run,
        )

        if not is_bootstrap_round and not args.dry_run:
            if use_permeability:
                write_subset_csv(candidates_perm, generated_perm, generated_helms)
            write_subset_csv(candidates_vina, generated_vina, generated_helms)

        run_command(
            python_cmd + [
                "scripts/train/train_dpo.py",
                "--config", str(round_config),
            ],
            log_path=log_path,
            dry_run=args.dry_run,
        )

        previous_checkpoint = checkpoint_dir / "dpo_latest.pt"
        if not args.dry_run and not previous_checkpoint.exists():
            raise FileNotFoundError(f"Round training did not produce checkpoint: {previous_checkpoint}")

        previous_candidates = candidates_path
        previous_vina = candidates_vina
        previous_perm = candidates_perm if use_permeability else None

        summary = {
            "round": round_idx,
            "mode": "bootstrap_base_only" if is_bootstrap_round else "generate_merge_train",
            "round_dir": str(round_dir),
            "round_config": str(round_config),
            "previous_checkpoint_for_generation": str(input_checkpoint),
            "checkpoint": str(previous_checkpoint),
            "candidate_sources": [str(p) for p in candidate_sources],
            "generated": None if generated_filtered is None else str(generated_filtered),
            "generated_postprocess": generated_postprocess,
            "generated_perm": None if generated_perm is None else str(generated_perm),
            "generated_vina": None if generated_vina is None else str(generated_vina),
            "candidates": str(candidates_path),
            "candidates_perm": None if candidates_perm is None else str(candidates_perm),
            "candidates_vina": str(candidates_vina),
            "epochs": epochs,
        }
        save_json(summary, round_dir / "round_summary.json")
        print(f"Round {round_idx} complete. Summary: {round_dir / 'round_summary.json'}")

    print("\nAll requested DPO rounds completed.")


if __name__ == "__main__":
    main()
