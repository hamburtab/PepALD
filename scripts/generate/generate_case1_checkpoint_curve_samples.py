"""Generate a chronological checkpoint sample curve for WP-DPO ablations.

For every round, all numbered Elite-SFT checkpoints are followed by all
numbered DPO checkpoints, matching the actual training order. Epoch counts are
read from the generated ablation config, so case1 yields 24 groups/2,400 lines
and the current case2 schedule yields 26 groups/2,600 lines.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUTODL_ABLATION_ROOT = Path("/root/autodl-tmp/wp_dpo_ablation")

if AUTODL_ABLATION_ROOT.parent.is_dir():
    DEFAULT_OUTPUT_ROOT = AUTODL_ABLATION_ROOT / "outputs"
    DEFAULT_CHECKPOINT_ROOT = AUTODL_ABLATION_ROOT / "checkpoints"
    DEFAULT_EVALUATION_ROOT = AUTODL_ABLATION_ROOT / "evaluations"
else:
    DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "ablations" / "wp_dpo"
    DEFAULT_CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints" / "ablations" / "wp_dpo"
    DEFAULT_EVALUATION_ROOT = (
        PROJECT_ROOT / "outputs" / "ablations" / "wp_dpo_evaluations"
    )

DEFAULT_CONFIGS = {
    "case1": PROJECT_ROOT / "configs" / "training" / "dpo.json",
    "case2": PROJECT_ROOT / "configs" / "training" / "dpo_case2.json",
}


@dataclass(frozen=True)
class CheckpointSampleTask:
    group_index: int
    round_idx: int
    stage: str
    stage_epoch: int
    checkpoint: Path
    sample_file: Path
    config_file: Path
    log_file: Path

    @property
    def label(self) -> str:
        return (
            f"r{self.round_idx:02d}_{self.stage}_epoch{self.stage_epoch:02d}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate samples from every numbered Elite-SFT/DPO epoch checkpoint "
            "and concatenate all groups in chronological order."
        )
    )
    parser.add_argument("--run_name", required=True, help="Ablation run name.")
    parser.add_argument("--case", choices=["case1", "case2"], default="case1")
    parser.add_argument("--arm", choices=["standard_dpo"], default="standard_dpo")
    parser.add_argument(
        "--start_round",
        type=int,
        default=0,
        help="First round index to sample (default: 0).",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=8,
        help="Number of consecutive rounds to sample (default: 8).",
    )
    parser.add_argument("--samples_per_checkpoint", type=int, default=100)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Generation seed used for every checkpoint. Keeping it identical "
            "reduces sampling noise when comparing checkpoints."
        ),
    )
    parser.add_argument(
        "--gpu_ids",
        default="0,1,2",
        help="Comma-separated generation GPU IDs (default: 0,1,2).",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional generation config override; normally inferred from the run manifest.",
    )
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--checkpoint_root", default=str(DEFAULT_CHECKPOINT_ROOT))
    parser.add_argument("--evaluation_root", default=str(DEFAULT_EVALUATION_ROOT))
    parser.add_argument(
        "--force_regenerate",
        action="store_true",
        help="Regenerate per-checkpoint files even when a valid 100-line file exists.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate all scheduled checkpoints and print the plan without generating.",
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


def infer_generation_config(
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
        config_path = DEFAULT_CONFIGS[case_name]
        if manifest_path.exists():
            manifest = load_json(manifest_path)
            config_key = "standard_config" if arm == "standard_dpo" else "wp_config"
            if manifest.get(config_key):
                config_path = resolve_path(manifest[config_key])

    if not config_path.exists():
        raise FileNotFoundError(f"Generation config not found: {config_path}")
    return config_path


def build_tasks(
    checkpoint_root: Path,
    evaluation_dir: Path,
    run_name: str,
    case_name: str,
    arm: str,
    start_round: int,
    rounds: int,
    rounds_config: dict[str, Any],
) -> list[CheckpointSampleTask]:
    if rounds < 1:
        raise ValueError("--rounds must be >= 1")
    if start_round < 0:
        raise ValueError("--start_round must be >= 0")

    rounds_root = checkpoint_root / run_name / case_name / "rounds"
    per_checkpoint_dir = evaluation_dir / "per_checkpoint"
    config_dir = evaluation_dir / "generation_configs"
    log_dir = evaluation_dir / "logs"
    tasks: list[CheckpointSampleTask] = []

    for round_idx in range(start_round, start_round + rounds):
        round_dir = rounds_root / f"{arm}_r{round_idx}"
        elite_sft_epochs = (
            int(rounds_config.get("elite_sft_num_epochs", 1))
            if bool(rounds_config.get("elite_sft_enabled", False))
            else 0
        )
        dpo_epochs = get_round_epochs(rounds_config, round_idx)
        checkpoints_by_round = [
            (
                "elite_sft",
                epoch,
                Path(f"elite_sft/checkpoint_epoch_{epoch}.pt"),
            )
            for epoch in range(1, elite_sft_epochs + 1)
        ]
        checkpoints_by_round.extend(
            ("dpo", epoch, Path(f"dpo_epoch_{epoch}.pt"))
            for epoch in range(1, dpo_epochs + 1)
        )
        for stage, stage_epoch, relative_checkpoint in checkpoints_by_round:
            group_index = len(tasks) + 1
            label = f"r{round_idx:02d}_{stage}_epoch{stage_epoch:02d}"
            tasks.append(
                CheckpointSampleTask(
                    group_index=group_index,
                    round_idx=round_idx,
                    stage=stage,
                    stage_epoch=stage_epoch,
                    checkpoint=round_dir / relative_checkpoint,
                    sample_file=per_checkpoint_dir / f"{label}.txt",
                    config_file=config_dir / f"dpo_{case_name}_{label}.json",
                    log_file=log_dir / f"{label}.log",
                )
            )
    return tasks


def get_round_epochs(rounds_config: dict[str, Any], round_idx: int) -> int:
    """Mirror scripts/train/run_dpo_rounds.py scheduling exactly."""
    if round_idx == 0:
        bootstrap_epochs = rounds_config.get("bootstrap_num_epochs")
        if bootstrap_epochs is not None:
            return int(bootstrap_epochs)
        schedule = rounds_config.get("epochs_per_round")
        if isinstance(schedule, list) and schedule:
            return int(schedule[0])
        return int(rounds_config.get("max_epochs", 5))

    schedule = rounds_config.get("epochs_per_round")
    if isinstance(schedule, list) and schedule:
        index = min(round_idx - 1, len(schedule) - 1)
        return int(schedule[index])
    max_epochs = int(rounds_config.get("max_epochs", 5))
    min_epochs = int(rounds_config.get("min_epochs", 3))
    decay_start = int(rounds_config.get("epoch_decay_start_round", 4))
    if round_idx < decay_start:
        return max_epochs
    return max(min_epochs, max_epochs - (round_idx - decay_start + 1))


def validate_checkpoints(tasks: Sequence[CheckpointSampleTask]) -> None:
    missing = [task.checkpoint for task in tasks if not task.checkpoint.exists()]
    if missing:
        details = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            f"Missing {len(missing)}/{len(tasks)} required checkpoints:\n{details}"
        )


def prepare_task_configs(
    tasks: Sequence[CheckpointSampleTask],
    base_config: dict[str, Any],
    samples_per_checkpoint: int,
    seed: int,
) -> None:
    for task in tasks:
        config = deepcopy(base_config)
        config.setdefault("training", {})
        config.setdefault("generation", {})
        config["training"]["device"] = "cuda"
        config["generation"]["checkpoint_path"] = str(task.checkpoint)
        config["generation"]["num_samples"] = int(samples_per_checkpoint)
        config["generation"]["output_file"] = str(task.sample_file)
        config["generation"]["seed"] = int(seed)
        save_json(config, task.config_file)


def sample_file_is_complete(path: Path, expected_count: int) -> bool:
    return path.exists() and len(load_nonempty_lines(path)) == expected_count


def generate_one_task(
    task: CheckpointSampleTask,
    gpu_id: str,
    samples_per_checkpoint: int,
    force_regenerate: bool,
) -> str:
    if not force_regenerate and sample_file_is_complete(
        task.sample_file, samples_per_checkpoint
    ):
        return f"[reuse] group {task.group_index:02d} {task.label}"

    task.sample_file.parent.mkdir(parents=True, exist_ok=True)
    task.log_file.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/generate/generate_peptides.py"),
        "--config",
        str(task.config_file),
        "--num_samples",
        str(samples_per_checkpoint),
        "--output",
        str(task.sample_file),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    env["PYTHONUNBUFFERED"] = "1"

    with task.log_file.open("w", encoding="utf-8") as log_handle:
        log_handle.write(
            f"CUDA_VISIBLE_DEVICES={gpu_id} {' '.join(command)}\n"
            f"Checkpoint: {task.checkpoint}\n"
        )
        log_handle.flush()
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Generation failed for group {task.group_index} on GPU {gpu_id}; "
            f"see {task.log_file}"
        )
    if not sample_file_is_complete(task.sample_file, samples_per_checkpoint):
        actual_count = (
            len(load_nonempty_lines(task.sample_file))
            if task.sample_file.exists()
            else 0
        )
        raise RuntimeError(
            f"{task.label} expected {samples_per_checkpoint} generated samples, "
            f"found {actual_count}: {task.sample_file}"
        )
    return f"[done] group {task.group_index:02d} {task.label} on GPU {gpu_id}"


def generate_task_partition(
    tasks: Sequence[CheckpointSampleTask],
    gpu_id: str,
    samples_per_checkpoint: int,
    force_regenerate: bool,
) -> list[str]:
    messages = []
    for task in tasks:
        print(
            f"[GPU {gpu_id}] generating group {task.group_index:02d}: "
            f"{task.label}",
            flush=True,
        )
        messages.append(
            generate_one_task(
                task,
                gpu_id,
                samples_per_checkpoint,
                force_regenerate,
            )
        )
    return messages


def generate_all_tasks(
    tasks: Sequence[CheckpointSampleTask],
    gpu_ids: Sequence[str],
    samples_per_checkpoint: int,
    force_regenerate: bool,
) -> None:
    partitions = [list(tasks[index:: len(gpu_ids)]) for index in range(len(gpu_ids))]
    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
        futures = {
            executor.submit(
                generate_task_partition,
                partition,
                gpu_id,
                samples_per_checkpoint,
                force_regenerate,
            ): gpu_id
            for gpu_id, partition in zip(gpu_ids, partitions)
            if partition
        }
        for future in as_completed(futures):
            for message in future.result():
                print(message, flush=True)


def concatenate_samples(
    tasks: Sequence[CheckpointSampleTask],
    samples_per_checkpoint: int,
    combined_path: Path,
    manifest_path: Path,
    seed: int,
) -> list[dict[str, Any]]:
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = combined_path.with_suffix(combined_path.suffix + ".tmp")
    records: list[dict[str, Any]] = []
    next_line = 1

    with temporary_path.open("w", encoding="utf-8") as combined_handle:
        for task in tasks:
            samples = load_nonempty_lines(task.sample_file)
            if len(samples) != samples_per_checkpoint:
                raise RuntimeError(
                    f"Cannot concatenate {task.sample_file}: expected "
                    f"{samples_per_checkpoint} lines, found {len(samples)}"
                )
            line_start = next_line
            for sample in samples:
                combined_handle.write(f"{sample}\n")
            line_end = line_start + len(samples) - 1
            records.append(
                {
                    "group_index": task.group_index,
                    "round": task.round_idx,
                    "stage": task.stage,
                    "stage_epoch": task.stage_epoch,
                    "checkpoint": str(task.checkpoint),
                    "sample_file": str(task.sample_file),
                    "num_samples": len(samples),
                    "line_start": line_start,
                    "line_end": line_end,
                    "seed": int(seed),
                }
            )
            next_line = line_end + 1

    os.replace(temporary_path, combined_path)
    with manifest_path.open("w", encoding="utf-8") as manifest_handle:
        for record in records:
            manifest_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return records


def main() -> None:
    args = parse_args()
    if args.samples_per_checkpoint < 1:
        raise ValueError("--samples_per_checkpoint must be >= 1")

    gpu_ids = parse_gpu_ids(args.gpu_ids)
    output_root = resolve_path(args.output_root)
    checkpoint_root = resolve_path(args.checkpoint_root)
    evaluation_root = resolve_path(args.evaluation_root)
    evaluation_dir = (
        evaluation_root
        / args.run_name
        / args.case
        / args.arm
        / "checkpoint_curve_samples"
    )
    group_manifest_path = evaluation_dir / "sample_groups.jsonl"
    run_manifest_path = evaluation_dir / "generation_manifest.json"

    config_path = infer_generation_config(
        args.config,
        output_root,
        args.run_name,
        args.case,
        args.arm,
    )
    base_config = load_json(config_path)
    rounds_config = base_config.get("dpo_rounds", {})
    if not rounds_config:
        raise ValueError(f"Config has no dpo_rounds section: {config_path}")
    tasks = build_tasks(
        checkpoint_root,
        evaluation_dir,
        args.run_name,
        args.case,
        args.arm,
        args.start_round,
        args.rounds,
        rounds_config,
    )
    validate_checkpoints(tasks)

    expected_groups = len(tasks)
    expected_total = expected_groups * args.samples_per_checkpoint
    combined_path = (
        evaluation_dir / f"all_{expected_groups}epochs_{expected_total}.txt"
    )
    print(f"Run:                    {args.run_name}")
    print(f"Case / arm:             {args.case} / {args.arm}")
    print(
        f"Round range:            r{args.start_round}.."
        f"r{args.start_round + args.rounds - 1}"
    )
    print(f"Checkpoint groups:      {len(tasks)} across {args.rounds} rounds")
    print(f"Samples per checkpoint: {args.samples_per_checkpoint}")
    print(f"Expected total samples: {expected_total}")
    print(f"Chronological order:    all Elite-SFT epochs -> all DPO epochs per round")
    print(f"Generation GPUs:        {', '.join(gpu_ids)}")
    print(f"Generation config:      {config_path}")
    print(f"Combined output:        {combined_path}")

    if args.dry_run:
        for task in tasks:
            print(f"  {task.group_index:02d}: {task.checkpoint}")
        print("Dry run complete; no samples were generated.")
        return

    prepare_task_configs(
        tasks,
        base_config,
        args.samples_per_checkpoint,
        args.seed,
    )
    generate_all_tasks(
        tasks,
        gpu_ids,
        args.samples_per_checkpoint,
        args.force_regenerate,
    )
    records = concatenate_samples(
        tasks,
        args.samples_per_checkpoint,
        combined_path,
        group_manifest_path,
        args.seed,
    )
    actual_total = len(load_nonempty_lines(combined_path))
    if len(records) != expected_groups or actual_total != expected_total:
        raise RuntimeError(
            f"Final output verification failed: groups={len(records)}/{expected_groups}, "
            f"samples={actual_total}/{expected_total}"
        )

    save_json(
        {
            "status": "complete",
            "run_name": args.run_name,
            "case": args.case,
            "arm": args.arm,
            "rounds": args.rounds,
            "start_round": args.start_round,
            "end_round": args.start_round + args.rounds - 1,
            "checkpoint_order": "all Elite-SFT epochs, then all DPO epochs per round",
            "round_epoch_schedule": [
                {
                    "round": round_idx,
                    "elite_sft_epochs": (
                        int(rounds_config.get("elite_sft_num_epochs", 1))
                        if bool(rounds_config.get("elite_sft_enabled", False))
                        else 0
                    ),
                    "dpo_epochs": get_round_epochs(rounds_config, round_idx),
                }
                for round_idx in range(
                    args.start_round, args.start_round + args.rounds
                )
            ],
            "seed_per_checkpoint": args.seed,
            "generation_gpu_ids": gpu_ids,
            "generation_config": str(config_path),
            "num_groups": len(records),
            "samples_per_group": args.samples_per_checkpoint,
            "total_samples": actual_total,
            "combined_sample_file": str(combined_path),
            "sample_group_manifest": str(group_manifest_path),
        },
        run_manifest_path,
    )
    print("\nCheckpoint sample generation complete and verified.")
    print(f"Combined {actual_total:,} samples: {combined_path}")
    print(f"Group manifest:          {group_manifest_path}")
    print(f"Generation manifest:     {run_manifest_path}")


if __name__ == "__main__":
    main()
