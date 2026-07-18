"""
Generate peptides with one sampler process per GPU.

The existing single-GPU generator remains the source of truth. This wrapper
splits generation.num_samples across GPUs, launches child generator processes
with CUDA_VISIBLE_DEVICES set, then concatenates shard outputs into the target
generated.txt.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="Generate peptides on multiple GPUs")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to config file."
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Optional merged output HELM file. Defaults to generation.output_file in the config."
    )
    parser.add_argument(
        "--num_samples", type=int, default=None,
        help="Optional total number of samples. Defaults to generation.num_samples."
    )
    parser.add_argument(
        "--gpu_ids", type=str, default=None,
        help="Comma-separated GPU IDs. If omitted, reads dpo_rounds.generation_gpu_ids, generation.gpu_ids, or dpo.unidock_gpu_ids."
    )
    parser.add_argument(
        "--lambda_gpt", "--lambda", dest="lambda_gpt", type=float, default=None,
        help="Override generation.lambda_gpt for every shard."
    )
    parser.add_argument(
        "--history_embedding_mode",
        choices=["token", "latent"],
        default=None,
        help=(
            "Override generation.history_embedding_mode for every shard: "
            "'token' uses selected-token Uni-Mol embeddings; 'latent' keeps "
            "the legacy diffusion-latent history."
        ),
    )
    constraint_group = parser.add_mutually_exclusive_group()
    constraint_group.add_argument(
        "--enforce_r1r2_constraints",
        "--enforce-r1r2-constraints",
        dest="enforce_r1r2_constraints",
        action="store_true",
        help="Enable the positional R1/R2 monomer mask for every shard.",
    )
    constraint_group.add_argument(
        "--disable_r1r2_constraints",
        "--disable-r1r2-constraints",
        "--no-r1r2-constraints",
        dest="enforce_r1r2_constraints",
        action="store_false",
        help="Disable the positional R1/R2 monomer mask for every shard.",
    )
    parser.set_defaults(enforce_r1r2_constraints=None)
    return parser.parse_args()


def resolve_path(path_str: str | Path) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def parse_gpu_ids(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [part.strip() for part in str(value).split(",") if part.strip()]


def detect_visible_gpu_ids() -> list[str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        visible = visible.strip()
        if visible and visible != "-1":
            return [part.strip() for part in visible.split(",") if part.strip()]
        return []

    try:
        import torch
        if torch.cuda.is_available():
            return [str(idx) for idx in range(torch.cuda.device_count())]
    except Exception:
        pass

    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return []

    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def split_counts(total: int, n_parts: int) -> list[int]:
    base = total // n_parts
    remainder = total % n_parts
    return [base + (1 if idx < remainder else 0) for idx in range(n_parts)]


def load_helm_list(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def write_merged_output(shard_outputs: list[Path], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    tmp_path = output_path.with_name(output_path.name + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as out:
        for shard_output in shard_outputs:
            for helm in load_helm_list(shard_output):
                out.write(f"{helm}\n")
                total += 1
    os.replace(tmp_path, output_path)
    return total


def main():
    args = parse_args()
    config_path = resolve_path(args.config)

    with open(config_path, "r", encoding="utf-8") as f:
        full_config = json.load(f)

    dpo_rounds_cfg = full_config.get("dpo_rounds", {})
    generation_cfg = full_config.get("generation", {})
    dpo_cfg = full_config.get("dpo", {})
    output_raw = args.output or generation_cfg.get("output_file")
    if not output_raw:
        raise ValueError("Set --output or generation.output_file in the config.")
    output_path = resolve_path(output_raw)

    gpu_source = "--gpu_ids"
    gpu_ids = parse_gpu_ids(args.gpu_ids)
    if not gpu_ids:
        gpu_source = "config"
        gpu_ids = parse_gpu_ids(
            dpo_rounds_cfg.get(
                "generation_gpu_ids",
                generation_cfg.get("gpu_ids", dpo_cfg.get("unidock_gpu_ids")),
            )
        )
    if not gpu_ids:
        gpu_source = "auto-detected visible GPUs"
        gpu_ids = detect_visible_gpu_ids()
    if not gpu_ids:
        raise ValueError(
            "No generation GPUs found. Set --gpu_ids, set generation.gpu_ids / "
            "dpo_rounds.generation_gpu_ids in the config, or expose GPUs with CUDA_VISIBLE_DEVICES."
        )

    total_samples = int(args.num_samples or generation_cfg.get("num_samples", 0))
    if total_samples <= 0:
        raise ValueError("generation.num_samples must be positive.")

    counts = split_counts(total_samples, len(gpu_ids))
    active_shards = [
        (shard_idx, gpu_id, count)
        for shard_idx, (gpu_id, count) in enumerate(zip(gpu_ids, counts))
        if count > 0
    ]
    if not active_shards:
        raise ValueError("No active generation shards.")

    shard_root = output_path.parent / f"{output_path.stem}.multigpu_shards"
    shard_root.mkdir(parents=True, exist_ok=True)

    base_seed = generation_cfg.get("seed")
    if base_seed is None:
        base_seed = int(time.time()) % 1_000_000_000
    else:
        base_seed = int(base_seed)

    print(f"Loading config from: {config_path}")
    print(f"Multi-GPU generation GPUs: {', '.join(gpu_ids)}")
    print(f"GPU source: {gpu_source}")
    print(f"Generating {total_samples} samples across {len(active_shards)} GPU shard(s)")
    print(f"Shard files: {shard_root}")

    processes = []
    log_handles = []
    shard_outputs: list[Path] = []

    try:
        for shard_idx, gpu_id, count in active_shards:
            shard_config = deepcopy(full_config)
            shard_config.setdefault("generation", {})
            shard_config["generation"]["num_samples"] = int(count)
            shard_config["generation"]["output_file"] = str(
                shard_root / f"shard_{shard_idx:02d}_gpu{gpu_id}.txt"
            )
            shard_config["generation"]["seed"] = int(base_seed + shard_idx)
            if args.lambda_gpt is not None:
                shard_config["generation"]["lambda_gpt"] = float(args.lambda_gpt)
            if args.history_embedding_mode is not None:
                shard_config["generation"]["history_embedding_mode"] = args.history_embedding_mode
            if args.enforce_r1r2_constraints is not None:
                shard_config["generation"]["enforce_r1r2_constraints"] = (
                    args.enforce_r1r2_constraints
                )

            shard_config_path = shard_root / f"shard_{shard_idx:02d}_gpu{gpu_id}.json"
            shard_output = Path(shard_config["generation"]["output_file"])
            log_path = shard_root / f"shard_{shard_idx:02d}_gpu{gpu_id}.log"
            save_json(shard_config, shard_config_path)
            shard_outputs.append(shard_output)

            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "scripts/generate/generate_peptides.py"),
                "--config", str(shard_config_path),
                "--output", str(shard_output),
            ]

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            env["PYTHONUNBUFFERED"] = "1"

            log_handle = open(log_path, "a", encoding="utf-8")
            log_handles.append(log_handle)
            log_handle.write(
                f"\nCUDA_VISIBLE_DEVICES={gpu_id} {' '.join(cmd)}\n"
                f"Shard sample count: {count}\n"
                f"Shard seed: {base_seed + shard_idx}\n"
            )
            log_handle.flush()

            print(f"Launching GPU {gpu_id}: {count} samples -> {shard_output}")
            proc = subprocess.Popen(
                cmd,
                cwd=PROJECT_ROOT,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            processes.append((gpu_id, count, proc, log_path, shard_output))

        failures = []
        for gpu_id, count, proc, log_path, shard_output in processes:
            returncode = proc.wait()
            if returncode != 0:
                failures.append((gpu_id, returncode, log_path))
                continue
            actual = len(load_helm_list(shard_output)) if shard_output.exists() else 0
            if actual != count:
                failures.append((gpu_id, f"expected {count} samples, got {actual}", log_path))

        if failures:
            details = "; ".join(
                f"GPU {gpu_id} exit={returncode}, log={log_path}"
                for gpu_id, returncode, log_path in failures
            )
            raise RuntimeError(f"One or more generation shards failed: {details}")

    except KeyboardInterrupt:
        print("Interrupted; terminating running generation shard processes...")
        for _, _, proc, _, _ in processes:
            if proc.poll() is None:
                proc.terminate()
        raise
    finally:
        for handle in log_handles:
            handle.close()

    merged_count = write_merged_output(shard_outputs, output_path)
    print(f"Merged {merged_count} generated HELM sequences -> {output_path}")
    print("\nDone. (Multi-GPU generation complete.)")


if __name__ == "__main__":
    main()
