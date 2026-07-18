"""Run a controlled Standard-DPO vs WP-DPO loss ablation.

Preference pairs are built once per target and reused verbatim by both arms.
Apart from artifact directories, generated arm configs differ only in
``dpo.dpop_winner_reg_alpha``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASE_CONFIGS = {
    "case1": PROJECT_ROOT / "configs" / "training" / "dpo.json",
    "case2": PROJECT_ROOT / "configs" / "training" / "dpo_case2.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Standard-DPO (alpha_win=0) loss ablations"
    )
    parser.add_argument("--case", choices=["case1", "case2", "all"], default="all")
    parser.add_argument(
        "--pepald_perm_checkpoint",
        required=True,
        help="One PepALD_perm checkpoint used to initialize every arm and target.",
    )
    parser.add_argument("--case1_config", default=str(DEFAULT_CASE_CONFIGS["case1"]))
    parser.add_argument("--case2_config", default=str(DEFAULT_CASE_CONFIGS["case2"]))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--run_name",
        default=None,
        help="Stable run name. Required for a later --stage train invocation.",
    )
    parser.add_argument("--stage", choices=["all", "prepare", "train"], default="all")
    parser.add_argument(
        "--rounds",
        type=int,
        default=8,
        help=(
            "Number of actual DPO training rounds per case/arm (default: 8). "
            "Use 1 for the original shared-pair single-round protocol."
        ),
    )
    parser.add_argument(
        "--arms",
        choices=["standard", "both"],
        default="standard",
        help=(
            "Arms to train in multi-round mode. 'standard' trains only the "
            "alpha_win=0 ablation (default); 'both' also retrains WP-DPO."
        ),
    )
    parser.add_argument("--output_root", default="outputs/ablations/wp_dpo")
    parser.add_argument("--checkpoint_root", default="checkpoints/ablations/wp_dpo")
    parser.add_argument("--candidate_file_case1", default=None)
    parser.add_argument("--candidate_file_case2", default=None)
    parser.add_argument("--vina_score_file_case1", default=None)
    parser.add_argument("--vina_score_file_case2", default=None)
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Candidates per case; defaults to dpo_rounds.num_samples_per_round.",
    )
    parser.add_argument(
        "--samples_per_epoch",
        type=int,
        default=100,
        help="Samples generated from each arm after every epoch (default: 100).",
    )
    parser.add_argument("--generation_gpu_ids", default=None)
    parser.add_argument("--unidock_gpu_ids", default=None)
    parser.add_argument("--wp_alpha_case1", type=float, default=None)
    parser.add_argument("--wp_alpha_case2", type=float, default=None)
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python prefix, e.g. 'conda run --no-capture-output -n base python'.",
    )
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_gpu_ids(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split(",") if item.strip()]


def differing_paths(left: Any, right: Any, prefix: str = "") -> list[str]:
    """Return leaf paths whose values differ in two JSON-compatible objects."""
    if isinstance(left, dict) and isinstance(right, dict):
        differences = []
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else key
            if key not in left or key not in right:
                differences.append(path)
            else:
                differences.extend(differing_paths(left[key], right[key], path))
        return differences
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return [prefix]
        differences = []
        for idx, (left_item, right_item) in enumerate(zip(left, right)):
            differences.extend(differing_paths(left_item, right_item, f"{prefix}[{idx}]"))
        return differences
    return [] if left == right else [prefix]


def build_arm_configs(
    base_config: dict,
    checkpoint_path: Path,
    standard_checkpoint_dir: Path,
    wp_checkpoint_dir: Path,
    seed: int,
    wp_alpha: float,
    samples_per_epoch: int = 100,
) -> tuple[dict, dict]:
    """Create both configs and enforce the allowed difference set."""
    if wp_alpha <= 0:
        raise ValueError(f"WP-DPO alpha_win must be > 0, got {wp_alpha}")
    if samples_per_epoch < 0:
        raise ValueError("samples_per_epoch must be >= 0")

    common = deepcopy(base_config)
    common.setdefault("training", {})
    common.setdefault("generation", {})
    common.setdefault("dpo", {})
    common["training"]["pretrained_checkpoint"] = str(checkpoint_path)
    common["generation"]["checkpoint_path"] = str(checkpoint_path)
    common["generation"]["seed"] = int(seed)
    common["dpo"]["seed"] = int(seed)
    common["dpo"]["deterministic"] = True
    common["dpo"]["audit_sampling_trace"] = True
    common["dpo"]["preserve_pairing"] = True
    common["dpo"]["epoch_sample_count"] = int(samples_per_epoch)
    common["dpo"]["epoch_sample_seed"] = int(seed) + 1_000_000
    winner_reg_mode = str(common["dpo"].get("dpop_winner_reg_mode", "external_reg"))
    if winner_reg_mode == "none":
        raise ValueError("WP-DPO arm cannot use dpop_winner_reg_mode='none'.")

    standard = deepcopy(common)
    standard["training"]["checkpoint_dir"] = str(standard_checkpoint_dir)
    standard["dpo"]["dpop_winner_reg_alpha"] = 0.0

    wp = deepcopy(common)
    wp["training"]["checkpoint_dir"] = str(wp_checkpoint_dir)
    wp["dpo"]["dpop_winner_reg_alpha"] = float(wp_alpha)

    expected = {"training.checkpoint_dir", "dpo.dpop_winner_reg_alpha"}
    actual = set(differing_paths(standard, wp))
    if actual != expected:
        raise AssertionError(
            "Uncontrolled ablation config differences: "
            f"expected={sorted(expected)}, actual={sorted(actual)}"
        )
    return standard, wp


def build_multiround_arm_config(
    base_config: dict,
    checkpoint_path: Path,
    seed: int,
    alpha_win: float,
    samples_per_epoch: int,
    rounds: int,
    arm_name: str,
    case_output_root: Path,
    case_checkpoint_root: Path,
    generation_gpu_ids: list[str],
    unidock_gpu_ids: list[str],
    python_prefix: str,
) -> dict:
    """Build one independent eight-round arm config for run_dpo_rounds.py."""
    if rounds <= 1:
        raise ValueError("Multi-round config requires rounds > 1")
    if samples_per_epoch < 0:
        raise ValueError("samples_per_epoch must be >= 0")
    config = deepcopy(base_config)
    config.setdefault("training", {})
    config.setdefault("generation", {})
    config.setdefault("dpo", {})
    config.setdefault("dpo_rounds", {})

    config["training"]["pretrained_checkpoint"] = str(checkpoint_path)
    config["generation"]["checkpoint_path"] = str(checkpoint_path)
    config["generation"]["seed"] = int(seed)
    config["dpo"]["dpop_winner_reg_alpha"] = float(alpha_win)
    config["dpo"]["seed"] = int(seed)
    config["dpo"]["deterministic"] = True
    config["dpo"]["audit_sampling_trace"] = True
    config["dpo"]["preserve_pairing"] = True
    # This data-dependent safeguard is scoped to generated ablation configs.
    # Existing/main-model configs retain the original strict window behavior.
    config["dpo"]["allow_loser_pool_fallback"] = True
    config["dpo"]["epoch_sample_count"] = int(samples_per_epoch)
    config["dpo"]["epoch_sample_seed"] = int(seed) + 1_000_000
    config["dpo"]["unidock_gpu_ids"] = [int(item) for item in unidock_gpu_ids]

    rounds_cfg = config["dpo_rounds"]
    # run_dpo_rounds uses inclusive round indices r0..rN.
    rounds_cfg["num_rounds"] = int(rounds) - 1
    rounds_cfg["initial_checkpoint"] = str(checkpoint_path)
    rounds_cfg["bootstrap_generate"] = True
    rounds_cfg["output_root"] = str(case_output_root)
    rounds_cfg["checkpoint_root"] = str(case_checkpoint_root)
    rounds_cfg["run_name"] = arm_name
    rounds_cfg["generation_gpu_ids"] = [int(item) for item in generation_gpu_ids]
    rounds_cfg["elite_replay_enabled"] = False
    rounds_cfg["elite_sft_enabled"] = False
    rounds_cfg["carry_forward_previous_candidates"] = False
    rounds_cfg["base_candidate_file"] = None
    rounds_cfg["base_vina_score_file"] = None
    rounds_cfg["base_perm_score_file"] = None
    rounds_cfg["base_merge_rounds"] = 0
    rounds_cfg["resume_dpo_training"] = True
    rounds_cfg["seed_vina_cache"] = False
    rounds_cfg["python"] = python_prefix
    return config


def configured_round_epochs(rounds_cfg: dict, round_idx: int) -> int:
    """Mirror run_dpo_rounds.get_round_epochs for artifact verification."""
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


def run_command(command: list[str], seed: int, dry_run: bool) -> None:
    print(f"$ {shlex.join(command)}", flush=True)
    if dry_run:
        return
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(seed)
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    env["PYTHONUNBUFFERED"] = "1"
    subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)


def case_arg(args: argparse.Namespace, stem: str, case_name: str):
    return getattr(args, f"{stem}_{case_name}")


def build_prepare_config(
    base_config: dict,
    checkpoint_path: Path,
    shared_dir: Path,
    candidate_path: Path,
    vina_score_path: Path,
    seed: int,
    num_samples: int,
    generation_gpu_ids: list[str],
    unidock_gpu_ids: list[str],
) -> dict:
    config = deepcopy(base_config)
    config.setdefault("training", {})
    config.setdefault("generation", {})
    config.setdefault("dpo", {})
    config.setdefault("dpo_rounds", {})
    config["training"]["pretrained_checkpoint"] = str(checkpoint_path)
    config["training"]["checkpoint_dir"] = str(shared_dir)
    config["generation"]["checkpoint_path"] = str(checkpoint_path)
    config["generation"]["output_file"] = str(shared_dir / "generated_raw.txt")
    config["generation"]["num_samples"] = int(num_samples)
    config["generation"]["seed"] = int(seed)
    config["dpo"]["sample_files"] = [str(candidate_path)]
    config["dpo"]["vina_score_file"] = str(vina_score_path)
    config["dpo"]["seed"] = int(seed)
    config["dpo"]["deterministic"] = True
    config["dpo"]["preserve_pairing"] = True
    config["dpo"]["unidock_gpu_ids"] = [int(item) for item in unidock_gpu_ids]
    config["dpo_rounds"]["generation_gpu_ids"] = [int(item) for item in generation_gpu_ids]
    return config


def prepare_pairs(
    args: argparse.Namespace,
    python_command: list[str],
    prepare_config: dict,
    prepare_config_path: Path,
    candidate_override: Path | None,
    vina_score_path: Path,
    generation_gpu_ids: list[str],
    unidock_gpu_ids: list[str],
) -> None:
    save_json(prepare_config, prepare_config_path)
    candidate_path = Path(prepare_config["dpo"]["sample_files"][0])

    if candidate_override is None:
        generated_raw = Path(prepare_config["generation"]["output_file"])
        generator_script = (
            "scripts/generate/generate_peptides_multigpu.py"
            if generation_gpu_ids
            else "scripts/generate/generate_peptides.py"
        )
        command = python_command + [
            generator_script,
            "--config", str(prepare_config_path),
            "--output", str(generated_raw),
            "--num_samples", str(prepare_config["generation"]["num_samples"]),
        ]
        if generation_gpu_ids:
            command.extend(["--gpu_ids", ",".join(generation_gpu_ids)])
        run_command(command, args.seed, args.dry_run)

        postprocess_mode = str(
            prepare_config.get("dpo_rounds", {}).get(
                "generated_postprocess", "force_r1r2_cyclize"
            )
        )
        if postprocess_mode != "force_r1r2_cyclize":
            raise ValueError(
                "The controlled runner requires generated_postprocess="
                f"force_r1r2_cyclize, got {postprocess_mode!r}"
            )
        run_command(
            python_command + [
                "scripts/data/vina_filter_r1r2_cyclize.py",
                "--input", str(generated_raw),
                "--output", str(candidate_path),
            ],
            args.seed,
            args.dry_run,
        )
    elif not args.dry_run and not candidate_override.exists():
        raise FileNotFoundError(f"Candidate file not found: {candidate_override}")

    vina_script = (
        "scripts/eval/export_train_vina_scores_multigpu.py"
        if unidock_gpu_ids
        else "scripts/eval/export_train_vina_scores.py"
    )
    command = python_command + [
        vina_script,
        "--config", str(prepare_config_path),
        "--sample_file", str(candidate_path),
        "--vina_score_file", str(vina_score_path),
        "--docking_mode", str(prepare_config["dpo"].get("docking_mode", "flexible")),
    ]
    if unidock_gpu_ids:
        command.extend(["--gpu_ids", ",".join(unidock_gpu_ids)])
    run_command(command, args.seed, args.dry_run)

    run_command(
        python_command + [
            "scripts/train/train_dpo.py",
            "--config", str(prepare_config_path),
            "--sample_file", str(candidate_path),
            "--vina_score_file", str(vina_score_path),
            "--prepare_pairs_only",
            "--seed", str(args.seed),
        ],
        args.seed,
        args.dry_run,
    )


def train_arms(
    args: argparse.Namespace,
    python_command: list[str],
    standard_config_path: Path,
    wp_config_path: Path,
    shared_pair_dir: Path,
) -> None:
    winner_file = shared_pair_dir / "winners.txt"
    loser_file = shared_pair_dir / "losers.txt"
    if not args.dry_run:
        for path in (winner_file, loser_file):
            if not path.exists():
                raise FileNotFoundError(f"Shared preference-pair artifact missing: {path}")

        for config_path in (standard_config_path, wp_config_path):
            config = load_json(config_path)
            checkpoint_dir = Path(config["training"]["checkpoint_dir"])
            if (checkpoint_dir / "dpo_latest.pt").exists():
                raise FileExistsError(
                    "Controlled ablation arms must start fresh, but an existing "
                    f"checkpoint was found in {checkpoint_dir}. Use a new --run_name."
                )

    for config_path in (standard_config_path, wp_config_path):
        run_command(
            python_command + [
                "scripts/train/train_dpo.py",
                "--config", str(config_path),
                "--skip_generate",
                "--winner_file", str(winner_file),
                "--loser_file", str(loser_file),
                "--seed", str(args.seed),
            ],
            args.seed,
            args.dry_run,
        )


def verify_artifacts(
    checkpoint_path: Path,
    expected_checkpoint_sha256: str,
    shared_pair_dir: Path,
    standard_checkpoint_dir: Path,
    wp_checkpoint_dir: Path,
    expected_epochs: int,
    samples_per_epoch: int,
) -> dict:
    manifests = [
        load_json(shared_pair_dir / "preference_manifest.json"),
        load_json(standard_checkpoint_dir / "dpo_data" / "preference_manifest.json"),
        load_json(wp_checkpoint_dir / "dpo_data" / "preference_manifest.json"),
    ]
    pair_keys = ("num_pairs", "winner_sha256", "loser_sha256", "pair_jsonl_sha256")
    for key in pair_keys:
        values = {manifest[key] for manifest in manifests}
        if len(values) != 1:
            raise RuntimeError(f"Preference-pair mismatch across arms for {key}: {values}")

    standard_trace = standard_checkpoint_dir / "sampling_trace.jsonl"
    wp_trace = wp_checkpoint_dir / "sampling_trace.jsonl"
    standard_trace_hash = sha256_file(standard_trace)
    wp_trace_hash = sha256_file(wp_trace)
    if standard_trace_hash != wp_trace_hash:
        raise RuntimeError(
            "Standard-DPO and WP-DPO sampling traces differ: batch order, "
            "timestep, or noise RNG stream was not controlled."
        )

    current_checkpoint_sha256 = sha256_file(checkpoint_path)
    if current_checkpoint_sha256 != expected_checkpoint_sha256:
        raise RuntimeError("PepALD_perm checkpoint changed during the ablation run.")

    epoch_sample_outputs = {}
    for arm_name, arm_dir in (
        ("standard_dpo", standard_checkpoint_dir),
        ("wp_dpo", wp_checkpoint_dir),
    ):
        sample_dir = arm_dir / "epoch_samples"
        records = []
        if samples_per_epoch > 0:
            manifest_path = sample_dir / "manifest.jsonl"
            if not manifest_path.exists():
                raise FileNotFoundError(f"Epoch-sample manifest missing: {manifest_path}")
            manifest_records = [
                json.loads(line)
                for line in manifest_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if len(manifest_records) != expected_epochs:
                raise RuntimeError(
                    f"{arm_name} expected {expected_epochs} epoch-sample records, "
                    f"found {len(manifest_records)}"
                )
            for epoch in range(1, expected_epochs + 1):
                sample_path = sample_dir / f"epoch_{epoch:03d}.txt"
                if not sample_path.exists():
                    raise FileNotFoundError(f"Epoch sample file missing: {sample_path}")
                sample_count = sum(
                    1
                    for line in sample_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                )
                if sample_count != samples_per_epoch:
                    raise RuntimeError(
                        f"{sample_path} expected {samples_per_epoch} samples, "
                        f"found {sample_count}"
                    )
                records.append(
                    {
                        "epoch": epoch,
                        "sample_file": str(sample_path),
                        "num_samples": sample_count,
                        "sha256": sha256_file(sample_path),
                    }
                )
        epoch_sample_outputs[arm_name] = records

    shared = manifests[0]
    return {
        "initial_checkpoint_sha256": current_checkpoint_sha256,
        "num_pairs": shared["num_pairs"],
        "winner_sha256": shared["winner_sha256"],
        "loser_sha256": shared["loser_sha256"],
        "pair_jsonl_sha256": shared["pair_jsonl_sha256"],
        "sampling_trace_sha256": standard_trace_hash,
        "sampling_traces_match": True,
        "epoch_samples": epoch_sample_outputs,
    }


def run_multiround_arm(
    args: argparse.Namespace,
    python_command: list[str],
    config_path: Path,
    config: dict,
    generation_gpu_ids: list[str],
    unidock_gpu_ids: list[str],
) -> None:
    rounds_cfg = config["dpo_rounds"]
    first_checkpoint_dir = (
        Path(rounds_cfg["checkpoint_root"])
        / f"{rounds_cfg['run_name']}_r0"
    )
    if not args.dry_run and (first_checkpoint_dir / "dpo_latest.pt").exists():
        raise FileExistsError(
            f"Multi-round arm already exists at {first_checkpoint_dir}. "
            "Use a new --run_name."
        )

    command = python_command + [
        "scripts/train/run_dpo_rounds.py",
        "--config", str(config_path),
        "--start_round", "0",
        "--num_rounds", str(args.rounds - 1),
    ]
    if generation_gpu_ids:
        command.extend(["--generation_gpu_ids", ",".join(generation_gpu_ids)])
    if unidock_gpu_ids:
        command.extend(["--unidock_gpu_ids", ",".join(unidock_gpu_ids)])
    if args.dry_run:
        command.append("--dry_run")
    run_command(command, args.seed, args.dry_run)


def verify_multiround_arm(config: dict, rounds: int, samples_per_epoch: int) -> dict:
    rounds_cfg = config["dpo_rounds"]
    run_name = str(rounds_cfg["run_name"])
    output_root = Path(rounds_cfg["output_root"])
    checkpoint_root = Path(rounds_cfg["checkpoint_root"])
    round_records = []
    total_epochs = 0
    total_epoch_samples = 0

    for round_idx in range(rounds):
        round_dir = output_root / f"{run_name}_r{round_idx}"
        checkpoint_dir = checkpoint_root / f"{run_name}_r{round_idx}"
        required = [
            round_dir / "round_summary.json",
            checkpoint_dir / "dpo_latest.pt",
            checkpoint_dir / "dpo_data" / "preference_manifest.json",
            checkpoint_dir / "sampling_trace.jsonl",
            checkpoint_dir / "epoch_metrics.jsonl",
        ]
        for path in required:
            if not path.exists():
                raise FileNotFoundError(f"Multi-round artifact missing: {path}")

        expected_epochs = configured_round_epochs(rounds_cfg, round_idx)
        total_epochs += expected_epochs
        sample_manifest_path = checkpoint_dir / "epoch_samples" / "manifest.jsonl"
        sample_records = []
        if samples_per_epoch > 0:
            if not sample_manifest_path.exists():
                raise FileNotFoundError(
                    f"Epoch-sample manifest missing: {sample_manifest_path}"
                )
            sample_records = [
                json.loads(line)
                for line in sample_manifest_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if len(sample_records) != expected_epochs:
                raise RuntimeError(
                    f"{sample_manifest_path} expected {expected_epochs} records, "
                    f"found {len(sample_records)}"
                )
            for sample_record in sample_records:
                if int(sample_record["num_samples"]) != samples_per_epoch:
                    raise RuntimeError(
                        "Unexpected samples/epoch in "
                        f"{sample_manifest_path}: {sample_record}"
                    )
                sample_path = Path(sample_record["sample_file"])
                if not sample_path.exists():
                    raise FileNotFoundError(f"Epoch sample file missing: {sample_path}")
                line_count = sum(
                    1
                    for line in sample_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                )
                if line_count != samples_per_epoch:
                    raise RuntimeError(
                        f"{sample_path} expected {samples_per_epoch} samples, "
                        f"found {line_count}"
                    )
                total_epoch_samples += line_count

        pair_manifest = load_json(
            checkpoint_dir / "dpo_data" / "preference_manifest.json"
        )
        round_records.append(
            {
                "round": round_idx,
                "epochs": expected_epochs,
                "checkpoint": str(checkpoint_dir / "dpo_latest.pt"),
                "num_pairs": pair_manifest["num_pairs"],
                "winner_sha256": pair_manifest["winner_sha256"],
                "loser_sha256": pair_manifest["loser_sha256"],
                "epoch_sample_manifest": str(sample_manifest_path),
            }
        )

    return {
        "rounds": rounds,
        "total_epochs": total_epochs,
        "total_epoch_samples": total_epoch_samples,
        "round_records": round_records,
    }


def main() -> None:
    args = parse_args()
    if args.rounds < 1:
        raise ValueError("--rounds must be >= 1")
    if args.rounds > 1 and args.stage != "all":
        raise ValueError("Multi-round mode currently requires --stage all")
    if args.rounds == 1 and args.arms != "both":
        raise ValueError(
            "The shared-pair single-round protocol requires --arms both."
        )
    if args.rounds > 1 and (
        args.candidate_file_case1
        or args.candidate_file_case2
        or args.vina_score_file_case1
        or args.vina_score_file_case2
    ):
        raise ValueError(
            "Multi-round mode generates and scores independent candidates in every "
            "arm/round; candidate and Vina file overrides are only valid with --rounds 1."
        )
    checkpoint_path = resolve_path(args.pepald_perm_checkpoint)
    if not args.dry_run and not checkpoint_path.exists():
        raise FileNotFoundError(f"PepALD_perm checkpoint not found: {checkpoint_path}")
    if args.stage == "train" and args.run_name is None:
        raise ValueError("--run_name is required with --stage train")

    run_name = args.run_name or datetime.now().strftime(f"%Y%m%d_%H%M%S_seed{args.seed}")
    output_root = resolve_path(args.output_root) / run_name
    checkpoint_root = resolve_path(args.checkpoint_root) / run_name
    python_command = shlex.split(args.python)
    selected_cases = ["case1", "case2"] if args.case == "all" else [args.case]
    checkpoint_sha256 = None if args.dry_run else sha256_file(checkpoint_path)

    print(f"Run name: {run_name}")
    print(f"PepALD_perm checkpoint: {checkpoint_path}")
    print(f"Cases: {', '.join(selected_cases)}")
    print(f"Seed: {args.seed}")
    print(f"DPO rounds per arm: {args.rounds}")
    print(f"Training arms: {args.arms}")

    for case_name in selected_cases:
        base_config_path = resolve_path(getattr(args, f"{case_name}_config"))
        base_config = load_json(base_config_path)
        shared_dir = output_root / case_name / "shared"
        shared_pair_dir = shared_dir / "dpo_data"
        standard_checkpoint_dir = checkpoint_root / case_name / "standard_dpo"
        wp_checkpoint_dir = checkpoint_root / case_name / "wp_dpo"

        candidate_arg = case_arg(args, "candidate_file", case_name)
        candidate_override = resolve_path(candidate_arg) if candidate_arg else None
        candidate_path = candidate_override or (shared_dir / "candidates.txt")
        vina_arg = case_arg(args, "vina_score_file", case_name)
        vina_score_path = resolve_path(vina_arg) if vina_arg else (shared_dir / "vina_scores.csv")

        rounds_cfg = base_config.get("dpo_rounds", {})
        dpo_cfg = base_config.get("dpo", {})
        num_samples = int(
            args.num_samples
            if args.num_samples is not None
            else rounds_cfg.get("num_samples_per_round", dpo_cfg.get("num_generate", 2000))
        )
        generation_gpu_ids = parse_gpu_ids(
            args.generation_gpu_ids
            if args.generation_gpu_ids is not None
            else rounds_cfg.get("generation_gpu_ids")
        )
        unidock_gpu_ids = parse_gpu_ids(
            args.unidock_gpu_ids
            if args.unidock_gpu_ids is not None
            else dpo_cfg.get("unidock_gpu_ids")
        )
        wp_alpha_override = case_arg(args, "wp_alpha", case_name)
        wp_alpha = float(
            wp_alpha_override
            if wp_alpha_override is not None
            else dpo_cfg.get("dpop_winner_reg_alpha", 0.0)
        )
        if args.arms == "both" and wp_alpha <= 0:
            raise ValueError(
                f"{case_name} WP-DPO alpha_win must be > 0, got {wp_alpha}"
            )

        if args.rounds > 1:
            multiround_base = deepcopy(base_config)
            if args.num_samples is not None:
                multiround_base.setdefault("generation", {})["num_samples"] = int(
                    args.num_samples
                )
                multiround_base.setdefault("dpo_rounds", {})[
                    "num_samples_per_round"
                ] = int(args.num_samples)

            case_round_output_root = output_root / case_name / "rounds"
            case_round_checkpoint_root = checkpoint_root / case_name / "rounds"
            standard_config = build_multiround_arm_config(
                multiround_base,
                checkpoint_path,
                args.seed,
                0.0,
                args.samples_per_epoch,
                args.rounds,
                "standard_dpo",
                case_round_output_root,
                case_round_checkpoint_root,
                generation_gpu_ids,
                unidock_gpu_ids,
                args.python,
            )
            wp_config = None
            if args.arms == "both":
                wp_config = build_multiround_arm_config(
                    multiround_base,
                    checkpoint_path,
                    args.seed,
                    wp_alpha,
                    args.samples_per_epoch,
                    args.rounds,
                    "wp_dpo",
                    case_round_output_root,
                    case_round_checkpoint_root,
                    generation_gpu_ids,
                    unidock_gpu_ids,
                    args.python,
                )
                expected_differences = {
                    "dpo.dpop_winner_reg_alpha",
                    "dpo_rounds.run_name",
                }
                actual_differences = set(differing_paths(standard_config, wp_config))
                if actual_differences != expected_differences:
                    raise AssertionError(
                        "Uncontrolled multi-round config differences: "
                        f"expected={sorted(expected_differences)}, "
                        f"actual={sorted(actual_differences)}"
                    )

            standard_config_path = (
                output_root / case_name / f"standard_dpo_{args.rounds}rounds.json"
            )
            wp_config_path = (
                output_root / case_name / f"wp_dpo_{args.rounds}rounds.json"
            )
            save_json(standard_config, standard_config_path)
            if wp_config is not None:
                save_json(wp_config, wp_config_path)
            candidates_per_round = int(
                standard_config["dpo_rounds"].get("num_samples_per_round", 2000)
            )
            trained_arms = ["standard_dpo"]
            if wp_config is not None:
                trained_arms.append("wp_dpo")
            manifest = {
                "case": case_name,
                "run_name": run_name,
                "protocol": "independent_multiround",
                "trained_arms": trained_arms,
                "rounds_per_arm": args.rounds,
                "round_indices": list(range(args.rounds)),
                "candidates_per_round": candidates_per_round,
                "estimated_candidates_all_trained_arms": (
                    candidates_per_round * args.rounds * len(trained_arms)
                ),
                "pepald_perm_checkpoint": str(checkpoint_path),
                "pepald_perm_checkpoint_sha256": checkpoint_sha256,
                "seed": args.seed,
                "standard_alpha_win": 0.0,
                "wp_alpha_win": wp_alpha if wp_config is not None else None,
                "standard_config": str(standard_config_path),
                "wp_config": str(wp_config_path) if wp_config is not None else None,
                "main_wp_dpo_retrained": wp_config is not None,
                "elite_sft_enabled": False,
                "elite_replay_enabled": False,
                "samples_per_epoch": args.samples_per_epoch,
                "status": "configured",
            }
            manifest_path = output_root / case_name / "ablation_manifest.json"
            save_json(manifest, manifest_path)

            print(f"\n=== {case_name}: Standard DPO, {args.rounds} rounds ===")
            run_multiround_arm(
                args,
                python_command,
                standard_config_path,
                standard_config,
                generation_gpu_ids,
                unidock_gpu_ids,
            )
            if wp_config is not None:
                print(f"\n=== {case_name}: WP-DPO, {args.rounds} rounds ===")
                run_multiround_arm(
                    args,
                    python_command,
                    wp_config_path,
                    wp_config,
                    generation_gpu_ids,
                    unidock_gpu_ids,
                )
            if not args.dry_run:
                current_checkpoint_sha256 = sha256_file(checkpoint_path)
                if current_checkpoint_sha256 != checkpoint_sha256:
                    raise RuntimeError(
                        "PepALD_perm checkpoint changed during the multi-round run."
                    )
                verification = {
                    "standard_dpo": verify_multiround_arm(
                        standard_config, args.rounds, args.samples_per_epoch
                    )
                }
                if wp_config is not None:
                    verification["wp_dpo"] = verify_multiround_arm(
                        wp_config, args.rounds, args.samples_per_epoch
                    )
                manifest["verification"] = verification
                manifest["status"] = "complete_verified"
                save_json(manifest, manifest_path)
                print(f"Verified multi-round ablation: {manifest_path}")
            continue

        prepare_config = build_prepare_config(
            base_config,
            checkpoint_path,
            shared_dir,
            candidate_path,
            vina_score_path,
            args.seed,
            num_samples,
            generation_gpu_ids,
            unidock_gpu_ids,
        )
        prepare_config_path = shared_dir / f"dpo_{case_name}_prepare_pairs.json"
        standard_config, wp_config = build_arm_configs(
            base_config,
            checkpoint_path,
            standard_checkpoint_dir,
            wp_checkpoint_dir,
            args.seed,
            wp_alpha,
            args.samples_per_epoch,
        )
        standard_config_path = output_root / case_name / "standard_dpo.json"
        wp_config_path = output_root / case_name / "wp_dpo.json"
        save_json(standard_config, standard_config_path)
        save_json(wp_config, wp_config_path)

        manifest = {
            "case": case_name,
            "run_name": run_name,
            "base_config": str(base_config_path),
            "pepald_perm_checkpoint": str(checkpoint_path),
            "pepald_perm_checkpoint_sha256": checkpoint_sha256,
            "seed": args.seed,
            "standard_alpha_win": 0.0,
            "wp_alpha_win": wp_alpha,
            "allowed_config_differences": [
                "training.checkpoint_dir",
                "dpo.dpop_winner_reg_alpha",
            ],
            "shared_winner_file": str(shared_pair_dir / "winners.txt"),
            "shared_loser_file": str(shared_pair_dir / "losers.txt"),
            "standard_config": str(standard_config_path),
            "wp_config": str(wp_config_path),
            "controlled_hyperparameters": {
                "num_epochs": dpo_cfg.get("num_epochs", 10),
                "batch_size": base_config.get("training", {}).get("batch_size"),
                "learning_rate": dpo_cfg.get("lr", 1e-5),
                "beta_dpo": dpo_cfg.get("beta_dpo", 0.1),
                "diffusion_steps": base_config.get("model", {}).get("num_diffusion_steps"),
                "deterministic": True,
                "audit_sampling_trace": True,
                "samples_per_epoch": args.samples_per_epoch,
                "epoch_sample_seed": args.seed + 1_000_000,
            },
            "status": "configured",
        }
        manifest_path = output_root / case_name / "ablation_manifest.json"
        save_json(manifest, manifest_path)

        print(f"\n=== {case_name}: prepare shared pairs ===")
        if args.stage in {"all", "prepare"}:
            prepare_pairs(
                args,
                python_command,
                prepare_config,
                prepare_config_path,
                candidate_override,
                vina_score_path,
                generation_gpu_ids,
                unidock_gpu_ids,
            )
            if not args.dry_run:
                manifest["status"] = "pairs_prepared"
                manifest["shared_pair_manifest"] = load_json(
                    shared_pair_dir / "preference_manifest.json"
                )
                save_json(manifest, manifest_path)

        print(f"\n=== {case_name}: train controlled arms ===")
        if args.stage in {"all", "train"}:
            train_arms(
                args,
                python_command,
                standard_config_path,
                wp_config_path,
                shared_pair_dir,
            )
            if not args.dry_run:
                manifest["verification"] = verify_artifacts(
                    checkpoint_path,
                    checkpoint_sha256,
                    shared_pair_dir,
                    standard_checkpoint_dir,
                    wp_checkpoint_dir,
                    int(dpo_cfg.get("num_epochs", 10)),
                    args.samples_per_epoch,
                )
                manifest["status"] = "complete_verified"
                save_json(manifest, manifest_path)
                print(f"Verified controlled ablation: {manifest_path}")

    print("\nAll requested DPO ablation work completed.")


if __name__ == "__main__":
    main()
