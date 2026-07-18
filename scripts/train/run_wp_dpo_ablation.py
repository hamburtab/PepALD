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
        description="Controlled Standard-DPO (alpha_win=0) vs WP-DPO ablation"
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
) -> tuple[dict, dict]:
    """Create both configs and enforce the allowed difference set."""
    if wp_alpha <= 0:
        raise ValueError(f"WP-DPO alpha_win must be > 0, got {wp_alpha}")

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

    shared = manifests[0]
    return {
        "initial_checkpoint_sha256": current_checkpoint_sha256,
        "num_pairs": shared["num_pairs"],
        "winner_sha256": shared["winner_sha256"],
        "loser_sha256": shared["loser_sha256"],
        "pair_jsonl_sha256": shared["pair_jsonl_sha256"],
        "sampling_trace_sha256": standard_trace_hash,
        "sampling_traces_match": True,
    }


def main() -> None:
    args = parse_args()
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
                )
                manifest["status"] = "complete_verified"
                save_json(manifest, manifest_path)
                print(f"Verified controlled ablation: {manifest_path}")

    print("\nAll requested WP-DPO ablation work completed.")


if __name__ == "__main__":
    main()
