"""
Generate checkpoint samples, force R1/R2 cyclization, and export flexible Vina scores.

Examples:
    python scripts/eval/evaluate_checkpoint_vina.py --config generate_case1.json
    python scripts/eval/evaluate_checkpoint_vina.py \
        --config generate_case1.json \
        --rounds_start 2 \
        --rounds_end 7 \
        --start_path elite_sft/checkpoint_epoch_1.pt
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate samples from checkpoint(s) and score them with flexible Vina."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Generation config path or basename under configs/inference, e.g. generate_case1.json.",
    )
    parser.add_argument("--rounds_start", type=int, default=None, help="First round to evaluate.")
    parser.add_argument(
        "--rounds_end",
        type=int,
        default=None,
        help="Exclusive end round. --rounds_start 2 --rounds_end 7 evaluates r2..r6.",
    )
    parser.add_argument(
        "--start_path",
        default=None,
        help=(
            "Checkpoint path relative to each round dir, e.g. "
            "elite_sft/checkpoint_epoch_1.pt. Defaults to the suffix from the config checkpoint."
        ),
    )
    return parser.parse_args()


def resolve_config(value: str) -> Path:
    path = Path(value).expanduser()
    candidates = [path]
    if path.suffix != ".json":
        candidates.append(path.with_suffix(".json"))
    candidates.extend([
        PROJECT_ROOT / "configs" / "inference" / path,
        PROJECT_ROOT / "configs" / "inference" / path.with_suffix(".json"),
    ])
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Generation config not found: {value}")


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def infer_case_dir(config_path: Path, config: dict) -> tuple[str, Path]:
    output_file = config.get("generation", {}).get("output_file")
    if output_file:
        output_path = resolve_project_path(output_file)
        parts = output_path.parts
        if "samples" in parts:
            idx = parts.index("samples")
            if idx + 1 < len(parts):
                case_tag = parts[idx + 1]
                return case_tag, output_path.parents[1]

    match = re.search(r"(case\d+)", config_path.stem)
    case_tag = match.group(1) if match else "case1"
    return case_tag, PROJECT_ROOT / "outputs" / "samples" / case_tag


def infer_dpo_config(case_tag: str) -> Path:
    mapping = {
        "case1": PROJECT_ROOT / "configs" / "training" / "dpo.json",
        "case2": PROJECT_ROOT / "configs" / "training" / "dpo_2axi.json",
        "case3": PROJECT_ROOT / "configs" / "training" / "dpo_case3.json",
    }
    path = mapping.get(case_tag, PROJECT_ROOT / "configs" / "training" / f"dpo_{case_tag}.json")
    if not path.exists():
        raise FileNotFoundError(f"Cannot infer DPO/Vina config for {case_tag}: {path}")
    return path


def split_round_checkpoint(checkpoint: Path) -> tuple[Path, str, int, Path]:
    parts = checkpoint.parts
    for idx, part in enumerate(parts):
        match = re.match(r"(.+)_r(\d+)$", part)
        if not match:
            continue
        round_parent = Path(checkpoint.anchor).joinpath(*parts[1:idx]) if checkpoint.is_absolute() else Path(*parts[:idx])
        suffix = Path(*parts[idx + 1 :])
        return round_parent, match.group(1), int(match.group(2)), suffix
    raise ValueError(f"Checkpoint path does not contain a round dir ending in _rN: {checkpoint}")


def checkpoint_label(checkpoint: Path) -> str:
    round_match = None
    for part in checkpoint.parts:
        match = re.match(r".+_r(\d+)$", part)
        if match:
            round_match = match.group(1)

    stem = checkpoint.stem
    epoch_match = re.search(r"(?:checkpoint_|dpo_)?epoch_?(\d+)$", stem)
    if epoch_match:
        tail = f"epoch{epoch_match.group(1)}"
    else:
        tail = re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_") or "checkpoint"

    prefix = f"r{round_match}" if round_match is not None else "checkpoint"
    if "elite_sft" in checkpoint.parts:
        return f"{prefix}_stf_{tail}"
    return f"{prefix}_{tail}"


def build_round_checkpoints(base_checkpoint: Path, args: argparse.Namespace) -> list[Path]:
    if args.rounds_start is None and args.rounds_end is None:
        return [base_checkpoint]
    if args.rounds_start is None or args.rounds_end is None:
        raise ValueError("--rounds_start and --rounds_end must be provided together.")
    if args.rounds_end <= args.rounds_start:
        raise ValueError("--rounds_end must be greater than --rounds_start.")

    round_parent, run_name, _, suffix = split_round_checkpoint(base_checkpoint)
    if args.start_path:
        suffix = Path(args.start_path)
    return [
        round_parent / f"{run_name}_r{round_idx}" / suffix
        for round_idx in range(args.rounds_start, args.rounds_end)
    ]


def run_command(cmd: list[str]) -> str:
    print("\n$ " + " ".join(str(part) for part in cmd), flush=True)
    proc = subprocess.Popen(
        cmd,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    lines = []
    for line in proc.stdout:
        print(line, end="")
        lines.append(line)
    rc = proc.wait()
    output = "".join(lines)
    if rc != 0:
        raise RuntimeError(f"Command failed with exit code {rc}: {' '.join(cmd)}")
    return output


def extract_vina_summary(output: str) -> str:
    marker = "=== Vina Summary ==="
    start = output.find(marker)
    if start < 0:
        return "(Vina summary not found in command output.)"
    summary = output[start:].split("\nDone.", 1)[0].strip()
    return summary


def evaluate_checkpoint(
    base_config: dict,
    dpo_config: Path,
    checkpoint: Path,
    output_dir: Path,
) -> tuple[Path, str]:
    label = checkpoint_label(checkpoint)
    raw_samples = output_dir / f".{label}.generated.txt"
    cyc_samples = output_dir / f".{label}.generated_cyc.txt"
    tmp_config = output_dir / f".{label}.generate_config.json"
    vina_csv = output_dir / f"{label}.csv"

    if vina_csv.exists():
        vina_csv.unlink()

    cfg = json.loads(json.dumps(base_config))
    cfg.setdefault("generation", {})
    cfg["generation"]["checkpoint_path"] = str(checkpoint)
    cfg["generation"]["output_file"] = str(raw_samples)
    save_json(cfg, tmp_config)

    try:
        run_command([
            sys.executable,
            "scripts/generate/generate_peptides.py",
            "--config",
            str(tmp_config),
        ])
        run_command([
            sys.executable,
            "scripts/data/vina_filter_r1r2_cyclize.py",
            "--input",
            str(raw_samples),
            "--output",
            str(cyc_samples),
        ])
        vina_output = run_command([
            sys.executable,
            "scripts/eval/export_train_vina_scores.py",
            "--config",
            str(dpo_config),
            "--sample_file",
            str(cyc_samples),
            "--vina_score_file",
            str(vina_csv),
            "--docking_mode",
            "flexible",
        ])
        return vina_csv, extract_vina_summary(vina_output)
    finally:
        for path in (raw_samples, cyc_samples, tmp_config):
            path.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    config_path = resolve_config(args.config)
    base_config = load_json(config_path)
    checkpoint_raw = base_config.get("generation", {}).get("checkpoint_path")
    if not checkpoint_raw:
        raise ValueError(f"{config_path} does not define generation.checkpoint_path")

    case_tag, case_dir = infer_case_dir(config_path, base_config)
    output_dir = case_dir / "samples_vina"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.txt"
    summary_path.write_text("", encoding="utf-8")

    dpo_config = infer_dpo_config(case_tag)
    checkpoints = build_round_checkpoints(Path(checkpoint_raw).expanduser(), args)

    print(f"Generation config: {config_path}")
    print(f"Vina config:       {dpo_config}")
    print(f"Output dir:        {output_dir}")
    print(f"Summary file:      {summary_path}")

    completed = 0
    with summary_path.open("a", encoding="utf-8") as summary_file:
        for checkpoint in checkpoints:
            if not checkpoint.exists():
                message = f"[SKIP] checkpoint not found: {checkpoint}"
                print(message)
                summary_file.write(message + "\n\n")
                summary_file.flush()
                continue

            print("\n" + "=" * 80)
            print(f"Evaluating checkpoint: {checkpoint}")
            print("=" * 80)
            vina_csv, vina_summary = evaluate_checkpoint(
                base_config=base_config,
                dpo_config=dpo_config,
                checkpoint=checkpoint,
                output_dir=output_dir,
            )
            completed += 1
            summary_file.write(f"Checkpoint: {checkpoint}\n")
            summary_file.write(f"CSV: {vina_csv}\n")
            summary_file.write(vina_summary + "\n\n")
            summary_file.flush()
            print(f"Saved Vina CSV: {vina_csv}")

    print(f"\nCompleted {completed}/{len(checkpoints)} checkpoint(s).")
    print(f"Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
