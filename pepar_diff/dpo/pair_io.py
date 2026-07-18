"""Persistence and integrity checks for aligned DPO preference pairs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def helm_list_sha256(helms: list[str]) -> str:
    """Hash the canonical one-HELM-per-line representation."""
    digest = hashlib.sha256()
    for helm in helms:
        digest.update(helm.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def validate_preference_pairs(winners: list[str], losers: list[str]) -> None:
    """Reject empty or misaligned pair files instead of silently truncating."""
    if not winners or not losers:
        raise ValueError("Preference pair files must be non-empty.")
    if len(winners) != len(losers):
        raise ValueError(
            "Preference pair files must have the same number of non-empty lines: "
            f"winners={len(winners)}, losers={len(losers)}"
        )


def _write_helm_list(helms: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for helm in helms:
            f.write(f"{helm}\n")


def save_preference_pair_snapshot(
    winners: list[str],
    losers: list[str],
    save_dir: str | Path,
    preserve_pairing: bool,
    source_winner_file: str | None = None,
    source_loser_file: str | None = None,
) -> Path:
    """Save split files, aligned JSONL, and content hashes for one pair set."""
    validate_preference_pairs(winners, losers)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    _write_helm_list(winners, save_dir / "winners.txt")
    _write_helm_list(losers, save_dir / "losers.txt")

    pair_jsonl = save_dir / "preference_pairs.jsonl"
    with pair_jsonl.open("w", encoding="utf-8") as f:
        for winner, loser in zip(winners, losers):
            f.write(
                json.dumps(
                    {"winner": winner, "loser": loser},
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

    manifest = {
        "num_pairs": len(winners),
        "preserve_pairing": bool(preserve_pairing),
        "winner_sha256": helm_list_sha256(winners),
        "loser_sha256": helm_list_sha256(losers),
        "pair_jsonl_sha256": hashlib.sha256(pair_jsonl.read_bytes()).hexdigest(),
        "source_winner_file": source_winner_file,
        "source_loser_file": source_loser_file,
    }
    manifest_path = save_dir / "preference_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest_path
