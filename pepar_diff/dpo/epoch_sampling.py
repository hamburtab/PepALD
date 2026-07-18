"""Deterministic, RNG-isolated sampling after DPO epochs."""

from __future__ import annotations

import json
import random
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np
import torch


@contextmanager
def isolated_sampling_seed(seed: int, device: torch.device) -> Iterator[None]:
    """Temporarily seed generation without perturbing subsequent training RNGs."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cuda_devices: list[int] = []
    if device.type == "cuda":
        cuda_devices = [
            device.index if device.index is not None else torch.cuda.current_device()
        ]

    try:
        with torch.random.fork_rng(devices=cuda_devices, enabled=True):
            random.seed(seed)
            np.random.seed(seed)
            torch.random.default_generator.manual_seed(seed)
            if cuda_devices:
                torch.cuda.default_generators[cuda_devices[0]].manual_seed(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def generate_epoch_samples(
    model,
    config,
    device: torch.device,
    epoch: int,
    num_samples: int,
    base_seed: int,
    output_dir: str | Path,
    alpha_win: float,
) -> Path:
    """Generate and save one deterministic sample set from the current model."""
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")

    epoch = int(epoch)
    epoch_seed = int(base_seed) + epoch
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"epoch_{epoch:03d}.txt"
    gen_cfg = config.generation
    was_training = model.training

    try:
        with isolated_sampling_seed(epoch_seed, device), torch.no_grad():
            model.eval()
            samples = model.sample(
                num_samples=int(num_samples),
                max_seq_len=gen_cfg.max_length,
                min_seq_len=gen_cfg.min_length,
                device=device,
                use_ddim=gen_cfg.use_ddim,
                ddim_steps=gen_cfg.ddim_steps if gen_cfg.use_ddim else None,
                lambda_gpt=gen_cfg.lambda_gpt,
                history_embedding_mode=gen_cfg.history_embedding_mode,
                predict_ring_bonds=gen_cfg.predict_ring_bonds,
                ring_threshold=gen_cfg.ring_bond_threshold,
                ring_top_k=gen_cfg.ring_top_k,
                verbose=False,
            )

        helms = []
        for sample in samples:
            if isinstance(sample, dict) and "tokens" in sample:
                tokens = sample["tokens"]
                ring_connections = sample.get("ring_connections", [])
            else:
                tokens = sample
                ring_connections = []
            helms.append(model.decode_to_helm(tokens, ring_connections))
    finally:
        model.train(was_training)

    if len(helms) != num_samples:
        raise RuntimeError(
            f"Epoch sampler requested {num_samples} samples but returned {len(helms)}"
        )
    with output_path.open("w", encoding="utf-8") as f:
        for helm in helms:
            f.write(f"{helm}\n")

    record = {
        "epoch": epoch,
        "seed": epoch_seed,
        "num_samples": len(helms),
        "sample_file": str(output_path),
        "alpha_win": float(alpha_win),
        "generation": {
            "max_length": gen_cfg.max_length,
            "min_length": gen_cfg.min_length,
            "use_ddim": gen_cfg.use_ddim,
            "ddim_steps": gen_cfg.ddim_steps,
            "lambda_gpt": gen_cfg.lambda_gpt,
            "history_embedding_mode": gen_cfg.history_embedding_mode,
            "predict_ring_bonds": gen_cfg.predict_ring_bonds,
        },
    }
    with (output_dir / "manifest.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return output_path
