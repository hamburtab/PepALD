# Uni-Dock GPU Setup

This project now uses `unidock` as the only docking backend for Vina scoring.

## Important platform note

Upstream Uni-Dock officially supports:

- Linux
- NVIDIA GPU

The local development machine in this workspace is `macOS arm64`, so the actual GPU binary cannot be installed or validated here.
Use the steps below on the Linux/NVIDIA training machine that owns the real `pepardiff`.

## Install into `pepardiff`

From the project root:

```bash
bash scripts/setup/setup_unidock_linux.sh pepardiff
```

This script will:

1. Install `unidock` from `conda-forge`
2. Install `Uni-Dock/unidock_tools` in editable mode
3. Verify that `unidock` is on `PATH`

## DPO config

`configs/training/dpo.json` is wired to the GPU docking path:

- `unidock_binary: "unidock"`
- `unidock_batch_size: 64`
- `unidock_search_mode: "fast"`
- `dock_box_size: 30.0`

If GPU memory is tight, reduce:

- `unidock_batch_size`

If accuracy is too low, increase:

- `unidock_search_mode` from `fast` to `balance`

## Runtime behavior

The code path is now:

`scripts/train/train_dpo.py -> pepar_diff/vina/dock.py -> pepar_diff/vina/unidock_backend.py`

The Uni-Dock backend:

1. Converts each HELM to SMILES
2. Builds one 3D ligand per HELM with RDKit
3. Runs Uni-Dock ligand preparation so fragment/torsion info is embedded in the SDF
4. Batches ligands through `unidock --ligand_index`
5. Reads the first pose energy from each `*_out.sdf`

## Errors

There is no CPU fallback. If `unidock` is missing, the platform is unsupported,
or docking fails at runtime, training raises an error immediately.
