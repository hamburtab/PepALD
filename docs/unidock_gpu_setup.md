# Uni-Dock GPU Setup

This project now supports a `unidock` docking backend in addition to the original Python `vina` backend.

## Important platform note

Upstream Uni-Dock officially supports:

- Linux
- NVIDIA GPU

The local development machine in this workspace is `macOS arm64`, so the actual GPU binary cannot be installed or validated here.
Use the steps below on the Linux/NVIDIA training machine that owns the real `molformer_env`.

## Install into `molformer_env`

From the project root:

```bash
bash scripts/setup_unidock_linux.sh molformer_env
```

This script will:

1. Install `unidock` from `conda-forge`
2. Install `Uni-Dock/unidock_tools` in editable mode
3. Verify that `unidock` is on `PATH`

## DPO config

`configs/dpo.json` is wired to the new backend:

- `docking_backend: "unidock"`
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

`train_dpo.py -> Vina/dock.py -> Vina/unidock_backend.py`

The Uni-Dock backend:

1. Converts each HELM to SMILES
2. Builds one 3D SDF per ligand with RDKit
3. Batches ligands through `unidock --ligand_index`
4. Reads the first pose energy from each `*_out.sdf`

## Fallback

If you want to switch back to the old backend, set in `configs/dpo.json`:

```json
"docking_backend": "vina"
```
