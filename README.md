# PepAR-Diff

PepAR-Diff, short for **Cyclopeptide Generation via AutoRegressive Latent Diffusion**, is a research codebase for HELM-based cyclic peptide generation. The framework combines autoregressive sequence modeling, latent diffusion denoising, ring-bond prediction, and downstream structure-aware evaluation to support peptide design workflows aimed at permeability- and docking-aware optimization.

The repository is organized for a publication-grade workflow: core model code lives in [`pepar_diff`](./pepar_diff), runnable entrypoints live in [`scripts`](./scripts), experiment settings live in [`configs`](./configs), curated assets live under [`data`](./data), and generated artifacts are written to [`outputs`](./outputs).

## Highlights

- Autoregressive latent diffusion for token-by-token HELM peptide generation.
- Dedicated cyclic-peptide fine-tuning with ring-bond prediction.
- DPO-based reward optimization integrating docking and permeability signals.
- Isolated permeability workflow via a separate `perm_env`.
- Uni-Dock based docking path for Linux + NVIDIA GPU environments.

## Repository Layout

```text
pepar_diff/            Core package
scripts/               Training, generation, preprocessing, evaluation, analysis
configs/training/      Pretraining, fine-tuning, DPO configs
configs/inference/     Inference-time configs
data/raw/              Source datasets
data/processed/        Processed HELM data, vocab, Uni-Mol embeddings
data/models/           Auxiliary predictive models
data/docking/          Docking receptor and reference ligand assets
outputs/samples/       Example/generated sequence outputs and cached scores
deprecated/            Archived legacy scripts and historical artifacts
```

## Environment Setup

Two conda environments are used intentionally.

### 1. Base modeling environment: `molformer_env`

The environment specification file is [`envs/pepardiff_env.yml`](./envs/pepardiff_env.yml). For compatibility with the existing workflow, it still creates the `molformer_env` conda environment.

```bash
conda env create -f envs/pepardiff_env.yml
conda activate molformer_env
pip install -e .
```

Use `molformer_env` for:

- data preprocessing
- Uni-Mol embedding preparation
- model pretraining and fine-tuning
- generation
- DPO training
- docking score export

### 2. Permeability environment: `perm_env`

Use the isolated permeability environment whenever you run permeability-related scoring code.

```bash
conda env create -f envs/perm_env.yml
conda activate perm_env
pip install -e .
```

Use `perm_env` for:

- `scripts/eval/export_permeability_scores.py`
- direct use of `pepar_diff.evaluation.Permeability`

## Data Preparation

Run preprocessing from the repository root.

```bash
conda activate molformer_env

python scripts/data/prepare_chembl32_data.py
python scripts/data/prepare_cycpeptmpdb_data.py
python scripts/data/extract_cyclic_peptides.py
python scripts/data/prepare_prior_data.py
```

If Uni-Mol monomer embeddings need to be regenerated:

```bash
python -m pepar_diff.embeddings.generator
```

## Training Pipeline

### Pretraining

```bash
conda activate molformer_env
python scripts/train/train_pretrain.py
```

Default config: [`configs/training/pretrain.json`](./configs/training/pretrain.json)

### Cyclic fine-tuning

```bash
conda activate molformer_env
python scripts/train/train_finetune.py --config configs/training/finetune_cyclic.json
```

### Permeability-focused fine-tuning

```bash
conda activate molformer_env
python scripts/train/train_finetune.py --config configs/training/finetune_permeability_top1000.json
```

### DPO training

Permeability scoring should be exported first in `perm_env`, then consumed by DPO training in `molformer_env`.

```bash
conda activate perm_env
python scripts/eval/export_permeability_scores.py \
  --input outputs/samples/helm_chembl32only_r1r2_cyclized.txt \
  --output outputs/samples/helm_chembl32only_r1r2_cyclized.perm.csv

conda activate molformer_env
python scripts/train/train_dpo.py --config configs/training/dpo.json
```

Default DPO config: [`configs/training/dpo.json`](./configs/training/dpo.json)

## Generation

Generate from the repository root with the base modeling environment.

```bash
conda activate molformer_env

python scripts/generate/generate_peptides.py --mode linear
python scripts/generate/generate_peptides.py --mode cyclic
python scripts/generate/generate_peptides.py --mode cpp
python scripts/generate/generate_peptides.py --config configs/training/dpo.json
```

Random baseline generation is also available:

```bash
python scripts/generate/generate_random_baseline.py
```

Generated sequences and score caches are written under [`outputs/samples`](./outputs/samples).

## Evaluation

### Permeability

```bash
conda activate perm_env
python scripts/eval/export_permeability_scores.py \
  --input outputs/samples/helm_dpo_samples.txt \
  --output outputs/samples/helm_dpo_samples.perm.csv
```

### Docking / Vina cache export

```bash
conda activate molformer_env
python scripts/eval/export_vina_scores.py \
  --config configs/training/dpo.json \
  --sample_file outputs/samples/combined_candidates.txt
```

### Full sample evaluation

```bash
conda activate molformer_env
python scripts/eval/evaluate_dpo_samples.py
python scripts/eval/evaluate_rewards.py --config configs/training/dpo.json
python scripts/eval/evaluate_validity_uniqueness.py
python scripts/eval/evaluate_full_metrics.py
python scripts/eval/evaluate_sample_quality.py outputs/samples/helm_dpo_samples.txt
```

## Docking Backend

PepAR-Diff uses Uni-Dock as the maintained docking backend. This path requires Linux and an NVIDIA GPU.

Setup instructions are documented in [`docs/unidock_gpu_setup.md`](./docs/unidock_gpu_setup.md).

## Verification Notes

During the repository refactor, the following lightweight checks were run:

- `conda run -n molformer_env ...` import checks for the base package and main training/generation/evaluation entrypoints
- `conda run -n perm_env ...` import checks for permeability modules and permeability scoring scripts
- `conda run -n molformer_env python -m compileall pepar_diff scripts`
