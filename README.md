# PepAR-Diff

PepAR-Diff, short for **Cyclopeptide Generation via AutoRegressive Latent Diffusion**, is a research codebase for HELM-based cyclic peptide generation. The framework combines autoregressive sequence modeling, latent diffusion denoising, ring-bond prediction, and downstream structure-aware evaluation to support peptide design workflows aimed at permeability- and docking-aware optimization.

The repository is organized for a publication-grade workflow: core model code lives in [`pepar_diff`](./pepar_diff), runnable entrypoints live in [`scripts`](./scripts), experiment settings live in [`configs`](./configs), curated assets live under [`data`](./data), and generated artifacts are written to [`outputs`](./outputs).

## Highlights

- Autoregressive latent diffusion for token-by-token HELM peptide generation.
- Dedicated cyclic-peptide fine-tuning with ring-bond prediction.
- DPO-based reward optimization integrating docking and permeability signals.
- Isolated permeability workflow via a separate `pepardiff-perm` environment.
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

PepAR-Diff uses two intentionally separated conda environments:

- `pepardiff`: main environment for preprocessing, embedding generation, training, generation, and docking-related evaluation
- `pepardiff-perm`: isolated environment for the permeability predictor, which depends on a separate legacy-compatible software stack

Create both environments from the repository root:

```bash
conda env create -f envs/pepardiff.yml
conda env create -f envs/pepardiff-perm.yml

conda activate pepardiff
pip install -e .

conda activate pepardiff-perm
pip install -e .
```

If the environments already exist, update them with:

```bash
conda env update -f envs/pepardiff.yml --prune
conda env update -f envs/pepardiff-perm.yml --prune
```

### Environment Usage

Use `pepardiff` for:

- `scripts/data/*`
- `python -m pepar_diff.embeddings.generator`
- `scripts/train/train_pretrain.py`
- `scripts/train/train_finetune.py`
- `scripts/train/train_dpo.py`
- `scripts/generate/*`
- `scripts/eval/export_train_vina_scores.py`
- `scripts/eval/evaluate_dpo_samples.py`
- `scripts/eval/eval_add/evaluate_rewards.py`
- `scripts/eval/evaluate_groundtruth_vina.py`

Use `pepardiff-perm` for:

- `scripts/eval/evaluate_permeability_scores.py`
- direct use of `pepar_diff.evaluation.Permeability`

This separation is recommended because the permeability workflow relies on a dedicated predictor environment and should not be mixed with the main training stack.

## Data Preparation

Run preprocessing from the repository root.

```bash
conda activate pepardiff

python scripts/data/prepare/prepare_chembl32_data.py
python scripts/data/prepare/prepare_cycpeptmpdb_data.py
python scripts/data/prepare/prepare_mergeCyclic.py
python scripts/data/prepare/prepare_prior_data.py
```

If Uni-Mol monomer embeddings need to be regenerated:

```bash
python -m pepar_diff.embeddings.generator
```

## Training Pipeline

### Pretraining

```bash
conda activate pepardiff
python scripts/train/train_pretrain.py
```

Default config: [`configs/training/pretrain.json`](./configs/training/pretrain.json)

### Cyclic fine-tuning

```bash
conda activate pepardiff
python scripts/train/train_finetune.py --config configs/training/finetune_cyclic.json
```

### Permeability-focused fine-tuning

```bash
conda activate pepardiff
python scripts/train/train_finetune.py --config configs/training/finetune_permeability_top1000.json
```

### DPO training

Permeability scoring should be exported first in `pepardiff-perm`, then consumed by DPO training in `pepardiff`.

```bash
conda activate pepardiff-perm
python scripts/eval/evaluate_permeability_scores.py \
  --input outputs/samples/dpo_train_data/combined_candidates.txt \
  --output outputs/samples/dpo_train_data/combined_candidates.perm.csv

conda activate pepardiff
python scripts/train/train_dpo.py --config configs/training/dpo.json
```

Default DPO config: [`configs/training/dpo.json`](./configs/training/dpo.json)

## Generation

Generate from the repository root with the base modeling environment.

```bash
conda activate pepardiff

python scripts/generate/generate_peptides.py --mode linear
python scripts/generate/generate_peptides.py --mode cyclic
python scripts/generate/generate_peptides.py --mode cpp
python scripts/generate/generate_peptides.py --mode dpo
```

Random baseline generation is also available:

```bash
python scripts/generate/generate_random_baseline.py
```

Generated sequences and score caches are written under [`outputs/samples`](./outputs/samples).

## Evaluation

### Permeability

```bash
conda activate pepardiff-perm
python scripts/eval/evaluate_permeability_scores.py \
  --input outputs/samples/dpo_generate_data/helm_dpo_samples.txt \
  --output outputs/samples/dpo_generate_data/helm_dpo_samples.perm.csv
```

### Docking / Vina cache export

```bash
conda activate pepardiff
python scripts/eval/export_train_vina_scores.py \
  --config configs/training/dpo.json \
  --sample_file outputs/samples/dpo_train_data/combined_candidates.txt
```

### Full sample evaluation

```bash
conda activate pepardiff
python scripts/eval/evaluate_dpo_samples.py
python scripts/eval/eval_add/evaluate_rewards.py --config configs/training/dpo.json
python scripts/eval/evaluate_validity_uniqueness.py
python scripts/eval/evaluate_full_metrics.py
python scripts/eval/eval_add/evaluate_sample_quality.py outputs/samples/dpo_generate_data/helm_dpo_samples.txt
```

## Docking Backend

PepAR-Diff uses Uni-Dock as the maintained docking backend. This path requires Linux and an NVIDIA GPU.

Setup instructions are documented in [`docs/unidock_gpu_setup.md`](./docs/unidock_gpu_setup.md).

## Verification Notes

During the repository refactor, the following lightweight checks were run:

- `conda run -n pepardiff ...` import checks for the base package and main training/generation/evaluation entrypoints
- `conda run -n pepardiff-perm ...` import checks for permeability modules and permeability scoring scripts
- `conda run -n pepardiff python -m compileall pepar_diff scripts`
