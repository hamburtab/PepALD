# WP-DPO loss ablation

By default, this runner trains only the Standard DPO ablation
(`alpha_win=0`) for case 1 and case 2 over eight independent-data optimization
rounds. The existing WP-DPO main models are not retrained. Use `--arms both`
only when a fresh Standard-DPO-versus-WP-DPO comparison is required.

Each case starts directly from PepALD_perm, generates, docks, and pairs its own
candidates in every round, and resumes from its own previous-round DPO
checkpoint. Elite SFT and elite replay are inherited unchanged from each main
model config, so `alpha_win` remains the intended ablation variable. After
every DPO training epoch, it generates 100 samples while restoring all RNG
states afterward, so epoch sampling cannot perturb later training steps.

## Full two-case run

Run from the repository root in the normal PepALD/DPO environment:

```bash
python scripts/train/run_wp_dpo_ablation.py \
  --case all \
  --pepald_perm_checkpoint /absolute/path/to/PepALD_perm.pt \
  --seed 42 \
  --run_name wp_dpo_seed42 \
  --rounds 8 \
  --arms standard \
  --samples_per_epoch 100
```

In every round, each case generates 20,000 candidates using the GPU lists in
its base config. Both trained case chains use `alpha_win=0`.

Eight rounds are represented as inclusive round directories `r0` through `r7`:

```text
outputs/ablations/wp_dpo/<run>/<case>/rounds/standard_dpo_r0 ... r7
checkpoints/ablations/wp_dpo/<run>/<case>/rounds/standard_dpo_r0 ... r7
```

Each round checkpoint contains its own `dpo_data/`, preference-pair hashes,
training metrics, sampling trace, checkpoints, and epoch samples. A successful
run ends with `status: complete_verified` in `ablation_manifest.json`.

Per-epoch samples are written under each arm checkpoint directory:

```text
checkpoints/ablations/wp_dpo/<run>/<case>/rounds/standard_dpo_rN/epoch_samples/
```

Each file contains 100 HELM sequences. `epoch_samples/manifest.jsonl` records
the epoch, generation seed, `alpha_win`, generation settings, and sample path.
