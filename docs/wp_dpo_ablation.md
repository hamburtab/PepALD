# WP-DPO loss ablation

This runner compares Standard DPO (`alpha_win=0`) with the configured WP-DPO
loss over eight independent-data optimization rounds. Both arms start from the
same PepALD_perm checkpoint. Each arm generates, docks, and pairs its own
candidates in every round; from round 1 onward their datasets may diverge.

The runner starts both arms directly from PepALD_perm, resumes each arm from its
own previous-round DPO checkpoint, and does not insert elite-SFT or elite replay.
After every training epoch, each arm generates 100 samples from its current
model. Epoch generation uses the same seed in both arms and restores all RNG
states afterward, so it cannot perturb later training steps.

## Full two-case run

Run from the repository root in the normal PepALD/DPO environment:

```bash
python scripts/train/run_wp_dpo_ablation.py \
  --case all \
  --pepald_perm_checkpoint /absolute/path/to/PepALD_perm.pt \
  --seed 42 \
  --run_name wp_dpo_seed42 \
  --rounds 8 \
  --samples_per_epoch 100
```

In every round and arm, both cases generate 20,000 candidates using the GPU
lists in their base configs. Case 1 uses its configured WP coefficient (`0.2`)
and case 2 uses its configured coefficient (`0.8`). Override these with
`--wp_alpha_case1` or `--wp_alpha_case2` when needed.

Eight rounds are represented as inclusive round directories `r0` through `r7`:

```text
outputs/ablations/wp_dpo/<run>/<case>/rounds/standard_dpo_r0 ... r7
outputs/ablations/wp_dpo/<run>/<case>/rounds/wp_dpo_r0 ... r7
checkpoints/ablations/wp_dpo/<run>/<case>/rounds/standard_dpo_r0 ... r7
checkpoints/ablations/wp_dpo/<run>/<case>/rounds/wp_dpo_r0 ... r7
```

Each round checkpoint contains its own `dpo_data/`, preference-pair hashes,
training metrics, sampling trace, checkpoints, and epoch samples. A successful
run ends with `status: complete_verified` in `ablation_manifest.json`.

Per-epoch samples are written under each arm checkpoint directory:

```text
checkpoints/ablations/wp_dpo/<run>/<case>/rounds/standard_dpo_rN/epoch_samples/
checkpoints/ablations/wp_dpo/<run>/<case>/rounds/wp_dpo_rN/epoch_samples/
```

Each file contains 100 HELM sequences. `epoch_samples/manifest.jsonl` records
the epoch, generation seed, `alpha_win`, generation settings, and sample path.
