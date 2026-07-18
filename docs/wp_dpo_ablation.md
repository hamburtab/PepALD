# WP-DPO loss ablation

This runner compares Standard DPO (`alpha_win=0`) with the configured WP-DPO
loss while holding the initial model, aligned preference pairs, hyperparameters,
data order, diffusion timesteps, and noise RNG stream fixed.

The runner starts both arms directly from PepALD_perm. It does not resume an
older DPO checkpoint and does not insert the multi-round elite-SFT stage.

## Full two-case run

Run from the repository root in the normal PepALD/DPO environment:

```bash
python scripts/train/run_wp_dpo_ablation.py \
  --case all \
  --pepald_perm_checkpoint /absolute/path/to/PepALD_perm.pt \
  --seed 42 \
  --run_name wp_dpo_seed42
```

By default, both cases generate 20,000 candidates using the GPU lists in their
base configs. Case 1 uses its configured WP coefficient (`0.2`) and case 2 uses
its configured coefficient (`0.8`). Override these with `--wp_alpha_case1` or
`--wp_alpha_case2` when needed.

To use already generated candidate pools while still rebuilding docking scores
and preference pairs:

```bash
python scripts/train/run_wp_dpo_ablation.py \
  --case all \
  --pepald_perm_checkpoint /absolute/path/to/PepALD_perm.pt \
  --candidate_file_case1 /path/to/case1_candidates.txt \
  --candidate_file_case2 /path/to/case2_candidates.txt \
  --seed 42 \
  --run_name wp_dpo_seed42
```

## Split preparation and training

The shared pair construction and the two training arms can be run separately:

```bash
python scripts/train/run_wp_dpo_ablation.py \
  --case all \
  --stage prepare \
  --pepald_perm_checkpoint /absolute/path/to/PepALD_perm.pt \
  --seed 42 \
  --run_name wp_dpo_seed42

python scripts/train/run_wp_dpo_ablation.py \
  --case all \
  --stage train \
  --pepald_perm_checkpoint /absolute/path/to/PepALD_perm.pt \
  --seed 42 \
  --run_name wp_dpo_seed42
```

Each case writes a shared `dpo_data/` directory containing `winners.txt`,
`losers.txt`, `preference_pairs.jsonl`, and SHA-256 hashes. After both arms
finish, the runner verifies that their pair hashes and per-step sampling traces
match. A successful run ends with `status: complete_verified` in
`ablation_manifest.json`.
