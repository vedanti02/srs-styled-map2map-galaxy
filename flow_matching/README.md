# flow_matching — standalone conditional flow-matching corrector (no GAN)

Ablation of the training objective: same data, patching (64³ cores, pad 0, naive stitch),
model space (log1p), validation protocol and evaluation scripts as the GAN in
`train_patch_cmass.py`, but the corrector is a conditional flow-matching model trained with
a single MSE on velocities. No adversarial term, no θ conditioning, no GAN encoder.

## Why
The GAN's remaining gaps (halo-count deficit, flattened count-histogram tail, 5–10 % large-scale
transfer deficit) are consistent with one mechanism: the L1 loss returns the per-voxel median,
which drops every voxel whose halo probability is below 0.5. Only a sampler of p(HF | LF) can
restore those counts. Flow matching gives such a sampler with no adversarial game to oscillate.
Design references: Fotiadis et al. 2024 (2410.19814), Albergo et al. 2023 (2310.03725),
De Bortoli et al. 2023 augmented bridge matching (2311.06978); see `paper/flow_matching_lit_review.md`.

## Method
* Model space `y = (log1p(n + u) − mean)/std`, `u ~ U[0,1)` (dequantisation; inverse = floor, exact).
  The LF conditioning channel is the exact `log1p(n_LF)` in the same standardisation.
* Path `x_t = (1−t)x0 + t·y_HF`, loss `E‖v_θ(x_t, t, y_LF) − (y_HF − x0)‖²`.
* Source `x0` (`--source`, the main ablation axis):
  `gaussian` N(0,I) · `lf_white` y_LF + σ_z ε (σ_z = residual RMSE) · `lf_spectral` y_LF + coloured ε
  (noise power = residual power spectrum, so large scales stay at LF).
* Velocity net: 3D UNet, circular convs (shift-equivariant on the torus), FiLM time embedding,
  levels 64→32→16→8, attention at 8³, 17.4 M params by default (`--ch-mult 1,2,4,4`).
* Training: 8 patches per step, 48 cube symmetries (`--aug`), bf16, AdamW, warm-up, weight EMA.
* Sampling: Heun (default 32 steps) or Euler (50 = paper). Deterministic per (box, draw) seed.
* Validation each epoch on stitched 128³ val boxes: L1/voxel in log1p space (selection, `--select l1`),
  L1 of the ensemble mean, fair CRPS, generation spread, total-count error, P(k) RMS (held-out check only).

## Files
| file | role |
|---|---|
| `space.py` | model-space transform + exact inverse |
| `augment.py` | cube symmetry groups `none / los16 / full48` |
| `interpolant.py` | sources, path, residual spectrum, Euler/Heun integrators, low-k projection |
| `unet3d.py` | periodic 3D UNet |
| `data.py` | count-patch dataset (raw counts) with optional in-RAM uint8 cache |
| `common.py` | EMA, checkpoint loading, per-box generation, validation metrics |
| `train_fm.py` | training (auto-resume from `last.pt`) |
| `sample_fm.py` | writes `sr_{idx}.npy` (+ `_d{k}` draws, `set{idx}_transformed.npy` symlinks) |
| `eval_spread.py` | T(k), r(k), spread-skill ratio, rank histograms, count-hist scatter from K draws |
| `check_isotropy.py` | real-space vs redshift-space test → picks `full48` vs `los16` |
| `tests/test_fm.py` | CPU tests on synthetic cubes (`python -m flow_matching.tests.test_fm`) |
| `slurm/` | `smoke_fm`, `train_fm`, `tier1_fm`, `tier2_fm` |

## Run
```bash
python -m flow_matching.tests.test_fm                         # CPU, no data needed
sbatch flow_matching/slurm/smoke_fm.slurm                     # 20 sims, end-to-end on real data
sbatch --export=ALL,TAG=fm_gauss,SOURCE=gaussian flow_matching/slurm/train_fm.slurm
sbatch --export=ALL,TAG=fm_gauss flow_matching/slurm/tier1_fm.slurm     # after best.pt exists
sbatch --export=ALL,TAG=fm_gauss flow_matching/slurm/tier2_fm.slurm     # survivors only (~3 GPU-h sampling)
```
Checkpoints: `/data/user_data/vkshirsa/cmass-ili/models/<TAG>/{last,best,epoch_N}.pt`.
SR fields: `/data/user_data/vkshirsa/cmass-ili/sr_fields/<TAG>/` (Tier-1/2) and `<TAG>_draws/` (8 draws × 60 val).

## Pre-registered ablation matrix and bar
Arms (same UNet, epochs, selection): `gaussian` (primary) · `lf_white` · `lf_spectral` · `gaussian --no-cond`.
Bar, fixed before the first run:
1. PQMass χ²/dof inside the null bracket on **three consecutive** saved epochs, not one.
2. At least half of the LF total-count deficit restored.
3. Large-scale transfer (k < 0.1) within 0.98–1.02.
4. Summary cross-fid KL not worse than LF within seed scatter; field-level at the floor.

## Known risks
Small data for a from-scratch 3D generative model (hence the modest net + augmentation);
a likely few-% P(k) error at high k; ~2–3 GPU-h per 1600-box sampling pass at 32 Heun steps.
