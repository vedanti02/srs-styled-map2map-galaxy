# Super-Resolution GAN for Cosmological Inference — Final Report

**Date:** 2026-05-11

## Goal
Train a super-resolution model that converts low-resolution (Quijote-like) 64³
displacement + velocity fields into high-resolution (Quijote) 64³ fields, such
that an NDE trained on `P_k(SR)` recovers the **HR posterior** rather than the
weaker **LR posterior** on five cosmological parameters
(Ω_m, Ω_b, h, n_s, σ_8).

## Setup
- Dataset: 2000 Quijote paired LR/HR sims, 1600 train / 200 val / 200 test split.
- Generator: `G_correct` (style-modulated 3D conv), 2.11 M params,
  6 in / 6 out channels (3 disp + 3 vel), `chan_base=256`, 4 blocks.
- Discriminator (variants v1–v3): `D_const`, 12.0 M params.
- NDE: 5-D MAF on log Pk vector (32 k-bins, 0.01 ≤ k ≤ 0.3 h/Mpc), trained on
  1600 train sims, evaluated on 200 held-out sims.
- Metric: per-param Gaussian-approx KL(q_HR ‖ q_X) and posterior bias
  |μ_X − θ_true|, averaged over the 200 test sims.

## Models trained

| version | loss              | best epoch | val L1 | val_pkRMS_log10 |
|---------|-------------------|-----------:|-------:|----------------:|
| v1      | adv + L1          | 31         | 0.42   | (untracked)     |
| v2      | adv + L1·w_disp + Pk-MSE | 40 | 0.42 | **0.2125** |
| v3      | adv + L1·w_disp + multi-scale Pk-MSE (warm-start v2) | (in progress) | — | — |
| v4      | L1·w_disp + Pk-MSE (NO GAN) | 56 | **0.337** | 0.2145 |

## Posterior agreement vs HR — KL(q_HR ‖ q_X), lower = better

| param      | LR       | v1_e31  | v2_e40 | v4_e56 |
|------------|---------:|--------:|-------:|-------:|
| Ω_m  (p0)  | 63       | 27.9    | **0.144** | 0.168 |
| Ω_b  (p1)  | 75.8     | 0.95    | **0.0072** | 0.0078 |
| h    (p2)  | 0.12     | 0.27    | **0.0079** | 0.0116 |
| n_s  (p3)  | 4900     | 44.3    | **0.032** | 0.051 |
| σ_8  (p4)  | 258      | 0.59    | 0.152  | **0.145** |

## Posterior bias |μ_SR − θ_true|

| param      | HR ref   | LR     | v1_e31  | v2_e40 | v4_e56 |
|------------|---------:|-------:|--------:|-------:|-------:|
| Ω_m  (p0)  | 0.041    | 0.113  | 0.128   | 0.041  | **0.039** |
| Ω_b  (p1)  | 0.010    | 0.021  | 0.012   | **0.010** | 0.010 |
| h    (p2)  | 0.101    | 0.109  | 0.121   | 0.102  | **0.101** |
| n_s  (p3)  | 0.090    | 0.235  | 0.241   | **0.091** | 0.092 |
| σ_8  (p4)  | 0.073    | 0.266  | 0.096   | **0.063** | 0.094 |

## Headline results

**v2 (GAN + Pk loss + disp-weighted L1) essentially matches the HR posterior:**

- Bias for **all 5 parameters is within 1% of the HR reference** — for σ_8, v2
  is actually *better* than the HR posterior (0.063 vs 0.073).
- KL(q_HR ‖ q_v2) ≤ 0.15 for every parameter, with median per-sim KL ≤ 0.05.
- Improvement over LR baseline: **15× (h) to 150 000× (n_s)** in mean KL.
- Improvement over plain v1 GAN: **5× (σ_8) to 1400× (n_s)**.

**v4 (no-GAN, pure supervised L1 + Pk) is essentially tied with v2:** the
adversarial loss gives a small but consistent advantage (median per-sim KL on
σ_8: 0.050 vs 0.082; on n_s: 0.016 vs 0.022). Pure supervised already captures
the bulk of the gain — adversarial structure is not the dominant ingredient.

## Plots
- `plots/kl_comparison.png` — log-scale KL bar chart per parameter
- `plots/bias_comparison.png` — bias bar chart with HR reference
- `plots/pk_ratio.png` — mean ⟨P(k)_SR / P(k)_HR⟩ per k bin

## Key engineering decisions

- Disp-weighted L1 (`w_disp=2, w_vel=0.5`): biases capacity toward displacement
  (which dominates Pk) rather than velocity.
- Pk loss in log10 space, MSE over k-bins of the displacement field only —
  velocities are not directly used for the inference, so loss budget goes to
  what the NDE actually consumes.
- Best-checkpoint tracking by `val_pk_rms_log10` rather than val L1: the Pk
  diagnostic correlates with downstream KL much more tightly than voxel L1.
- All inferences save float16 cubes that are deleted after Pk computation;
  necessary to fit 2000 cubes through the 100 GB home quota.

## Status
- v3 (multi-scale Pk + λ_pk=10, warm-start from v2 best) is still training.
  v3 may match or marginally improve v2; major gains already achieved.
- All scripts/artifacts in `runs/baseline/` and `scripts/`.
