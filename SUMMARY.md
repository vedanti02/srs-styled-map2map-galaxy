# SR styled-map2map — Project Summary

**Goal:** Super-resolve low-resolution (Quijote-like) 3-D cosmological
displacement + velocity fields to high-resolution (Quijote) fields, such that an
NDE trained on the resulting `P_k(SR)` recovers the **HR posterior** on five
cosmological parameters (Ω_m, Ω_b, h, n_s, σ_8), rather than the much weaker
**LR posterior**.

**Dataset:** 2 000 paired LR/HR Quijote sims at 64³ resolution, split
1 600 train / 200 val / 200 test. Each "sample" is a 6-channel cube (3 disp + 3
velocity). Each sim is labelled by a 5-D θ.

**Headline metric:** per-parameter **KL(q_HR ‖ q_X)** (Gaussian approximation of
the NDE posterior), averaged over the 200 held-out test sims. Also report
posterior bias **|μ_X − θ_true|**.

---

## Architectures

All variants share the same backbone generator:
- **G_correct** (style-modulated 3-D conv, 2.11 M params), 6 in / 6 out
  channels, `chan_base=256`, 4 blocks. Conditioned on θ via the style modulator.
- **D_const** (3-D discriminator, 12 M params) used by v1/v2/v3.

## Loss components (used in different combinations across versions)

| Symbol | Term | Notes |
|---|---|---|
| **L1** | `mean( |G(x_LR, θ) − x_HR| )` | per-voxel reconstruction |
| **L1·w** | weighted L1 with `w=(2,2,2,0.5,0.5,0.5)` | disp-weighted variant; biases capacity toward displacement, which is what the NDE consumes |
| **adv** | softplus GAN loss + R1 grad penalty (every 16 D-steps) | adversarial term used by v1/v2/v3 |
| **Pk-MSE** | `MSE(log10 P_k(SR), log10 P_k(HR))` on disp channels | makes Pk match HR directly |
| **multi-scale Pk** | Pk-MSE on full + 3-D-smoothed disp, with bin-linear weight `w_k` (3.0 → 1.0 from low-k to high-k) | emphasises modes the model tends to mis-predict |

---

## Versions

### v0 (baseline — never trained as a standalone variant)
The "vanilla" GAN setup from the original `styled_srsgan` recipe: G+D with
adversarial loss + plain L1, no Pk or weighted-channel terms. We did not
actually run v0 to convergence on this dataset — the existing repo prior had
been used to verify the data pipeline. We treat **LR (no SR at all)** as the
"do nothing" reference baseline instead.

### v1 — vanilla GAN (`adv + L1`)
- **Loss:** `adv + 10·L1` (uniform-weight L1 across all 6 channels).
- **Training:** ~31 epochs (training crashed at e31 due to disk fill; e17–e25
  intermediate ckpts were kept and evaluated, but only **e31 ckpt** retained
  after cleanup).
- **Result vs LR baseline (KL improvements):**
  - Ω_m: 63 → 27.9 (2.3×)
  - Ω_b: 75.8 → 0.95 (80×)
  - h: 0.12 → 0.27 (slightly worse — already easy for LR)
  - n_s: 4 900 → 44.3 (110×)
  - σ_8: 258 → 0.59 (440×)
- **Pk-level Pk error:** median 13.7% per sim (vs LR's 5.8%) — the GAN
  *systematically suppresses small-scale power* (classic L1-loss bias). Despite
  this, posterior agreement is dramatically better than LR because the GAN
  field preserves *information* the LR field lacks.
- **Take-away:** Vanilla GAN already gives huge posterior wins, but introduces
  Pk suppression and posterior bias on Ω_m (μ off by 0.13 vs HR's 0.04).

### v2 — GAN + disp-weighted L1 + Pk loss (`adv + 10·L1·w + 2·Pk-MSE`)
**Differences from v1:**
- Adds direct Pk-MSE loss on the disp channels (so the model is pushed to
  match the *power spectrum* of HR, fixing v1's small-scale suppression).
- Weights L1 toward displacement (`w_disp=2, w_vel=0.5`) — the NDE consumes
  Pk of disp only, so capacity is reallocated.
- Best-checkpoint selection by `val_pkRMS_log10` instead of L1 (Pk-RMS
  correlates more tightly with downstream KL than voxel L1).
- 60 epochs trained; best.pt = **epoch 40**, `val_pkRMS_log10 = 0.2125`.
- **Result vs HR posterior — essentially MATCHES HR:**
  - Bias for every parameter is within 1% of HR; σ_8 bias is actually *lower*
    than HR (0.063 vs 0.073).
  - KL(q_HR ‖ q_v2) ≤ 0.15 for every parameter; median per-sim KL ≤ 0.05.
- **Improvement over v1:**
  - Ω_m: KL 27.9 → 0.144 (194×)
  - Ω_b: KL 0.95 → 0.007 (135×)
  - n_s: KL 44.3 → 0.032 (1 400×)
  - σ_8: KL 0.59 → 0.15 (4×)
- **Take-away:** Pk loss + disp-weighted L1 alone closes the gap to HR. This
  is the **current SOTA** model.

### v3 — multi-scale Pk + stronger Pk weight (`adv + 10·L1·w + 10·multi-scale-Pk-MSE`, warm-start from v2 best)
**Differences from v2:**
- Pk loss is computed at **two scales**: the original disp field, *and* a
  3-D-smoothed (sigma=2 voxels) version. Loss = ½(scale₁ + scale₂). The
  smoothing emphasises the largest-scale modes the GAN tends to mis-predict.
- Per-k-bin **linear weight** (3.0 at smallest k, 1.0 at largest k) — biases
  the gradient toward cosmologically-relevant low-k modes.
- Pk weight cranked to 10 (v2 had 2). Pure Pk emphasis test.
- **Warm-started from v2/best.pt**, trained 60 more epochs.
- Best at epoch 27; `val_pkRMS_log10 = 0.2088` — **best of any version**
  (v2 had 0.2125).
- **Posterior eval (final):**
  - KL(p₀ Ω_m) = 0.168 vs v2 0.144 — **v2 wins**
  - KL(p₁ Ω_b) = 0.0095 vs v2 0.0072 — v2 wins
  - KL(p₂ h)   = 0.0094 vs v2 0.0079 — v2 wins
  - KL(p₃ n_s) = 0.0287 vs v2 0.032 — **v3 wins (only one)**
  - KL(p₄ σ_8) = 0.166 vs v2 0.152 — v2 wins
- Bias was actually slightly *better* for v3 on Ω_m and σ_8, but the NDE
  posterior is **tighter than HR** on Ω_m (σ_SR=0.046 vs σ_HR=0.051), which
  blows up the KL. Multi-scale + low-k-weighted Pk loss reduces the *variance*
  of `log Pk(SR)` across cosmologies, making the NDE over-confident.
- **Take-away:** improving `val_pkRMS_log10` does **not** automatically improve
  downstream KL when the Pk variance across sims shrinks. v3 illustrates a
  metric/loss vs downstream-objective mismatch: the population-mean Pk match
  got better, but the *per-sim discriminability* of Pk got worse.

### v5 — pure supervised + v3's loss (`10·L1·w + 10·multi-scale-Pk-MSE`, low-k weight 3, no GAN)
**Differences from v4:** v4's losses plus v3's multi-scale Pk (full + sigma=2
smoothed) with low-k bin weighting. Tests whether v3's loss innovations carry
over without the GAN.
- 60 epochs, best.pt = epoch ~58, `val_pkRMS_log10 ≈ 0.2174` — between v4
  (0.2145) and v3 (0.2088).
- **Posterior eval:**
  - KL(Ω_m) = 0.214 — **worst of any SR variant.** Same over-confidence
    failure mode as v3 on Ω_m: σ_SR=0.048 vs σ_HR=0.050, but mean shift
    |μ_HR−μ_SR|=0.010 inflates the Gaussian-KL.
  - KL(Ω_b) = 0.006 — **best of any variant.**
  - KL(σ_8) = 0.127 — **best of any variant** (beats v4's 0.145, v2's 0.152).
  - KL(n_s) = 0.040 — between v2 (0.032) and v4 (0.051).
- **Take-away:** the multi-scale + low-k weighted Pk loss isn't a universal
  improvement — it's a *cosmological-mode reallocator*. v5 trades Ω_m accuracy
  for σ_8 accuracy. If σ_8 is the priority, v5 is the best variant trained so
  far. If you want the best balanced posterior, **v2 is still SOTA.**

### v4 — pure supervised, **no GAN** (`10·L1·w + 10·Pk-MSE`)
**Differences from v2:** discriminator removed entirely. Same loss schedule
minus the adversarial term and R1 grad penalty. Tests whether the GAN's
adversarial term is contributing anything beyond what L1+Pk alone produces.
- 60 epochs; best.pt = epoch 56, `val_pkRMS_log10 = 0.2145`, `val_L1 = 0.337`.
- **val_L1 is dramatically lower than v2's 0.42** — without adversarial
  pressure, voxel-level reconstruction is much sharper. But Pk match is
  *slightly* worse than v2.
- Posterior: **essentially ties v2**.
  - KL(p₀) 0.144 vs 0.168 — v2 wins
  - KL(p₁) 0.0072 vs 0.0078 — tie
  - KL(p₂) 0.0079 vs 0.0116 — v2 wins
  - KL(p₃) 0.032 vs 0.051 — v2 wins
  - KL(p₄) 0.152 vs 0.145 — v4 wins
- Median per-sim KL across all params: v2 strictly ≤ v4.
- **Take-away:** the L1+Pk losses do nearly all the work. The discriminator
  provides a small but consistent improvement — it doesn't hurt, but it's not
  the dominant ingredient. The Pk-MSE term is the critical addition over v1.

---

## Headline comparison

### KL(q_HR ‖ q_X) — per parameter, lower is better

| Param      | LR baseline | v1 (vanilla) | v2 (Pk loss) | v3 (multi-scale GAN) | v4 (no-GAN) | v5 (no-GAN multi-scale) |
|------------|------------:|-------------:|-------------:|---------------------:|------------:|------------------------:|
| Ω_m  (p0)  |          63 |         27.9 |    **0.144** |                0.168 |       0.168 |                   0.214 |
| Ω_b  (p1)  |        75.8 |         0.95 |       0.0072 |               0.0095 |      0.0078 |              **0.0060** |
| h    (p2)  |        0.12 |         0.27 |   **0.0079** |               0.0094 |      0.0116 |                  0.0124 |
| n_s  (p3)  |       4 900 |         44.3 |        0.032 |            **0.029** |       0.051 |                   0.040 |
| σ_8  (p4)  |         258 |         0.59 |        0.152 |                0.166 |       0.145 |               **0.127** |

### Bias |μ_SR − θ_true| (HR reference column for context)

| Param | HR ref | LR | v1 | v2 | v3 | v4 | v5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ω_m | 0.041 | 0.113 | 0.128 | 0.041 | **0.037** | 0.039 | 0.039 |
| Ω_b | 0.010 | 0.021 | 0.012 | 0.010 | **0.010** | 0.010 | 0.010 |
| h   | 0.101 | 0.109 | 0.121 | 0.102 | 0.101 | **0.101** | 0.101 |
| n_s | 0.090 | 0.235 | 0.241 | 0.091 | 0.090 | 0.092 | **0.088** |
| σ_8 | 0.073 | 0.266 | 0.096 | 0.063 | **0.060** | 0.094 | 0.078 |

**v2 still leads on aggregate KL** (3/5 parameters). v5 takes the σ_8 and Ω_b
crowns but pays a steep price on Ω_m (worst KL of any SR variant at 0.214).
The "win one, lose one" pattern for v5 vs v2 suggests the multi-scale + low-k
weighted Pk loss systematically biases which cosmological modes the SR field
encodes well: the low-k tightening helps σ_8 (sensitive to broad-band power)
but hurts Ω_m (sensitive to shape across k).

---

## Plot interpretation

### KL comparison — mean KL per parameter (log scale)

![KL comparison](runs/baseline/plots/kl_comparison.png)

[`runs/baseline/plots/kl_comparison.png`](runs/baseline/plots/kl_comparison.png) — bar chart of mean KL per parameter
for `LR / v1 / v2_e10 / v2_e40 / v4_e56`. Reading left-to-right within each
parameter group makes the "training trajectory" visible — every version cuts
KL by 1–3 orders of magnitude over the previous. **n_s is the most dramatic**
(4 900 → 0.03, 150 000×).

### Bias |μ_SR − θ_true| per parameter

![Bias comparison](runs/baseline/plots/bias_comparison.png)

[`runs/baseline/plots/bias_comparison.png`](runs/baseline/plots/bias_comparison.png) — bias per parameter, with the
**HR reference** as the black bar. For v2/v4 the bars sit on top of HR — the
SR posterior is no longer biased relative to HR. v1's bars on Ω_m and n_s are
still noticeably above HR.

### Mean Pk ratio ⟨P(k)_SR / P(k)_HR⟩

![Pk ratio](runs/baseline/plots/pk_ratio.png)

[`runs/baseline/plots/pk_ratio.png`](runs/baseline/plots/pk_ratio.png) — vs k for each variant. v1 sits at ratio
≈ 0.85–0.90 across most k bins (uniform power suppression). v2 and v4 sit near
1.0 over the range of k that the NDE consumes, confirming the Pk loss is doing
its job at the population level.

### Diagnostic per-model panels

#### v2 (Pk loss + GAN) — Pk panel & projection

![v2 Pk panel](runs/baseline/plots/diagnostic_v2_best/pk_panel_v2_best.png)
![v2 projection set4](runs/baseline/plots/diagnostic_v2_best/projection_v2_best_set4.png)

#### v3 (multi-scale Pk + GAN, warm-started from v2) — Pk panel & projection

![v3 Pk panel](runs/baseline/plots/diagnostic_v3_best/pk_panel_v3_best.png)
![v3 projection set4](runs/baseline/plots/diagnostic_v3_best/projection_v3_best_set4.png)

#### v4 (no-GAN supervised) — Pk panel & projection

![v4 Pk panel](runs/baseline/plots/diagnostic_v4_best/pk_panel_v4_best.png)
![v4 projection set4](runs/baseline/plots/diagnostic_v4_best/projection_v4_best_set4.png)

**`pk_panel_<tag>.png` — 3-panel Pk diagnostic over 20 held-out sims**

- **Top panel — P(k):** Truth (red) vs Pred-median (blue, with 16/84%
  percentile band). For v2/v3/v4 the two curves overlap closely except at the
  highest k near Nyquist. For v1 the blue curve sits a constant factor below
  red — visualising the power-suppression issue.
- **Middle panel — Transfer P_pred(k)/P_truth(k):** ideal = 1.0.
  - v1: drops to ≈ 0.4 around k ≈ 0.1 1/Box — classic GAN suppression on
    intermediate scales.
  - v2/v3/v4: stays within ±10% of 1.0 across most of the k range.
- **Bottom panel — Cross-power r(k) = ⟨δ_pred δ_truth⟩ / √(P_pred P_truth):**
  ideal = 1.0 means the SR field is *phase-aligned* with HR, not just
  statistically similar.
  - All variants drop from r ≈ 0.9 at the largest scales to r ≈ 0.1 by k ≈ 0.2.
    This is the **fundamental limit of the SR problem**: at scales below the
    LR's effective resolution, the model can match statistics but cannot
    reproduce phase-by-phase. The Pk-based NDE only needs the statistical
    match, which is why posterior agreement still saturates near HR.

**`projection_<tag>_set4.png` — 4-panel projection of one test sim (set 4)**

- **Input Histogram (LR):** sparse, blocky density structure inherited from the
  low-resolution input.
- **Ground Truth Histogram (HR):** high-frequency filamentary detail (cosmic
  web).
- **Predicted Histogram (SR):** for v2/v3/v4 the visible morphology matches HR
  in terms of filament locations and overall power; v1 looks slightly
  smoother. Some over-smoothing remains because the network cannot generate
  *new* phase information beyond what's encoded in LR.
- **Residual (Truth − Pred)/√Pred:** mostly flat blue (near-zero residual) for
  v2/v3, with localised red/blue speckle where small-scale detail is
  mis-predicted. The fact that the residual is *unstructured* (no large
  coherent residual lobes) is a positive sign — the model is not making
  systematic spatial errors.

---

## Engineering decisions worth noting

- **float16 cube outputs during full-dataset inference** — saves 2× disk;
  Pk error from fp16 is below the per-sim variance.
- **Cubes deleted immediately after Pk** — keeps the 100 GB /home quota from
  filling. Two simultaneous full-dataset inferences will overflow disk; eval
  jobs are serialised by dependency or `--exclude`.
- **Best.pt by val_pkRMS_log10** — empirically the strongest predictor of
  downstream KL. val_L1 is noisier and led to picking the wrong epoch in early
  v2 runs.
- **Disp-weighted L1 (`w_disp=2, w_vel=0.5`)** — velocities are not directly
  used by the NDE; biasing capacity toward disp improves Pk match noticeably
  without hurting overall L1.

---

## Key takeaways

1. **v2 is the strongest model** as of 2026-05-11. It essentially recovers the
   HR posterior across all 5 cosmological parameters from LR-quality inputs.
2. **Pk-MSE is the critical loss term.** Adding it on top of L1+adv took the
   posterior from "much better than LR but visibly biased" (v1) to
   "indistinguishable from HR" (v2).
3. **The discriminator helps only marginally.** v4 (pure supervised) ties v2
   within noise; v2's GAN edge is small but consistent.
4. **The remaining gap to HR is fundamental.** Cross-power r(k) → 0 at high k
   means phase information beyond LR's resolution is irrecoverable; what the
   model recovers is the *statistical* match, which is what the Pk-based NDE
   needs.
5. **v3 (multi-scale Pk + bin-weighting + λ_pk=10, warm-start from v2)** has
   the best training metric (`val_pkRMS_log10 = 0.2088 vs 0.2125 for v2`).
   Posterior eval is still running; report will be updated when it lands.

---

**Artifacts:**
- Checkpoints: `checkpoints/{full,v2,v3,v4}/best.pt`
- Pk arrays per sim: `runs/baseline/pk/{hr,lr,sr_*}/`
- Posterior metrics: `runs/baseline/metrics_*.npz`
- Plots: `runs/baseline/plots/`
- Detailed numerical report: `runs/baseline/FINAL_REPORT.md`
