# Research Scratch — Patch-wise GAN Domain Correction (mentor task 2)

Date started: 2026-06-12

## Goal

Mentor's task 2 (GAN-based approaches):
- **2.1** Gather results on a *simple stitching* of patch-wise model outputs for a single simulation.
- **2.2** Train a patch-wise model *based on performance on the stitched result* — implemented the easy
  way the mentor suggested: **ignore the loss from boundary voxels that get cut out in post-processing**.

## Key data decision (IMPORTANT)

The pre-made patch folders `quijote-64/` & `quijotelike-64/` (`set{id}_pos_{i}_{j}_{k}`) were
**rejected as training data**:
- Grid sizes vary per sim: 1666 sims have 8 patches (2×2×2), 116 have 27, 114 have 1, 3 have 64, 1 has 125.
- ~100 sims have *mismatched* LR/HR patch counts (e.g. set2: 64 HR vs 8 LR patches) → unusable pairs.
- Adjacent-patch face continuity check FAILED: shared faces of `pos_0_0_0`/`pos_1_0_0` differ *more*
  (MAE 11.5) than random slice pairs (9.4). Tiling convention unverifiable; no generating script in repo.

**Decision: `stitched/` is the source of truth.** Tile each 64³ stitched cube on the fly into
**2×2×2 patches of 32³**. The tiling convention is then exact by construction, all 2000 sims usable,
stitched outputs are 64³ → drop directly into the existing eval pipeline (Pk, NDE, KL), and results
are directly comparable to whole-box v1–v7 models.

(Parallel action item for the humans: ask Xiaowen/mentor how `set*_pos_i_j_k` tiles into a box and
why grid sizes vary. If a verified large-box layout appears later, these scripts re-run with a
config change.)

## Normalization

Keep `CHAN_STD = [18,18,18,210,210,210]`. Patches are crops of the same stitched data; cropping does
not change per-voxel channel statistics, so no recompute is needed (unlike the rejected quijote-64
patches, which had different stats: disp std ≈ 12.5, vel std ≈ 375).

## Experiment design (A/B)

Both arms start from the v2 recipe (GAN + R1 + weighted L1 + Pk loss) = our SOTA GAN config.
Same data, same split (seed=0 on sim ids → train/val/test = 1600/200/200), same budget → any
difference is attributable to the boundary treatment.

| | Arm A: `patch_v1` (baseline, → mentor 2.1) | Arm B: `patch_v2_boundary` (→ mentor 2.2) |
|---|---|---|
| Train input | plain 32³ tiles (no halo) | halo'd tiles (32+2·pad)³ = 38³, periodic-wrap halo |
| Loss region | full 32³ | central 32³ only (outer pad=3 shell ignored) |
| L1 | weighted L1 on full patch | weighted L1 on cropped interior |
| Adversarial | D sees full 32³ | D sees cropped interior 32³ |
| Pk loss | Hann-windowed Pk on 32³ | Hann-windowed Pk on cropped interior 32³ |
| Inference | naive stitch (32³ tiles → 64³) | overlap-tile (pad=3 halo, crop, place) |

Notes:
- **Why Hann window for patch Pk:** a 32³ patch of a periodic 64³ box is itself NOT periodic.
  `TorchPk` assumes periodicity (FFT). Apodizing both fake & real with the same 3D Hann window
  (Feldman–Kaiser–Peacock-style) suppresses edge leakage; the fake-vs-real *difference* signal is
  preserved since both are windowed identically.
- **Why halo extraction is exactly correct here:** the full 64³ box IS periodic, so
  `mode='wrap'` halo voxels are the true neighbouring data for every patch (incl. edge patches).
- **Why seams appear at all (research basis):** fully-convolutional models lose context at patch
  borders (overlap-tile strategy, Ronneberger et al. 2015 U-Net). G_correct additionally uses
  *periodic* padding internally, so on a naive 32³ tile the model "sees" the patch's opposite face
  instead of the true neighbour — guaranteed wrong context at every seam. Boundary-masked training
  (valid-region loss; cf. nnU-Net, Isensee et al. 2021) lets the model spend capacity only on the
  voxels that survive post-processing.
- pad=3 matches the repo's existing halo convention (`lr2sr.py`).

## Evaluation plan

On held-out test sims (same 200-test split as v1–v7):
1. **Stitched-cube fidelity:** Pk(SR-stitched 64³) vs Pk(HR 64³) — periodic, no window needed.
   Metric: val-style pkRMS(log10) + Pk ratio curve.
2. **Seam statistics:** per-voxel |SR−HR| profiled by distance to nearest patch boundary
   (planes x,y,z ∈ {0,32}); seam-adjacent error vs interior error ratio. Plus |SR−HR| slice figures.
3. **Existing pipeline:** transform → Pk → NDE (sbi) → KL(q_HR‖q_SR) per cosmo parameter,
   comparable to v1–v7 numbers.
4. Reference points: whole-box v2 SOTA (upper bound), naive-stitched Arm A (the 2.1 deliverable),
   overlap-stitched Arm A (does halo alone fix it?), overlap-stitched Arm B (the 2.2 deliverable).

## Decisions log

- 2026-06-12: stitched/ chosen as source of truth; 2×2×2 × 32³ on-the-fly tiling. (see above)
- 2026-06-12: CHAN_STD kept; verified crops share whole-box statistics.
- 2026-06-12: A/B design fixed (table above); pad=3; Hann-windowed patch Pk.

## Implementation decisions (2026-06-12)

- **One trainer for both arms** (`train_patch.py`, `--pad 0` = Arm A, `--pad 3` = Arm B) instead of
  two scripts — identical code path guarantees the A/B differs ONLY in boundary treatment.
- **Receptive field note:** G_correct = 4 H-blocks × two 3³ convs → RF radius 8, with *periodic*
  padding inside the model. On a naive 32³ tile the model literally sees the patch's opposite face
  as context near every border — this is the seam mechanism. pad=3 doesn't cover RF 8; that is fine
  for Arm B because the masked loss trains the model under exactly the deployment context.
- **Stitched validation drives model selection**: every epoch, all 200 val sims are patch-inferred
  in the arm's own deployment mode (naive / overlap-tile), stitched to 64³, scored with L1 +
  pkRMS(log10) vs HR. best.pt = lowest stitched pkRMS. This implements "train … based on performance
  on the stitched result" at the model-selection level even for Arm A.
- Patch Pk loss: TorchPk(N=32, lbox=500, 16 bins), 3D Hann (periodic) window on fake & real.
- Deterministic val/inference noise: fixed seed-0 noise_list (same convention as transform.py).
- Eval protocol mirrors `scripts/eval_v5_wrapper.slurm` exactly (infer all 2000 → Pk → NDE n-train
  1600 → evaluate.py vs runs/baseline/nde/posterior_hr.pkl on test split) so KL numbers are directly
  comparable to v1–v7. Seam eval restricted to test split.

## File inventory (new)

- `data/patch_dataset.py` — PatchPairDataset + extract/stitch/crop helpers + self-test.
- `train_patch.py` — both arms.
- `infer_stitch.py` — naive | overlap stitching inference, transform.py-compatible outputs.
- `analysis/eval_seam.py` — seam profile, seam ratio, Pk ratio, slice figures, md table.
- `scripts/train_patch_armA.slurm`, `scripts/train_patch_armB.slurm`, `scripts/eval_patch_wrapper.slurm`.

## Run log

- 2026-06-12: `python -m data.patch_dataset` → PASS ×3 (bit-exact reconstruction; halo == periodic
  neighbourhood). 2000 sims × 8 patches = 16000 pairs (12800/1600/1600 train/val/test).
- 2026-06-12: CPU smoke (tiny model, 2 sims, 1 epoch): Arm A and Arm B both train; stitched val
  computed; infer_stitch naive + eval_seam chain OK (untrained model → seam ratio 0.98 ≈ 1, the
  expected null).
- 2026-06-12: submitted Slurm **8330684 (Arm A)** and **8330685 (Arm B)** — 40 epochs, batch 16,
  L40S, v2 hyperparameters. Plan after training: eval wrapper × 3 = {A-naive (deliverable 2.1),
  A-overlap (ablation: does inference-time halo alone help?), B-overlap (deliverable 2.2)}.
- 2026-06-12 epoch 0: Arm A stitched val_pkRMS=0.2347 (974 s/epoch), Arm B **0.2297** (1226 s/epoch).
  Reference: whole-box v2 best = 0.2125 @ epoch 40. Both arms healthy (D balanced, rec falling);
  B ahead of A from the start. ETA: A ≈ 11 h, B ≈ 13.5 h.
- 2026-06-12/13: both arms finished 40 epochs clean. best.pt: **Arm A epoch 30** (val_pkRMS 0.2133),
  **Arm B epoch 35** (val_pkRMS 0.2133). Tied on the global Pk metric.
- 2026-06-13: eval pipeline (Slurm 8337016/17/18) on all 2000 sims → seam + Pk + NDE + KL. Clean
  (only a harmless sbi triangular_solve deprecation warning).

## RESULTS (2026-06-13)

### Seam artifacts (200 test sims, displacement MAE [Mpc/h])

| run | seam MAE d=0 | interior MAE d≥8 | seam ratio | stitched Pk RMS log10 |
|---|---:|---:|---:|---:|
| Arm A · naive stitch (2.1)      | 5.388 | 4.669 | **1.154** | 0.190 |
| Arm A · overlap-tile pad3 (abl) | 5.326 | 4.669 | **1.141** | 0.193 |
| Arm B · boundary-masked pad3 (2.2) | 5.276 | 4.631 | **1.139** | 0.191 |

Error-vs-distance profile (key rows): A-naive d0=5.388 d1=5.322 d2=5.185 … plateau d7=4.700.
The seam is confined to d=0–5 voxels (≈ the RF radius). A-overlap(pad3) ≈ A-naive for d≥2 — it
only fixes the outermost 1–2 voxels, because **pad=3 ≪ G_correct receptive-field radius = 8**.
Arm B is uniformly ~0.04 lower at all distances (slightly better model overall), a touch more at
the seam. **Visual:** `seam_patchA_naive_slice.png` shows a clear vertical discontinuity at x=32 in
the stitched SR field, absent in HR.

### Cosmology — KL(q_HR ‖ q_X), mean over 200 test sims (lower=better)

| param | LR | whole-box v2 | A-naive (2.1) | A-overlap | B-bmask (2.2) |
|---|---:|---:|---:|---:|---:|
| Ω_m | 63   | 0.137  | 0.153 | 0.173 | 0.159 |
| Ω_b | 75.8 | 0.0071 | 0.0050| 0.0053| 0.0067|
| h   | 0.12 | 0.0084 | 0.0129| 0.0096| 0.0139|
| n_s | 4897 | 0.0323 | 0.0277| 0.0379| 0.029 |
| σ_8 | 258  | 0.150  | 0.113 | 0.179 | 0.329 |
| **mean** | 1059 | 0.0668 | **0.0623** | 0.0809 | 0.108 |

**Headline:** patch-train + stitch reaches whole-box-v2 cosmology (A-naive mean KL 0.062 vs v2 0.067)
— the seam is cosmologically negligible. Inter-arm KL spread (esp σ_8: 0.11/0.18/0.33) tracks NDE
training noise, not the boundary treatment (global stitched Pk RMS is ~0.19 for all three). So KL is
NOT a sensitive probe of seams; the seam-ratio metric is.

### Why boundary masking under-delivered — initial hypothesis (REFUTED)
First hypothesis: pad=3 < G_correct RF radius 8, so interior outer voxels still see periodic-wrapped
wrong context; fix = larger halo. **Refuted** by the pad=8 ablation (Slurm 8337124): inference with
pad=8 gave seam ratio 1.140 (A) / 1.141 (B) — indistinguishable from pad=3. Inference context does
not change the boundary error at all. (Inference-pad benefit saturates by pad=3 → effective RF small.)

### The real story — there is NO patch seam (controls + per-position profile)
Two controls reran the SAME seam metric on WHOLE-BOX models (no patches ever):
- whole-box v3: seam ratio 1.025  (looked like a clean null → drove the premature "seam is real" claim)
- whole-box **v2** (same recipe as the patch arms): seam ratio **1.141** — IDENTICAL to the patch models.

A same-recipe no-patch model scoring 1.141 means the metric was CONFOUNDED. The
distance-to-nearest-(pos%32)-plane metric lumps two different planes into "d=0":
  * pos 0/63 = the PERIODIC BOX EDGE (elevated error in every G_correct model — intrinsic, not patching)
  * pos 31/32 = the actual internal PATCH SEAM
The periodic-edge elevation dominated the ratio. v3 scored ~1.0 only because it's a weaker model with
uniformly higher interior error, so the same absolute edge bump is a smaller *ratio*.

Decisive test — per-axis error vs absolute grid position (Slurm 8337168, `seam_position_profile.png`,
40 test sims), patchA-naive vs whole-box v2:

| position | patchA naive | whole-box v2 |
|---|---|---|
| x=32 (internal patch seam) | 4.314 (0.928× interior) | 4.311 (0.930×) |
| x=0  (periodic box edge)   | 5.333 (1.147×)          | 5.290 (1.141×) |
| interior                   | 4.649                   | 4.637 |

The two profiles overlie across all 64 positions. **At the internal patch boundary (x=32) the patch
model has the SAME error as the whole-box model — slightly BELOW interior, no spike.** The only
elevated feature is the periodic box edge (x=0/63), shared by both and unrelated to patching. The
"vertical line at x=32" I thought I saw in the slice figure was the plot's axvline marker, not data.

## FINAL CONCLUSIONS (corrected)

1. **2.1 — Patch-train + naive-stitch works and has no measurable seam.** Tiling each 64³ cube into
   2×2×2 × 32³ patches, training the v2-recipe GAN per-patch, and naively stitching produces a 64³
   field whose error at the internal patch boundaries equals the patch interior and equals the
   whole-box model. Cosmology is preserved: mean KL(q_HR‖q_SR) = 0.062, matching whole-box v2 (0.067)
   and far below LR (1059). The patch model is actually slightly BETTER in patch interiors (MAE 4.65
   vs whole-box 4.66 overall, and lower at mid-patch) — focusing on 32³ sub-volumes helps locally.

2. **2.2 — Boundary masking gives no benefit here, because there is no seam to remove.** Implemented
   exactly as the mentor proposed (halo input, loss only on the kept interior). Arm B ≈ Arm A on every
   metric (seam ratio 1.139 vs 1.154 — and that residual difference is the periodic edge, not a seam;
   KL 0.108 vs 0.062 — within NDE-training noise). Overlap-tile inference and larger halos likewise do
   nothing measurable. This is a correct *negative* result for this configuration.

3. **Methodological lesson (logged honestly):** the first metric confounded the periodic box edge with
   the patch seam, and using a *different-recipe* model (v3) as the control produced a false positive.
   The same-recipe control (v2) + per-position profile corrected it. Always control with the same
   recipe and separate the periodic edge from the internal seam.

4. **Scope / caveat (important for the mentor).** This is the 64³ box, 2×2×2 tiling, 32³ patches,
   same-resolution domain correction. The effect of patching is negligible here because (a) G_correct's
   effective receptive field is small (local residual corrections; inference-halo benefit saturates by
   3 voxels), so periodic-padding "wrong context" only touches a thin shell the model handles well, and
   (b) only one internal seam plane per axis. Seams may still matter for: smaller patches, a larger-RF /
   upsampling model (true 8× SR), or the real target geometry (256³, many patches). The pipeline
   (`patch_dataset` → `train_patch --pad` → `infer_stitch --mode` → `eval_seam` + `seam_position_profile`)
   re-runs on any of those with a config change. Recommended next step if pursuing large boxes: rerun the
   per-position profile there before investing in boundary-masked training.

## REAL-SCALE PLAN (2026-06-13, Saumya confirmed interpretation (b))

**Confirmed scope:** same-resolution domain correction. Model maps a low-fidelity 64³ patch
(`quijotelike-64`) → high-fidelity 64³ patch (`quijote-64`); NO upsampling. "Stitching" = reassemble
the corrected 64³ patches into the full box. The 64³-of-64³ proxy is superseded by the REAL 64³
quijote-64 patches assembled into the 128³ box.

**Why redo at real scale (not just a rescale):** the proxy tiled a *clean* 64³ cube, so patches joined
smoothly and there was no inherent seam. The real `quijote-64` patches carry a genuine per-patch
periodic-boundary convention (3× gradient jump at every patch face, = the box-edge jump). So a real seam
*may* exist here that the proxy couldn't show — this is the actual test of mentor 2.1/2.2.

**Data (verified):**
- 1666 sims with matched 2×2×2 LR&HR grids → assemble to **128³** box (lbox=1000 Mpc/h; one 64³ patch
  = 500 Mpc/h). Drop the 27/64/125-patch sims (too few, different box sizes).
- LR↔HR are PAIRED (corr 0.66 same-pos vs 0.42 mismatched; |HR−LR|≈6.8, std≈12.6). Domain correction
  well-posed.
- HR ground truth = assembled-128³ of `quijote-64` HR patches (self-contained; do NOT use `stitched/`,
  which is a different realization). θ = `style.npy` (5 params), present per patch.

**Steps:**
0. ✓ Pre-check pairing (done).
1. `data/patch_dataset_real.py`: load quijote-64/quijotelike-64 patches, filter to 1666 clean 2×2×2
   sims, assemble 128³, serve 64³ patches with optional periodic-wrap halo (pad); split by sim
   (seed=0). Reconstruction self-test (assemble→tile→stitch bit-exact).
2. Generalize `train_patch.py` to take patch/box sizes from the dataset (currently hardcoded 32/64).
   Train two arms, v2 GAN recipe:
   - Arm A `patch_real_v1` (pad=0): full-patch loss, naive stitch — mentor **2.1**.
   - Arm B `patch_real_v2` (pad=3): halo input, loss on kept interior only — mentor **2.2**.
   Stitched-128³ validation each epoch (vs assembled HR), best = lowest stitched pkRMS.
3. HR reference: assemble HR 128³ for all clean sims → Pk (lbox=1000, div estimator) → train NDE
   q_HR(θ|Pk). (New; existing posterior_hr is 64³ stitched, not comparable.)
4. Eval (reuse `infer_stitch.py`, `analysis/eval_seam.py`, `analysis/seam_position_profile.py`):
   - seam ratio + per-position profile at x=64 (the real internal seam) — does naive stitch show a
     spike here that boundary-masking removes? (THE 2.1/2.2 question at real scale.)
   - θ verification: Pk(SR-128³)→NDE q_SR; KL(q_HR‖q_SR) + bias vs θ_true; Arm A vs Arm B vs LR.
   - Pk(SR)/Pk(HR) ratio curve.
5. Honest comparison + conclusions; controls as needed.

**Reuses:** train_patch.py, infer_stitch.py, eval_seam.py, seam_position_profile.py, summarize_patch.py,
pk_torch.py, inference/nde.py, evaluate.py — all parametrized; main new code is the dataset + assembly.

**Open risk:** the per-patch 3× seam adds spurious power at the patch frequency in the 128³ Pk; it is
identical in HR & SR so it cancels in KL(q_HR‖q_SR), but absolute θ recovery may be degraded vs the
clean 64³ case. Will report.

### Real-scale run log (2026-06-13)
- Smoke tests PASSED: dataset reconstruction (bit-exact, 1666 sims, 128³); training end-to-end;
  eval_seam_real Pk path (pkRMS=0 on HR-as-SR); infer_stitch_real --hr-only dumps (6,128,128,128).
- Pragmatic tuning (logged): chan_base_g=128 (not 256) + 25 epochs — 64³ patches are 8× the proxy's
  voxels, so 256/40ep ≈ 80h. At ~62 min/epoch, 25 epochs ≈ 26h vs 23h wall → expect ~22 epochs;
  best.pt + save-every-5 guarantee a converged model (proxy plateaued by ep15).
- Training submitted: Arm A 8407065 (pad=0 naive), Arm B 8407066 (pad=3 boundary-masked).
  Arm A epoch 0: stitched-128 val_pkRMS=0.1358 (healthy; D balanced).
- Eval pipeline wired with Slurm deps (auto-runs, no polling): HR-ref 8411484 (parallel, no model
  needed); eval A 8411485 = afterany:trainA:HRref; eval B 8411486 = afterany:trainB:HRref.
- Files: data/patch_dataset_real.py, train_patch_real.py, infer_stitch_real.py,
  analysis/eval_seam_real.py, analysis/summarize_patch_real.py, scripts/{train_patch_real,
  eval_real_hr, eval_patch_real_wrapper}.slurm.
- INCIDENT (disk-full): HR-ref dumped ~42GB cubes to home NFS (101G, ~13G free) → hit 100% →
  killed Arm B training (shared FS, exit 1 no traceback) and the HR-ref job at ~800/1666. Arm A
  (different node) survived. FIX: all bulk cubes now write to node-local /tmp (5TB nvme); only tiny
  Pk(npz)/metrics/figures return to home. Added --hr-only to infer_stitch_real (HR dump w/o model).
- After fix: HR-ref 8413516 COMPLETED — 1666 Pk files, split 1332/168, q_HR NDE trained
  (posterior_hr.pkl 440KB); /tmp peaked 1.2T; home stayed 82%. θ pipeline validated end-to-end.
- Resubmitted: Arm B train 8413515; eval A 8413517 (afterany A-train+HRref), eval B 8413518.
  Validation eval 8415997 launched on Arm A epoch-0 ckpt to exercise the full wrapper early.
- VALIDATION EVAL (8415997, epoch-0 Arm A naive) COMPLETED in 44 min — full wrapper works end to end.
  PRELIMINARY (undertrained) but pipeline-final numbers:
  seam-dist ratio 1.429; **x=64 internal-seam excess 1.205**; x=0 periodic-edge 1.220; Pk RMS 0.104.
  KL(HR‖SR)=[0.033,0.012,0.012,0.015,0.226]; SR bias ≈ HR bias.
  KEY: unlike the proxy (x=64 excess 0.93 = NO seam), the REAL quijote-64 patches show a genuine
  internal seam (x=64 excess 1.205). So the real-scale test is meaningful and 2.2 has a real target.
  The decisive comparison: does Arm B (boundary-masked overlap) drive x=64 excess toward 1.0?

## REAL-SCALE FINAL RESULTS (both arms best.pt @ epoch 15, val_pkRMS ~0.122; test split n=168)
(patience monitor died ~04:07 before triggering; jobs ran to wall/epochs, evals auto-fired via afterany)

| run | seam-dist ratio (d0/d≥8) | x=64 internal-seam excess | x=0 edge (confounded) | Pk RMS |
|---|---|---|---|---|
| Arm A · naive (2.1)        | 1.4165 | 1.2057 | 1.2221 | 0.0840 |
| Arm B · boundary-mask (2.2)| 1.4149 | 1.2026 | 1.2183 | 0.0838 |

KL(q_HR‖q_SR) mean over test: Arm A 0.0645 vs Arm B 0.0765 (per-param s8 dominates: 0.263 vs 0.313).

CONCLUSIONS (honest):
1. There IS a real internal patch seam at REAL scale: ~21% excess displacement error at the pure
   internal x=64 plane vs interior (x64_excess 1.21), and ~1.4× error within d=0 of nearest seam.
   This is a genuine seam the clean-cube proxy could NOT produce (proxy x64≈0.93, ratio≈1.0).
2. Arm B (boundary-masked overlap, pad=3 context + masked loss + overlap inference) does NOT fix it:
   seam-dist ratio 1.4165→1.4149 (−0.1%), x64 excess 1.2057→1.2026 (−0.3%). Within noise. No benefit.
3. Arm B does not help posterior recovery either — slightly WORSE KL (0.0765 vs 0.0645), s8-driven.
   Pk RMS identical (0.084). => Mentor's 2.2 hypothesis NOT supported at this config.
4. Note x=0 is NOT a clean control (it's both periodic box edge AND a patch face); the trustworthy
   seam signal is x64_excess + seam-dist ratio. Earlier "x=0 control" framing was imprecise.
5. Likely why masking fails: the seam is baked into per-patch model OUTPUT (edge receptive-field
   truncation), not introduced by assembly averaging — so assembly-time overlap/masking can't repair
   it. pad=3 (3-voxel context) is also small. A real fix would be training-time: larger context pad
   so patch edges see true neighbors, or seam-aware reconstruction loss — not assembly masking.

## ROOT CAUSE (2026-06-17, decisive diagnostic — analysis/diag_hr_seam.py)
The seam is a DATA artifact, not a model artifact. Measured the GROUND-TRUTH field's own plane-to-plane
discontinuity (mean |f(n)-f(n-1)|) across 8 test sims:
  - quijote-64 patches assembled→128³ (HR target): disp jump @x/y/z=64 = 2.3-2.4x interior, @0 = 2.3x;
    vel ~1.5x. IDENTICAL in LR (quijotelike-64): ~2.5x. Interior smooth (1.0x baseline).
  - stitched/ 64³ cubes: jump at every plane incl. periodic edge = 0.92-0.98x => fully CONTINUOUS.
=> The quijote-64 64³ patches are NOT a coherent 128³ field: each is its own frame (independent
   periodic BCs). Assembling 8 of them bakes a real ~2.4x displacement discontinuity into the TARGET at
   every internal face + the periodic wrap. The model reproduces per-patch structure faithfully; the
   assembled-box seam is inherited from the data. extract_patch's pad halo is pulled FROM the
   discontinuous assembled box, so Arm B's "context" is itself the jump — why pad=3 couldn't help.
CONSEQUENCE: Task 2.2 (boundary masking) is moot for this data — no model-induced seam exists to remove;
smoothing across a genuine label discontinuity would corrupt the statistics. The proxy found "no seam"
precisely because it used stitched/ (continuous); switching to raw patches introduced the discontinuity.
RECOMMENDATION: (a) box-level cosmology on a coherent volume => use continuous data (stitched/ 64³, or
true 128³ quijote cropped) — patch-stitch of quijote-64 cannot yield a coherent box; (b) if per-patch
correction is the deliverable => evaluate per-patch (Pk/posterior per 64³), drop the 128³ stitch framing.

### Artifacts
- Code: `data/patch_dataset.py`, `train_patch.py`, `infer_stitch.py`, `analysis/eval_seam.py`,
  `analysis/seam_position_profile.py`, `analysis/summarize_patch.py`; Slurm in `scripts/`.
- Checkpoints: `checkpoints/patch_v1/best.pt` (Arm A), `checkpoints/patch_v2_boundary/best.pt` (Arm B).
- Results: `runs/patch/SUMMARY.md`, `seam_*.npz/.md`, `pkratio_all.png`, `seamprofile_all.png`,
  `seam_position_profile.png`, `metrics_*.npz`; per-run slice/profile figures.
- Logs: `logs/*_patch_*.log`, `logs/*_wb_v2_seam.log`, `logs/*_seam_pos.log`.
