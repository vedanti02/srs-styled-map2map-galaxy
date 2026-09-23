# Research Scratch 2.0 — CMASS count-field patch GAN (mentor task 2, redone on valid data)

Date started: 2026-06-28

## Why 2.0 (the pivot)

Task-2 (GAN domain correction at patch level, then stitch; 2.1 naive vs 2.2 boundary-masked)
was first attempted on the `lagrangian_output_64` displacement patches. A decisive diagnostic
(`analysis/diag_hr_seam.py`) proved those `quijote-64` patches are **independent volumes**
(cross-boundary corr ≈ −0.02 vs +0.60 interior): assembling them bakes a discontinuity into the
*labels*, so the "seam" was a data artifact and 2.2 had nothing to fix. Full story in
`research_scratch.md` + memory `patch-seam-finding`.

**2.0 pivot:** drop displacement entirely; use the **CMASS-ILI Eulerian halo-count fields** — one
continuous, periodic 128³ box per sim that we crop ourselves, so patches are genuine neighbours and
the seam test is finally valid.

## Data (verified)
- `/data/group_data/universedata/cmass-ili/processed/{idx:04d}_{input,label}.npy`, idx==lhid, 2000 sims.
- Each (128,128,128) **halo count field**, periodic **L=1000 Mpc/h** (cell 7.81 Mpc/h). NGP binning
  (round-to-nearest), total counts == #halos (~122k). input=LR(FastPM), label=HR(N-body), matchIC.
- LR/HR paired: smoothed corr ~0.66@32³, ~0.87@16³ (voxel ~0.06 = sparsity, mean ~0.06–0.25/cell).
- θ (5 params) from `quijote/nbody/L1000-N128/{idx}/config.yaml` → cached `data/cmass_theta.npz`
  (+ global count scale 0.247). Splits: `processed/{train,val,test}_list.npy` = 1600/200/200.

## Design (confirmed with user)
- Model space = **log1p(n)**; non-neg recovered via expm1. Pk always on overdensity δ=n/n̄−1 (per-box).
- Primary A/B = **naive (pad=0) vs overlap boundary-masked (pad=8)**, GAN kept (mentor 2.1 vs 2.2).
- Patches: 2×2×2 of 64³ cores, periodic-wrap halo (the diagram's D=80,d=16,core=64). Overlap at load.
- Model selection: stitched-128³ Pk-RMS on δ. θ from cmass_theta.npz. Bulk cubes → node-local /tmp.
- Reuses (channel-agnostic): G_correct/D_const (in_chan=out_chan=1), TorchPk(is_density=True),
  extract_patch/stitch_patches/crop_interior, evaluate.py/nde.py (+ new `--theta-npz`).

## Build + validation (all on CPU, PASSED)
- **Phase 0** `data/patch_dataset_cmass.py`: + `to_model_space/to_counts/counts_to_delta` (np+torch),
  `transform` arg. log1p round-trips exactly; δ zero-mean per-box/per-sample.
- **Phase 1** `analysis/power_spectrum.py`: + `cube_pk_counts` + `--estimator counts`; `--theta-npz`
  in `inference/nde.py` + `evaluate.py`. Offline Pk(counts) == TorchPk(is_density) to 3e-6.
- **Phase 2** `train_patch_cmass.py`: 1-ch G/D, log1p, plain-L1 + Pk(δ) + GAN(R1); stitched-128 val.
  Tiny CPU run: adv/rec/pk finite, val computed, best.pt saved. ✓
- **Phase 3** `infer_stitch_cmass.py` (+`--hr-only/--lr-only/--max-sims`) and
  `analysis/eval_seam_cmass.py`. HR-identity sanity: pkRMS=0.0 (zero error); untrained SR:
  x64_excess=0.994 (≈1, no artificial seam — as expected for continuous data). ✓
- **Phase 4** slurm: `train_patch_cmass`, `eval_cmass_hr` (HR ref + **LR baseline**),
  `eval_cmass_wrapper`, `summarize_cmass`.
- **Phase 5** integrated cosmology-core smoke (`scratchpad/smoke_cmass.py`): HR→Pk(counts)→q_HR(NDE,
  theta-npz)→evaluate HR-vs-HR ⇒ **mean KL ≈ 0.004** (sampling noise) — full wiring correct. ✓

### Intermediate plots (runs/patch_cmass/smoke/)
- `data_pair.png` — LR vs HR projected count maps + (HR−LR): correlated structure, real correction signal.
- `pk_hr_lr.png` — mean P(k): FastPM≈N-body, LR slightly high at small k-tail (the signal to correct).
- `smokeSR_slice.png` — seam slice machinery (untrained model).

## Full run (Phase 6) — launched 2026-06-28
Jobs (ids in `.cmasschain.txt`): HR-ref(+LR baseline) ∥ train A (pad0) ∥ train B (pad8);
eval A/B afterany(train,HR); summary afterany(evalA,evalB). 30 epochs, chan_base_g=128.
Deliverable: `runs/patch_cmass/SUMMARY.md` + pkratio/posprofile/kl_bars figures. Key question:
on continuous data, is there any seam (x64_excess>1)? does overlap (B) beat naive (A)? does either
beat the LR baseline on KL(q_HR‖q_X)?

### HR-ref + LR baseline COMPLETED (8841136, 1h07) — at-scale validation PASSED
- 2000 Pk(counts) files, posterior_hr.pkl, split_sids (1600/200). /tmp healthy (4.5T free).
- **LR baseline KL(q_HR‖q_LR) = 0.0047** (per-param [0.004,0.006,0.004,0.006,0.004]).
  => FastPM halo Pk already gives ~the same cosmology posterior as N-body — very little headroom for
  the model at the Pk→θ level (matches LR≈HR Pk plot). NDE recovery σ~0.11, bias ~0.10 (Pk-only summary,
  intrinsic). So the cosmology KL is NOT a sensitive discriminator here; the seam metrics are the
  real test of 2.1/2.2. Both arms likely land near KL~0.005 too.
- Training A (naive) 8841137 + B (overlap) 8841138 RUNNING; eval+summary chained.

### Mid-training (~12h in): both healthy
- Arm A (naive): epoch 9, best val_pkRMS ≈0.108, ~71 min/epoch.
- Arm B (overlap pad8): epoch 5, best val_pkRMS ≈0.096, ~115 min/epoch (80³ inputs, slower).
- Both decreasing, no NaN, best.pt saving. Arm B ahead on stitched Pk-RMS so far (overlap context
  appears to help on continuous data — opposite of the broken lagrangian patches). Will confirm at eval.
- Note: background monitor shells get killed periodically (exit ~144); self-heal by relaunching on the
  failure notification. Slurm chain is independent and unaffected.

## RESULTS (test split n=200)
Final best.pt: Arm A (naive) val_pkRMS 0.0947 @ep18; Arm B (overlap pad8) 0.0963 @ep~4.

### Arm A (naive) — COMPLETE, clean
| metric | value |
|---|---|
| x=64 internal-seam excess | **0.9934 (≈1 → NO seam)** |
| seam-dist ratio (d0/d≥8) | 1.173 |
| stitched Pk RMS (log10) | 0.0637 |
| KL(q_HR‖q_SR) mean | **0.0033** |
- KL per param Om/Ob/h/ns/s8 = .0038/.0042/.0032/.0023/.0030 — **beats LR baseline (0.0047)** on every
  param (model correction helps, modestly; ns improves most 0.0057→0.0023). SR bias ≈ HR bias (~0.10,
  the Pk-only NDE recovery floor). Visual `seam_cmassA_naive_slice.png`: SR≈HR, no stripe at x=64.
- KEY: on continuous count data, naive stitch has NO internal seam (x64≈1) — the clean result the
  broken lagrangian patches could never give. The seam_ratio 1.17 is the generic plane effect, not a seam.

### BUG + fix (Arm B first eval)
Arm B's first eval gave all-NaN: float16 overflow. The overlap model occasionally predicts a log1p-space
value >11 at a rare voxel → expm1 >65504 → inf in the float16-saved cube → one bad sim NaN-poisons the
aggregates (and nflows spline assert fails in q_SR). Training val never showed it (float32). FIX:
clamp SR counts to a physical cap (CMAX=200; HR max ~12) + nan_to_num at save in infer_stitch_cmass,
plus defensive nan_to_num on load in eval_seam_cmass. Re-ran Arm B eval + summary.

### FINAL RESULTS (200 test boxes) — both arms complete
Seam: A x64=0.9934 ratio 1.173 PkRMS 0.0637; B x64=0.9998 ratio 1.003 PkRMS 0.0678. Both seamless.
KL(q_HR‖q_X) mean: LR 0.0047, A 0.0033, B 0.0033 (both beat LR; A≈B). Bias ~0.10 all, matched HR/A/B.
=> 2.1 ≈ 2.2: no real seam on continuous data, overlap adds nothing. Both beat raw LR slightly.
SR-vs-HR match (Arm A, analysis/match_stats_cmass.py, 16 boxes): total count ratio 0.965±0.172,
void frac HR 0.833 / SR 0.855, std ratio 1.008±0.100 (NOT blurred), cross-corr r(k) 0.95 large / 0.40 small.
KEY INSIGHT: SR is NOT smoothed (variance + Pk match); it matches HR statistics + large-scale structure
but decorrelates at small scales (r→0.4) because that info isn't in LR (voxel corr 0.06, shot noise).
Only addressable flaw: ±17% per-box count scatter (a Poisson-count head would tighten it).
Deliverable: FINAL.md (repo root) + figures_cmass/ (box+patch Pk panels, projections, comparison, schematic).
Diag: analysis/diag_cmass.py (box+patch, HR/SR labels); scripts/diag_cmass.slurm.
