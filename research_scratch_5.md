# research_scratch_5.md: Flow+GAN follow-up (started 2026-09-28)

Main objective (unchanged): a posterior trained on corrected (SR) boxes must give the same answer on real HR
boxes as a posterior trained on HR, i.e. q_SR(theta | HR) matches q_HR(theta | HR). The decisive numbers are the
cross-fidelity KLs (summary level = P(k) NDE, field level = 3D CNN). P(k), coherence, PQMass, bispectrum and
halo counts are supporting diagnostics only.

Starting point (from 2026-09-24, see scratch/why/overnight_log.md and paper/section7_flowgan.tex):
- Flow+GAN (models/Flow+GAN, best.pt = epoch 15 of 20, CRPS selection): summary KL 0.064 +/- 0.010
  (LR 0.052, HR floor 0.034), field KL 0.024 (floor 0.018). Power within 1.1% RMS of HR.
- sigma8 bias gone (+0.21 widths, LR +0.19); remaining gap to LR = scatter (sigma8 scatter 0.45 vs LR 0.38),
  from extra box-to-box variation of the largest waves.

----------------------------------------------------------------------------------------------------------
## MORNING SUMMARY (written 2026-09-28 08:40, details in R1-R11 below)
1. Main objective, summary level: Flow+GAN trained to 40 epochs (CRPS pick, epoch 39) gives summary cross-fid KL
   0.047 +/- 0.008 vs LR 0.049 +/- 0.010 (8 seeds, 200 test boxes; HR floor 0.031). Final epoch 40: 0.055 +/- 0.011.
   First SR arm that is NOT worse than LR at summary level. Adjacent epochs differ by 0.008, so say "equal to LR
   within noise", not "beats LR". Mechanism: longer training removed the leftover s8 bias (+0.23 -> -0.005 widths)
   and part of the extra Om/ns scatter (R10).
2. Field level: all Flow+GAN checkpoints stay near the HR floor (0.024-0.027 vs 0.018/0.019; LR 0.47).
3. Step 1 (smooth large-scale guide during generation): works as designed on the fields (largest-scale scatter
   0.41 -> 0.35, LR 0.30) but no KL gain (0.069 vs 0.065). Not adopted. Hard LK swap hurts every flow even when k=0
   is kept (R3, R7).
4. Step 5 (tier-1): PQMass at epoch 15 FAILED (3.6); diagnosed as an integer-specific one-point artefact of the flow
   (2 vs 3 halo voxels) plus a too-long tail; not a decoder issue (decoder study R5). At 40 epochs PQMass PASSES
   (1.20; LR 0.83, GAN 2.22). Bispectrum 1.106 (LR 1.112). Halo totals -1.4% (LR -1.9%). All plots checked by eye.
5. Step 3 (calibration): draws under-dispersed (spread-skill 0.15), unchanged by longer training (R4, R11).
6. Step 6 (8 NDE seeds): done for all main arms; numbers moved slightly (LR 0.052 -> 0.049, patch flow 0.074 -> 0.087).
7. Step 4 (test-split boxes): generated for epoch 15, guide, crps40, final40 (sr_fields/<tag>, test split).
8. Not done (not requested): paper edits (step 7). Recommended paper number: crps40, with final40 and epoch 15 as a
   checkpoint-noise row. Frozen checkpoint: models/Flow+GAN/best_crps40_epoch39.pt.
9. Failures overnight: only my own (sbatch comma split, a wrong theta column in a diagnostic, a typo); all fixed and
   rerun, listed in D5, R7 and scratch/why/ignore_jobs.txt.

----------------------------------------------------------------------------------------------------------
## Log of steps and decisions (2026-09-28)

### D1. Resume Flow+GAN training to 40 epochs (user request)
- Why: P(k) validation error was still falling at epoch 20 (0.061 -> 0.055) even though CRPS was flat.
- Action: saved the first-run best as models/Flow+GAN/best_epoch15_first20.pt (and val_log_first20.jsonl);
  resumed from last.pt (epoch 20) to 40 epochs, same recipe (1 GPU, lr 1e-4, CRPS selection). Job 3248215.
- Epoch-trend check (validation only, boxes 16-55, the first 16 were used for selection): epochs 5, 10, 15, 20
  generated (jobs 3248216-19), analysed by scratch/why/epoch_trend.py (job 3248221). Uses one combined band
  per scale range to avoid the single-bin artefact found on 2026-09-24.

### D2. Checkpoint selection for the longer run (user chose option 3, fixed BEFORE any test result)
- Evaluate BOTH the CRPS-best epoch (models/Flow+GAN_crps40 -> Flow+GAN/best.pt) and the final epoch 40
  (models/Flow+GAN_final40 -> Flow+GAN/epoch_40.pt). Report both. Full chains (post_v2.sh):
  FGcrps40 (gen 3248243-52, lowk 3248253, NDE 3248254-55, summary 3248256, field 3248257-60),
  FGfinal40 (gen 3248261-70, lowk 3248271, NDE 3248272-73, summary 3248274, field 3248275-78).

### D3. Step 1: smooth large-scale constraint DURING generation (new code)
- Why: the remaining gap to LR is extra box-to-box scatter of the largest waves; the literature fixes large
  scales to the low-resolution input (Rouhiainen et al. 2311.05217) and uses data-consistency projections
  during sampling (e.g. DDNM, arXiv 2212.00490; masked velocity in Albergo et al. 2310.03725).
- Implementation (flow_matching/common.py: lowk_taper, make_lowk_guided; sample_fm.py --lowk-guide k1,k2):
  at each ODE step, predicted end point x1_hat = x + (1-t) v; its Fourier modes are pulled to the LR field's
  with weight w(k) = 1 for k < k1, cosine taper to 0 at k2, 0 above; velocity corrected to v - d/(1-t).
  The k = 0 mode (total halo count) is never changed. Unit-tested: end-point low-k equals LR, high-k and mean
  untouched.
- Choice k1 = 0.02, k2 = 0.05 h/Mpc: LR/HR coherence is about 0.97 for k < 0.03 and still 0.9 at 0.05, so LR's
  large waves are trustworthy there; fading out by 0.05 avoids forcing LR's small errors at larger k.
- Runs on the epoch-15 checkpoint (models/Flow+GAN_guide -> best_epoch15_first20.pt), so it is a clean
  "sampling only" change vs the epoch-15 numbers. Chain FGguide: gen 3248339-48, lowk 3248349, NDE 3248350-51,
  summary 3248352, field 3248353-56; scale-by-scale guided vs unguided job 3248393.

### D4. A likely flaw found in the 2026-09-24 hard low-k swap test, and a direct re-test
- The swap mask included k = 0, i.e. it replaced the SR box's total halo count by LR's. LR's per-box count is
  off by ~12% (std) box to box; with delta = n/nbar - 1 this rescales every small-scale mode (shot noise),
  which the posterior reads as Omega_m. This would explain why the swap helped the GAN (its count already equals
  LR's) but hurt every flow (their counts differ from LR's).
- Re-test "LK3" = same k < 0.05 swap but keeping the model's own k = 0 (scratch/why/lowk.py ... nodc):
  AfixLK3, fmLK3, FlowGANLK3 (NDE jobs 3248333, 3248335, 3248337). Previous LK2 numbers stay in the record.

### D5. Step 4: test-split boxes
- Epoch-15 Flow+GAN test boxes into sr_fields/Flow+GAN (jobs 3248322-25), and for crps40 / final40 / guide
  (3248378-86, 3248395-98). Needed so paper figures and the bispectrum use test boxes.
- Bug caught in my own submission: passing "--lowk-guide 0.02,0.05" inside sbatch --export splits at the comma.
  Cancelled 3248388-92 and resubmitted via the environment (3248395-99). The main guide chain was unaffected
  (it inherited GEN_EXTRA from the environment). sample_fm now prints lowk_guide and full_box in its header so
  every log shows what was used.

### D6. Step 5: tier-1 statistics (PQMass val 200, bispectrum test 32, halo-count deficit)
- scratch/why/tier1_any.slurm: FlowGAN_ep15 (3248326), Flow+GAN_crps40 (3248382), Flow+GAN_final40 (3248387),
  Flow+GAN_guide (3248399). Same scripts and boxes as all earlier arms (PQMass on 200 val boxes by protocol;
  note boxes 0-15 of val were also used for Flow+GAN checkpoint selection).

### D7. Step 3: calibration (spread-skill, rank histograms)
- 8 draws x 60 val boxes for epoch 15 (sr_fields/FlowGAN_ep15_draws, jobs 3248327-30), eval_spread (3248331).
  Old flow had spread-skill 0.20 (ideal 1).

### D8. Step 6: 8 posterior seeds instead of 4
- scripts/crossfid_reps.slurm now takes SEEDS (default "0 1 2 3"); seeds 4-7 for hr, LR, Afixfid,
  fm_gauss_nodq, FlowGAN, FGcrps40, FGfinal40, FGguide (jobs 3248370-77).
- analysis/seed_xfid_check.py now loops over seeds 0-15 (skips missing), so HR floor uses all seed pairs.
- Final combined summary table: job 3248394 (afterany on all NDE jobs), log logs/3248394_seedxfid.log.

----------------------------------------------------------------------------------------------------------
## Results
(filled in below as jobs finish; an automatic collector job also appends raw results at the end)

### D9. Automatic results collection
- scratch/why/collect_0928.slurm (job 3248410) appends every result log to this file when the queue ends, so the record is complete even if no one is watching.

### R1. Epoch trend (job 3248221, val boxes 16-55, combined bands k<0.03 / 0.03-0.1 / 0.1-0.2 / 0.2-0.4)
| | median P/P_HR | box-to-box scatter of P/P_HR | median coherence |
| --- | --- | --- | --- |
| LR | 1.007 1.025 1.037 1.012 | 0.302 0.376 0.610 1.132 | 0.968 0.795 0.352 -0.089 |
| epoch 5 | 0.945 0.966 0.975 0.998 | 0.410 0.423 0.565 0.787 | 0.958 0.868 0.676 0.378 |
| epoch 10 | 0.971 0.992 0.999 1.003 | 0.412 0.438 0.585 0.808 | 0.960 0.872 0.682 0.385 |
| epoch 15 | 0.974 0.993 1.006 1.011 | 0.414 0.440 0.590 0.815 | 0.960 0.873 0.682 0.386 |
| epoch 20 | 0.985 0.999 1.004 1.004 | 0.411 0.435 0.581 0.800 | 0.961 0.874 0.684 0.386 |

Reading: longer training still moves the mean large-scale power toward HR (0.945 -> 0.985) and coherence is
stable after epoch 10, but the large-scale box-to-box scatter (0.41 vs LR 0.30) does not change with epochs.
That scatter is what separates Flow+GAN from LR at the summary level (sigma8 scatter), so longer training alone is
not expected to close the gap; the guided-generation test (D3) targets it directly. Decision: keep both runs
as queued; no change.

### R2. Guided generation vs unguided, same epoch-15 checkpoint (job 3248393, val boxes 16-55)
Header check: logs/3248347_genv2.log shows `lowk_guide=0.02,0.05 full_box=True` on models/Flow+GAN_guide (epoch 15),
so the guide flag reached the jobs.
| | median P/P_HR | box-to-box scatter of P/P_HR | median coherence |
| --- | --- | --- | --- |
| LR | 1.007 1.025 1.037 1.012 | 0.302 0.376 0.610 1.132 | 0.968 0.795 0.352 -0.089 |
| epoch 15 | 0.974 0.993 1.006 1.011 | 0.414 0.440 0.590 0.815 | 0.960 0.873 0.682 0.386 |
| epoch 15 + guide | 0.979 0.991 1.006 1.010 | 0.349 0.432 0.585 0.809 | 0.970 0.875 0.684 0.388 |

Reading: the guide does what it was designed to do. Largest-scale box-to-box scatter drops 0.414 -> 0.349
(LR 0.302), about 60% of the gap closed, and largest-scale coherence rises to 0.970 (LR 0.968). Small scales are
untouched (bands 0.1-0.4 identical within 0.01), so the field-level advantage should survive. Scatter is not fully
LR's because the taper only fully locks k < 0.02 and fades by 0.05 (the 0.03-0.1 band barely moves, 0.440 -> 0.432).
The decisive test is the summary KL (job 3248352); no change to the queue.

### R3. LK3 (hard k<0.05 swap, keeping the model's own k=0) vs LK2 (swap incl. k=0) vs no swap (job 3248610, 200 test boxes, nde_v2)
Summary cross-fid KL, seeds 0-3 (LR, FlowGAN, HR floor now use 8 seeds):
| | no swap | LK2 (k=0 swapped too) | LK3 (k=0 kept) |
| --- | --- | --- | --- |
| GAN (Afix) | 0.194 +/- 0.033 (AfixRe) | 0.106 +/- 0.031 | 0.084 +/- 0.012 |
| patch flow (fm) | 0.072 +/- 0.006 (fmRe) | 0.163 +/- 0.015 | 0.150 +/- 0.031 |
| Flow+GAN | 0.065 +/- 0.011 (8 seeds) | 0.088 +/- 0.010 | 0.075 +/- 0.005 |
| LR | 0.049 +/- 0.010 (8 seeds) | | |
| HR floor | 0.031 +/- 0.009 (8 seeds) | | |

Reading: keeping k=0 helps every arm (D4 hypothesis is partly right: swapping the total count was costing
0.01-0.02), and the GAN with LK3 (0.084) is its best ever. But a hard swap still hurts both flows relative to no
swap. So the total-count effect is only part of the story; the remaining harm is plausibly the sharp seam at
k=0.05 where LR's large-scale modes meet the flow's own (the flow's small scales were generated to be consistent
with its own large scales, not LR's). The guided run (D3) makes the flow generate small scales consistent with
the constrained large scales, which is the literature-backed way to do this; its KL (job 3248352) is the test.
Numbers also slightly updated by 8 seeds: LR 0.049 (was 0.052 on 4), Flow+GAN 0.065 (was 0.064), floor 0.031.

### R4. Epoch-15 Flow+GAN tier-1 + calibration (jobs 3248326, 3248331); plots checked by eye
- Bispectrum (test 32): equilateral mid-k B/B_HR: SR 1.056, LR 1.112 (GAN fixcond earlier ~1.11, stage0_R ~1.05).
  Plot figures_cmass/bispectrum_FlowGAN_ep15.png: same layout as earlier arms; k<0.03 bins are noisy with 32 boxes
  (sign flips at k~0.012 in HR, LR and SR alike, i.e. sample noise, not a bug). k>0.03 ratios 1.02-1.2 for both.
- Halo counts (400 SR boxes): SR total deficit mean -1.18% (LR -1.85%, -2.12% on the same boxes); std 10.3% vs
  LR 8.7%; plot figures_cmass/halo_count_deficit_FlowGAN_ep15.png looks right (long left tail in LR too).
- Calibration (60 val x 8 draws): spread-skill ratio 0.15 (old flow 0.20; 1 = calibrated), U-shaped rank
  histograms at mid and high k, per-voxel draw std 0.43 of HR std. So draws are NOT calibrated: the differences
  between draws are much smaller than the error vs HR at the P(k) level. This matters for any use of multiple draws as
  an uncertainty estimate; our cross-fid protocol uses one draw per box, so it does not enter the KLs directly.
  Plot figures_cmass/spread_FlowGAN_ep15.png checked: T(k) median within 3% of 1 everywhere, coherence above LR
  for k>0.04, SSR decreasing with k.
- PQMass (val 200, protocol: 10 count-histogram bins + log mean): SR chi2/dof 3.59, frac(p<0.05) 0.90; LR 0.83.
  Worse than the GAN (2.22) and stage0_R (0.75). Plot figures_cmass/pqmass_FlowGAN_ep15.png is drawn correctly
  (LR histogram sits on the chi2_9 null, p-values of LR near uniform), so this is a real failure, not a plotting bug.

### D10. Diagnosing the PQMass failure (all by me on the login node, scripts in scratch/why/)
1. Per-feature mean shift (pq_feat.py): all bins shift < 0.3 HR-sd except the >20 bin (+0.77, sd ratio 2.1).
   Extreme tail check: HR max voxel count median 16, max 25, 0.14 voxels >20 per box (10% of boxes have one);
   Flow+GAN median max 19, max 40, 0.50 voxels >20 per box (30% of boxes); GAN max 17; LR (CHARM) max 12.
   So the flow over-produces a handful of very dense voxels.
2. Robustness variant with the top two bins merged (>11) (scratch/why/pqmass_top11.py, NOT the protocol):
   Flow+GAN 2.80, GAN 1.88, stage0_R 0.50, guided 2.82, LR 0.63. Still fails: the tail is not the whole story.
3. Histogram shape at FIXED total count (residual after a quadratic fit of each bin on log mean count, fit on HR):
   Flow+GAN has too few 2-halo voxels (-2.5 residual sd) and too many 3-halo voxels (+3.8 residual sd), and its
   shape varies too little box to box (residual sd 0.5-0.7 vs LR 1.1). In absolute terms this is ~2% of 2-halo
   voxels moved to 3; PQMass sees it because HR's shape at a given total count is extremely tight (residual sd is
   4% of the total sd for these bins). GAN and stage0_R have much smaller residuals (< 0.9 sd).
4. Hypothesis: decoder bias. The decode is n = round(z) with z = expm1(y) in count space, but the model makes its
   errors in log space. Around a true 2 the round-up boundary (ln 3.5 - ln 3 = 0.154) is closer than the round-down
   boundary (ln 3 - ln 2.5 = 0.182), so symmetric log-space errors push 2 -> 3 more than 2 -> 1; the same bias at
   every n also lengthens the tail. Test (decision taken autonomously; cheap, no retraining):
   - added --save-model-space to flow_matching/sample_fm.py (saves y_{idx}.npy float32); the sampler is seeded and
     deterministic, so the continuous output reproduces the stored epoch-15 boxes exactly (script checks this).
   - generate epoch-15 continuous outputs: 200 val (jobs 3248772-73), 40 train (3248774) into
     sr_fields/FlowGAN_ep15_ycont, from best_epoch15_first20.pt (best.pt was overwritten by the resumed run at 02:46).
   - scratch/why/decoder_study.py (job 3248776): decoders round (current), log (nearest integer in log space) and
     qm (thresholds fitted on the 40 TRAIN boxes so the pooled SR histogram equals HR's; empirical quantile mapping,
     Panofsky and Brier 1968, standard bias correction in statistical downscaling). Reports histogram ratios, total
     count, P(k) per band, and PQMass (protocol + merged) per decoder on val.
   - Decision rule, fixed now: adopt a new decoder only if it brings PQMass toward 1 AND does not move P(k) by more
     than 1% in any band or the total count by more than 0.5%. If adopted, the full chain (train/val/test re-decode,
     NDE, summary and field KL) is rerun, since the main objective is the posterior, not PQMass.

### R5. Decoder study result (job 3248776): decoder hypothesis REJECTED, keep round()
- Sanity: round() on the saved continuous output reproduces the stored epoch-15 boxes (1e-6 to 1e-5 of voxels differ,
  bf16 non-determinism), so the study is on the exact samples that were evaluated.
- Val 200, pooled histogram ratio to HR (bins 0,1,2,3,4,5,6-7,8-11,12-20,>20), total count, median P/P_HR
  (k<0.03, .03-.1, .1-.2, .2-.41, .41-.6), PQMass protocol / merged-top variant:
| decoder | hist/HR | N/N_HR | P/P_HR | PQMass | PQMass merged |
| --- | --- | --- | --- | --- | --- |
| round (current) | 1.006 0.981 0.911 1.044 0.931 0.892 0.898 0.990 1.344 3.536 | 0.996 | 0.986 0.999 1.003 1.008 1.014 | 3.56 | 2.80 |
| log-space nearest | 1.006 0.971 0.918 1.092 0.960 0.911 0.913 1.001 1.352 3.536 | 1.005 | 0.989 1.000 1.005 1.009 1.013 | 3.65 | 2.85 |
| quantile-matched (train fit) | 0.982 1.114 1.084 1.091 1.088 1.074 1.067 1.039 1.011 3.536 | 1.092 | 1.082 1.058 0.980 0.927 0.936 | 2.85 | 1.92 |
| LR | 1.006 0.978 0.949 0.936 0.934 0.924 0.935 0.947 0.807 0 | 0.990 | 1.010 1.026 1.035 1.013 0.970 | 0.83 | 0.63 |
- The log-space decoder changes nothing (the 3-bin excess even grows). Quantile matching improves PQMass a little
  but adds 9% halos on val and moves P(k) by up to 8% (its 0|1 threshold, z = 0.004, sits on a sharp spike of
  near-empty voxels, so it does not transfer from train to val). Both fail the decision rule fixed in D10.
  Decision: keep round(); no chain rerun.
- Where the artefact comes from (100 val boxes, pooled ratio to HR, bins 0,1,2,3,4,5,6,7,8,9-11,12-20,>20):
  GAN (Afix)   1.008 0.967 0.928 0.904 0.883 0.875 0.920 1.016 1.104 1.225 1.343 0
  Flow+GAN     1.008 0.970 0.893 1.023 0.904 0.863 0.856 0.873 0.908 0.965 1.210 3.188
  stage0_R     1.008 0.967 0.927 0.897 0.888 0.902 0.938 1.002 1.036 1.082 0.949 0
  patch flow   1.009 0.967 0.938 0.910 0.695 0.733 0.691 0.650 0.624 0.612 0.646 1.625
  LR           1.008 0.967 0.927 0.908 0.903 0.893 0.891 0.919 0.945 0.897 0.760 0
  The GAN input is smooth across 2-3-4; the 3-halo spike is created by the flow itself (the older patch flow has a
  different integer-specific step, between 3 and 4). So it is a property of the learned velocity field near the
  integer lattice in log space, not of the decoder or the GAN base. Fixing it needs a training change (e.g. the
  dequantised model space, which was tried in Sept and gave spurious halos, or a lattice-aware loss); not
  attempted overnight. Note for the paper: all arms including LR are ~10% off HR in bins 2-8; Flow+GAN's
  distinctive failures are the 2/3 imbalance and a too-long tail (>20 voxels 3x HR).
- Conclusion for PQMass: Flow+GAN fails the one-point PQMass test (3.6), worse than the GAN (2.2) and stage0_R
  (0.75), while being the best SR arm on the posterior KLs. PQMass here is a one-point test at fixed total count;
  it is not what the posterior uses.

### R6. Guided generation summary KL (job 3248352, 200 test boxes, 8 seeds): NO improvement
| | summary KL, 8 seeds |
| --- | --- |
| LR | 0.049 +/- 0.010 |
| Flow+GAN epoch 15 (unguided) | 0.065 +/- 0.011 |
| Flow+GAN epoch 15 + guide (FGguide) | 0.069 +/- 0.020 |
| FGguide + LK hard swap (4 seeds) | 0.080 +/- 0.003 |
| GAN (Afixfid) | 0.202 +/- 0.038 |
| patch flow (fm_gauss_nodq) | 0.087 +/- 0.027 (was 0.074 with 4 seeds) |
| HR floor | 0.031 +/- 0.009 |
Per seed (same NDE seed and same HR posterior): unguided 0.060 0.078 0.064 0.052 0.081 0.073 0.055 0.054, guided
0.073 0.079 0.048 0.053 0.079 0.112 0.058 0.054. Paired difference +0.004 (driven by seed 5); zero within noise.
Reading: the guide cut the largest-scale box-to-box scatter by 60% of the gap to LR (R2) but the summary KL did not
move. So that scatter is NOT what separates Flow+GAN from LR at the summary level; my D3 hypothesis was wrong.
The remaining 0.016 gap (1.5 sigma of the seed scatter) must come from elsewhere; candidates: the high-k shot-noise
regime, where the flow's one-point shape (R5) sets P(k) at k>0.3 (P/P_HR 1.008-1.014 there, LR 1.013/0.970), or
simply the NDE noise level. Field KL for the guided run pending (job 3248356).

### D11. Posterior-shift breakdown to find what really separates Flow+GAN from LR
- scratch/why/post_diag_any.py (job 3249092): for LR, FlowGAN, FGguide, FlowGANLK3, fm_gauss_nodq, Afixfid, per seed,
  posterior mean shift vs the HR-trained posterior on the same HR test box, in HR-posterior widths: mean (bias),
  box-to-box std (scatter), and width ratio, for Om and s8. Same quantity as the 2026-09-23 sigma8 diagnosis, now with
  8 seeds. Purpose: decide whether the remaining gap is bias, scatter, or width, and in which parameter.

### R7. Posterior-shift breakdown (job 3249151; 3249092 was wrong, see note), 200 HR test boxes, 8 seeds
Note: the first run (3249092) took theta columns 0,1 as (Om, s8); the order is Om, Ob, h, ns, s8 (checked against the
Quijote LH ranges in data/cmass_theta.npz), so column 1 was Ob. Discarded; 3249143 failed on a typo I introduced
(NameError), fixed and rerun as 3249151.
Shift = (posterior mean of q_X - posterior mean of q_HR) / HR posterior width, same seed, same HR box.
| | bias Om Ob h ns s8 | scatter Om Ob h ns s8 | width ratio Om Ob h ns s8 | 0.5 mean z^2 summed |
| --- | --- | --- | --- | --- |
| LR | +.007 -.017 -.023 +.138 +.182 | .165 .106 .078 .268 .381 | 1.00 1.02 1.01 1.09 0.94 | 0.175 |
| Flow+GAN | -.016 +.015 -.055 -.031 +.228 | .221 .106 .085 .342 .394 | 1.02 1.01 1.00 1.08 0.99 | 0.213 |
| FGguide | -.014 +.035 -.050 -.072 +.260 | .199 .116 .079 .343 .406 | 1.01 1.01 1.01 1.08 0.98 | 0.221 |
| FlowGANLK3 (4 seeds) | -.010 +.021 -.003 -.014 +.179 | .219 .124 .080 .337 .623 | 1.03 1.01 1.01 1.08 1.03 | 0.318 |
| patch flow | +.016 +.027 +.014 -.259 -.096 | .207 .109 .079 .324 .456 | 1.03 1.01 1.00 1.08 1.02 | 0.242 |
| GAN (Afixfid) | -.107 +.019 -.105 -.082 +1.055 | .181 .096 .082 .256 .519 | 1.01 1.02 1.02 1.11 0.89 | 0.776 |
| GAN calibrated (4 seeds) | -.040 +.011 +.018 +.034 -.449 | .183 .103 .077 .286 .476 | 1.01 1.00 1.01 1.11 0.97 | 0.308 |
(The last column ignores width terms and is only a rough proxy for the KL; it ranks arms like the real KL except
that it penalises the calibrated GAN's s8 overshoot more.)
Reading:
- The GAN's problem is one thing (s8 bias +1.06 widths), confirming the 2026-09-23 diagnosis.
- Flow+GAN vs LR: the excess is spread over three parameters: Om scatter (0.22 vs 0.17), ns scatter (0.34 vs 0.27)
  and s8 bias (+0.23 vs +0.18). No single dominant term, so no single targeted fix is indicated.
- The guide lowers Om scatter slightly (0.199) but raises s8 bias (+0.26); net zero, matching R6.
- The hard swap LK3 blows up s8 scatter (0.62): splicing LR's large-scale modes onto the flow's small scales makes the
  large/small-scale power ratio (the s8 lever) noisier box to box; this is the mechanism behind R3.
- Total halo count is NOT the cause of the extra Om scatter: per-box ln(N_X/N_HR) on test, robust scatter LR 0.0223,
  Flow+GAN 0.0215, GAN 0.0229; median offset LR -0.010, Flow+GAN -0.004. Flow+GAN's count is at least as good as LR's.
- Overall: Flow+GAN's remaining gap to LR (0.065 vs 0.049, about 1.5 sigma of seed scatter) is small and diffuse.
  With the evidence so far, I would describe Flow+GAN as "close to LR at summary level, better than LR at field
  level", not as "matching HR at summary level".

### R8. Guided run: field KL and tier-1 (jobs 3248353-56, 3248399); plots checked
- Field KL (3D CNN, SR seeds 100-102 vs HR refs): guided 0.0265 +/- 0.0057 (unguided epoch 15: 0.024; HR floor 0.018,
  pairwise retrain floor 0.019). Same within noise. SR closer to HR-ref than LR in 57% of combos.
- PQMass (val 200): 3.44 (unguided 3.59). Bispectrum equilateral mid-k B/B_HR: 1.024 (unguided 1.056, LR 1.112).
  Halo counts: mean -1.28%, same as unguided.
- Plot figures_cmass/pqmass_Flow+GAN_guide.png checked: same shape as epoch 15. Note on reading all PQMass plots: the
  LR p-values pile up near 1 (better than the null) because LR and HR boxes share initial conditions, so the two
  samples are paired, not independent. PQMass on paired boxes therefore under-rejects; an SR failure is real.
- Verdict on step 1 (guided generation): it does what it is designed to do at the field level (largest-scale scatter
  and coherence toward LR, small scales untouched, bispectrum slightly better) but gives no gain on the summary or field
  KL. Not adopted as the default; reported as a tested ablation.

### R9. Resumed training finished (job 3248215, 4h30, epochs 21-40)
- Val (first 16 val boxes) CRPS 0.0748 (epoch 20) -> 0.0740 (flat from epoch 30); P(k) RMS error 0.0549 -> 0.0517-0.0534;
  L1 0.1452 -> 0.1434; count error -0.5% to -2% (noisy). Gains are small and saturating.
- CRPS-best checkpoint = stored epoch index 38 (the 39th epoch; CRPS ties at 0.0740 from epoch 30 on, and the
  "<=" rule keeps the latest tie). Final = epoch_40.pt (index 39). The two pre-registered picks (D2) are therefore
  adjacent epochs and should give nearly the same numbers; both are still evaluated as planned.
- Chains started automatically at 06:37 (FGcrps40 gens running, FGfinal40 queued).

### R10. MAIN RESULT: longer training closes the summary-level gap to LR (jobs 3248256, 3248274, 3248260, 3248278, 3248382, 3248387, 3251464)
Summary cross-fid KL, 200 HR test boxes, 8 NDE seeds (nde_v2):
| arm | summary KL | per seed | field KL (3D CNN) | PQMass (val 200) | bispectrum eq. mid-k B/B_HR |
| --- | --- | --- | --- | --- | --- |
| HR floor | 0.031 +/- 0.009 | | 0.018 (retrain floor 0.019) | | 1 |
| LR (CHARM) | 0.049 +/- 0.010 | .060 .062 .039 .047 .038 .058 .048 .037 | 0.47 +/- 0.78 | 0.83 | 1.112 |
| Flow+GAN epoch 15 (old best.pt) | 0.065 +/- 0.011 | .060 .078 .064 .052 .081 .073 .055 .054 | 0.024 | 3.59 | 1.056 |
| Flow+GAN epoch 15 + guide | 0.069 +/- 0.020 | | 0.027 | 3.44 | 1.024 |
| **Flow+GAN crps40 (epoch 39, CRPS pick)** | **0.047 +/- 0.008** | .046 .051 .042 .067 .039 .051 .042 .041 | 0.027 +/- 0.011 | 1.20 | 1.106 |
| Flow+GAN final40 (epoch 40) | 0.055 +/- 0.011 | .056 .059 .042 .067 .071 .060 .044 .043 | 0.027 +/- 0.015 | 1.20 | 1.111 |
| crps40 + hard LK swap (4 seeds) | 0.075 +/- 0.008 | | | | |
| GAN (Afixfid) | 0.202 +/- 0.038 | | | 2.22 | ~1.11 |
- Paired with LR (same NDE seed, same HR posterior): crps40 - LR per seed = -.014 -.011 +.003 +.020 +.001 -.007 -.006
  +.004, mean -0.001. final40 - LR mean +0.006. Both are statistically indistinguishable from LR; crps40 is the
  first SR arm that is not worse than LR at the summary level. Field level: all Flow+GAN checkpoints stay near the HR
  floor (0.024-0.027 vs 0.018/0.019) while LR is 0.47, so SR still wins decisively there.
- Honest caveat: crps40 and final40 are ADJACENT epochs yet differ by 0.008 in summary KL and 0.17 widths in s8 bias
  (below). So checkpoint-to-checkpoint noise is about the size of the seed noise. The pre-registered pick (D2) was
  "report both"; the fair statement is "0.047-0.055, equal to LR within noise", not "0.047 beats LR".
- Mechanism (posterior-shift breakdown, job 3251464, HR-posterior widths):
  | | bias Om Ob h ns s8 | scatter Om Ob h ns s8 |
  | --- | --- | --- |
  | LR | +.007 -.017 -.023 +.138 +.182 | .165 .106 .078 .268 .381 |
  | epoch 15 | -.016 +.015 -.055 -.031 +.228 | .221 .106 .085 .342 .394 |
  | crps40 | +.032 +.019 +.002 -.005 -.005 | .177 .115 .079 .301 .369 |
  | final40 | +.012 +.001 -.015 -.051 +.167 | .181 .108 .077 .317 .364 |
  Longer training removed the residual s8 bias (crps40) and cut the extra Om and ns scatter (0.22 -> 0.18, 0.34 -> 0.30).
  Scale by scale (val 16-55): median large-scale power 0.974 -> 0.997 (k<0.03) and 0.993 -> 1.005 (0.03-0.1),
  coherence at k 0.1-0.2 0.682 -> 0.699; the largest-scale box-to-box scatter barely moves (0.414 -> 0.405).
  So the gap was mostly the small large-scale power deficit (the s8 lever found on 09-23), not the scatter; this
  also explains why the guide (which targeted scatter) did not help (R6).
- One-point (PQMass) now passes: 1.20 (LR 0.83, GAN 2.22, epoch 15 3.59); plot figures_cmass/pqmass_Flow+GAN_crps40.png
  checked (p-values close to uniform, mild excess at small p). Pooled histogram ratio to HR (bins 0..8, 9-11, 12-20,
  >20): 1.008 0.960 0.972 0.855 0.975 0.894 0.867 0.896 0.906 0.879 0.956 1.562; max voxel count median 17, max 32
  (HR 16 / 25). An integer-specific wobble is still there but FLIPPED sign vs epoch 15 (3-bin now low, was high),
  so it is training noise near the integer lattice, not a fixed feature; the >20 tail shrank from 3.2x to 1.6x HR.
- Bispectrum is back to LR's level (1.106 vs 1.112); epoch 15 was 1.056. Not a posterior input; note only.
  Plots bispectrum_Flow+GAN_crps40.png and halo_count_deficit_Flow+GAN_crps40.png checked (same layout as others;
  halo totals mean -1.43%, LR -1.85%).
- Hard LK swap still hurts (0.075, 0.072): R3/R7 mechanism holds.

### D12. Follow-ups queued after R10 (decided autonomously)
- Froze the CRPS pick as models/Flow+GAN/best_crps40_epoch39.pt (copy) so later runs cannot overwrite it.
- Calibration (step 3) for crps40: 8 draws x 60 val boxes (jobs 3251508-11) -> eval_spread (3251512). Epoch 15 had
  spread-skill 0.15; check whether longer training changes it.
- Recommendation for the paper (not done, step 7 not requested): use crps40 as the Flow+GAN result, report final40 and
  epoch 15 alongside to show checkpoint noise, quote summary KL as 0.047-0.055 vs LR 0.049.

### Automatic collection Mon Sep 28 08:10:26 AM CDT 2026

Job states:
```
3248215           Flow+GAN_resume  COMPLETED   04:30:19 
3248216                     genep  COMPLETED   00:11:03 
3248217                     genep  COMPLETED   00:11:08 
3248218                     genep  COMPLETED   00:11:04 
3248219                     genep  COMPLETED   00:11:02 
3248221                   eptrend  COMPLETED   00:00:19 
3248243                     genv2  COMPLETED   00:39:01 
3248244                     genv2  COMPLETED   00:39:12 
3248245                     genv2  COMPLETED   00:39:13 
3248246                     genv2  COMPLETED   00:39:03 
3248247                     genv2  COMPLETED   00:39:14 
3248248                     genv2  COMPLETED   00:39:05 
3248249                     genv2  COMPLETED   00:39:12 
3248250                     genv2  COMPLETED   00:39:19 
3248251                     genv2  COMPLETED   00:19:44 
3248252                     genv2  COMPLETED   00:19:39 
3248253                    lowkv2  COMPLETED   00:16:47 
3248254            xfrep_FGcrps40  COMPLETED   00:04:29 
3248255          xfrep_FGcrps40LK  COMPLETED   00:04:14 
3248256                  seedxfid  COMPLETED   00:08:55 
3248257                   fieldv2  COMPLETED   00:03:16 
3248258                   fieldv2  COMPLETED   00:03:19 
3248259                   fieldv2  COMPLETED   00:03:18 
3248260                   fieldev  COMPLETED   00:00:18 
3248261                     genv2  COMPLETED   00:39:01 
3248262                     genv2  COMPLETED   00:39:03 
3248263                     genv2  COMPLETED   00:39:06 
3248264                     genv2  COMPLETED   00:39:13 
3248265                     genv2  COMPLETED   00:39:03 
3248266                     genv2  COMPLETED   00:39:08 
3248267                     genv2  COMPLETED   00:39:06 
3248268                     genv2  COMPLETED   00:39:06 
3248269                     genv2  COMPLETED   00:19:38 
3248270                     genv2  COMPLETED   00:19:42 
3248271                    lowkv2  COMPLETED   00:16:49 
3248272           xfrep_FGfinal40  COMPLETED   00:04:15 
3248273         xfrep_FGfinal40LK  COMPLETED   00:04:19 
3248274                  seedxfid  COMPLETED   00:09:21 
3248275                   fieldv2  COMPLETED   00:03:15 
3248276                   fieldv2  COMPLETED   00:03:15 
3248277                   fieldv2  COMPLETED   00:03:15 
3248278                   fieldev  COMPLETED   00:00:19 
3248322                     genv2  COMPLETED   00:09:54 
3248323                     genv2  COMPLETED   00:09:54 
3248324                     genv2  COMPLETED   00:09:54 
3248325                     genv2  COMPLETED   00:09:53 
3248326                     tier1  COMPLETED   00:01:22 
3248327                  gendraws  COMPLETED   00:23:37 
3248328                  gendraws  COMPLETED   00:23:27 
3248329                  gendraws  COMPLETED   00:23:28 
3248330                  gendraws  COMPLETED   00:23:24 
3248331                    spread  COMPLETED   00:01:54 
3248332                     lowk2  COMPLETED   00:09:32 
3248333             xfrep_AfixLK3  COMPLETED   00:04:29 
3248334                     lowk2  COMPLETED   00:09:31 
3248335               xfrep_fmLK3  COMPLETED   00:04:25 
3248336                     lowk2  COMPLETED   00:09:30 
3248337          xfrep_FlowGANLK3  COMPLETED   00:03:55 
3248339                     genv2  COMPLETED   00:39:19 
3248340                     genv2  COMPLETED   00:39:07 
3248341                     genv2  COMPLETED   00:39:08 
3248342                     genv2  COMPLETED   00:39:11 
3248343                     genv2  COMPLETED   00:39:06 
3248344                     genv2  COMPLETED   00:39:06 
3248345                     genv2  COMPLETED   00:39:07 
3248346                     genv2  COMPLETED   00:39:06 
3248347                     genv2  COMPLETED   00:19:42 
3248348                     genv2  COMPLETED   00:19:37 
3248349                    lowkv2  COMPLETED   00:16:48 
3248350             xfrep_FGguide  COMPLETED   00:04:35 
3248351           xfrep_FGguideLK  COMPLETED   00:04:26 
3248352                  seedxfid  COMPLETED   00:08:46 
3248353                   fieldv2  COMPLETED   00:03:18 
3248354                   fieldv2  COMPLETED   00:03:16 
3248355                   fieldv2  COMPLETED   00:03:16 
3248356                   fieldev  COMPLETED   00:00:20 
3248370                seeds47_hr  COMPLETED   00:04:30 
3248371                seeds47_LR  COMPLETED   00:04:33 
3248372           seeds47_Afixfid  COMPLETED   00:04:35 
3248373      seeds47_fm_gauss_no+  COMPLETED   00:04:44 
3248374           seeds47_FlowGAN  COMPLETED   00:04:21 
3248375          seeds47_FGcrps40  COMPLETED   00:04:19 
3248376         seeds47_FGfinal40  COMPLETED   00:04:28 
3248377           seeds47_FGguide  COMPLETED   00:05:32 
3248378                     genv2  COMPLETED   00:09:58 
3248379                     genv2  COMPLETED   00:09:55 
3248380                     genv2  COMPLETED   00:09:55 
3248381                     genv2  COMPLETED   00:09:55 
3248382                     tier1  COMPLETED   00:01:20 
3248383                     genv2  COMPLETED   00:09:53 
3248384                     genv2  COMPLETED   00:09:58 
3248385                     genv2  COMPLETED   00:09:58 
3248386                     genv2  COMPLETED   00:09:53 
3248387                     tier1  COMPLETED   00:01:34 
3248388                     genv2 CANCELLED+   00:00:00 
3248389                     genv2 CANCELLED+   00:00:00 
3248390                     genv2 CANCELLED+   00:00:00 
3248391                     genv2 CANCELLED+   00:00:00 
3248393                   eptrend  COMPLETED   00:00:14 
3248394                  seedxfid  COMPLETED   00:17:32 
3248395                     genv2  COMPLETED   00:09:52 
3248396                     genv2  COMPLETED   00:09:54 
3248397                     genv2  COMPLETED   00:09:53 
3248398                     genv2  COMPLETED   00:09:50 
3248399                     tier1  COMPLETED   00:01:27 
3248610                  seedxfid  COMPLETED   00:09:22 
3248772                     genv2  COMPLETED   00:19:40 
3248773                     genv2  COMPLETED   00:19:38 
3248774                     genv2  COMPLETED   00:07:54 
3248776                  decstudy  COMPLETED   00:01:35 
3249092                  postdiag  COMPLETED   00:03:53 
3249143                  postdiag     FAILED   00:00:10 
3249151                  postdiag  COMPLETED   00:04:02 
3251464                  postdiag  COMPLETED   00:03:00 
```

#### Epoch trend (val 16-55) (job 3248221)
```
40 held-out val boxes (16-55). Bands: ['k<0.03 (largest)', '0.03-0.1', '0.1-0.2', '0.2-0.4']
           | median P/P_HR per band            | box-to-box scatter of P/P_HR       | median coherence r
LR         | 1.007 1.025 1.037 1.012 | 0.302 0.376 0.610 1.132 | 0.968 0.795 0.352 -0.089
epoch 5    | 0.945 0.966 0.975 0.998 | 0.410 0.423 0.565 0.787 | 0.958 0.868 0.676 0.378
epoch 10   | 0.971 0.992 0.999 1.003 | 0.412 0.438 0.585 0.808 | 0.960 0.872 0.682 0.385
epoch 15   | 0.974 0.993 1.006 1.011 | 0.414 0.440 0.590 0.815 | 0.960 0.873 0.682 0.386
epoch 20   | 0.985 0.999 1.004 1.004 | 0.411 0.435 0.581 0.800 | 0.961 0.874 0.684 0.386
```

#### Guided vs unguided (val 16-55) (job 3248393)
```
40 held-out val boxes (16-55). Bands: ['k<0.03 (largest)', '0.03-0.1', '0.1-0.2', '0.2-0.4']
           | median P/P_HR per band            | box-to-box scatter of P/P_HR       | median coherence r
LR         | 1.007 1.025 1.037 1.012 | 0.302 0.376 0.610 1.132 | 0.968 0.795 0.352 -0.089
epoch 15   | 0.974 0.993 1.006 1.011 | 0.414 0.440 0.590 0.815 | 0.960 0.873 0.682 0.386
epoch guide | 0.979 0.991 1.006 1.010 | 0.349 0.432 0.585 0.809 | 0.970 0.875 0.684 0.388
```

#### FINAL summary table (job 3248394)
```
200 test boxes

per-seed summary cross-fid KL (q_source seed s vs q_HR seed s):
  LR      : [0.06, 0.062, 0.039, 0.047, 0.038, 0.058, 0.048, 0.037]  -> mean 0.049 +/- 0.010
  Afixfid : [0.218, 0.234, 0.137, 0.15, 0.254, 0.198, 0.213, 0.213]  -> mean 0.202 +/- 0.038
  AfixLK3 : [0.082, 0.104, 0.071, 0.081]  -> mean 0.084 +/- 0.012
  fm_gauss_nodq: [0.064, 0.067, 0.104, 0.062, 0.138, 0.114, 0.089, 0.059]  -> mean 0.087 +/- 0.027
  fmLK3   : [0.203, 0.124, 0.134, 0.138]  -> mean 0.150 +/- 0.031
  FlowGAN : [0.06, 0.078, 0.064, 0.052, 0.081, 0.073, 0.055, 0.054]  -> mean 0.065 +/- 0.011
  FlowGANLK3: [0.079, 0.082, 0.069, 0.071]  -> mean 0.075 +/- 0.005
  FGcrps40: [0.046, 0.051, 0.042, 0.067, 0.039, 0.051, 0.042, 0.041]  -> mean 0.047 +/- 0.008
  FGcrps40LK: [0.063, 0.087, 0.073, 0.077]  -> mean 0.075 +/- 0.008
  FGfinal40: [0.056, 0.059, 0.042, 0.067, 0.071, 0.06, 0.044, 0.043]  -> mean 0.055 +/- 0.011
  FGfinal40LK: [0.063, 0.073, 0.068, 0.083]  -> mean 0.072 +/- 0.007
  FGguide : [0.073, 0.079, 0.048, 0.053, 0.079, 0.112, 0.058, 0.054]  -> mean 0.069 +/- 0.020
  FGguideLK: [0.082, 0.084, 0.08, 0.075]  -> mean 0.080 +/- 0.003
  HR-floor : [0.047, 0.026, 0.028, 0.018, 0.038, 0.043, 0.037, 0.021, 0.026, 0.028]  -> mean 0.031 +/- 0.009
SEED_XFID_DONE
SEEDXFID_ALL_DONE
```

#### FGcrps40 summary (job 3248256)
```
200 test boxes

per-seed summary cross-fid KL (q_source seed s vs q_HR seed s):
  LR      : [0.06, 0.062, 0.039, 0.047, 0.038, 0.058, 0.048, 0.037]  -> mean 0.049 +/- 0.010
  Afixfid : [0.218, 0.234, 0.137, 0.15, 0.254, 0.198, 0.213, 0.213]  -> mean 0.202 +/- 0.038
  fm_gauss_nodq: [0.064, 0.067, 0.104, 0.062, 0.138, 0.114, 0.089, 0.059]  -> mean 0.087 +/- 0.027
  fmFull  : [0.117, 0.087, 0.078, 0.068]  -> mean 0.088 +/- 0.018
  fmFullLK: [0.451, 0.417, 0.3, 0.374]  -> mean 0.385 +/- 0.056
  FGcrps40: [0.046, 0.051, 0.042, 0.067, 0.039, 0.051, 0.042, 0.041]  -> mean 0.047 +/- 0.008
  FGcrps40LK: [0.063, 0.087, 0.073, 0.077]  -> mean 0.075 +/- 0.008
  HR-floor : [0.047, 0.026, 0.028, 0.018, 0.038, 0.043, 0.037, 0.021, 0.026, 0.028]  -> mean 0.031 +/- 0.009
SEED_XFID_DONE
SEEDXFID_ALL_DONE
```

#### FGfinal40 summary (job 3248274)
```
200 test boxes

per-seed summary cross-fid KL (q_source seed s vs q_HR seed s):
  LR      : [0.06, 0.062, 0.039, 0.047, 0.038, 0.058, 0.048, 0.037]  -> mean 0.049 +/- 0.010
  Afixfid : [0.218, 0.234, 0.137, 0.15, 0.254, 0.198, 0.213, 0.213]  -> mean 0.202 +/- 0.038
  fm_gauss_nodq: [0.064, 0.067, 0.104, 0.062, 0.138, 0.114, 0.089, 0.059]  -> mean 0.087 +/- 0.027
  fmFull  : [0.117, 0.087, 0.078, 0.068]  -> mean 0.088 +/- 0.018
  fmFullLK: [0.451, 0.417, 0.3, 0.374]  -> mean 0.385 +/- 0.056
  FGfinal40: [0.056, 0.059, 0.042, 0.067, 0.071, 0.06, 0.044, 0.043]  -> mean 0.055 +/- 0.011
  FGfinal40LK: [0.063, 0.073, 0.068, 0.083]  -> mean 0.072 +/- 0.007
  HR-floor : [0.047, 0.026, 0.028, 0.018, 0.038, 0.043, 0.037, 0.021, 0.026, 0.028]  -> mean 0.031 +/- 0.009
SEED_XFID_DONE
SEEDXFID_ALL_DONE
```

#### FGguide summary (job 3248352)
```
200 test boxes

per-seed summary cross-fid KL (q_source seed s vs q_HR seed s):
  LR      : [0.06, 0.062, 0.039, 0.047, 0.038, 0.058, 0.048, 0.037]  -> mean 0.049 +/- 0.010
  Afixfid : [0.218, 0.234, 0.137, 0.15, 0.254, 0.198, 0.213, 0.213]  -> mean 0.202 +/- 0.038
  fm_gauss_nodq: [0.064, 0.067, 0.104, 0.062, 0.138, 0.114, 0.089, 0.059]  -> mean 0.087 +/- 0.027
  fmFull  : [0.117, 0.087, 0.078, 0.068]  -> mean 0.088 +/- 0.018
  fmFullLK: [0.451, 0.417, 0.3, 0.374]  -> mean 0.385 +/- 0.056
  FGguide : [0.073, 0.079, 0.048, 0.053, 0.079, 0.112, 0.058, 0.054]  -> mean 0.069 +/- 0.020
  FGguideLK: [0.082, 0.084, 0.08, 0.075]  -> mean 0.080 +/- 0.003
  HR-floor : [0.047, 0.026, 0.028, 0.018, 0.038, 0.043, 0.037, 0.021, 0.026, 0.028]  -> mean 0.031 +/- 0.009
SEED_XFID_DONE
SEEDXFID_ALL_DONE
```

#### Field FGcrps40 (job 3248260)
```
qhr_s100: recovery |mu-theta|(norm)=0.221
qhr_s101: recovery |mu-theta|(norm)=0.218
qhr_s102: recovery |mu-theta|(norm)=0.219
qhr_s103: recovery |mu-theta|(norm)=0.218
qhr_s104: recovery |mu-theta|(norm)=0.217
qhr_s105: recovery |mu-theta|(norm)=0.218
qlr_s100: recovery |mu-theta|(norm)=0.219
qlr_s101: recovery |mu-theta|(norm)=0.220
qlr_s102: recovery |mu-theta|(norm)=0.184
qlr_s103: recovery |mu-theta|(norm)=0.219
qlr_s104: recovery |mu-theta|(norm)=0.189
qlr_s105: recovery |mu-theta|(norm)=0.218
qlr_s106: recovery |mu-theta|(norm)=0.219
qlr_s107: recovery |mu-theta|(norm)=0.220
qsr_s100: recovery |mu-theta|(norm)=0.220
qsr_s101: recovery |mu-theta|(norm)=0.219
qsr_s102: recovery |mu-theta|(norm)=0.218
  SR: mean cross-fid KL over refs = 0.0267 +/- 0.0114
  LR: mean cross-fid KL over refs = 0.4711 +/- 0.7824
  HR: mean cross-fid KL over refs = 0.0180 +/- 0.0091
  SR<LR fraction over 144 (ref,SR,LR) combos: min=8% max=100% mean=60%
```

#### Field FGfinal40 (job 3248278)
```
qhr_s100: recovery |mu-theta|(norm)=0.221
qhr_s101: recovery |mu-theta|(norm)=0.218
qhr_s102: recovery |mu-theta|(norm)=0.219
qhr_s103: recovery |mu-theta|(norm)=0.218
qhr_s104: recovery |mu-theta|(norm)=0.217
qhr_s105: recovery |mu-theta|(norm)=0.218
qlr_s100: recovery |mu-theta|(norm)=0.219
qlr_s101: recovery |mu-theta|(norm)=0.220
qlr_s102: recovery |mu-theta|(norm)=0.184
qlr_s103: recovery |mu-theta|(norm)=0.219
qlr_s104: recovery |mu-theta|(norm)=0.189
qlr_s105: recovery |mu-theta|(norm)=0.218
qlr_s106: recovery |mu-theta|(norm)=0.219
qlr_s107: recovery |mu-theta|(norm)=0.220
qsr_s100: recovery |mu-theta|(norm)=0.219
qsr_s101: recovery |mu-theta|(norm)=0.218
qsr_s102: recovery |mu-theta|(norm)=0.219
  SR: mean cross-fid KL over refs = 0.0274 +/- 0.0149
  LR: mean cross-fid KL over refs = 0.4711 +/- 0.7824
  HR: mean cross-fid KL over refs = 0.0180 +/- 0.0091
  SR<LR fraction over 144 (ref,SR,LR) combos: min=8% max=100% mean=58%
```

#### Field FGguide (job 3248356)
```
qhr_s100: recovery |mu-theta|(norm)=0.221
qhr_s101: recovery |mu-theta|(norm)=0.218
qhr_s102: recovery |mu-theta|(norm)=0.219
qhr_s103: recovery |mu-theta|(norm)=0.218
qhr_s104: recovery |mu-theta|(norm)=0.217
qhr_s105: recovery |mu-theta|(norm)=0.218
qlr_s100: recovery |mu-theta|(norm)=0.219
qlr_s101: recovery |mu-theta|(norm)=0.220
qlr_s102: recovery |mu-theta|(norm)=0.184
qlr_s103: recovery |mu-theta|(norm)=0.219
qlr_s104: recovery |mu-theta|(norm)=0.189
qlr_s105: recovery |mu-theta|(norm)=0.218
qlr_s106: recovery |mu-theta|(norm)=0.219
qlr_s107: recovery |mu-theta|(norm)=0.220
qsr_s100: recovery |mu-theta|(norm)=0.220
qsr_s101: recovery |mu-theta|(norm)=0.218
qsr_s102: recovery |mu-theta|(norm)=0.222
  SR: mean cross-fid KL over refs = 0.0265 +/- 0.0057
  LR: mean cross-fid KL over refs = 0.4711 +/- 0.7824
  HR: mean cross-fid KL over refs = 0.0180 +/- 0.0091
  SR<LR fraction over 144 (ref,SR,LR) combos: min=19% max=100% mean=57%
```

#### Tier-1 job 3248326
```
  SR vs HR : chi2/dof median=3.59  mean_p=0.033  frac(p<0.05)=0.90   (match => chi2/dof~1, p~uniform)
  LR vs HR : chi2/dof median=0.83  mean_p=0.539  frac(p<0.05)=0.12   (match => chi2/dof~1, p~uniform)
saved figures_cmass/pqmass_FlowGAN_ep15.png
saved figures_cmass/bispectrum_FlowGAN_ep15.png
LR: equilateral mid-k B ratio to HR = 1.112
SR: equilateral mid-k B ratio to HR = 1.056
=== halo-count deficit ===
  LR fractional deficit (LR-HR)/HR: mean -1.85%  std 8.71%  median -1.03%
  corr(LR deficit, Om) = -0.171
  corr(LR deficit, s8) = -0.164
  SR fractional deficit (SR-HR)/HR: mean -1.18%  std 10.27%  median -0.35%
  LR deficit on the SAME boxes:      mean -2.12%
  share of missing halos SR restores, (SR-LR)/(HR-LR): median 18.7%  mean -252.7%
saved figures_cmass/halo_count_deficit_FlowGAN_ep15.png
TIER1_DONE
```

#### Tier-1 job 3248382
```
  SR vs HR : chi2/dof median=1.20  mean_p=0.376  frac(p<0.05)=0.13   (match => chi2/dof~1, p~uniform)
  LR vs HR : chi2/dof median=0.83  mean_p=0.539  frac(p<0.05)=0.12   (match => chi2/dof~1, p~uniform)
saved figures_cmass/pqmass_Flow+GAN_crps40.png
saved figures_cmass/bispectrum_Flow+GAN_crps40.png
LR: equilateral mid-k B ratio to HR = 1.112
SR: equilateral mid-k B ratio to HR = 1.106
=== halo-count deficit ===
  LR fractional deficit (LR-HR)/HR: mean -1.85%  std 8.71%  median -1.03%
  corr(LR deficit, Om) = -0.171
  corr(LR deficit, s8) = -0.164
  SR fractional deficit (SR-HR)/HR: mean -1.43%  std 10.11%  median -0.26%
  LR deficit on the SAME boxes:      mean -2.12%
  share of missing halos SR restores, (SR-LR)/(HR-LR): median 15.9%  mean -236.1%
saved figures_cmass/halo_count_deficit_Flow+GAN_crps40.png
TIER1_DONE
```

#### Tier-1 job 3248387
```
  SR vs HR : chi2/dof median=1.20  mean_p=0.351  frac(p<0.05)=0.17   (match => chi2/dof~1, p~uniform)
  LR vs HR : chi2/dof median=0.83  mean_p=0.539  frac(p<0.05)=0.12   (match => chi2/dof~1, p~uniform)
saved figures_cmass/pqmass_Flow+GAN_final40.png
saved figures_cmass/bispectrum_Flow+GAN_final40.png
LR: equilateral mid-k B ratio to HR = 1.112
SR: equilateral mid-k B ratio to HR = 1.111
=== halo-count deficit ===
  LR fractional deficit (LR-HR)/HR: mean -1.85%  std 8.71%  median -1.03%
  corr(LR deficit, Om) = -0.171
  corr(LR deficit, s8) = -0.164
  SR fractional deficit (SR-HR)/HR: mean -1.57%  std 10.14%  median -0.56%
  LR deficit on the SAME boxes:      mean -2.12%
  share of missing halos SR restores, (SR-LR)/(HR-LR): median 11.4%  mean -233.4%
saved figures_cmass/halo_count_deficit_Flow+GAN_final40.png
TIER1_DONE
```

#### Tier-1 job 3248399
```
  SR vs HR : chi2/dof median=3.44  mean_p=0.042  frac(p<0.05)=0.88   (match => chi2/dof~1, p~uniform)
  LR vs HR : chi2/dof median=0.83  mean_p=0.539  frac(p<0.05)=0.12   (match => chi2/dof~1, p~uniform)
saved figures_cmass/pqmass_Flow+GAN_guide.png
saved figures_cmass/bispectrum_Flow+GAN_guide.png
LR: equilateral mid-k B ratio to HR = 1.112
SR: equilateral mid-k B ratio to HR = 1.024
=== halo-count deficit ===
  LR fractional deficit (LR-HR)/HR: mean -1.85%  std 8.71%  median -1.03%
  corr(LR deficit, Om) = -0.171
  corr(LR deficit, s8) = -0.164
  SR fractional deficit (SR-HR)/HR: mean -1.28%  std 10.27%  median -0.47%
  LR deficit on the SAME boxes:      mean -2.12%
  share of missing halos SR restores, (SR-LR)/(HR-LR): median 15.8%  mean -246.1%
saved figures_cmass/halo_count_deficit_Flow+GAN_guide.png
TIER1_DONE
```

#### Spread / calibration (job 3248331)
```
0.225  1.001 [0.967,1.056]  1.022   0.542  0.024  0.02
0.333  1.005 [0.977,1.047]  1.010   0.358  -0.137  0.01

large-scale (k<0.1) T_SR median 0.990  T_LR 1.021
SSR mean over k: 0.149  (1 = calibrated; <1 under-dispersed)
per-voxel generation std across draws / HR field std: 0.4324
rank-hist (low/mid/high k), ideal flat = 0.111:
  low: 0.128 0.092 0.067 0.087 0.095 0.087 0.090 0.122 0.233
  mid: 0.278 0.063 0.050 0.048 0.035 0.043 0.070 0.087 0.325
  high: 0.409 0.056 0.024 0.024 0.033 0.028 0.017 0.028 0.381
box-to-box std of count-histogram features (bins 0,1,2,3,4,5,7,11,20,inf ; logmean):
  HR: 0.0718 0.0412 0.0174 0.0075 0.0033 0.0015 0.0010 0.0003 0.0000 0.0000 0.1037
  LR: 0.0608 0.0363 0.0144 0.0061 0.0026 0.0011 0.0008 0.0002 0.0000 0.0000 0.0876
  SR: 0.0610 0.0366 0.0136 0.0065 0.0026 0.0011 0.0007 0.0002 0.0000 0.0000 0.0880
SPREAD_DONE -> figures_cmass/spread_FlowGAN_ep15.png
```
- FlowGANLK3 NDE job 3248337:  COMPLETED 
- AfixLK3 NDE job 3248333:  COMPLETED 
- fmLK3 NDE job 3248335:  COMPLETED 

### R11. Final combined 8-seed table (job 3248394) and crps40 calibration (job 3251512)
- Final table (logs/3248394_seedxfid.log) reproduces every number in R3, R6, R10 exactly (same pickles); nothing new.
- Calibration crps40 (60 val x 8 draws): spread-skill 0.154 (epoch 15: 0.149), same U-shaped rank histograms at mid/high
  k, per-voxel draw std 0.42 of HR std. Longer training did not change calibration: draws still vary far less than
  their error vs HR at the P(k) level. Median transfer now within 1-2% of 1 at every k (large-scale median 1.004, LR
  1.021). Plot figures_cmass/spread_FGcrps40.png checked.
- Verdict on step 3: the flow's draws are under-dispersed (a known property of CRPS/L1-selected conditional flows run
  with an ODE sampler); they should not be used as a calibrated uncertainty. This does not affect the cross-fid
  protocol (one draw per box). Not fixed overnight: possible fixes are SDE sampling with extra noise (tried in Sept
  for the patch flow with little effect) or a larger source noise scale at train time; both need new training runs.

----------------------------------------------------------------------------------------------------------
