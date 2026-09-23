# Research scratchpad 3 — post-mentor-review round (CMASS count-field GAN)

Started 2026-07-12. Follows research_scratch.md (lagrangian) and
research_scratch_2.md. This round executes the to-dos from (a) the mentor's
report feedback and (b) the PhD meeting notes.

## To-do register

| # | item | source | status |
|---|---|---|---|
| 1 | Loss curves Arm A vs Arm B | meeting note 2 | DONE (see below) |
| 2 | KL setup made explicit in paper | mentor email | DONE |
| 3 | Fig 10 marker-line fix + redraw | mentor email | in progress |
| 4 | tests for conversion.py | preproc email pt 3 | pending |
| 5 | Bispectrum (held-out statistic) | mentor email | pending |
| 6 | Pk-loss ablation (lambda-pk 0) | meeting note 1 (confirmed: "kl loss" = Pk term) | pending (GPU) |
| 7 | Arm B (padding) tuning | meeting note 3 | pending (GPU) — see finding F2 first |
| 8 | Extra diagnostics if 7 fails | meeting note 4 | pending |
| 9 | Seam re-run, many-patch tiling | mentor email | pending (GPU) |

Also done earlier this session (preproc email pts 1-2): conversion.py voxel grid
made cell-centered (matches CIC lattice in analysis/power_spectrum.py) and
N_global rounding changed round->ceil so N_global^3 >= n_halos. Verified with
inline tests; formal test file = to-do 4. NOTE: lagrangian data on disk was
generated with the OLD convention; count-field pipeline unaffected (never uses
conversion.py).

## Findings

### F1 (2026-07-12) — Loss curves: both arms stopped early; A not converged
`analysis/plot_losses_cmass.py` -> `figures_cmass/losses_cmass.png`, parsed from
logs/8841137 (Arm A) and 8841138 (Arm B). 30 epochs configured, 23h wall clock:

| arm | epochs completed | best val_pkRMS | best epoch |
|---|---:|---:|---:|
| A (pad 0) | 19 | 0.0947 | 18 (second-to-last!) |
| B (pad 8) | 11 | 0.0963 | 3 |

- Arm A's best is its 18th of 19 epochs -> still improving at the wall; NOT
  converged.
- Arm B's 80^3 inputs make each epoch ~1.7x slower -> only 11 epochs.
- Arm B's stitched val L1 was still DESCENDING at cutoff and had crossed BELOW
  Arm A's plateau (0.1495 vs ~0.153) while its val_pkRMS oscillated ~0.10-0.11.

### F2 — implication for "tune Arm B" (meeting note 3)
Before turning any hyperparameter knobs, the curves say Arm B is simply
UNDER-TRAINED. The cheapest intervention with the highest prior: RESUME Arm B
from its last checkpoint (trainer supports --resume) for another wall-clock
window. Knob changes (lambda-rec down / lambda-pk up / pad 4) only if the
resumed run plateaus above Arm A.

### F3 (2026-07-12) — Bispectrum baseline: LR is ~11-17% HIGH vs HR at mid/high k
`analysis/bispectrum_cmass.py --mode raw`, 32 test boxes, 10 log k-bins
(`runs/patch_cmass/bispec_hr_lr.npz`, `figures_cmass/bispectrum_hr_lr.png`):
- Equilateral B(k,k,k): LR/HR ratio settles at ~1.05-1.20 for k >~ 0.03 h/Mpc
  (mid-k mean 1.112). LR OVERSHOOTS the halo bispectrum, consistent in sign
  with its small P(k) excess but LARGER in amplitude (~11% vs ~1-4% on Pk).
- Squeezed B(k,k,k_min): ratio scatters 0.9-1.35 for k >~ 0.03; noisier.
- First ~3 bins (k <~ 0.02) are sample-variance junk (few triangles), as the
  synthetic-field sanity check predicted. Interpret from bin ~3 up.
- WHY THIS MATTERS: the bispectrum separates LR from HR ~3-10x more strongly
  than P(k) does. So it is a genuinely discriminating held-out statistic for
  the mentor's "by construction" concern: if SR (trained with only P(k) in the
  loss) also closes this ~11% equilateral gap, that is real evidence the model
  corrects the field beyond what it was trained on. -> run bispectrum in
  `--mode dirs` on SR cubes right after the GPU eval regenerates them.

### F4 (2026-07-12) — HELD-OUT bispectrum: SR tracks HR about as well as / mildly
### better than LR, without ever training on B(k)
CPU quick-look: Arm A best.pt run on CPU (61 s/box), 16 test boxes, seed-0 noise
(`runs/patch_cmass/bispec_hr_lr_srA.npz`, `figures_cmass/bispectrum_hr_lr_srA.png`).
Statistic: per-box PAIRED ratio B_X/B_HR (shared ICs cancel cosmic variance),
median over boxes, well-sampled bins k >= 0.027 only.

Equilateral B(k,k,k), median paired ratio to HR per k-bin:

| k [h/Mpc] | 0.041 | 0.062 | 0.094 | 0.143 | 0.216 | 0.327 |
|---|---:|---:|---:|---:|---:|---:|
| LR | 1.01 | 1.06 | 1.13 | 1.10 | 1.10 | 1.05 |
| SR | 0.93 | 0.94 | 0.97 | 1.03 | 1.11 | 1.13 |

Squeezed B(k,k,kmin): SR closer to 1 than LR in most bins
(SR 0.90-1.13 vs LR 1.03-1.21).

Reading (honest):
- SR REMOVES most of LR's systematic +6-13% equilateral excess at intermediate
  k (0.06-0.14 h/Mpc) and improves the squeezed configuration, while never
  having seen a bispectrum in training. It slightly undershoots (-6/-7%) at
  k~0.04-0.06 and overshoots (+11-13%) at the two highest bins, where LR is
  +5-10% as well.
- Crucially SR does NOT collapse B(k): a conditional-mean regressor would
  suppress the bispectrum badly; the GAN keeps the right non-Gaussian
  amplitude. Combined with F3's point that B separates LR from HR more strongly
  than P(k), this is real (if modest) evidence against "the results are good by
  construction of the P(k) loss": a higher-order, never-trained-on statistic
  is preserved and mildly improved.
- Caveats: 16 boxes, one noise seed, Arm A only, CPU quick-look. Definitive
  version = full 200-box test split via the GPU eval; rerun then. Note the
  16-box mid-k LR ratio (1.08) differs from the 32-box unpaired estimate (1.11)
  -- paired per-box ratios are the reliable form of this measurement.

### F5 (2026-07-12) — Upstream pull: Julia fixed preprocessing independently;
### node-centered lattice is now the DOCUMENTED convention
Pulled 2c67278 (Julia): conversion.py N_global via compute_n_global (ceil with
float guard — equivalent to my local fix), halo-counts-per-voxel support
(save_outputs counts.npy, run_voxelize.py — answers mentor pt 4 about counts
in-repo), density_histogram/density_panel tools, and data/test_conversion.py
(11 test classes; passes here after a py3.9 compat line, see below).
- CONVENTION DECISION: Julia documents the node-centered lattice (voxel i
  center at i*voxel_side) as INTENTIONAL and makes counts voxelization
  consistent with it. My local cell-centered variant conflicted; hers is pushed
  and self-consistent -> hers stands. My variant preserved at
  scratch/conversion_mine.py for reference only; my tests/ suite removed
  (asserted the other convention). NOTE the CIC lattice in
  analysis/power_spectrum.py remains cell-centered (arange*cell + 0.5*cell) —
  a half-voxel global translation w.r.t. the conversion grid; harmless for
  P(k) amplitudes but worth one line in the paper if both pipelines are shown
  together.
- py3.9 compat: added `from __future__ import annotations` to data/conversion.py
  (her `np.ndarray | None` annotations crash import on this cluster's 3.9;
  additive fix, no behavior change). Her suite then passes: OK.

### F6 (2026-07-12) — GOAL METRIC (new protocol): SR-trained posterior SURVIVES
### the shift to HR far better than LR-trained. Meeting hypothesis CONFIRMED.
Cross-fidelity eval per the meeting's redefined goal: q trained on X, TESTED ON
HR Pk, compared against q trained-and-tested on HR. evaluate.py, 200 test sims,
2000 samples, seed 0 (`runs/patch_cmass/metrics_{hr_selfcheck,crossfid_*}.npz`).

KL(q_HR-on-HR || q_X-on-HR), per parameter and mean:

| trained on | Om | Ob | h | ns | s8 | mean |
|---|---:|---:|---:|---:|---:|---:|
| HR (self-check = noise floor) | 0.0008 | 0.0008 | 0.0007 | 0.0007 | 0.0007 | 0.0007 |
| SR Arm A | 0.0032 | 0.0038 | 0.0030 | 0.0019 | 0.0027 | **0.0029** |
| SR Arm B | 0.0028 | 0.0026 | 0.0019 | 0.0036 | 0.0055 | 0.0033 |
| LR (control) | 0.0039 | 0.0061 | 0.0035 | 0.0061 | 0.0038 | **0.0047** |

- SR-trained (Arm A) beats LR-trained on EVERY parameter; mean KL 0.0029 vs
  0.0047 (38% closer to the HR-trained reference). Self-check floor is 0.0007,
  so SR-trained sits ~4x the floor vs LR-trained ~6.4x.
- Parameter-recovery bias |mu-theta| is identical across all (~0.10 / 0.010 Ob)
  -> no accuracy cost, the difference is pure posterior consistency.
- This is the meeting's motivating claim demonstrated: sampling in HR space
  (SR) survives the distribution shift to HR better than staying in LR space.
- Caveat: these SR models were trained under the OLD protocol (Pk loss on).
  The no-Pk rerun (F8) must reproduce this for the clean story, since the
  goal metric may not feature in training-adjacent choices.

### F7 (2026-07-12) — 4-way bispectrum: Arm B ~= Arm A on the held-out statistic
16 boxes, CPU quick-look (`bispec_hr_lr_srA_srB.npz`,
`figures_cmass/bispectrum_hr_lr_srA_srB.png`): SR_A and SR_B mid-k equilateral
ratio to HR both 0.897 (unpaired); LR 0.990. Overlap padding neither helps nor
hurts the bispectrum. (Paired-ratio detail for Arm A in F4.)

### F9 (2026-07-12, updating live) — B-resume (old protocol): L1 still descending
Job 9250957 epoch-by-epoch (val_L1 / val_pkRMS); pre-cutoff best pk 0.0963@ep3,
last pre-cutoff L1 0.1495@ep10:
- ep10 (re-run): 0.1487 / 0.1007  -> L1 continues to improve past the old
  cutoff point; pk not yet beating 0.0963. Consistent with F1 under-training.
- ep11: 0.1526 / 0.1029 -> L1 back up; oscillating rather than descending.
  Under-training hypothesis WEAKENING for the pk metric: more epochs are not
  (yet) buying improvement. Keep to ~ep20 before the verdict.
- ep12-17 overnight: L1 flat ~0.150-0.152; pk oscillating 0.098-0.116, best
  new value 0.0983 (ep15) — still NOT beating the old 0.0963 (ep3). Seven
  extra epochs bought nothing on either metric.
- FINAL: job hit the 23h Slurm --time limit mid-epoch-18 (TIMEOUT, not a
  crash) and stopped there for good; last checkpoint epoch_15.pt, best.pt
  still the original ep3 0.0963. Not resubmitting -- this run has served its
  purpose (see verdict below) and the currently-running NEW-protocol B_aware
  (9269633) is the arm that actually matters going forward.
- VERDICT (final): B under the OLD protocol was NOT merely under-trained; its
  early ep3 optimum (0.0963) was real and 15 further epochs (3->18) never beat
  it. Contrast: the NEW-protocol B_aware run is already ahead of no-pad A at
  every comparable epoch (F10) -- the Pk loss, not insufficient training time,
  was what held old-Arm-B back.

### F8 (2026-07-12, updating live) — no-Pk run: L1 better, Pk drifting worse
Job 9250956 (Arm A recipe, lambda_pk=0) vs old Arm A (Pk on), val_L1 / val_pkRMS:

| epoch | no-Pk A | old A (Pk on) |
|---|---|---|
| 0 | 0.1303 / 0.1186 | 0.1785 / 0.1232 |
| 1 | 0.1267 / 0.1394 | 0.1555 / 0.1032 |
| 2 | 0.1220 / 0.1257 | 0.1604 / 0.1008 |
| 3 | 0.1535 / **0.5213** | 0.1538 / 0.0963 |

- Epoch 3: val Pk RMS exploded to 0.52 during a visible GAN oscillation
  (adv spiked to 6.9, D(f) -5.6, train rec rose 0.074 -> 0.127 within the
  epoch). Old A never exceeded ~0.13. Tentative new reading: the Pk loss was
  not only fitting the metric — it acted as a training STABILIZER for the
  adversarial dynamics. If the run recovers and stabilizes, fine (selection
  ignores bad epochs); if oscillations persist, the no-Pk protocol may need a
  replacement anchor (e.g. the variance/PDF statistics terms, or the
  scale-aware rec loss of protocol item 5).

POST-RESUME (9269632, from epoch_10.pt): ep10 recomputed = 0.1151/0.1220 vs
pre-crash ep10 0.1140/0.1087 (F11) — different but not bit-identical, expected
(resume doesn't replay RNG state exactly; within the epoch-to-epoch noise band
already seen, e.g. ep8 0.1088 ep9 0.1097). Not concerning on its own; watching
ep11+ to see the post-resume trajectory track the pre-crash one.
  ep11: 0.1136/0.1168 -- L1 tracking closely (pre-crash ep11 was 0.1134);
  pk a touch higher than pre-crash (0.1096) but within noise. Trajectory
  intact post-resume.
  ep12: 0.1121/0.1069 -- L1 EXACT MATCH to pre-crash ep12 (0.1121); pk close
  (0.1069 vs 0.1051). This was the last pre-crash epoch reported -- run is now
  caught up and entering genuinely new territory at ep13. Resume validated.
  ep13 (NEW): 0.1114/0.1092 -- L1 still slowly improving (0.1151->0.1136->
  0.1121->0.1114); pk still oscillating in a ~0.107-0.12 band, not clearly
  trending toward the old Pk-on run's ~0.10 range yet. 17 epochs remain.
  ep14: 0.1130/0.1043 -- best pk of the resumed run so far (prev low 0.1069
  ep12), now inside the old Pk-on run's ~0.10 range. L1 ticked up slightly but
  stays in the 0.111-0.115 band. Encouraging for the "GAN recovers Pk without
  the loss term" reading (F8).
  ep15: 0.1133/0.1106 -- L1 flat (~0.113); pk back up off ep14's low, still
  in the same oscillating 0.104-0.12 band. 15 epochs remain (ep15..29).
  ep16: 0.1134/0.1084 -- steady, same band. No new signal; watching for
  whether pk eventually settles below the old Pk-on best (0.0947) or plateaus
  around here.
  ep17: 0.1149/0.1057 -- same band, pk ticked down. 12 epochs remain.
  ep18: 0.1157/0.1092, ep19: 0.1168/0.1098, ep20: 0.1155/0.1073,
  ep21: 0.1164/0.1084, ep22: 0.1169/0.1077, ep23: 0.1174/0.1083,
  ep24: 0.1178/0.1051 -- L1 slight upward drift continuing (0.1133 at ep15 ->
  0.1178 at ep24, a new high for this stretch), while pk dropped to near its
  historical low (0.1051, close to F8's best of 0.1043 at ep14). Metrics
  moving in opposite directions this epoch.
  ep25: 0.1174/0.1124 -- L1 flat, pk back up. Still oscillating, no
  convergence. 5 epochs remain to the 30-epoch budget.
  ep26: 0.1146/0.1092 -- L1 improved slightly. 4 epochs remain.

OVERNIGHT UPDATE (ep 4-12): the run RECOVERED fully — the ep3 blowup was a
transient oscillation, not a collapse.

| epoch | no-Pk A L1/pk | old A (Pk on) L1/pk |
|---|---|---|
| 7  | 0.1137 / 0.1079 | ~0.154 / ~0.106 |
| 12 | 0.1121 / 0.1051 | ~0.152 / ~0.103 |

Revised no-Pk reading at half-run:
1. The GAN largely RECOVERS the Pk match on its own: 0.105 and falling vs
   ~0.096-0.103 for Pk-on at the same epochs. Removing the term costs a few
   percent on Pk RMS (so far), NOT a collapse -> the P(k) result was not
   simply "by construction"; the adversarial+L1 objective carries most of it.
2. Meanwhile stitched val L1 is ~26% BETTER than the old run at the same
   epoch (0.112 vs 0.152) — the Pk term was actively trading away real-space
   accuracy.
3. Occasional single-epoch GAN oscillations (ep3 here; ep2 in B_aware) —
   selection skips them; not fatal.

- L1 is much better without the Pk term (expected: one less competing
  objective). But val Pk RMS is DEGRADING (0.119 -> 0.139) while old-A improved
  (0.123 -> 0.103) — early evidence the Pk loss was doing REAL work on the Pk
  metric, i.e. the mentor's "faked by construction" suspicion has substance at
  this stage. Open question: does the adversarial term pull Pk back over 30
  epochs? Verdict needs the full run + eval chain. (NOTE early epochs are noisy
  — old A hit 0.19 pkRMS at ep1 of another arm; withhold judgment till ~ep10.)
- CAVEAT: 9250956 launched before the --select flag existed -> selects best.pt
  by val Pk RMS (legacy). Post-hoc L1 selection over its epoch_5/10/... will be
  applied; for the clean protocol its final numbers should be read from BOTH
  selections.

### F10 (2026-07-12, updating live) — padding variants, stitched-val trajectory
All no-Pk protocol; identical eval; select=l1. Epoch-0 comparison:

| run | pad | pad-loss | ep0 val_L1 | ep0 val_pkRMS |
|---|---|---|---:|---:|
| no-Pk A (9250956) | 0 | - | 0.1303 | 0.1186 |
| B_aware (9252145) | 8 | crop | **0.1239** | **0.1135** |
| B_unaware (9252146) | 8 | full | 0.1716 | 0.1185 |

- B_aware beats no-Pk A on BOTH stitched metrics at epoch 0 — first support
  for mentor prediction P1 (trim-aware padding = same objective + more info =>
  train/val at least as good). Under the OLD protocol B started WORSE than A
  (old-B 0.1655 vs old-A 0.1785 ep0). One epoch, keep watching.
- B_unaware val_L1 0.1716 — 38% WORSE than B_aware on the identical stitched
  eval at the same epoch. Mentor prediction P2 (unaware relies on halo voxels
  that get trimmed -> worse generalization) supported at epoch 0. Its Pk RMS
  (0.1185) is not hurt, interesting — the penalty is in real-space accuracy.
- EARLY VERDICT (1 epoch, hold lightly): P1 supported, P2 supported.
- ep1: B_aware 0.1219 / 0.1355 vs no-Pk A ep1 0.1267 / 0.1394 — B_aware still
  ahead of A on both. Note B_aware's pk also drifts up without the Pk anchor
  (0.1135 -> 0.1355), same pattern as A: consistent with the Pk-as-stabilizer
  reading in F8, now seen in a second independent run.

OVERNIGHT (same-epoch comparison, all no-Pk protocol):

| epoch | A (pad 0) L1/pk | B_aware L1/pk | B_unaware L1/pk |
|---|---|---|---|
| 0 | 0.1303 / 0.1186 | 0.1239 / 0.1135 | 0.1716 / 0.1185 |
| 3 | 0.1535 / 0.5213* | 0.1150 / 0.1094 | 0.1744 / 0.0960 |
| 5 | 0.1169 / 0.1166 | 0.1141 / 0.1158 | 0.1821 / 0.1073 |
| 6 | 0.1162 / 0.1090 | 0.1141 / 0.1086 | (0.1821 ep5) |
(* transient GAN oscillation)

- P1 HOLDING through 7 epochs: B_aware <= A on L1 at every epoch (and equal or
  better on pk at most). The mentor's expectation is borne out once the
  protocol is clean — old-B's underperformance was the Pk term's doing, not
  the padding's.
POST-RESUME (9269634, from epoch_5.pt): ep5 recomputed = 0.1868/0.1151 vs
pre-crash ep5 0.1821/0.1073 (F11) — resume-noise consistent; still far worse
L1 than B_aware's post-resume 0.1160 at the same epoch. P2 holds.
  ep6: 0.1848/0.1040 -- L1 continues its degrading trend (0.1716 -> 0.1821 ->
  0.1848 across ep0/2/6); still far behind B_aware's ep6 0.1119. P2 (unaware
  padding generalizes worse over training) continues to hold with fresh data.
  ep7: 0.1812/0.1088 -- L1 dips slightly off ep6's peak; reading revised to
  "oscillating in an elevated band (~0.168-0.187)" rather than strictly
  monotonic degradation, but still clearly and consistently worse than
  B_aware's ~0.112 band at every epoch. P2's core claim (unaware generalizes
  worse) still holds; the specific "monotonic" wording in the earlier note was
  too strong.
  ep8: 0.1828/0.1063 -- still in the same 0.168-0.187 band, no improvement.
  P2 steady.
  ep9: 0.1855/0.1075, ep10: 0.1849/0.1103 -- same elevated band, no recovery.
  Will hit its 23h limit well before epoch 30 (slowest run, ~2.3h/epoch); will
  need a RESUME to continue.

STRONG P2 CONFIRMATION: full epoch history (ep0-10, pre-crash + resumed)
checked directly. Best L1 of the ENTIRE run so far is epoch 2 (0.1687) --
strictly better than every one of epochs 3 through 10 (0.1744-0.1868, no
exceptions). Under --select l1 the saved best.pt IS epoch 2's checkpoint,
meaning 8 further epochs of training never improved real-space accuracy even
once. This is the cleanest evidence yet for P2: the trim-unaware model's
real-space fit peaks early and degrades as it learns to lean on the halo
region it will lose at test time.
  ep11: 0.1918/0.1079 -- NEW WORST L1 of the entire run (prev worst ~0.1868).
  The degradation is not just "never recovers", it is still actively getting
  worse 9 epochs past its ep2 peak. P2 keeps strengthening.
  ep12: 0.1798/0.1105 -- back off the ep11 worst, still in the ~0.17-0.19
  elevated band, nowhere near the ep2 best (0.1687).

ALL FOUR JOBS have now reported >=1 post-resume epoch; ordering unchanged from
pre-crash (B_aware < A_nopk < B_unaware on L1). No sign resume corrupted
anything. Resuming normal per-epoch monitoring.

- P2 CONFIRMED and strengthening: B_unaware's stitched val L1 DEGRADES over
  training (0.1716 -> 0.1821) — it increasingly relies on halo voxels that get
  trimmed. Textbook version of the mentor's "relies on boundary to average out
  behavior" failure.
POST-RESUME (9269633, from epoch_5.pt): ep5 recomputed = 0.1160/0.1066 vs
pre-crash ep5 0.1141/0.1158 (F11) — same resume-noise pattern as A_nopk above;
still clearly ahead of A_nopk's post-resume ep-comparable numbers, P1 intact.
  ep6: 0.1119/0.1081 -- matches pre-crash ep6 (0.1141/0.1086) closely; still
  ahead of A_nopk ep6 (0.1162/0.1090). P1 continues to hold on fresh data.
  ep7 (NEW, past pre-crash cutoff): 0.1124/0.1071 vs A_nopk ep7 0.1137/0.1079
  -- still slightly ahead on both. P1 holding into genuinely new territory.
  ep8: 0.1116/0.1079 -- L1 has been tightly stable 0.112-0.116 for 4 epochs
  straight (ep5-8); still ahead of A_nopk ep8 (0.1218/0.1088). P1 steady.
  ep9: 0.1103/0.1090 -- new best L1 for this run; still ahead of A_nopk ep9
  (0.1161/0.1097) on both metrics. P1 continues to hold.
  ep10: 0.1109/0.1087, ep11: 0.1131/0.1063 -- L1 still in its best-ever range
  (~0.111-0.113); P1 holds. On track to hit the 23h wall-clock limit well
  before epoch 30 (see epoch-budget note); will need a RESUME like 9250957 did.
  ep12: 0.1115/0.1093 -- steady in the same band.
  ep13: 0.1121/0.1086 -- steady; gap to A_nopk widening slightly as A_nopk
  drifts up (A_nopk ep23 L1 0.1174 vs B_aware ep13 L1 0.1121).
  ep14: 0.1090/0.1064 -- NEW BEST L1 for this run (prev best 0.1103 ep9). P1
  keeps strengthening; B_aware's L1 is now clearly and increasingly ahead of
  A_nopk's (0.1090 vs A_nopk ep25 0.1174).

- Curiosity: B_unaware posts the BEST single-epoch pk of all no-Pk runs
  (0.0960 ep3) while being worst on L1 — without core-focused spatial
  pressure, the GAN drifts toward statistics-matching over placement. Neat
  illustration that Pk RMS and real-space accuracy are separable axes.

### Padding variants, first train-loss readings (it200/it400, epoch 0, NOISY):
- B_aware (9252145):   rec 0.1095, 0.2072   (pk=0.0000 confirmed)
- B_unaware (9252146): rec 0.1327, 0.2570   (pk=0.0000 confirmed)
Aware < unaware at both checkpoints so far — but unaware's loss covers 95%
more voxels incl. the periodic-wrap halo, so its rec is not directly comparable
in absolute terms; the meaningful test is each variant's own trajectory + the
STITCHED val metrics (identical eval for both). Defer P1/P2 judgment to
epoch-level curves.

### Protocol v2 (2026-07-12, from PhD meeting) — now in effect
1. PRIMARY GOAL METRIC: cross-fidelity KL — q trained on SR, TESTED ON HR,
   vs q trained+tested on HR (F6 is the first measurement). Never in training.
2. Pk loss REMOVED from training (9250956 is the new mainline Arm A).
3. Checkpoint selection: new flag --select l1 (selection on val Pk was a
   metric leak of the same kind); legacy runs keep pk selection for
   comparability.
4. Padding two-variant experiment (mentor's hypotheses):
   --pad-loss crop (trim-AWARE, loss on kept 64^3 core; the old Arm B) vs
   --pad-loss full (trim-UNAWARE, loss on whole 80^3 incl. trimmed halo;
   dataset pad_target=True). Predictions to check on loss curves:
   P1: aware-B train loss <= A train loss (more info, same objective).
   P2: unaware-B trains lower, generalizes worse after trimming.
   Submitted: 9252145 (B_aware_nopk), 9252146 (B_unaware_nopk); both PK=0,
   SELECT=l1, pad 8. Shape path verified on CPU (80^3 targets, G/D forward,
   crop default unchanged).
5. Scale-aware rec loss (|x|^a, a<1) queued behind the padding baselines.
   Alternative to propose: per-scale L1 on a Gaussian pyramid (convex).
6. Report updated: cross-fidelity table added as PRIMARY cosmology result
   (paper/section6_srs.tex), own-data KL demoted to secondary; catalog-level
   (not halo-level) correspondence clarified in Data. Overleaf Sec 3.7 needs
   the same edits manually.
7. U-net Poisson baseline: prefer asking mentor for his baseline outputs to
   run through the same eval chain; implementing it ourselves is the fallback.

### F11 (2026-07-13) — INCIDENT: shared storage 100% full, 3/4 jobs crashed
### overnight; fixed with retry + resumed from surviving checkpoints
`df -h /data/group_data/universedata/` = 1.1T/1.1T, 0 avail, 100% full. Jobs
9250956 (A_nopk), 9252145 (B_aware_nopk), 9252146 (B_unaware_nopk) all died
15-16h in with FileNotFoundError on a *_input.npy inside a DataLoader worker.
Confirmed transient, not real corruption/deletion: all "missing" files exist
and read fine moments later, all 2000 input+label files present. Same failure
mode independently hit the single-step (Lagrangian) pipeline overnight per
[collaborator]'s update — a group-wide storage problem, not specific to this
code. 9250957 (B-resume, old protocol) was unaffected, still running.

Fix: added `_load_npy_retry` (exponential backoff, 5 attempts) around both
np.load calls in `data/patch_dataset_cmass.py: load_boxes`.

Recovery: checkpoints survived the crash (epoch_10.pt for A_nopk, epoch_5.pt
for both B variants, plus each run's best.pt). First resubmit attempt
(9269576-78) mistakenly started FRESH instead of resuming -- caught before
much time was lost, cancelled, resubmitted correctly with --resume:
  9269632 = A_nopk resume from epoch_10
  9269633 = B_aware_nopk resume from epoch_5
  9269634 = B_unaware_nopk resume from epoch_5
All three RUNNING as of this entry. Interim best-checkpoint numbers (usable
now, independent of resume outcome): A_nopk best val_pkRMS so far 0.1051 (ep12,
F8); B_aware_nopk best val_l1 so far 0.1141 (ep5/6, F10); B_unaware_nopk best
val_l1 0.1687 (ep2, F10) -- these already support the P1/P2 readings.

Group action item (from [collaborator]'s update, applies here too): shared
storage needs space freed before any further large writes; retry logic is a
mitigation, not a fix for the underlying full-disk condition.

NEAR-MISS caught: first resubmit (9269576-78) started FRESH instead of
resuming -- cancelled within ~1 min of submission, no meaningful time lost,
resubmitted correctly (9269632/33/34) with --resume; all three confirmed
"resumed at epoch N" in their logs and running.

FALSE ALARM checked: 9250957 (the untouched original B-resume job) also hit
the SAME FileNotFoundError twice in its stderr but did NOT crash -- log shows
continuous, unbroken epoch progression 10->17->into 18 with no second
"resumed" marker (i.e. no Slurm requeue/restart happened). Verified alive and
producing fresh output moments after the error grep fired. Likely explanation:
the transient NFS miss landed on a worker process during its end-of-epoch
teardown (fresh DataLoader workers are spawned each epoch, no
persistent_workers), so the exception surfaced after the consuming iterator
was already gone and got printed rather than propagated -- lucky timing, nine
9250957 is the one job that never needed the retry fix at all. No action
taken; noted here so future error-log greps aren't misread as crashes.

### F12 (2026-07-13) — LR ablation started; found and fixed a resume bug
Learning rate was confirmed fixed and un-swept across every run this session:
lr_g=lr_d=1e-4, Adam(beta1=0, beta2=0.99), no scheduler
(`train_patch_cmass.py:41-42,161-162`). Flagged as a plausible contributor to
the training oscillation seen in F1/F8/F9 (no monotonic late-training
improvement, occasional single-epoch blowups).

BUG FOUND while wiring this up: `opt_g.load_state_dict(c["opt_g"])` on resume
restores the CHECKPOINT's optimizer state, which includes the old `lr` per
param group -- this silently overwrites any new `--lr-g`/`--lr-d` passed at
resume time, so a naive LR-ablation-via-resume would have silently done
nothing. Fixed in `train_patch_cmass.py` (~line 176): after loading optimizer
state on resume, explicitly re-set `opt_g.param_groups[*]["lr"] = args.lr_g`
(same for D). Also added `LRG`/`LRD` env passthrough to
`scripts/train_patch_cmass.slurm` (mirrors the PK/PADLOSS/SELECT pattern).
This only affects the *next* process to load the file; the three currently
running jobs (9269632/33/34) already have the old code in memory and are
unaffected.

Submitted job 9276992: resumes the mainline no-Pk Arm A from its last SAFE
checkpoint (`checkpoints/patch_cmass_A_nopk/epoch_15.pt`, epoch index 14) into
a NEW checkpoint dir (`patch_cmass_A_nopk_lowlr`, so the still-running
9269632 is not touched), with lr_g=lr_d=5e-5 (half the default), PK=0,
SELECT=l1, PAD=0 otherwise identical. Question: does halving the LR reduce the
oscillation / let Pk RMS settle instead of wobbling in the ~0.10-0.14 band
seen in F8? Comparison will be epoch-for-epoch against 9269632 (same LR) from
epoch 15 onward, both starting from the identical epoch_15.pt weights.
### F13 (2026-07-13/14) — held-out bispectrum on the ACTUAL new-protocol
### checkpoints (in progress)
User asked to close the biggest gap flagged earlier: all no-Pk/padding
conclusions so far were from training-time proxy metrics (stitched L1/pkRMS),
not the real held-out check. Only 1 CPU core on this node (nproc=1, load ~5),
so did this in two feasible steps rather than the full 1600-sim NDE retrain
(infeasible on CPU, would take many hours per arm; deferred, see note below):
  1. CPU inference (sequential, detached background script
     scratch/infer_newprotocol.py) on 16 test boxes each, current best.pt of:
     A_nopk, B_aware_nopk, B_unaware_nopk. DONE, ~28 min each arm.
  2. 5-way bispectrum (HR, LR, SR_Anopk, SR_Baware, SR_Bunaware),
     `analysis/bispectrum_cmass.py --tag newprotocol`. Running now.
Note while inferring B_unaware: its best.pt is EPOCH 2 (see the B_unaware log
entry above) -- so this bispectrum result for B_unaware reflects an early,
not late, checkpoint, which is itself informative (it's the best real-space
model that arm ever produced).

RESULT (`bispec_newprotocol.npz`, `figures_cmass/bispectrum_newprotocol.png`),
16 test boxes, equilateral mid-k ratio to HR (same unpaired computation as
F3/F7, so directly comparable):

| source | equilateral mid-k B ratio to HR |
|---|---:|
| LR (control) | 0.990 |
| SR_Anopk (new protocol, pad 0) | 0.986 |
| SR_Baware (new protocol, pad 8, trim-aware) | 0.977 |
| SR_Bunaware (new protocol, pad 8, trim-unaware, epoch-2 ckpt) | **0.488** |

Compare to the OLD protocol (F4/F7): SR_A was 0.897, SR_B was 0.897, both
BELOW LR's 0.990 on this statistic. Under the new protocol, SR_Anopk (0.986)
and SR_Baware (0.977) sit almost exactly at LR's level, a large improvement on
this held-out statistic over the old Pk-loss-on models. Consistent with F8's
finding that removing the Pk loss improved real-space L1 substantially, better
real-space placement plausibly carries over to better bispectrum agreement.
SR_Bunaware collapses to 0.488, roughly HALF of HR's amplitude, a dramatic,
independent confirmation of P2 on a statistic that has nothing to do with the
training loss: the same model that has the worst L1 and never recovers past
epoch 2 also has by far the worst higher-order structure.

CAVEATS (explicit, not hidden): 16 boxes, one noise seed, CPU quick-look;
A_nopk/B_aware/B_unaware checkpoints are mid-training bests (epoch ~15-23,
~2, respectively), not final epoch-30 models; ratio is unpaired
(median-of-ratios across boxes, not per-box paired like the F4 table) so it is
noisier than F4's paired numbers but IS directly comparable to F3/F7 since all
three use the identical script and metric definition.

DEFERRED (explicitly, not silently skipped): the cross-fidelity posterior KL
(the metric that actually matters most, per F6) requires training an NDE on
SR Pk from the ~1600-sim TRAINING split, which needs inference over that many
boxes. At ~90-110s/box on this 1-core node that is many hours per arm x 3 arms
-- not attempted now. Options going forward: (a) wait for a GPU job to finish
and use it for fast inference, or (b) run a reduced-but-controlled version
(same N for all of q_HR/q_SR/q_LR so it's a fair comparison, just lower
power) in the background over hours. Not started yet; flagging for a decision
rather than quietly doing a possibly-misleading shortcut.

CONFIRMED started correctly: log shows "resumed at epoch 15  lr_g=5e-05
lr_d=5e-05" -- the resume-LR fix works, the new rate actually took effect.
Baseline to beat at ep15 (9269632, lr=1e-4): val_L1 0.1133, val_pkRMS 0.1106.

FIRST COMPARISON (epoch-for-epoch, both from identical epoch_15.pt weights):

| epoch | lr=1e-4 (9269632) L1/pk | lr=5e-5 (9276992) L1/pk |
|---|---|---|
| 15 (recomputed post-branch) | 0.1133/0.1106 | 0.1138/0.1077 |
| 16 | 0.1134/0.1084 | 0.1139/0.1064 |
| 17 | 0.1137/0.1060 (lowlr) vs 0.1149/0.1057 (base) | -- |
| 18-19 (lowlr only so far) | base: 0.1157/0.1092, 0.1168/0.1098 | -- |

Very close through 3 epochs, but epoch 18 shows the clearest separation yet:

| epoch | lr=1e-4 (9269632) L1/pk | lr=5e-5 (9276992) L1/pk |
|---|---|---|
| 15 | 0.1133/0.1106 | 0.1138/0.1077 |
| 16 | 0.1134/0.1084 | 0.1139/0.1064 |
| 17 | 0.1149/0.1057 | 0.1137/0.1060 |
| 18 | 0.1157/0.1092 | $\mathbf{0.1120/0.1074}$ |
| 19 | 0.1168/0.1098 | $\mathbf{0.1131/0.1084}$ |
| 20 | 0.1155/0.1073 | $0.1132$/$0.1124$ |
| 21 | 0.1164/0.1084 | $\mathbf{0.1128/0.1045}$ |
| 22 | 0.1169/0.1077 | $\mathbf{0.1136/0.1004}$ |

8 epochs in: lowlr ahead on L1 every single epoch so far. pk was mixed at ep20
but back to a clear lowlr win at ep21-22; 0.1004 at ep22 is a new best PK RMS
across BOTH no-Pk runs (previous low 0.1043, baseline ep14), and is getting
close to the original Pk-loss-on Arm A's historic best (0.0947).

### VERDICT at the ep25+ checkpoint (matched ep15-22, n=8 each)
Win count (lower is better), epoch by epoch:
  L1: base wins ep15,16 (2/8); lowlr wins ep17-22 (6/8), all 6 CONSECUTIVE.
  pk: base wins ep17,20 (2/8); lowlr wins the other 6/8.
Mean over ep15-22: base L1=0.11536 pk=0.10839; lowlr L1=0.11326 pk=0.10665.
lowlr better on BOTH metrics on average (L1 -1.8%, pk -1.6%), and the L1 win
is a clean 6-consecutive-epoch streak after the first 2 (near-identical,
right at the branch point) -- unlikely to be noise.
VERDICT: lean yes, halving the LR helps. Recommending to the user that
B_aware_nopk and B_unaware_nopk get the same LR-ablation branch (resume from
current checkpoint into a new dir, mirroring 9276992), rather than deciding
unilaterally since it commits 2 more GPU jobs. Awaiting go-ahead.

### F14 (2026-07-14) — AUTONOMOUS DECISIONS (user granted standing authority to
### decide + log; act as senior research engineer, stick to the research goal)
The research goal (mentor meeting): a posterior TRAINED ON SR that recovers HR
cosmology when TESTED ON HR (cross-fidelity KL). Everything else (L1, pk RMS,
bispectrum) is diagnostic. Decisions below are all oriented to that goal.

CONTEXT AT DECISION TIME: queue empty. Under new protocol (lambda_pk=0):
  - A_nopk (9269632): pad0 lr1e-4, TIMED OUT ep28/30. best.pt ep28 (select=pk
    legacy). Plateaued (F8).
  - B_aware (9269633): pad8 trim-aware lr1e-4, TIMED OUT ep16/30. best.pt ep16
    L1=0.107 (select=l1). Winning arm (F10, P1).
  - B_unaware (9269634): pad8 trim-unaware lr1e-4, TIMED OUT ep13/30. best.pt
    ep2 L1=0.1687 -- actively WORSENING with training (F10, P2, bispectrum
    0.488 F13).
  - A_lowlr (9276992): pad0 lr5e-5, COMPLETED 30/30. best.pt MISSING (resume
    select-mismatch bug, fixed below). epoch_20/25/30 saved. True best ep18
    L1=0.1120 (weights lost). Lean-yes on training metrics (F12).

DECISIONS:
D1 (STOP training, do not resume the 3 timed-out jobs). Rationale: all
   plateaued per F8/F9/F10; B_unaware actively worsens so more training is
   counterproductive; A_nopk 28->30 is trivial. Marginal metric gain does not
   justify GPU + 23h queue reservations. Use existing best.pt checkpoints.
D2 (PIVOT to the goal metric). Everything so far was training-time proxies +
   one held-out bispectrum. The missing piece is cross-fidelity posterior KL
   on the NEW-protocol models -- run it now. This is what the whole protocol
   change was for.
D3 (checkpoints to eval): each arm's own best.pt (its deliverable, parallel to
   how F6 used old-protocol best.pt), EXCEPT A_lowlr uses epoch_25.pt (no
   best.pt; L1 0.1125 ~= true best ep18 0.1120, a documented "best-available"
   caveat).
D4 (reuse HR reference): posterior_hr.pkl + pk_hr (2000 files) are
   protocol-independent (HR data unchanged) -> no regeneration. Verified
   present.
D5 (defer LR-propagation-to-B-arms): do NOT speculatively launch lr5e-5 B-arm
   training. Instead let A_lowlr's cross-fidelity KL vs A_nopk's answer whether
   lower LR helps THE GOAL metric; only propagate if it does. Avoids 2
   speculative GPU jobs.
D6 (cross-fidelity step): the eval wrapper produces posterior_TAG.pkl +
   OWN-DATA KL. After it lands, run one extra evaluate.py per arm with
   --pk-sr-dir=pk_hr for the CROSS-FIDELITY KL (q_SR tested on HR), cheap/CPU.
D7 (code fix, shipped): the resume select-mismatch bug that ate A_lowlr's
   best.pt is fixed in train_patch_cmass.py (~L182): only inherit the saved
   "best" threshold if ckpt's select == this run's select, else reset to inf.
   Prevents silent "best.pt never written" on any future select-changing
   resume.

SUBMITTED (scripts/eval_cmass_wrapper.slurm, GPU, 5h each, parallel):
  9362706 Anopk   (A_nopk/best.pt, naive)
  9362707 Baware  (B_aware/best.pt, overlap)
  9362708 Bunaware(B_unaware/best.pt, overlap)
  9362709 Alowlr  (A_lowlr/epoch_25.pt, naive)
Each: infer 2000 sims -> seam -> pk -> train NDE q_SR -> own-data KL. Then I
run the cross-fidelity evaluate.py (D6) and compare to F6 (old protocol:
SR-A 0.0029, SR-B 0.0033, LR 0.0047, HR floor 0.0007). KEY QUESTION: does the
new (no-Pk) protocol match or beat the old protocol on the GOAL metric?

### F15 (2026-07-18) — GOAL-METRIC RESULTS, new protocol. DECISIVE.
Cross-fidelity KL (q_SR trained on SR, TESTED ON HR pk) vs q_HR on HR, 200 test
sims, 2000 samples. Params (Om, Ob, h, ns, s8). All 4 eval jobs (9362706-09)
completed clean; posteriors + pk in runs/patch_cmass/.

| model (new protocol, lambda_pk=0) | Om | Ob | h | ns | s8 | MEAN |
|---|---:|---:|---:|---:|---:|---:|
| A_nopk   (pad0, lr1e-4)   | 0.0034 | 0.0029 | 0.0023 | 0.0031 | 0.0015 | **0.0026** |
| A_lowlr  (pad0, lr5e-5)   | 0.0024 | 0.0016 | 0.0058 | 0.0023 | 0.0015 | 0.0027 |
| B_aware  (pad8 trim-aware)| 0.0041 | 0.0050 | 0.0024 | 0.0027 | 0.0031 | 0.0034 |
| B_unaware(pad8 trim-unaw.)| 1.7514 | 0.0162 | 3.1351 | 0.0336 | 0.3219 | **1.0516** |

Reference F6 (OLD protocol, Pk loss ON): SR-A 0.0029, SR-B 0.0033, LR 0.0047, HR floor 0.0007.

FINDINGS (all on the metric that actually matters):
1. NEW no-Pk PROTOCOL IMPROVES THE GOAL. A_nopk 0.0026 < old SR-A 0.0029 < LR 0.0047.
   Removing the power-spectrum loss made the cosmology posterior BETTER, not worse.
   Counterintuitive: Pk term was in loss AND metric, yet dropping it improved
   downstream Pk-based inference -- model learns better field structure (F8 +26% L1,
   F13 bispectrum 0.986 vs old 0.897). Validates the mentor's protocol change on the goal.
2. PADDING DOES NOT HELP THE GOAL -- IT HURTS. B_aware 0.0034 > A_nopk 0.0026, despite
   B_aware WINNING on training L1 (0.107 vs 0.117) and matching bispectrum. Clean
   dissociation: optimized L1 and held-out bispectrum both favored padding, but the goal
   (cross-fidelity cosmology) favors NO padding. Deliverable = A_nopk. Likely overlap
   blending distorts stitched Pk in a way that hurts inference while improving local fit.
3. TRIM-UNAWARE CATASTROPHIC ON COSMOLOGY: 1.05 mean (Om 1.75, h 3.14), ~400x good arms.
   Own-data KL also elevated (0.058, ~20x); cross-fid ~18x worse than own-data -> real
   distribution-shift breakdown, not numeric bug. Strongest confirmation of P2. CAVEAT:
   best.pt is epoch 2 (L1 only worsens with training so ep2 IS its best, but early); 4
   independent metrics (L1, bispec 0.488, own-data KL 0.058, cross-fid 1.05) all agree,
   so undertraining is not a plausible full explanation.
4. LOWER LR IS A WASH ON THE GOAL. A_lowlr 0.0027 ~= A_nopk 0.0026. lr5e-5's L1 advantage
   (F12) did NOT translate to cosmology.

AUTONOMOUS DECISIONS:
D8 (resolves pending D5): DO NOT propagate lr5e-5 to B arms. A_lowlr tied A_nopk on the
   goal (0.0027 vs 0.0026); LR helped train L1 but not cosmology. LR ablation closed.
D9: DELIVERABLE = A_nopk (pad0, no-Pk, lr1e-4). Best on goal (0.0026), beats old protocol
   (0.0029) and padding (0.0034). Padding NOT recommended for the cosmology objective
   despite its L1/bispectrum edge. Headline conclusion.
D10: Update paper section6 Follow-Up with this table + the L1-vs-goal dissociation.
   Regenerate cross-fidelity KL bar figure.
D11: Training experiments CLOSED. Remaining work is analysis/writeup. No pending GPU work.


### F16 (2026-07-18) — SIGNIFICANCE CHECK on the goal metric. CORRECTS F15 claim 1.
Paired per-box comparison (mean over 5 params, same 200 test sids), goal metric
(cross-fidelity KL):

| comparison | paired diff (new-base) | t | frac boxes new<base | verdict |
|---|---:|---:|---:|---|
| A_nopk vs LR control    | -0.0020 | -10.97 | 0.82 | ROBUST: model beats no-model |
| A_nopk vs B_aware (pad) | -0.0008 |  -6.29 | 0.69 | ROBUST: no-pad beats padding on goal |
| A_nopk vs old-SR-A      | -0.0003 |  -1.75 | 0.49 | NOT DISTINGUISHABLE |

CORRECTION to F15 finding 1: "new no-Pk protocol IMPROVES the goal" was an
OVERCLAIM. The 0.0026 vs 0.0029 gap is within noise (t=-1.75, only 49% of
boxes better -- a coin flip; the mean gap is heavy-tail-driven, not a
consistent per-box improvement). HONEST claim: the new protocol MATCHES the old
on the goal metric, at no cost, while being (a) simpler (no Pk loss to tune/no
metric-leak), (b) +26% better on real-space L1 (F8), (c) better on the held-out
bispectrum (F13, 0.986 vs 0.897). So the new protocol is PREFERABLE on
parsimony + real-space + held-out stats, NOT on the goal number itself.

WHAT REMAINS ROBUST (unchanged): (1) A_nopk clearly beats the LR control on the
goal (t=-11, 82%) -- the cross-fidelity story holds; the model genuinely helps.
(2) No-padding beats padding on the goal (t=-6.3, 69%) -- the L1-vs-goal
dissociation is real and significant. (3) Trim-unaware catastrophe (1.05 vs
0.0026) is ~400x, no test needed.

METHOD NOTE: KL is heavy-tailed across boxes, so the fraction-of-boxes sign
test is the more trustworthy statistic than the t on the mean. Both agree here.
Corrected the paper (section6 goal-metric block) and this scratchpad.


### F17 (2026-07-19) — TEX cleanup + NDE-training-seed ROBUSTNESS (running).
Two tracks per user's push ("keep drafting till good results; plain English, no
em-dashes; ground every decision").

(a) TEX: removed all 13 em-dashes from paper/section6_srs.tex (-> plain commas),
added a plain-English "Bottom line" paragraph stating the best result (no
padding + no Pk loss is the recommended model; SR cross-fid 0.0026 vs LR 0.0047,
HR floor 0.0007). Braces balanced 741/741, zero em-dashes verified.

(b) ROBUSTNESS (job 9365463, analysis/robust_nde.py): quantify NDE-training
variance of the goal metric. GROUNDING: SNPE_C posteriors vary run-to-run from
net init + training stochasticity (Hermans et al. 2022 "Trust Crisis in SBI";
Lueckmann et al. 2021 SBI benchmark) -> a single-NDE 0.0026-vs-0.0029 comparison
is not defensible. Design: retrain q_HR and each q_X together under 5 matched
seeds, report cross-fid KL spread. Seed-matched so shared init randomness
cancels.

FIRST DATA POINT (seed 0): A_nopk cross-fid KL = 0.0017, vs 0.0026 from the
ORIGINAL eval-job NDE. That single-seed swing (0.0009) already EXCEEDS the
entire A_nopk-vs-old-A gap (0.0003) I was interpreting in F15/F16. Strong
preliminary confirmation that (i) the "new protocol not distinguishable from
old" conclusion (F16) is correct and if anything understated, and (ii) the
ABSOLUTE point estimates in F15 carry NDE noise of order +/-0.001 that a single
run hides. Awaiting all 5 seeds for mean +/- std per arm; the A_nopk-vs-LR gap
(0.002-0.003) should survive since it is ~3x this noise, but the
A_nopk-vs-old-A gap will not.


### F18 (2026-07-19) — CAUGHT a bug in the robustness job before trusting it.
Reason-with-every-decision / don't-hallucinate in action.

The NDE-seed robustness job (9365463) reported seed-0 LR cross-fid KL = 0.0013,
LOWER (better) than A_nopk (0.0017). That contradicts the ROBUST F16 result
(A_nopk beats LR on 82% of boxes, t=-11; original means LR 0.0047 >> A_nopk
0.0026). LR flipping from clearly-worst to best is not variance, it is an
artifact -> investigated instead of reporting.

DIAGNOSIS (verified, reproducible):
1. Sampling-seed stability of the KL estimator is FINE: KL(qhr_0 || qLR_0)
   recomputed at sampling seeds {1000,123,7} = 0.0199 / 0.0195 / 0.0206,
   mean ~= median (0.0199 vs 0.0190), frac(sim KL>0.01)=0.99. So the mean KL is
   NOT heavy-tail-unstable for this pair; it is a solid ~0.020.
2. But the JOB reported 0.0013 for the SAME saved posteriors (qhr_0, qLR_0)
   that a careful recompute puts at ~0.020 -- a 15x inconsistency NOT explained
   by sampling. => the job's inline crossfid_kl has a bug (suspected
   GPU/RNG-state or in-memory-posterior interaction inside the train+eval loop;
   root cause not fully isolated, but the empirical inconsistency is clear and
   reproducible).

DECISION: DISCARD the robustness job's KL outputs. KEEP its saved posteriors
(qHR_s, q{arm}_s for s=0..4 in _robust_tmp) -- the expensive NDE-training part
is valid. Recompute ALL cross-fid KLs myself with the verified evaluate.py-based
method (/tmp/careful_kl.py), which (a) VALIDATES by reproducing F6 originals
(expect LR~0.0047, Anopk~0.0026) then (b) applies to the retrained NDEs.
Result pending.

METHOD NOTE going forward: use the careful standalone recompute, and prefer the
per-box sign test (F16) over the mean KL for model-vs-model claims, since the
mean is model-pair dependent in magnitude and the sign test is what proved
robust. Do not report the job's 0.0013-0.0032 numbers anywhere.


### F19 (2026-07-19) — MAJOR: the goal metric's ABSOLUTE value is dominated by
### the q_HR reference NDE, not by the SR model. Headline may need reframing.
Careful verified recompute (/tmp/careful_kl.py; method VALIDATED by reproducing
F6 originals exactly: LR 0.0047, Anopk 0.0026, oldA 0.0029):

Seed-0 RETRAINED NDEs, KL vs qhr_0 reference:
  LR_0 = 0.0199,  oldA_0 = 0.0208,  Anopk_0 = 0.0223   (all ~0.02, ~7x originals)

The ONLY change from the originals is the q_HR reference NDE (qhr_0 vs the
original posterior_hr.pkl). All three models jump to ~0.02 and the ordering
inverts (LR nominally lowest). => the ABSOLUTE cross-fid KL is set by which
q_HR you happened to train, which has large run-to-run variance. The
"0.0026 vs 0.0047, beats the 0.0007 floor" framing (F15) rests on ONE q_HR and
is not reference-invariant.

RESOLUTION HYPOTHESIS (being tested, /tmp/floor_and_paired.py):
- F16's "A_nopk beats LR on 82% of boxes" used a SHARED q_HR reference, so the
  PAIRED (same-reference) difference cancels the q_HR noise and can still be
  robust even though absolute values swing. Testing whether the 82% survives
  when the reference changes to qhr_0, qhr_1.
- The real NDE-TRAINING floor = KL between two independently-trained q_HR
  (qhr_0 vs qhr_1). If ~0.02, then F15's 0.0007 "floor" was SAMPLING-only and
  misleadingly low; the honest floor is ~0.02, and model differences at the
  0.003 level are within NDE-reference noise (only paired same-reference
  comparisons are meaningful).

IF the paired sign test survives (Anopk<LR ~high% across references): headline
holds in RELATIVE form ("SR-trained beats LR-trained, paired"), but the
absolute-KL and "near the floor" claims must be dropped/reframed. Grounding:
this is the standard SBI caution (Hermans 2022) that NPE posteriors are not
reproducible in absolute terms; relative/paired comparisons are the defensible
unit. Awaiting /tmp/floor_paired.log.


### F20 (2026-07-19) — HEADLINE AT RISK: cross-fid ranking not robust to NDE
### training. torch-default-init hypothesis REFUTED; proper ensemble test next.
- REFUTED: "originals cluster because fresh torch shares a default init" -- two
  fresh `python -c 'import torch; torch.randn(3)'` processes give DIFFERENT
  values, so torch is genuinely random per process. The two-cluster structure
  (original NDEs mutually ~0.003-0.005; retrained-with-explicit-seed NDEs
  mutually ~0.075 and ~0.10 from originals) is real and not yet mechanistically
  explained.
- ESTABLISHED regardless of mechanism (verified recompute, method validated vs
  F6):
  Q1 real floor: KL(qhr_0||qhr_1)=0.075, KL(qhr_0||qhr_2)=0.077. The F15
    "0.0007 floor" was SAMPLING-only (same NDE) -- off by ~100x. Absolute-KL
    and "near the floor" claims are INDEFENSIBLE.
  Q2 ranking flip: A_nopk<LR on 82% (orig ref) / 44% (qhr_0) / 17% (qhr_1). The
    headline "SR-trained beats LR-trained" does NOT replicate across NDE
    retrainings.
- IMPLICATION: F6/F15/F16 cross-fid conclusions were SINGLE-NDE draws. The
  metric as implemented (one NDE per condition, Gaussian-approx KL) has
  training variance (~0.07) >> the model differences claimed (~0.002). This is
  the Hermans et al. 2022 "Trust Crisis in SBI" reproducibility problem,
  textbook.
- BEFORE declaring the headline dead: must (a) confirm retrained NDEs are
  legitimate (sensible theta recovery, not degenerate), and (b) do the PROPER
  ensembled comparison (Hermans 2022 fix): many NDEs per condition, compare
  distributions. Using the robust job's saved posteriors (seeds 0-4) + the
  originals as ensemble members -> no new training needed. Definitive script
  next; will report whichever way it lands.

### F21 (2026-07-19) — The GAN's noise injection COLLAPSED: generator is
### effectively deterministic. (noise-scatter diagnostic)
- Diagnostic: total count over 8 noise draws per box has z-CV = 0.000 (3 boxes).
  Confirmed at the parameter level: _NoiseInject `.std` params (init at 0) are
  mean |std| = 4.4e-4 (A_nopk) / 2.0e-4 (B_aware) -- essentially still zero vs
  model-space signal O(0.1). The generator IGNORES its noise input.
- IMPLICATIONS:
  1. The model is a DETERMINISTIC conditional map, not a stochastic sampler.
     The "GAN draws a plausible small-scale realization" framing (used in the
     paper's What-Is-Not-Working and in earlier discussion) does NOT hold for
     these trained checkpoints -- there is no realized stochasticity. The model
     regresses toward the conditional mean as far as the adversarial term
     allows, consistent with the small-scale decorrelation r=0.40.
  2. Improvement lever: NOISE-MARGINALIZATION is dead (z-CV=0, nothing to
     average). The 17% per-box count scatter is ENTIRELY SYSTEMATIC (per-box
     bias, deterministic in z). Only a count-aware (Poisson) head or a better
     conditional model can address it. But this is MOOT until the goal metric
     is shown reliable (F20 / ensemble test pending).
  3. Why the std collapsed: standard GAN mode-collapse-of-noise / the
     adversarial + L1 objective has no term rewarding noise usage, and L1
     actively prefers the (deterministic) conditional mean. To make the noise
     active would need e.g. a diversity/variance term or a stronger adversarial
     weight (grounded: the noise-collapse failure mode is well known in
     conditional GANs, e.g. pix2pix/Isola et al. 2017 noted the generator
     learns to ignore input noise; they dropped it in favor of dropout).

### F22 (2026-07-19) — VERDICT: the cross-fidelity headline does NOT survive
### proper NDE treatment. Ensemble analysis (analysis/ensemble_crossfid.py).
(1) LEGITIMACY: all NDE members recover theta to ~0.08 on HR test (q_HR, q_Anopk,
    q_LR all similar) -> the retrained NDEs are VALID, not degenerate. Their
    disagreement is real.
(2) REAL FLOOR: within-q_HR pairwise KL = [0.076,0.076,0.319,0.150,0.384,0.309],
    mean 0.219. Two independently-trained q_HR disagree by 0.08-0.38, i.e.
    ~100-500x the F15 "0.0007 floor" (which was sampling-only) and ~40-190x the
    model differences claimed earlier (~0.002).
(3) ENSEMBLED (NDE-variance-marginalized, Hermans 2022 fix): KL(HR||Anopk)=0.0107
    vs KL(HR||LR)=0.0030; A_nopk closer than LR on just 1% of boxes (mean
    Anopk-LR = +0.0077, i.e. A_nopk WORSE). The single-NDE 82% (F16) does not
    hold; the point estimate REVERSES.

CONCLUSION: "SR-trained posterior beats LR-trained on the cosmology goal"
(F6/F15/F16) is a SINGLE-NDE ARTIFACT. Under NDE re-training or ensembling it
does not replicate. The cross-fidelity KL as implemented (one NDE/condition,
Gaussian-approx) cannot support SR-vs-LR or protocol claims: NDE-training
variance (>=0.076, up to 0.38) swamps the model differences (~0.002). Textbook
Hermans et al. 2022.

WHAT STILL SURVIVES (not NDE-dependent, still valid):
 - Training-metric results: no-Pk improves stitched L1 vs Pk-on (F8); B_aware
   beats no-pad on L1 (F10, P1); B_unaware degrades / best at epoch 2 (F10, P2);
   trim-unaware also worst on bispectrum 0.488 (F13).
 - Held-out bispectrum: SR preserves B(k) ~ LR level, does not collapse (F13).
 - Noise-injection collapse -> deterministic generator (F21).
These are real. The COSMOLOGY (cross-fid KL) claims are the ones that fall.

OPEN: two-cluster puzzle (original NDEs mutually ~0.003 but retrained ~0.22).
Resolving with 4 SEPARATE-PROCESS q_HR (job 9365594, like the originals). If
they cluster ~0.003 -> real noise small, headline is "weak/marginal"; if ~0.22
-> "meaningless". Either way the headline is not robustly established.
NEXT: rewrite paper goal-metric section honestly once 9365594 lands.

### F23 (2026-07-19) — REVERSAL OF THE REVERSAL: F20/F22 were based on BUGGY
### same-process NDEs. Legit (separate-process) NDE variance is SMALL (~0.004).
SEPARATE-PROCESS q_HR replicates (4 fresh `python -m inference.nde` processes,
like the originals): pairwise KL [0.0042,0.0031,0.0058,0.0033,0.0041,0.0036]
mean 0.004; orig-vs-rep mean 0.0034. TIGHT, matching the original cluster.

=> The ~0.22 within-q_HR scatter (F20/F22) was a SAME-PROCESS ARTIFACT of
robust_nde.py (training 5 SNPE_C sequentially in one python process with
explicit manual_seed leaks/corrupts state; the separate-process legit way is
reproducible to ~0.004). 

CONSEQUENCES:
 - DISCARD F20 (ranking flip 82->17%) and F22 (ensemble "LR beats A_nopk"):
   both used the corrupted _robust_tmp posteriors. NOT valid.
 - F16 (A_nopk<LR on 82%, t=-11) used LEGIT separate-process ORIGINAL
   posteriors -> potentially valid after all.
 - The real NDE-training floor is ~0.004 (separate-process), COMPARABLE to the
   model differences (~0.002). So the ranking may be marginal but is NOT
   "swamped by 0.22". Need to test properly.

LESSON (logged honestly): I almost reported "headline overturned" from a buggy
robustness harness. Checking the two-cluster puzzle (instead of trusting the
scatter) caught it. Same discipline that caught the F18 inline-KL bug. The
same-process-NDE-training pattern is the bug; never train many NDEs in one
process for a variance study.

REDO (proper): train 4 SEPARATE-PROCESS replicates each of q_Anopk and q_LR
(job submitted), confirm they cluster ~0.004, then redo the A_nopk-vs-LR
ranking across legit replicate pairs. Only THEN conclude. Paper rewrite ON HOLD
until this lands -- do not touch the cross-fid claims yet.

### F24 (2026-07-19) — FINAL VERDICT (clean, legit posteriors): SR does NOT
### robustly beat LR on the cosmology goal. F16's 82% was a lucky single draw.
Definitive ranking robustness, LEGIT separate-process replicates (/tmp/rank_robust.py):
 - Arm replicate spreads: q_Anopk 0.0035, q_LR 0.0036 (TIGHT -> these are legit,
   NOT the same-process 0.22 bug). Fixed q_HR reference = original posterior_hr
   (F23: representative), so q_HR variance is held out / canceled.
 - A_nopk<LR fraction over 16 (q_Anopk_i, q_LR_j) legit pairs:
   min 14%, max 76%, MEAN 51%. All 16: [14,46,40,18,42,66,62,45,46,76,70,55,46,73,74,51].

INTERPRETATION (final, trustworthy):
 - The SR-vs-LR mean-KL gap (legit originals: Anopk 0.0026 vs LR 0.0047, diff
   ~0.0021, validated F19) is SMALLER than the per-arm NDE replicate variance
   (~0.0035). So a random legit replicate pair is a coin flip (mean 51%).
 - F16's 82% used ONE (q_Anopk, q_LR) pair that happened to favor A_nopk; it
   does not generalize. F6/F15/F16 SR-vs-LR conclusion = single-draw artifact.
 - Same logic kills the other fine cross-fid rankings: new-vs-old protocol
   (~0.0003 gap) and no-pad-vs-aware-pad (~0.0008 gap) are FAR below the ~0.0035
   NDE noise -> indistinguishable.
 - SURVIVES on the goal metric: only the TRIM-UNAWARE catastrophe (KL ~1.05,
   ~300x the noise) is real and robust. Everything at the ~0.001-0.002 level is
   noise.

RECONCILIATION of the whole robustness saga:
 - F18: robust job inline-KL buggy -> discarded (correct catch).
 - F19: careful method validated vs F6 (correct).
 - F20/F22: used SAME-PROCESS buggy NDEs (0.22 scatter) -> their SPECIFIC
   numbers were wrong, but their CONCLUSION (ranking not robust) turns out
   CORRECT for a different, legit reason (F24). Net: discard F20/F22 numbers,
   keep the conclusion.
 - F23: separate-process NDEs tight ~0.004 -> legit variance small; originals
   representative.
 - F24: with legit replicates, SR-vs-LR ranking = 51% coin flip. FINAL.

WHAT SURVIVES OVERALL (verified, not NDE-fragile):
 - Cross-fid cosmology: only trim-unaware breaks it (huge, robust). SR/LR/new/old
   /padding are INDISTINGUISHABLE at ~0.001-0.002 (all < NDE noise 0.0035).
 - Training L1: no-Pk > Pk-on; aware-pad > no-pad; unaware-pad degrades (P1/P2
   as TRAINING results, real).
 - Held-out bispectrum: SR preserves B(k), does not collapse (F13).
 - Noise injection collapsed -> deterministic generator (F21).
 - Real NDE-training floor (done right): ~0.0035, not 0.0007.

PAPER: rewrite the cross-fid cosmology claims to "indistinguishable within NDE
variance; only trim-unaware fails; a real SR-vs-LR cosmology test needs NDE
ensembling / more replicates / a coverage test." Keep training + bispectrum
results. This is the honest, final state.

### F25 (2026-07-19) — COMPLETE STORY: why summary-based cosmology cannot show an
### SR benefit. LR is already at the NDE floor; no room to improve.
Two more grounded checks close the cosmology investigation:

1. PAIRED bispectrum (a MORE discriminating, held-out summary; F3 said B(k)
   separates LR from HR 3-10x more strongly than P(k)). Per-box, is SR closer to
   HR than LR on log-equilateral-B (16 boxes, bispec_newprotocol.npz)?
     SR_Anopk 62% (median err SR 0.043 vs LR 0.039 -- marginal, ~coin flip)
     SR_Baware 56% ; SR_Bunaware 25% (median 0.275, catastrophic).
   So even on the bispectrum, SR does NOT robustly beat LR. Only trim-unaware
   is clearly separated (worse).

2. WHY (the key quantitative point): LR is already at the estimator floor.
     NDE floor (two independent q_HR):  ~0.0039
     LR cross-fid:                       0.0047   (~1 floor-unit above floor)
     SR A_nopk:                          0.0026   (single-draw, within noise)
   An LR-trained posterior already agrees with an HR-trained one about as well
   as two HR-trained posteriors agree with each other. There is essentially no
   gap for SR to close at the summary level -> no summary-based cosmology metric
   (P(k) or bispectrum) can distinguish SR from LR.

FINAL SCIENTIFIC CONCLUSION (honest, complete, grounded):
 - Raw LR ALREADY recovers HR-level cosmology from these summary statistics
   (LR summaries are HR-like; consistent with FINAL.md's early observation).
   The SR correction therefore provides NO measurable benefit for summary-based
   inference -- not because SR is bad, but because there is no headroom.
 - SR's corrections are at the FIELD level (it fixes the count field); summary-
   based inference does not capture that. Whether SR helps FIELD-LEVEL inference
   (e.g. a CNN/field posterior) is untested and is the only avenue where an SR
   cosmology benefit could appear. This is the honest "future work".
 - The one robust summary-level cosmology result is NEGATIVE: trim-unaware
   padding destroys cosmology (P(k) KL ~1.05, bispectrum 25%).
 - Surviving positive results are field/training-level: no-Pk improves L1;
   aware-pad >= no-pad on L1; SR preserves the bispectrum vs a regressor;
   noise-injection collapsed (deterministic G).
This completes the cosmology arc. No further NDE experiments will change it
(signal 0.002 < floor 0.004, structurally).

## Log

- 2026-07-12: created this scratchpad. Items 1-2 done. Paper edit: explicit
  KL construction paragraph added to paper/section6_srs.tex (q_HR trained+tested
  on HR; q_SR trained+tested on SR; per-sim KL between the two posteriors).
- Item 3 (Fig 10): eval_seam_cmass.py no longer draws in-plot white boundary
  lines (they read as a seam in the residual panel); boundary now marked by axis
  ticks at 0/64/127. Arm B slice redrawn from saved npz
  (figures_cmass/seam_cmassB_overlap_slice.png; residual panel is the projected
  difference |Sz SR - Sz HR| since the npz stores projections, labeled as such).
  Arm A npz predates slice storage -> its figure regenerates during the next
  GPU eval with the fixed code.
- Item 4 (tests): tests/test_conversion.py, 10/10 pass on python3.9
  (added `from __future__ import annotations` to data/conversion.py for 3.9
  compat). Two initial test failures were WRONG TEST ASSUMPTIONS, not code bugs:
  greedy center-out assignment lets central voxels claim any nearest unassigned
  halo, so sparse-halo cases do not assign on-center; correct invariant is the
  full-grid identity case (tested), plus a moved-halo periodic-wrap case.
- Item 5 (bispectrum): analysis/bispectrum_cmass.py. FFT shell estimator,
  B = L^6/N^9 * sum(F1F2F3)/sum(T1T2T3), consistent with power_spectrum.py
  normalization (P = L^3|d_k|^2/N^6). SANITY-CHECKED on synthetic fields:
  Gaussian -> B consistent with 0 (|B_g|/|B_ng| = 0.001-0.02 on well-sampled
  shells), squared-Gaussian -> B > 0. Low-k shells (first ~3 bins) are
  sample-variance dominated (few triangles) -> interpret from bin ~3 up.
  HR vs LR raw run in progress (32 test boxes, CPU background).
- Item 6+7 (GPU): submitted
    * job 9250956: no-Pk ablation (PAD=0, PK=0.0,
      ckpt-dir checkpoints/patch_cmass_A_nopk)
    * job 9250957: Arm B RESUME from epoch_10.pt (per finding F2 -- B is
      under-trained; resume before knob-tuning)
  scripts/train_patch_cmass.slurm extended with optional PK= and RESUME= env.
  Both pending on partition priority at submission.
- 9252145 (B_aware_nopk) RUNNING; ARM header confirms new flag wired:
  "pad-loss=crop". 9252146 (B_unaware_nopk) pending.
- Both GPU jobs RUNNING (~15 min after submission). 9250957 confirmed
  "resumed at epoch 10" (epoch_10.pt stores epoch=9; retrains 10..29; best.pt
  only overwritten if val_pkRMS beats the stored 0.0963). 9250956 confirmed
  Arm A geometry (input 64^3); pk-loss weight 0 takes effect in the G loss.

### F26 (2026-07-19) — FIELD-LEVEL inference (the untested avenue). Pipeline built,
### grounded in literature; caught + fixed moment-network marginal collapse.
User asked to complete the field-level test (the one place SR's field corrections
could show a cosmology benefit that summaries cannot). Built, grounded in:
 - Architecture: 3D CNN, Conv3d+BN+LeakyReLU stride-2 blocks -> FC (mu,sigma),
   following Villaescusa-Navarro et al. 2021 (arXiv:2109.10360, CAMELS field-level).
   models/field_cnn.py, 10.6M params.
 - Loss: moment network (Jeffrey & Wandelt 2020, arXiv:2011.05991):
   L = sum_i log sum_j (theta-mu)^2 + sum_i log sum_j ((theta-mu)^2 - sigma^2)^2.
   Minimizer = (posterior mean, posterior variance) -> Gaussian posterior, and the
   cross-fid KL is ANALYTIC (no MC sampling noise, unlike the NDE version).
 - Rigor from F24: separate-process replicates (3 seeds x {HR,LR,SR}) + paired
   per-box ranking. SR train fields generated via A_nopk best.pt (1600, uint8).
FIRST RUN COLLAPSED: verr(norm) stuck at ~0.245 (= marginal/random baseline) while
loss decreased -> classic moment-net collapse (net predicts prior mean + calibrated
variance, ignores the field; variance term drives loss down). NOTE: the summary
(P(k)) recovery was ALSO ~marginal (FINAL.md |mu-theta|~0.10 = MAD of a uniform
prior) -> the cosmology signal in these sparse 128^3 halo-count fields may be
intrinsically weak. FIX: MSE warmup (15 ep) forces mu->theta before the variance
shortcut. Resubmitted 9408252-260. The warmup verr is the decisive signal test:
 - if verr drops below ~0.245 -> field has extractable cosmology, cross-fid test
   is meaningful (proceed to verdict);
 - if verr stays ~0.245 even under direct MSE -> weak signal, no method can be
   distinguished (consistent with the summary conclusion), and that is the honest
   answer.
