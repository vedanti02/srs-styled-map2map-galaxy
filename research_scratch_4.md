# Research scratchpad 4 — FIELD-LEVEL cosmology inference (the untested avenue)

Started 2026-07-19. Follows research_scratch_3.md (F1-F26). This scratchpad logs
the field-level SBI phase in detail (every decision), per user request.

## Why this phase
research_scratch_3 conclusion (F24/F25): summary-based cross-fidelity (P(k), and
paired bispectrum) CANNOT distinguish SR from LR on cosmology, because LR is
already HR-like at the summary level (~at the NDE floor 0.004). SR corrects the
FIELD; the only untested avenue where an SR cosmology benefit could appear is
FIELD-LEVEL inference (a CNN posterior on the full 128^3 box, not summaries).
This phase builds and runs that test.

## Design + grounding (literature)
- Architecture: 3D CNN, Conv3d(stride2)+BN+LeakyReLU blocks -> global pool -> FC
  head outputting (mu, sigma) for 5 params. Follows Villaescusa-Navarro et al.
  2021 "Robust marginalization ... at the field level" (arXiv:2109.10360, CAMELS).
  models/field_cnn.py, 10.6M params (their range 10-30M).
- Loss: moment network (Jeffrey & Wandelt 2020, arXiv:2011.05991):
  L = sum_i log sum_j (theta-mu)^2 + sum_i log sum_j ((theta-mu)^2 - sigma^2)^2.
  Minimizer = (posterior mean, variance). => cross-fid KL is ANALYTIC (no MC
  sampling noise, an advantage over the NDE version).
- Cross-fid test: train q on field source S (hr/lr/sr); apply to HR TEST fields;
  KL(q_HR on HR || q_S on HR). Each model uses its OWN input normalization,
  applied to HR at eval (honest distribution shift).
- Rigor (from F24): 3 SEPARATE-PROCESS seeds per source + paired per-box ranking.
- Input: log(1+n) count field, standardized by source-specific mean/std.
- Params normalized to [0,1] by Quijote LH priors (Om[.1,.5] Ob[.03,.07]
  h[.5,.9] ns[.8,1.2] s8[.6,1.]).

## Decision log
D-F1: Store SR fields as uint8 (counts 0-200 fit) in home (3.4GB, disk was 10G
  free). Only TRAIN split needed for q_SR (evaluated on HR test). Later also
  generated VAL split (needed for val monitoring). SR from A_nopk best.pt.
D-F2: SR gen (job 9408047 train, 9408296 val). uint8 worked, 1.6GB.
D-F3: First 9-job run had 2 bugs (float16 .std() overflow -> float64 accum; flip
  axis before unsqueeze). Fixed, no GPU wasted (died at load).
D-F4: FIRST TRAINED RUN COLLAPSED to the marginal: verr(norm) flat ~0.245 (=
  MAD of the ~uniform prior) while loss decreased -> classic moment-net collapse
  (net predicts prior mean + calibrated sigma, ignores field; variance term
  drives loss down). FIX: MSE warmup (15 ep) forces mu->theta before the
  variance shortcut. Resubmitted 9408252-260.
D-F5: SR jobs (9408258-260) failed: needed SR fields for the VAL split too.
  Generated SR val fields (9408296); rerun q_SR held until signal confirmed.
D-F6 (OPEN BUG in my own diagnostic): after warmup, verr STILL ~0.243-0.245
  through ep19 (both MSE warmup and moment loss). Ran a per-param signal
  diagnostic (field_diag.py, job 9408326) BUT it loaded qhr_s0.pt which is an
  EPOCH-0 (untrained) checkpoint (that job started slow) -> uninformative:
  corr(mu,theta) ~0, std(mu)~0.005 (constant output) = expected for untrained.
  MUST redo on a TRAINED checkpoint (ep~39) + add a TRAIN-vs-VAL recovery check,
  because with 10.6M params / 1600 fields OVERFITTING is a real alternative to
  "weak signal" (flat val + dropping train = overfit; flat both = no signal/bug).

## Status (live)
- 6 HR/LR field CNNs (9408252-257) training to ep40 (~10s/ep, ~8 min total).
- 3 SR CNNs held pending signal confirmation.
- NEXT: when HR/LR done, redo diagnostic on a TRAINED ckpt with train-vs-val
  recovery to decide: overfit (fixable: more data/reg/smaller net) vs weak
  signal (fundamental: 1 box, 5 params, sparse halos -> data uninformative).

## D-F7 (2026-07-19) — trained CNN fails to fit even TRAIN (not overfitting).
field_diag.py on qhr_s1.pt (ep34), TRAIN-vs-TEST recovery:
  TRAIN: mean recov 0.242 (marginal 0.241), corr~0 all params, std(mu)~0.003
  TEST : mean recov 0.250 (marginal 0.248), corr~0 all params, std(mu)~0.003
=> NOT overfitting (train also marginal). The CNN collapsed to NEAR-CONSTANT
output (std(mu) 0.003 vs std(theta) 0.28), cannot fit even training data. This
is an OPTIMIZATION/ARCHITECTURE failure on the sparse count field (mostly zeros;
first-layer stride-2 conv + BN likely washing out the sparse halo signal),
DISTINCT from weak cosmology signal.
CONTEXT: summary NDE recovery was ALSO ~marginal (FINAL.md |mu-theta|~0.10 =
uniform-prior MAD), so weak signal is plausible independently. But I must not
claim "weak signal" while my CNN can't fit train. DECISIVE test: can the CNN
MEMORIZE 32 fields (overfit, no aug/dropout, MSE, many epochs)?
  - memorizes (train recov -> ~0): architecture OK; marginal-on-full is a
    signal/data-quantity issue -> field genuinely weakly informative.
  - cannot memorize 32: architecture is BROKEN -> must fix the net before any
    conclusion; "weak signal" would be unsupported.
Running overfit_test.py (job below).

## D-F8 (2026-07-19) — Overfit test: architecture is FINE (memorizes 32 fields).
overfit_test.py (32 HR fields, pure MSE, no aug/dropout): train_recov 0.229
(marginal) -> 0.036, loss -> 0.0000, std(mu) -> 0.27 (=std(theta)). The 10.6M
CNN HAS capacity + can fit; output DOES vary with input. => "broken arch" RULED
OUT. The full-1600 collapse-to-marginal is a GENERALIZATION failure, not
capacity.
KEY RECONCILIATION: the summary P(k)-NDE recovery (FINAL.md) is
0.10/0.10/0.10/0.10/0.010 for Om/Ob... wait Om/h/ns/s8 0.10, Ob 0.010. The
uniform-prior MAD (=range/4) is EXACTLY: Om .4/4=.10, h .4/4=.10, ns .4/4=.10,
s8 .4/4=.10, Ob .04/4=.010. So the summary method recovers EXACTLY the marginal
for ALL 5 params -> extracts ~ZERO cosmology info per box. Strong INDEPENDENT
evidence the single CMASS-ILI box weakly constrains these 5 params.
=> The field CNN's marginal recovery is consistent with genuinely weak per-box
signal. But field inference CAN in principle beat P(k) (CAMELS); my CNN's
generalization failure might be improvable. ONE fair attempt before concluding:
GroupNorm (BN is a known small-batch/shift failure mode) + rotation aug +
weight decay, retrain HR, check if recovery beats marginal.
D-F9: if improved CNN still marginal -> field-level ALSO cannot extract
cosmology -> no method distinguishes SR/LR (data-limited); VERDICT. If it beats
marginal -> proceed to full SR/LR/HR comparison.

## D-F10 (2026-07-19) — VERDICT: field-level inference is DATA-LIMITED, cannot
## distinguish SR from LR. (GroupNorm+aug+wd attempt also marginal.)
Fair improved attempt (job 9408405, GroupNorm replacing BatchNorm, + 90deg
rotation aug + weight_decay 1e-4): MSE warmup ep0->14 moved verr 0.244->0.241,
i.e. FLAT at the marginal. MSE warmup is the BEST case for fitting the mean; its
failure to generalize means the field->cosmology mapping is not learnable from
1600 boxes with a capacity-verified CNN.

CONVERGENT EVIDENCE (3 independent methods, all ~marginal recovery):
 1. Summary P(k)-NDE: recovery EXACTLY the uniform-prior MAD for all 5 params
    (0.10/0.10/0.10/0.10/0.010) -> ~zero info extracted.
 2. Field CNN (BatchNorm): marginal recovery; overfit-32 test PROVES capacity.
 3. Field CNN (GroupNorm+aug+wd): also marginal.

CONCLUSION (honest, complete): a single CMASS-ILI halo-count box carries
negligible cosmological information for these 5 parameters. No inference method
tested (P(k), bispectrum, field CNN) recovers cosmology beyond the prior. This
UNIFIES the whole investigation: SR, LR, and HR all give near-prior posteriors,
so the cross-fidelity KL is small for everyone and the tiny differences are NDE
noise (scratch_3 F24). There is no cosmological signal for the SR field
correction to improve, so SR cannot be distinguished from LR on cosmology at ANY
level (summary or field). This is a DATA/SETUP limitation (1 box, 5 params,
sparse halos), not an SR failure.
CAVEAT (honest): "field carries negligible info" is demonstrated for THIS CNN
(capacity-verified) + THIS setup (1600 sims, 5 params, single box). A much
larger training set, fewer target params (e.g. Om/s8 only, as CAMELS does), or
multi-box/multi-field inputs could extract more; that is future work. But within
the project's setup, the field-level avenue does not rescue an SR cosmology
benefit.
WHAT SURVIVES (unchanged): trim-unaware padding breaks cosmology (robust);
training-metric + bispectrum + noise-collapse results (scratch_3).

## D-F10 addendum — per-param diagnostic on the GroupNorm ckpt (definitive).
field_diag.py on qhr_s7.pt (GroupNorm+aug+wd, best ep21), 150 train + test boxes:
  TRAIN: Om recov0.265 corr-0.018 | Ob 0.242 -0.076 | h 0.244 -0.068 |
         ns 0.232 -0.028 | s8 0.231 +0.015 | mean recov 0.243 (marginal 0.241)
  TEST : Om 0.242 +0.020 | Ob 0.247 -0.100 | h 0.258 -0.053 |
         ns 0.256 -0.011 | s8 0.248 +0.026 | mean recov 0.250 (marginal 0.248)
  std(mu)=0.000 for ALL params on BOTH splits.
READING: corr(mu,theta) ~ 0 (|corr|<0.10, random sign) for every param on BOTH
train and test; std(mu)=0 means the net output is CONSTANT (= prior mean).
Recovery = marginal on train AND test. So even on TRAINING data the field-CNN
output carries zero cosmology info, while the overfit-32 test (D-F8) proved the
same arch memorizes 32 fields (std(mu)->0.27). => generalization/signal failure,
NOT capacity. This is the definitive confirmation of the D-F10 verdict: the
field->cosmology map is not learnable from 1600 CMASS-ILI boxes for these 5
params. FIELD-LEVEL AVENUE CLOSED.

## D-F11 (2026-07-22) *** CRITICAL BUG: label scramble invalidates the no-info verdict ***
While running the Fisher "information ceiling" check (D-F10 follow-up), the sanity
test FAILED: corr(mean log P(k), s8) = +0.03, emulator R^2 = 0.002. Physically
impossible if fields<->theta are matched (P(k) amplitude MUST track s8). Grounded
in physics (halo abundance depends strongly on Om, s8), I traced it:

ROOT CAUSE (proven 2000/2000): the processed count fields {k:04d}_label.npy are
ordered by STRING-sorted sim id, sorted(range(2000),key=str) = [0,1,10,100,1000,
...], but cmass_theta.npz indexes cosmology NUMERICALLY. So processed field k is
the halo field of numeric sim string_sorted[k], NOT sim k. Every field in the
whole pipeline was paired with the WRONG cosmology.
EXACT PROOF: field_count[k] == catalog_count(halos.h5)[string_sorted[k]] for
2000/2000 sims (0 mismatches); identity match only 2/2000 (the fixed points 0,1).
perm saved -> data/string_sort_perm.npy.

EVIDENCE CHAIN:
 - halos.h5[sim s] count vs cosmo[s] (guaranteed-correct pairing): corr(N,Om)=+0.85,
   corr(N,S8=s8*sqrt(Om/0.3))=+0.93. Signal is REAL and STRONG.
 - processed field count[k] vs theta[k] (current/buggy): corr ~ 0 for all params.
 - apply string-sort fix -> corr(count,S8)=0.93, corr(count,Om)=0.84 restored.
 - P(k) emulator R^2: 0.002 (buggy) -> 0.98 (fixed). Fisher (fixed, single HR box):
     Om  sigma/prior 0.21-0.29  CONSTRAINED (marg sigma ~0.06-0.08)
     s8  sigma/prior 0.34-0.57  CONSTRAINED (~0.10-0.16)
     ns  0.54-0.60  weak ;  h 0.75-0.79 weak ;  Ob 0.77-0.84 unconstrained
   = textbook: low-z halo field constrains Om & s8, not Ob/h/ns.

SCOPE OF THE BUG:
 INVALIDATED (theta-dependent): ALL cross-fidelity KL, NDE recovery, field-CNN
   recovery, and the entire "single box carries no cosmology info" verdict
   (D-F4..D-F10, scratch_3 F24/F25). Those showed marginal recovery because the
   net was learning field -> WRONG cosmology. NOT a data limitation.
 STILL VALID (theta-independent): P(k) SR-vs-HR, bispectrum, field-match, seam,
   real-space L1 (these compare SR vs HR for the same box; no theta used).
 GAN training conditions on style=theta (G_correct(x,style), D(x,theta)) with the
   scrambled labels -> suboptimal conditioning, not "invalid"; retrain with correct
   labels may improve it.

RETRACTION: the D-F10 "field-level is data-limited / no method distinguishes SR
from LR because the data is uninformative" conclusion is WRONG. It was a labeling
artifact. The data richly constrains Om and s8 (P(k) R^2=0.98). The cross-fidelity
SR-vs-LR question must be RE-RUN with corrected labels before any verdict.

FIX: rebuild theta in field order: theta_fixed[k] = theta_numeric[string_sorted[k]].
Least-invasive (all code pairs field k with theta[k]) -> save cmass_theta_fixed.npz.

## D-F12 (2026-07-22) *** FIX VALIDATED END-TO-END: field-CNN now recovers cosmology ***
Retrained q_HR (job 9408796, qhr_s100.pt) after correcting cmass_theta.npz to
field order. Per-param diag (field_diag.py, corrected theta), corr(mu,theta):
             corr(TRAIN) corr(TEST)  recov(TEST)   [was, scrambled: corr~0, std(mu)=0]
  Om          +0.901      +0.903       0.109        marginal 0.25 -> Om error more than halved
  ns          +0.301      +0.300       0.236
  h           +0.280      +0.195       0.231
  s8          +0.111      +0.245       0.271
  Ob          +0.036      +0.008       0.257        (unconstrained, as Fisher predicted)
  std(mu) Om: 0.24 (was 0.000 = constant). mean recov 0.221 vs marginal 0.250.
Train==Test corr (0.901/0.903) => genuine generalization, NOT overfit.
vloss best -5.20 (corrected) vs -1.79 (scrambled) => far more informative posteriors.

MATCHES the Fisher forecast pattern exactly: Om strong, s8/ns/h weak-moderate,
Ob zero. This CONCLUSIVELY proves: (1) the string-sort label bug was real,
(2) the fix is correct, (3) a single CMASS box richly constrains Om (corr 0.90)
and moderately s8/ns/h, (4) field-level inference WORKS. The entire prior
"data is cosmology-uninformative / no method separates SR from LR" verdict
(D-F4..D-F10, scratch_3 F24/F25) is RETRACTED: it was a labeling artifact.

NEXT (the real deliverable, now answerable): corrected cross-fidelity. Train NDE
q on SR/LR/HR summaries with corrected labels, evaluate on HR. Does SR-trained
beat LR-trained (the project goal metric)? Caveat: existing SR fields came from a
GAN conditioned on the SCRAMBLED theta (style), so a clean answer needs a GAN
retrain with correct conditioning; a first-pass corrected cross-fid on existing
SR is informative but not final.

## D-F13 (2026-07-22) GAN theta-sensitivity -> existing SR is the FAIR first-pass.
gan_theta_sensitivity.py on A_nopk best.pt, SR relative change under different
theta conditioning (5 test boxes):
  correct vs scrambled theta (both valid vectors): 0.04-0.09
  correct vs extreme theta:                        0.08-0.13
  correct vs zero theta:                           0.47-0.49
READING: G does NOT ignore theta (zero-theta -> 48% change; it modulates on the
style). But across VALID cosmology vectors the change is only 4-9%.
IMPLICATION for cross-fid design:
 - Existing SR was generated with SCRAMBLED theta => it does NOT inject each box's
   TRUE cosmology into the field. That makes it the FAIR, NON-LEAKING choice: if we
   regenerated SR conditioned on TRUE theta, the GAN (which uses theta) would inject
   true cosmology into the SR field and q_SR could cheat (leak). Also, at deployment
   on HR you do NOT know theta. So the launched first-pass cross-fid (existing SR +
   corrected labels for the POSTERIOR pairing) is valid.
 - A GAN retrain is a real DESIGN question (what theta to condition on when it is
   unknown at test; whether to drop style conditioning), not a trivial bug fix.
   Defer until the first-pass cross-fid says whether SR helps at all.

## D-F14 (2026-07-22) CORRECTED cross-fidelity (summary/P(k) level) -- REAL verdict.
12 NDEs (q_HR, q_Anopk, q_LR x4 fresh-process seeds) on CORRECTED labels, ensemble
eval on 200 HR test boxes (analysis/ensemble_crossfid.py, jobs 9408874-79):
 (1) NDE recovery |post.mean - theta_true| on HR test: q_HR 0.051, q_Anopk 0.055,
     q_LR 0.052 (all seeds consistent). Prior-mean baseline ~0.082 => REAL recovery
     now (was ~prior when labels scrambled). Confirms fix at the NDE level too.
 (2) Real floor (within-q_HR seed-pair KL): mean 0.028 (0.019-0.034). [Note: higher
     than the old bogus 0.0007/0.004 because posteriors are now INFORMATIVE.]
 (3) Ensembled cross-fid KL to HR: LR 0.032 (~= floor 0.028), SR/Anopk 0.088 (~3x
     floor). SR closer than LR on only 12% of boxes; mean(SR-LR)=+0.056.
VERDICT (summary level, valid now): raw LR is ALREADY at the HR floor (transfers as
well as HR->HR). SR DEGRADES summary-level cosmology transfer (0.088 vs 0.032). So
at the P(k) level SR does not help and actively hurts. This is a real, interpretable
result, NOT the prior-collapse artifact.
WHY: LR & HR share the large-scale P(k) that carries the cosmology (Om); the GAN
alters small-scale P(k) in a way inconsistent with how HR P(k) responds to cosmology,
so q_SR mis-transfers. No summary headroom for SR (LR already ~HR).
OPEN (the one place SR could still help): FIELD-LEVEL cross-fid. SR corrects the
FIELD, not just P(k). Train field-CNN q on SR fields / LR fields / HR fields, test on
HR fields. field-CNN now works (corr Om 0.90). SR train fields exist
(runs/patch_cmass/sr_fields_Anopk). This is the decisive remaining test.

## D-F15 (2026-07-22) CORRECTED FIELD-LEVEL cross-fidelity -- SR HELPS (the positive result).
8 field CNNs launched (q_HR/q_LR/q_SR x3, corrected labels); 2 died on transient CUDA
errors, eval ran on 2 HR + 2 LR + 3 SR (job 9409473, eval_field_crossfid.py).
Cross-fid KL to HR (field-level, analytic Gaussian):
  SR: 0.0315 +/- 0.0108   (~= HR floor)
  LR: 0.969  +/- 0.948    (~40x floor, HIGH variance / unstable)
  HR: 0.0236 +/- 0.0014   (self floor 0.025)
Paired SR<LR per box: mean 66% (min 16% max 100% over 12 ref/SR/LR combos; noisy,
only 2 LR + 2 HR members).
Recovery (sanity, all working): HR 0.22, LR 0.18-0.22, SR 0.22 (avg over 5 params;
Om dominates the recovered signal).

VERDICT: REVERSAL vs the summary level. At the FIELD level the raw LR box is OOD for
a CNN (KL ~1, unstable) -> an LR-trained field posterior does NOT transfer to HR. SR
corrects the FIELD to match HR -> an SR-trained field posterior transfers at the floor
(0.03 ~ 0.025). THIS is SR's value, and it appears exactly where SR operates (field),
not at the forgiving summary (P(k)) level where LR is already HR-like.

COHERENT PICTURE (corrected, valid):
  summary/P(k):  LR 0.032 ~ floor 0.028; SR 0.088  -> SR neutral/hurts (LR already HR-like)
  field-level :  LR 0.97 (fails);        SR 0.032 ~ floor -> SR STRONGLY helps
=> SR helps cosmology transfer at the FIELD level, which is what it is designed to fix.
CAVEAT: LR KL has huge variance (2 members only; 2 CNN jobs died on CUDA). Firming up
with more HR/LR seeds before finalizing the magnitude and the SR<LR fraction.

## D-F15 firmed (4 HR / 4 LR / 3 SR, job 9409500):
Cross-fid KL to HR:  SR 0.0243 +/- 0.0114 (~= floor 0.022);  LR 0.488 +/- 0.808
(~20x floor, BIMODAL/unstable); HR floor 0.0204 +/- 0.0091. SR<LR fraction 60%
(16-100%, noisy due to LR bimodality).
REFINED: SR transfers to HR RELIABLY at the floor. LR is UNSTABLE -- some seeds
transfer OK, some fail catastrophically (KL up to ~2). Note LR gets the POINT
estimate on HR ~right (recovery 0.22 like all) but the posterior SHAPE wrong ->
high KL; SR fixes the shape. Net: SR is the reliable choice for field-level
cross-fidelity; raw LR is a gamble. This is SR's demonstrated cosmology value.

## D-F16 (2026-07-22) CORRECTED padding cross-fidelity (summary/P(k) level).
Added q_Baware, q_Bunaware NDEs (corrected labels, 4 seeds) to the _robust_tmp
ensemble; analysis/padding_crossfid.py (jobs 9478712-13 train, 9478738 eval):
  LR 0.031 (~=floor 0.028); SR no-pad(Anopk) 0.087; SR trim-aware(Baware) 0.127;
  SR trim-unaware(Bunaware) 0.172; HR floor 0.028.
(Anopk 0.087 ~= main-run 0.088, LR 0.031 ~= 0.032: consistent.)
READING: ordering no-pad < trim-aware < trim-unaware PRESERVED under correct
labels -> padding still hurts the summary goal metric, trim-unaware worst; no-pad
is the SR variant to use. BUT the buggy "trim-unaware = 1.05 (~400x, catastrophic)"
was a LABEL ARTIFACT: corrected it is 0.172, only ~2x the no-pad model at the
summary level (std large, +/-0.19). So RETRACT "trim-unaware breaks
catastrophically"; it is the weakest variant (worst goal metric + worst real-space
L1, fig protocolv2), not a blow-up. All SR variants remain worse than LR at the
summary level (LR already HR-like). Field-level padding cross-fid not run (would
need SR fields for Baware/Bunaware).

## D-F17 (2026-08-04) Mentor-endorsed next runs: correct-conditioning GAN retrain + firm LR margin.
Mentor confirmed (a) LR-fails-to-transfer = FastPM OOD-ness (expected), (b) fixing the
generator's cosmology conditioning is the biggest lever, (c) no averaging in stitching
(verified: crop_interior drops the pad halo, stitch_patches does direct assignment of
non-overlapping 64^3 cores -> the paper's "overlap blending" phrase is a wording bug to fix).

LAUNCHED:
1. GAN retrain with CORRECT conditioning (job 9753716). Reproduces A_nopk EXACTLY
   (pad=0, PK=0.0, select=pk, lr 1e-4, adv 1.0, 30 ep) from scratch; the ONLY change is
   cmass_theta.npz is now field-ordered (correct), so G conditions on the true cosmology
   instead of the scrambled one it saw before (D-F13: it was effectively conditioning on
   noise). Checkpoints -> /data/user_data/vkshirsa/cmass-ili/models/patch_cmass_A_nopk_fixcond
   (313GB free; HOME is 92% full / 8GB, do NOT save large models there).
2. +4 LR, +2 HR field CNNs (9753718-723) to firm the noisy field-level LR number
   (0.49 +/- 0.81 from only 4 members) and the HR floor -> total 8 LR / 6 HR / 3 SR.

OPEN METHODOLOGICAL CAVEAT (do NOT hand-wave when the retrain lands): with a correctly
conditioned G, generating SR by conditioning on each box's TRUE theta would inject true
cosmology into the SR field (a leak; q_SR could read the stamp). At deployment on HR you
do NOT know theta. So the FAIR SR-generation protocol mirrors deployment: condition on a
fiducial theta (prior mean) or marginalize, NOT the per-box truth. Plan: when retrain done,
generate SR both ways (true-theta and fiducial) and report the fiducial as the headline;
the gap between them quantifies the conditioning leak. Flagged to user/mentor.

## D-F18 (2026-08-04) FIRMED field-level cross-fid (8 LR / 6 HR / 3 SR seeds).
job 9754743, eval_field_crossfid on the doubled member set:
  SR 0.0223 +/- 0.0102 (~= floor);  LR 0.4711 +/- 0.7824 (~24x floor);  HR floor 0.0193
  (15 within-HR pairwise combos). SR<LR fraction 60% mean (10-100% over 144 combos).
KEY: doubling LR seeds (4->8) barely moved the spread (0.81->0.78). So the huge LR
variance is INTRINSIC bimodality, not small-sample noise: some LR-trained field CNNs
transfer to HR OK, others break (~2). SR is reliably at the floor. Right framing for
the paper = KL means + reliability (SR reliable, LR a gamble), NOT the noisy SR<LR
fraction. Paper numbers to update on next edit pass: LR 0.49->0.47, floor 0.022->0.019
(SR 0.024->0.022) -- all within noise, story unchanged. This is the STILL-scrambled-SR
result; the SR side gets redone once the correct-conditioning retrain (9753716) lands.

## D-F19 (2026-08-05) Correct-conditioning retrain: field cross-fid (fiducial vs oracle). NUANCED.
Retrained GAN (correct conditioning, val_pk 0.065 vs scrambled 0.104). Regenerated SR
two ways, trained 3 qsr each, field cross-fid (6 HR / 8 LR / 3 SR):
  old scrambled SR (~unconditioned):   0.022 +/- 0.010   (D-F18)
  new correct, ORACLE (true theta):    0.024 +/- 0.011
  new correct, FIDUCIAL (deployment):  0.037 +/- 0.014
  LR: 0.471 +/- 0.782 ;  HR floor 0.0193
READING (honest, do not spin):
 1. ALL SR variants transfer at/near the floor (0.022-0.037), 13-24x better than LR
    (0.47). The "SR helps at the FIELD level" headline is ROBUST across conditioning.
 2. The conditioning FIX did NOT improve the deployment goal metric. The field metric
    was already saturated at the floor (old SR already 0.022 ~ floor 0.019), so there
    was no headroom. Oracle (0.024) ~ old (0.022).
 3. FIDUCIAL (the deployable version, cosmology unknown at test) is somewhat WORSE:
    0.037 vs oracle 0.024. Mechanism: feeding a fixed prior-mean theta to a
    cosmology-conditioned G imposes a mild cosmology bias on SR, making it less like
    each box's true HR. NOTE the gap (0.013) ~ the metric noise (+/-0.014), so
    suggestive not decisive.
 4. So for DEPLOYMENT (theta unknown), an UNCONDITIONED / ignore-theta G is at least as
    good as the conditioned one; the conditioning benefit shows in field FIDELITY
    (val_pk 0.065 vs 0.104, measured with TRUE theta = oracle) but not in the saturated
    field cross-fid. Correct conditioning != free win on the goal metric.
WHERE conditioning might still help the GOAL: the SUMMARY-level cross-fid (old SR 0.088
worse than LR 0.032). Worth rerunning summary cross-fid with the new SR P(k). TBD.

## D-F20 (2026-08-05) Conditioned SR: SUMMARY cross-fid + PQMass + panels. HONEST/NEGATIVE for conditioning at deployment.
Regenerated conditioned-model evals (retrained correct-conditioning GAN, fiducial deployment):
SUMMARY-level cross-fid KL to HR (padding_crossfid, corrected labels):
  LR 0.031 (~floor 0.028) ; scrambled SR (Anopk) 0.087 ; conditioned SR fiducial (Afixfid) 0.137
  => conditioned+fiducial SR is WORSE than BOTH scrambled SR and LR at the summary level.
  Cause: fiducial conditioning imposes a per-box cosmology bias that distorts SR P(k) ->
  q_SR mis-transfers. The "headroom" the summary level had (0.088) was NOT unlocked; it got worse.
PQMass (1-point count histogram, 200 val boxes, my own impl since not pip-installable):
  LR vs HR chi2/dof=0.83 (MATCHES); SR vs HR chi2/dof=2.22, 64% p<0.05 (DISTINGUISHABLE).
  => SR's count PDF (rounded exp(model)) is statistically off from real halo counts; LR (real
  halos) matches HR. Gross stats fine (SR total 418k~LR 417k vs HR 442k; void 0.871=LR).
  PQMass catches a generation/rounding artifact P(k) hides.
PANELS (diag_cmass, LR added to pk panel, fiducial + oracle):
  cross-coherence r(k): SR stays high (~0.5 at Nyquist), LR decorrelates to ~0/neg -> SR>>LR on PHASE.
  projection overshoot (SR brighter) is a log1p-delta VIZ effect, present in oracle AND original too,
  NOT a fiducial artifact (I initially mis-attributed it; corrected).
FULL HONEST PICTURE of the conditioning retrain:
  - val_pk (oracle field fidelity, theta known): 0.065 vs scrambled 0.104 -> conditioning helps IF theta known.
  - field cross-fid: fiducial 0.037 / oracle 0.024 / scrambled 0.022 (all ~floor; saturated; no gain).
  - summary cross-fid: fiducial 0.137 (WORSE than scrambled 0.087 and LR 0.031).
  - PQMass: SR 1-point worse than LR.
  CONCLUSION: correct conditioning helps ONLY in the oracle sense (theta known); at DEPLOYMENT
  (theta unknown -> fiducial) it does NOT improve, and hurts the summary metric. The mentor's
  "conditioning hugely important" holds only when cosmology is known, which is not the inference setting.
NEXT: compute ORACLE (true-theta) summary cross-fid to quantify the deployment gap at summary level too.

## D-F20 addendum: oracle summary cross-fid + level-dependent reconciliation.
Oracle (true-theta) summary cross-fid: Afixtrue 0.160 +/- 0.112 (vs fiducial Afixfid 0.137 +/- 0.066;
within noise of each other). Both >> scrambled SR 0.087 >> LR 0.031 (floor 0.028).
=> Cosmology-conditioning the GENERATOR (either mode) HURTS the summary cross-fid: SR encodes G's
imperfect P(k)<->cosmology mapping, which misleads q_SR at transfer. theta-independent (scrambled/
unconditioned) SR is better; raw LR best.

LEVEL-DEPENDENT (the two goal-metric levels point OPPOSITE on conditioning):
  FIELD cross-fid:   oracle 0.024 < fiducial 0.037 (< but all ~floor 0.019; conditioning helps slightly)
  SUMMARY cross-fid: oracle 0.160 ~ fiducial 0.137 (both WORSE than scrambled 0.087; conditioning hurts)
Reconcile: at the field level, true-theta makes SR more HR-like per-box -> better field transfer.
At the summary level, the P(k) IS the cosmology channel, so G stamping its own P(k)-cosmology mapping
MISLEADS the summary posterior. So generator cosmology-conditioning helps the field metric marginally
but harms the summary metric.

FINAL VERDICT on the conditioning retrain (honest, for mentor):
 - It improves field FIDELITY when theta is known (val_pk 0.065 vs 0.104) -- oracle only.
 - It does NOT help the deployment goal metric: field cross-fid stays at the (saturated) floor; summary
   cross-fid gets WORSE (0.14-0.16 vs scrambled 0.087, LR 0.031).
 - PQMass: conditioned SR 1-point distinguishable from HR (fiducial 2.22 > oracle 1.58 > LR 0.83) -- both
   worse than LR; fiducial adds bias.
 - The robust SR win remains small-scale PHASE coherence (SR>>LR), field-level, regardless of conditioning.
 => "Correct conditioning is hugely important for predictions" holds only in the oracle/theta-known sense,
   NOT for deployment. For the goal metric, an unconditioned generator (or raw LR at summary) is at least
   as good. This refutes the strong form of the hypothesis, honestly.

================================================================================
D-F21  DISCRETIZATION / STOCHASTICITY PROBE (refutes rounding + Poisson fixes)
================================================================================
Motivation: pin down WHY conditioned SR is (a) worse at summary cross-fid and (b) PQMass-
distinguishable. Candidate causes: rounding (round(exp(y)-1)), missing shot noise (verified noise
std ~0, near-deterministic generator), or the P(k)-theta mapping/leak. Built 3 count-field variants
from ONE continuous generator output (val split, fiducial theta) via gen_sr_fields.py --no-round:
  cont = clip(exp(y)-1,0,200)   round = round(cont)   pois = Poisson(cont).
analysis/discretization_test.py, 200 val boxes. Job 10268552 (preempt).

RESULTS:
                PDF-L1 to HF   PQMass chi2/dof   P(k) transfer @k=0.05/0.20/0.38
  LF (control)  0.0102         0.95             --
  cont          0.0103         2.37             0.92 / 0.98 / 1.02
  round(deploy) 0.0103         2.30             0.96 / 1.06 / 1.08
  pois          0.1065        30.01             0.99 / 1.37 / 1.69

FINDINGS (two candidate fixes REFUTED):
 1. Rounding does ~nothing: cont~=round on every metric. De-rounding is NOT a fix. (Also: the PQMass
    count-histogram feature bins at half-integers = rounding thresholds, so it is rounding-invariant
    by construction -- confirmed empirically.)
 2. Poisson sampling is ACTIVELY WRONG: pois worst everywhere (PQMass 30, PDF-L1 10x, P(k)->1.7 at
    high k = added white shot noise). The generator reproduces the CLUSTERED count field directly
    (L1 on log-counts), so cont already IS the count field, not a latent rate. "Add a Poisson head"
    is the wrong fix.
 3. The real signals are NOT mean-level: round's mean 1-pt PDF matches HF as well as LF (L1 0.0103 vs
    0.0102) and mean P(k) transfers within ~8%, yet PQMass rejects (2.30 vs 0.95) and summary cross-fid
    is worse (0.137 vs 0.031). So:
      - PQMass badness = box-to-box DISTRIBUTION (higher moments) of the count stats. Consistent with
        UNDER-DISPERSION from the near-deterministic (noise~0) L1-mean-regressed generator.
      - Summary badness = P(k)-theta MAPPING distortion / conditioning leak, not the mean P(k), not
        discretization.

IMPLICATION: not a bug. De-rounding and Poisson heads are dead ends. Real levers = (i) distribution-
matching objective to revive the L1-killed stochastic scatter so box-to-box dispersion matches real
fields (not Poisson), and (ii) theta-independent generator for the summary leak. Directly supports the
paper's distribution-matching thesis. Cheap confirm pending: per-box std of each PQMass feature bin,
SR vs HF, to show under-dispersion explicitly. Fig: figures_cmass/discretization_Afix_fiducial.png

================================================================================
D-F22  SAMPLE VARIANCE (generator is deterministic; PQMass gap is small + N-dependent)
================================================================================
M=12 noise draws/box, 60 val boxes, fiducial theta, deployed (rounded) SR. analysis/
sample_variance.py, job 10268883 (preempt).

RESULTS:
 - GENERATION variance = 0.00% (per-voxel std/mean across 12 samples). The DEPLOYED (rounded)
   SR is bit-identical across noise draws: tiny noise (std~2e-4) is erased by round(). The
   generator is effectively deterministic -> drawing more samples gives the SAME field.
 - Transfer T(k)=P_SR/P_HF band widths (mean over k): generation=0.0001 vs box-to-box=0.435.
   Generation band is ~4000x thinner than field-to-field scatter -> razor-thin line.
 - PQMass chi2/dof vs HF at 60 boxes: SR=1.02 +/- 0.03, LF=0.64, HF-self floor~1.08.
   The +/-0.03 is TESSELLATION randomness (fields identical), NOT generation spread.

CAVEAT / RECONCILE with D-F21 (200 boxes: SR 2.30, LF 0.95): PQMass two-sample chi2 grows
with N for a fixed distributional gap. At 60 boxes SR sits inside the null bracket (LF 0.64,
floor 1.08); at 200 boxes it separates (2.3). => the SR 1-point distributional mismatch is
REAL but SMALL: needs ~200 boxes to detect. Softens "PQMass is bad" to "subtle, N-dependent".

"IF VARIANCE TOO HIGH" branch: NOT triggered. Variance is ~0 (too LOW). Averaging more samples
does nothing (identical fields), so inference-vs-#samples is moot. The lever is to MAKE the
generator stochastic (distribution-matching objective to revive the L1-killed noise), not to
sample more. Fig: figures_cmass/sample_variance_Afix_fiducial.png

================================================================================
D-F23  PLOT AUDIT: per-box P(k) error, hidden-by-axes, and honest cross-fid error bars
================================================================================
Trigger: mentor + user flagged the P(k) panel (LR~SR~HR) as inconsistent with the cross-fid
bar (SR worse at summary), and insisted SR should be CLOSER to HR since LR is a low-quality sim.

1. CORRECTION to an earlier claim. I had said "LR already matches HR's P(k), nothing to fix."
   WRONG: that was the MEDIAN transfer (~1.02). Per-box (paired, same IC box) RMS P(k) error
   vs HR over 1600 train boxes:  LR 12.1%   SR conditioned 13.5%   SR scrambled 14.0%.
   SR does NOT reduce LR's P(k) error; it is marginally worse and beats LR on only 26% of
   boxes. Mentor is right: SR should fix P(k) and currently does not. Caveat: part of the 12%
   is per-box high-k shot noise (irreducible), but SR reduces none of the structured part.
   Cause: P(k) loss disabled (_nopk); nothing pulls per-box ratios to 1. Lever: re-add P(k) loss.

2. WHY THE PANEL HID IT (make_pk_panel, diag_cmass.py): (a) top P(k) band is 16-84 pct across
   boxes spanning the cosmology prior -> factor ~5 band that swamps a 12% matched-box error;
   (b) transfer panel uses log y 1e-2..1e2 (4 decades) so 12% is a sliver at y=1, AND drops
   the LR band entirely (tl_med,_,_=band(tlr)); (c) both plot the median, cancelling +/- per-box
   deviations. Data was computed consistently (same k-grid/estimator/normalization) -> NOT a
   computation bug, an aggregation/axis choice that erased the effect.
   Fix figure: figures_cmass/pk_transfer_perbox_honest.png (linear axis, both bands, RMS hist).

3. SEED ROBUSTNESS of the summary cross-fid (analysis/seed_xfid_check.py, job 10361507, 200
   test boxes, per-seed q_X(s) vs q_HR(s)):
     HR floor pairs: 0.031,0.034,0.022,[0.52],0.034,0.033,[0.54],0.025,[0.54],[0.53]
     LR:   0.048,0.045,0.049,0.055,[0.012]    Anopk: 0.149,0.161,0.101,0.071,[0.004]
     Afixfid: 0.157,0.129,0.133,0.220 (seeds 0-3 only)
   => NDE SEED 4 IS BROKEN: every HR-vs-HR pair with seed 4 explodes to ~0.53 (floor should be
   ~0). Its LR/Anopk values (0.012/0.004) are garbage (compared against broken q_HR). ~1 in 5
   NDE fits fails catastrophically -> individual cross-fid numbers NOT trustworthy to 2 decimals.
   Healthy seeds only: floor 0.030+/-0.005 < LR 0.049+/-0.004 < SR scrambled 0.121+/-0.036
   < SR conditioned 0.160+/-0.036. ORDERING IS ROBUST (SR worse at summary is real), but the
   spread is wide, so the old single-number bar (0.137) with tiny error bars was misleading.
   NOTE per-seed LR mean (0.049) > earlier ensembled LR (0.031): ensembling pools samples across
   seeds and lowers the KL; both methods agree LR << SR.

4. FIELD level error bars (from eval_field_crossfid logs, mean+/-std over HR-ref x replicate):
   LR 0.471+/-0.782 (bimodal, std>mean)  SR scrambled 0.022+/-0.010  SR conditioned 0.037+/-0.014
   floor 0.019. Field SR>>LR is a 10x effect, robust.

5. HONEST CHART: analysis/plot_crossfid_honest.py -> figures_cmass/crossfid_honest_cmass.png.
   Supersedes plot_crossfid_conditioned.py / crossfid_conditioned_cmass.png (tiny error bars,
   single fit). Both delivered to user via SendUserFile.
TAKEAWAYS: (i) SR must reduce per-box P(k) error and doesn't -> re-add P(k) loss; (ii) the
cross-fid metric needs per-seed reporting + a broken-fit check (HR floor per pair) before any
number is quoted; (iii) the P(k) panel should use a linear transfer axis with both bands.

================================================================================
D-F24  P(k) PANEL FIXED (linear transfer axis, both bands) + SR large-scale power DEFICIT
================================================================================
make_pk_panel (analysis/diag_cmass.py) patched: transfer panel is now LINEAR y 0.6..1.5 (was
log 1e-2..1e2), draws BOTH LR and SR 16-84 bands (LR band was discarded), and prints the RMS
of (P_X/P_HR - 1) in the legend. Regenerated cmassAfix panels (fixcond model, fiducial theta,
16 test sims, n_bins=40), job 10361866 (preempt). diag_cmass.slurm header fixed to preempt recipe.

RMS err vs HR now visible on the figure:
  per-patch 64^3:  LR 36.9%   SR 35.8%
  full box 128^3:  LR 33.1%   SR 31.3%
=> SR ~= LR; SR is NOT tighter. Consistent with D-F23 (per-box 12.1% vs 13.5%).
CAVEAT on the absolute RMS: it is binning/mask/N dependent. D-F23's 12% used n_bins=32, 1600
boxes, low-k bins masked; the panel uses n_bins=40, 16 sims, and keeps the very-low-k few-mode
bins (SR dips to ~0.78 at k~0.012) which inflate it to ~33%. Per-patch is larger still (8x less
volume -> bigger shot-noise term). Quote only the ROBUST statement: SR does not reduce LR's
per-box P(k) error. Do not quote one RMS number as "the" error without stating the binning.
Also: one k-bin (~0.05) spikes in both LR and SR bands = a few-mode bin, not a model effect.

NEW, now visible on the full-box panel: SR/HR MEDIAN sits at 0.90-0.97 for k < 0.1, i.e. SR has
a systematic 5-10% LARGE-SCALE POWER DEFICIT, while LR/HR sits at ~1.03-1.05 (slight excess).
Matches the per-k Afixfid/HR transfer in the plot-audit (0.89-0.95 at low-mid k). This is the
STRUCTURED (fixable) part of the error: the generator under-powers large scales. Neither LR nor
SR is right, but SR's error is a coherent deficit -> precisely what a P(k)-matching loss fixes.
(P(k) loss is disabled in the fixcond model, _nopk.)

BUGS / LOOSE ENDS:
 - Legend hardcoded "per-patch RMS err": WRONG on the full-box panel (it is per-box). Fixed in
   code -> "RMS err vs HR" (title carries patch/box context). Box panel needs a RERUN to relabel.
 - Paper tex (section6_srs.tex fig:srs-pkpanel) includes pk_panel_box_cmassA.png (OLD arm A
   tag), NOT the regenerated cmassAfix. Regenerating cmassAfix does NOT update the paper figure
   until the tex is repointed to cmassAfix or the diag is rerun with TAG=cmassA. USER-GATED.
Delivered: pk_panel_patch_cmassAfix.png via SendUserFile (correctly labeled). Box panel held
until relabel rerun.
D-F24 addendum: relabel rerun done, job 10361940 (preempt). Both cmassAfix panels regenerated
with legend "RMS err vs HR" (box 33.1%/31.3%, patch 36.9%/35.8%); content unchanged, label bug
closed. Box panel delivered to user via SendUserFile. Paper tex pointer (cmassA vs cmassAfix)
still USER-GATED, not touched.
D-F24 addendum 2 (paper): USER APPROVED repointing fig:srs-pkpanel. section6_srs.tex line 604
includegraphics -> pk_panel_box_cmassAfix.png; caption rewritten to match the honest panel
(conditioned generator, fiducial theta, 16 test boxes; linear transfer with BOTH bands + RMS in
legend; SR median <1 for k<0.1 = 5-10% large-scale deficit; SR does not tighten scatter vs LR;
SR r(k) stays high while LR decorrelates). Old caption claimed "Transfer ~1 across scales" and
"Arm A": both now wrong for this figure, removed. RMS numbers deliberately NOT hardcoded in the
caption (binning-dependent, D-F24 caveat); caption says the legend reports them.
LEFT UNTOUCHED, FLAGGED TO USER (outside the approval):
 - lines 598-600 prose names projection_box_cmassA_set23.png / projection_patch_cmassA_set23_p0.png
   (old-tag); regenerated files are ..._cmassAfix_... .
 - line 584 seam_cmassA_naive_slice.png: a different (seam/stitching) old-Arm-A figure.
 - line ~507 table caption "Transfer ~1" over 200 boxes: MEDIAN-only claim; per-box error 12%+
   (D-F23). Needs a deliberate content correction, not a silent edit.
D-F24 addendum 3 (paper, scope decision): USER CHOSE "free repoints only". section6_srs.tex
lines 597-599 prose filenames repointed cmassA -> cmassAfix (pk_panel_patch, projection_box_set23,
projection_patch_set23_p0); all three cmassAfix files confirmed on disk (16:40, job 10361940).
USER DECLINED (do NOT re-raise as pending; only revisit if asked):
 - regenerating seam figure (line 584 seam_cmassA_naive_slice.png) + match stats (line 532
   match_stats_cmassA.txt) for cmassAfix. No cmassAfix versions exist; would need an eval_seam /
   match-stats preempt job on the fixcond model. Section still shows Arm A for these two.
 - correcting the "Transfer ~1" claim (line ~507, 200-box table caption). Still median-only;
   per-box error 12%+ and 5-10% SR large-scale deficit (D-F23/D-F24). Left as written.
Net state of Section 6: P(k) panel figure+caption and the patch/projection prose document the
current cmassAfix model; seam figure, match-stats ref, and the line-507 transfer claim still
document old Arm A. This inconsistency is KNOWN and user-accepted for now.

================================================================================
D-F25  SAMPLE-AVERAGED SR P(k) PANEL: averaging 10 draws changes nothing (confirms D-F22)
================================================================================
User asked for the per-patch P(k) panel with SR as the MEAN of 10 noise draws vs 1 LR patch vs
the paired HR patch. analysis/pk_panel_sampleavg.py, job 10362355 (preempt). 16 test sims x 8
patches, M=10 draws, n_bins=40, lbox_patch=500; SAME estimator as diag_cmass; seed 0 == the
existing panel's SR, so SR-single IS pk_panel_patch_cmassAfix's SR (verified: RMS 35.8% matches).
SR kept as CONTINUOUS to_counts output (no rounding), matching the existing panel, so the ONLY
difference between SR-single and SR-avg is the averaging.

RESULT: the 10-sample mean is indistinguishable from a single sample.
  averaging effect on the FIELD: mean|avg-single|/mean = 0.058%   (continuous; rounding -> 0.00%, D-F22)
  RMS err vs HR:  LR 36.9%   SR-single 35.8%   SR-avg10 35.8%
  max |T_avg - T_single| over k = 0.00008     max |r_avg - r_single| over k = 0.00001
  Figure: green dashed (avg) sits exactly on blue solid (single) in P(k), transfer, coherence.
CONCLUSION: "average multiple SR samples to get closer to HR" is NOT a viable lever with this
generator: there is no sample diversity to average over (noise std ~2e-4). A posterior-mean
field would only reduce small-scale power / tighten transfer if the generator were genuinely
stochastic, which requires the distribution-matching objective (revive the L1-killed noise),
not more draws. Consistent with D-F22 (0.00% sample variance) and the inference-vs-#samples
branch being moot. The 0.058% field difference is the un-rounded noise-std footprint, not signal.
Delivered: pk_panel_patch_sampleavg_cmassAfix.png via SendUserFile.

================================================================================
D-F26  WHAT THE MODEL IS FED: real box sample + a conditioning no-op in block0
================================================================================
analysis/show_input_sample.py, job 10363187 (preempt), real weights (fixcond), real box idx=23.
DATA: /data/group_data/universedata/cmass-ili/processed/{idx:04d}_{input,label}.npy (LR=FastPM,
HR=Quijote), cosmology from quijote/nbody/L1000-N128/{idx}/config.yaml -> data/cmass_theta.npz.
INPUT = a (1,128,128,128) integer halo-count grid (7.8 Mpc/h cells) + the raw 5-vector theta.
Box 23: LR 555,162 halos (mean 0.2647/voxel, max 12); HR 568,139 (0.2709, max 13). ~82.6% of
voxels are EMPTY (1.73M of 2.1M); ~245k hold 1; ~77k hold 2. theta=[0.3069 0.0561 0.8349 1.0145 0.7043].
Concrete phase point: in a 6x6 slab at z=64, LR and HR have halos in DIFFERENT cells despite
being the same universe (e.g. LR[0,0]=1 vs HR[0,0]=0; HR[5,0]=1 vs LR[5,0]=0). This is the
real-space face of "same P(k), different phases".
SR output for the box: total 556,082 (mean 0.2652, max 15) ~= LR's total, NOT HR's 568,139.
SR did not recover the ~13k halo deficit: concrete instance of SR not fixing amplitude/count.
DIMENSION FLOW (printed live): (1,128^3) counts -> log1p -> (8,1,64^3) patches; theta (5,)->(8,5);
block0 -> (8,128,64^3); 4x HBlock_const -> (8,64,64^3); out (8,1,64^3) -> expm1 -> stitch
(1,128^3) -> round uint8. Grid size never changes (same-resolution correction).
CORRECTION to what I told the user earlier: block0's style_block is Linear(5->1), NOT 5->128,
because block0 has in_chan=1. It yields ONE scale (0.9897) for the single input channel.
And with Cin=1, modulation multiplies the whole kernel by one scalar and demodulation divides
it straight back out (w*s/||w*s|| = w/||w||), so theta has ZERO effect on block0. Verified
empirically with a random ConvStyled3d on CPU (two thetas, same input):
Cin=1  conv, two different thetas: max|out_a - out_b| = 7.15e-07  -> NO-OP: theta cancelled by demodulation
Cin=16 conv, same two thetas:      max|out_a - out_b| = 3.11e-01  -> theta HAS effect
=> theta's conditioning first BITES at HBlock_const[0].conv0 (128 input channels -> 128
relative scales survive demod). "theta modulates every layer" was overstated: every layer
receives it, but the 1-channel first conv provably ignores it.

================================================================================
D-F27  LR HALO-COUNT DEFICIT is a SPARSE CATASTROPHIC TAIL in high-Om/high-s8 boxes
================================================================================
analysis/halo_count_deficit.py. First attempt (all 2000 boxes, job 10364325) TIMED OUT at
30 min: ~32 GB of raw count fields is too slow on the universedata NFS (~18 MB/s). Rerun on
every-4th box (500, evenly spread over the prior) + 200 SR boxes, job 10365213, 14.6 min.

LR vs HR total halos (500 boxes): mean HR 495,323, mean LR 479,811.
  deficit (LR-HR)/HR: MEDIAN -1.03%  IQR -2.27%..+0.65%  |  mean -1.85%  std 8.71%
  range -84.3% .. +16.4%  |  only 66.2% of boxes have LR < HR.
  linear corr with theta is weak: Om -0.17, s8 -0.16, others ~0.
=> For the TYPICAL box LR's halo count is nearly right. QUOTE THE MEDIAN, not the mean:
   the mean/std are dominated by a tail.

THE TAIL is a cosmology corner, not a glitch. 5 worst boxes (57-84% of halos MISSING) are
ALL high-Om (0.38-0.45) AND high-s8 (0.85-0.99), with huge HR totals (920k-1.03M vs 495k
avg). 5 largest LR EXCESS boxes are all Om~0.10 (least clustered). Binned by Om*s8:
  bottom50: mean -0.55% / median -1.15%      50-80: -1.03% / -1.08%
  80-95:    mean -4.30% / median -0.72%      top5:  mean -12.45% / median -0.13%
Mean-vs-median gap in the top bin = SPARSE catastrophic failure (~1-2% of boxes), not a
smooth trend: even in the most clustered 5%, half the boxes are fine. This is FastPM
collapsing in the most strongly clustered universes. Physical (tracks Om,s8), not corruption.

SR does NOT restore the missing halos (200 boxes, deployment/fiducial): SR deficit -2.32%
vs LR -2.52% on the same boxes; share of missing halos restored: MEDIAN 3.7% (mean -16%,
unstable when HR~LR). The residual generator inherits LR's halo number, so in the
catastrophic boxes it starts 60-84% short and stays there. Strongest concrete case for a
count/P(k)-matching term.

RECONCILIATION with the summary cross-fid: P(k) is computed on delta = n/nbar - 1 with a
per-box nbar (cube_pk_counts -> counts_to_overdensity), which normalizes the TOTAL count
away. A missing-halo deficit therefore mostly does not reach the P(k) posterior; what leaks
through is a milder bias/shot-noise shift (the ~1.02-1.05 LR transfer), which is why LR
transfers well at the summary level despite a cosmology-dependent count error.
REFRAME of "LR is a low-quality sim that needs fixing": true, but LR needs its small-scale
PHASES fixed everywhere and its halo COUNT fixed only in a minority of extreme-clustering
boxes, which our SR currently cannot do.
HYPOTHESIS -> D-F28: the field CNN sees raw log1p counts (NOT nbar-normalized), so it DOES
see the deficit; that may be why LR's field cross-fid is poor (0.47) and "bimodal". NOTE
D-F18's bimodality is across SEEDS (~2 of 8 LR estimators broken), so the test must separate
seed vs box axes. Fig: figures_cmass/halo_count_deficit.png; data runs/patch_cmass/halo_count_deficit.npz

================================================================================
D-F28  PER-BOX FIELD KL: count-deficit hypothesis REFUTED; and good-LR estimators are AT THE FLOOR
================================================================================
analysis/field_kl_perbox.py, job 10368405 (preempt, 17 s). field_ckpts_fiducial (6 qhr / 8 qlr /
3 qsr deployment), 200 HR test boxes, per-box KL = kl_gauss(ref,X).mean over 5 params.
SANITY PASSED: means reproduce D-F19 exactly: LR 0.4711, SR 0.0369, HR floor 0.0193.
Seed split reproduces D-F18: exactly 2 bad LR seeds (qlr_s102 1.87, qlr_s104 1.78); the other
6 are 0.0179-0.0244.

BOX AXIS (good LR seeds) -> HYPOTHESIS REFUTED:
  per-box KL median 0.017, p95 0.042, max 0.082 (max/p95 = 2x: NOT heavy-tailed).
  corr(KL, count deficit) = +0.02 (zero); corr(KL,|deficit|) = +0.12 (weak).
  corr(KL, Om*s8) = -0.30 (NEGATIVE: high-clustering boxes transfer BETTER, not worse; SR -0.47).
  KL by Om*s8 bin is FLAT (0.023/0.016/0.017/0.021); top-5% boxes carry 5.4% of KL (5% = uniform).
  8 catastrophic-count boxes (deficit<-20%, incl. one at -73%): KL 0.024 vs 0.020 elsewhere,
  both ~floor. The count catastrophe does NOT hurt field-level transfer.
SEED AXIS: the 2 bad seeds fail UNIFORMLY (excess 1.77 on cat boxes vs 1.81 elsewhere;
  corr(excess,deficit)=+0.10). They are globally broken training runs, not data-driven failures.
=> LR's field problem is SEED-LEVEL TRAINING INSTABILITY, unrelated to the count tail. The
   tail (D-F27) is a real data property worth stating, but NOT the explanation of the metric.

THE BIGGER FINDING (unanticipated, must go in the paper): a SUCCESSFULLY trained LR field
estimator is AT THE HR FLOOR (0.018-0.024 vs floor 0.019) and BEATS SR (0.037, ~2x floor).
The "SR 0.037 vs LR 0.47 at the field level" headline compares SR to a seed-average
contaminated by 2 broken runs. Honest framing (sharpening D-F18): SR's field-level advantage
is RELIABILITY, not a better ceiling: LR-trained estimators hit the floor 6/8 times and fail
catastrophically 2/8; SR-trained hit ~2x floor 3/3 (small n). Why 25% of LR trainings break
is now the open question (training instability: degenerate posterior / near-constant output?),
and it is a property of training on LR data, so "LR is a gamble" stands as a claim about the
SOURCE, not a fluke. Fig: figures_cmass/field_kl_perbox.png; data runs/patch_cmass/field_kl_perbox.npz

================================================================================
D-F29  STAGE 0 LAUNCHED: fine-tune fixcond with a rebalanced loss (P(k) loss OFF), 2 arms
================================================================================
GOAL (user): minimize the HR-vs-SR DISTRIBUTION gap ("SR closer to HR"). Diagnosis (D-F21..D-F28):
under-dispersion, 5-10% large-scale power deficit, inherited count deficit all trace to L1
(lambda_rec 2.0) dominating the adversarial term (1.0): mean-regression kills variance/power.
RETRACTED: my repeated "turn the P(k) loss back on" advice. Protocol item 2 (scratch_3:376)
removed it deliberately ("the goal metric may not feature in training-adjacent choices"), and
with it ON the SR bispectrum was 0.897 < LR 0.990; OFF it recovered to 0.986 (F13). P(k) loss
stays OFF; any re-enable is mentor-gated. Literature basis logged in chat (Goodfellow 2014;
Ledig SRGAN 1609.04802; Blau&Michaeli 1711.06077; Isola pix2pix 1611.07004 incl. "generator
ignores noise under pixel loss" = our noise std->0; Mathieu 1511.05440; Salimans 1606.03498).

DESIGN: short fine-tunes from fixcond/best.pt (epoch 18), 5 EXTRA epochs, lr 5e-5 (half),
select=l1 (protocol item 3), save-every 1, adv 1.0, pk 0. scripts/stage0_finetune.slurm.
  Arm R (rebalance):   lambda_rec 0.5 (L1:adv 2:1 -> 1:2, a 4x shift), sigma 0.   job 10368804
  Arm S (smoothed L1): lambda_rec 2.0, rec_smooth_sigma 1.5 vox (~12 Mpc/h; L1 cutoff ~k 0.08-0.1
     = where the large-scale deficit lives and LR/HR coherence breaks).           job 10368805
  ckpts -> /data/user_data/vkshirsa/cmass-ili/models/stage0_{R,S}/
GOTCHA: --epochs is the ABSOLUTE final epoch (loop = range(start, epochs)); --epochs 5 on a
resume from epoch 18 runs ZERO epochs. Launcher computes EPOCHS = ckpt_epoch+1+EXTRA on-node.
--resume re-applies --lr-g/--lr-d (code handles it), and resets best when select metric changes.

SPLIT DECISION (user): keep 1600/200/200 EXACTLY (select on val, test on the 200 test boxes),
so all Stage-0 numbers are directly comparable to every prior number. (Data is already 80/20
train/held-out; a true re-split would invalidate all prior models and was declined.)

PRE-REGISTERED SUCCESS CRITERIA (held-out only; P(k) used as a CHECK, never trained on):
  PQMass chi2/dof 2.3 -> toward 1 (200 val boxes, same convention as D-F20/21)
  large-scale transfer (k<0.1) 0.90-0.97 -> toward 1; per-box P(k) RMS <= LR's 12%
  halo-count deficit closes (SR restores > 3.7% of missing halos)
  bispectrum NOT below 0.986; r(k) NOT degraded; summary cross-fid NOT worse than 0.137
  baseline for every metric = fixcond itself, re-evaluated identically.
Interpretation fixed in advance: if a metric improves only because fidelity was bought with
more theta-leak (cross-fid worsens), that arm FAILS.
D-F29 addendum (evaluation design, fixed before results):
TIER 1 (cheap, all arms + fixcond BASELINE re-evaluated by the identical script,
scripts/stage0_eval.slurm): gen SR val+test (fiducial) -> PQMass (val, 200) -> diag panels
(test, 16: transfer = held-out CHECK only, cross-coherence) -> bispectrum (test, 32) ->
halo-count deficit (SR part; LR/HR totals cached in halo_count_lrhr.npz, seeded from the
500-box run so no arm re-reads them). Baseline job launched NOW (its bispectrum was never
measured: F13's 0.986 was the old Anopk model), so comparison numbers arrive during training.
TIER 2 (expensive, survivors only): gen 1600 train SR -> pk_from_srfields -> NDE -> summary
cross-fid. It is a GATE for declaring a winner, not a screening metric.
Quirks handled: bispectrum_cmass --mode dirs expects set{sid}_transformed.npy (launcher
symlinks to sr_{sid}.npy); halo_count_deficit.py now takes SRDIR TAG and suffixes outputs, so
arms do not overwrite each other. Caveat: the count stat uses every-4th boxes that HAVE SR:
~100 for R/S (val+test only) vs up to 400 for the baseline (train SR exists); note N in readout.
D-F29 addendum 2: BASELINE (fixcond) re-evaluated by the identical Tier-1 script, job 10368814,
10.6 min. These are THE reference numbers for Arms R/S (same script, same boxes, same seeds):
  PQMass chi2/dof (val, 200):            LR 0.83   |  SR(fixcond) 2.22
  bispectrum, equilateral mid-k ratio to HR (test, 32):  LR 1.112  |  SR 1.050   (1.0 = matches HR)
  halo-count deficit (400 SR boxes):     LR -2.12% |  SR -1.92% (median -0.80%); SR restores median 3.4%
  large-scale transfer k<0.1 (from diag, same model as cmassAfix): SR 0.90-0.97, LR 1.03-1.05
  summary cross-fid (prior, D-F20/D-F23): LR 0.049, SR 0.137-0.160, floor ~0.03
CRITERION CLARIFICATION (same intent, comparable statistic): F13's "bispectrum 0.986" was the
OLD Anopk model and is NOT the statistic this script prints, so it cannot serve as the bar. The
bar is the identically-measured baseline: an arm's SR bispectrum ratio must not move FURTHER
from 1.0 than 1.050 (fixcond). Note SR (1.050) already beats LR (1.112) on this statistic.
Pass/fail for R and S therefore = vs THIS table: PQMass 2.22 -> toward 1; transfer 0.90-0.97
-> toward 1; count restore 3.4% -> higher; bispectrum |ratio-1| <= 0.050; r(k) not degraded;
Tier-2 cross-fid not worse than 0.137.
D-F29 HANDOFF (user disconnecting; in-session watchers will die, SLURM jobs will not):
  training  Arm R 10368804 -> logs/10368804_stage0ft.log ; Arm S 10368805 -> logs/10368805_stage0ft.log
  Tier-1 eval  baseline 10368814 (DONE) ; Arm R 10368818 (afterok R) ; Arm S 10368819 (afterok S)
            -> logs/{JID}_stage0eval.log ; figures_cmass/*_stage0_{R,S}.png ; runs/patch_cmass/halo_count_deficit_stage0_{R,S}.npz
  checkpoints -> /data/user_data/vkshirsa/cmass-ili/models/stage0_{R,S}/ (best.pt = val-l1 selected)
  RESUME: `squeue -u vkshirsa`; then read the three eval logs, compare to the baseline table in
  addendum 2, apply the pre-registered criteria, and ONLY THEN decide Tier-2 (scripts/stage0_tier2.slurm,
  not submitted; ~8-10 GPU-h per arm, launch for survivors only).

================================================================================
D-F30  STAGE 0 TIER-1 RESULTS: Arm R (adv up / L1 down) CLOSES the PQMass gap; Arm S fails
================================================================================
Training (5 extra epochs each, ~70 min/epoch, ~5.9 h): R job 10368804, S job 10368805.
  R val_L1: .1173 .1180 .1150(best,ep21) .1171 .1220 | val_pkRMS .107 .125 .062 .088 .134  (oscillating)
  S val_L1: .1205 .1225 .1194 .1222 .1182(best,ep23) | val_pkRMS .110 .106 .133 .115 .087
  NOTE file epoch_N.pt = log epoch N-1 (saver uses epoch+1). R best.pt = log ep21 (val_pk 0.062,
  BETTER than fixcond's 0.0654 despite L1 weight /4). GAN oscillation is real (R pk 0.06->0.13).
Tier-1 (identical script; baseline = fixcond, same boxes/seeds), evals 10368818 (R) 10368819 (S):
                          LR      fixcond    Arm R     Arm S      pre-registered bar
  PQMass chi2/dof (val)   0.83    2.22       0.75      2.47       toward 1
  PQMass frac(p<.05)      0.12    0.64       0.13      0.67
  bispec eq mid-k ratio   1.112   1.050      1.044     1.110      |ratio-1| <= 0.050
  transfer RMS err        33.1%   31.3%      31.6%     31.8%      not worse
  large-scale T (k<0.1)   1.03-5  0.90-0.97  ~0.93-.97 ~0.93-.98  toward 1
  r(k)                    collapse high      unchanged unchanged  not degraded
  count restore (median)  --      3.4%(400b) 0.2%(90b) 16.4%(90b) higher
VERDICT (per criteria fixed in D-F29):
  ARM R SURVIVES: PQMass 2.22 -> 0.75 = statistically indistinguishable from HR, as good as the
    real-halo LR control (0.83); frac(p<.05) 0.13 = null rate. Bispectrum 1.044 (pass, slightly
    better than baseline). r(k), transfer unchanged. val P(k) not degraded (0.062). The one-point
    DISTRIBUTION gap is CLOSED by giving the adversarial (distribution-matching) term more weight,
    exactly as SRGAN / Blau-Michaeli predict. NOT fixed: large-scale power deficit (still 0.93-0.97)
    and halo count (restore ~0). These are amplitude/mean-level gaps the adversarial term does not
    target -> need a different lever (count/amplitude term), not more of this one.
  ARM S FAILS: PQMass WORSE (2.47) and bispectrum 1.110 = LR level, outside the 0.050 bar. Blurring
    L1 removed the small-scale anchor without strengthening the distribution matcher, so small
    scales drifted. Its one positive, count restore median 16.4% vs 3.4%, is on 90 boxes with
    std 12.6%: SUGGESTIVE only. Does not advance.
CAVEATS: count stat for R/S is on 90 boxes (every-4th of val+test) vs 400 for baseline; compare
  each arm to LR-on-the-same-boxes (-3.16%), not to the baseline's -1.92%. R's 0.75 is from ONE
  val-selected epoch of an oscillating GAN -> robustness check across all saved epochs launched
  (PQMass vs epoch). fixcond's own val_L1 not in hand; L1 fidelity cost of R not quantified yet.
NEXT: (i) PQMass-vs-epoch for R; (ii) Tier-2 cross-fid gate for R (stage0_tier2.slurm);
  (iii) if R passes Tier-2, it replaces fixcond as the working model and the remaining gaps
  (large-scale power, count) get their own lever.
D-F30 addendum: Tier-2 for Arm R = job 10373720 (stage0_tier2.slurm); PQMass-vs-epoch job in logs/*_pqepochR.log
D-F30 addendum (robustness of Arm R's PQMass), job 10373719, 9.8 min, PQMass on EVERY saved epoch:
  log epoch:   19     20     21(best)  22     23        | fixcond baseline 2.22, LR control 0.83
  PQMass:      4.31   1.19   0.75      1.32   2.36
  val_pkRMS:   .107   .125   .062      .088   .134
  frac(p<.05): 0.90   0.23   0.13      0.28   0.64
VERDICT: NOT a lone lucky epoch: three consecutive epochs (20-22: 1.19, 0.75, 1.32) sit in the
matches-HR regime, bracketing the selected one, so the rebalanced objective genuinely REACHES the
distribution-matched regime. BUT it is NOT STABLE: epoch 19 (4.31) is the shock right after the
loss-weight change, and epoch 23 (2.36) has drifted back to baseline level. The good regime is a
~3-epoch WINDOW, not a plateau: classic GAN oscillation / non-convergence. The protocol's val-L1
selection happened to pick the PQMass optimum (ep21), which is reassuring for the protocol but is
one run. Reading for the paper: the distribution gap is CLOSABLE by rebalancing (real, reproducible
over a window), and the remaining problem is training STABILITY, not the objective.
FIX DIRECTION (not launched): generator weight EMA (G_ema, standard in StyleGAN2, Karras et al.
1912.04958; averages over the oscillation), and/or lower LR / checkpoint averaging over the window.
Tier-2 (cross-fid gate) proceeds on best.pt (ep21) as pre-registered.

================================================================================
D-F31  STAGE 0 FINAL: Arm R PASSES the Tier-2 cross-fid gate -> Stage-0 winner (with a stability caveat)
================================================================================
Tier-2 job 10373720, 45.8 min (gen 1600 train SR 23.5 min, P(k) set, 4 NDE seeds x 101 epochs,
ensemble cross-fid via padding_crossfid with stage0_R added). SAME run, SAME HR members/seeds/test
boxes as the comparison rows, so directly comparable:
  summary cross-fid KL to HR (mean over test boxes, +/- box std):
    stage0_R  0.1363 +/- 0.081   <- Arm R
    Afixfid   0.1373 +/- 0.066   <- fixcond baseline (deployment)
    Afixtrue  0.1599             <- fixcond oracle
    Anopk     0.0870             <- old theta-independent SR
    LR        0.0306   HR floor 0.0278
GATE: "not worse than 0.137" -> PASS (0.1363, identical within the +/-0.07 spread). Arm R did
NOT buy its distribution win with extra cosmology leak. It also did not REDUCE the leak: still
~4.5x LR, ~5x floor. Expected and important: the rebalance targets the DISTRIBUTION axis; the leak
lives in the CONDITIONING. Stage 0 is direct evidence the two axes are separable levers
(PQMass 2.22 -> 0.75 with cross-fid unchanged): Blau-Michaeli fidelity/distribution axis vs the
decorrelation/theta-independence axis. Neither lever moves the other.

STAGE-0 WINNER: Arm R (lambda_rec 0.5, adv 1.0, P(k) loss OFF, best.pt = log ep21).
  vs fixcond: PQMass 2.22 -> 0.75 (matches HR; LR control 0.83) | bispectrum 1.050 -> 1.044 |
  transfer RMS 31.3 -> 31.6% (=) | r(k) = | val P(k) 0.0654 -> 0.062 | cross-fid 0.137 -> 0.136 (=)
  NOT fixed: large-scale power deficit (0.93-0.97), halo count (~0 restored), theta-leak.
  CAVEAT: the matched regime is a ~3-epoch WINDOW (ep20-22: 1.19/0.75/1.32; ep23 back to 2.36).
  Arm S: failed Tier-1 (PQMass 2.47, bispectrum 1.110); did not advance.
Checkpoint: /data/user_data/vkshirsa/cmass-ili/models/stage0_R/best.pt (NOT yet copied to
group_data; adoption as the working model is user-gated). Tier-2 artifacts: sr_fields/stage0_R
(train+val+test), runs/patch_cmass/pk_stage0_R_train, _robust_tmp/qstage0_R_{0..3}.pkl.
NEXT (user to choose): (A) G_ema stabilization fine-tune of R (window -> plateau, ~6 h);
(B) the leak: theta-independent G / adversarial decorrelation (new code); (C) amplitude/count
term for the large-scale deficit + halo count (new code; count is not an eval metric, hygiene OK);
(D) write Stage 0 up for the mentor before more compute.
D-F31 addendum (user chose: pause compute, write Stage 0 up for the mentor first):
Memo published as a private artifact (share from the page's menu):
  https://claude.ai/code/artifact/838f7505-671d-4847-93b7-48a505068553   (title "Stage 0 Loss Rebalance", v1)
  source: scratchpad/stage0_memo.html (543 KB, 4 figures embedded: baseline + R PQMass, R per-epoch
  stability curve [new, from D-F30 numbers], R full-box panel). No em-dashes. Contents: TL;DR, what R/S
  are, diagnosis, pre-registered design, Tier-1 table with pass/fail, stability, Tier-2 gate, open gaps,
  decision A/B/C for the mentor (A recommended), provenance footer with all job IDs.
No compute launched. Next action waits on the mentor's steer (A: G_ema stabilise+adopt, B: leak, C: amplitude/count).

================================================================================
D-F32  10 NOISE DRAWS, ONE BOX, BOTH MODELS: rebalancing did NOT revive the noise channel
================================================================================
analysis/pk_panel_noise10.py, job 10378222 (55 s). Box set23, fiducial theta, pad 0, 10 draws each
from a fresh np.random.default_rng(seed) N(0,1) set of 8 x (1,64,64,64) cubes (seed 0 == diag's noise).
  model                learned noise |std|      field std/mean    max_k std(T)   max_k std(r)
  fixcond (baseline)   mean 2.30e-4, max 2.4e-3   0.050%            0.00016        0.00002
  Arm R (stage0_R)     mean 2.82e-4, max 2.3e-3   0.128%            0.00049        0.00006
=> Arm R's noise amplitude is +20% (2.8e-4 vs 2.3e-4): still effectively zero. Field spread across
draws 2.5x larger but still 0.13%. All 10 P(k)/T(k)/r(k) curves coincide for BOTH models.
CONCLUSION: lowering the L1 weight 4x made the generator match the HR distribution across BOXES
(PQMass 0.75) WITHOUT making it stochastic WITHIN a box. Box-to-box distribution matching and
per-input stochasticity are separable too: the adversarial term at weight 1.0 does not pressure the
noise channel on its own. Reviving per-box sampling needs an explicit mechanism (e.g. a diversity /
mode-seeking term, or a loss that scores samples rather than the conditional mean), not weight
rebalancing. Fig: figures_cmass/pk_panel_noise10_set23.png
D-F32 addendum (plot fix, job 10380966): user noticed HR/LR LOOKED different between the two
columns. Cause: the top (log P(k)) row auto-scaled its y-axis per column to that column's SR
curves (Arm R's low-k SR spike 1.38x vs 1.27x HR made its y-range taller), so identical HR/LR
curves sat at different heights. The data were identical (computed once, reused; the fixed-axis
transfer and coherence rows were pixel-identical). Fixed with sharey="row" + masking non-positive
SR bins on the log axis; the script now prints the single HR/LR values used in both columns:
P_HR(k~0.02/0.1/0.3) = 41495/10942/3457, P_LR = 41914/11568/3554, LR/HR = 1.010/1.057/1.028.
Harmless RuntimeWarning from the LR/HR divide at empty k-bins in that print. Conclusions unchanged.
D-F32 addendum 2 (the 10 noise draws, exact): per draw 8 cubes x (1,64,64,64) = 2,097,152 values
~ N(0,1) from numpy default_rng(seed), seeds 0-9 (seed 0 == diag's). Per-draw min/max:
  s0 [-5.35, 5.00]  s1 [-4.81, 5.04]  s2 [-5.19, 5.60]  s3 [-5.62, 4.96]  s4 [-5.17, 5.00]
  s5 [-4.82, 4.76]  s6 [-4.90, 4.82]  s7 [-4.79, 5.53]  s8 [-4.96, 5.38]  s9 [-4.74, 5.19]
  every draw: mean ~0 (|mean| < 0.0013), std 0.999-1.000. OVERALL RANGE [-5.615, 5.595].
As INJECTED (x + learned_std * noise): baseline std 2.30e-4 -> perturbation in [-1.3e-3, +1.3e-3],
Arm R std 2.82e-4 -> [-1.6e-3, +1.6e-3], typical +/-2.5e-4, against feature magnitudes ~2e-2.
That ~1% perturbation of the features is why 10 draws collapse to one field. Figure title now
carries the range (regen job in logs/*_noise10.log).

================================================================================
D-F33  FLOW TAIL-FIX ARMS: evals were never queued; launched by user decision (2026-09-20)
================================================================================
State found: fm_gauss_nodq_tlate trained 60/60 (TRAIN_DONE 09-18 12:23) but NO Tier-1 for 3 days;
fm_gauss_nodq_lw2 at epoch 54/60, preempted twice (09-18 18:14, 09-19 20:57), pending behind ~780
jobs, priority decaying 3922->3877. The train_fm.slurm chain only re-submits TRAINING links; nothing
ever submits tier1. "Keep waiting" therefore could not complete the evals. Surfaced; user chose:
launch tlate Tier-1 now, gate lw2's on TRAIN_DONE.
Actions: tlate Tier-1 = tier1_fm.slurm TAG=fm_gauss_nodq_tlate (see chat for job id).
lw2 gate = scripts/lw2_tier1_gate.slurm, job 10509088, afterany:10491448; it submits tier1 iff
$CK/TRAIN_DONE exists, else re-gates itself on the newest fm_fm_gauss_nodq_lw2 link and exits
(robust to further preemptions; holds no idle slot; stops if the chain dies).
Val-curve prior (both arms): best.pt is an early epoch (lw2 best 0.1847 <= ep19; tlate ~0.177 ep34-37);
tlate OVERcounts (+1..+6%) with pkRMS 0.08-0.09; lw2 undercounts 5-6%. Neither curve suggests the
dense tail filled; the Tier-1 histogram is the actual test.

================================================================================
D-F34  BOTH DENSE-TAIL FIXES FAIL the pre-registered bar (Tier-1, 2026-09-20/21)
================================================================================
tlate Tier-1 job 10509086 (09-20); lw2 trained 60/60 (TRAIN_DONE 09-21 14:40), gate 10509088 fired
and submitted Tier-1 10522937 (09-21). Same script/boxes/seeds as prior arms.
                         fm_gauss_nodq(ref)   tlate          lw2            bar
  PQMass chi2/dof        2.85                 4.24 (worse)   8.13 (much worse)   null bracket
  frac(p<.05)            0.81                 0.93           0.99
  bispec eq ratio        1.093                1.085          0.966 (pass)        |r-1|<=0.05
  count (SR-HR)/HR med   -2.4%                +6.2% (OVER)   -0.9%
  count restored (med)   -8%                  110% (overshoot)  27.9%            >=50%
  T(k<0.1) median        0.966                0.941          0.875 (worse)       0.98-1.02
  r(k) at k~0.10         (LF 0.63)            0.70 (> LF)    0.44 (< LF!)
  SSR mean               --                   0.20           0.20               ~1
  gen std / HR std       --                   0.54           0.59
  rank-hist mid/high k   --                   0.73/0.82 in TOP bin  0.87/0.82 in TOP bin
READING: neither lever filled the tail; each broke something else.
 - lw2 (dense-region loss weight): traded distribution for count. Halo count nearly right
   (-0.9%, restores 28%) and bispectrum passes, but PQMass 8.1 (worst of any arm), large-scale
   power deficit DEEPENED to 0.875, and mid-k coherence collapsed BELOW LF (0.44 vs 0.63 at
   k~0.1). Upweighting dense LF regions made the model put halos in roughly the right number
   but the wrong places / wrong spectrum. Val curve had warned: L1 never beat its ep<=19 best.
 - tlate (late-t emphasis): OVERcounts (+6% median, restores 110%), PQMass 4.2, T(k) 0.94,
   bispec 8.5%. Emphasising late t added halos indiscriminately (spurious low-count voxels),
   consistent with its val cnt_err +1..+6% through training.
 - Both: SSR 0.20 and rank histograms piled in the top bin at mid/high k = HR ranks ABOVE all
   draws = the sampler is systematically low on small-scale power AND under-dispersed. The
   "flow is a calibrated sampler" premise does not hold for these arms at k>0.05.
IMPLICATION: the reweighting/time-sampling hypothesis for the dense-tail deficit is NOT
supported. The plain fm_gauss_nodq remains the best flow arm (summary KL 0.058, PQMass 2.85).
Remaining candidate causes: capacity (17.4M; --ch-mult 1,2,4,8 = 41.7M untested), or the
gaussian source itself (regenerating large scales from noise) -- the lf_spectral source with
the round decoder scored 0.144/3.41, so that is not a free win either. Histogram job
(scripts/fm_hist_tailfix.slurm, diag_hist on nodq/lw2/tlate/stage0_R) pending for the per-bin view.
D-F34 addendum (per-bin histogram, job 10535167, 100 val boxes, diag_hist; ratio to HR by bin):
  bin:           0      1      2      3      4      5     6-7   8-11  12-20   >20   | box-to-box std/HR std (bins 3..8-11)
  LR             1.008  .967   .927   .908   .903   .893   .900   .923   .760   0     | .90 .87 .85 .86 .88
  fm_gauss_nodq  1.008  .969   .955   .935   .736   .787   .741   .703   .795   1.6   | .92 .74 .79 .73 .66
  lw2            1.003  1.007  1.017  .788   .809   .753   .734   .792   1.19   5.1   | .80 .85 .77 .72 .72
  tlate          1.001  1.011  .983   .957   .948   .789   .887   .808   .769   .19   | .80 .79 .64 .69 .59
  stage0_R (GAN) 1.008  .967   .927   .897   .888   .902   .958   1.057  .948   0     | .88 .85 .87 .93 1.00
THREE FINDINGS:
 1. lw2 "restored" the count with the WRONG voxels: extra 1s and 2s (1.007, 1.017, above HR) plus a
    handful of absurd >20 spikes (5.1x HR; box-to-box std 2.6x), while the dense bins 3..6-7 stayed
    at 0.73-0.81 (n=3 got WORSE, .935 -> .788). Explains PQMass 8.1 and the coherence collapse.
 2. tlate has the BEST mean dense tail of any flow arm (n=3 .957, n=4 .948, 6-7 .887, 8-11 .808 vs
    nodq .935/.736/.741/.703), i.e. late-t emphasis DID move the mean tail toward HR. It fails PQMass
    (4.2) because it is the most UNDER-DISPERSED across boxes (std ratios 0.53-0.89 in every tail bin)
    and over-produces 1s. So the tail deficit is movable; the price was box-to-box diversity.
 3. The GAN (stage0_R) passes PQMass (0.75) WITHOUT a better mean tail than tlate (n=4 .888 < .948):
    it is near-identity to LR up to n=5 and edits only n>=6, and its box-to-box std ratios are the
    closest to 1 of any source (0.85-1.00). => PQMass, as implemented (per-box histogram features,
    tessellated across boxes), is dominated by BOX-TO-BOX DISPERSION of the histogram, not by the
    mean tail. The flow's PQMass failure is therefore primarily its under-dispersion (SSR 0.20,
    rank-hist piled in the top bin), i.e. p(HF|LF) learned too narrow, not the mean dense tail.
REFRAME: "dense-voxel tail deficit" was a partial description. The flow's real defect is a
conditional distribution that is too narrow (under-dispersed) at k>0.05, which shows up as (a) HR
above all draws, (b) box-to-box histogram std 0.6-0.8 of HR, (c) a mean tail deficit. Levers for
WIDTH (not emphasis): a stochastic interpolant noise term gamma(t)z (Albergo 2310.03725), larger
sigma/temperature with a source that keeps large scales (lf_spectral), SDE sampling re-checked on
box-to-box std rather than the mean tail, or capacity (41.7M) if the velocity field is over-smoothed.
