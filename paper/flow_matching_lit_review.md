# Flow matching for LF -> HF halo-field correction: literature review

Compiled 2026-09-09. Scope: replacing the deterministic, under-dispersed GAN + L1
corrector (CHARM -> Quijote halo count fields, 128^3, paired by initial conditions)
with a flow-matching or diffusion corrector whose outputs support unbiased
downstream inference. Four independent searches, about 100 arXiv abstract pages
fetched. Every arXiv ID below was checked against its abstract page; full texts
were not read, so treat design details as "per abstract" until you open the paper.

## 1. Bottom line

The literature outside cosmology has converged on one recipe for exactly this
problem (cheap solver -> expensive solver, paired but imperfectly aligned,
stochastic small scales):

1. Start the generative process from the LF field (or from a deterministic
   prediction made from it), not from Gaussian noise.
2. Keep the velocity/score network conditioned on the LF field for the whole
   trajectory, otherwise the bridge preserves the HF marginal but not the pairing.
3. Add an explicit noise term whose scale is calibrated to the empirical LF-HF
   residual, per scale. This is what sets box-to-box spread.
4. Validate spread directly: spread-skill ratio, CRPS, rank histograms per k-bin,
   plus a coverage test on the corrector's conditional samples.

Nobody has applied this to halo count fields, and nobody validates a corrected 3D
field by the bias of a downstream SBI posterior (the closest, Zeghal et al. 2025,
does it in 2D lensing). That combination is open and is the paper's claim.

Read first, in this order:

| arXiv | Paper | Why first |
|---|---|---|
| 2410.19814 | Fotiadis et al. 2024, Stochastic Flow Matching for Resolving Small-Scale Physics | Closest template: encoder maps LF to a latent base, FM transports to HF, noise scale learned from residual variance. Built for misaligned LF/HF from different solvers. |
| 2310.03725 | Albergo et al. 2023, Stochastic interpolants with data-dependent couplings | The clean theory: x0 = LF, x1 = paired HF, plus a gamma(t) z noise term. Code: github.com/interpolants/couplings |
| 2311.06978 | De Bortoli et al. 2023, Augmented Bridge Matching | The caveat: vanilla bridge/FM loses the coupling; fix by feeding x0 to the network throughout. |
| 2510.24631 | Zeghal et al. 2025, Bridging Simulators with Conditional Optimal Transport | Cosmology, LPT -> N-body lensing maps with conditional OT-FM, validated by recovering the true field-level SBI posterior. Your evaluation criterion, already used. |
| 2510.19224 | Horowitz, Cuesta-Lazaro, Yehia 2025, BaryonBridge | Cosmology, 3D, paired-by-ICs, stochastic interpolant from FastPM fields to hydro fields, fully convolutional. Nearest 3D cosmology precedent. |
| 2606.30821 | Kim, Soma, Dean 2026, Mind the Residual Gap | Shows mean + residual-diffusion (CorrDiff style) is still biased and under-dispersed in practice, and how to fix it. Your exact symptom. |

## 2. Three formulations and where each paper sits

| Formulation | Idea | Precedents | Fit to your setup |
|---|---|---|---|
| A. Conditional generation from noise | Sample p(HF given LF) by diffusion/FM from N(0,I), LF fed as a conditioning channel | SR3 2104.07636, Palette 2111.05826, Schanz 2310.06929, Rouhiainen 2311.05217, Bourdin 2408.00839, Sether 2412.05131, Riveros 2502.17087 | Proven in 3D cosmology. Ignores that LF is already 90 percent of the answer, so the model must regenerate large scales it could have copied. Longest trajectories. |
| B. Bridge from LF to HF | Interpolant or bridge whose endpoints are the paired (LF, HF) boxes, with a noise term | Albergo couplings 2310.03725, DDBM 2309.16948, I2SB 2302.05872, InDI 2303.11435, ResShift 2307.12348, Aligned DSB 2302.11419, A2BM 2607.16294, Wu et al. 2603.21717 | Most natural for paired-by-IC data. Learns only the residual transport. Needs x0-conditioning (2311.06978). A2BM adds a per-pair alignment score for approximate correspondence. |
| C. Mean predictor + residual generator | Deterministic net predicts E[HF given LF]; generative model samples the residual | CorrDiff 2309.15214, Fotiadis SFM 2410.19814, Chen et al. MFFM 2605.16118, Bhola and Duraisamy 2512.12749, ECMWF 2604.03303, TAUDiff 2412.13627, PDE-Refiner 2308.05732 | Lowest-risk path: reuse the existing GAN as the mean. Residual is small and roughly stationary. Known failure: residual model still under-dispersed unless the source noise is calibrated (2606.30821, 2410.19814). |

Recommendation: C first, using the Fotiadis/MFFM construction (source = mean
prediction + noise whose per-k variance matches the residual spectrum; velocity net
conditioned on LF; no theta conditioning). Then B as the ablation that drops the
GAN entirely. A is the baseline a referee will expect because it is what the
cosmology literature has used.

## 3. Method papers to adopt (tier A)

- 2410.19814 Fotiadis, Brenowitz, Geffner, Cohen, Pritchard, Vahdat, Mardani 2024. Stochastic Flow Matching for Resolving Small-Scale Physics (ICML 2025 version: Adaptive Flow Matching). Encoder to latent base, adaptive noise scale from maximum-likelihood residual variance. Compared against conditional diffusion and conditional FM from noise on 25 km -> 2 km Taiwan weather and Kolmogorov flow.
- 2310.03725 Albergo, Goldstein, Boffi, Ranganath, Vanden-Eijnden 2023. Stochastic interpolants with data-dependent couplings. x_t = alpha(t) x0 + beta(t) x1 + gamma(t) z with x0 drawn conditionally on x1. Demonstrated on super-resolution and inpainting.
- 2311.06978 De Bortoli, Liu, Chen, Theodorou, Nie 2023. Augmented Bridge Matching. Proves flow/bridge matching preserves marginals not couplings unless OT-optimal; fix is to condition the drift on x0.
- 2309.16948 Zhou, Lou, Khanna, Ermon 2023. Denoising Diffusion Bridge Models. Bridges between paired endpoints; reverse process starts from the source; EDM preconditioning and samplers with tunable stochasticity.
- 2302.05872 Liu, Vahdat, Huang, Theodorou, Nie, Anandkumar 2023. I2SB. Tractable Schrodinger bridge with analytic marginals given pairs; simulation-free training; starts from the degraded input. Well-documented NVlabs code.
- 2303.11435 Delbracio, Milanfar 2023. Inversion by Direct Iteration. Framed explicitly around regression-to-the-mean of single-step supervised losses; iterative move from input to target with a stochastic variant. Simplest bridge-style ablation baseline.
- 2605.16118 Chen, Liu, Tang, Li 2026. Multi-Fidelity Flow Matching: Cascaded Refinement of PDE Solutions. Conditional residual FM where the source distribution is calibrated to the empirical LF-HF residual statistics including local correlation; stacks across fidelity levels; eight PDE benchmarks.
- 2512.12749 Bhola, Duraisamy 2025. Residual-augmented flow matching operators. Learns the HF-LF discrepancy as a probabilistic operator conditioned on the LF solution; better UQ at small data budgets than learning HF directly.
- 2309.15214 Mardani et al. 2023. CorrDiff. UNet mean + EDM residual diffusion for km-scale downscaling; the canonical two-stage template, evaluated with MAE, CRPS, spectra, rank histograms.
- 2606.30821 Kim, Soma, Dean 2026. Mind the Residual Gap. Mean-plus-residual is biased and under-dispersive because the residual distribution at training differs from test; proposes ReMatch (OT residual matching in PCA space). Use its SSR and CRPS diagnostics.
- 2607.16294 Okabayashi et al. 2026. A2BM: Alignment-Aware Bridge Matching. Bridge matching for weakly aligned pairs with a per-pair alignment score the model conditions on. A per-box large-scale LF/HF cross-correlation could play this role.
- 2603.21717 Wu, Zhang, Yeung-Levy, Lundberg, Fox 2026. Uncertainty-Aware Distribution-to-Distribution Flow Matching for Scientific Imaging. Paired FM from the source image with a diffusion term and score correction; Bayesian variant separates epistemic and aleatoric uncertainty.
- 2305.15618 Wan et al. 2023. Debias Coarsely, Sample Conditionally. OT map removes coarse-scale solver bias, then conditional diffusion upsamples. Relevant because CHARM is a biased emulator, not merely a low-resolution one; a separate debias stage is testable on its own.

## 4. Cosmology precedents (tier B)

Stochastic LF -> HF or DM -> tracer generation:
- 2510.19224 Horowitz, Cuesta-Lazaro, Yehia 2025. BaryonBridge. Stochastic interpolant, FastPM -> Lyman-alpha hydro fields on CAMELS, paired by ICs, conditioned on cosmology and astrophysics, runs at 256^3. Validated on summary statistics only.
- 2510.24631 Zeghal, Remy, Hezaveh, Lanusse, Perreault-Levasseur 2025. Bridging Simulators with Conditional Optimal Transport. LPT -> PM lensing maps with COT-FM on unpaired data; validated by full-field SBI recovering the true posterior. Note: they condition the transport on cosmology so the bridge does not shuffle cosmologies. That looks in tension with your ablation, where theta conditioning hurt the summary-level posterior. The likely resolution is that in their unpaired setting conditioning is what defines the coupling, whereas your pairing by ICs already fixes it. Read closely before citing either way.
- 2310.06929 Schanz, List, Hahn 2023. Stochastic Super-resolution of Cosmological Simulations with Denoising Diffusion Models. Explicitly motivated by GAN sample-diversity collapse; filter-boosted loss reweighting scales. The direct citation for "GAN correctors are under-dispersed".
- 2311.05217 Rouhiainen, Gira, Munchmeyer, Lee, Shiu 2023. 3D conditional Palette diffusion, LR DM -> TNG300, iterative outpainting to 8x the training volume. Reference 3D architecture and a stitching recipe.
- 2408.00839 Bourdin, Legin, Ho, Adam, Hezaveh, Perreault-Levasseur 2024. Inpainting Galaxy Counts onto N-Body Simulations. Conditional score model painting galaxy count fields onto DM. The only cosmology paper generating a count-valued 3D field; check how they treat discreteness.
- 2412.05131 Sether, Giusarma, Reyes-Hurtado 2024. Probabilistic Galaxy Field Generation with Diffusion Models. VDM, 3D DM -> galaxy fields, paired, beats HOD.
- 2403.10648 Ono, Park, Mudur, Ni, Cuesta-Lazaro, Villaescusa-Navarro 2024. Debiasing with Diffusion. Galaxies -> DM posterior samples; cross-model generalization test is analogous to your cross-fidelity test.
- 2601.14377 Mishra, Trotta, Viel 2026. Cosmo-FOLD. Overlap latent diffusion generating arbitrarily large 3D fields from small training boxes. Principled answer to patch seams and 128^3 memory.
- 2602.10172 Islam et al. 2026. Cosmo3DFlow. Wavelet flow matching on 128^3 Quijote halo fields (halo -> IC direction). Demonstrates FM working on your exact data and resolution.
- 2502.17087 Riveros et al. 2025. Conditional Diffusion-Flow models for 3D cosmic density fields. Head-to-head: diffusion slightly more accurate on P(k), bispectrum, PDF; FM faster. Cite when choosing FM over diffusion.
- 2312.07534 Mudur, Cuesta-Lazaro, Finkbeiner 2023 and 2405.05255 Diffusion-HMC. Diffusion model as emulator and as field-level likelihood. If you go diffusion, the likelihood gives a field-level test between SR and HF.
- 2409.11401 Pandey et al. 2024, GOTHAM, and 2511.08438 Galactification 2025. Transformer point-cloud halo/galaxy generators from the CHARM group. Alternative stochastic halo generators; cite as baselines.
- 2311.17141 Cuesta-Lazaro, Mishra-Sharma 2023; 2409.02980 NeHOD 2024; 2607.19808 Moriwaki, Osato, Yoshida 2026. Point-cloud diffusion for halos/galaxies. Alternative representation to voxel counts.

GAN and deterministic lineage your current model descends from:
- 2010.06608 Li et al. 2020 (PNAS) and 2408.09051 Zhang et al. 2024 (SR IV, deterministic emulator). SR IV is the deterministic analogue of your L1 model; with shared ICs a deterministic map is well-posed at large scales, the question is the irreducible small-scale scatter.
- 2111.06393 Schaurecker et al. 2021. U-Net + cGAN halo SR with shared ICs. Closest GAN analogue; natural baseline.
- 2605.09004 Fremstad, Adamek, Mota 2026. Separate Universe SR emulator. GAN with injected noise reports about 10 percent small-scale power suppression and halo-abundance deficits, the same failure mode as your halo_count_deficit plots.
- 2111.02441 NECOLA, 2206.04594 Jamieson et al. 2022, 2312.09271 Doeser et al. 2023. Deterministic CNN correctors (COLA/LPT -> N-body).

## 5. Competing framings a referee will raise (tier C)

The nearest competitors are not other field correctors. They correct the
posterior instead of the simulator, and your cross-fidelity KL is the right
metric to compare against them.

- 2502.08416 Krouglova et al. 2025. Multifidelity SBI (MF-NPE): pretrain NPE on LF sims, fine-tune on few HF sims. The canonical baseline.
- 2505.21215 Saoulis et al. 2025 (MNRAS). Transfer learning MF-SBI in cosmology: DM-only -> CAMELS hydro, 8-15x fewer HF sims.
- 2606.23346 Saoulis et al. 2026. Field-level weak lensing with 60 simulations using MF-SBI.
- 2506.06087 Hikida, Bharti, Jeffrey, Briol 2025. Multilevel neural SBI (telescoping NPE loss); includes a CAMELS example.
- 2507.00514 Thiele, Bayer, Takeishi 2025. Multi-fidelity SBI via feature matching and distillation. Bayer is a CHARM co-author, so this is the framing closest to home.
- 2509.23385 Ruhlmann et al. 2025. Flow Matching Calibration for SBI under misspecification: flow matching used to correct the posterior rather than the field. Natural sibling to your method.
- 2405.08719 Wehenkel et al. 2024 (RoPE) and 2305.15871 Huang et al. 2023. Summaries-side or calibration-side fixes for misspecification.
- 2507.03086 Pierre et al. 2025. Drop inconsistent summary components and transport observations onto the training support (SimBIG).

Suggested framing: CHARM (2409.09124) is already tuned to give unbiased P(k) and
bispectrum posteriors, so a field corrector can only add variance at the summary
level. The residual mismatch lives at the field level, which is where your
corrector helps. Posterior-side MF-SBI needs HF sims at inference-model training
time; a field corrector produces an HF-like suite once, reusable for any
downstream estimator. That is the argument for the field route.

## 6. Evaluation and calibration (tier D)

- 2606.10023 Doeser, Jasche 2026. Posterior reliability of generative models for IC inference: matching means, marginals, and cross-correlation does not imply correct posterior geometry. The citation for why P(k) passes while PQMass fails.
- 2505.13620 Bayer et al. 2025. Field-level comparison of seven N-body codes: OOD tests plus field-level SBI trained on one code and tested on another; smoothing-scale sweep. Closest published version of your protocol.
- 2508.05744 Akhmetzhanova, Cuesta-Lazaro, Mishra-Sharma 2025. Scale-dependent normalizing flows for misspecification detection. Would localize in scale where SR fails; complements PQMass, which is scale-blind.
- 2302.03026 Lemos et al. 2023. TARP coverage test. Run it on the corrector's conditional samples treating HF as truth given LF; this quantifies under-dispersion directly.
- 2402.04355 Lemos et al. 2024. PQMass (already cited).
- 2512.13987 Rampal et al. 2025. Intercomparison of GAN, diffusion, and FM for precipitation downscaling: GANs systematically under-dispersive, diffusion and FM calibrated; all methods underestimate changes in extremes. Cleanest published evidence for the switch, and a warning to test cross-cosmology generalization.
- 2501.14822 Merizzi, Evangelista, Loukos 2025. Ensemble variance of DDIM sampling is set by the number of reverse steps; a post-hoc knob for matching box-to-box variance.
- 2604.00897 Delefosse et al. 2026. Re-coarsening consistency check (coarse-grained output should match input). For you, compare against coarse-grained Quijote, not CHARM, since LF is biased.
- 2312.15796 GenCast and 2309.15214 CorrDiff for the ensemble validation vocabulary: CRPS, spread-skill, rank histograms per scale.
- 1711.06077 Blau, Michaeli 2018. Perception-distortion tradeoff. The theorem that any distortion loss (L1, L2) necessarily worsens distributional fidelity; the formal reason your L1 term drove noise strength to zero. Palette (2111.05826) documents L1 reducing sample diversity empirically.
- 2209.01845 Cannon, Ward, Schmon 2022 and 2109.10360 Villaescusa-Navarro et al. 2021 for misspecification framing.

## 7. Count data (tier E)

Your discretization probe showed the generator reproduces clustered counts
directly, so a Poisson head adds shot noise on top. Continuous bridges with
rounding are standard (Bourdin 2408.00839 treats galaxy counts continuously).
Two count-native transports exist if sparse voxels turn out lossy:
- 2603.04730 Fishman et al. 2026. Count Bridges. Integer-valued bridge with closed-form conditionals, both endpoints observed count distributions.
- 2605.07746 Wei, Pearson 2026. Flow Matching for Count Data. Birth-death process FM between arbitrary count-distributed source and target. No spatial architecture provided; you would supply the 3D conv rate network.
- 2505.05082 ItDPDM and 2605.00360 Binomial flows: noise-to-data Poisson/binomial processes, not bridges; their Tweedie-style denoisers could be adapted.
- 2605.29016 Xia, Wise 2026 (21cm lightcones) reports that preprocessing (Yeo-Johnson plus amplitude compression) dominates stability for skewed voxel distributions; relevant to your log(1+n) choice.
- 2410.14171 Pandey et al. 2024. Heavy-tailed diffusion (Student-t kernels) if high-count voxel residuals are heavy-tailed.

## 8. Implications for the specific failures in the draft

- L1 drove noise strength to zero. Flow matching regresses a velocity, not a
  sample, so there is no per-sample reconstruction term at all. Stochasticity is
  by construction from gamma(t) z (Albergo) or an SDE sampler (DDBM). The
  perception-distortion theorem says no reweighting of L1 would have fixed this.
- Poisson head made things worse. Consistent with the literature: model the
  count field continuously and round, or use a count-native bridge.
- Theta conditioning hurt the summary posterior. Fotiadis, MFFM, and BaryonBridge
  all condition on the LF field only or on parameters that are known at
  deployment. Zeghal conditions on cosmology but in an unpaired setting; read it
  before deciding. Debeire et al. 2604.03459 add soft conservation constraints to
  Fotiadis's method; a total-halo-count or large-scale P(k) conservation term is
  the analogue for you and would address the 5-10 percent large-scale deficit
  without a cosmology-dependent P(k) loss.
- Patch seams. Cosmo-FOLD's overlap-latent scheme, Rouhiainen's outpainting, and
  Schiodt et al. 2508.13770 (patch-wise stochastic interpolants on turbulence)
  are the three precedents.

## 9. Minimal experiment plan implied by the literature

1. Freeze the current unconditioned GAN as the mean predictor m(LF).
2. Compute residuals r = HF - m(LF) on the training split, in log(1+n) space.
   Measure the residual power spectrum P_r(k).
3. Source distribution: x0 = m(LF) + eps, with eps Gaussian and per-k variance
   matched to P_r(k) (Chen et al. MFFM recipe), or a scalar sigma learned as in
   Fotiadis.
4. Train a 3D conv velocity field v(x_t, t, LF) with the linear interpolant
   x_t = (1-t) x0 + t HF. Condition on LF throughout (Augmented Bridge Matching).
   No theta. Same patching as now.
5. Sample with an ODE (deterministic given x0) and an SDE; compare spread.
6. Report: per-box transfer RMS, r(k), PQMass on the count histogram, spread-skill
   ratio per k-bin across 12+ draws, TARP on conditional samples, then the
   cross-fidelity KL at summary and field level.
7. Ablations: (a) bridge straight from LF without the GAN mean; (b) conditional
   FM from noise (formulation A) as the cosmology-standard baseline; (c) MF-NPE
   fine-tuning as the posterior-side baseline.

## 10. Gaps this work can claim

- No flow-matching or bridge model on 3D halo count fields.
- No generative corrector of an ML halo emulator (CHARM-like) at all.
- No 3D corrected-field evaluation by downstream SBI posterior bias; Zeghal does
  it in 2D lensing, Doeser and Jasche for IC inference.
- No paper measures posterior bias from training SBI on generated rather than
  simulated fields. Your cross-fidelity KL fills that.

## 11. Caveats

- IDs were verified against arXiv abstract pages, not full texts. Several 2026
  entries are recent preprints; check for revisions.
- The claim that CorrDiff's own residual ensemble is under-dispersive comes from
  secondary sources and Kim et al. 2606.30821; confirm in the CorrDiff appendix.
- The summary of Li et al. 2010.06608 in one search read it as non-GAN; it is a
  GAN with noise injection.
- CosmoFlow 2507.11842 (the paper that started this thread) is a
  representation-learning paper and does not belong in this list beyond a passing
  citation.
