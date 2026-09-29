"""Section 7 figures (paper/section7_flowgan.tex): the two figures only.
  1. pk_panel_box_flowgan.png : full-box P(k), transfer T(k)=P_X/P_HR (median + 16-84 band, RMS in legend),
     cross-coherence r(k) with HR, for LR, GAN (deployed, fiducial theta), Flow (Gaussian source) and Flow+GAN
     (40-epoch run, CRPS-best checkpoint). All 200 held-out TEST boxes (updated 2026-09-28; the first version used
     validation boxes 16..115 because Flow+GAN had no test fields then).
  2. crossfid_flowgan.png : cross-fidelity KL to HR, summary (P(k) NDE, nde_v2, per-seed mean +/- std over 8 seeds)
     and field (3D CNN, mean +/- std over HR-reference x replicate), numbers copied from the eval logs listed below.
  python -m flow_matching.plot_sec7            (source scripts/env_delta.sh first)
"""
import os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from flow_matching.eval_spread import PkBins, delta
from paths import PROCESSED as PROC

C = os.environ.get("SRS_CMASS_ROOT", "/work/nvme/bdne/vkshirsagar1/cmass-ili") + "/sr_fields"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
# name -> (source dir or "input", colour, linestyle); models take the first three categorical slots, LR is the grey control
SRC = {"LR (raw)": ("input", "#8a8984", "--"),
       "GAN": (f"{C}/Afix_fiducial", "#eb6834", "-"),
       "Flow": (f"{C}/fm_gauss_nodq", "#1baf7a", "-"),
       "Flow+GAN": (f"{C}/Flow+GAN_crps40", "#2a78d6", "-")}


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)


def pk_panel(out="figures_cmass/pk_panel_box_flowgan.png"):
    ids = sorted(int(d["idx"]) for d in np.load(f"{PROC}/test_list.npy", allow_pickle=True))
    pkb = PkBins(); ok = pkb.n_modes > 0; k = pkb.k[ok]
    P = {t: [] for t in SRC}; T = {t: [] for t in SRC}; R = {t: [] for t in SRC}; Ph_all = []
    for j, i in enumerate(ids):
        fh = pkb.fft(delta(np.load(f"{PROC}/{i:04d}_label.npy"))); Ph = pkb.auto(fh); Ph_all.append(Ph[ok])
        for t, (p, _, _) in SRC.items():
            x = np.load(f"{PROC}/{i:04d}_input.npy") if p == "input" else np.load(f"{p}/sr_{i}.npy")
            fx = pkb.fft(delta(x)); Px = pkb.auto(fx)
            P[t].append(Px[ok]); T[t].append((Px / Ph)[ok]); R[t].append((pkb.cross(fx, fh) / np.sqrt(Px * Ph))[ok])
        if (j + 1) % 25 == 0:
            print(f"  {j+1}/{len(ids)}", flush=True)
    np.savez("runs/patch_cmass/pk_panel_box_flowgan.npz", k=k, Ph=np.array(Ph_all), ids=np.array(ids),
             **{f"T_{t}": np.array(v) for t, v in T.items()}, **{f"r_{t}": np.array(v) for t, v in R.items()})

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    ax[0].plot(k, np.median(Ph_all, 0), color=INK, lw=2.2, label="HR (target)")
    for t, (_, c, ls) in SRC.items():
        Tt, Rt = np.array(T[t]), np.array(R[t])
        rms = np.sqrt(np.mean((np.median(Tt, 0) - 1) ** 2))
        ax[0].plot(k, np.median(P[t], 0), color=c, ls=ls, lw=2, label=t)
        ax[1].plot(k, np.median(Tt, 0), color=c, ls=ls, lw=2, label=f"{t}  (RMS of median $-1$: {100*rms:.1f}%)")
        if t in ("LR (raw)", "Flow+GAN"):
            ax[1].fill_between(k, np.percentile(Tt, 16, 0), np.percentile(Tt, 84, 0), color=c, alpha=0.15, lw=0)
        ax[2].plot(k, np.median(Rt, 0), color=c, ls=ls, lw=2, label=t)
        low = k < 0.1
        print(f"{t:>10s}: T(k<0.1) median {np.median(Tt[:, low]):.3f} | T(k>0.2) median {np.median(Tt[:, k > 0.2]):.3f} "
              f"| median-T RMS {rms:.4f} | r at k~0.125 {np.median(Rt[:, np.argmin(np.abs(k-0.125))]):.3f}")
    ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_ylabel(r"$P(k)$ [$(h^{-1}{\rm Mpc})^3$]")
    ax[0].set_title("Power spectrum (median over boxes)", fontsize=11, color=INK)
    ax[1].axhline(1, color=INK, lw=0.9); ax[1].set_xscale("log"); ax[1].set_ylim(0.75, 1.25)
    ax[1].set_ylabel(r"$T(k)=P_X/P_{\rm HR}$"); ax[1].set_title("Transfer to HR (median; 16-84% band for LR and Flow+GAN)", fontsize=11, color=INK)
    ax[2].set_xscale("log"); ax[2].set_ylim(-0.2, 1.02); ax[2].axhline(0, color=INK2, lw=0.6)
    ax[2].set_ylabel(r"$r(k)$ with HR"); ax[2].set_title("Cross-coherence with HR (median)", fontsize=11, color=INK)
    for a in ax:
        style(a); a.set_xlabel(r"$k$ [$h\,{\rm Mpc}^{-1}$]"); a.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(out, dpi=150); print("saved", out)


# Cross-fidelity KL to HR (lower is better). (mean, std); None = not run.
#   summary: logs/3248394_seedxfid.log (nde_v2, 8 seeds, 200 HR test boxes; Flow+GAN = tag FGcrps40);
#            floor = HR seed pairs (0.031).
#   field  : Flow+GAN logs/3248260_fieldev.log (Flow+GAN_crps40); LR and floor (pairwise HR retrains, 0.0193) same log;
#            GAN (deployed, fiducial) and Flow (Gaussian, patches) from the babel field evals (research_scratch_4 D-F19,
#            flow_matching/plot_crossfid_arms.py), same HR reference CNNs.
ARMS = ["LR (raw)", "GAN", "Flow", "Flow+GAN"]
SUMM = [(0.049, 0.010), (0.202, 0.038), (0.087, 0.027), (0.047, 0.008)]; SUMM_FLOOR = 0.031
FIELD = [(0.471, 0.782), (0.037, 0.014), (0.029, 0.008), (0.027, 0.011)]; FIELD_FLOOR = 0.019


def crossfid(out="figures_cmass/crossfid_flowgan.png"):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    for a, vals, floor, title in ((ax[0], SUMM, SUMM_FLOOR, r"Summary level: NDE on $\log P(k)$"),
                                  (ax[1], FIELD, FIELD_FLOOR, r"Field level: 3D CNN on the $128^3$ box")):
        x = np.arange(len(ARMS))
        for i, (m, e) in enumerate(vals):
            c = SRC[ARMS[i]][1]
            a.bar(i, m, width=0.62, color=c, edgecolor="white", lw=2)
            a.errorbar(i, m, yerr=[[min(e, 0.9 * m)], [e]], fmt="none", ecolor=INK2, capsize=4, lw=1.1)
            a.text(i + 0.33, m, f"{m:.3f}", ha="left", va="center", fontsize=9, color=INK)
        a.axhline(floor, ls="--", color=INK, lw=1, label=f"HR-to-HR floor ({floor:.3f})")
        a.legend(loc="upper right", fontsize=9, frameon=False)
        a.set_yscale("log"); a.set_ylim(0.012, 1.6); a.set_xticks(x); a.set_xticklabels(ARMS, fontsize=10)
        a.set_ylabel(r"KL$(q_{\rm HR}\,\|\,q_X)$ on HR test boxes"); a.set_title(title, fontsize=11, color=INK)
        style(a); a.grid(axis="x", visible=False)
    fig.tight_layout(); fig.savefig(out, dpi=150); print("saved", out)


if __name__ == "__main__":
    import sys
    crossfid()
    if "--bars-only" not in sys.argv:
        pk_panel()
