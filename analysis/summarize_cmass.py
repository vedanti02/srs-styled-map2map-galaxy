"""Final CMASS comparison: Arm A (naive) vs Arm B (overlap) vs LR baseline, all vs HR.
Reads runs/patch_cmass/{seam_<tag>.npz, metrics_<tag>.npz, metrics_LRbase.npz} and writes
SUMMARY.md + comparison figures (Pk ratio, per-position seam profile, KL bars)."""
import os
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

R = "runs/patch_cmass"
PARAMS = ["Om", "Ob", "h", "ns", "s8"]
ARMS = [("cmassA_naive", "Arm A · naive (2.1)"), ("cmassB_overlap", "Arm B · overlap (2.2)")]


def _load(path):
    return np.load(path, allow_pickle=True) if os.path.exists(path) else None


def main():
    seam = {tag: _load(f"{R}/seam_{tag}.npz") for tag, _ in ARMS}
    metr = {tag: _load(f"{R}/metrics_{tag}.npz") for tag, _ in ARMS}
    lr = _load(f"{R}/metrics_LRbase.npz")

    L = ["# CMASS count-field patch experiment — summary\n",
         "Eulerian halo-count fields (no Lagrangian). 128³ box, 2×2×2 of 64³ patches, "
         "log1p model space, GAN. Mentor 2.1 (naive) vs 2.2 (boundary-masked overlap), test split.\n"]

    # ---- seam table ----
    L.append("\n## Seam artifacts & fidelity (vs HR count field)\n")
    L.append("| run | seam-dist ratio | x=64 internal-seam excess | x=0 edge | stitched Pk RMS |")
    L.append("|---|---:|---:|---:|---:|")
    for tag, name in ARMS:
        s = seam[tag]
        if s is None: L.append(f"| {name} | (missing) | | | |"); continue
        L.append(f"| {name} | {float(s['seam_ratio']):.4f} | **{float(s['x64_excess']):.4f}** | "
                 f"{float(s['x0_excess']):.4f} | {float(s['pk_rms']):.4f} |")

    # ---- KL table ----
    L.append("\n## Posterior agreement KL(q_HR‖q_X), mean over test sims (lower=better)\n")
    cols = [("LR baseline", lr)] + [(name, metr[tag]) for tag, name in ARMS]
    L.append("| param | " + " | ".join(c[0] for c in cols) + " |")
    L.append("|---|" + "|".join(["---:"] * len(cols)) + "|")
    klmean = {}
    for c, (name, z) in enumerate(cols):
        if z is not None:
            klmean[name] = z["kl_hr_to_sr"].mean(0)
    for i, pn in enumerate(PARAMS):
        row = [pn] + [f"{klmean[name][i]:.4f}" if name in klmean else "—" for name, _ in cols]
        L.append("| " + " | ".join(row) + " |")
    meanrow = ["**mean**"] + [f"**{klmean[name].mean():.4f}**" if name in klmean else "—" for name, _ in cols]
    L.append("| " + " | ".join(meanrow) + " |")

    # ---- bias table (recovery of true theta) ----
    L.append("\n## Parameter recovery |μ−θ_true|, mean over test sims (lower=better)\n")
    L.append("| param | HR | " + " | ".join(name for _, name in ARMS) + " |")
    L.append("|---|---:|" + "|".join(["---:"] * len(ARMS)) + "|")
    for i, pn in enumerate(PARAMS):
        ref = next((m for m in metr.values() if m is not None), None)
        hr_b = np.abs(ref["mu_hr"] - ref["theta_true"]).mean(0)[i] if ref is not None else None
        cells = [pn, f"{hr_b:.4f}" if hr_b is not None else "—"]
        for tag, _ in ARMS:
            z = metr[tag]
            cells.append(f"{np.abs(z['mu_sr']-z['theta_true']).mean(0)[i]:.4f}" if z is not None else "—")
        L.append("| " + " | ".join(cells) + " |")

    os.makedirs(R, exist_ok=True)
    with open(f"{R}/SUMMARY.md", "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))

    # ---- Fig 1: Pk ratio SR/HR ----
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for tag, name in ARMS:
        s = seam[tag]
        if s is None: continue
        k = s["k"]; ratio = 10 ** (s["pk_sr"] - s["pk_hr"])
        mr = ratio.mean(0); m = s["pk_hr"].mean(0) > -10
        ax.plot(k[m], mr[m], lw=1.8, label=name)
    ax.axhline(1.0, color="grey", ls=":")
    ax.set_xscale("log"); ax.set_xlabel("k [h/Mpc]"); ax.set_ylabel("P_SR(k) / P_HR(k)")
    ax.set_ylim(0.8, 1.2); ax.legend(); ax.set_title("Stitched Pk ratio vs HR (test sims)")
    fig.tight_layout(); fig.savefig(f"{R}/pkratio_cmass.png", dpi=130); plt.close(fig)

    # ---- Fig 1b: ABSOLUTE Pk overlay — HR vs SR-A vs SR-B vs LR (the SR-vs-HR comparison) ----
    refseam = next((s for s in seam.values() if s is not None), None)
    if refseam is not None:
        k = refseam["k"]; m = refseam["pk_hr"].mean(0) > -10
        fig, (a0, a1) = plt.subplots(1, 2, figsize=(13, 5))
        a0.loglog(k[m], 10 ** refseam["pk_hr"].mean(0)[m], "k-", lw=2.2, label="HR (N-body)")
        for tag, name in ARMS:
            s = seam[tag]
            if s is not None:
                a0.loglog(k[m], 10 ** s["pk_sr"].mean(0)[m], lw=1.6, label=name)
        # LR baseline Pk from pk_LRbase test files
        import glob
        lrpks = [np.load(f)["pk"] for f in sorted(glob.glob(f"{R}/pk_LRbase/pk_set*_transformed.npz"))[:200]]
        if lrpks:
            lr_mean = np.log10(np.maximum(np.array(lrpks).mean(0), 1e-12))
            a0.loglog(k[m], 10 ** lr_mean[m], "r:", lw=1.6, label="LR (FastPM)")
        a0.set_xlabel("k [h/Mpc]"); a0.set_ylabel("P(k) [(Mpc/h)³]"); a0.legend()
        a0.set_title("Absolute P(k): SR vs HR (mean over test sims)")
        # right: ratio zoom
        for tag, name in ARMS:
            s = seam[tag]
            if s is not None:
                a1.plot(k[m], (10 ** (s["pk_sr"] - s["pk_hr"])).mean(0)[m], lw=1.8, label=name)
        a1.axhline(1.0, color="grey", ls=":"); a1.set_xscale("log"); a1.set_ylim(0.85, 1.15)
        a1.set_xlabel("k [h/Mpc]"); a1.set_ylabel("P_SR/P_HR"); a1.legend(); a1.set_title("Pk ratio vs HR")
        fig.tight_layout(); fig.savefig(f"{R}/pk_overlay_cmass.png", dpi=130); plt.close(fig)

    # ---- Fig 2: per-position seam profile A vs B ----
    fig, ax = plt.subplots(figsize=(9, 5))
    for tag, name in ARMS:
        s = seam[tag]
        if s is None: continue
        ax.plot(range(len(s["pos_profile"])), s["pos_profile"], lw=1.5, label=name)
    for x in (0, 64): ax.axvline(x, color="grey", ls=":", lw=0.9)
    ax.set_xlabel("grid position"); ax.set_ylabel("count MAE"); ax.legend()
    ax.set_title("Per-position error — internal seam at x=64")
    fig.tight_layout(); fig.savefig(f"{R}/posprofile_cmass.png", dpi=130); plt.close(fig)

    # ---- Fig 2b: combined SR-vs-HR projected count maps (HR | SR-A | SR-B) ----
    sA, sB = seam.get("cmassA_naive"), seam.get("cmassB_overlap")
    if sA is not None and "slice_hr2d" in sA and sA["slice_hr2d"].ndim == 2:
        hr2d = sA["slice_hr2d"]; vmax = np.percentile(hr2d, 99.5)
        panels = [("HR (N-body)", hr2d, "viridis", 0, vmax)]
        if sA is not None: panels.append(("SR-A naive", sA["slice_sr2d"], "viridis", 0, vmax))
        if sB is not None and "slice_sr2d" in sB:
            panels.append(("SR-B overlap", sB["slice_sr2d"], "viridis", 0, vmax))
            panels.append(("SR-B − HR", sB["slice_sr2d"] - hr2d, "RdBu_r", -vmax * 0.3, vmax * 0.3))
        fig, ax = plt.subplots(1, len(panels), figsize=(4.5 * len(panels), 4.6))
        for a, (t, img, cm, lo, hi) in zip(np.atleast_1d(ax), panels):
            a.imshow(img, origin="lower", cmap=cm, vmin=lo, vmax=hi); a.set_title(t)
            a.axvline(64, color="w", ls=":", lw=0.7); a.axhline(64, color="w", ls=":", lw=0.7)
        fig.suptitle(f"SR vs HR projected counts (set{int(sA['slice_sid'])}, ∑z)")
        fig.tight_layout(); fig.savefig(f"{R}/srhr_slice_cmass.png", dpi=130); plt.close(fig)

    # ---- Fig 3: KL bars ----
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(PARAMS)); w = 0.8 / max(len(klmean), 1)
    for j, (name, _) in enumerate(cols):
        if name not in klmean: continue
        ax.bar(x + j * w, klmean[name], w, label=name)
    ax.set_xticks(x + w); ax.set_xticklabels(PARAMS); ax.set_ylabel("KL(q_HR‖q_X)")
    ax.legend(); ax.set_title("Posterior disagreement vs HR (lower=better)")
    fig.tight_layout(); fig.savefig(f"{R}/kl_bars_cmass.png", dpi=130); plt.close(fig)

    print(f"\nwrote {R}/SUMMARY.md + pkratio_cmass.png, posprofile_cmass.png, kl_bars_cmass.png")


if __name__ == "__main__":
    main()
