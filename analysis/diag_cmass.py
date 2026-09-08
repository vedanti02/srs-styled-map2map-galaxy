"""CMASS diagnostic plots (count fields), matching analysis/plot_diagnostic.py format,
at BOTH the full 128^3 box level and the individual 64^3 patch level:

  1) Pk panel: P(k) median+band (Truth=HR, Pred=SR), transfer P_SR/P_HR, cross-power r(k).
  2) Projection panel for one example: LR input / HR truth / SR pred / residual.

Density is the count overdensity delta = n/nbar - 1 (no displacement/CIC). SR is produced
by running the trained generator per patch (periodic-wrap halo for overlap), cropping, and
stitching, exactly as infer_stitch_cmass does.
"""
import argparse, os
import numpy as np
import torch

from map2map.models.styled_srsgan import G_correct
from data.patch_dataset_cmass import (
    PatchPairDatasetCmass, extract_patch, stitch_patches, crop_interior,
    to_counts, counts_to_delta, PATCH, N_PATCHES, N_FULL,
)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from analysis.plot_diagnostic import cross_pk


def build_noise_list(grid, num_blocks, seed, device):
    rng = np.random.default_rng(seed)
    return [torch.from_numpy(rng.standard_normal((1, grid, grid, grid)).astype(np.float32)).to(device)
            for _ in range(2 * num_blocks)]


def make_pk_panel(results, k_nyq, outpath, title="Power Spectrum"):
    """P(k) for HR/SR/LR (median+band), transfer X/HR, cross-power r(k) for SR and LR.
    results = list of (k, P_HR, P_SR, P_LR, r_SR, r_LR)."""
    Ks  = np.stack([r[0] for r in results])
    Phr = np.stack([r[1] for r in results]); Psr = np.stack([r[2] for r in results])
    Plr = np.stack([r[3] for r in results]); Rsr = np.stack([r[4] for r in results])
    Rlr = np.stack([r[5] for r in results])
    k = np.nanmedian(Ks, axis=0); v = k > 0; k = k[v]
    Phr, Psr, Plr, Rsr, Rlr = Phr[:, v], Psr[:, v], Plr[:, v], Rsr[:, v], Rlr[:, v]
    def band(A): return np.nanmedian(A, 0), np.nanpercentile(A, 16, 0), np.nanpercentile(A, 84, 0)
    fig, ax = plt.subplots(3, 1, figsize=(7.5, 11), sharex=True)
    # --- top: P(k) HR, SR, LR ---
    hr_med, _, _ = band(Phr); sr_med, sr_lo, sr_hi = band(Psr); lr_med, lr_lo, lr_hi = band(Plr)
    ax[0].plot(k, hr_med, color="red", lw=2, label="HR")
    ax[0].fill_between(k, sr_lo, sr_hi, color="C0", alpha=0.25); ax[0].plot(k, sr_med, color="C0", lw=2, label="SR median")
    ax[0].fill_between(k, lr_lo, lr_hi, color="C3", alpha=0.12); ax[0].plot(k, lr_med, color="C3", lw=1.8, ls=":", label="LR median")
    ax[0].axvline(k_nyq, color="gray", lw=1.5, label="Nyquist (approx)")
    ax[0].set_yscale("log"); ax[0].set_xscale("log"); ax[0].set_ylabel("P(k)"); ax[0].set_title(title)
    ax[0].legend(fontsize=10); ax[0].grid(alpha=0.3, which="both")
    # --- middle: transfer X/HR (SR and LR), LINEAR zoomed axis with BOTH bands ---
    # The old log 1e-2..1e2 axis compressed a ~12% per-patch error into an invisible sliver
    # at y=1, and the LR band was discarded, so LR and SR looked identical (D-F23).
    tsr = Psr / np.maximum(Phr, 1e-30); tlr = Plr / np.maximum(Phr, 1e-30)
    ts_med, ts_lo, ts_hi = band(tsr); tl_med, tl_lo, tl_hi = band(tlr)
    rms_sr = np.sqrt(np.nanmean((tsr - 1.0) ** 2)); rms_lr = np.sqrt(np.nanmean((tlr - 1.0) ** 2))
    ax[1].fill_between(k, tl_lo, tl_hi, color="C3", alpha=0.15)
    ax[1].plot(k, tl_med, color="C3", lw=1.8, ls=":", label=f"LR/HR (RMS err vs HR {rms_lr*100:.1f}%)")
    ax[1].fill_between(k, ts_lo, ts_hi, color="C0", alpha=0.25)
    ax[1].plot(k, ts_med, color="C0", lw=2, label=f"SR/HR (RMS err vs HR {rms_sr*100:.1f}%)")
    ax[1].axhline(1.0, color="k", lw=1, ls="--"); ax[1].axvline(k_nyq, color="gray", lw=1.5)
    ax[1].set_xscale("log"); ax[1].set_ylim(0.6, 1.5)
    ax[1].set_ylabel("Transfer (X/HR), linear"); ax[1].legend(fontsize=9); ax[1].grid(alpha=0.3, which="both")
    # --- bottom: cross-coherence r(k) for SR and LR ---
    rs_med, rs_lo, rs_hi = band(Rsr); rl_med, _, _ = band(Rlr)
    ax[2].fill_between(k, rs_lo, rs_hi, color="C0", alpha=0.25); ax[2].plot(k, rs_med, color="C0", lw=2, label="SR x HR")
    ax[2].plot(k, rl_med, color="C3", lw=1.8, ls=":", label="LR x HR")
    ax[2].axhline(1.0, color="k", lw=1, ls="--"); ax[2].axhline(0.0, color="k", lw=1, ls="--")
    ax[2].axvline(k_nyq, color="gray", lw=1.5); ax[2].set_xscale("log"); ax[2].set_ylim(-0.25, 1.1)
    ax[2].set_xlabel("k [1/Box]"); ax[2].set_ylabel("Cross-power (X x HR)"); ax[2].legend(fontsize=9); ax[2].grid(alpha=0.3, which="both")
    plt.tight_layout(); plt.savefig(outpath, dpi=120); plt.close(fig); print(f"saved {outpath}")


def make_projection_panel(delta_lr, delta_hr, delta_sr, outpath):
    """4 panels: LR input / HR truth / SR prediction / residual (HR - SR)/sqrt(SR)."""
    def proj(d): return np.log1p(np.maximum(d, -0.999)).sum(axis=2)
    p_lr, p_hr, p_sr = proj(delta_lr), proj(delta_hr), proj(delta_sr)
    safe = np.maximum(p_sr - p_sr.min() + 1e-3, 1e-3)
    resid = (p_hr - p_sr) / np.sqrt(safe)
    vmin = min(p_lr.min(), p_hr.min(), p_sr.min())
    vmax = max(np.percentile(p_lr, 99.5), np.percentile(p_hr, 99.5), np.percentile(p_sr, 99.5))
    fig, ax = plt.subplots(1, 4, figsize=(20, 5.5))
    titles = ["LR input", "HR truth", "SR prediction", r"Residual (HR - SR)/$\sqrt{\rm SR}$"]
    for a, img, t in zip(ax[:3], [p_lr, p_hr, p_sr], titles[:3]):
        a.imshow(img, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
        a.set_title(t, fontsize=16); a.set_xticks([]); a.set_yticks([])
    rabs = max(abs(np.percentile(resid, 1)), abs(np.percentile(resid, 99)))
    im = ax[3].imshow(resid, origin="lower", cmap="bwr", vmin=-rabs, vmax=rabs)
    ax[3].set_title(titles[3], fontsize=16); ax[3].set_xticks([]); ax[3].set_yticks([])
    plt.colorbar(im, ax=ax[3], fraction=0.046, pad=0.04)
    plt.tight_layout(); plt.savefig(outpath, dpi=120); plt.close(fig); print(f"saved {outpath}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--mode", default="naive", choices=["naive", "overlap"])
    p.add_argument("--theta-mode", default="fiducial", choices=["true", "fiducial"],
                   help="fiducial = prior-mean cosmology for all boxes (deployment-faithful); "
                        "true = per-box cosmology (oracle)")
    p.add_argument("--tag", default="cmassA")
    p.add_argument("--out-dir", default="figures_cmass")
    p.add_argument("--n-sims", type=int, default=12)
    p.add_argument("--n-bins", type=int, default=40)
    p.add_argument("--lbox", type=float, default=1000.0)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    FIDUCIAL = np.array([0.30, 0.05, 0.70, 1.00, 0.80], np.float32)  # prior mean (Om,Ob,h,ns,s8)
    print(f"theta-mode={args.theta_mode}", flush=True)

    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    saved = ck.get("args", {}) or {}
    cb, nb = saved.get("chan_base_g", 128), saved.get("num_blocks", 4)
    transform = saved.get("transform", "log1p")
    pad = 0 if args.mode == "naive" else (saved.get("pad", 8))
    G = G_correct(1, 1, 5, chan_base=cb, num_blocks=nb).to(dev)
    G.load_state_dict(ck["model"]); G.eval()
    nl = build_noise_list(PATCH + 2 * pad, nb, 0, dev)

    ds = PatchPairDatasetCmass(split="test", pad=pad, transform=transform)  # model space LR
    ds_raw = PatchPairDatasetCmass(split="test", pad=0, normalize_inputs=False)  # raw counts
    ids = ds.ids[:args.n_sims]

    lbox_box, lbox_patch = args.lbox, args.lbox * PATCH / N_FULL  # 1000, 500
    k_nyq_box = np.pi * N_FULL / lbox_box
    k_nyq_patch = np.pi * PATCH / lbox_patch

    box_results, patch_results = [], []
    box_cache = patch_cache = None
    with torch.no_grad():
        for i, idx in enumerate(ids):
            lr_m, _ = ds.load_boxes(idx)                       # model space LR box
            lr_c, hr_c = ds_raw.load_boxes(idx)               # raw count boxes
            patches = np.stack([extract_patch(lr_m, q, pad) for q in range(N_PATCHES)])
            xb = torch.from_numpy(patches).to(dev)
            th_vec = ds.theta[idx] if args.theta_mode == "true" else FIDUCIAL
            th = torch.from_numpy(np.asarray(th_vec, np.float32)).unsqueeze(0).expand(N_PATCHES, -1).to(dev)
            fake = crop_interior(G(xb, th, nl), pad).cpu().numpy()        # (8,1,64,64,64) model space
            sr_patches = to_counts(fake, transform, ds.scale)            # counts per patch
            sr_c = stitch_patches(sr_patches)                            # (1,128,128,128) counts

            # ---- box level ----
            d_hr = counts_to_delta(hr_c); d_sr = counts_to_delta(sr_c); d_lr = counts_to_delta(lr_c)
            k, Pa, Pb, Pab, m = cross_pk(d_hr[0], d_sr[0], lbox_box, n_bins=args.n_bins)
            r = np.zeros_like(k); r[m] = Pab[m] / np.sqrt(np.maximum(Pa * Pb, 1e-60))[m]
            _, _, Plr, Pab_l, ml = cross_pk(d_hr[0], d_lr[0], lbox_box, n_bins=args.n_bins)
            r_l = np.zeros_like(k); r_l[ml] = Pab_l[ml] / np.sqrt(np.maximum(Pa * Plr, 1e-60))[ml]
            box_results.append((k, Pa, Pb, Plr, r, r_l))
            if i == 0:
                box_cache = (d_lr[0], d_hr[0], d_sr[0], idx)

            # ---- patch level (each of the 8 cores is a 64^3 sub-box) ----
            hr_pp = [extract_patch(hr_c, q, 0)[0] for q in range(N_PATCHES)]
            lr_pp = [extract_patch(lr_c, q, 0)[0] for q in range(N_PATCHES)]
            for q in range(N_PATCHES):
                dh = counts_to_delta(hr_pp[q]); dsr = counts_to_delta(sr_patches[q][0]); dl = counts_to_delta(lr_pp[q])
                kk, Qa, Qb, Qab, mm = cross_pk(dh, dsr, lbox_patch, n_bins=args.n_bins)
                rr = np.zeros_like(kk); rr[mm] = Qab[mm] / np.sqrt(np.maximum(Qa * Qb, 1e-60))[mm]
                _, _, Qlr, Qab_l, mml = cross_pk(dh, dl, lbox_patch, n_bins=args.n_bins)
                rr_l = np.zeros_like(kk); rr_l[mml] = Qab_l[mml] / np.sqrt(np.maximum(Qa * Qlr, 1e-60))[mml]
                patch_results.append((kk, Qa, Qb, Qlr, rr, rr_l))
                if i == 0 and q == 0:
                    patch_cache = (dl, dh, dsr, idx, q)
            print(f"  sim {i+1}/{len(ids)} (set{idx}) done", flush=True)

    t = args.tag
    make_pk_panel(box_results, k_nyq_box, f"{args.out_dir}/pk_panel_box_{t}.png",
                  title=f"Power Spectrum, full box ({t})")
    make_pk_panel(patch_results, k_nyq_patch, f"{args.out_dir}/pk_panel_patch_{t}.png",
                  title=f"Power Spectrum, per patch 64^3 ({t})")
    if box_cache:
        dl, dh, dsr, idx = box_cache
        make_projection_panel(dl, dh, dsr, f"{args.out_dir}/projection_box_{t}_set{idx}.png")
    if patch_cache:
        dl, dh, dsr, idx, q = patch_cache
        make_projection_panel(dl, dh, dsr, f"{args.out_dir}/projection_patch_{t}_set{idx}_p{q}.png")
    print("diag_cmass done")


if __name__ == "__main__":
    main()
