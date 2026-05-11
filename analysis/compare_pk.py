"""Diagnostic comparison of Pk(HR), Pk(LR), Pk(SR).

Loads spherically averaged Pk arrays from runs/baseline/pk/{hr,lr,sr_*}
and reports:
  - per-bin mean and stdev across simulations
  - SR/HR and LR/HR ratios
  - mean fractional Pk error vs HR (over a chosen k-range)

Saves a numpy summary and (if matplotlib is available) a PNG plot.
"""
import argparse
import glob
import os
import re
import numpy as np

_SET_RE = re.compile(r"set(\d+)")


def _load_pk_dir(d, prefix=""):
    """Returns dict {sid: pk_vec} and a 1D k array."""
    files = sorted(glob.glob(os.path.join(d, f"pk_{prefix}*.npz")))
    if not files:
        return {}, None
    out = {}
    k_ref = None
    for f in files:
        m = _SET_RE.search(os.path.basename(f))
        if not m:
            continue
        z = np.load(f)
        sid = int(m.group(1))
        out[sid] = z["pk"]
        if k_ref is None:
            k_ref = z["k"]
    return out, k_ref


def _summarize(pks, name):
    arr = np.stack(list(pks.values()))
    m, s = arr.mean(0), arr.std(0)
    print(f"  {name}: n={arr.shape[0]} sims, k-bins={arr.shape[1]}")
    return arr, m, s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pk-hr-dir", default="runs/baseline/pk/hr")
    p.add_argument("--pk-lr-dir", default="runs/baseline/pk/lr")
    p.add_argument("--pk-sr-dir", required=True,
                   help="e.g. runs/baseline/pk/sr_e17")
    p.add_argument("--output", default=None,
                   help="Output prefix for .npz/.png; defaults next to --pk-sr-dir")
    args = p.parse_args()

    if args.output is None:
        args.output = os.path.join(args.pk_sr_dir, "..", "compare_" + os.path.basename(args.pk_sr_dir))

    pk_hr, k_hr = _load_pk_dir(args.pk_hr_dir, prefix="quijote_")
    pk_lr, k_lr = _load_pk_dir(args.pk_lr_dir, prefix="quijotelike_")
    pk_sr, k_sr = _load_pk_dir(args.pk_sr_dir, prefix="set")
    if not pk_hr or not pk_sr:
        raise RuntimeError(f"No Pk found. HR={len(pk_hr)} LR={len(pk_lr)} SR={len(pk_sr)}")

    common = sorted(set(pk_hr) & set(pk_sr) & (set(pk_lr) if pk_lr else set(pk_hr)))
    # Filter out sims with any NaN in HR or SR Pk vectors (model occasionally
    # diverges on outlier inputs).
    bad = [s for s in common
           if np.isnan(pk_hr[s]).any()
           or np.isnan(pk_sr[s]).any()
           or (pk_lr and np.isnan(pk_lr[s]).any())]
    if bad:
        print(f"skipping {len(bad)} sims with NaN Pk: {bad[:8]}{'...' if len(bad) > 8 else ''}")
    common = [s for s in common if s not in set(bad)]
    print(f"common sims (clean): {len(common)}")

    pk_hr_v = np.stack([pk_hr[s] for s in common])
    pk_sr_v = np.stack([pk_sr[s] for s in common])
    pk_lr_v = np.stack([pk_lr[s] for s in common]) if pk_lr else None
    k = k_hr if k_hr is not None else k_sr

    valid = (k > 0) & (pk_hr_v.mean(0) > 0)

    print("\n=== mean Pk (over sims) ===")
    print("k:        ", np.round(k[valid][:8], 4))
    print("Pk_HR:    ", np.round(pk_hr_v.mean(0)[valid][:8], 2))
    print("Pk_SR:    ", np.round(pk_sr_v.mean(0)[valid][:8], 2))
    if pk_lr_v is not None:
        print("Pk_LR:    ", np.round(pk_lr_v.mean(0)[valid][:8], 2))

    print("\n=== ratios (mean over sims) ===")
    ratio_sr = pk_sr_v[:, valid].mean(0) / pk_hr_v[:, valid].mean(0)
    print("SR/HR:    ", np.round(ratio_sr, 3))
    if pk_lr_v is not None:
        ratio_lr = pk_lr_v[:, valid].mean(0) / pk_hr_v[:, valid].mean(0)
        print("LR/HR:    ", np.round(ratio_lr, 3))

    # fractional error vs HR per sim, averaged over a k-range
    eps = 1e-12
    per_sim_err = np.abs(pk_sr_v - pk_hr_v) / np.maximum(pk_hr_v, eps)
    mean_err_band = per_sim_err[:, valid].mean(1)
    print(f"\nmean |Pk_SR - Pk_HR| / Pk_HR (per sim): "
          f"median={np.median(mean_err_band):.3f}, mean={mean_err_band.mean():.3f}, "
          f"p95={np.percentile(mean_err_band, 95):.3f}")
    if pk_lr_v is not None:
        per_sim_lr = np.abs(pk_lr_v - pk_hr_v) / np.maximum(pk_hr_v, eps)
        print(f"mean |Pk_LR - Pk_HR| / Pk_HR (per sim): "
              f"median={np.median(per_sim_lr[:, valid].mean(1)):.3f}")

    np.savez(args.output + ".npz",
             k=k, pk_hr=pk_hr_v, pk_sr=pk_sr_v,
             pk_lr=pk_lr_v if pk_lr_v is not None else np.array([]),
             common_sids=np.array(common))
    print(f"\nsaved {args.output}.npz")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8), sharex=True,
                                        gridspec_kw={"height_ratios": [3, 2]})
        ax1.loglog(k[valid], pk_hr_v.mean(0)[valid], label="Pk(HR=quijote)", lw=2)
        if pk_lr_v is not None:
            ax1.loglog(k[valid], pk_lr_v.mean(0)[valid], label="Pk(LR=quijotelike)", lw=1, ls="--")
        ax1.loglog(k[valid], pk_sr_v.mean(0)[valid], label="Pk(SR=GAN output)", lw=1.5, ls="-.")
        ax1.set_ylabel(r"$P(k)$ [(Mpc/h)$^3$]")
        ax1.legend()
        ax1.grid(True, which="both", alpha=0.3)
        ax2.semilogx(k[valid], ratio_sr, label="SR/HR")
        if pk_lr_v is not None:
            ax2.semilogx(k[valid], ratio_lr, label="LR/HR", ls="--")
        ax2.axhline(1.0, color="k", lw=0.5)
        ax2.set_ylim(0.0, 2.0)
        ax2.set_xlabel(r"$k$ [h/Mpc]")
        ax2.set_ylabel(r"$P/P_{HR}$")
        ax2.legend()
        ax2.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.output + ".png", dpi=120)
        print(f"saved {args.output}.png")
    except ImportError:
        print("matplotlib not available; skipped PNG plot")


if __name__ == "__main__":
    main()
