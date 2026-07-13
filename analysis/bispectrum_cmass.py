"""Bispectrum of CMASS halo-count overdensity fields — a HELD-OUT statistic
(the training loss contains P(k) but never the bispectrum; mentor request).

Estimator: standard FFT shell method. With delta_d = fftn(delta) (unnormalized,
matching analysis/power_spectrum.py where P = L^3 |delta_d|^2 / N^6):

    F_i(x) = ifftn(mask_i * delta_d)      (field filtered to k-shell i)
    T_i(x) = ifftn(mask_i)                (shell triangle-counting field)
    B(k1,k2,k3) = L^6/N^9 * sum_x Re(F1 F2 F3) / sum_x Re(T1 T2 T3)

Configurations: equilateral B(k,k,k) over log-spaced shells, plus one squeezed
set B(k, k, k_min-bin).

Inputs, either mode:
  --mode raw : read HR/LR directly from processed/{idx}_label.npy / _input.npy
               (CPU-only, no model needed).
  --mode dirs: read set{sid}_transformed.npy cubes from one or more label=dir
               pairs (e.g. SR output dirs from infer_stitch_cmass).

Output: runs/patch_cmass/bispec_<tag>.npz + figures_cmass/bispectrum_<tag>.png
"""
import argparse
import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROCESSED = "/data/group_data/universedata/cmass-ili/processed"


def counts_to_delta(counts):
    c = np.asarray(counts, dtype=np.float64)
    if c.ndim == 4:
        c = c[0]
    return c / max(c.mean(), 1e-6) - 1.0


def shell_masks(N, lbox, n_bins):
    kx = np.fft.fftfreq(N, d=lbox / N) * 2 * np.pi
    kgrid = np.sqrt(kx[:, None, None]**2 + kx[None, :, None]**2 + kx[None, None, :]**2)
    k_min, k_nyq = 2 * np.pi / lbox, np.pi * N / lbox
    edges = np.logspace(np.log10(k_min * 1.01), np.log10(k_nyq), n_bins + 1)
    masks = [(kgrid >= lo) & (kgrid < hi) for lo, hi in zip(edges[:-1], edges[1:])]
    k_cen = np.sqrt(edges[:-1] * edges[1:])
    return masks, k_cen


class BispecEstimator:
    """Precomputes shell fields T_i once; per-cube computes F_i and triangle sums."""

    def __init__(self, N, lbox, n_bins=10):
        self.N, self.lbox = N, lbox
        self.masks, self.k = shell_masks(N, lbox, n_bins)
        self.T = [np.fft.ifftn(m.astype(np.float64)).real for m in self.masks]
        self.norm = lbox**6 / N**9
        # triangle counts (denominators), config-independent of the data
        self.tri_eq = np.array([ (t**3).sum() for t in self.T ])
        self.tri_sq = np.array([ (t * t * self.T[0]).sum() for t in self.T ])

    def __call__(self, delta):
        dk = np.fft.fftn(delta)
        F = [np.fft.ifftn(m * dk).real for m in self.masks]
        b_eq = np.array([ (f**3).sum() for f in F ]) / np.maximum(self.tri_eq, 1e-300)
        b_sq = np.array([ (f * f * F[0]).sum() for f in F ]) / np.maximum(self.tri_sq, 1e-300)
        return self.norm * b_eq, self.norm * b_sq


def load_cubes(mode, spec, sids, max_sims):
    """Yields (sid, cube) for one labeled source.

    spec 'label'/'input' always reads the raw processed cubes; any other spec is
    a directory of set{sid}_transformed.npy — so raw and model-output sources
    can be mixed in one invocation regardless of --mode."""
    n = 0
    raw = spec in ("label", "input")
    for sid in sids:
        if max_sims and n >= max_sims:
            return
        if raw:
            path = os.path.join(PROCESSED, f"{sid:04d}_{spec}.npy")
        else:
            path = os.path.join(spec, f"set{sid}_transformed.npy")
        if not os.path.exists(path):
            continue
        yield sid, np.load(path).astype(np.float32)
        n += 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["raw", "dirs"], default="raw")
    p.add_argument("--sources", nargs="+", default=["HR=label", "LR=input"],
                   help="label=spec pairs. raw mode: spec in {label,input}; "
                        "dirs mode: spec is a directory of set*_transformed.npy")
    p.add_argument("--split-sids", default="runs/patch_cmass/split_sids.npz")
    p.add_argument("--which-sids", default="test_sids")
    p.add_argument("--max-sims", type=int, default=32)
    p.add_argument("--n-bins", type=int, default=10)
    p.add_argument("--lbox", type=float, default=1000.0)
    p.add_argument("--tag", default="hr_lr")
    args = p.parse_args()

    sids = np.load(args.split_sids)[args.which_sids].tolist()
    est = None
    results = {}
    for pair in args.sources:
        label, spec = pair.split("=", 1)
        eqs, sqs, used = [], [], []
        for sid, cube in load_cubes(args.mode, spec, sids, args.max_sims):
            if est is None:
                est = BispecEstimator(cube.shape[-1], args.lbox, args.n_bins)
            b_eq, b_sq = est(counts_to_delta(cube))
            eqs.append(b_eq); sqs.append(b_sq); used.append(sid)
            if len(used) % 8 == 0:
                print(f"  {label}: {len(used)} boxes", flush=True)
        results[label] = (np.array(eqs), np.array(sqs), np.array(used))
        print(f"{label}: {len(used)} boxes done")

    os.makedirs("runs/patch_cmass", exist_ok=True)
    save = {"k": est.k}
    for label, (eqs, sqs, used) in results.items():
        save[f"beq_{label}"] = eqs; save[f"bsq_{label}"] = sqs; save[f"sids_{label}"] = used
    np.savez(f"runs/patch_cmass/bispec_{args.tag}.npz", **save)

    # ---- figure: equilateral + squeezed, each with ratio-to-HR panel ----
    ref = "HR" if "HR" in results else list(results)[0]
    colors = {"HR": "k", "SR": "C0", "SR_A": "C0", "SR_B": "C2", "LR": "C3"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    for col, (name, idx) in enumerate([("equilateral B(k,k,k)", 0),
                                       (r"squeezed B(k,k,$k_{\min}$)", 1)]):
        ax, axr = axes[0, col], axes[1, col]
        ref_med = np.nanmedian(results[ref][idx], 0)
        for label, r in results.items():
            arr = r[idx]
            med = np.nanmedian(arr, 0)
            lo, hi = np.nanpercentile(arr, 16, 0), np.nanpercentile(arr, 84, 0)
            c = colors.get(label, None)
            ax.fill_between(est.k, lo, hi, alpha=0.15, color=c)
            ax.plot(est.k, med, lw=2, color=c, ls="--" if label == ref else "-", label=label)
            if label != ref:
                axr.plot(est.k, med / np.where(np.abs(ref_med) > 0, ref_med, np.nan),
                         lw=2, color=c, label=label)
        ax.set_xscale("log"); ax.set_yscale("symlog")
        ax.set_title(name); ax.legend(); ax.grid(alpha=0.3, which="both")
        axr.axhline(1.0, color="k", ls="--", lw=1)
        axr.set_xscale("log"); axr.set_ylim(0, 2)
        axr.set_xlabel(r"$k\ [h/{\rm Mpc}]$"); axr.set_ylabel(f"ratio to {ref}")
        axr.grid(alpha=0.3, which="both")
    axes[0, 0].set_ylabel(r"$B(k)\ [({\rm Mpc}/h)^6]$")
    plt.tight_layout()
    out = f"figures_cmass/bispectrum_{args.tag}.png"
    plt.savefig(out, dpi=130)
    print("saved", out)

    # console summary: mid-k ratio to reference
    mid = slice(len(est.k) // 3, 2 * len(est.k) // 3)
    for label, r in results.items():
        if label == ref:
            continue
        ratio = np.nanmedian(r[0], 0)[mid] / np.nanmedian(results[ref][0], 0)[mid]
        print(f"{label}: equilateral mid-k B ratio to {ref} = {np.nanmean(ratio):.3f}")


if __name__ == "__main__":
    main()
