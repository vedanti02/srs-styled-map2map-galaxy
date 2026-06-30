"""Decisive seam test: mean displacement error as a function of absolute grid
position (0..63) along each axis, averaged over the other two axes and sims.

A genuine 2x2x2 patch seam shows sharp spikes at x=0 and x=32 (patch faces) that
are ABSENT in a whole-box model. The periodic box edge (x=0) is shared by all
G_correct models; the *internal* plane at x=32 is the patch-specific signature.

Compares any number of SR dirs against HR. Writes a per-axis profile figure +
prints the x=32 (internal) and x=0 (periodic) excess over the patch interior.
"""
import argparse
import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.pair_dataset import _load_6ch


def position_profile(sr_dir, data_root, snap, sids):
    """Return (3,64) mean |SR-HR| disp error per position, per axis."""
    prof = np.zeros((3, 64))
    n = 0
    for sid in sids:
        f = os.path.join(sr_dir, f"set{sid}_transformed.npy")
        if not os.path.exists(f):
            continue
        sr = np.load(f).astype(np.float32)[:3]
        hr = _load_6ch(os.path.join(data_root, f"set{sid}_quijote"), snap)[:3]
        err = np.abs(sr - hr).mean(axis=0)          # (64,64,64)
        prof[0] += err.mean(axis=(1, 2))            # along x
        prof[1] += err.mean(axis=(0, 2))            # along y
        prof[2] += err.mean(axis=(0, 1))            # along z
        n += 1
    return prof / max(n, 1), n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sr-dirs", nargs="+", required=True, help="label=dir pairs")
    p.add_argument("--data-root", default="/data/group_data/universedata/lagrangian_output_64/stitched/")
    p.add_argument("--snap", default="PART_009")
    p.add_argument("--out", required=True)
    p.add_argument("--max-sims", type=int, default=40)
    args = p.parse_args()

    runs = [s.split("=", 1) for s in args.sr_dirs]
    # sims = those present in the first dir
    d0 = runs[0][1]
    sids = sorted(int(m.group(1)) for f in os.listdir(d0)
                  if (m := re.match(r"set(\d+)_transformed\.npy$", f)))[:args.max_sims]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    print(f"positions: internal patch plane=32, periodic edge=0; interior ref = mean(8..24,40..56)")
    for label, d in runs:
        prof, n = position_profile(d, args.data_root, args.snap, sids)
        mean_axis = prof.mean(axis=0)               # average x,y,z profiles
        interior = np.concatenate([mean_axis[8:25], mean_axis[40:57]]).mean()
        ax.plot(range(64), mean_axis, lw=1.6, label=f"{label} (n={n})")
        print(f"  {label:>22}: x=0 {mean_axis[0]:.3f} ({mean_axis[0]/interior:.3f})  "
              f"x=31 {mean_axis[31]:.3f}  x=32 {mean_axis[32]:.3f} ({mean_axis[32]/interior:.3f})  "
              f"interior {interior:.3f}")
    for s in (0, 32):
        ax.axvline(s, color="grey", ls=":", lw=0.8)
    ax.set_xlabel("grid position (averaged over other 2 axes & sims)")
    ax.set_ylabel("displacement MAE [Mpc/h]")
    ax.set_title("Per-position error — patch seam = spike at x=32 absent in whole-box")
    ax.legend()
    fig.tight_layout(); fig.savefig(f"{args.out}.png", dpi=140)
    print(f"wrote {args.out}.png")


if __name__ == "__main__":
    main()
