"""Seam-artifact + fidelity evaluation of stitched SR cubes vs HR ground truth.

For every test sim it computes, on the displacement channels (physical units):
  * |SR - HR| profiled by distance to the nearest patch seam plane
    (seams at x,y,z in {0, 32}; voxels at q=0 or q=31 within a patch touch one)
  * seam ratio = mean error at seam-adjacent voxels (d=0) / interior (d>=8)
  * log10 Pk of SR and HR (periodic 64^3, divergence estimator)

Outputs: <out>.npz with all arrays, <out>.md table, and figures
<out>_pkratio.png, <out>_seamprofile.png, <out>_slice.png.

Usage:
  python -m analysis.eval_seam --sr-dir runs/patch/transformed_v1_naive \
      --out runs/patch/seam_v1_naive [--label "Arm A naive"]
"""
import argparse
import os
import re
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.pk_torch import TorchPk
from data.pair_dataset import _load_6ch
from data.patch_dataset import PATCH

MAX_D = PATCH // 2  # 16 distance bins (0..15)


def seam_distance_grid(n=64):
    """d(v) = min over axes of distance to the nearest seam face; seams between
    voxels 31|32 and 63|0 along each axis."""
    q = np.arange(n) % PATCH
    d1 = np.minimum(q, PATCH - 1 - q)            # per-axis distance, 0..15
    return np.minimum.reduce(np.meshgrid(d1, d1, d1, indexing="ij"))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sr-dir", required=True, help="dir of set{sid}_transformed.npy (denormalized)")
    p.add_argument("--data-root", default="/data/group_data/universedata/lagrangian_output_64/stitched/")
    p.add_argument("--out", required=True, help="output prefix")
    p.add_argument("--label", default="")
    p.add_argument("--lbox", type=float, default=1000.0)
    p.add_argument("--n-pk-bins", type=int, default=32)
    p.add_argument("--snap", default="PART_009")
    p.add_argument("--max-sims", type=int, default=0)
    p.add_argument("--split", default="test", choices=["train", "val", "test", "all"],
                   help="restrict to sims of this split (same seed-0 split as training)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    label = args.label or os.path.basename(args.out)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pk = TorchPk(N=64, lbox=args.lbox, n_bins=args.n_pk_bins, device=device)
    dgrid = seam_distance_grid()
    dmask = [dgrid == d for d in range(MAX_D)]

    files = sorted(
        (int(re.match(r"set(\d+)_transformed\.npy$", f).group(1)), f)
        for f in os.listdir(args.sr_dir)
        if re.match(r"set(\d+)_transformed\.npy$", f)
    )
    if args.split != "all":
        from data.patch_dataset import PatchPairDataset
        keep = set(PatchPairDataset(args.data_root, split=args.split,
                                    seed=args.seed).ids)
        files = [(s, f) for s, f in files if s in keep]
        print(f"split={args.split}: {len(files)} sims")
    if args.max_sims > 0:
        files = files[:args.max_sims]
    assert files, f"no set*_transformed.npy in {args.sr_dir} (split={args.split})"

    prof_sum = np.zeros(MAX_D)        # per-distance |err| accumulators (disp)
    prof_cnt = np.zeros(MAX_D)
    pk_sr_all, pk_hr_all = [], []
    slice_saved = False
    for n, (sid, fname) in enumerate(files):
        sr = np.load(os.path.join(args.sr_dir, fname)).astype(np.float32)
        hr = _load_6ch(os.path.join(args.data_root, f"set{sid}_quijote"), args.snap)
        err = np.abs(sr[:3] - hr[:3]).mean(axis=0)        # (64,64,64) disp error
        for d in range(MAX_D):
            prof_sum[d] += err[dmask[d]].sum()
            prof_cnt[d] += dmask[d].sum()
        pk_sr_all.append(pk(torch.from_numpy(sr[:3]).unsqueeze(0).to(device)).cpu().numpy()[0])
        pk_hr_all.append(pk(torch.from_numpy(hr[:3]).unsqueeze(0).to(device)).cpu().numpy()[0])
        if not slice_saved:
            fig, axes = plt.subplots(1, 3, figsize=(16, 5))
            vmax = np.percentile(np.abs(hr[0]), 99)
            for ax, (img, title) in zip(axes, [
                (hr[0][:, :, 32], f"HR disp_x (set{sid}, z=32)"),
                (sr[0][:, :, 32], "SR stitched disp_x"),
                (err[:, :, 32], "|SR-HR| disp mean"),
            ]):
                im = ax.imshow(img, origin="lower",
                               vmax=vmax if "HR" in title or "SR" in title else None,
                               vmin=-vmax if "|" not in title else 0,
                               cmap="RdBu_r" if "|" not in title else "magma")
                for s in (0, 32):
                    ax.axhline(s - 0.5, color="lime", lw=0.6, ls="--")
                    ax.axvline(s - 0.5, color="lime", lw=0.6, ls="--")
                ax.set_title(title); fig.colorbar(im, ax=ax, shrink=0.8)
            fig.suptitle(label)
            fig.tight_layout()
            fig.savefig(f"{args.out}_slice.png", dpi=130)
            plt.close(fig)
            slice_saved = True
        if (n + 1) % 25 == 0 or n + 1 == len(files):
            print(f"  {n+1}/{len(files)} sims", flush=True)

    profile = prof_sum / prof_cnt
    seam_err, interior_err = profile[0], profile[8:].mean()
    seam_ratio = seam_err / interior_err
    pk_sr = np.array(pk_sr_all); pk_hr = np.array(pk_hr_all)
    m = pk_hr.mean(0) > -10
    pk_rms = float(np.sqrt(((pk_sr - pk_hr) ** 2)[:, m].mean()))
    ratio = 10 ** (pk_sr - pk_hr)                       # P_SR/P_HR per sim/bin
    k_centers = np.sqrt(pk.edges[:-1].cpu().numpy() * pk.edges[1:].cpu().numpy())

    np.savez(f"{args.out}.npz",
             sids=np.array([s for s, _ in files]),
             profile=profile, seam_err=seam_err, interior_err=interior_err,
             seam_ratio=seam_ratio, pk_rms=pk_rms,
             pk_sr=pk_sr, pk_hr=pk_hr, k=k_centers, label=label)

    with open(f"{args.out}.md", "w") as f:
        f.write(f"# Seam evaluation — {label}\n\n"
                f"sims: {len(files)}  (sr-dir: {args.sr_dir})\n\n"
                f"| metric | value |\n|---|---|\n"
                f"| seam-adjacent disp MAE (d=0) | {seam_err:.4f} |\n"
                f"| interior disp MAE (d>=8) | {interior_err:.4f} |\n"
                f"| **seam ratio** | **{seam_ratio:.4f}** |\n"
                f"| Pk RMS (log10, vs HR) | {pk_rms:.4f} |\n"
                f"| P_SR/P_HR @ low k (first 4 bins) | {ratio[:, m][:, :4].mean():.4f} |\n"
                f"| P_SR/P_HR @ high k (last 4 bins) | {ratio[:, m][:, -4:].mean():.4f} |\n")

    fig, ax = plt.subplots(figsize=(7, 5))
    rm, rs = ratio.mean(0), ratio.std(0)
    ax.semilogx(k_centers[m], rm[m], lw=2, label=label)
    ax.fill_between(k_centers[m], (rm - rs)[m], (rm + rs)[m], alpha=0.25)
    ax.axhline(1.0, color="k", ls=":")
    ax.set_xlabel("k [h/Mpc]"); ax.set_ylabel("P_SR(k) / P_HR(k)")
    ax.set_title(f"Stitched Pk ratio — {label}"); ax.legend()
    fig.tight_layout(); fig.savefig(f"{args.out}_pkratio.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(range(MAX_D), profile, "o-")
    ax.set_xlabel("voxel distance to nearest seam"); ax.set_ylabel("disp MAE [Mpc/h]")
    ax.set_title(f"Error vs seam distance — {label}  (seam ratio {seam_ratio:.3f})")
    fig.tight_layout(); fig.savefig(f"{args.out}_seamprofile.png", dpi=130); plt.close(fig)

    print(f"\n== {label} ==")
    print(f"seam MAE={seam_err:.4f}  interior MAE={interior_err:.4f}  "
          f"seam ratio={seam_ratio:.4f}  pkRMS={pk_rms:.4f}")
    print(f"wrote {args.out}.npz/.md and 3 figures")


if __name__ == "__main__":
    main()
