"""Real-scale (128³, 64³ patches) seam + fidelity evaluation vs assembled HR.

Combines what the proxy split across eval_seam + seam_position_profile:
  * seam ratio: disp MAE at the internal patch boundary (x=64 plane) vs interior
  * per-position error profile along each axis (the decisive seam test: spike at 64?)
  * stitched Pk(SR)/Pk(HR) ratio (128³, periodic)
  * |SR-HR| slice figure (marker-free)
HR is assembled on the fly from quijote-64 patches (NOT stitched/, a diff realization).

Internal patch faces are at planes 0 and 64 (2×2×2 of 64³ in a 128³ box). The plane
at 0 is also the periodic box edge; the plane at 64 is the PURE internal seam — so we
report x=64 separately as the patch-specific signature.
"""
import argparse, os, re
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

from analysis.pk_torch import TorchPk
from data.patch_dataset_real import (
    PatchPairDatasetReal, assemble_box, PATCH, N_FULL,
)

MAXD = PATCH // 2  # 32


def seam_distance_grid(n=N_FULL):
    q = np.arange(n) % PATCH
    d1 = np.minimum(q, PATCH - 1 - q)
    return np.minimum.reduce(np.meshgrid(d1, d1, d1, indexing="ij"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sr-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--label", default="")
    p.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    p.add_argument("--lbox", type=float, default=1000.0)
    p.add_argument("--n-pk-bins", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-sims", type=int, default=0)
    args = p.parse_args()
    label = args.label or os.path.basename(args.out)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pk = TorchPk(N=N_FULL, lbox=args.lbox, n_bins=args.n_pk_bins, device=dev)
    ds = PatchPairDatasetReal(split=args.split, seed=args.seed)
    keep = set(ds.ids)

    files = sorted((int(m.group(1)), f) for f in os.listdir(args.sr_dir)
                   if (m := re.match(r"set(\d+)_transformed\.npy$", f)) and int(m.group(1)) in keep)
    if args.max_sims > 0:
        files = files[:args.max_sims]
    assert files, f"no matching SR cubes in {args.sr_dir} for split {args.split}"

    dgrid = seam_distance_grid()
    dmask = [dgrid == d for d in range(MAXD)]
    prof_sum = np.zeros(MAXD); prof_cnt = np.zeros(MAXD)
    pos_sum = np.zeros((3, N_FULL)); pos_n = 0
    pk_sr_all, pk_hr_all = [], []
    slice_done = False
    root, snap = ds.root, ds.snap
    for n, (sid, fname) in enumerate(files):
        sr = np.load(os.path.join(args.sr_dir, fname)).astype(np.float32)
        hr = assemble_box(root, "quijote-64", sid, snap).astype(np.float32)
        err = np.abs(sr[:3] - hr[:3]).mean(axis=0)  # (128,128,128)
        for d in range(MAXD):
            prof_sum[d] += err[dmask[d]].sum(); prof_cnt[d] += dmask[d].sum()
        pos_sum[0] += err.mean(axis=(1, 2)); pos_sum[1] += err.mean(axis=(0, 2)); pos_sum[2] += err.mean(axis=(0, 1))
        pos_n += 1
        pk_sr_all.append(pk(torch.from_numpy(sr[:3]).unsqueeze(0).to(dev)).cpu().numpy()[0])
        pk_hr_all.append(pk(torch.from_numpy(hr[:3]).unsqueeze(0).to(dev)).cpu().numpy()[0])
        if not slice_done:
            fig, ax = plt.subplots(1, 3, figsize=(16, 5))
            vmax = np.percentile(np.abs(hr[0]), 99)
            ax[0].imshow(hr[0][:, :, 64], origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax); ax[0].set_title(f"HR disp_x (set{sid}, z=64)")
            ax[1].imshow(sr[0][:, :, 64], origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax); ax[1].set_title("SR stitched disp_x")
            im = ax[2].imshow(err[:, :, 64], origin="lower", cmap="magma"); ax[2].set_title("|SR-HR| disp mean"); fig.colorbar(im, ax=ax[2], shrink=.8)
            fig.suptitle(label); fig.tight_layout(); fig.savefig(f"{args.out}_slice.png", dpi=130); plt.close(fig)
            slice_done = True
        if (n + 1) % 25 == 0 or n + 1 == len(files):
            print(f"  {n+1}/{len(files)}", flush=True)

    profile = prof_sum / prof_cnt
    seam_err, interior_err = profile[0], profile[8:].mean()
    seam_ratio = seam_err / interior_err
    posp = pos_sum.mean(0) / pos_n
    interior_pos = np.concatenate([posp[8:25], posp[40:57]]).mean()
    x64_excess = posp[64] / interior_pos          # PURE internal seam
    x0_excess = posp[0] / interior_pos            # periodic edge (control)
    pk_sr, pk_hr = np.array(pk_sr_all), np.array(pk_hr_all)
    m = pk_hr.mean(0) > -10
    pk_rms = float(np.sqrt(((pk_sr - pk_hr) ** 2)[:, m].mean()))
    ratio = 10 ** (pk_sr - pk_hr)
    k = np.sqrt(pk.edges[:-1].cpu().numpy() * pk.edges[1:].cpu().numpy())

    np.savez(f"{args.out}.npz", sids=np.array([s for s, _ in files]), profile=profile,
             seam_err=seam_err, interior_err=interior_err, seam_ratio=seam_ratio,
             pos_profile=posp, x64_excess=x64_excess, x0_excess=x0_excess,
             pk_rms=pk_rms, pk_sr=pk_sr, pk_hr=pk_hr, k=k, label=label)
    with open(f"{args.out}.md", "w") as f:
        f.write(f"# Real-scale seam eval — {label}\n\nsims: {len(files)} ({args.split})\n\n"
                f"| metric | value |\n|---|---|\n"
                f"| seam-dist ratio (d=0 / d>=8) | {seam_ratio:.4f} |\n"
                f"| **x=64 internal-seam excess** | **{x64_excess:.4f}** |\n"
                f"| x=0 periodic-edge excess (control) | {x0_excess:.4f} |\n"
                f"| stitched Pk RMS (log10) | {pk_rms:.4f} |\n")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(range(N_FULL), posp, lw=1.6)
    for s in (0, 64): ax.axvline(s, color="grey", ls=":", lw=0.9)
    ax.set_xlabel("grid position"); ax.set_ylabel("disp MAE [Mpc/h]")
    ax.set_title(f"Per-position error — {label}  (x=64 excess {x64_excess:.3f})")
    fig.tight_layout(); fig.savefig(f"{args.out}_posprofile.png", dpi=130); plt.close(fig)

    print(f"\n== {label} ==  seam_ratio={seam_ratio:.4f}  x64_excess={x64_excess:.4f}  "
          f"x0_excess={x0_excess:.4f}  pkRMS={pk_rms:.4f}")
    print(f"wrote {args.out}.npz/.md + figures")


if __name__ == "__main__":
    main()
