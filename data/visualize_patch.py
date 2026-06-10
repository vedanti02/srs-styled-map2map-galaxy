#!/usr/bin/env python3
"""
Quick visualisation of a single (or all) patch displacement density fields.

For each patch found under quijote-{N}/set{id}_pos_*/, computes the CIC
density from the displacement field and plots a z-slice and z-mean projection
side by side.  Patches are laid out in a grid arranged by their (i,j,k) index.

Usage:
    python -m data.visualize_patch --sim-id 42
    python -m data.visualize_patch --sim-id 42 --patch 0 0 0   # single patch
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SNAPSHOT = "PART_009"
DEFAULT_BOX_SIZE = 1000.0
DEFAULT_NMESH = 64


# ---------------------------------------------------------------------------
# CIC (same as density_panel.py)
# ---------------------------------------------------------------------------

def _cic_accumulate(pos_cell: np.ndarray, N: int) -> np.ndarray:
    rho = np.zeros((N, N, N), dtype=np.float64)
    i0 = np.floor(pos_cell).astype(np.int64) % N
    f  = pos_cell - np.floor(pos_cell)
    i1 = (i0 + 1) % N
    one_f = 1.0 - f
    for wx, ix in [(one_f[0], i0[0]), (f[0], i1[0])]:
        for wy, iy in [(one_f[1], i0[1]), (f[1], i1[1])]:
            for wz, iz in [(one_f[2], i0[2]), (f[2], i1[2])]:
                np.add.at(rho, (ix, iy, iz), wx * wy * wz)
    return rho.astype(np.float32)


def disp_to_ratio(disp: np.ndarray, box_size: float) -> np.ndarray:
    """CIC-deposit displaced lattice → rho/rho_bar."""
    N = disp.shape[1]
    cell = box_size / N
    g = np.arange(N) * cell + 0.5 * cell
    lx = g[:, None, None]; ly = g[None, :, None]; lz = g[None, None, :]
    px = ((disp[0] + lx) % box_size) / cell
    py = ((disp[1] + ly) % box_size) / cell
    pz = ((disp[2] + lz) % box_size) / cell
    rho = _cic_accumulate(np.stack([px, py, pz]).reshape(3, -1), N)
    return rho / max(float(rho.mean()), 1e-12)


# ---------------------------------------------------------------------------
# Patch discovery
# ---------------------------------------------------------------------------

_PATCH_RE = re.compile(r"set\d+_pos_(\d+)_(\d+)_(\d+)$")

def find_patches(data_root: Path, sim_id: int, N: int,
                 sim_type: str = "quijote") -> list[tuple[tuple, Path]]:
    """Return [(ijk, disp_path), ...] sorted by ijk."""
    patch_root = data_root / f"{sim_type}-{N}"
    out = []
    for d in sorted(patch_root.glob(f"set{sim_id}_pos_*")):
        m = _PATCH_RE.match(d.name)
        if m:
            ijk = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
            disp_path = d / SNAPSHOT / "disp.npy"
            if disp_path.exists():
                out.append((ijk, disp_path))
    return sorted(out)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_single_patch(ijk: tuple, disp_path: Path, sim_id: int,
                      box_size: float, out_path: Path) -> None:
    disp  = np.load(disp_path).astype(np.float32)
    ratio = disp_to_ratio(disp, box_size)
    N     = ratio.shape[0]

    z_slice = ratio[:, :, N // 2]
    z_mean  = ratio.mean(axis=2)
    vmax = float(np.percentile(ratio, 99.5))

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, img, title in [
        (axes[0], z_slice, f"z-slice (k={N//2})"),
        (axes[1], z_mean,  "z-mean projection"),
    ]:
        im = ax.imshow(img, origin="lower", cmap="viridis", vmin=0, vmax=vmax)
        ax.set_title(title, fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    axes[0].set_ylabel(r"$\rho/\hat{\rho}$", fontsize=11)
    i, j, k = ijk
    fig.suptitle(
        f"set{sim_id}  patch {i}_{j}_{k}  |  {N}³ voxels  |  "
        f"box={box_size:.0f} Mpc/h",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def plot_all_patches(patches: list[tuple[tuple, Path]], sim_id: int,
                     box_size: float, out_path: Path) -> None:
    """Grid of z-slice projections, one cell per patch."""
    ijks   = [p[0] for p in patches]
    i_vals = sorted(set(t[0] for t in ijks))
    j_vals = sorted(set(t[1] for t in ijks))

    n_rows = len(i_vals)
    n_cols = len(j_vals)
    # Stack all patches along k for each (i,j) cell
    ratio_map: dict[tuple, list] = {}
    for ijk, disp_path in patches:
        i, j, _ = ijk
        disp  = np.load(disp_path).astype(np.float32)
        ratio = disp_to_ratio(disp, box_size)
        ratio_map.setdefault((i, j), []).append(ratio)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4 * n_cols, 4 * n_rows),
                             squeeze=False)

    N = next(iter(ratio_map.values()))[0].shape[2]
    global_vmax = 0.0
    imgs = {}
    for (i, j), ratios in ratio_map.items():
        merged = np.mean(ratios, axis=0)[:, :, N // 2]  # z-slice at mid-plane, averaged across k-patches
        imgs[(i, j)] = merged
        global_vmax = max(global_vmax, float(np.percentile(merged, 99.5)))

    for ri, i in enumerate(i_vals):
        for ci, j in enumerate(j_vals):
            ax = axes[ri][ci]
            if (i, j) in imgs:
                im = ax.imshow(imgs[(i, j)], origin="lower", cmap="viridis",
                               vmin=0, vmax=global_vmax)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                ax.set_title(f"i={i}, j={j}", fontsize=10)
            else:
                ax.set_visible(False)
            ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(
        f"set{sim_id}  |  all {len(patches)} patches  |  "
        f"z-slice (k={N//2})  |  box={box_size:.0f} Mpc/h",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Visualise patch displacement density fields"
    )
    p.add_argument("--data-root",
                   default="/home/juliahul/projects/stuff/universedata/lagrangian_output_64")
    p.add_argument("--sim-id", type=int, default=42)
    p.add_argument("--patch", nargs=3, type=int, default=None,
                   metavar=("I", "J", "K"),
                   help="Single patch to plot (default: plot all patches)")
    p.add_argument("--out-dir", default=None,
                   help="Default: <data-root>/analysis/patch_vis")
    p.add_argument("--box-size", type=float, default=DEFAULT_BOX_SIZE)
    p.add_argument("--nmesh",    type=int,   default=DEFAULT_NMESH)
    p.add_argument("--sim-type", default="quijote",
                   help="Subfolder prefix (default: quijote → quijote-{nmesh})")
    args = p.parse_args()

    data_root = Path(args.data_root)
    out_dir   = (Path(args.out_dir) if args.out_dir
                 else data_root / "analysis" / "patch_vis")
    out_dir.mkdir(parents=True, exist_ok=True)

    patches = find_patches(data_root, args.sim_id, args.nmesh, args.sim_type)
    if not patches:
        raise RuntimeError(f"No patches found for set{args.sim_id} under {data_root}")
    print(f"Found {len(patches)} patches for set{args.sim_id}")

    if args.patch is not None:
        ijk = tuple(args.patch)
        match = [(t, dp) for t, dp in patches if t == ijk]
        if not match:
            raise ValueError(f"Patch {ijk} not found. Available: {[t for t,_ in patches]}")
        i, j, k = ijk
        out_path = out_dir / f"{args.sim_type}_set{args.sim_id}_patch_{i}_{j}_{k}.png"
        plot_single_patch(ijk, match[0][1], args.sim_id, args.box_size, out_path)
    else:
        out_path = out_dir / f"{args.sim_type}_set{args.sim_id}_all_patches.png"
        plot_all_patches(patches, args.sim_id, args.box_size, out_path)


if __name__ == "__main__":
    main()
