#!/usr/bin/env python3
"""
4-panel density projection panels per simulation.

Panels: Input (prepatch 0_0_0) | Ground Truth (halos) | Predicted (stitched) | Residual

Two PNG files are written per sim per projection axis:
  set{id}_density_panel_z_ratio.png      — single mid-slice along z
  set{id}_density_panel_z_mean_ratio.png — mean projection along z

Usage:
    python -m data.density_panel
    python -m data.density_panel --sim-ids 0 100 --out-dir /tmp/panels
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SNAPSHOT = "PART_009"
DEFAULT_BOX_SIZE = 1000.0
DEFAULT_NMESH = 64
DEFAULT_REDSHIFT_KEY = "0.666667"


# ---------------------------------------------------------------------------
# CIC helpers (shared with density_histogram.py)
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


def disp_to_rho(disp: np.ndarray, box_size: float,
                voxel_side: float | None = None) -> np.ndarray:
    """CIC-deposit displaced Lagrangian lattice. Returns (N,N,N) rho summing to N^3.

    Lattice nodes are at q = i * voxel_side (node-centred, matching
    conversion.build_global_voxel_grid — NOT cell-centred, which would shift
    every reconstructed halo by half a voxel).  voxel_side defaults to
    box_size / N (exact for fixed-mesh outputs); pass the meta.json value for
    conversion.py outputs, where N * voxel_side != box_size in general.
    """
    N = disp.shape[1]
    cell = box_size / N
    if voxel_side is None:
        voxel_side = cell
    g = np.arange(N) * voxel_side
    lx = g[:, None, None]; ly = g[None, :, None]; lz = g[None, None, :]
    px = ((disp[0] + lx) % box_size) / cell
    py = ((disp[1] + ly) % box_size) / cell
    pz = ((disp[2] + lz) % box_size) / cell
    return _cic_accumulate(np.stack([px, py, pz], axis=0).reshape(3, -1), N)


def halos_to_rho(halo_pos: np.ndarray, box_size: float, N: int) -> np.ndarray:
    """CIC-deposit halo positions. Returns (N,N,N) rho summing to n_halos."""
    cell = box_size / N
    return _cic_accumulate((halo_pos % box_size).T / cell, N)


def rho_to_ratio(rho: np.ndarray) -> np.ndarray:
    """Return rho / rho_bar (so mean = 1)."""
    return rho / max(float(rho.mean()), 1e-12)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_disp(path: Path) -> np.ndarray:
    return np.load(path).astype(np.float32)


def load_halo_positions(halo_dir: Path, sim_id: int,
                        redshift_key: str = DEFAULT_REDSHIFT_KEY) -> np.ndarray:
    h5path = (halo_dir / "quijote" / "nbody" / "L1000-N128"
              / str(sim_id) / "halos.h5")
    if not h5path.exists():
        raise FileNotFoundError(f"Halo catalog not found: {h5path}")
    with h5py.File(h5path, "r") as f:
        if redshift_key not in f:
            raise KeyError(f"Redshift key {redshift_key!r} not in {h5path}")
        pos = f[redshift_key]["pos"][:]
    return pos.astype(np.float32)


# ---------------------------------------------------------------------------
# Projection helpers
# ---------------------------------------------------------------------------

def project(rho_ratio: np.ndarray, axis: int, mode: str) -> np.ndarray:
    """Project (N,N,N) density ratio along `axis`.

    mode='slice' : single mid-plane slice (no reduction).
    mode='mean'  : mean along `axis`.
    """
    if mode == "slice":
        idx = rho_ratio.shape[axis] // 2
        return np.take(rho_ratio, idx, axis=axis)
    return rho_ratio.mean(axis=axis)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _make_panel(rho_pre: np.ndarray, rho_orig: np.ndarray, rho_st: np.ndarray,
                mae: float, patch_dir_count: int, sim_id: int,
                axis: int, mode: str, out_path: Path) -> None:
    axis_name = "xyz"[axis]
    mode_label = "mean" if mode == "mean" else "slice"

    p_input = project(rho_pre,  axis, mode)
    p_truth = project(rho_orig, axis, mode)
    p_pred  = project(rho_st,   axis, mode)

    # Residual on projections
    safe_pred = np.where(np.abs(p_pred) > 1e-6, p_pred, 1e-6)
    residual  = (p_truth - p_pred) / safe_pred

    vmax = float(np.percentile(p_truth, 99.5))
    vmin = 0.0
    rabs = float(max(abs(np.percentile(residual, 5)),
                     abs(np.percentile(residual, 95))))

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    for ax, img, title in [
        (axes[0], p_input, "Input Density"),
        (axes[1], p_truth, "Ground Truth Density"),
        (axes[2], p_pred,  "Predicted Density"),
    ]:
        im = ax.imshow(img, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=13)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    im_r = axes[3].imshow(residual, origin="lower", cmap="bwr",
                          vmin=-rabs, vmax=rabs)
    axes[3].set_title("Residual (Truth - Pred) / Pred", fontsize=13)
    axes[3].set_xticks([]); axes[3].set_yticks([])
    plt.colorbar(im_r, ax=axes[3], fraction=0.046, pad=0.04)

    axes[1].set_xlabel(r"$\rho/\hat{\rho}$", fontsize=11)

    fig.suptitle(
        f"set{sim_id} density comparison | axis={axis_name} | "
        f"patch dirs={patch_dir_count} | stitched MAE={mae:.3f}\n"
        f"input=pre-stitch patch 0_0_0, truth=original halo mesh, "
        f"predicted=stitched converted mesh",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_and_plot(sim_id: int, data_root: Path, halo_dir: Path,
                     box_size: float, N: int, redshift_key: str,
                     out_dir: Path) -> None:
    prepatch_path = (data_root / f"quijote-{N}"
                     / f"set{sim_id}_pos_0_0_0" / SNAPSHOT / "disp.npy")
    stitched_path = (data_root / "stitched"
                     / f"set{sim_id}_quijote" / SNAPSHOT / "disp.npy")

    print(f"Loading halos for sim {sim_id} ...")
    halo_pos = load_halo_positions(halo_dir, sim_id, redshift_key)
    n_halos  = len(halo_pos)
    print(f"  {n_halos:,} halos")

    print(f"  Computing density fields ...")
    # Lattice spacing of the conversion.py grid these disp fields were built on
    # (N * voxel_side != box_size in general, so the default would be wrong).
    voxel_side = (box_size ** 3 / n_halos) ** (1.0 / 3.0)

    rho_orig = halos_to_rho(halo_pos, box_size, N)
    rho_pre  = disp_to_rho(load_disp(prepatch_path), box_size, voxel_side=voxel_side)
    rho_st   = disp_to_rho(load_disp(stitched_path),  box_size, voxel_side=voxel_side)

    ratio_orig = rho_to_ratio(rho_orig)
    ratio_pre  = rho_to_ratio(rho_pre)
    ratio_st   = rho_to_ratio(rho_st)

    delta_orig = ratio_orig - 1.0
    delta_st   = ratio_st   - 1.0
    mae = float(np.mean(np.abs(delta_st - delta_orig)))
    print(f"  stitched MAE vs original = {mae:.4f}")

    patch_dir_count = len(list(
        (data_root / f"quijote-{N}").glob(f"set{sim_id}_pos_*")
    ))

    out_dir.mkdir(parents=True, exist_ok=True)

    for mode, suffix in [("slice", "z_ratio"), ("mean", "z_mean_ratio")]:
        out_path = out_dir / f"set{sim_id}_density_panel_{suffix}.png"
        _make_panel(ratio_pre, ratio_orig, ratio_st,
                    mae, patch_dir_count, sim_id,
                    axis=2, mode=mode, out_path=out_path)


def main() -> None:
    p = argparse.ArgumentParser(
        description="4-panel density projection panels (prepatch / truth / stitched / residual)"
    )
    p.add_argument(
        "--data-root",
        default="/home/juliahul/projects/stuff/universedata/lagrangian_output_64",
    )
    p.add_argument(
        "--halo-dir",
        default="/home/juliahul/projects/stuff/universedata/cmass-ili",
    )
    p.add_argument("--sim-ids", nargs="+", type=int, default=[42])
    p.add_argument("--out-dir", default=None,
                   help="Default: <data-root>/analysis/density_panels")
    p.add_argument("--box-size", type=float, default=DEFAULT_BOX_SIZE)
    p.add_argument("--nmesh",    type=int,   default=DEFAULT_NMESH)
    p.add_argument("--redshift", default=DEFAULT_REDSHIFT_KEY)
    args = p.parse_args()

    data_root = Path(args.data_root)
    halo_dir  = Path(args.halo_dir)
    out_dir   = (Path(args.out_dir) if args.out_dir
                 else data_root / "analysis" / "density_panels")

    for sim_id in args.sim_ids:
        process_and_plot(sim_id, data_root, halo_dir,
                         args.box_size, args.nmesh, args.redshift, out_dir)


if __name__ == "__main__":
    main()
