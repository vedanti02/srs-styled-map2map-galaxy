#!/usr/bin/env python3
"""
Density histogram comparison: original halo catalog vs prepatch000 vs stitched.

For each voxel on the Lagrangian lattice we treat it as a particle at position
q (lattice site) and displace it by Psi(q) to get the Eulerian position
x = q + Psi. Each displaced particle's mass is then distributed to the 8
surrounding mesh cells via trilinear (CIC) weighting, yielding a density field
rho. Dividing by the mean and subtracting 1 gives the overdensity delta.

Three fields are compared per sim:
  - original  : halo positions CIC-deposited onto N^3 grid directly from catalog
  - prepatch000: density reconstructed from set{id}_pos_0_0_0 displacement patch
  - stitched  : density reconstructed from stitched/set{id}_quijote full-box field

Saves:
  <out_dir>/set{a}_set{b}_density_histograms.png
  <out_dir>/set{a}_set{b}_density_histograms_summary.json

Usage:
    python -m data.density_histogram \\
        --data-root /home/juliahul/projects/stuff/universedata/lagrangian_output_64 \\
        --halo-dir  /home/juliahul/projects/stuff/universedata/cmass-ili \\
        --sim-ids 0 100
"""

from __future__ import annotations

import argparse
import json
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
# CIC helpers
# ---------------------------------------------------------------------------

def _cic_accumulate(pos_cell: np.ndarray, N: int) -> np.ndarray:
    """Trilinear deposit of unit-mass particles at fractional cell positions.

    Args:
        pos_cell: (3, M) particle positions in cell units (0 .. N).
        N: grid side.

    Returns:
        (N, N, N) rho (sum = M).
    """
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
    """Displace Lagrangian lattice particles by `disp` and CIC-deposit.

    Each of the N^3 voxels is treated as a particle at its lattice position
    q. After displacement to x = q + Psi (mod L), it contributes to the 8
    neighbouring cells with trilinear weights.

    Lattice convention: conversion.build_global_voxel_grid places voxel i's
    centre at q = i * voxel_side (node-centred, voxel 0 at the origin), and
    displacements are measured FROM those nodes.  So reconstruction must start
    particles at q = i * voxel_side too — a cell-centred lattice
    ((i + 0.5) * cell) would shift every reconstructed halo by half a voxel.

    Args:
        disp: (3, N, N, N) displacement field in the same units as box_size.
        box_size: periodic box side length.
        voxel_side: lattice spacing used when `disp` was generated.  Defaults
            to box_size / N, which is exact for fixed-mesh outputs
            (run_voxelize.py).  For conversion.py outputs pass the meta.json
            voxel_side, since there N * voxel_side != box_size in general.

    Returns:
        (N, N, N) rho, summing to N^3.
    """
    N = disp.shape[1]
    cell = box_size / N                      # CIC deposit mesh spacing
    if voxel_side is None:
        voxel_side = cell
    g = np.arange(N) * voxel_side            # node-centred lattice positions
    lx = g[:, None, None]
    ly = g[None, :, None]
    lz = g[None, None, :]

    px = ((disp[0] + lx) % box_size) / cell
    py = ((disp[1] + ly) % box_size) / cell
    pz = ((disp[2] + lz) % box_size) / cell

    pos_cell = np.stack([px, py, pz], axis=0).reshape(3, -1)
    return _cic_accumulate(pos_cell, N)


def halos_to_rho(halo_pos: np.ndarray, box_size: float, N: int) -> np.ndarray:
    """CIC-deposit halo positions onto N^3 grid.

    Returns:
        (N, N, N) rho, summing to len(halo_pos).
    """
    cell = box_size / N
    pos_cell = (halo_pos % box_size).T / cell    # (3, M)
    return _cic_accumulate(pos_cell, N)


def rho_to_delta(rho: np.ndarray) -> np.ndarray:
    rho_bar = rho.mean()
    return rho / max(float(rho_bar), 1e-12) - 1.0


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_disp(path: Path) -> np.ndarray:
    return np.load(path).astype(np.float32)


def load_halo_positions(
    halo_dir: Path, sim_id: int, redshift_key: str = DEFAULT_REDSHIFT_KEY
) -> np.ndarray:
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
# Per-sim processing
# ---------------------------------------------------------------------------

def process_sim(
    sim_id: int,
    data_root: Path,
    halo_dir: Path,
    box_size: float,
    N: int,
    redshift_key: str,
) -> dict:
    prepatch_path = (data_root / f"quijote-{N}"
                     / f"set{sim_id}_pos_0_0_0" / SNAPSHOT / "disp.npy")
    stitched_path = (data_root / "stitched"
                     / f"set{sim_id}_quijote" / SNAPSHOT / "disp.npy")

    # Original halo catalog → density
    halo_pos = load_halo_positions(halo_dir, sim_id, redshift_key)
    n_halos = len(halo_pos)
    voxel_side = (box_size**3 / n_halos) ** (1.0 / 3.0)

    rho_orig = halos_to_rho(halo_pos, box_size, N)
    delta_orig = rho_to_delta(rho_orig)

    # Prepatch000 — pass the conversion-grid lattice spacing (N * voxel_side
    # != box_size for conversion.py outputs, so the default would be wrong).
    disp_pre = load_disp(prepatch_path)
    rho_pre = disp_to_rho(disp_pre, box_size, voxel_side=voxel_side)
    delta_pre = rho_to_delta(rho_pre)
    zero_vox_pre = int(np.all(disp_pre == 0, axis=0).sum())

    patch_dir_count = len(list(
        (data_root / f"quijote-{N}").glob(f"set{sim_id}_pos_*")
    ))

    # Stitched — same conversion-grid lattice spacing as the patches.
    disp_st = load_disp(stitched_path)
    rho_st = disp_to_rho(disp_st, box_size, voxel_side=voxel_side)
    delta_st = rho_to_delta(rho_st)
    zero_vox_st = int(np.all(disp_st == 0, axis=0).sum())

    metrics = {
        "original_total_counts":           float(rho_orig.sum()),
        "prepatch000_total_counts":        float(rho_pre.sum()),
        "stitched_total_counts":           float(rho_st.sum()),
        "prepatch000_zero_vector_voxels":  zero_vox_pre,
        "stitched_zero_vector_voxels":     zero_vox_st,
        "prepatch000_mse_vs_original":     float(np.mean((delta_pre - delta_orig) ** 2)),
        "prepatch000_mae_vs_original":     float(np.mean(np.abs(delta_pre - delta_orig))),
        "stitched_mse_vs_original":        float(np.mean((delta_st - delta_orig) ** 2)),
        "stitched_mae_vs_original":        float(np.mean(np.abs(delta_st - delta_orig))),
    }

    return {
        "sim_id":             sim_id,
        "box_size":           box_size,
        "nmesh":              N,
        "n_halos":            n_halos,
        "voxel_side_from_halos": voxel_side,
        "patch_dir_count":    patch_dir_count,
        "metrics":            metrics,
        "_deltas":            (delta_orig, delta_pre, delta_st),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_histograms(results: list[dict], out_path: Path, n_bins: int = 80) -> None:
    n_sims = len(results)
    fig, axes = plt.subplots(1, n_sims, figsize=(7 * n_sims, 5), squeeze=False)

    for col, r in enumerate(results):
        ax = axes[0, col]
        delta_orig, delta_pre, delta_st = r["_deltas"]
        sid = r["sim_id"]

        lo = min(np.percentile(delta_orig, 1),
                 np.percentile(delta_pre,  1),
                 np.percentile(delta_st,   1))
        hi = max(np.percentile(delta_orig, 99),
                 np.percentile(delta_pre,  99),
                 np.percentile(delta_st,   99))
        bins = np.linspace(lo, hi, n_bins + 1)

        ax.hist(delta_orig.ravel(), bins=bins, histtype="step", density=True,
                label=f"original (N_h={r['n_halos']:,})", color="black", lw=1.5)
        ax.hist(delta_pre.ravel(), bins=bins, histtype="step", density=True,
                label="prepatch000", color="C0", lw=1.5, linestyle="--")
        ax.hist(delta_st.ravel(),  bins=bins, histtype="step", density=True,
                label="stitched", color="C1", lw=1.5)

        m = r["metrics"]
        ax.set_title(
            f"set{sid}  |  N={r['nmesh']}³  |  L={r['box_size']:.0f} Mpc/h\n"
            f"prepatch MSE={m['prepatch000_mse_vs_original']:.3f}  "
            f"stitched MSE={m['stitched_mse_vs_original']:.3f}",
            fontsize=10,
        )
        ax.set_xlabel(r"$\delta = \rho/\bar{\rho} - 1$")
        ax.set_ylabel("PDF")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Compare density histograms: original halos vs prepatch vs stitched"
    )
    p.add_argument(
        "--data-root",
        default="/home/juliahul/projects/stuff/universedata/lagrangian_output_64",
        help="Root of the lagrangian_output_64 tree (contains quijote-64/, stitched/)",
    )
    p.add_argument(
        "--halo-dir",
        default="/home/juliahul/projects/stuff/universedata/cmass-ili",
        help="Root of halo catalogs (contains quijote/nbody/L1000-N128/)",
    )
    p.add_argument("--sim-ids", nargs="+", type=int, default=[0, 100])
    p.add_argument(
        "--out-dir", default=None,
        help="Output directory (default: <data-root>/analysis/density_histograms)",
    )
    p.add_argument("--box-size", type=float, default=DEFAULT_BOX_SIZE)
    p.add_argument("--nmesh", type=int, default=DEFAULT_NMESH)
    p.add_argument("--redshift", default=DEFAULT_REDSHIFT_KEY)
    p.add_argument("--n-bins", type=int, default=80)
    args = p.parse_args()

    data_root = Path(args.data_root)
    halo_dir  = Path(args.halo_dir)
    out_dir   = (Path(args.out_dir) if args.out_dir
                 else data_root / "analysis" / "density_histograms")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for sim_id in args.sim_ids:
        print(f"Processing sim {sim_id} ...")
        r = process_sim(sim_id, data_root, halo_dir,
                        args.box_size, args.nmesh, args.redshift)
        results.append(r)
        m = r["metrics"]
        print(f"  n_halos={r['n_halos']:,}  patch_dirs={r['patch_dir_count']}")
        print(f"  prepatch  MSE={m['prepatch000_mse_vs_original']:.4f}  "
              f"MAE={m['prepatch000_mae_vs_original']:.4f}  "
              f"zero_vox={m['prepatch000_zero_vector_voxels']}")
        print(f"  stitched  MSE={m['stitched_mse_vs_original']:.4f}  "
              f"MAE={m['stitched_mae_vs_original']:.4f}  "
              f"zero_vox={m['stitched_zero_vector_voxels']}")

    tag = "set" + "_set".join(str(r["sim_id"]) for r in results)

    plot_histograms(results, out_dir / f"{tag}_density_histograms.png",
                   n_bins=args.n_bins)

    summary = [{k: v for k, v in r.items() if not k.startswith("_")}
               for r in results]
    json_path = out_dir / f"{tag}_density_histograms_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved {json_path}")


if __name__ == "__main__":
    main()
