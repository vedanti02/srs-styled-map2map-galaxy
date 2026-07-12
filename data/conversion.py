#!/usr/bin/env python3
"""
conversion.py
=============
Converts dark-matter halo catalogs into Lagrangian displacement/velocity fields
on overlapping voxel-grid patches.

Pipeline overview
-----------------
For each simulation (quijote + quijotelike pair):
  1. Load halo positions and velocities from HDF5.
  2. Derive voxel_side = (L^3/M)^(1/3) (mean inter-halo spacing) and
     N_global = ceil(L / voxel_side) (number of voxels per axis in the full box;
     rounding UP guarantees N_global^3 >= M so every halo can be assigned).
  3. Run assign_halos_to_global_voxels once over the full N_global^3 grid.
     Each halo is claimed by exactly one voxel; no halo is double-counted across
     overlapping patches.
  4. Slice the resulting global displacement/velocity arrays into patches of size
     at most N voxels per axis.  Patches step by stride = N - overlap voxels, so
     consecutive patches share exactly `overlap` voxels.  The last patch along
     each axis is clipped to the box boundary and may be smaller than N.
  5. Optionally stitch all patches back into the full box (--stitch).

Slurm array-job chunking
------------------------
--slurm-job-size / --slurm-job-idx partition the simulation list across Slurm array
jobs.  Job j processes simulations [j*size, j*size + size).
"""

import argparse
import json
import logging
import re
from pathlib import Path

import h5py
import numpy as np
import yaml
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# Global defaults
# ---------------------------------------------------------------------------

DEFAULT_DATA_DIR   = "/home/juliahul/projects/stuff/universedata/cmass-ili"
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_SIM_IDS    = list(range(2000))
DEFAULT_REDSHIFT   = "0.666667"
DEFAULT_N          = 64    # maximum patch size in voxels per axis
DEFAULT_OVERLAP    = 16    # overlap between adjacent patches in voxels; stride = N - overlap
DEFAULT_BOX_LENGTH = 1000.0   # Mpc/h
DEFAULT_SEED       = 42


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def halo_path(data_dir: Path, catalog: str, sim_id: int) -> Path:
    """Return the HDF5 path for catalog (quijote|quijotelike) and simulation sim_id."""
    if catalog == "quijote":
        return data_dir / "quijote" / "nbody" / "L1000-N128" / str(sim_id) / "halos.h5"
    elif catalog == "quijotelike":
        return data_dir / "quijotelike" / "fastpm" / "L1000-N128" / str(sim_id) / "halos.h5"
    else:
        raise ValueError(f"Unknown catalog type: {catalog!r}")


def config_path(data_dir: Path, catalog: str, sim_id: int) -> Path:
    """Return the YAML config path for simulation sim_id."""
    if catalog == "quijote":
        return data_dir / "quijote" / "nbody" / "L1000-N128" / str(sim_id) / "config.yaml"
    elif catalog == "quijotelike":
        return data_dir / "quijotelike" / "fastpm" / "L1000-N128" / str(sim_id) / "config.yaml"
    else:
        raise ValueError(f"Unknown catalog type: {catalog!r}")


def paired_catalogs_exist(data_dir: Path, sim_id: int) -> bool:
    """Return True only when both quijote and quijotelike halos.h5 files exist."""
    q  = halo_path(data_dir, "quijote",     sim_id)
    ql = halo_path(data_dir, "quijotelike", sim_id)
    return q.exists() and ql.exists()


def load_halos(data_dir: Path, catalog: str, sim_id: int, redshift_key: str):
    """
    Load halo positions and velocities from HDF5.

    Parameters
    ----------
    redshift_key : str  group name inside the HDF5 file, e.g. '0.666667'

    Returns
    -------
    pos : (M, 3) float32  halo positions [Mpc/h], or None if file is missing
    vel : (M, 3) float32  halo velocities [km/s], or None
    """
    h5path = halo_path(data_dir, catalog, sim_id)

    if not h5path.exists():
        return None, None

    with h5py.File(h5path, "r") as f:
        if redshift_key not in f:
            return None, None
        grp = f[redshift_key]
        pos = grp["pos"][:]
        vel = grp["vel"][:]

    return pos.astype(np.float32), vel.astype(np.float32)


def load_cosmo(data_dir: Path, catalog: str, sim_id: int):
    """
    Load cosmological parameters from the YAML config.

    Returns
    -------
    cosmo : (C,) float32  parameter vector, or None if config is missing
    """
    cfg_path = config_path(data_dir, catalog, sim_id)

    if not cfg_path.exists():
        return None

    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    return np.array(cfg["nbody"]["cosmo"], dtype=np.float32)


def save_outputs(out_dir: Path, disp: np.ndarray, vel: np.ndarray,
                 style: np.ndarray, skip_existing: bool = True,
                 counts: np.ndarray | None = None) -> bool:
    """
    Save displacement, velocity, cosmological style, and (optionally) halo-count
    arrays to disk.

    Parameters
    ----------
    out_dir : Path
    disp    : (3, Ni, Nj, Nk) float32  Lagrangian displacement [Mpc/h]
    vel     : (3, Ni, Nj, Nk) float32  halo velocity [km/s]
    style   : (C,) float32 or None     cosmological parameters
    counts  : (1, Ni, Nj, Nk) float32 or None  halos per voxel (halo_counts_field)

    Returns
    -------
    bool  True if files were written, False if skipped due to skip_existing.
    """
    if skip_existing and (out_dir / "disp.npy").exists():
        return False

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "disp.npy", disp.astype(np.float32))
    np.save(out_dir / "vel.npy",  vel.astype(np.float32))

    if style is not None:
        np.save(out_dir / "style.npy", style.astype(np.float32))

    if counts is not None:
        np.save(out_dir / "counts.npy", counts.astype(np.float32))

    return True


# ---------------------------------------------------------------------------
# Mesh construction
# ---------------------------------------------------------------------------

def build_global_voxel_grid(N_global: int, voxel_side: float,
                             box_length: float) -> np.ndarray:
    """
    Build a uniform N_global^3 voxel grid covering the periodic simulation box.

    Voxel (gx, gy, gz) has its centre at:
        x = (gx * voxel_side) mod box_length
        y = (gy * voxel_side) mod box_length
        z = (gz * voxel_side) mod box_length

    Parameters
    ----------
    N_global   : int    number of voxels per axis; typically compute_n_global(box_length, voxel_side)
    voxel_side : float  physical side length of each voxel [Mpc/h]
    box_length : float  periodic box side [Mpc/h]

    Returns
    -------
    np.ndarray, shape (N_global, N_global, N_global, 3), dtype float32

    Example
    -------
    N_global=3, voxel_side=10.0, box_length=30.0 gives centres at
    x in {0, 10, 20} along each axis, so voxel (0,1,2) → [0., 10., 20.].
    """
    coords = np.arange(N_global, dtype=np.float64) * voxel_side
    gx, gy, gz = np.meshgrid(coords, coords, coords, indexing="ij")
    vc = np.stack([gx, gy, gz], axis=-1) % box_length   # (N_global, N_global, N_global, 3)
    return vc.astype(np.float32)


def build_voxel_mesh(center: np.ndarray, voxel_side: float,
                     Ni: int, Nj: int, Nk: int,
                     box_length: float) -> np.ndarray:
    """
    Build an (Ni, Nj, Nk) voxel mesh centred at `center` (retained for reference
    and testing).

    The main pipeline uses build_global_voxel_grid + array slicing; this helper
    is superseded but kept for verification via generate_patch_center().  Ni, Nj,
    Nk may differ since the last patch along each axis can be smaller than N.

    Parameters
    ----------
    center     : (3,) float32  physical centre of the mesh [Mpc/h]
    voxel_side : float         voxel side length [Mpc/h]
    Ni, Nj, Nk : int           number of voxels along each axis
    box_length : float         periodic box side [Mpc/h]

    Returns
    -------
    np.ndarray, shape (Ni, Nj, Nk, 3), dtype float32
    """
    oi = (np.arange(Ni) - (Ni - 1) / 2.0) * voxel_side
    oj = (np.arange(Nj) - (Nj - 1) / 2.0) * voxel_side
    ok = (np.arange(Nk) - (Nk - 1) / 2.0) * voxel_side
    ox, oy, oz = np.meshgrid(oi, oj, ok, indexing="ij")

    voxel_centers = np.stack([
        (center[0] + ox) % box_length,
        (center[1] + oy) % box_length,
        (center[2] + oz) % box_length,
    ], axis=-1)

    return voxel_centers.astype(np.float32)


# ---------------------------------------------------------------------------
# Periodic distance utilities
# ---------------------------------------------------------------------------

def periodic_disp_vec(a: np.ndarray, b: np.ndarray, box_length: float) -> np.ndarray:
    """
    Minimum-image displacement vector from b to a in a periodic box.

    Parameters
    ----------
    a, b       : (..., 3)  positions [Mpc/h]
    box_length : float     periodic box side [Mpc/h]

    Returns
    -------
    np.ndarray, shape (..., 3)  displacement (a - b) wrapped to (-L/2, L/2]^3
    """
    d = a - b
    d -= box_length * np.round(d / box_length)
    return d


def periodic_dist_sq(a: np.ndarray, b: np.ndarray, box_length: float) -> np.ndarray:
    """
    Squared minimum-image distance between a and b.

    Parameters
    ----------
    a, b       : (..., 3)
    box_length : float

    Returns
    -------
    np.ndarray, shape (...)  squared distances [Mpc/h]^2
    """
    d = periodic_disp_vec(a, b, box_length)
    return np.einsum("...i,...i->...", d, d)


# ---------------------------------------------------------------------------
# Halo-to-voxel assignment
# ---------------------------------------------------------------------------

def assign_halos_to_global_voxels(
    halo_pos:  np.ndarray,   # (M, 3) halo positions [Mpc/h]
    halo_vel:  np.ndarray,   # (M, 3) halo velocities [km/s]
    global_vc: np.ndarray,   # (N_global, N_global, N_global, 3) voxel centres [Mpc/h]
    box_length: float,
    logger: logging.Logger,
    K_max: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Greedy center-out halo-to-voxel assignment over the full simulation box.

    Each of the M halos is assigned to exactly one voxel; each voxel receives at most
    one halo.  Voxels with no available halo nearby (when M < N_global^3) are
    zero-filled (zero displacement, zero velocity).

    Algorithm
    ---------
    1. Sort all N_global^3 voxels by distance from the geometric box centre
       (box_length/2, box_length/2, box_length/2).  Processing center-first ensures
       that densely-packed inner regions claim their nearest halo before edge voxels
       compete for the same halos.
    2. Build a 27-copy KD-tree: replicate all M halos across the 27 periodic image
       shifts (±L along each axis) so that the tree returns the correct nearest-image
       distance for halos near the box boundary.
    3. Batch-query the tree for each voxel's K_max nearest halo candidates in one call
       (workers=-1 uses all available CPU threads).
    4. Iterate voxels in center-out order; for each voxel, assign the nearest candidate
       halo that has not already been claimed.
    5. If any voxel exhausts its K_max candidates (rare when K_max >= 128), rebuild the
       tree once from the remaining unassigned halos and handle all such voxels.

    Parameters
    ----------
    halo_pos  : (M, 3) float32  positions in [0, box_length)^3 [Mpc/h]
    halo_vel  : (M, 3) float32  velocities [km/s]
    global_vc : (N_global, N_global, N_global, 3) float32  from build_global_voxel_grid()
    box_length : float
    logger    : logging.Logger
    K_max     : int  candidate pool size; increase to reduce fallback rebuilds at cost of memory

    Returns
    -------
    disp_field : (3, N_global, N_global, N_global) float32
        Displacement from each voxel centre to its assigned halo [Mpc/h].
        Zero for voxels with no assigned halo.
    vel_field  : (3, N_global, N_global, N_global) float32
        Velocity of each voxel's assigned halo [km/s].  Zero for unassigned voxels.

    Example
    -------
    With M=500 halos, box_length=100, N_global=8, voxel_side=12.5:
        global_vc = build_global_voxel_grid(8, 12.5, 100)   # (8,8,8,3)
        disp, vel = assign_halos_to_global_voxels(pos, vel, global_vc, 100, logger)
        # disp.shape == (3, 8, 8, 8)
    """
    N_global = global_vc.shape[0]
    n_voxels = N_global ** 3    # total voxels in the full grid
    M = len(halo_pos)           # number of halos

    vc_flat = global_vc.reshape(n_voxels, 3)   # (n_voxels, 3) flattened voxel centres

    # Sort voxels by distance from the geometric box centre.
    box_center = np.array([box_length / 2.0] * 3, dtype=np.float32)
    dsq_from_center = periodic_dist_sq(vc_flat, box_center, box_length)
    voxel_order = np.argsort(dsq_from_center)   # (n_voxels,) center-out ordering

    # 27-copy periodic image set: ±L along each axis in all 3^3 = 27 combinations.
    shifts = np.array([
        [dx, dy, dz]
        for dx in [-box_length, 0.0, box_length]
        for dy in [-box_length, 0.0, box_length]
        for dz in [-box_length, 0.0, box_length]
    ], dtype=np.float32)   # (27, 3)

    def build_tree(orig_indices):
        """Build a 27-copy KD-tree from a subset of halos identified by orig_indices.
        Returns (tree, image_to_halo) where image_to_halo[k] maps tree entry k back
        to an original halo index."""
        pos = halo_pos[orig_indices]
        rep = (pos[np.newaxis] + shifts[:, np.newaxis]).reshape(-1, 3)  # (27*len, 3)
        return cKDTree(rep), np.tile(orig_indices, 27)

    tree, image_to_halo = build_tree(np.arange(M))

    # Batch query: all n_voxels queries in one call; workers=-1 uses all CPU threads.
    k_query = min(K_max, len(image_to_halo))
    _, img_idx_all = tree.query(vc_flat, k=k_query, workers=-1)
    # img_idx_all : (n_voxels, k_query) — indices into the 27-copy array
    if k_query == 1:
        img_idx_all = img_idx_all[:, np.newaxis]
    halo_cands = image_to_halo[img_idx_all]   # (n_voxels, k_query) original halo indices

    assigned       = np.full(M, False)
    disp_flat      = np.zeros((n_voxels, 3), dtype=np.float32)   # (n_voxels, 3)
    vel_flat       = np.zeros((n_voxels, 3), dtype=np.float32)   # (n_voxels, 3)
    n_assigned     = 0
    n_zero_filled  = 0
    n_rebuilds     = 0
    needs_fallback = []

    # Greedy center-out pass (no tree queries inside this loop).
    for vi in voxel_order:
        found = False
        for ci in range(k_query):
            h = halo_cands[vi, ci]
            if not assigned[h]:
                assigned[h]    = True
                disp_flat[vi]  = periodic_disp_vec(halo_pos[h], vc_flat[vi], box_length)
                vel_flat[vi]   = halo_vel[h]
                n_assigned    += 1
                found          = True
                break
        if not found:
            needs_fallback.append(vi)

    # Fallback: voxels whose K_max nearest halos were all already claimed.
    # One tree rebuild over remaining unassigned halos covers all such voxels.
    if needs_fallback:
        fb_voxels = np.array(needs_fallback)
        unassigned_idxs = np.where(~assigned)[0]
        if len(unassigned_idxs) == 0:
            n_zero_filled += len(fb_voxels)
        else:
            fb_tree, fb_image_to_halo = build_tree(unassigned_idxs)
            n_rebuilds = 1
            k_fb = min(K_max, len(fb_image_to_halo))
            _, fb_img = fb_tree.query(vc_flat[fb_voxels], k=k_fb, workers=-1)
            if k_fb == 1:
                fb_img = fb_img[:, np.newaxis]
            fb_cands = fb_image_to_halo[fb_img]   # (len(fb_voxels), k_fb)
            for i, vi in enumerate(fb_voxels):
                found = False
                for ci in range(k_fb):
                    h = fb_cands[i, ci]
                    if not assigned[h]:
                        assigned[h]    = True
                        disp_flat[vi]  = periodic_disp_vec(halo_pos[h], vc_flat[vi], box_length)
                        vel_flat[vi]   = halo_vel[h]
                        n_assigned    += 1
                        found          = True
                        break
                if not found:
                    n_zero_filled += 1

    logger.info(
        f"    assigned={n_assigned}/{n_voxels}, zero-filled={n_zero_filled}, "
        f"tree-rebuilds={n_rebuilds}"
    )

    # Reshape (n_voxels, 3) → (3, N_global, N_global, N_global)
    disp_field = disp_flat.reshape(N_global, N_global, N_global, 3).transpose(3, 0, 1, 2)
    vel_field  = vel_flat.reshape(N_global, N_global, N_global, 3).transpose(3, 0, 1, 2)
    return disp_field, vel_field


def halo_counts_field(halo_pos: np.ndarray, voxel_side: float,
                      N_global: int, box_length: float) -> np.ndarray:
    """
    Histogram halos onto the voxel grid: counts per voxel (many-to-one).

    Unlike assign_halos_to_global_voxels (one-to-one matching, multiplicity
    discarded), every halo is counted in exactly one voxel and a voxel may hold
    any number of halos.  Uses the SAME node-centred lattice convention as
    build_global_voxel_grid: voxel i's centre is at i * voxel_side, so a halo is
    counted at its NEAREST node, i.e. voxel index round(pos / voxel_side) mod
    N_global.  (Plain floor(pos / voxel_side) would bin into the cell whose
    lower corner is the node — shifted half a voxel from the displacement
    field's catchment regions.)

    Parameters
    ----------
    halo_pos   : (M, 3) float  halo positions [Mpc/h]
    voxel_side : float         voxel side length [Mpc/h]
    N_global   : int           voxels per axis
    box_length : float         periodic box side [Mpc/h]

    Returns
    -------
    counts : (1, N_global, N_global, N_global) float32
        Halo count per voxel; counts.sum() == M.  Leading channel axis matches
        the (C, N, N, N) layout of the displacement/velocity fields so the same
        patch slicing applies unchanged.

    Example
    -------
    box_length=4, voxel_side=2, N_global=2 (nodes at {0, 2} per axis):
    a halo at x=0.9 → round(0.45)=0 → voxel 0; at x=1.1 → round(0.55)=1 →
    voxel 1; at x=3.1 → round(1.55)=2 ≡ 0 (mod 2) → wraps to voxel 0 (node 0
    at distance 0.9 through the periodic boundary is nearer than node 2 at 1.1).
    """
    idx = np.round((halo_pos % box_length) / voxel_side).astype(np.int64) % N_global
    counts = np.zeros((N_global, N_global, N_global), dtype=np.float32)
    np.add.at(counts, (idx[:, 0], idx[:, 1], idx[:, 2]), 1.0)
    return counts[np.newaxis]   # (1, N_global, N_global, N_global)


# ---------------------------------------------------------------------------
# Patch geometry helper (kept for reference and unit testing)
# ---------------------------------------------------------------------------

def generate_patch_center(
    i: int, j: int, k: int,
    Ni: int, Nj: int, Nk: int,
    stride: int, voxel_side: float, box_length: float,
) -> np.ndarray:
    """
    Physical coordinates of the geometric centre of patch (i, j, k).

    Patch (i, j, k) starts at global voxel index (i*stride, j*stride, k*stride)
    and spans (Ni, Nj, Nk) voxels (the actual per-axis patch size; the last patch
    along an axis can be smaller than the maximum patch size N — see bug 3/4 fix
    in generate_patches_for_sim).  Its geometric centre is at global voxel index
    (i*stride + (Ni-1)/2, j*stride + (Nj-1)/2, k*stride + (Nk-1)/2), which maps to
    physical position (i*stride + (Ni-1)/2) * voxel_side  mod  box_length (and
    likewise for j, k).

    Using the maximum patch size N instead of the patch's actual size for a
    clipped last patch computes the wrong point entirely (it can fall outside the
    patch and wrap around to near the box origin), so the caller must pass the
    true per-axis sizes.

    Parameters
    ----------
    i, j, k    : int    patch indices along each axis (0-based)
    Ni, Nj, Nk : int    actual patch size in voxels along each axis
    stride     : int    voxels between consecutive patch origins (= N - overlap)
    voxel_side : float  physical side length of one voxel [Mpc/h]
    box_length : float  periodic box side [Mpc/h]

    Returns
    -------
    center : (3,) float32  physical patch centre [Mpc/h]

    Note
    ----
    The main pipeline uses direct global-array slicing rather than per-patch mesh
    construction, so this function is not called in production.  It is retained for
    verification: build_voxel_mesh(generate_patch_center(i, j, k, Ni, Nj, Nk, stride,
    v, L), v, Ni, Nj, Nk, L) should recover the patch voxel centres up to
    floating-point rounding.
    """
    center = np.array([
        (i * stride + (Ni - 1) / 2.0) * voxel_side,
        (j * stride + (Nj - 1) / 2.0) * voxel_side,
        (k * stride + (Nk - 1) / 2.0) * voxel_side,
    ], dtype=np.float64) % box_length
    return center.astype(np.float32)


def compute_n_global(box_length: float, voxel_side: float) -> int:
    """
    Number of voxels per axis needed to cover the box at spacing voxel_side.

    Uses ceil, NOT round: with voxel_side = (L^3/M)^(1/3) the exact ratio
    L / voxel_side equals M^(1/3), so round() gives N_global^3 < M whenever the
    fractional part of M^(1/3) is below 0.5 — i.e. fewer voxels than halos, and
    since assign_halos_to_global_voxels grants each voxel at most one halo, some
    halos could then never be assigned.  Rounding up guarantees N_global^3 >= M.

    The small epsilon guards against floating-point noise pushing an exactly
    integer ratio (M a perfect cube) just above the integer, which would
    otherwise over-allocate one full extra layer of voxels.

    Parameters
    ----------
    box_length : float  periodic box side [Mpc/h]
    voxel_side : float  voxel side length [Mpc/h]

    Returns
    -------
    int  smallest N_global with N_global * voxel_side >= box_length (up to fp noise)

    Example
    -------
    M=800 halos, L=100:  voxel_side = (1e6/800)^(1/3) ≈ 10.772,
    L/voxel_side = 800^(1/3) ≈ 9.283 → ceil → 10 (10^3 = 1000 >= 800).
    round() would give 9 (9^3 = 729 < 800: 71 halos unassignable).
    """
    return int(np.ceil(box_length / voxel_side - 1e-9))


def compute_n_patches(N_global: int, N: int, stride: int) -> int:
    """
    Minimum number of N-voxel patches, stepped by `stride` voxels, needed to cover
    an N_global-voxel axis.

    Derivation: the last patch starts at global voxel index (n-1)*stride and must
    reach the end of the axis:  (n-1)*stride + N >= N_global  =>  n >= (N_global - N)/stride + 1.

    Parameters
    ----------
    N_global : int  size of the full axis in voxels
    N        : int  maximum patch size in voxels
    stride   : int  voxels between consecutive patch origins (= N - overlap)

    Returns
    -------
    int  minimal number of patches that fully covers the axis
    """
    if N_global <= N:
        return 1
    return int(np.ceil((N_global - N) / stride)) + 1


# ---------------------------------------------------------------------------
# Patch generation
# ---------------------------------------------------------------------------

def generate_patches_for_sim(
    data_dir: Path,
    catalog: str,
    sim_id: int,
    redshift: str,
    N: int,
    overlap: int,
    box_length: float,
    output_dir: Path,
    logger: logging.Logger,
    skip_existing: bool = True,
) -> list:
    """
    Generate all overlapping patches for one simulation catalog.

    Design
    ------
    1. Assign all halos globally (assign_halos_to_global_voxels) — each halo is
       claimed by exactly one voxel, so no halo is double-counted across patches.
    2. Slice the (3, N_global, N_global, N_global) displacement and velocity fields
       into patches and save each to disk.

    Patch layout
    ------------
    stride = N - overlap.
    Patch (i, j, k) starts at global voxel indices (i*stride, j*stride, k*stride).
    Its size along axis a (with a_idx being i, j, or k) is:
        Na = min(N, N_global - a_idx * stride)   [voxels]
    so the last patch along each axis is clipped to the box boundary, while all
    non-terminal patches have exactly N voxels and consecutive patches share exactly
    `overlap` voxels.

    Parameters
    ----------
    N       : int  maximum patch size in voxels per axis
    overlap : int  overlap between adjacent patches in voxels (stride = N - overlap)

    Returns
    -------
    list of (Path, (i, j, k))  output subdirectory and patch index for each patch

    Output layout
    -------------
    output_dir/{catalog}-{N}/set{sim_id}_pos_{i}_{j}_{k}/PART_009/
        disp.npy   (3, Ni, Nj, Nk) float32  Lagrangian displacement [Mpc/h]
        vel.npy    (3, Ni, Nj, Nk) float32  velocity [km/s]
        style.npy  (C,) float32  cosmological parameters (same for all patches)
    output_dir/{catalog}-{N}/set{sim_id}_meta.json
        N_global, N, overlap, stride, voxel_side, n_patches (read by stitch_patches)

    Example
    -------
    M=500000 halos, box_length=1000, N=64, overlap=16 (stride=48):
        voxel_side ≈ 12.6 Mpc/h, N_global ≈ 79, n_patches = 2
        Patch (0,0,0): shape (3, 64, 64, 64) — full interior patch
        Patch (1,0,0): shape (3, 31, 64, 64) — last-along-x; 79 - 1*48 = 31 voxels
    """
    stride = N - overlap

    logger.info(f"  Loading {catalog} halos for sim {sim_id}...")
    pos, vel = load_halos(data_dir, catalog, sim_id, redshift)
    cosmo    = load_cosmo(data_dir, catalog, sim_id)

    if pos is None or vel is None:
        logger.warning(f"  Missing {catalog} halos/redshift for sim {sim_id}; skipping")
        return []

    M = len(pos)
    logger.info(f"  {catalog}: {M} halos")

    # Mean inter-halo spacing: assume M halos fill a box of volume L^3 uniformly.
    voxel_side = (box_length ** 3 / M) ** (1.0 / 3.0)   # [Mpc/h]
    # N_global: voxels per axis, rounded UP so N_global^3 >= M and every halo
    # can be claimed by some voxel (see compute_n_global docstring).
    N_global = compute_n_global(box_length, voxel_side)
    logger.info(f"  voxel_side={voxel_side:.4f} Mpc/h, N_global={N_global}")

    n_patches = compute_n_patches(N_global, N, stride)
    logger.info(
        f"  Grid: {n_patches}^3 = {n_patches**3} patches "
        f"(N={N}, overlap={overlap}, stride={stride}, N_global={N_global})"
    )

    # Early exit: if all patches already exist, skip the expensive global assignment.
    if skip_existing:
        all_exist = all(
            (output_dir / f"{catalog}-{N}" / f"set{sim_id}_pos_{i}_{j}_{k}"
             / "PART_009" / "disp.npy").exists()
            for i in range(n_patches)
            for j in range(n_patches)
            for k in range(n_patches)
        )
        if all_exist:
            logger.info(f"  All {n_patches**3} patches already exist, skipping global assignment")
            return [
                (output_dir / f"{catalog}-{N}" / f"set{sim_id}_pos_{i}_{j}_{k}" / "PART_009",
                 (i, j, k))
                for i in range(n_patches)
                for j in range(n_patches)
                for k in range(n_patches)
            ]

    # --- Step 1: global halo-to-voxel assignment ---
    logger.info(f"  Building global {N_global}^3 grid and running assignment...")
    global_vc = build_global_voxel_grid(N_global, voxel_side, box_length)
    # global_vc : (N_global, N_global, N_global, 3)
    global_disp, global_vel = assign_halos_to_global_voxels(
        pos, vel, global_vc, box_length, logger
    )
    # global_disp, global_vel : (3, N_global, N_global, N_global)

    # Write per-simulation metadata so stitch_patches can read N_global and stride.
    catalog_dir = output_dir / f"{catalog}-{N}"
    catalog_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "N_global":   N_global,
        "N":          N,
        "overlap":    overlap,
        "stride":     stride,
        "voxel_side": float(voxel_side),
        "n_patches":  n_patches,
        "sim_id":     sim_id,
        "catalog":    catalog,
    }
    with open(catalog_dir / f"set{sim_id}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # --- Step 2: slice global fields into patches and save ---
    patch_dirs = []
    ch = np.arange(3)   # channel indices (x, y, z components)

    for i in range(n_patches):
        for j in range(n_patches):
            for k in range(n_patches):
                # Actual patch sizes; last patch along each axis may be < N.
                Ni = min(N, N_global - i * stride)
                Nj = min(N, N_global - j * stride)
                Nk = min(N, N_global - k * stride)

                out_subdir = (
                    output_dir
                    / f"{catalog}-{N}"
                    / f"set{sim_id}_pos_{i}_{j}_{k}"
                    / "PART_009"
                )

                if skip_existing and (out_subdir / "disp.npy").exists() and (out_subdir / "vel.npy").exists():
                    logger.info(f"  Patch ({i},{j},{k}) size=({Ni},{Nj},{Nk}) already exists, skipping")
                    patch_dirs.append((out_subdir, (i, j, k)))
                    continue

                # Global voxel index ranges with periodic wrapping for edge patches.
                xi = np.arange(i * stride, i * stride + Ni) % N_global   # (Ni,)
                yj = np.arange(j * stride, j * stride + Nj) % N_global   # (Nj,)
                zk = np.arange(k * stride, k * stride + Nk) % N_global   # (Nk,)

                # np.ix_ creates an open mesh for cross-product indexing → (3, Ni, Nj, Nk)
                patch_disp = global_disp[np.ix_(ch, xi, yj, zk)]
                patch_vel  = global_vel [np.ix_(ch, xi, yj, zk)]

                save_outputs(out_subdir, patch_disp, patch_vel, cosmo, skip_existing=False)
                logger.info(f"  Saved patch ({i},{j},{k}) size=({Ni},{Nj},{Nk}) -> {out_subdir}")
                patch_dirs.append((out_subdir, (i, j, k)))

    return patch_dirs


# ---------------------------------------------------------------------------
# Stitching
# ---------------------------------------------------------------------------

def make_weight_window(Ni: int, Nj: int, Nk: int,
                       mode: str = "hann", eps: float = 1e-3) -> np.ndarray:
    """
    Build a separable 3-D blending window of shape (1, Ni, Nj, Nk).

    The window is the outer product of three 1-D windows (one per axis).  The Hann
    window tapers each patch's contribution toward its edges, reducing boundary
    artefacts when overlapping patches are blended together.

    Parameters
    ----------
    Ni, Nj, Nk : int    patch dimensions in voxels along each axis
    mode        : str   'hann' (raised cosine) or 'uniform' (all-ones)
    eps         : float minimum weight to prevent division by zero during stitching

    Returns
    -------
    np.ndarray, shape (1, Ni, Nj, Nk), dtype float32

    Example
    -------
    make_weight_window(4, 4, 4, mode='hann')[0, :, 0, 0]
    # ≈ [0.001, 0.75, 0.75, 0.001]  (Hann values floored at eps=0.001)
    """
    def hann1d(N):
        if mode == "uniform":
            return np.ones(N, dtype=np.float32)
        elif mode == "hann":
            return np.maximum(np.hanning(N).astype(np.float32), eps)
        else:
            raise ValueError(f"Unknown weight mode: {mode!r}")

    wi = hann1d(Ni)   # (Ni,)
    wj = hann1d(Nj)   # (Nj,)
    wk = hann1d(Nk)   # (Nk,)
    w3 = wi[:, None, None] * wj[None, :, None] * wk[None, None, :]   # (Ni, Nj, Nk)
    return w3[None].astype(np.float32)   # (1, Ni, Nj, Nk)


def add_patch_periodic(
    accum:   np.ndarray,   # (C, global_N, global_N, global_N) running weighted sum
    weights: np.ndarray,   # (1, global_N, global_N, global_N) running weight sum
    patch:   np.ndarray,   # (C, Ni, Nj, Nk) patch field values
    origin:  tuple,        # (oi, oj, ok) global voxel start indices (before modulo)
    window:  np.ndarray,   # (1, Ni, Nj, Nk) per-voxel blending weights
) -> None:
    """
    Accumulate a patch into the global weighted-sum buffers with periodic wrapping.

    Parameters
    ----------
    accum   : (C, global_N, global_N, global_N)  modified in-place
    weights : (1, global_N, global_N, global_N)  modified in-place
    patch   : (C, Ni, Nj, Nk)
    origin  : (oi, oj, ok)  patch start in global voxel coordinates (oi = i*stride, etc.)
    window  : (1, Ni, Nj, Nk)  from make_weight_window()
    """
    C, global_N, _, _ = accum.shape
    _, Ni, Nj, Nk = patch.shape   # actual patch dims (may be < N for last patches)

    xs = (origin[0] + np.arange(Ni)) % global_N   # (Ni,) wrapped x-indices
    ys = (origin[1] + np.arange(Nj)) % global_N   # (Nj,)
    zs = (origin[2] + np.arange(Nk)) % global_N   # (Nk,)

    accum  [np.ix_(np.arange(C), xs, ys, zs)] += patch * window
    weights[np.ix_(np.arange(1), xs, ys, zs)] += window


def find_patch_dirs(patch_root: Path, sim_id: int) -> list:
    """
    Scan patch_root for patch subdirectories belonging to sim_id.

    Parameters
    ----------
    patch_root : Path  e.g. output_dir / "{catalog}-{N}"
    sim_id     : int

    Returns
    -------
    list of (Path, (i, j, k))  sorted by directory name; Path points to PART_009/
    """
    label_re = re.compile(r"set(?P<sim>\d+)_pos_(?P<i>\d+)_(?P<j>\d+)_(?P<k>\d+)$")
    out = []
    prefix = f"set{sim_id}_pos_"

    for p in sorted(patch_root.glob(f"{prefix}*")):
        if not p.is_dir():
            continue
        m = label_re.match(p.name)
        if m is None:
            continue
        i, j, k = int(m.group("i")), int(m.group("j")), int(m.group("k"))
        out.append((p / "PART_009", (i, j, k)))

    return out


def stitch_field(
    patch_dirs_with_origins: list,   # list of (Path, (oi, oj, ok)) with voxel origins
    field: str,
    global_N: int,
    weight_mode: str,
) -> np.ndarray:
    """
    Weighted-average stitch of overlapping patches into a full (C, global_N^3) field.

    Each patch is accumulated into the global grid weighted by a separable window
    computed from the patch's actual shape (so last-axis patches with Ni < N are
    handled correctly with a Ni-point Hann window instead of an N-point one).

    Parameters
    ----------
    patch_dirs_with_origins : list of (Path, (oi, oj, ok))
        Each tuple contains a PART_009/ directory and the patch's starting voxel
        indices in global coordinates (oi = i*stride, oj = j*stride, ok = k*stride).
    field       : str   'disp' or 'vel'
    global_N    : int   full-grid size per axis (from set{sim_id}_meta.json)
    weight_mode : str   'hann' or 'uniform'

    Returns
    -------
    np.ndarray, shape (C, global_N, global_N, global_N), dtype float32
    """
    accum   = None
    weights = np.zeros((1, global_N, global_N, global_N), dtype=np.float32)
    used    = 0

    for part_dir, origin in patch_dirs_with_origins:
        fpath = part_dir / f"{field}.npy"
        if not fpath.exists():
            continue

        patch = np.load(fpath).astype(np.float32)   # (C, Ni, Nj, Nk)
        if patch.ndim != 4:
            raise ValueError(f"Expected 4-D array in {fpath}, got shape {patch.shape}")

        _, Ni, Nj, Nk = patch.shape
        window = make_weight_window(Ni, Nj, Nk, mode=weight_mode)   # (1, Ni, Nj, Nk)

        if accum is None:
            C = patch.shape[0]
            accum = np.zeros((C, global_N, global_N, global_N), dtype=np.float32)

        add_patch_periodic(accum, weights, patch, origin, window)
        used += 1

    if accum is None or used == 0:
        raise RuntimeError(f"No patches found for field '{field}'")

    stitched = accum / np.maximum(weights, 1e-12)

    uncovered = int(np.sum(weights[0] == 0))
    if uncovered > 0:
        print(f"WARNING: {uncovered} global voxels not covered for field '{field}'.")

    print(
        f"Stitched {used} patches for '{field}'; "
        f"coverage min={weights.min():.4g}, max={weights.max():.4g}"
    )
    return stitched.astype(np.float32)


def stitch_patches(
    patch_root: Path,
    output_dir: Path,
    sim_id: int,
    catalog: str,
    N: int,
    weight_mode: str = "hann",
    logger: logging.Logger | None = None,
) -> Path:
    """
    Reconstruct the full simulation box by stitching all saved patches.

    Reads N_global and stride from set{sim_id}_meta.json (written by
    generate_patches_for_sim) so these values do not need to be re-specified.

    Parameters
    ----------
    patch_root  : Path  directory containing the '{catalog}-{N}' subdirectory
    output_dir  : Path  where to write stitched output
    sim_id      : int
    catalog     : str   'quijote' or 'quijotelike'
    N           : int   max patch size in voxels (used to locate the catalog subdir)
    weight_mode : str   'hann' or 'uniform'
    logger      : optional

    Returns
    -------
    stitched_dir : Path  contains disp.npy, vel.npy, and (optionally) style.npy
    """
    catalog_dir = patch_root / f"{catalog}-{N}"

    # Load metadata written during patch generation.
    meta_path = catalog_dir / f"set{sim_id}_meta.json"
    if not meta_path.exists():
        raise RuntimeError(f"Missing patch metadata: {meta_path}")
    with open(meta_path) as f:
        meta = json.load(f)
    global_N = meta["N_global"]   # true full-box grid size
    stride   = meta["stride"]

    patch_dirs = find_patch_dirs(catalog_dir, sim_id)
    if not patch_dirs:
        raise RuntimeError(f"No patch directories found under {catalog_dir} for sim {sim_id}")

    if logger:
        logger.info(
            f"  Found {len(patch_dirs)} patches for stitching "
            f"(global_N={global_N}, stride={stride})"
        )

    # Convert (i, j, k) patch indices to voxel-coordinate origins for add_patch_periodic.
    patch_dirs_with_origins = [
        (part_dir, (i * stride, j * stride, k * stride))
        for part_dir, (i, j, k) in patch_dirs
    ]

    stitched_dir = output_dir / "stitched" / f"set{sim_id}_{catalog}" / "PART_009"
    stitched_dir.mkdir(parents=True, exist_ok=True)

    for field in ["disp", "vel"]:
        stitched = stitch_field(patch_dirs_with_origins, field, global_N, weight_mode)
        np.save(stitched_dir / f"{field}.npy", stitched)
        if logger:
            logger.info(f"  Saved stitched {field}: {stitched.shape}")

    style_src = patch_dirs[0][0] / "style.npy"
    if style_src.exists():
        style = np.load(style_src).astype(np.float32)
        np.save(stitched_dir / "style.npy", style)
        if logger:
            logger.info(f"  Saved style: {style.shape}")

    return stitched_dir


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    stride     = args.N - args.overlap

    logger.info(f"Data dir      : {data_dir}")
    logger.info(f"Output dir    : {output_dir}")
    logger.info(f"Sim IDs       : {args.sim_ids}")
    logger.info(f"N={args.N}, overlap={args.overlap}, stride={stride}, seed={args.seed}")
    logger.info(f"Redshift key  : {args.redshift}")
    logger.info(f"Stitch        : {args.stitch}")
    logger.info(f"Skip existing : {args.skip_existing}")

    for sim_id in args.sim_ids:
        logger.info(f"=== Simulation {sim_id} ===")

        if not paired_catalogs_exist(data_dir, sim_id):
            logger.warning(
                f"Skipping sim {sim_id}: missing paired quijote/quijotelike halos.h5"
            )
            continue

        quijote_stitched_dir     = output_dir / "stitched" / f"set{sim_id}_quijote"     / "PART_009"
        quijotelike_stitched_dir = output_dir / "stitched" / f"set{sim_id}_quijotelike" / "PART_009"

        quijote_done     = (quijote_stitched_dir     / "disp.npy").exists() and (quijote_stitched_dir     / "vel.npy").exists()
        quijotelike_done = (quijotelike_stitched_dir / "disp.npy").exists() and (quijotelike_stitched_dir / "vel.npy").exists()

        if quijote_done and quijotelike_done:
            logger.info(f"Both stitched outputs exist for sim {sim_id}, skipping")
            continue

        if not quijote_done:
            logger.info(f"[Step 1] Generating quijote patches for sim {sim_id}...")
            patch_dirs_q = generate_patches_for_sim(
                data_dir=data_dir, catalog="quijote", sim_id=sim_id,
                redshift=args.redshift, N=args.N, overlap=args.overlap,
                box_length=args.box_length, output_dir=output_dir,
                logger=logger, skip_existing=args.skip_existing,
            )
            logger.info(f"  Generated/found {len(patch_dirs_q)} quijote patches")

            if args.stitch:
                logger.info("[Step 2] Stitching quijote patches...")
                try:
                    stitch_patches(output_dir, output_dir, sim_id, "quijote",
                                   args.N, args.weight_mode, logger)
                except Exception as e:
                    logger.warning(f"  Quijote stitching failed for sim {sim_id}: {e}")

        if not quijotelike_done:
            logger.info(f"[Step 3] Generating quijotelike patches for sim {sim_id}...")
            patch_dirs_ql = generate_patches_for_sim(
                data_dir=data_dir, catalog="quijotelike", sim_id=sim_id,
                redshift=args.redshift, N=args.N, overlap=args.overlap,
                box_length=args.box_length, output_dir=output_dir,
                logger=logger, skip_existing=args.skip_existing,
            )
            logger.info(f"  Generated/found {len(patch_dirs_ql)} quijotelike patches")

            if args.stitch:
                logger.info("[Step 4] Stitching quijotelike patches...")
                try:
                    stitch_patches(output_dir, output_dir, sim_id, "quijotelike",
                                   args.N, args.weight_mode, logger)
                except Exception as e:
                    logger.warning(f"  Quijotelike stitching failed for sim {sim_id}: {e}")

    logger.info("Done.")


def parse_args():
    p = argparse.ArgumentParser(
        description="Convert halo catalogs to Lagrangian grid format with stitching."
    )

    p.add_argument("--data-dir",    default=DEFAULT_DATA_DIR)
    p.add_argument("--output-dir",  default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--sim-ids",     nargs="+", type=int, default=DEFAULT_SIM_IDS)

    # Slurm array-job chunking: job j processes sims [j*size, j*size+size).
    p.add_argument("--total-sims",      type=int, default=2000)
    p.add_argument("--slurm-job-size",  type=int, default=20,
                   help="Simulations per Slurm array job (formerly --chunk-size)")
    p.add_argument("--slurm-job-idx",   type=int, default=None,
                   help="Slurm array job index (0-based); selects sim range when set")

    p.add_argument("--redshift",    default=DEFAULT_REDSHIFT)
    p.add_argument("--N",           type=int, default=DEFAULT_N,
                   help="Maximum patch size in voxels per axis (default: %(default)s)")
    p.add_argument("--overlap",     type=int, default=DEFAULT_OVERLAP,
                   help="Overlap between adjacent patches in voxels; stride = N - overlap "
                        "(default: %(default)s)")
    p.add_argument("--box-length",  type=float, default=DEFAULT_BOX_LENGTH)
    p.add_argument("--seed",        type=int, default=DEFAULT_SEED)

    p.add_argument("--skip-existing",    action="store_true", default=True,
                   help="Skip patch generation if output already exists")
    p.add_argument("--no-skip-existing", dest="skip_existing", action="store_false",
                   help="Regenerate patches even if output exists")
    p.add_argument("--stitch",           action="store_true", default=False,
                   help="Stitch patches into full box after generation")
    p.add_argument("--weight-mode",      choices=["uniform", "hann"], default="hann",
                   help="Weighting mode for stitching")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.slurm_job_idx is not None:
        start = args.slurm_job_idx * args.slurm_job_size
        end   = min(start + args.slurm_job_size, args.total_sims)
        args.sim_ids = list(range(start, end))

    main(args)
