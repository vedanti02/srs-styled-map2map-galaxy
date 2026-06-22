#!/usr/bin/env python3
"""
voxelize.py
===========
Periodic voxelization: extraction of overlapping cubic patches ("voxels") from
a periodic simulation box.

This module is extraction-only — it does not stitch/reassemble patches back
into a box (see conversion.py's stitch_field/stitch_patches for that).
"""

import numpy as np


def voxelize(volume: np.ndarray, D: int, d: int):
    """
    Extract overlapping periodic patches of size D from a cubic box of side L.

    The box has periodic boundary conditions: matter exiting one face re-enters
    the opposite face.  Patches are laid out on a regular grid with stride
    `D - d` per axis, so consecutive patches share exactly `d` voxels of
    overlap.  Patches near the box boundary wrap around (pulling data from the
    opposite face) rather than being clamped or zero-padded.

    Parameters
    ----------
    volume : np.ndarray, shape (..., L, L, L)
        Field defined on a periodic cubic box of side L per axis.  Any leading
        dimensions (e.g. a channel axis) before the trailing (L, L, L) spatial
        axes are preserved as-is in each patch.
    D : int
        Patch size in voxels, per axis.
    d : int
        Overlap between neighboring patches, in voxels (0 <= d < D).

    Returns
    -------
    patches : np.ndarray, shape (N, N, N, ..., D, D, D)
        N**3 patches of size D per axis, where N = L // (D - d).  Leading
        dims of `volume` (if any) are preserved between the (N, N, N) patch
        index and the (D, D, D) spatial axes.
    origins : np.ndarray, shape (N, N, N, 3), dtype int64
        origins[i, j, k] = (i, j, k) * (D - d): the patch's starting voxel
        index along (x, y, z), reported in the ORIGINAL [0, L) box frame (not
        the padded frame used internally for the periodic wrap).

    Raises
    ------
    ValueError  if d is not in [0, D), or the trailing 3 axes of `volume` are
                not equal (not a cubic box).
    AssertionError  if L is not evenly divisible by the stride D - d.

    Example
    -------
    L=4, D=2, d=0 (stride=2, N=2): patch (1, 0, 0) is volume[2:4, 0:2, 0:2],
    with origin (2, 0, 0) — exactly tiling the box, no wrap needed.

    L=4, D=3, d=1 (stride=2, N=2): patch (1, 0, 0) starts at origin (2, 0, 0)
    and spans voxels {2, 3, 0} along x (wrapping past the x=L boundary back to
    x=0), since the box is periodic.
    """
    if not (0 <= d < D):
        raise ValueError(f"overlap d={d} must satisfy 0 <= d < D={D}")

    *lead_shape, Lx, Ly, Lz = volume.shape
    if not (Lx == Ly == Lz):
        raise ValueError(f"volume must be cubic in its trailing 3 axes, got {(Lx, Ly, Lz)}")
    L = Lx

    stride = D - d
    assert L % stride == 0, (
        f"box size L={L} must be evenly divisible by stride D-d={stride} "
        f"(D={D}, d={d})"
    )
    N = L // stride

    # Pad the spatial axes only (leading channel dims, if any, are untouched)
    # by D on each side with periodic wraparound, so that even a patch
    # starting at the last grid position (origin (N-1)*stride) can be sliced
    # contiguously without any boundary special-casing.
    n_lead = len(lead_shape)
    pad_width = [(0, 0)] * n_lead + [(D, D)] * 3
    padded = np.pad(volume, pad_width=pad_width, mode="wrap")

    patches = np.empty((N, N, N, *lead_shape, D, D, D), dtype=volume.dtype)
    origins = np.empty((N, N, N, 3), dtype=np.int64)

    for i in range(N):
        oi = i * stride
        pi = oi + D   # index into the padded array, shifted by the D-wide pad
        for j in range(N):
            oj = j * stride
            pj = oj + D
            for k in range(N):
                ok = k * stride
                pk = ok + D
                patches[i, j, k] = padded[..., pi:pi + D, pj:pj + D, pk:pk + D]
                origins[i, j, k] = (oi, oj, ok)

    return patches, origins
