#!/usr/bin/env python3
"""
conversion_approx.py
====================
Probabilistic and approximate halo-to-voxel assignment methods, built on top
of the exact GPU baseline in conversion_gpu.py.

All methods share:
  - Identical I/O, mesh construction, stitching, and CLI infrastructure.
  - The same 27-copy periodic boundary handling (halos replicated across all
    27 image shifts before indexing).
  - The same output format: disp.npy (3,N,N,N), vel.npy (3,N,N,N), style.npy.
  - A method.json saved alongside each patch recording method + parameters.

Methods
-------
exact_gpu       Exact IndexFlatL2 on GPU. Identical to conversion_gpu.py.
                Greedy center-out assignment, one-halo-per-voxel.

ivf_gpu         FAISS IVFFlat on GPU. Approximate greedy assignment.
                Fast; small fraction of voxels get a non-nearest halo.
                Key params: --nlist (Voronoi cells), --nprobe (cells searched).

hnsw_cpu        FAISS HNSW graph index on CPU (FAISS GPU does not support HNSW).
                Approximate greedy assignment. High recall at lower nprobe cost.
                Key params: --M-hnsw (graph degree), --ef-search.

random_proj     Random-projection candidate search. Projects 27-copy halos into
                R-dim space, finds candidates there, re-ranks by true 3-D L2,
                then runs greedy center-out on the re-ranked candidates.
                Key params: --n-proj, --n-candidates, --seed.

stochastic_topk Relaxes one-halo-per-voxel. Queries k nearest halos per voxel,
                samples one via Gumbel-max over distance-weighted log-probs.
                Fully vectorised; no greedy loop. Run with multiple --seed values
                to measure assignment variance.
                Key params: --k, --sigma-scale, --seed.

soft_knn        Relaxes one-halo-per-voxel. Computes a Gaussian-kernel weighted
                average of displacement and velocity over the k nearest halos.
                Fields are smooth; small-scale power is reduced.
                Key params: --k, --sigma-scale, --kernel.

sinkhorn        Local entropy-regularised optimal transport. Divides the patch
                into spatial blocks, finds k_local nearest halos per block via
                FAISS, and runs log-domain Sinkhorn to produce a soft balanced
                assignment within each block. No external OT library required.
                Key params: --k-local, --block-side, --sinkhorn-eps,
                            --sinkhorn-iters.

Usage
-----
python conversion_approx.py \\
    --method soft_knn \\
    --data-dir /path/to/cmass-ili \\
    --output-dir /path/to/output \\
    --k 16 --sigma-scale 1.0 \\
    --stitch --chunk-idx 0 --chunk-size 10

Requires: faiss-gpu (conda install -c pytorch -c nvidia faiss-gpu)
"""

import argparse
import json
import logging
import re
from pathlib import Path

import faiss
import h5py
import numpy as np
import yaml
from scipy.spatial import cKDTree


DEFAULT_DATA_DIR = "/home/juliahul/projects/stuff/universedata/cmass-ili"
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_SIM_IDS = list(range(2000))
DEFAULT_REDSHIFT = "0.666667"
DEFAULT_N = 64
DEFAULT_STRIDE = 48
DEFAULT_BOX_LENGTH = 1000.0
DEFAULT_SEED = 42


# ---------------------------------------------------------------------------
# I/O helpers  (identical to conversion_gpu.py)
# ---------------------------------------------------------------------------

def halo_path(data_dir: Path, catalog: str, sim_id: int) -> Path:
    if catalog == "quijote":
        return data_dir / "quijote" / "nbody" / "L1000-N128" / str(sim_id) / "halos.h5"
    elif catalog == "quijotelike":
        return data_dir / "quijotelike" / "fastpm" / "L1000-N128" / str(sim_id) / "halos.h5"
    else:
        raise ValueError(f"Unknown catalog type: {catalog!r}")


def config_path(data_dir: Path, catalog: str, sim_id: int) -> Path:
    if catalog == "quijote":
        return data_dir / "quijote" / "nbody" / "L1000-N128" / str(sim_id) / "config.yaml"
    elif catalog == "quijotelike":
        return data_dir / "quijotelike" / "fastpm" / "L1000-N128" / str(sim_id) / "config.yaml"
    else:
        raise ValueError(f"Unknown catalog type: {catalog!r}")


def paired_catalogs_exist(data_dir: Path, sim_id: int) -> bool:
    q = halo_path(data_dir, "quijote", sim_id)
    ql = halo_path(data_dir, "quijotelike", sim_id)
    return q.exists() and ql.exists()


def _read_float32(dset) -> np.ndarray:
    """Low-level h5py read that handles non-standard HDF5 float types."""
    shape = dset.id.get_space().get_simple_extent_dims()
    out = np.empty(shape, dtype=np.float32)
    mem_type = h5py.h5t.py_create(np.dtype("float32"), logical=True)
    dset.id.read(h5py.h5s.ALL, h5py.h5s.ALL, out, mem_type)
    return out


def load_halos(data_dir: Path, catalog: str, sim_id: int, redshift_key: str):
    h5path = halo_path(data_dir, catalog, sim_id)
    if not h5path.exists():
        return None, None
    with h5py.File(h5path, "r") as f:
        if redshift_key not in f:
            return None, None
        grp = f[redshift_key]
        pos = _read_float32(grp["pos"])
        vel = _read_float32(grp["vel"])
    return pos, vel


def load_cosmo(data_dir: Path, catalog: str, sim_id: int):
    cfg_path = config_path(data_dir, catalog, sim_id)
    if not cfg_path.exists():
        return None
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    return np.array(cfg["nbody"]["cosmo"], dtype=np.float32)


def save_outputs(out_dir: Path, disp: np.ndarray, vel: np.ndarray,
                 style: np.ndarray, meta: dict, skip_existing: bool = True):
    if skip_existing and (out_dir / "disp.npy").exists():
        return False
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "disp.npy", disp.astype(np.float32))
    np.save(out_dir / "vel.npy", vel.astype(np.float32))
    if style is not None:
        np.save(out_dir / "style.npy", style.astype(np.float32))
    with open(out_dir / "method.json", "w") as f:
        json.dump(meta, f, indent=2)
    return True


# ---------------------------------------------------------------------------
# Mesh construction  (identical to conversion_gpu.py)
# ---------------------------------------------------------------------------

def build_voxel_mesh(center: np.ndarray, voxel_side: float, N: int,
                     box_length: float) -> np.ndarray:
    offsets_1d = (np.arange(N) - (N - 1) / 2.0) * voxel_side
    ox, oy, oz = np.meshgrid(offsets_1d, offsets_1d, offsets_1d, indexing="ij")
    voxel_centers = np.stack([
        (center[0] + ox) % box_length,
        (center[1] + oy) % box_length,
        (center[2] + oz) % box_length,
    ], axis=-1)
    return voxel_centers.astype(np.float32)


def periodic_disp_vec(a: np.ndarray, b: np.ndarray, box_length: float) -> np.ndarray:
    d = a - b
    d -= box_length * np.round(d / box_length)
    return d


def periodic_dist_sq(a: np.ndarray, b: np.ndarray, box_length: float) -> np.ndarray:
    d = periodic_disp_vec(a, b, box_length)
    return np.einsum("...i,...i->...", d, d)


# ---------------------------------------------------------------------------
# Shared 27-copy periodic image builder
# ---------------------------------------------------------------------------

def _build_27copy(halo_pos: np.ndarray, orig_indices: np.ndarray,
                  box_length: float):
    """
    Returns (rep, image_to_halo) where rep is the (27*len(orig_indices), 3)
    array of all periodic images and image_to_halo maps each row of rep back
    to an index in orig_indices (not in halo_pos directly).
    """
    shifts = np.array(
        [[dx, dy, dz]
         for dx in (-box_length, 0.0, box_length)
         for dy in (-box_length, 0.0, box_length)
         for dz in (-box_length, 0.0, box_length)],
        dtype=np.float32,
    )
    pos = halo_pos[orig_indices]
    rep = np.ascontiguousarray(
        (pos[np.newaxis] + shifts[:, np.newaxis]).reshape(-1, 3),
        dtype=np.float32,
    )
    image_to_halo = np.tile(orig_indices, 27)
    return rep, image_to_halo


# ---------------------------------------------------------------------------
# Greedy center-out assignment  (shared by exact_gpu, ivf_gpu, hnsw_cpu,
#                                 random_proj — all hard-assignment methods)
# ---------------------------------------------------------------------------

def _greedy_assign(
    halo_pos: np.ndarray,
    halo_vel: np.ndarray,
    vc_flat: np.ndarray,
    box_length: float,
    voxel_order: np.ndarray,
    img_idx_all: np.ndarray,      # (n_voxels, k_query) — indices into image_to_halo
    image_to_halo: np.ndarray,    # (27*M_orig,) — maps image idx → original halo idx
    M: int,                       # total number of halos
):
    """
    Greedy center-out assignment shared by all hard-assignment methods.
    img_idx_all must already be in distance order (closest first) per row.
    Returns (disp_flat, vel_flat, assigned, n_assigned, n_zero_filled, needs_fallback).
    """
    n_voxels, k_query = img_idx_all.shape
    halo_cands = image_to_halo[np.clip(img_idx_all, 0, len(image_to_halo) - 1)]

    assigned = np.full(M, False)
    disp_flat = np.zeros((n_voxels, 3), dtype=np.float32)
    vel_flat = np.zeros((n_voxels, 3), dtype=np.float32)
    n_assigned = 0
    n_zero_filled = 0
    needs_fallback = []

    for vi in voxel_order:
        found = False
        for ci in range(k_query):
            if img_idx_all[vi, ci] < 0:
                break
            h = halo_cands[vi, ci]
            if not assigned[h]:
                assigned[h] = True
                disp_flat[vi] = periodic_disp_vec(halo_pos[h], vc_flat[vi], box_length)
                vel_flat[vi] = halo_vel[h]
                n_assigned += 1
                found = True
                break
        if not found:
            needs_fallback.append(vi)

    return disp_flat, vel_flat, assigned, n_assigned, n_zero_filled, needs_fallback


def _fallback_assign(
    halo_pos, halo_vel, vc_flat, box_length, needs_fallback,
    assigned, disp_flat, vel_flat, n_assigned, n_zero_filled,
    M, box_length_arg, K_max, res, gpu_id,
):
    """
    One GPU index rebuild on remaining unassigned halos to cover any voxels
    that exhausted all K_max candidates in the main pass.
    """
    n_rebuilds = 0
    if not needs_fallback:
        return n_assigned, n_zero_filled, n_rebuilds

    fb_voxels = np.array(needs_fallback)
    unassigned_idxs = np.where(~assigned)[0]
    if len(unassigned_idxs) == 0:
        n_zero_filled += len(fb_voxels)
        return n_assigned, n_zero_filled, n_rebuilds

    fb_rep, fb_image_to_halo = _build_27copy(halo_pos, unassigned_idxs, box_length)
    fb_index = faiss.index_cpu_to_gpu(res, gpu_id, faiss.IndexFlatL2(3))
    fb_index.add(fb_rep)
    n_rebuilds = 1

    k_fb = min(K_max, len(fb_image_to_halo))
    _, fb_img = fb_index.search(np.ascontiguousarray(vc_flat[fb_voxels]), k_fb)
    fb_cands = fb_image_to_halo[np.clip(fb_img, 0, len(fb_image_to_halo) - 1)]

    for i, vi in enumerate(fb_voxels):
        found = False
        for ci in range(k_fb):
            if fb_img[i, ci] < 0:
                break
            h = fb_cands[i, ci]
            if not assigned[h]:
                assigned[h] = True
                disp_flat[vi] = periodic_disp_vec(halo_pos[h], vc_flat[vi], box_length)
                vel_flat[vi] = halo_vel[h]
                n_assigned += 1
                found = True
                break
        if not found:
            n_zero_filled += 1

    return n_assigned, n_zero_filled, n_rebuilds


# ---------------------------------------------------------------------------
# Method 0 — exact_gpu
# ---------------------------------------------------------------------------

def assign_exact_gpu(
    halo_pos, halo_vel, voxel_centers, box_length, logger,
    *, K_max=128, gpu_id=0,
):
    """Exact IndexFlatL2 on GPU. Identical semantics to conversion_gpu.py."""
    N = voxel_centers.shape[0]
    n_voxels = N ** 3
    M = len(halo_pos)
    vc_flat = voxel_centers.reshape(n_voxels, 3)
    voxel_order = np.argsort(periodic_dist_sq(
        vc_flat, voxel_centers[N // 2, N // 2, N // 2], box_length))

    res = faiss.StandardGpuResources()
    rep, image_to_halo = _build_27copy(halo_pos, np.arange(M), box_length)
    index = faiss.index_cpu_to_gpu(res, gpu_id, faiss.IndexFlatL2(3))
    index.add(rep)

    k_query = min(K_max, len(image_to_halo))
    _, img_idx_all = index.search(np.ascontiguousarray(vc_flat), k_query)

    disp_flat, vel_flat, assigned, n_assigned, n_zero_filled, needs_fallback = \
        _greedy_assign(halo_pos, halo_vel, vc_flat, box_length,
                       voxel_order, img_idx_all, image_to_halo, M)

    n_assigned, n_zero_filled, n_rebuilds = _fallback_assign(
        halo_pos, halo_vel, vc_flat, box_length, needs_fallback,
        assigned, disp_flat, vel_flat, n_assigned, n_zero_filled,
        M, box_length, K_max, res, gpu_id,
    )

    logger.info(f"    exact_gpu: assigned={n_assigned}/{n_voxels}, "
                f"zero-filled={n_zero_filled}, rebuilds={n_rebuilds}")
    return (disp_flat.reshape(N, N, N, 3).transpose(3, 0, 1, 2),
            vel_flat.reshape(N, N, N, 3).transpose(3, 0, 1, 2),
            {"method": "exact_gpu", "K_max": K_max, "gpu_id": gpu_id,
             "n_assigned": n_assigned, "n_zero_filled": n_zero_filled})


# ---------------------------------------------------------------------------
# Method 1 — ivf_gpu
# ---------------------------------------------------------------------------

def assign_ivf_gpu(
    halo_pos, halo_vel, voxel_centers, box_length, logger,
    *, K_max=128, nlist=512, nprobe=64, gpu_id=0,
):
    """
    FAISS IVFFlat on GPU.

    Partitions the 27-copy index into nlist Voronoi cells via k-means, then
    searches only nprobe cells per query.  Greedy center-out assignment is
    applied to the returned candidates exactly as in exact_gpu.

    nlist: number of Voronoi cells (coarser → faster, lower recall).
    nprobe: cells searched per query (higher → slower, higher recall).
    Typical operating point for 3D uniform data: nlist=512, nprobe=64 gives
    >99% recall@1 while being ~4–8× faster than IndexFlatL2 at query time.
    """
    N = voxel_centers.shape[0]
    n_voxels = N ** 3
    M = len(halo_pos)
    vc_flat = voxel_centers.reshape(n_voxels, 3)
    voxel_order = np.argsort(periodic_dist_sq(
        vc_flat, voxel_centers[N // 2, N // 2, N // 2], box_length))

    res = faiss.StandardGpuResources()
    rep, image_to_halo = _build_27copy(halo_pos, np.arange(M), box_length)

    nlist_actual = min(nlist, len(rep) // 10)  # need ≥10 points per centroid
    quantizer = faiss.IndexFlatL2(3)
    index_cpu = faiss.IndexIVFFlat(quantizer, 3, nlist_actual, faiss.METRIC_L2)
    index_cpu.train(rep)
    index_cpu.add(rep)
    index_cpu.nprobe = nprobe
    index = faiss.index_cpu_to_gpu(res, gpu_id, index_cpu)

    k_query = min(K_max, len(image_to_halo))
    _, img_idx_all = index.search(np.ascontiguousarray(vc_flat), k_query)

    disp_flat, vel_flat, assigned, n_assigned, n_zero_filled, needs_fallback = \
        _greedy_assign(halo_pos, halo_vel, vc_flat, box_length,
                       voxel_order, img_idx_all, image_to_halo, M)

    n_assigned, n_zero_filled, n_rebuilds = _fallback_assign(
        halo_pos, halo_vel, vc_flat, box_length, needs_fallback,
        assigned, disp_flat, vel_flat, n_assigned, n_zero_filled,
        M, box_length, K_max, res, gpu_id,
    )

    logger.info(f"    ivf_gpu: assigned={n_assigned}/{n_voxels}, "
                f"zero-filled={n_zero_filled}, rebuilds={n_rebuilds}, "
                f"nlist={nlist_actual}, nprobe={nprobe}")
    return (disp_flat.reshape(N, N, N, 3).transpose(3, 0, 1, 2),
            vel_flat.reshape(N, N, N, 3).transpose(3, 0, 1, 2),
            {"method": "ivf_gpu", "K_max": K_max, "nlist": nlist_actual,
             "nprobe": nprobe, "n_assigned": n_assigned, "n_zero_filled": n_zero_filled})


# ---------------------------------------------------------------------------
# Method 2 — hnsw_cpu
# ---------------------------------------------------------------------------

def assign_hnsw_cpu(
    halo_pos, halo_vel, voxel_centers, box_length, logger,
    *, K_max=128, M_hnsw=32, ef_construction=200, ef_search=128,
):
    """
    FAISS HNSW on CPU.

    FAISS GPU does not support HNSW, so this runs entirely on CPU.  The
    navigable small-world graph gives O(log N) query time and high recall
    without needing nprobe tuning.  Build cost is O(N * M_hnsw * log N)
    — expect 30–90 s for the 27-copy index at M~120K.

    M_hnsw: graph degree. 32 is a good default; higher → better recall, more
            memory and build time.
    ef_construction: beam width at build time. Higher → better graph quality.
    ef_search: beam width at query time. Higher → better recall, slower queries.
    """
    N = voxel_centers.shape[0]
    n_voxels = N ** 3
    M = len(halo_pos)
    vc_flat = voxel_centers.reshape(n_voxels, 3)
    voxel_order = np.argsort(periodic_dist_sq(
        vc_flat, voxel_centers[N // 2, N // 2, N // 2], box_length))

    rep, image_to_halo = _build_27copy(halo_pos, np.arange(M), box_length)

    index = faiss.IndexHNSWFlat(3, M_hnsw)
    index.hnsw.efConstruction = ef_construction
    index.add(rep)
    index.hnsw.efSearch = ef_search

    k_query = min(K_max, len(image_to_halo))
    _, img_idx_all = index.search(np.ascontiguousarray(vc_flat), k_query)

    # Use res=None, gpu_id=0 placeholder for fallback — fallback uses GPU exact
    res = faiss.StandardGpuResources()
    disp_flat, vel_flat, assigned, n_assigned, n_zero_filled, needs_fallback = \
        _greedy_assign(halo_pos, halo_vel, vc_flat, box_length,
                       voxel_order, img_idx_all, image_to_halo, M)

    n_assigned, n_zero_filled, n_rebuilds = _fallback_assign(
        halo_pos, halo_vel, vc_flat, box_length, needs_fallback,
        assigned, disp_flat, vel_flat, n_assigned, n_zero_filled,
        M, box_length, K_max, res, 0,
    )

    logger.info(f"    hnsw_cpu: assigned={n_assigned}/{n_voxels}, "
                f"zero-filled={n_zero_filled}, rebuilds={n_rebuilds}, "
                f"M_hnsw={M_hnsw}, ef_search={ef_search}")
    return (disp_flat.reshape(N, N, N, 3).transpose(3, 0, 1, 2),
            vel_flat.reshape(N, N, N, 3).transpose(3, 0, 1, 2),
            {"method": "hnsw_cpu", "K_max": K_max, "M_hnsw": M_hnsw,
             "ef_construction": ef_construction, "ef_search": ef_search,
             "n_assigned": n_assigned, "n_zero_filled": n_zero_filled})


# ---------------------------------------------------------------------------
# Method 3 — random_proj
# ---------------------------------------------------------------------------

def assign_random_proj(
    halo_pos, halo_vel, voxel_centers, box_length, logger,
    *, K_max=128, n_proj=16, n_candidates=256, seed=42,
):
    """
    Random-projection candidate search.

    Projects all 27-copy halo positions into an n_proj-dimensional space using
    random unit vectors, builds a KD-tree in that space, retrieves n_candidates
    halos per voxel from projected-space proximity, re-ranks them by true 3-D
    periodic L2 distance, and applies the standard greedy center-out assignment
    on the resulting K_max top candidates.

    This is cheaper to build than HNSW for small n_proj but gives lower recall
    than IVF or HNSW.  Useful as a control ablation since the approximation
    error is purely from discarding candidates outside the projected neighbourhood.

    n_proj: projection dimensionality. Higher → better recall, slower KD-tree.
    n_candidates: how many candidates to retrieve in projected space before
                  re-ranking. Must be >= K_max.
    seed: controls the random projection matrix.
    """
    N = voxel_centers.shape[0]
    n_voxels = N ** 3
    M = len(halo_pos)
    vc_flat = voxel_centers.reshape(n_voxels, 3)
    voxel_order = np.argsort(periodic_dist_sq(
        vc_flat, voxel_centers[N // 2, N // 2, N // 2], box_length))

    rep, image_to_halo = _build_27copy(halo_pos, np.arange(M), box_length)

    rng = np.random.default_rng(seed)
    P = rng.standard_normal((3, n_proj)).astype(np.float32)
    P /= np.linalg.norm(P, axis=0, keepdims=True)  # unit columns

    rep_proj = rep @ P          # (27M, n_proj)
    vc_proj = vc_flat @ P       # (n_voxels, n_proj)

    proj_tree = cKDTree(rep_proj)
    n_cand_actual = min(n_candidates, len(rep))
    _, proj_img_idx = proj_tree.query(vc_proj, k=n_cand_actual, workers=-1)
    # proj_img_idx: (n_voxels, n_cand_actual) — indices into rep ordered by projected dist

    # Re-rank by true 3-D periodic L2 in chunks to keep memory bounded (~6 MB/chunk).
    CHUNK = 4096
    k_keep = min(K_max, n_cand_actual)
    img_idx_all = np.empty((n_voxels, k_keep), dtype=np.int64)

    for start in range(0, n_voxels, CHUNK):
        end = min(start + CHUNK, n_voxels)
        cand_pos = rep[proj_img_idx[start:end]]          # (chunk, n_cand, 3)
        chunk_vc = vc_flat[start:end, np.newaxis, :]     # (chunk, 1, 3)
        d = cand_pos - chunk_vc
        d -= box_length * np.round(d / box_length)
        d_sq = (d ** 2).sum(axis=-1)                     # (chunk, n_cand)
        top_k = np.argpartition(d_sq, k_keep - 1, axis=1)[:, :k_keep]
        # sort the top-k by distance
        chunk_len = end - start
        row_idx = np.arange(chunk_len)[:, None]
        top_k_sorted = top_k[row_idx, np.argsort(d_sq[row_idx, top_k], axis=1)]
        img_idx_all[start:end] = proj_img_idx[start:end][row_idx, top_k_sorted]

    res = faiss.StandardGpuResources()
    disp_flat, vel_flat, assigned, n_assigned, n_zero_filled, needs_fallback = \
        _greedy_assign(halo_pos, halo_vel, vc_flat, box_length,
                       voxel_order, img_idx_all, image_to_halo, M)

    n_assigned, n_zero_filled, n_rebuilds = _fallback_assign(
        halo_pos, halo_vel, vc_flat, box_length, needs_fallback,
        assigned, disp_flat, vel_flat, n_assigned, n_zero_filled,
        M, box_length, K_max, res, 0,
    )

    logger.info(f"    random_proj: assigned={n_assigned}/{n_voxels}, "
                f"zero-filled={n_zero_filled}, rebuilds={n_rebuilds}, "
                f"n_proj={n_proj}, n_candidates={n_cand_actual}")
    return (disp_flat.reshape(N, N, N, 3).transpose(3, 0, 1, 2),
            vel_flat.reshape(N, N, N, 3).transpose(3, 0, 1, 2),
            {"method": "random_proj", "K_max": K_max, "n_proj": n_proj,
             "n_candidates": n_cand_actual, "seed": seed,
             "n_assigned": n_assigned, "n_zero_filled": n_zero_filled})


# ---------------------------------------------------------------------------
# Method 4 — stochastic_topk
# ---------------------------------------------------------------------------

def assign_stochastic_topk(
    halo_pos, halo_vel, voxel_centers, box_length, logger,
    *, k=16, sigma_scale=1.0, seed=42, gpu_id=0,
):
    """
    Stochastic top-k assignment. Relaxes the one-halo-per-voxel constraint.

    Queries the k nearest halos for every voxel simultaneously (one GPU call),
    computes a Gaussian log-weight over distance, adds Gumbel noise (Gumbel-max
    trick), and picks the argmax as the assigned halo.  Each voxel independently
    samples one halo; the same halo can be used by multiple voxels.

    Because there is no greedy sequential loop, this is fully vectorised and
    ~100–500× faster than the exact baseline per patch.  Run with multiple
    --seed values to measure assignment variance across stochastic realisations.

    k: candidate pool size.  Larger k → better coverage of halos far from the
       voxel but rarely sampled.  Typical: 16–64.
    sigma_scale: bandwidth multiplier.  sigma = (L^3/M)^(1/3) * sigma_scale.
                 sigma_scale=1 sets sigma to the mean inter-halo spacing.
    seed: controls Gumbel noise.  Different seeds → different fields.
    """
    N = voxel_centers.shape[0]
    n_voxels = N ** 3
    M = len(halo_pos)
    vc_flat = voxel_centers.reshape(n_voxels, 3)

    sigma = (box_length ** 3 / M) ** (1.0 / 3.0) * sigma_scale

    res = faiss.StandardGpuResources()
    rep, image_to_halo = _build_27copy(halo_pos, np.arange(M), box_length)
    index = faiss.index_cpu_to_gpu(res, gpu_id, faiss.IndexFlatL2(3))
    index.add(rep)

    k_actual = min(k, len(image_to_halo))
    dists_sq, img_idx = index.search(np.ascontiguousarray(vc_flat), k_actual)
    # dists_sq: (n_voxels, k_actual) — squared L2 in flat (non-periodic) space.
    # Because of the 27-copy, the returned distance is already the periodic
    # minimum-image distance squared for halos close to the voxel.

    halo_idx = image_to_halo[np.clip(img_idx, 0, len(image_to_halo) - 1)]

    # Gaussian log-weights
    log_w = -dists_sq / (2.0 * sigma ** 2)

    # Gumbel-max sampling: argmax(log_w + Gumbel(0,1)) ~ categorical(softmax(log_w))
    rng = np.random.default_rng(seed)
    u = rng.uniform(size=log_w.shape).astype(np.float32)
    gumbel = -np.log(-np.log(u + 1e-20) + 1e-20)
    chosen_k = np.argmax(log_w + gumbel, axis=1)          # (n_voxels,)

    chosen_halo = halo_idx[np.arange(n_voxels), chosen_k]
    disp_flat = periodic_disp_vec(halo_pos[chosen_halo], vc_flat, box_length)
    vel_flat = halo_vel[chosen_halo]

    disp_field = disp_flat.reshape(N, N, N, 3).transpose(3, 0, 1, 2)
    vel_field = vel_flat.reshape(N, N, N, 3).transpose(3, 0, 1, 2)

    logger.info(f"    stochastic_topk: k={k_actual}, sigma={sigma:.3f}, seed={seed}")
    return (disp_field, vel_field,
            {"method": "stochastic_topk", "k": k_actual, "sigma": float(sigma),
             "sigma_scale": sigma_scale, "seed": seed,
             "one_halo_per_voxel": False, "is_soft": False})


# ---------------------------------------------------------------------------
# Method 5 — soft_knn
# ---------------------------------------------------------------------------

def assign_soft_knn(
    halo_pos, halo_vel, voxel_centers, box_length, logger,
    *, k=16, sigma_scale=1.0, kernel="gaussian", gpu_id=0,
):
    """
    Soft k-NN assignment. Relaxes the one-halo-per-voxel constraint.

    Queries the k nearest halos per voxel and computes the output displacement
    and velocity as a distance-weighted average using a Gaussian (or top-hat)
    kernel.  Fully vectorised — no sequential loop, no assignment tracking.

    The resulting fields are smooth approximations of the Lagrangian displacement
    field.  Small-scale power is suppressed relative to the exact assignment at
    wavenumbers k > 2π/sigma.  Useful as a fast, deterministic preprocessing
    variant when the downstream model is expected to be robust to smoothing.

    k: number of neighbours to average over.
    sigma_scale: bandwidth multiplier. sigma = (L^3/M)^(1/3) * sigma_scale.
    kernel: 'gaussian' (softmax over -d^2/2sigma^2) or 'tophat' (uniform
            average over the k nearest halos regardless of distance).
    """
    N = voxel_centers.shape[0]
    n_voxels = N ** 3
    M = len(halo_pos)
    vc_flat = voxel_centers.reshape(n_voxels, 3)

    sigma = (box_length ** 3 / M) ** (1.0 / 3.0) * sigma_scale

    res = faiss.StandardGpuResources()
    rep, image_to_halo = _build_27copy(halo_pos, np.arange(M), box_length)
    index = faiss.index_cpu_to_gpu(res, gpu_id, faiss.IndexFlatL2(3))
    index.add(rep)

    k_actual = min(k, len(image_to_halo))
    dists_sq, img_idx = index.search(np.ascontiguousarray(vc_flat), k_actual)
    halo_idx = image_to_halo[np.clip(img_idx, 0, len(image_to_halo) - 1)]
    # halo_idx: (n_voxels, k_actual) — original halo indices for each candidate

    # Distance-based weights
    if kernel == "gaussian":
        log_w = -dists_sq / (2.0 * sigma ** 2)
        log_w -= log_w.max(axis=1, keepdims=True)   # numerical stability
        w = np.exp(log_w)
    elif kernel == "tophat":
        w = np.ones_like(dists_sq)
    else:
        raise ValueError(f"Unknown kernel: {kernel!r}. Choose 'gaussian' or 'tophat'.")
    w /= w.sum(axis=1, keepdims=True)               # (n_voxels, k_actual)

    # Periodic displacement from voxel to each candidate halo: (n_voxels, k, 3)
    cand_pos = halo_pos[halo_idx]                   # (n_voxels, k, 3)
    cand_disp = periodic_disp_vec(cand_pos, vc_flat[:, np.newaxis, :], box_length)
    cand_vel = halo_vel[halo_idx]                   # (n_voxels, k, 3)

    disp_flat = (w[:, :, np.newaxis] * cand_disp).sum(axis=1)   # (n_voxels, 3)
    vel_flat = (w[:, :, np.newaxis] * cand_vel).sum(axis=1)     # (n_voxels, 3)

    disp_field = disp_flat.reshape(N, N, N, 3).transpose(3, 0, 1, 2)
    vel_field = vel_flat.reshape(N, N, N, 3).transpose(3, 0, 1, 2)

    logger.info(f"    soft_knn: k={k_actual}, sigma={sigma:.3f}, kernel={kernel}")
    return (disp_field, vel_field,
            {"method": "soft_knn", "k": k_actual, "sigma": float(sigma),
             "sigma_scale": sigma_scale, "kernel": kernel,
             "one_halo_per_voxel": False, "is_soft": True})


# ---------------------------------------------------------------------------
# Method 6 — sinkhorn
# ---------------------------------------------------------------------------

def _sinkhorn_log(
    a: np.ndarray,    # (n_vox,) voxel marginal
    b: np.ndarray,    # (n_halo,) halo marginal
    C: np.ndarray,    # (n_vox, n_halo) cost matrix (squared distances)
    epsilon: float,
    n_iters: int,
) -> np.ndarray:
    """
    Log-domain Sinkhorn. Returns the (n_vox, n_halo) transport plan T.

    Solves:  min_T <C, T> - epsilon * H(T)
             s.t.  T 1 = a,  T^T 1 = b,  T >= 0

    where H(T) = -sum_{ij} T_{ij} log T_{ij} is the transport entropy.
    Log-domain formulation avoids underflow for small epsilon.
    """
    log_K = -C / epsilon          # (n_vox, n_halo) log of Gibbs kernel
    log_a = np.log(a + 1e-20)
    log_b = np.log(b + 1e-20)
    u = np.zeros(len(a), dtype=np.float64)
    v = np.zeros(len(b), dtype=np.float64)

    for _ in range(n_iters):
        # u_i = log a_i - log sum_j exp(log_K_ij + v_j)
        lse_v = (log_K + v[np.newaxis, :]).max(axis=1) + \
                np.log(np.exp(log_K + v[np.newaxis, :] -
                              (log_K + v[np.newaxis, :]).max(axis=1, keepdims=True)
                              ).sum(axis=1) + 1e-20)
        u = log_a - lse_v
        # v_j = log b_j - log sum_i exp(log_K_ij + u_i)
        lse_u = (log_K + u[:, np.newaxis]).max(axis=0) + \
                np.log(np.exp(log_K + u[:, np.newaxis] -
                              (log_K + u[:, np.newaxis]).max(axis=0, keepdims=True)
                              ).sum(axis=0) + 1e-20)
        v = log_b - lse_u

    log_T = log_K + u[:, np.newaxis] + v[np.newaxis, :]
    return np.exp(log_T).astype(np.float32)


def assign_sinkhorn(
    halo_pos, halo_vel, voxel_centers, box_length, logger,
    *, k_local=512, block_side=8, epsilon=0.05, n_iters=50, gpu_id=0,
):
    """
    Local entropy-regularised optimal transport (Sinkhorn) assignment.

    Divides the N^3 patch into (N/block_side)^3 non-overlapping spatial blocks.
    For each block, uses the FAISS GPU index to find the k_local nearest halos
    to the block centre, then runs log-domain Sinkhorn OT on the local
    (n_vox_block × k_local) cost matrix.  The resulting transport plan T[i,j]
    gives the fractional contribution of halo j to voxel i within the block.
    The output displacement and velocity are the T-weighted sums:

        disp[i] = sum_j T_norm[i,j] * periodic_disp_vec(halo_pos[j], vc[i], L)
        vel[i]  = sum_j T_norm[i,j] * halo_vel[j]

    where T_norm[i,:] = T[i,:] / T[i,:].sum() (row-normalised transport plan).

    This enforces approximate balance — each halo contributes to multiple
    nearby voxels in proportion to inverse distance, with the entropy term
    controlling the sharpness of the assignment.  Unlike soft_knn it respects
    the halo budget within each block.

    block_side: voxels per block edge.  N must be divisible by block_side.
                Default 8 → 8^3=512 voxels per block for N=64.
    k_local: halos retrieved per block via FAISS.  Should be 2–4× block_side^3
             to ensure full coverage.
    epsilon: entropy regularisation.  Smaller → sharper (approaches hard
             assignment); larger → smoother (approaches uniform average).
             Values around 0.05–0.5 times median cost work well in practice.
    n_iters: Sinkhorn iterations.  50 is sufficient for epsilon ≥ 0.01.
    """
    N = voxel_centers.shape[0]
    if N % block_side != 0:
        raise ValueError(f"N={N} must be divisible by block_side={block_side}")

    n_voxels = N ** 3
    M = len(halo_pos)
    vc_flat = voxel_centers.reshape(n_voxels, 3)

    # Build one exact GPU index for block-centre queries
    res = faiss.StandardGpuResources()
    rep, image_to_halo = _build_27copy(halo_pos, np.arange(M), box_length)
    index = faiss.index_cpu_to_gpu(res, gpu_id, faiss.IndexFlatL2(3))
    index.add(rep)

    n_blocks_per_dim = N // block_side
    n_blocks_total = n_blocks_per_dim ** 3
    n_vox_per_block = block_side ** 3

    # Precompute voxel grid indices for fast block extraction
    # Reshape vc_flat into (n_blocks_per_dim, block_side, ...) along each axis
    # voxel_centers shape: (N, N, N, 3) with ij-indexing
    vc_grid = voxel_centers  # (N, N, N, 3)

    disp_flat = np.zeros((n_voxels, 3), dtype=np.float32)
    vel_flat = np.zeros((n_voxels, 3), dtype=np.float32)

    k_local_actual = min(k_local, len(image_to_halo))

    # Compute block centres for batch FAISS query
    block_centres = np.zeros((n_blocks_total, 3), dtype=np.float32)
    block_idx_map = []  # each entry: (flat_voxel_indices,)

    bi = 0
    for bx in range(n_blocks_per_dim):
        for by in range(n_blocks_per_dim):
            for bz in range(n_blocks_per_dim):
                # Voxel index ranges for this block
                xs = slice(bx * block_side, (bx + 1) * block_side)
                ys = slice(by * block_side, (by + 1) * block_side)
                zs = slice(bz * block_side, (bz + 1) * block_side)

                block_vox = vc_grid[xs, ys, zs, :]  # (bs, bs, bs, 3)
                block_flat_vox = block_vox.reshape(-1, 3)  # (n_vox_per_block, 3)

                # Centre of the block in physical coordinates
                block_centres[bi] = block_flat_vox.mean(axis=0)

                # Global flat voxel indices belonging to this block
                xi = np.arange(bx * block_side, (bx + 1) * block_side)
                yi = np.arange(by * block_side, (by + 1) * block_side)
                zi = np.arange(bz * block_side, (bz + 1) * block_side)
                gxi, gyi, gzi = np.meshgrid(xi, yi, zi, indexing="ij")
                global_flat = (gxi * N * N + gyi * N + gzi).ravel()

                block_idx_map.append((global_flat, block_flat_vox))
                bi += 1

    # Batch-query k_local halos per block centre
    _, block_img_idx = index.search(
        np.ascontiguousarray(block_centres), k_local_actual
    )  # (n_blocks, k_local)

    n_blocks_done = 0
    for bi in range(n_blocks_total):
        global_flat, block_vc = block_idx_map[bi]
        img_idx_block = block_img_idx[bi]                    # (k_local,)
        valid = img_idx_block >= 0
        img_idx_block = img_idx_block[valid]
        if len(img_idx_block) == 0:
            continue

        halo_idx_block = image_to_halo[img_idx_block]       # (k_local_valid,)
        # Deduplicate: same original halo can appear via different periodic images
        halo_idx_block = np.unique(halo_idx_block)

        h_pos = halo_pos[halo_idx_block]                    # (n_h, 3)
        h_vel = halo_vel[halo_idx_block]                    # (n_h, 3)
        n_h = len(halo_idx_block)

        # Cost matrix: squared periodic distances, shape (n_vox_per_block, n_h)
        d = periodic_disp_vec(
            h_pos[np.newaxis, :, :],    # (1, n_h, 3)
            block_vc[:, np.newaxis, :], # (n_vox, 1, 3)
            box_length,
        )  # (n_vox, n_h, 3)
        C = (d ** 2).sum(axis=-1).astype(np.float64)        # (n_vox, n_h)

        # Uniform marginals
        a = np.ones(n_vox_per_block, dtype=np.float64) / n_vox_per_block
        b = np.ones(n_h, dtype=np.float64) / n_h

        # Scale epsilon by median cost for invariance to physical units
        eps_scaled = epsilon * float(np.median(C)) + 1e-12
        T = _sinkhorn_log(a, b, C, eps_scaled, n_iters)     # (n_vox, n_h)

        # Row-normalise: each voxel's fractional weights sum to 1
        T_norm = T / (T.sum(axis=1, keepdims=True) + 1e-20)

        # Weighted displacement and velocity
        disp_local = periodic_disp_vec(
            h_pos[np.newaxis, :, :],
            block_vc[:, np.newaxis, :],
            box_length,
        )  # (n_vox, n_h, 3)
        disp_flat[global_flat] = (T_norm[:, :, np.newaxis] * disp_local).sum(axis=1)
        vel_flat[global_flat] = T_norm @ h_vel               # (n_vox, 3)

        n_blocks_done += 1

    disp_field = disp_flat.reshape(N, N, N, 3).transpose(3, 0, 1, 2)
    vel_field = vel_flat.reshape(N, N, N, 3).transpose(3, 0, 1, 2)

    logger.info(f"    sinkhorn: blocks={n_blocks_done}/{n_blocks_total}, "
                f"block_side={block_side}, k_local={k_local_actual}, "
                f"epsilon={epsilon}, n_iters={n_iters}")
    return (disp_field, vel_field,
            {"method": "sinkhorn", "k_local": k_local_actual,
             "block_side": block_side, "epsilon": epsilon, "n_iters": n_iters,
             "one_halo_per_voxel": False, "is_soft": True})


# ---------------------------------------------------------------------------
# Method registry and dispatcher
# ---------------------------------------------------------------------------

_METHODS = {
    "exact_gpu":       assign_exact_gpu,
    "ivf_gpu":         assign_ivf_gpu,
    "hnsw_cpu":        assign_hnsw_cpu,
    "random_proj":     assign_random_proj,
    "stochastic_topk": assign_stochastic_topk,
    "soft_knn":        assign_soft_knn,
    "sinkhorn":        assign_sinkhorn,
}


def dispatch_assign(method: str, halo_pos, halo_vel, voxel_centers,
                    box_length, logger, cfg: dict):
    """Call the appropriate assignment method using only the kwargs it accepts."""
    import inspect
    fn = _METHODS[method]
    sig = inspect.signature(fn)
    valid = {p for p in sig.parameters if p not in
             ("halo_pos", "halo_vel", "voxel_centers", "box_length", "logger")}
    kwargs = {k: v for k, v in cfg.items() if k in valid}
    return fn(halo_pos, halo_vel, voxel_centers, box_length, logger, **kwargs)


# ---------------------------------------------------------------------------
# Patch generation
# ---------------------------------------------------------------------------

def generate_patch_center(i, j, k, stride, voxel_side, box_length):
    origin = np.array([i * stride, j * stride, k * stride], dtype=np.float32)
    return (origin * voxel_side) % box_length


def generate_patches_for_sim(
    data_dir, catalog, sim_id, redshift, N, stride, box_length,
    output_dir, logger, method, assign_cfg, skip_existing=True,
):
    logger.info(f"  Loading {catalog} halos for sim {sim_id}...")
    pos, vel = load_halos(data_dir, catalog, sim_id, redshift)
    cosmo = load_cosmo(data_dir, catalog, sim_id)

    if pos is None or vel is None:
        logger.warning(f"  Missing {catalog} halos/redshift for sim {sim_id}; skipping")
        return []

    logger.info(f"  {catalog}: {len(pos)} halos")
    n_halos = len(pos)
    voxel_side = (box_length ** 3 / n_halos) ** (1.0 / 3.0)
    logger.info(f"  Voxel side: {voxel_side:.4f} Mpc/h")

    n_patches = max(1, int(np.ceil(box_length / (stride * voxel_side))))
    logger.info(f"  Grid: {n_patches}^3 = {n_patches**3} patches "
                f"(stride={stride}, voxel_side={voxel_side:.4f})")

    patch_dirs = []
    for i in range(n_patches):
        for j in range(n_patches):
            for k in range(n_patches):
                out_subdir = (output_dir / f"{catalog}-{N}"
                              / f"set{sim_id}_pos_{i}_{j}_{k}" / "PART_009")

                if skip_existing and (out_subdir / "disp.npy").exists() \
                        and (out_subdir / "vel.npy").exists():
                    patch_dirs.append((out_subdir, (i, j, k)))
                    continue

                center = generate_patch_center(i, j, k, stride, voxel_side, box_length)
                voxel_centers = build_voxel_mesh(center, voxel_side, N, box_length)
                disp, vel_field, meta = dispatch_assign(
                    method, pos, vel, voxel_centers, box_length, logger, assign_cfg)

                save_outputs(out_subdir, disp, vel_field, cosmo, meta,
                             skip_existing=False)
                logger.info(f"  Saved patch {i},{j},{k} -> {out_subdir}")
                patch_dirs.append((out_subdir, (i, j, k)))

    return patch_dirs


# ---------------------------------------------------------------------------
# Stitching  (identical to conversion_gpu.py)
# ---------------------------------------------------------------------------

def make_weight_window(N, mode="hann", eps=1e-3):
    if mode == "uniform":
        w1 = np.ones(N, dtype=np.float32)
    elif mode == "hann":
        w1 = np.hanning(N).astype(np.float32)
        w1 = np.maximum(w1, eps)
    else:
        raise ValueError(f"Unknown weight mode: {mode}")
    w3 = w1[:, None, None] * w1[None, :, None] * w1[None, None, :]
    return w3[None].astype(np.float32)


def add_patch_periodic(accum, weights, patch, origin, window):
    C, global_N, _, _ = accum.shape
    _, N, _, _ = patch.shape
    xs = (origin[0] + np.arange(N)) % global_N
    ys = (origin[1] + np.arange(N)) % global_N
    zs = (origin[2] + np.arange(N)) % global_N
    accum[np.ix_(np.arange(C), xs, ys, zs)] += patch * window
    weights[np.ix_(np.arange(1), xs, ys, zs)] += window


def find_patch_dirs(patch_root, sim_id):
    label_re = re.compile(r"set(?P<sim>\d+)_pos_(?P<i>\d+)_(?P<j>\d+)_(?P<k>\d+)$")
    out = []
    for p in sorted(patch_root.glob(f"set{sim_id}_pos_*")):
        if not p.is_dir():
            continue
        m = label_re.match(p.name)
        if m is None:
            continue
        out.append((p / "PART_009", (int(m.group("i")), int(m.group("j")), int(m.group("k")))))
    return out


def stitch_field(patch_dirs, field, global_N, patch_N, stride_vox, weight_mode):
    accum = None
    weights = np.zeros((1, global_N, global_N, global_N), dtype=np.float32)
    window = make_weight_window(patch_N, mode=weight_mode)
    used = 0
    for part_dir, ijk in patch_dirs:
        fpath = part_dir / f"{field}.npy"
        if not fpath.exists():
            continue
        patch = np.load(fpath).astype(np.float32)
        if patch.ndim != 4 or patch.shape[1:] != (patch_N, patch_N, patch_N):
            raise ValueError(f"Unexpected patch shape {patch.shape} in {fpath}")
        if accum is None:
            accum = np.zeros((patch.shape[0], global_N, global_N, global_N),
                             dtype=np.float32)
        origin = tuple((x * stride_vox) % global_N for x in ijk)
        add_patch_periodic(accum, weights, patch, origin, window)
        used += 1
    if accum is None or used == 0:
        raise RuntimeError(f"No patches found for field {field}")
    stitched = accum / np.maximum(weights, 1e-12)
    uncovered = int((weights[0] == 0).sum())
    if uncovered:
        print(f"WARNING: {uncovered} voxels uncovered for {field}.")
    print(f"Stitched {used} patches for {field}; "
          f"coverage min={weights.min():.4g} max={weights.max():.4g}")
    return stitched.astype(np.float32)


def stitch_patches(patch_root, output_dir, sim_id, catalog,
                   global_N, patch_N, stride_vox, weight_mode="hann", logger=None):
    patch_root = patch_root / f"{catalog}-{global_N}"
    patch_dirs = find_patch_dirs(patch_root, sim_id)
    if not patch_dirs:
        raise RuntimeError(f"No patches under {patch_root} for sim {sim_id}")
    if logger:
        logger.info(f"  Found {len(patch_dirs)} patch directories for stitching")
    stitched_dir = output_dir / "stitched" / f"set{sim_id}_{catalog}" / "PART_009"
    stitched_dir.mkdir(parents=True, exist_ok=True)
    for field in ("disp", "vel"):
        np.save(stitched_dir / f"{field}.npy",
                stitch_field(patch_dirs, field, global_N, patch_N, stride_vox, weight_mode))
        if logger:
            logger.info(f"  Saved stitched {field}")
    style_src = patch_dirs[0][0] / "style.npy"
    if style_src.exists():
        np.save(stitched_dir / "style.npy", np.load(style_src).astype(np.float32))
        if logger:
            logger.info("  Saved style")
    return stitched_dir


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    logger = logging.getLogger(__name__)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    assign_cfg = dict(
        # shared
        K_max=args.K_max, gpu_id=args.gpu_id,
        # IVF
        nlist=args.nlist, nprobe=args.nprobe,
        # HNSW
        M_hnsw=args.M_hnsw, ef_construction=args.ef_construction,
        ef_search=args.ef_search,
        # random proj
        n_proj=args.n_proj, n_candidates=args.n_candidates,
        # stochastic / soft
        k=args.k, sigma_scale=args.sigma_scale, kernel=args.kernel,
        seed=args.seed,
        # sinkhorn
        k_local=args.k_local, block_side=args.block_side,
        epsilon=args.sinkhorn_eps, n_iters=args.sinkhorn_iters,
    )

    logger.info(f"Method        : {args.method}")
    logger.info(f"Data dir      : {data_dir}")
    logger.info(f"Output dir    : {output_dir}")
    logger.info(f"GPU ID        : {args.gpu_id}")
    logger.info(f"FAISS version : {faiss.__version__}")
    logger.info(f"Assign cfg    : {assign_cfg}")

    for sim_id in args.sim_ids:
        logger.info(f"=== Simulation {sim_id} ===")
        if not paired_catalogs_exist(data_dir, sim_id):
            logger.warning(f"Skipping sim {sim_id}: missing paired halos.h5")
            continue

        q_dir = output_dir / "stitched" / f"set{sim_id}_quijote" / "PART_009"
        ql_dir = output_dir / "stitched" / f"set{sim_id}_quijotelike" / "PART_009"
        q_done = (q_dir / "disp.npy").exists() and (q_dir / "vel.npy").exists()
        ql_done = (ql_dir / "disp.npy").exists() and (ql_dir / "vel.npy").exists()

        if q_done and ql_done:
            logger.info(f"Both stitched outputs exist for sim {sim_id}, skipping")
            continue

        for catalog, done in (("quijote", q_done), ("quijotelike", ql_done)):
            if done:
                continue
            step = 1 if catalog == "quijote" else 3
            logger.info(f"[Step {step}] Generating {catalog} patches for sim {sim_id}...")
            patch_dirs = generate_patches_for_sim(
                data_dir=data_dir, catalog=catalog, sim_id=sim_id,
                redshift=args.redshift, N=args.N, stride=args.stride,
                box_length=args.box_length, output_dir=output_dir,
                logger=logger, method=args.method, assign_cfg=assign_cfg,
                skip_existing=args.skip_existing,
            )
            logger.info(f"  Generated/found {len(patch_dirs)} {catalog} patches")

            if args.stitch:
                logger.info(f"[Step {step+1}] Stitching {catalog} patches...")
                try:
                    stitch_patches(output_dir, output_dir, sim_id, catalog,
                                   args.N, args.N, args.stride, args.weight_mode, logger)
                except Exception as e:
                    logger.warning(f"  Stitching failed for {catalog} sim {sim_id}: {e}")

    logger.info("Done.")


def parse_args():
    p = argparse.ArgumentParser(
        description="Approximate/probabilistic halo-to-voxel preprocessing ablations.")

    # Run config
    p.add_argument("--method", choices=list(_METHODS), default="soft_knn",
                   help="Assignment method to use.")
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--sim-ids", nargs="+", type=int, default=DEFAULT_SIM_IDS)
    p.add_argument("--total-sims", type=int, default=2000)
    p.add_argument("--chunk-size", type=int, default=20)
    p.add_argument("--chunk-idx", type=int, default=None)
    p.add_argument("--redshift", default=DEFAULT_REDSHIFT)
    p.add_argument("--N", type=int, default=DEFAULT_N)
    p.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    p.add_argument("--box-length", type=float, default=DEFAULT_BOX_LENGTH)
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--skip-existing", action="store_true", default=True)
    p.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    p.add_argument("--stitch", action="store_true", default=False)
    p.add_argument("--weight-mode", choices=["uniform", "hann"], default="hann")

    # Shared approximate params
    p.add_argument("--K-max", type=int, default=128,
                   help="Candidate pool for hard-assignment methods (exact_gpu, ivf_gpu, "
                        "hnsw_cpu, random_proj).")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help="RNG seed for stochastic_topk and random_proj.")

    # IVF GPU
    p.add_argument("--nlist", type=int, default=512,
                   help="[ivf_gpu] Number of Voronoi cells.")
    p.add_argument("--nprobe", type=int, default=64,
                   help="[ivf_gpu] Cells searched per query.")

    # HNSW CPU
    p.add_argument("--M-hnsw", type=int, default=32,
                   help="[hnsw_cpu] Graph degree.")
    p.add_argument("--ef-construction", type=int, default=200,
                   help="[hnsw_cpu] Build beam width.")
    p.add_argument("--ef-search", type=int, default=128,
                   help="[hnsw_cpu] Query beam width.")

    # Random projection
    p.add_argument("--n-proj", type=int, default=16,
                   help="[random_proj] Projection dimensionality.")
    p.add_argument("--n-candidates", type=int, default=256,
                   help="[random_proj] Candidates retrieved in projected space.")

    # Stochastic / soft k-NN
    p.add_argument("--k", type=int, default=16,
                   help="[stochastic_topk, soft_knn] Neighbour count.")
    p.add_argument("--sigma-scale", type=float, default=1.0,
                   help="[stochastic_topk, soft_knn] sigma = mean_spacing * sigma_scale.")
    p.add_argument("--kernel", choices=["gaussian", "tophat"], default="gaussian",
                   help="[soft_knn] Distance weighting kernel.")

    # Sinkhorn
    p.add_argument("--k-local", type=int, default=512,
                   help="[sinkhorn] Halos retrieved per block.")
    p.add_argument("--block-side", type=int, default=8,
                   help="[sinkhorn] Voxels per block edge (N must be divisible).")
    p.add_argument("--sinkhorn-eps", type=float, default=0.05,
                   help="[sinkhorn] Entropy regularisation epsilon.")
    p.add_argument("--sinkhorn-iters", type=int, default=50,
                   help="[sinkhorn] Sinkhorn iterations.")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.chunk_idx is not None:
        start = args.chunk_idx * args.chunk_size
        end = min(start + args.chunk_size, args.total_sims)
        args.sim_ids = list(range(start, end))
    main(args)
