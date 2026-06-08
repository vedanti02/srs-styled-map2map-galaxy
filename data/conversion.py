#!/usr/bin/env python3
"""
convert_halos_to_grid_stitched.py
==================================
Converts dark-matter halo catalogs into Lagrangian displacement/velocity fields
on fixed N^3 grids.

This version:
- Uses convolutional-style overlapping patches.
- Optionally stitches patches into full boxes.
- Supports Slurm chunking over sim IDs.
- Skips simulations unless BOTH quijote and quijotelike halos.h5 exist.
"""

import argparse
import logging
import re
from pathlib import Path

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
# I/O helpers
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


def load_halos(data_dir: Path, catalog: str, sim_id: int, redshift_key: str):
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
    cfg_path = config_path(data_dir, catalog, sim_id)

    if not cfg_path.exists():
        return None

    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    return np.array(cfg["nbody"]["cosmo"], dtype=np.float32)


def save_outputs(out_dir: Path, disp: np.ndarray, vel: np.ndarray,
                 style: np.ndarray, skip_existing: bool = True):
    if skip_existing and (out_dir / "disp.npy").exists():
        return False

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "disp.npy", disp.astype(np.float32))
    np.save(out_dir / "vel.npy", vel.astype(np.float32))

    if style is not None:
        np.save(out_dir / "style.npy", style.astype(np.float32))

    return True


# ---------------------------------------------------------------------------
# Mesh construction
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


# ---------------------------------------------------------------------------
# Periodic distance utilities
# ---------------------------------------------------------------------------

def periodic_disp_vec(a: np.ndarray, b: np.ndarray, box_length: float) -> np.ndarray:
    d = a - b
    d -= box_length * np.round(d / box_length)
    return d


def periodic_dist_sq(a: np.ndarray, b: np.ndarray, box_length: float) -> np.ndarray:
    d = periodic_disp_vec(a, b, box_length)
    return np.einsum("...i,...i->...", d, d)


# ---------------------------------------------------------------------------
# Halo-to-voxel assignment
# ---------------------------------------------------------------------------

def assign_halos_to_voxels(
    halo_pos: np.ndarray,
    halo_vel: np.ndarray,
    voxel_centers: np.ndarray,
    box_length: float,
    logger: logging.Logger,
    K_max: int = 128,
):
    N = voxel_centers.shape[0]
    n_voxels = N * N * N
    M = len(halo_pos)

    vc_flat = voxel_centers.reshape(n_voxels, 3)
    patch_center = voxel_centers[N // 2, N // 2, N // 2]

    dsq_from_center = periodic_dist_sq(vc_flat, patch_center, box_length)
    voxel_order = np.argsort(dsq_from_center)

    shifts = np.array([
        [dx, dy, dz]
        for dx in [-box_length, 0.0, box_length]
        for dy in [-box_length, 0.0, box_length]
        for dz in [-box_length, 0.0, box_length]
    ], dtype=np.float32)

    def build_tree(orig_indices):
        pos = halo_pos[orig_indices]
        rep = (pos[np.newaxis] + shifts[:, np.newaxis]).reshape(-1, 3)
        return cKDTree(rep), np.tile(orig_indices, 27)

    tree, image_to_halo = build_tree(np.arange(M))

    # Query all voxels in one batch call instead of one call per voxel.
    # workers=-1 uses all available CPU threads for the query.
    k_query = min(K_max, len(image_to_halo))
    _, img_idx_all = tree.query(vc_flat, k=k_query, workers=-1)
    if k_query == 1:
        img_idx_all = img_idx_all[:, np.newaxis]
    halo_cands = image_to_halo[img_idx_all]  # (n_voxels, k_query)

    assigned = np.full(M, False)
    disp_flat = np.zeros((n_voxels, 3), dtype=np.float32)
    vel_flat = np.zeros((n_voxels, 3), dtype=np.float32)

    n_assigned = 0
    n_zero_filled = 0
    n_rebuilds = 0
    needs_fallback = []

    # Greedy center-out assignment: same order as before, but no tree queries in the loop.
    for vi in voxel_order:
        found = False
        for ci in range(k_query):
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

    # Fallback: voxels whose K_max nearest halos were all taken.
    # One tree rebuild from the remaining unassigned halos handles all of them.
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
            fb_cands = fb_image_to_halo[fb_img]  # (len(fb_voxels), k_fb)

            # fb_voxels is already in voxel_order (appended in that order above).
            for i, vi in enumerate(fb_voxels):
                found = False
                for ci in range(k_fb):
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

    logger.info(
        f"    assigned={n_assigned}/{n_voxels}, zero-filled={n_zero_filled}, "
        f"tree-rebuilds={n_rebuilds}"
    )

    disp_field = disp_flat.reshape(N, N, N, 3).transpose(3, 0, 1, 2)
    vel_field = vel_flat.reshape(N, N, N, 3).transpose(3, 0, 1, 2)

    return disp_field, vel_field


# ---------------------------------------------------------------------------
# Patch generation
# ---------------------------------------------------------------------------

def generate_patch_center(i: int, j: int, k: int, stride: int,
                          voxel_side: float, box_length: float) -> np.ndarray:
    origin = np.array([i * stride, j * stride, k * stride], dtype=np.float32)
    center = (origin * voxel_side) % box_length
    return center


def generate_patches_for_sim(
    data_dir: Path,
    catalog: str,
    sim_id: int,
    redshift: str,
    N: int,
    stride: int,
    box_length: float,
    output_dir: Path,
    logger: logging.Logger,
    skip_existing: bool = True,
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

    n_patches = int(np.ceil(box_length / (stride * voxel_side)))
    n_patches = max(1, n_patches)

    logger.info(
        f"  Grid: {n_patches}^3 = {n_patches ** 3} patches "
        f"(stride={stride}, voxel_side={voxel_side:.4f})"
    )

    patch_dirs = []

    for i in range(n_patches):
        for j in range(n_patches):
            for k in range(n_patches):
                out_subdir = (
                    output_dir
                    / f"{catalog}-{N}"
                    / f"set{sim_id}_pos_{i}_{j}_{k}"
                    / "PART_009"
                )

                if skip_existing and (out_subdir / "disp.npy").exists() and (out_subdir / "vel.npy").exists():
                    logger.info(f"  Patch {i},{j},{k} already exists, skipping")
                    patch_dirs.append((out_subdir, (i, j, k)))
                    continue

                center = generate_patch_center(i, j, k, stride, voxel_side, box_length)
                voxel_centers = build_voxel_mesh(center, voxel_side, N, box_length)
                disp, vel_field = assign_halos_to_voxels(
                    pos, vel, voxel_centers, box_length, logger
                )

                save_outputs(out_subdir, disp, vel_field, cosmo, skip_existing=False)
                logger.info(f"  Saved patch {i},{j},{k} -> {out_subdir}")
                patch_dirs.append((out_subdir, (i, j, k)))

    return patch_dirs


# ---------------------------------------------------------------------------
# Stitching
# ---------------------------------------------------------------------------

def make_weight_window(N: int, mode: str = "hann", eps: float = 1e-3) -> np.ndarray:
    if mode == "uniform":
        w1 = np.ones(N, dtype=np.float32)
    elif mode == "hann":
        w1 = np.hanning(N).astype(np.float32)
        w1 = np.maximum(w1, eps)
    else:
        raise ValueError(f"Unknown weight mode: {mode}")

    w3 = w1[:, None, None] * w1[None, :, None] * w1[None, None, :]
    return w3[None].astype(np.float32)


def add_patch_periodic(accum: np.ndarray, weights: np.ndarray, patch: np.ndarray,
                       origin: tuple[int, int, int], window: np.ndarray) -> None:
    C, global_N, _, _ = accum.shape
    _, N, _, _ = patch.shape

    xs = (origin[0] + np.arange(N)) % global_N
    ys = (origin[1] + np.arange(N)) % global_N
    zs = (origin[2] + np.arange(N)) % global_N

    idx = np.ix_(np.arange(C), xs, ys, zs)
    accum[idx] += patch * window
    weights[np.ix_(np.arange(1), xs, ys, zs)] += window


def find_patch_dirs(patch_root: Path, sim_id: int):
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


def stitch_field(patch_dirs, field: str, global_N: int, patch_N: int,
                 stride_vox: int, weight_mode: str) -> np.ndarray:
    accum = None
    weights = np.zeros((1, global_N, global_N, global_N), dtype=np.float32)
    window = make_weight_window(patch_N, mode=weight_mode)

    used = 0

    for part_dir, ijk in patch_dirs:
        fpath = part_dir / f"{field}.npy"
        if not fpath.exists():
            continue

        patch = np.load(fpath).astype(np.float32)

        if patch.ndim != 4:
            raise ValueError(f"Expected {fpath} shape (C,N,N,N), got {patch.shape}")

        if patch.shape[1:] != (patch_N, patch_N, patch_N):
            raise ValueError(f"Expected patch size {patch_N} for {fpath}, got {patch.shape}")

        if accum is None:
            C = patch.shape[0]
            accum = np.zeros((C, global_N, global_N, global_N), dtype=np.float32)

        origin = tuple((x * stride_vox) % global_N for x in ijk)
        add_patch_periodic(accum, weights, patch, origin, window)
        used += 1

    if accum is None or used == 0:
        raise RuntimeError(f"No patches found for field {field}")

    stitched = accum / np.maximum(weights, 1e-12)

    uncovered = int(np.sum(weights[0] == 0))
    if uncovered > 0:
        print(f"WARNING: {uncovered} global voxels were not covered for field {field}.")

    print(
        f"Stitched {used} patches for {field}; "
        f"coverage min={weights.min():.4g}, max={weights.max():.4g}"
    )

    return stitched.astype(np.float32)


def stitch_patches(
    patch_root: Path,
    output_dir: Path,
    sim_id: int,
    catalog: str,
    global_N: int,
    patch_N: int,
    stride_vox: int,
    weight_mode: str = "hann",
    logger: logging.Logger | None = None,
) -> Path:
    patch_root = patch_root / f"{catalog}-{global_N}"
    patch_dirs = find_patch_dirs(patch_root, sim_id)

    if not patch_dirs:
        raise RuntimeError(f"No patch directories found under {patch_root} for sim {sim_id}")

    if logger:
        logger.info(f"  Found {len(patch_dirs)} patch directories for stitching")

    stitched_dir = output_dir / "stitched" / f"set{sim_id}_{catalog}" / "PART_009"
    stitched_dir.mkdir(parents=True, exist_ok=True)

    for field in ["disp", "vel"]:
        stitched = stitch_field(
            patch_dirs=patch_dirs,
            field=field,
            global_N=global_N,
            patch_N=patch_N,
            stride_vox=stride_vox,
            weight_mode=weight_mode,
        )
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

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    logger.info(f"Data dir      : {data_dir}")
    logger.info(f"Output dir    : {output_dir}")
    logger.info(f"Sim IDs       : {args.sim_ids}")
    logger.info(f"N={args.N}, stride={args.stride}, seed={args.seed}")
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

        quijote_stitched_dir = output_dir / "stitched" / f"set{sim_id}_quijote" / "PART_009"
        quijotelike_stitched_dir = output_dir / "stitched" / f"set{sim_id}_quijotelike" / "PART_009"

        quijote_exists = (quijote_stitched_dir / "disp.npy").exists() and (quijote_stitched_dir / "vel.npy").exists()
        quijotelike_exists = (quijotelike_stitched_dir / "disp.npy").exists() and (quijotelike_stitched_dir / "vel.npy").exists()

        if quijote_exists and quijotelike_exists:
            logger.info(f"Both stitched outputs exist for sim {sim_id}, skipping")
            continue

        if not quijote_exists:
            logger.info(f"[Step 1] Generating quijote patches for sim {sim_id}...")
            patch_dirs_q = generate_patches_for_sim(
                data_dir=data_dir,
                catalog="quijote",
                sim_id=sim_id,
                redshift=args.redshift,
                N=args.N,
                stride=args.stride,
                box_length=args.box_length,
                output_dir=output_dir,
                logger=logger,
                skip_existing=args.skip_existing,
            )
            logger.info(f"  Generated/found {len(patch_dirs_q)} quijote patches")

            if args.stitch:
                logger.info("[Step 2] Stitching quijote patches...")
                try:
                    stitch_patches(
                        patch_root=output_dir,
                        output_dir=output_dir,
                        sim_id=sim_id,
                        catalog="quijote",
                        global_N=args.N,
                        patch_N=args.N,
                        stride_vox=args.stride,
                        weight_mode=args.weight_mode,
                        logger=logger,
                    )
                except Exception as e:
                    logger.warning(f"  Quijote stitching failed for sim {sim_id}: {e}")

        if not quijotelike_exists:
            logger.info(f"[Step 3] Generating quijotelike patches for sim {sim_id}...")
            patch_dirs_ql = generate_patches_for_sim(
                data_dir=data_dir,
                catalog="quijotelike",
                sim_id=sim_id,
                redshift=args.redshift,
                N=args.N,
                stride=args.stride,
                box_length=args.box_length,
                output_dir=output_dir,
                logger=logger,
                skip_existing=args.skip_existing,
            )
            logger.info(f"  Generated/found {len(patch_dirs_ql)} quijotelike patches")

            if args.stitch:
                logger.info("[Step 4] Stitching quijotelike patches...")
                try:
                    stitch_patches(
                        patch_root=output_dir,
                        output_dir=output_dir,
                        sim_id=sim_id,
                        catalog="quijotelike",
                        global_N=args.N,
                        patch_N=args.N,
                        stride_vox=args.stride,
                        weight_mode=args.weight_mode,
                        logger=logger,
                    )
                except Exception as e:
                    logger.warning(f"  Quijotelike stitching failed for sim {sim_id}: {e}")

    logger.info("Done.")


def parse_args():
    p = argparse.ArgumentParser(
        description="Convert halo catalogs to Lagrangian grid format with stitching."
    )

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
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)

    p.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip patch generation if output already exists",
    )

    p.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="Regenerate patches even if output exists",
    )

    p.add_argument(
        "--stitch",
        action="store_true",
        default=False,
        help="Also stitch patches into full box after generation",
    )

    p.add_argument(
        "--weight-mode",
        choices=["uniform", "hann"],
        default="hann",
        help="Weighting mode for stitching",
    )

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.chunk_idx is not None:
        start = args.chunk_idx * args.chunk_size
        end = min(start + args.chunk_size, args.total_sims)
        args.sim_ids = list(range(start, end))

    main(args)