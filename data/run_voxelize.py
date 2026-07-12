#!/usr/bin/env python3
"""
run_voxelize.py
================
Rasterizes halo catalogs onto a fixed-resolution periodic mesh (matching the
simulation's native L1000-N128 mesh resolution, N_mesh=128) and extracts
overlapping patches with voxelize.voxelize().

Unlike conversion.py's generate_patches_for_sim (whose grid resolution
N_global is derived per-sim from the halo count and therefore varies between
simulations), this script rasterizes onto a FIXED N_mesh^3 grid so that
voxelize()'s exact-divisibility requirement (L % (D-d) == 0) holds uniformly
across every simulation.

For each simulation (quijote + quijotelike, kept in separate output subdirs):
  1. Load halo positions/velocities from HDF5.
  2. Rasterize onto a fixed N_mesh^3 voxel grid via assign_halos_to_global_voxels
     (each halo claimed by exactly one voxel; zero-filled where none assigned).
     Also histogram halo counts per voxel via halo_counts_field (many-to-one;
     multiplicity preserved, same node-centred lattice).
  3. Run voxelize() with the configured (D, d) to extract overlapping,
     periodically-wrapped patches (same slicing for disp, vel, and counts).
  4. Save each patch's displacement/velocity/counts/style arrays to disk.

Slurm array-job chunking
------------------------
--slurm-job-size / --slurm-job-idx partition the simulation list across Slurm array
jobs, same convention as conversion.py.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

import conversion as c
import voxelize as v


DEFAULT_DATA_DIR   = "/home/juliahul/projects/stuff/universedata/cmass_ili_new/cmass-ili"
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_REDSHIFT   = "0.666667"
DEFAULT_N_MESH     = 128     # fixed rasterization resolution (matches L1000-N128)
DEFAULT_D          = 64      # patch size in voxels per axis
DEFAULT_D_OVERLAP  = 32      # overlap between adjacent patches in voxels
DEFAULT_BOX_LENGTH = 1000.0  # Mpc/h
DEFAULT_K_MAX      = 64      # KD-tree candidate pool size for assignment


def voxelize_sim(
    data_dir: Path,
    catalog: str,
    sim_id: int,
    redshift: str,
    N_mesh: int,
    D: int,
    d: int,
    box_length: float,
    output_dir: Path,
    logger: logging.Logger,
    k_max: int,
    skip_existing: bool = True,
) -> list:
    """
    Rasterize one simulation's halos onto a fixed N_mesh^3 grid and voxelize.

    Returns
    -------
    list of (Path, (i, j, k))  output subdirectory and patch index for each patch
    """
    stride = D - d
    assert N_mesh % stride == 0, (
        f"N_mesh={N_mesh} must be divisible by stride=D-d={stride} (D={D}, d={d})"
    )
    N_patches = N_mesh // stride

    catalog_dir = output_dir / f"{catalog}-{D}"

    if skip_existing:
        all_exist = all(
            (catalog_dir / f"set{sim_id}_pos_{i}_{j}_{k}" / "PART_009" / fname).exists()
            for i in range(N_patches) for j in range(N_patches) for k in range(N_patches)
            for fname in ("disp.npy", "counts.npy")
        )
        if all_exist:
            logger.info(f"  All {N_patches**3} patches already exist for {catalog} sim {sim_id}, skipping")
            return [
                (catalog_dir / f"set{sim_id}_pos_{i}_{j}_{k}" / "PART_009", (i, j, k))
                for i in range(N_patches) for j in range(N_patches) for k in range(N_patches)
            ]

    logger.info(f"  Loading {catalog} halos for sim {sim_id}...")
    pos, vel = c.load_halos(data_dir, catalog, sim_id, redshift)
    cosmo    = c.load_cosmo(data_dir, catalog, sim_id)

    if pos is None or vel is None:
        logger.warning(f"  Missing {catalog} halos/redshift for sim {sim_id}; skipping")
        return []

    logger.info(f"  {catalog}: {len(pos)} halos -> rasterizing onto fixed {N_mesh}^3 grid")
    voxel_side = box_length / N_mesh
    global_vc  = c.build_global_voxel_grid(N_mesh, voxel_side, box_length)
    disp, velf = c.assign_halos_to_global_voxels(
        pos, vel, global_vc, box_length, logger, K_max=k_max
    )
    # disp, velf : (3, N_mesh, N_mesh, N_mesh)

    # Halo counts per voxel (many-to-one histogram; multiplicity that the
    # one-to-one assignment above discards).  Same node-centred lattice.
    counts = c.halo_counts_field(pos, voxel_side, N_mesh, box_length)
    # counts : (1, N_mesh, N_mesh, N_mesh)

    disp_patches,   origins = v.voxelize(disp,   D, d)
    vel_patches,    _       = v.voxelize(velf,   D, d)
    counts_patches, _       = v.voxelize(counts, D, d)
    # disp_patches, vel_patches : (N_patches, N_patches, N_patches, 3, D, D, D)
    # counts_patches            : (N_patches, N_patches, N_patches, 1, D, D, D)

    catalog_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "N_mesh":     N_mesh,
        "D":          D,
        "d":          d,
        "stride":     stride,
        "voxel_side": float(voxel_side),
        "n_patches":  N_patches,
        "sim_id":     sim_id,
        "catalog":    catalog,
    }
    with open(catalog_dir / f"set{sim_id}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    patch_dirs = []
    for i in range(N_patches):
        for j in range(N_patches):
            for k in range(N_patches):
                out_subdir = catalog_dir / f"set{sim_id}_pos_{i}_{j}_{k}" / "PART_009"
                if skip_existing and (out_subdir / "disp.npy").exists() and (out_subdir / "vel.npy").exists():
                    # Backfill counts.npy for patches generated before counts existed.
                    if not (out_subdir / "counts.npy").exists():
                        np.save(out_subdir / "counts.npy",
                                counts_patches[i, j, k].astype(np.float32))
                    patch_dirs.append((out_subdir, (i, j, k)))
                    continue
                c.save_outputs(
                    out_subdir, disp_patches[i, j, k], vel_patches[i, j, k], cosmo,
                    skip_existing=False, counts=counts_patches[i, j, k],
                )
                patch_dirs.append((out_subdir, (i, j, k)))

    logger.info(f"  Saved {len(patch_dirs)} patches for {catalog} sim {sim_id} (origins start at {origins[0,0,0]})")
    return patch_dirs


def main(args):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    logger.info(f"Data dir      : {data_dir}")
    logger.info(f"Output dir    : {output_dir}")
    logger.info(f"Sim IDs       : {args.sim_ids}")
    logger.info(f"N_mesh={args.n_mesh}, D={args.D}, d={args.d}, k_max={args.k_max}")

    for sim_id in args.sim_ids:
        logger.info(f"=== Simulation {sim_id} ===")
        if not c.paired_catalogs_exist(data_dir, sim_id):
            logger.warning(f"Skipping sim {sim_id}: missing paired quijote/quijotelike halos.h5")
            continue

        for catalog in ["quijote", "quijotelike"]:
            voxelize_sim(
                data_dir=data_dir, catalog=catalog, sim_id=sim_id,
                redshift=args.redshift, N_mesh=args.n_mesh, D=args.D, d=args.d,
                box_length=args.box_length, output_dir=output_dir,
                logger=logger, k_max=args.k_max, skip_existing=args.skip_existing,
            )

    logger.info("Done.")


def parse_args():
    p = argparse.ArgumentParser(
        description="Rasterize halo catalogs onto a fixed mesh and voxelize into overlapping patches."
    )
    p.add_argument("--data-dir",    default=DEFAULT_DATA_DIR)
    p.add_argument("--output-dir",  default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--sim-ids",     nargs="+", type=int, default=list(range(2000)))

    p.add_argument("--total-sims",      type=int, default=2000)
    p.add_argument("--slurm-job-size",  type=int, default=20)
    p.add_argument("--slurm-job-idx",   type=int, default=None)

    p.add_argument("--redshift",    default=DEFAULT_REDSHIFT)
    p.add_argument("--n-mesh",      type=int,   default=DEFAULT_N_MESH,
                   help="Fixed rasterization grid resolution per axis (default: %(default)s)")
    p.add_argument("--D",           type=int,   default=DEFAULT_D,
                   help="Patch size in voxels per axis (default: %(default)s)")
    p.add_argument("--d",           type=int,   default=DEFAULT_D_OVERLAP,
                   help="Overlap between adjacent patches in voxels (default: %(default)s)")
    p.add_argument("--box-length",  type=float, default=DEFAULT_BOX_LENGTH)
    p.add_argument("--k-max",       type=int,   default=DEFAULT_K_MAX,
                   help="KD-tree candidate pool size for halo assignment (default: %(default)s)")

    p.add_argument("--skip-existing",    action="store_true", default=True)
    p.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.slurm_job_idx is not None:
        start = args.slurm_job_idx * args.slurm_job_size
        end   = min(start + args.slurm_job_size, args.total_sims)
        args.sim_ids = list(range(start, end))

    main(args)
