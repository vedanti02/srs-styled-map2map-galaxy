"""On-the-fly patch tiling of the `stitched/` 64^3 cubes.

Source of truth is `stitched/` (same data, split and normalization as
``pair_dataset.PairDataset``). Each 64^3 cube is tiled into a 2x2x2 grid of
32^3 patches; an optional halo of ``pad`` voxels is extracted with periodic
wrap (exactly correct, since the full box IS periodic).

Patch index convention: ``p = i*4 + j*2 + k`` with patch origin
``(32i, 32j, 32k)`` — i.e. ``np.unravel_index(p, (2,2,2))``, matching
`utils.utils.cropfield`. Stitching with `stitch_patches` reproduces the
original cube exactly (verified by ``python -m data.patch_dataset``).
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset

from .pair_dataset import (  # reuse: identical split & normalization
    CHAN_STD, normalize, denormalize, _scan_set_ids, _load_6ch,
)

N_FULL = 64          # stitched cube size
N_SPLIT = 2          # patches per axis
PATCH = N_FULL // N_SPLIT   # 32
N_PATCHES = N_SPLIT ** 3    # 8


def extract_patch(cube, p, pad=0):
    """cube: (C, 64, 64, 64) array. Returns patch p in the 2x2x2 grid with a
    periodic-wrap halo of ``pad`` voxels per face: (C, 32+2*pad, ...)."""
    ijk = np.unravel_index(p, (N_SPLIT,) * 3)
    x = cube
    for d, i in enumerate(ijk):
        lo = i * PATCH - pad
        x = x.take(range(lo, lo + PATCH + 2 * pad), axis=1 + d, mode="wrap")
    return x


def stitch_patches(patches):
    """patches: sequence/array of N_PATCHES arrays (C, 32, 32, 32) in patch-index
    order (NO halo — crop it before calling). Returns (C, 64, 64, 64)."""
    patches = np.asarray(patches)
    assert patches.shape[0] == N_PATCHES and patches.shape[-1] == PATCH, patches.shape
    C = patches.shape[1]
    out = np.empty((C, N_FULL, N_FULL, N_FULL), dtype=patches.dtype)
    for p in range(N_PATCHES):
        i, j, k = np.unravel_index(p, (N_SPLIT,) * 3)
        out[:, i*PATCH:(i+1)*PATCH, j*PATCH:(j+1)*PATCH, k*PATCH:(k+1)*PATCH] = patches[p]
    return out


def crop_interior(x, pad):
    """Crop a halo'd patch (..., 32+2p, 32+2p, 32+2p) back to the central 32^3.
    Works on numpy arrays and torch tensors (trailing 3 dims)."""
    if pad == 0:
        return x
    return x[..., pad:-pad, pad:-pad, pad:-pad]


class PatchPairDataset(Dataset):
    """Paired (LR patch, HR patch, theta) samples; one item per (sim, patch).

    Each item:
        x_lr: (6, 32+2*pad, 32+2*pad, 32+2*pad)   halo'd LR patch
        x_hr: (6, 32+2*pad, ...)                  halo'd HR patch (crop for loss)
        theta: (5,), sid: int, p: int (patch index 0..7)

    Split semantics identical to PairDataset (split over *sims*, seed=0
    default), so test sims here are the same test sims as v1–v7.
    """

    def __init__(self, stitched_root, split="train", pad=0,
                 train_frac=0.8, val_frac=0.1, snap="PART_009", seed=0,
                 normalize_inputs=True):
        self.root = stitched_root
        self.snap = snap
        self.pad = pad
        self.normalize_inputs = normalize_inputs

        ids = _scan_set_ids(stitched_root)
        rng = np.random.default_rng(seed)
        ids = list(ids)
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(train_frac * n)
        n_val = int(val_frac * n)
        if split == "train":
            self.ids = sorted(ids[:n_train])
        elif split == "val":
            self.ids = sorted(ids[n_train:n_train + n_val])
        elif split == "test":
            self.ids = sorted(ids[n_train + n_val:])
        elif split == "all":
            self.ids = sorted(ids)
        else:
            raise ValueError(f"unknown split {split!r}")

    def __len__(self):
        return len(self.ids) * N_PATCHES

    def load_cubes(self, sid):
        """Full normalized (or raw) LR/HR cubes + theta for one sim."""
        lr_dir = os.path.join(self.root, f"set{sid}_quijotelike")
        hr_dir = os.path.join(self.root, f"set{sid}_quijote")
        x_lr = _load_6ch(lr_dir, self.snap)
        x_hr = _load_6ch(hr_dir, self.snap)
        theta = np.load(os.path.join(lr_dir, self.snap, "style.npy")).astype(np.float32)
        if self.normalize_inputs:
            x_lr = normalize(x_lr)
            x_hr = normalize(x_hr)
        return x_lr, x_hr, theta

    def __getitem__(self, idx):
        sid = self.ids[idx // N_PATCHES]
        p = idx % N_PATCHES
        x_lr, x_hr, theta = self.load_cubes(sid)
        return (
            torch.from_numpy(np.ascontiguousarray(extract_patch(x_lr, p, self.pad))),
            torch.from_numpy(np.ascontiguousarray(extract_patch(x_hr, p, self.pad))),
            torch.from_numpy(theta),
            sid,
            p,
        )


if __name__ == "__main__":
    # Validation: tiling + stitching must reproduce the cube EXACTLY, and the
    # halo must equal the periodic neighbourhood.
    root = "/data/group_data/universedata/lagrangian_output_64/stitched"
    ds = PatchPairDataset(root, split="val", pad=3)
    print(f"sims: train={len(PatchPairDataset(root,'train').ids)} "
          f"val={len(ds.ids)} test={len(PatchPairDataset(root,'test').ids)}")
    print(f"patches/sim: {N_PATCHES}  patch size: {PATCH}^3  "
          f"items (val): {len(ds)}")

    sid = ds.ids[0]
    x_lr, x_hr, _ = ds.load_cubes(sid)

    # 1) exact reconstruction (pad=0 path)
    rec = stitch_patches([extract_patch(x_hr, p, 0) for p in range(N_PATCHES)])
    assert rec.shape == x_hr.shape and np.array_equal(rec, x_hr), "stitch mismatch"
    print("PASS: pad=0 tile→stitch reproduces the 64^3 cube bit-exactly")

    # 2) halo'd extraction: interior crop must equal the pad=0 patch,
    #    and halo voxels must equal the periodic neighbours
    pad = 3
    for p in (0, 5, 7):
        full = extract_patch(x_hr, p, pad)
        assert full.shape == (6,) + (PATCH + 2 * pad,) * 3, full.shape
        assert np.array_equal(crop_interior(full, pad), extract_patch(x_hr, p, 0))
    # explicit wrap value check for patch 0: halo voxel (axis 0, index -1) == cube row 63
    full0 = extract_patch(x_hr, 0, pad)
    assert np.array_equal(full0[:, 0, pad:-pad, pad:-pad], x_hr[:, 64 - pad, 0:PATCH, 0:PATCH])
    print("PASS: pad=3 halo equals true periodic neighbourhood")

    # 3) reconstruction from halo'd patches after interior crop
    rec2 = stitch_patches([crop_interior(extract_patch(x_hr, p, pad), pad)
                           for p in range(N_PATCHES)])
    assert np.array_equal(rec2, x_hr)
    print("PASS: halo'd extract → crop → stitch reproduces the cube bit-exactly")

    print("summary: 2000 sims x 8 patches = 16000 patch pairs "
          "(train 12800 / val 1600 / test 1600)")
