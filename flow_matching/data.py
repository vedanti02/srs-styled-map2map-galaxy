"""Count-field patch dataset for flow matching.

Wraps data.patch_dataset_cmass.PatchPairDatasetCmass with normalize_inputs=False so
that RAW COUNTS reach the trainer (dequantisation must be re-drawn every step, so the
transform lives on the GPU, not in the loader). Optional in-RAM uint8 cache of all
boxes (1600 train pairs ~ 6.7 GB) so an epoch is compute-bound, not NFS-bound.
"""
import os
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from torch.utils.data import Dataset

from data.patch_dataset_cmass import (PatchPairDatasetCmass, extract_patch, stitch_patches,
                                      crop_interior, PATCH, N_PATCHES, N_FULL, N_SPLIT, PROCESSED)

__all__ = ["CountPatchDataset", "FullBoxDataset", "extract_patch", "stitch_patches", "crop_interior",
           "stitch_torch", "PATCH", "N_PATCHES", "N_FULL", "PROCESSED"]


def stitch_torch(patches: torch.Tensor) -> torch.Tensor:
    """(8, C, 64, 64, 64) -> (C, 128, 128, 128), same ordering as stitch_patches (numpy)."""
    C = patches.shape[1]
    box = patches.new_empty((C, N_FULL, N_FULL, N_FULL))
    for p in range(N_PATCHES):
        i, j, k = np.unravel_index(p, (N_SPLIT,) * 3)
        box[:, i * PATCH:(i + 1) * PATCH, j * PATCH:(j + 1) * PATCH, k * PATCH:(k + 1) * PATCH] = patches[p]
    return box


class CountPatchDataset(Dataset):
    """Indexed by sim. __getitem__ -> (lr (8,1,64+2pad,..) float counts, hr (8,1,64,64,64), idx)."""

    def __init__(self, split="train", pad=0, cache=False, max_sets=0, workers=8, verbose=True):
        self.base = PatchPairDatasetCmass(split=split, pad=pad, normalize_inputs=False)
        self.ids = list(self.base.ids[:max_sets]) if max_sets > 0 else list(self.base.ids)
        self.pad = pad
        self.theta = self.base.theta
        self.cache = None
        if cache:
            self._build_cache(workers, verbose)

    def __len__(self):
        return len(self.ids)

    def _build_cache(self, workers, verbose):
        n = len(self.ids)
        lr = np.empty((n, N_FULL, N_FULL, N_FULL), np.uint8)
        hr = np.empty((n, N_FULL, N_FULL, N_FULL), np.uint8)
        clipped = [0]
        t0 = time.time()

        def load(j):
            l, h = self.base.load_boxes(self.ids[j])  # (1,128^3) float32 counts
            for src, dst in ((l[0], lr), (h[0], hr)):
                if src.max() > 255:
                    clipped[0] += int((src > 255).sum())
                dst[j] = np.clip(src, 0, 255).astype(np.uint8)
            return j

        with ThreadPoolExecutor(max_workers=workers) as ex:
            for m, _ in enumerate(ex.map(load, range(n))):
                if verbose and ((m + 1) % 200 == 0 or m + 1 == n):
                    print(f"  cache {m+1}/{n} boxes {time.time()-t0:.0f}s", flush=True)
        if clipped[0]:
            print(f"WARNING: {clipped[0]} voxels > 255 clipped in cache", flush=True)
        self.cache = (lr, hr)
        if verbose:
            print(f"cached {n} boxes ({(lr.nbytes+hr.nbytes)/1e9:.2f} GB) in {time.time()-t0:.0f}s", flush=True)

    def load_boxes(self, idx):
        """-> (lr, hr) each (1,128,128,128) float32 physical counts."""
        if self.cache is not None:
            j = self.ids.index(idx)
            return (self.cache[0][j].astype(np.float32)[None], self.cache[1][j].astype(np.float32)[None])
        return self.base.load_boxes(idx)

    def __getitem__(self, k):
        idx = self.ids[k]
        lr, hr = self.load_boxes(idx)
        lr_p = np.stack([extract_patch(lr, p, self.pad) for p in range(N_PATCHES)])
        hr_p = np.stack([extract_patch(hr, p, 0) for p in range(N_PATCHES)])
        return torch.from_numpy(lr_p), torch.from_numpy(hr_p), idx


class FullBoxDataset(Dataset):
    """Whole periodic 128^3 boxes, no patching. Optional third field per box (e.g. a deterministic
    corrector's output) read from <extra_dir>/sr_{idx}.npy. __getitem__ -> (lr, hr, extra, idx), each
    (1,128,128,128) float32 counts; extra is an empty tensor when extra_dir is None."""

    def __init__(self, split="train", extra_dir=None, cache=False, max_sets=0, workers=8, verbose=True):
        self.base = CountPatchDataset(split, pad=0, cache=False, max_sets=max_sets, verbose=False)
        self.ids = self.base.ids
        self.theta = self.base.theta
        self.extra_dir = extra_dir or None
        self.cache = None
        if cache:
            n = len(self.ids); nf = 3 if self.extra_dir else 2
            buf = np.empty((nf, n, N_FULL, N_FULL, N_FULL), np.uint8); t0 = time.time()

            def load(j):
                for f_, a in enumerate(self._load(self.ids[j])[:nf]):
                    buf[f_, j] = np.clip(a[0], 0, 255).astype(np.uint8)
            with ThreadPoolExecutor(max_workers=workers) as ex:
                list(ex.map(load, range(n)))
            self.cache = buf
            if verbose:
                print(f"cached {n} full boxes x {nf} fields ({buf.nbytes/1e9:.1f} GB) in {time.time()-t0:.0f}s", flush=True)

    def __len__(self):
        return len(self.ids)

    def _load(self, idx):
        lr, hr = self.base.load_boxes(idx)
        ex = np.load(f"{self.extra_dir}/sr_{idx}.npy").astype(np.float32)[None] if self.extra_dir else None
        return lr, hr, ex

    def load_boxes(self, idx):
        """-> (lr, hr, extra or None), each (1,128,128,128) float32 counts."""
        if self.cache is not None:
            j = self.ids.index(idx)
            out = [self.cache[f_, j].astype(np.float32)[None] for f_ in range(self.cache.shape[0])]
            return out[0], out[1], (out[2] if len(out) > 2 else None)
        return self._load(idx)

    def __getitem__(self, k):
        idx = self.ids[k]
        lr, hr, ex = self.load_boxes(idx)
        ex_t = torch.from_numpy(ex) if ex is not None else torch.zeros(0)
        return torch.from_numpy(lr), torch.from_numpy(hr), ex_t, idx
