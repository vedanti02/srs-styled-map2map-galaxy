"""CMASS-ILI patch dataset — Eulerian halo COUNT fields (no Lagrangian/displacement).

Drop-in analogue of data/patch_dataset_real.py, but on the cmass-ili data:
  * source: /data/group_data/universedata/cmass-ili/processed/{idx:04d}_{input,label}.npy
      input = LR (FastPM), label = HR (Quijote N-body); each (128,128,128) integer
      halo-count field on a PERIODIC L=1000 Mpc/h box. idx == lhid (0..1999).
  * cosmology theta: quijote/nbody/L1000-N128/{idx}/config.yaml -> nbody.cosmo (5 params).
  * splits: processed/{train,val,test}_list.npy (lists of dicts with 'idx').

Patching = the SAME overlap + periodic-wrap scheme as patch_dataset_real.extract_patch:
  2x2x2 grid of 64^3 CORES (stride 64, tile the 128^3 box exactly); each patch is the
  core grown by `pad` on every side via periodic wrap -> (64+2*pad)^3 input. Neighbours
  overlap by 2*pad; a window that runs off the box edge wraps from the opposite side.
  HR target is the bare 64^3 core (pad=0). Stitching places cores -> exact 128^3.

Because each sim is ONE continuous periodic box that we crop ourselves, the patches are
real neighbours by construction -> no data seam (unlike the lagrangian patches). Verified
by `python -m data.patch_dataset_cmass`.
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset

# reuse the exact, channel-agnostic patch ops from the lagrangian pipeline
from data.patch_dataset_real import extract_patch, stitch_patches, crop_interior

N_FULL = 128
N_SPLIT = 2
PATCH = 64
N_PATCHES = N_SPLIT ** 3            # 8
DATA_ROOT = "/data/group_data/universedata/cmass-ili"
PROCESSED = os.path.join(DATA_ROOT, "processed")
NBODY = os.path.join(DATA_ROOT, "quijote", "nbody", "L1000-N128")
THETA_CACHE = os.path.join(os.path.dirname(__file__), "cmass_theta.npz")


def _read_cosmo(idx):
    """5 cosmo params for sim idx from its config.yaml (no yaml dep: parse the cosmo block)."""
    p = os.path.join(NBODY, str(idx), "config.yaml")
    with open(p) as f:
        lines = f.readlines()
    vals, grab = [], False
    for ln in lines:
        s = ln.strip()
        if s.startswith("cosmo:"):
            grab = True
            continue
        if grab:
            if s.startswith("- "):
                vals.append(float(s[2:]))
            else:
                break
    assert len(vals) == 5, f"idx {idx}: parsed {vals}"
    return np.array(vals, dtype=np.float32)


def build_theta_cache(n=2000):
    """idx -> (5,) cosmo table + a global count scale (mean count over train sims), cached."""
    theta = np.stack([_read_cosmo(i) for i in range(n)])
    # count scale from a sample of train sims (preserve absolute amplitude => one global scale)
    tr = [d["idx"] for d in np.load(os.path.join(PROCESSED, "train_list.npy"), allow_pickle=True)]
    sample = tr[:50]
    scale = float(np.mean([np.load(f"{PROCESSED}/{i:04d}_label.npy").mean() for i in sample]))
    np.savez(THETA_CACHE, theta=theta, scale=np.float32(scale))
    return theta, scale


def _load_theta():
    if not os.path.exists(THETA_CACHE):
        return build_theta_cache()
    z = np.load(THETA_CACHE)
    return z["theta"], float(z["scale"])


# --------- reversible count transforms (shared by trainer / inference / seam eval) ---------
# Count fields are sparse non-negative integers. The CNN learns in "model space"
# (default log1p); Pk is always computed on the overdensity delta=n/nbar-1 of the
# physical counts. These helpers accept numpy arrays OR torch tensors.

def _is_torch(x):
    return hasattr(x, "clamp_min")


def to_model_space(counts, transform="log1p", scale=None):
    """Physical counts -> model space (what the generator predicts)."""
    if transform == "log1p":
        return torch.log1p(counts) if _is_torch(counts) else np.log1p(counts)
    if transform == "scale":
        assert scale, "transform=scale needs a scale"
        return counts / scale
    if transform == "delta":          # model space IS the overdensity (experimental)
        return counts_to_delta(counts)
    raise ValueError(transform)


def to_counts(model, transform="log1p", scale=None):
    """Inverse of to_model_space: model space -> physical counts (clamped >= 0)."""
    if transform == "log1p":
        if _is_torch(model):
            return torch.expm1(model.clamp_min(0.0))
        return np.expm1(np.clip(model, 0.0, None))
    if transform == "scale":
        assert scale, "transform=scale needs a scale"
        if _is_torch(model):
            return model.clamp_min(0.0) * scale
        return np.clip(model, 0.0, None) * scale
    if transform == "delta":          # delta -> counts: n = nbar*(1+delta); nbar unknown, use (1+delta)
        if _is_torch(model):
            return (1.0 + model).clamp_min(0.0)
        return np.clip(1.0 + model, 0.0, None)
    raise ValueError(transform)


def counts_to_delta(counts, nbar=None, eps=1e-6):
    """Counts -> overdensity delta = n/nbar - 1. Per-sample mean for batched torch
    (B,*); global mean for a single numpy cube."""
    if _is_torch(counts):
        if nbar is None:
            dims = tuple(range(1, counts.dim())) if counts.dim() > 3 else tuple(range(counts.dim()))
            nbar = counts.mean(dim=dims, keepdim=True).clamp_min(eps)
        return counts / nbar - 1.0
    arr = np.asarray(counts, dtype=np.float64)
    if nbar is None:
        nbar = max(float(arr.mean()), eps)
    return (arr / nbar - 1.0).astype(np.float32)


def _split_ids(split):
    if split == "all":
        return list(range(2000))
    lst = np.load(os.path.join(PROCESSED, f"{split}_list.npy"), allow_pickle=True)
    return sorted(int(d["idx"]) for d in lst)


class PatchPairDatasetCmass(Dataset):
    """Indexed by SIM. __getitem__ -> (lr_in, hr_tgt, theta, idx):
        lr_in:  (8, 1, 64+2*pad, ...)  overlap+wrap LR patches (model input)
        hr_tgt: (8, 1, 64, 64, 64)     HR cores (target)
    """

    def __init__(self, split="train", pad=8, normalize_inputs=True, transform="log1p"):
        self.pad = pad
        self.normalize_inputs = normalize_inputs
        self.transform = transform
        self.ids = _split_ids(split)
        self.theta, self.scale = _load_theta()

    def __len__(self):
        return len(self.ids)

    def load_boxes(self, idx):
        lr = np.load(f"{PROCESSED}/{idx:04d}_input.npy").astype(np.float32)[None]  # (1,128,128,128)
        hr = np.load(f"{PROCESSED}/{idx:04d}_label.npy").astype(np.float32)[None]
        if self.normalize_inputs:
            lr = to_model_space(lr, self.transform, self.scale)
            hr = to_model_space(hr, self.transform, self.scale)
        return lr, hr

    def __getitem__(self, k):
        idx = self.ids[k]
        lr, hr = self.load_boxes(idx)
        lr_in = np.stack([extract_patch(lr, p, self.pad) for p in range(N_PATCHES)])
        hr_tgt = np.stack([extract_patch(hr, p, 0) for p in range(N_PATCHES)])
        return (torch.from_numpy(lr_in), torch.from_numpy(hr_tgt),
                torch.from_numpy(self.theta[idx]), idx)


if __name__ == "__main__":
    print("building/loading theta cache ...")
    theta, scale = _load_theta()
    print(f"theta {theta.shape}; count scale (mean HR count/cell) = {scale:.5f}")
    print(f"splits: train {len(_split_ids('train'))} / val {len(_split_ids('val'))} "
          f"/ test {len(_split_ids('test'))} / all {len(_split_ids('all'))}")

    pad = 8
    ds = PatchPairDatasetCmass(split="test", pad=pad, normalize_inputs=False)
    idx = ds.ids[0]
    lr, hr = ds.load_boxes(idx)
    print(f"\nsim {idx}: box {hr.shape}  theta {theta[idx]}")

    # 1) reconstruction bit-exact (cores tile the box)
    rec = stitch_patches([extract_patch(hr, p, 0) for p in range(N_PATCHES)])
    assert rec.shape == hr.shape and np.array_equal(rec, hr)
    print("PASS: 8x 64^3 cores stitch back to the 128^3 box bit-exactly")

    # 2) count conservation
    assert abs(rec.sum() - hr.sum()) < 1e-3
    print(f"PASS: counts conserved (box total {hr.sum():.0f})")

    # 3) overlap+wrap halo == true periodic neighbourhood
    full0 = extract_patch(hr, 0, pad)
    assert full0.shape == (1, PATCH + 2 * pad, PATCH + 2 * pad, PATCH + 2 * pad)
    assert np.array_equal(crop_interior(full0, pad), extract_patch(hr, 0, 0))
    # patch 0 left face wraps to the opposite side of the box
    assert np.array_equal(full0[:, 0, pad:-pad, pad:-pad], hr[:, N_FULL - pad, 0:PATCH, 0:PATCH])
    print(f"PASS: pad={pad} halo is the true periodic-wrap neighbourhood; interior==core")

    # 4) NO SEAM: cross-boundary correlation at x=64 vs interior (continuous box => ~equal)
    def corr(a, b): return np.corrcoef(a.ravel(), b.ravel())[0, 1]
    cb = corr(hr[0, 63], hr[0, 64]); ci = corr(hr[0, 31], hr[0, 32])
    print(f"PASS-CHECK: corr across core boundary x=64 = {cb:+.3f} vs interior x=32 = {ci:+.3f} "
          f"(should be ~equal => no seam)")

    # 5) theta sanity + dataset item shapes
    assert np.allclose(theta[0], [0.1755, 0.06681, 0.7737, 0.8849, 0.6641], atol=1e-4)
    print("PASS: theta[0] matches config.yaml cosmo")
    lr_in, hr_tgt, th, sid = ds[0]
    print(f"item: lr_in {tuple(lr_in.shape)}  hr_tgt {tuple(hr_tgt.shape)}  theta {tuple(th.shape)}  idx {sid}")
    assert lr_in.shape == (8, 1, 64 + 2 * pad, 64 + 2 * pad, 64 + 2 * pad)
    assert hr_tgt.shape == (8, 1, 64, 64, 64)
    print("PASS: dataset item shapes correct (pad=8 -> 80^3 input, 64^3 target)")
