"""Real-scale patch dataset: the actual `quijote-64`/`quijotelike-64` 64³ patches,
assembled into the 128³ full box (2×2×2 grid). Same-resolution domain correction
(interpretation (b), confirmed by Saumya): LR 64³ patch → HR 64³ patch, stitch → 128³.

Only the 1666 sims with a matched 2×2×2 LR&HR grid are used. Indexed BY SIM (not by
patch) so each sim's 8 patches are read once per epoch; the trainer flattens the
8-patch axis into the batch.

Patch index p = unravel(p,(2,2,2)) → origin (64i,64j,64k). With pad>0 the halo is the
periodic-wrap neighbourhood of the assembled 128³ box (real neighbour data on internal
faces). Verified by `python -m data.patch_dataset_real`.
"""
import os
import re
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import Counter

# HR-patch per-channel std (disp x3, vel x3), measured over the quijote-64 patches.
# Differs from the stitched constants because patches retain small-scale power.
CHAN_STD = np.array([31.1, 30.2, 30.8, 336.0, 341.0, 335.0], dtype=np.float32)

N_FULL = 128          # assembled box
N_SPLIT = 2
PATCH = 64            # native patch size
N_PATCHES = N_SPLIT ** 3   # 8
DATA_ROOT = "/data/group_data/universedata/lagrangian_output_64"


def normalize(x):   return x / CHAN_STD.reshape(6, 1, 1, 1)
def denormalize(x): return x * CHAN_STD.reshape(6, 1, 1, 1)


def _scan_clean_sids(root):
    """Sids whose LR AND HR both have a full 2×2×2 (=8) patch grid."""
    def grid(kind):
        per = Counter()
        for e in os.listdir(os.path.join(root, kind)):
            m = re.match(r"set(\d+)_pos", e)
            if m:
                per[int(m.group(1))] += 1
        return per
    hr, lr = grid("quijote-64"), grid("quijotelike-64")
    return sorted(s for s in (set(hr) & set(lr)) if hr[s] == N_PATCHES and lr[s] == N_PATCHES)


def _load_patch(root, kind, sid, i, j, k, snap):
    d = os.path.join(root, kind, f"set{sid}_pos_{i}_{j}_{k}", snap)
    disp = np.load(os.path.join(d, "disp.npy"))
    vel = np.load(os.path.join(d, "vel.npy"))
    return np.concatenate([disp, vel], axis=0).astype(np.float32)


def assemble_box(root, kind, sid, snap="PART_009"):
    """Assemble the 8 patches into (6, 128, 128, 128)."""
    box = np.empty((6, N_FULL, N_FULL, N_FULL), dtype=np.float32)
    for p in range(N_PATCHES):
        i, j, k = np.unravel_index(p, (N_SPLIT,) * 3)
        box[:, i*PATCH:(i+1)*PATCH, j*PATCH:(j+1)*PATCH, k*PATCH:(k+1)*PATCH] = \
            _load_patch(root, kind, sid, i, j, k, snap)
    return box


def extract_patch(box, p, pad=0):
    """Patch p from the 128³ box with periodic-wrap halo of `pad`: (6,64+2p,...)."""
    ijk = np.unravel_index(p, (N_SPLIT,) * 3)
    x = box
    for d, i in enumerate(ijk):
        lo = i * PATCH - pad
        x = x.take(range(lo, lo + PATCH + 2 * pad), axis=1 + d, mode="wrap")
    return np.ascontiguousarray(x)


def stitch_patches(patches):
    """8 patches (6,64,64,64) in index order → (6,128,128,128)."""
    patches = np.asarray(patches)
    assert patches.shape[0] == N_PATCHES and patches.shape[-1] == PATCH, patches.shape
    out = np.empty((patches.shape[1], N_FULL, N_FULL, N_FULL), dtype=patches.dtype)
    for p in range(N_PATCHES):
        i, j, k = np.unravel_index(p, (N_SPLIT,) * 3)
        out[:, i*PATCH:(i+1)*PATCH, j*PATCH:(j+1)*PATCH, k*PATCH:(k+1)*PATCH] = patches[p]
    return out


def crop_interior(x, pad):
    if pad == 0:
        return x
    return x[..., pad:-pad, pad:-pad, pad:-pad]


class PatchPairDatasetReal(Dataset):
    """Indexed by SIM. __getitem__ → (lr_in, hr_tgt, theta, sid):
        lr_in:  (8, 6, 64+2*pad, ...)  halo'd LR patches (model input)
        hr_tgt: (8, 6, 64, 64, 64)     HR tiles (target / placed)
    """

    def __init__(self, root=DATA_ROOT, split="train", pad=0,
                 train_frac=0.8, val_frac=0.1, snap="PART_009", seed=0,
                 normalize_inputs=True):
        self.root = root
        self.snap = snap
        self.pad = pad
        self.normalize_inputs = normalize_inputs
        ids = _scan_clean_sids(root)
        rng = np.random.default_rng(seed)
        rng.shuffle(ids)
        n = len(ids); ntr = int(train_frac*n); nval = int(val_frac*n)
        self.ids = {"train": ids[:ntr], "val": ids[ntr:ntr+nval],
                    "test": ids[ntr+nval:], "all": ids}[split]
        self.ids = sorted(self.ids)

    def __len__(self):
        return len(self.ids)

    def load_boxes(self, sid):
        lr = assemble_box(self.root, "quijotelike-64", sid, self.snap)
        hr = assemble_box(self.root, "quijote-64", sid, self.snap)
        theta = np.load(os.path.join(self.root, "quijote-64",
                        f"set{sid}_pos_0_0_0", self.snap, "style.npy")).astype(np.float32)
        if self.normalize_inputs:
            lr, hr = normalize(lr), normalize(hr)
        return lr, hr, theta

    def __getitem__(self, idx):
        sid = self.ids[idx]
        lr, hr, theta = self.load_boxes(sid)
        lr_in = np.stack([extract_patch(lr, p, self.pad) for p in range(N_PATCHES)])
        hr_tgt = np.stack([extract_patch(hr, p, 0) for p in range(N_PATCHES)])
        return (torch.from_numpy(lr_in), torch.from_numpy(hr_tgt),
                torch.from_numpy(theta), sid)


if __name__ == "__main__":
    ds = PatchPairDatasetReal(split="all", pad=3)
    print(f"clean 2x2x2 sims: {len(ds.ids)}  "
          f"(train {len(PatchPairDatasetReal(split='train').ids)} / "
          f"val {len(PatchPairDatasetReal(split='val').ids)} / "
          f"test {len(PatchPairDatasetReal(split='test').ids)})")
    sid = ds.ids[0]
    lr, hr, theta = ds.load_boxes(sid)
    print(f"assembled box shape: {hr.shape}  theta {theta}")

    # 1) bit-exact reconstruction (no halo)
    rec = stitch_patches([extract_patch(hr, p, 0) for p in range(N_PATCHES)])
    assert rec.shape == hr.shape and np.array_equal(rec, hr), "stitch mismatch"
    print("PASS: assemble → tile(pad0) → stitch reproduces the 128³ box bit-exactly")

    # 2) halo interior == no-halo patch; halo == periodic neighbour
    pad = 3
    for p in (0, 3, 7):
        full = extract_patch(hr, p, pad)
        assert full.shape == (6,) + (PATCH + 2*pad,) * 3, full.shape
        assert np.array_equal(crop_interior(full, pad), extract_patch(hr, p, 0))
    full0 = extract_patch(hr, 0, pad)   # patch0 face at x=0 wraps to box row 128-pad
    assert np.array_equal(full0[:, 0, pad:-pad, pad:-pad], hr[:, N_FULL-pad, 0:PATCH, 0:PATCH])
    print("PASS: pad=3 halo == true periodic neighbourhood; interior crop == pad0 patch")

    # 3) halo'd → crop → stitch == box
    rec2 = stitch_patches([crop_interior(extract_patch(hr, p, pad), pad) for p in range(N_PATCHES)])
    assert np.array_equal(rec2, hr)
    print("PASS: halo'd extract → crop → stitch reproduces the 128³ box bit-exactly")

    # 4) dataset item shapes
    lr_in, hr_tgt, th, s = ds[0]
    print(f"item: lr_in {tuple(lr_in.shape)}  hr_tgt {tuple(hr_tgt.shape)}  theta {tuple(th.shape)}  sid {s}")
    assert lr_in.shape == (8, 6, 70, 70, 70) and hr_tgt.shape == (8, 6, 64, 64, 64)
    print("PASS: dataset item shapes correct (pad=3 → 70³ input, 64³ target)")
