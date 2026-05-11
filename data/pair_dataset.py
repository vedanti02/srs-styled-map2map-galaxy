import os
import re
import numpy as np
import torch
from torch.utils.data import Dataset


_SET_RE = re.compile(r"^set(\d+)_(quijote|quijotelike)$")

# Per-channel global std across the stitched dataset (computed over 100 sims).
# Channels: [dx, dy, dz, vx, vy, vz]. Mean offsets are negligible relative to std
# and we keep zero ↔ no displacement, so we only z-score by std.
CHAN_STD = np.array([18.0, 18.0, 18.0, 210.0, 210.0, 210.0], dtype=np.float32)


def normalize(x):
    """x: (6, ...) array → divide each channel by CHAN_STD."""
    return x / CHAN_STD.reshape(6, 1, 1, 1)


def denormalize(x):
    """Inverse of normalize."""
    return x * CHAN_STD.reshape(6, 1, 1, 1)


def _scan_set_ids(stitched_root):
    quijote_ids, like_ids = set(), set()
    for entry in os.listdir(stitched_root):
        m = _SET_RE.match(entry)
        if not m:
            continue
        set_id, kind = int(m.group(1)), m.group(2)
        (quijote_ids if kind == "quijote" else like_ids).add(set_id)
    return sorted(quijote_ids & like_ids)


def _load_6ch(set_dir, snap):
    disp = np.load(os.path.join(set_dir, snap, "disp.npy"))
    vel = np.load(os.path.join(set_dir, snap, "vel.npy"))
    return np.concatenate([disp, vel], axis=0).astype(np.float32)


class PairDataset(Dataset):
    """Paired (LR=quijotelike, HR=quijote, theta=style) loader for `stitched/`.

    Each sample: x_LR (6,64,64,64), x_HR (6,64,64,64), theta (5,).
    """

    def __init__(self, stitched_root, split="train",
                 train_frac=0.8, val_frac=0.1, snap="PART_009", seed=0,
                 normalize_inputs=True):
        self.root = stitched_root
        self.snap = snap
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
        return len(self.ids)

    def __getitem__(self, i):
        sid = self.ids[i]
        lr_dir = os.path.join(self.root, f"set{sid}_quijotelike")
        hr_dir = os.path.join(self.root, f"set{sid}_quijote")

        x_lr = _load_6ch(lr_dir, self.snap)
        x_hr = _load_6ch(hr_dir, self.snap)
        theta = np.load(os.path.join(lr_dir, self.snap, "style.npy")).astype(np.float32)

        if self.normalize_inputs:
            x_lr = normalize(x_lr)
            x_hr = normalize(x_hr)

        return (
            torch.from_numpy(x_lr),
            torch.from_numpy(x_hr),
            torch.from_numpy(theta),
            sid,
        )
