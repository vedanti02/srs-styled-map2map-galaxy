"""Redshift-augmented pair dataset.

Wraps PairDataset and at __getitem__ time:
  - samples z ~ U[z_min, z_max]
  - computes linear-theory scalings
        s_disp(z, Om) = D(z, Om) / D(0, Om)
        s_vel(z, Om)  = D(z)·H(z)·f(z)/(1+z) / [D(0)·H(0)·f(0)]
    via map2map.norms.cosmology
  - scales the *normalized* LR and HR fields by these factors
    (consistent for both, so the SR task is preserved at scaled amplitude)
  - returns theta_6d = (Om, Ob, h, n_s, sigma_8, z)

Caveat: this is strictly linear theory. At small scales (high k), nonlinear
clustering does NOT follow D(z) scaling. For the NDE which uses k bins down to
~k_Ny ≈ 0.20 h/Mpc, the linear approximation is rough but order-correct.
Useful for plumbing a redshift dimension into the model; not a replacement for
real multi-snapshot training data.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from map2map.norms.cosmology import D, H, f
from data.pair_dataset import PairDataset


def disp_scale(z: float, Om: float) -> float:
    return float(D(z, Om=Om) / D(0.0, Om=Om))


def vel_scale(z: float, Om: float) -> float:
    num = D(z, Om=Om) * H(z, Om=Om) * f(z, Om=Om) / (1.0 + z)
    den = D(0.0, Om=Om) * H(0.0, Om=Om) * f(0.0, Om=Om)
    return float(num / den)


class PairDatasetZAug(Dataset):
    """LR/HR/θ pair with synthetic linear-growth redshift augmentation.

    theta returned is 6-D: (Om, Ob, h, n_s, sigma_8, z).
    """

    def __init__(self, stitched_root, split="train", train_frac=0.8, val_frac=0.1,
                 snap="PART_009", seed=0, normalize_inputs=True,
                 z_min=0.0, z_max=1.5, fixed_z=None):
        self.base = PairDataset(stitched_root, split=split, train_frac=train_frac,
                                val_frac=val_frac, snap=snap, seed=seed,
                                normalize_inputs=normalize_inputs)
        self.z_min = float(z_min)
        self.z_max = float(z_max)
        self.fixed_z = fixed_z  # if set, override sampling (useful for val/eval)
        # Per-worker RNG keyed off seed + worker id at __getitem__ time
        self._seed = seed

    @property
    def ids(self):
        return self.base.ids

    @ids.setter
    def ids(self, v):
        self.base.ids = v

    def __len__(self):
        return len(self.base)

    def _draw_z(self, sid: int) -> float:
        if self.fixed_z is not None:
            return float(self.fixed_z)
        # Per-call RNG mixing dataset seed + sid + a tiny bit of time variability
        # via numpy's global hash-of-tuple seeding pattern.
        rng = np.random.default_rng((self._seed, sid, np.random.SeedSequence().entropy & 0xFFFFFFFF))
        return float(rng.uniform(self.z_min, self.z_max))

    def __getitem__(self, i):
        x_lr, x_hr, theta5, sid = self.base[i]
        Om = float(theta5[0].item())
        z = self._draw_z(int(sid))

        s_d = disp_scale(z, Om)
        s_v = vel_scale(z, Om)

        x_lr = x_lr.clone()
        x_hr = x_hr.clone()
        x_lr[:3] *= s_d; x_hr[:3] *= s_d
        x_lr[3:] *= s_v; x_hr[3:] *= s_v

        theta6 = torch.cat([theta5, torch.tensor([z], dtype=theta5.dtype)], dim=0)
        return x_lr, x_hr, theta6, sid
