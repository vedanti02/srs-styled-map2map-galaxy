"""Symmetry augmentation for periodic cubes, applied identically to LF and HF.

Groups:
  none   : identity only
  los16  : the 16 ops that keep the last axis (line of sight) fixed. Use this if the
           count fields are in redshift space (RSD breaks isotropy along the LOS).
  full48 : the full octahedral group (6 axis permutations x 8 reflections). Valid for
           real-space fields, which are statistically isotropic.
Run flow_matching/check_isotropy.py on the HR boxes to decide between los16 and full48.
"""
import itertools
import torch

PERMS = list(itertools.permutations(range(3)))                      # 6
FLIPS = [(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)]  # 8


def all_ops(group: str):
    if group == "none":
        return [((0, 1, 2), (0, 0, 0))]
    if group == "full48":
        return [(p, f) for p in PERMS for f in FLIPS]
    if group == "los16":
        return [(p, f) for p in PERMS if p[2] == 2 for f in FLIPS]
    raise ValueError(f"unknown augmentation group {group!r}")


def random_op(group: str, generator=None):
    ops = all_ops(group)
    i = int(torch.randint(len(ops), (1,), generator=generator).item())
    return ops[i]


def apply_op(x: torch.Tensor, op):
    """x: (B, C, D, H, W). Permute spatial axes then flip the selected ones."""
    perm, flips = op
    x = x.permute(0, 1, 2 + perm[0], 2 + perm[1], 2 + perm[2])
    dims = [2 + i for i in range(3) if flips[i]]
    if dims:
        x = x.flip(dims)
    return x.contiguous()
