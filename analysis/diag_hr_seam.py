"""Decisive diagnostic: is the patch seam a DATA artifact or a MODEL artifact?

Measures the GROUND-TRUTH field's own discontinuity across the internal patch
faces (x/y/z = 64) vs interior, for the assembled HR (quijote-64) and LR
(quijotelike-64) boxes. If HR itself jumps at 64, the target is discontinuous
=> data problem. If HR is smooth at 64, the seam in SR error is a model (RF) problem.
"""
import numpy as np
from data.patch_dataset_real import PatchPairDatasetReal, assemble_box, DATA_ROOT

ds = PatchPairDatasetReal(split="test", pad=0, normalize_inputs=False)
sids = ds.ids[:8]


def plane_jump(box, ax):
    """Per-plane mean |field(n) - field(n-1)| along axis `ax` (periodic), disp(0:3) & vel(3:6)."""
    out = {}
    for name, sl in (("disp", slice(0, 3)), ("vel", slice(3, 6))):
        f = box[sl]
        g = np.abs(f - np.roll(f, 1, axis=ax + 1))  # +1: channel axis is 0
        out[name] = g.mean(axis=tuple(a for a in range(4) if a != ax + 1))  # -> (128,)
    return out


for kind, label in (("quijote-64", "HR"), ("quijotelike-64", "LR")):
    agg = {("disp", a): [] for a in range(3)}
    agg.update({("vel", a): [] for a in range(3)})
    for sid in sids:
        box = assemble_box(DATA_ROOT, kind, sid)
        for ax in range(3):
            pj = plane_jump(box, ax)
            for fld in ("disp", "vel"):
                agg[(fld, ax)].append(pj[fld])
    print(f"\n=== {label} ({kind}); mean over {len(sids)} test sims ===")
    for fld in ("disp", "vel"):
        for ax, axn in enumerate("xyz"):
            P = np.mean(agg[(fld, ax)], axis=0)
            interior = np.concatenate([P[8:25], P[40:57]]).mean()
            r0, r64 = P[0] / interior, P[64] / interior
            print(f"  {fld} {axn}: jump@{axn}=0 {r0:5.2f}x   jump@{axn}=64 {r64:5.2f}x   "
                  f"(interior abs {interior:.4f})")
