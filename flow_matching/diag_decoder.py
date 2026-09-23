"""Decoder diagnostic for the dequantised model: are the extra halos and the power deficit
produced by model-space samples crossing the floor() boundary (a spurious-halo mechanism that
dilutes delta and adds shot noise), or by genuinely wrong spatial structure?
Re-decodes the SAME model-space samples with a shifted threshold n = floor(z - shift) and
reports count error, T(k) and r(k) per shift. If a shift restores count AND T(k) AND r(k),
the velocity field is fine and the decoder is the problem.
  python -m flow_matching.diag_decoder --ckpt .../best.pt --n-boxes 10
"""
import argparse, numpy as np, torch
from flow_matching.data import CountPatchDataset
from flow_matching.common import build_from_ckpt, generate_box
from flow_matching.eval_spread import PkBins, delta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--n-boxes", type=int, default=10)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--shifts", default="0,0.1,0.2,0.3,0.4,0.5")
    a = ap.parse_args()
    dev = torch.device("cuda")
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    net, space, interp, args = build_from_ckpt(ck, dev)
    ds = CountPatchDataset("val", pad=0, max_sets=a.n_boxes)
    pkb = PkBins(); ok = pkb.n_modes > 0; k = pkb.k
    sel = [b for b in np.where(ok)[0] if k[b] > 0.012][::4]
    shifts = [float(s) for s in a.shifts.split(",")]
    res = {s: {"ce": [], "T": [], "r": [], "frac1": []} for s in shifts}
    hr_frac = []
    print(f"{len(ds)} val boxes, heun {a.steps} | {space}", flush=True)
    print("k-bins: " + " ".join(f"{k[b]:.3f}" for b in sel))
    for j, idx in enumerate(ds.ids):
        lr, hr = ds.load_boxes(idx)
        _, ms = generate_box(net, space, interp, lr, dev, n_draws=1, steps=a.steps, seed=idx, cond=args["cond"],
                             amp_dtype=torch.bfloat16, return_model_space=True)
        z = np.expm1(ms[0] * space.std + space.mean)                      # continuous counts (1,128^3)
        fh = pkb.fft(delta(hr)); Ph = pkb.auto(fh); hr_frac.append((hr >= 1).mean())
        for s in shifts:
            n = np.clip(np.floor(z - s + 1e-4), 0, 200)
            fs = pkb.fft(delta(n)); Ps = pkb.auto(fs)
            res[s]["ce"].append(n.sum() / hr.sum() - 1); res[s]["T"].append(Ps / Ph)
            res[s]["r"].append(pkb.cross(fs, fh) / np.sqrt(Ps * Ph)); res[s]["frac1"].append((n >= 1).mean())
    print(f"HR occupied-voxel fraction {np.mean(hr_frac):.4f}")
    for s in shifts:
        T = np.median(np.stack(res[s]["T"]), 0); r = np.median(np.stack(res[s]["r"]), 0)
        print(f"\n[shift {s:.1f}] count err {np.mean(res[s]['ce']):+.3f}  occupied frac {np.mean(res[s]['frac1']):.4f}")
        print("  T " + " ".join(f"{T[b]:.3f}" for b in sel)); print("  r " + " ".join(f"{r[b]:.3f}" for b in sel), flush=True)
    print("DIAG_DONE")


if __name__ == "__main__":
    main()
