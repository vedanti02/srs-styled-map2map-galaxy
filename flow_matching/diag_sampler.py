"""Sampler diagnostic: is the power deficit a model or an integration problem?
For N val boxes and several (method, steps) configs: K draws each, then per k-bin
  T(k) of single draws (median), T(k) of the K-draw MEAN field, r(k) of draw0, r(k) of the mean,
  per-voxel draw-to-draw std / HR std, total-count error.
  python -m flow_matching.diag_sampler --ckpt .../best.pt --n-boxes 10 --draws 4
"""
import argparse, time
import numpy as np
import torch
from flow_matching.data import CountPatchDataset
from flow_matching.common import build_from_ckpt, generate_box
from flow_matching.eval_spread import PkBins, delta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--n-boxes", type=int, default=10)
    ap.add_argument("--draws", type=int, default=4)
    ap.add_argument("--configs", default="heun:16,heun:32,heun:64,heun:128,euler:50")
    ap.add_argument("--no-ema", action="store_true")
    a = ap.parse_args()
    dev = torch.device("cuda")
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    if a.no_ema: ck["ema"] = None
    net, space, interp, args = build_from_ckpt(ck, dev)
    ds = CountPatchDataset("val", pad=0, max_sets=a.n_boxes)
    pkb = PkBins(); ok = pkb.n_modes > 0; k = pkb.k
    sel = [b for b in np.where(ok)[0] if k[b] > 0.012][::4]
    boxes = [ds.load_boxes(i) for i in ds.ids]
    fh = [pkb.fft(delta(hr)) for _, hr in boxes]; Ph = [pkb.auto(f) for f in fh]
    fl = [pkb.fft(delta(lr)) for lr, _ in boxes]
    print(f"{len(boxes)} val boxes, {a.draws} draws | ckpt epoch {ck.get('epoch')} ema={not a.no_ema}", flush=True)
    print("LF reference: T(k) " + " ".join(f"{np.median([pkb.auto(fl[j])[b]/Ph[j][b] for j in range(len(boxes))]):.3f}" for b in sel))
    print("              r(k) " + " ".join(f"{np.median([pkb.cross(fl[j], fh[j])[b]/np.sqrt(pkb.auto(fl[j])[b]*Ph[j][b]) for j in range(len(boxes))]):.3f}" for b in sel))
    print("k-bins:            " + " ".join(f"{k[b]:.3f}" for b in sel), flush=True)
    for cfg in a.configs.split(","):
        method, steps = cfg.split(":"); steps = int(steps); t0 = time.time()
        Td, Tm, rd, rm, gv, ce = [], [], [], [], [], []
        for j, (lr, hr) in enumerate(boxes):
            srs = generate_box(net, space, interp, lr, dev, n_draws=a.draws, steps=steps, method=method,
                               seed=ds.ids[j], cond=args["cond"], amp_dtype=torch.bfloat16)
            fs0 = pkb.fft(delta(srs[0])); Ps0 = pkb.auto(fs0)
            mean_field = np.mean(np.stack(srs), 0); fsm = pkb.fft(delta(mean_field)); Psm = pkb.auto(fsm)
            Td.append(Ps0 / Ph[j]); Tm.append(Psm / Ph[j])
            rd.append(pkb.cross(fs0, fh[j]) / np.sqrt(Ps0 * Ph[j])); rm.append(pkb.cross(fsm, fh[j]) / np.sqrt(Psm * Ph[j]))
            gv.append(np.log1p(np.stack(srs)).std(0).mean() / np.log1p(hr).std()); ce.append(srs[0].sum() / hr.sum() - 1)
        Td, Tm, rd, rm = (np.median(np.stack(x), 0) for x in (Td, Tm, rd, rm))
        print(f"\n[{method} {steps} steps] {time.time()-t0:.0f}s  voxel-std/HR {np.mean(gv):.3f}  count err {np.mean(ce):+.3f}")
        print("  T draw    " + " ".join(f"{Td[b]:.3f}" for b in sel))
        print("  T mean(K) " + " ".join(f"{Tm[b]:.3f}" for b in sel))
        print("  r draw    " + " ".join(f"{rd[b]:.3f}" for b in sel))
        print("  r mean(K) " + " ".join(f"{rm[b]:.3f}" for b in sel), flush=True)
    print("DIAG_DONE")


if __name__ == "__main__":
    main()
