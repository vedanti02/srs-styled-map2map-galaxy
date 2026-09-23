"""Does widening the source noise at sampling time (temperature) fill the missing high-count tail?
For N val boxes and several source scales: count-histogram ratio to HR, mean count, T(k), r(k).
  python -m flow_matching.diag_temp --ckpt .../best.pt --scales 1.0,1.1,1.2,1.35 --n-boxes 10
"""
import argparse, copy, numpy as np, torch
from flow_matching.data import CountPatchDataset
from flow_matching.common import build_from_ckpt, generate_box
from flow_matching.eval_spread import PkBins, delta
from flow_matching.diag_hist import hist, LAB


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--n-boxes", type=int, default=10)
    ap.add_argument("--scales", default="1.0"); ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--sde-gammas", default="", help="comma list; each runs the SDE sampler at scale 1")
    a = ap.parse_args()
    dev = torch.device("cuda")
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    net, space, interp, args = build_from_ckpt(ck, dev)
    ds = CountPatchDataset("val", pad=0, max_sets=a.n_boxes)
    pkb = PkBins(); ok = pkb.n_modes > 0; k = pkb.k
    sel = [b for b in np.where(ok)[0] if k[b] > 0.012][::4]
    boxes = [ds.load_boxes(i) for i in ds.ids]
    Hh = np.mean([hist(hr) for _, hr in boxes], 0); mc_h = np.mean([hr.mean() for _, hr in boxes])
    print(f"{len(boxes)} val boxes | {interp} | bins " + " ".join(f"{l:>6s}" for l in LAB[:8]))
    print(f"HR mean count {mc_h:.4f}")
    for sc in [float(x) for x in a.scales.split(",")]:
        it = copy.deepcopy(interp); it.sigma_scale = sc
        Hs, mcs, T, r = [], [], [], []
        for j, (lr, hr) in enumerate(boxes):
            sr = generate_box(net, space, it, lr, dev, n_draws=1, steps=a.steps, seed=ds.ids[j], cond=args["cond"], amp_dtype=torch.bfloat16)[0]
            Hs.append(hist(sr)); mcs.append(sr.mean())
            fs, fh = pkb.fft(delta(sr)), pkb.fft(delta(hr)); Ps, Ph = pkb.auto(fs), pkb.auto(fh)
            T.append(Ps / Ph); r.append(pkb.cross(fs, fh) / np.sqrt(Ps * Ph))
        rat = np.mean(Hs, 0) / Hh; T = np.median(np.stack(T), 0); r = np.median(np.stack(r), 0)
        print(f"\n[scale {sc:.2f}] mean count {np.mean(mcs):.4f} ({np.mean(mcs)/mc_h-1:+.3f})")
        print("  hist/HR " + " ".join(f"{x:6.3f}" for x in rat[:8]))
        print("  T(k)    " + " ".join(f"{T[b]:.3f}" for b in sel) + "   r(k) " + " ".join(f"{r[b]:.3f}" for b in sel), flush=True)
    for gm in [float(x) for x in a.sde_gammas.split(",") if x]:
        Hs, mcs, T, r = [], [], [], []
        for j, (lr, hr) in enumerate(boxes):
            sr = generate_box(net, space, interp, lr, dev, n_draws=1, steps=a.steps, method="sde", sde_gamma=gm,
                              seed=ds.ids[j], cond=args["cond"], amp_dtype=torch.bfloat16)[0]
            Hs.append(hist(sr)); mcs.append(sr.mean())
            fs, fh = pkb.fft(delta(sr)), pkb.fft(delta(hr)); Ps, Ph = pkb.auto(fs), pkb.auto(fh)
            T.append(Ps / Ph); r.append(pkb.cross(fs, fh) / np.sqrt(Ps * Ph))
        rat = np.mean(Hs, 0) / Hh; T = np.median(np.stack(T), 0); r = np.median(np.stack(r), 0)
        print(f"\n[SDE gamma {gm:.2f}, {a.steps} steps] mean count {np.mean(mcs):.4f} ({np.mean(mcs)/mc_h-1:+.3f})")
        print("  hist/HR " + " ".join(f"{x:6.3f}" for x in rat[:8]))
        print("  T(k)    " + " ".join(f"{T[b]:.3f}" for b in sel) + "   r(k) " + " ".join(f"{r[b]:.3f}" for b in sel), flush=True)
    print("DIAG_DONE")


if __name__ == "__main__":
    main()
