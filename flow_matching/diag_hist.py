"""Mean one-point count histogram (the PQMass representation) per source on the val split,
plus per-bin ratio to HR. Tells WHICH count bins a decoder distorts.
  python -m flow_matching.diag_hist TAG=/path/to/srdir [TAG2=/path ...] --n-boxes 100
"""
import argparse, os, numpy as np
from paths import PROCESSED as PROC
CBINS = np.array([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 7.5, 11.5, 20.5, np.inf])
LAB = ["0", "1", "2", "3", "4", "5", "6-7", "8-11", "12-20", ">20"]


def hist(n):
    h, _ = np.histogram(np.asarray(n, np.float32).ravel(), bins=CBINS); return h / h.sum()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("sources", nargs="+"); ap.add_argument("--n-boxes", type=int, default=100)
    a = ap.parse_args()
    va = sorted(int(d["idx"]) for d in np.load(f"{PROC}/val_list.npy", allow_pickle=True))
    srcs = {"HR": None, "LR": None}
    for s in a.sources:
        t, p = s.split("="); srcs[t] = p
    ids = [i for i in va if all(p is None or os.path.exists(f"{p}/sr_{i}.npy") for p in srcs.values())][:a.n_boxes]
    print(f"{len(ids)} val boxes | bins: " + " ".join(f"{l:>7s}" for l in LAB) + "    mean count")
    H = {}
    for t, p in srcs.items():
        hs, means = [], []
        for i in ids:
            n = np.load(f"{PROC}/{i:04d}_label.npy") if t == "HR" else np.load(f"{PROC}/{i:04d}_input.npy") if t == "LR" else np.load(f"{p}/sr_{i}.npy")
            hs.append(hist(n)); means.append(float(np.asarray(n, np.float64).mean()))
        H[t] = (np.mean(hs, 0), np.std(hs, 0), np.mean(means))
    for t in H:
        m, s, mc = H[t]
        print(f"{t:>14s} frac  " + " ".join(f"{x:7.4f}" for x in m) + f"    {mc:.4f}")
    print("ratio to HR:")
    for t in H:
        if t == "HR": continue
        print(f"{t:>14s}       " + " ".join(f"{x/y if y>0 else float('nan'):7.3f}" for x, y in zip(H[t][0], H['HR'][0])))
    print("box-to-box std / HR std:")
    for t in H:
        if t == "HR": continue
        print(f"{t:>14s}       " + " ".join(f"{x/y if y>0 else float('nan'):7.3f}" for x, y in zip(H[t][1], H['HR'][1])))


if __name__ == "__main__":
    main()
