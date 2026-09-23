"""How much does each corrector actually change the LF count field, voxel by voxel?
Fraction of voxels where SR != LR, split by LR count, plus the mean |SR-LR| in counts.
  python -m flow_matching.diag_edit_fraction TAG=/srdir [...] --n-boxes 50
"""
import argparse, os, numpy as np
PROC = "/data/group_data/universedata/cmass-ili/processed"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("sources", nargs="+"); ap.add_argument("--n-boxes", type=int, default=50)
    a = ap.parse_args()
    va = sorted(int(d["idx"]) for d in np.load(f"{PROC}/val_list.npy", allow_pickle=True))
    srcs = dict(s.split("=") for s in a.sources)
    ids = [i for i in va if all(os.path.exists(f"{p}/sr_{i}.npy") for p in srcs.values())][:a.n_boxes]
    print(f"{len(ids)} val boxes. Columns: overall changed fraction | changed fraction among LR=0,1,2,3+ | mean |SR-LR| | HR!=LR fraction")
    for t, p in srcs.items():
        ch, byb, mad, hrch = [], [], [], []
        for i in ids:
            lr = np.load(f"{PROC}/{i:04d}_input.npy").astype(np.int32); hr = np.load(f"{PROC}/{i:04d}_label.npy").astype(np.int32)
            sr = np.load(f"{p}/sr_{i}.npy").astype(np.int32)
            d = sr != lr; ch.append(d.mean()); hrch.append((hr != lr).mean()); mad.append(np.abs(sr - lr).mean())
            byb.append([d[lr == 0].mean(), d[lr == 1].mean(), d[lr == 2].mean(), d[lr >= 3].mean()])
        b = np.mean(byb, 0)
        print(f"{t:>16s}: {np.mean(ch):.4f} | " + " ".join(f"{x:.3f}" for x in b) + f" | {np.mean(mad):.4f} | {np.mean(hrch):.4f}")


if __name__ == "__main__":
    main()
