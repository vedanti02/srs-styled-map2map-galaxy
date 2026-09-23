"""Are the count fields in real space (isotropic) or redshift space (anisotropic along the LOS)?
Decides the augmentation group: full48 (isotropic) vs los16 (LOS = last axis fixed).
Compares the power in modes within 25 deg of each axis, per k-bin, over a few HR boxes.
  python -m flow_matching.check_isotropy --n-boxes 10
"""
import argparse, numpy as np
PROC = "/data/group_data/universedata/cmass-ili/processed"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n-boxes", type=int, default=10); ap.add_argument("--which", default="label")
    a = ap.parse_args()
    ids = sorted(int(d["idx"]) for d in np.load(f"{PROC}/train_list.npy", allow_pickle=True))[:a.n_boxes]
    N, L = 128, 1000.0
    kx = np.fft.fftfreq(N, d=L / N) * 2 * np.pi
    KX, KY, KZ = np.meshgrid(kx, kx, kx, indexing="ij")
    kk = np.sqrt(KX ** 2 + KY ** 2 + KZ ** 2); kk[0, 0, 0] = 1e-12
    bins = np.logspace(np.log10(2 * np.pi / L * 1.01), np.log10(np.pi * N / L), 13)
    cos_lim = np.cos(np.radians(25))
    masks = [np.abs(K) / kk > cos_lim for K in (KX, KY, KZ)]
    acc = np.zeros((3, len(bins) - 1)); cnt = np.zeros((3, len(bins) - 1))
    for i in ids:
        n = np.load(f"{PROC}/{i:04d}_{a.which}.npy").astype(np.float64)
        d = n / n.mean() - 1.0
        p = np.abs(np.fft.fftn(d) / N ** 3) ** 2
        for ax in range(3):
            s, _ = np.histogram(kk[masks[ax]], bins=bins, weights=p[masks[ax]]); c, _ = np.histogram(kk[masks[ax]], bins=bins)
            acc[ax] += s; cnt[ax] += c
    P = acc / np.maximum(cnt, 1)
    print(f"{a.n_boxes} {a.which} boxes. P along axis / mean of the other two, per k-bin (1.00 = isotropic):")
    print("k        x       y       z")
    for b in range(len(bins) - 1):
        if cnt[0, b] == 0: continue
        m = P[:, b].mean()
        print(f"{np.sqrt(bins[b]*bins[b+1]):.3f}  " + "  ".join(f"{P[ax, b]/m:6.3f}" for ax in range(3)))
    ratio_z = (P[2] / np.maximum(0.5 * (P[0] + P[1]), 1e-30))
    w = cnt[2] * (np.sqrt(bins[:-1] * bins[1:]) > 0.05)          # skip the cosmic-variance-dominated lowest bins
    rz = float(np.sum(ratio_z * w) / max(np.sum(w), 1))
    print(f"\nmode-weighted P_z / P_xy over bins with k > 0.05: {rz:.3f}  (lowest bins have too few modes per axis "
          f"to be informative). Within a few % of 1.00 -> real space, use --aug full48; clearly != 1 -> redshift space "
          f"along z, use --aug los16.")


if __name__ == "__main__":
    main()
