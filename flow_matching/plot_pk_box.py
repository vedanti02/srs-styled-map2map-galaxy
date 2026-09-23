"""Full-box (128^3, L=1000 Mpc/h) power-spectrum panel across corrector arms, from saved SR dirs:
P(k) median, transfer T(k)=P_X/P_HR (median + 16-84 band), cross-coherence r(k) with HR (median).
All on the count overdensity delta = n/nbar - 1 with per-box nbar, test split.
  python -m flow_matching.plot_pk_box LR=input GAN=/dir FM=/dir ... --n-boxes 100 --tag all_arms
"""
import argparse, os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from flow_matching.eval_spread import PkBins, delta
from paths import PROCESSED as PROC


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("sources", nargs="+"); ap.add_argument("--n-boxes", type=int, default=100)
    ap.add_argument("--split", default="test"); ap.add_argument("--tag", default="all_arms"); ap.add_argument("--out-dir", default="figures_cmass")
    a = ap.parse_args()
    srcs = dict(s.split("=") for s in a.sources)
    ids = sorted(int(d["idx"]) for d in np.load(f"{PROC}/{a.split}_list.npy", allow_pickle=True))
    ids = [i for i in ids if all(p == "input" or os.path.exists(f"{p}/sr_{i}.npy") for p in srcs.values())][:a.n_boxes]
    pkb = PkBins(); ok = pkb.n_modes > 0; k = pkb.k[ok]
    print(f"{len(ids)} {a.split} boxes; sources {list(srcs)}", flush=True)
    P = {t: [] for t in srcs}; T = {t: [] for t in srcs}; R = {t: [] for t in srcs}; Ph_all = []
    for j, i in enumerate(ids):
        hr = np.load(f"{PROC}/{i:04d}_label.npy").astype(np.float32); fh = pkb.fft(delta(hr)); Ph = pkb.auto(fh); Ph_all.append(Ph[ok])
        for t, p in srcs.items():
            x = np.load(f"{PROC}/{i:04d}_input.npy").astype(np.float32) if p == "input" else np.load(f"{p}/sr_{i}.npy").astype(np.float32)
            fx = pkb.fft(delta(x)); Px = pkb.auto(fx)
            P[t].append(Px[ok]); T[t].append((Px / Ph)[ok]); R[t].append((pkb.cross(fx, fh) / np.sqrt(Px * Ph))[ok])
        if (j + 1) % 20 == 0: print(f"  {j+1}/{len(ids)}", flush=True)
    np.savez(f"runs/patch_cmass/pk_box_{a.tag}.npz", k=k, Ph=np.array(Ph_all), **{f"P_{t}": np.array(v) for t, v in P.items()},
             **{f"T_{t}": np.array(v) for t, v in T.items()}, **{f"r_{t}": np.array(v) for t, v in R.items()}, ids=np.array(ids))
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    ax[0].plot(k, np.median(Ph_all, 0), "k-", lw=2, label="HR")
    print("k-bin  " + "  ".join(f"{t:>14s}" for t in srcs))
    for c, t in enumerate(srcs):
        Tm, Tlo, Thi = np.median(T[t], 0), np.percentile(T[t], 16, 0), np.percentile(T[t], 84, 0)
        ax[0].plot(k, np.median(P[t], 0), label=t, color=f"C{c}")
        ax[1].plot(k, Tm, color=f"C{c}", label=t); ax[1].fill_between(k, Tlo, Thi, color=f"C{c}", alpha=0.12)
        ax[2].plot(k, np.median(R[t], 0), color=f"C{c}", label=t)
        low = k < 0.1; print(f"{t:>14s}: T(k<0.1) median {np.median(np.array(T[t])[:, low]):.3f} | r at k~0.06 {np.median(np.array(R[t])[:, np.argmin(np.abs(k-0.062))]):.3f} | per-box RMS(T-1) {np.sqrt(np.mean((np.array(T[t])-1)**2)):.3f}")
    ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_title("P(k), median over boxes"); ax[0].set_xlabel("k [h/Mpc]"); ax[0].legend(fontsize=8)
    ax[1].axhline(1, c="k", lw=0.8); ax[1].set_xscale("log"); ax[1].set_ylim(0.7, 1.3); ax[1].set_title("transfer $P_X/P_{HR}$ (median, 16-84 band)"); ax[1].set_xlabel("k [h/Mpc]"); ax[1].legend(fontsize=8)
    ax[2].set_xscale("log"); ax[2].set_ylim(-0.05, 1.05); ax[2].set_title("cross-coherence with HR (median)"); ax[2].set_xlabel("k [h/Mpc]"); ax[2].legend(fontsize=8)
    fig.suptitle(f"Full 128$^3$ box, {len(ids)} {a.split} boxes"); fig.tight_layout()
    fig.savefig(f"{a.out_dir}/pk_box_{a.tag}.png", dpi=130); print(f"PKBOX_DONE {a.out_dir}/pk_box_{a.tag}.png")


if __name__ == "__main__":
    main()
