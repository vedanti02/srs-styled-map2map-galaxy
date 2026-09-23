"""Spread / calibration diagnostics for a stochastic corrector, from K draws per box.

Reads <srdir>/sr_{idx}.npy (draw 0) and sr_{idx}_d{k}.npy, HR/LR from the processed dir.
Per k-bin (32 log bins on the count overdensity, as everywhere else in the repo):
  T(k)   = P_SR/P_HF   median + 16-84 band over boxes x draws (amplitude; LF for reference)
  r(k)   = P_x,HF / sqrt(P_x P_HF)   cross-coherence of draw 0 and of LF (phase)
  SSR(k) = <std_draws logP> / RMSE(<logP>_draws, logP_HF)   spread-skill ratio, ideal 1
  rank histogram of logP_HF among the K draws, pooled over low / mid / high k
Plus the per-voxel generation variance and the box-to-box scatter of the count-histogram
features (the PQMass representation) for HR, LR, SR.

  python -m flow_matching.eval_spread <srdir> <tag> --draws 8 --n-boxes 60 --split val
"""
import argparse, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

from paths import PROCESSED as PROC
CBINS = np.array([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 7.5, 11.5, 20.5, np.inf])


def hist_feat(n):
    n = np.asarray(n, np.float32).ravel(); h, _ = np.histogram(n, bins=CBINS); h = h / h.sum()
    return np.concatenate([h, [np.log1p(n.mean())]]).astype(np.float32)


def delta(n):
    c = np.asarray(n, np.float64); c = c[0] if c.ndim == 4 else c
    return c / max(c.mean(), 1e-6) - 1.0


class PkBins:
    def __init__(self, N=128, lbox=1000.0, n_bins=32):
        kx = np.fft.fftfreq(N, d=lbox / N) * 2 * np.pi
        self.kgrid = np.sqrt(kx[:, None, None] ** 2 + kx[None, :, None] ** 2 + kx[None, None, :] ** 2)
        k_nyq, k_min = np.pi * N / lbox, 2 * np.pi / lbox
        self.bins = np.logspace(np.log10(k_min * 1.01), np.log10(k_nyq), n_bins + 1)
        self.n_modes, _ = np.histogram(self.kgrid, bins=self.bins)
        ks, _ = np.histogram(self.kgrid, bins=self.bins, weights=self.kgrid)
        self.k = np.where(self.n_modes > 0, ks / np.maximum(self.n_modes, 1), 0.0)
        self.N, self.lbox = N, lbox

    def fft(self, d):
        return np.fft.fftn(d) / self.N ** 3

    def bin(self, grid):
        s, _ = np.histogram(self.kgrid, bins=self.bins, weights=grid)
        return np.where(self.n_modes > 0, s / np.maximum(self.n_modes, 1), np.nan) * self.lbox ** 3

    def auto(self, fk):
        return self.bin(np.abs(fk) ** 2)

    def cross(self, fa, fb):
        return self.bin((fa * np.conj(fb)).real)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("srdir"); ap.add_argument("tag")
    ap.add_argument("--draws", type=int, default=8)
    ap.add_argument("--n-boxes", type=int, default=60)
    ap.add_argument("--split", default="val")
    ap.add_argument("--out-dir", default="figures_cmass")
    ap.add_argument("--npz-dir", default="runs/patch_cmass")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True); os.makedirs(a.npz_dir, exist_ok=True)
    ids = sorted(int(d["idx"]) for d in np.load(f"{PROC}/{a.split}_list.npy", allow_pickle=True))

    def draw_path(i, k):
        return f"{a.srdir}/sr_{i}.npy" if k == 0 else f"{a.srdir}/sr_{i}_d{k}.npy"
    ids = [i for i in ids if all(os.path.exists(draw_path(i, k)) for k in range(a.draws))][:a.n_boxes]
    K = a.draws
    print(f"spread eval [{a.tag}]: {len(ids)} {a.split} boxes x {K} draws from {a.srdir}", flush=True)
    pkb = PkBins()
    nb = len(pkb.k)
    logP_hr = np.zeros((len(ids), nb)); logP_lr = np.zeros_like(logP_hr); logP_sr = np.zeros((len(ids), K, nb))
    r_sr = np.zeros((len(ids), nb)); r_lr = np.zeros_like(r_sr)
    genvar = np.zeros(len(ids)); feats = {"HR": [], "LR": [], "SR": []}
    for j, i in enumerate(ids):
        hr = np.load(f"{PROC}/{i:04d}_label.npy").astype(np.float32)
        lr = np.load(f"{PROC}/{i:04d}_input.npy").astype(np.float32)
        srs = np.stack([np.load(draw_path(i, k)).astype(np.float32) for k in range(K)])
        fh = pkb.fft(delta(hr)); fl = pkb.fft(delta(lr))
        Ph = pkb.auto(fh); Pl = pkb.auto(fl)
        logP_hr[j] = np.log(Ph); logP_lr[j] = np.log(Pl)
        r_lr[j] = pkb.cross(fl, fh) / np.sqrt(Pl * Ph)
        for k in range(K):
            fs = pkb.fft(delta(srs[k])); Ps = pkb.auto(fs)
            logP_sr[j, k] = np.log(Ps)
            if k == 0:
                r_sr[j] = pkb.cross(fs, fh) / np.sqrt(Ps * Ph)
        genvar[j] = np.log1p(srs).std(0).mean() / max(np.log1p(hr).std(), 1e-6)
        feats["HR"].append(hist_feat(hr)); feats["LR"].append(hist_feat(lr)); feats["SR"].append(hist_feat(srs[0]))
        if (j + 1) % 10 == 0:
            print(f"  {j+1}/{len(ids)}", flush=True)

    k = pkb.k; ok = pkb.n_modes > 0
    T_sr = np.exp(logP_sr - logP_hr[:, None]); T_lr = np.exp(logP_lr - logP_hr)
    mean_sr = logP_sr.mean(1)
    ssr = logP_sr.std(1, ddof=1).mean(0) / np.sqrt(((mean_sr - logP_hr) ** 2).mean(0)) if K > 1 else np.full(nb, np.nan)
    # rank of HR among draws, per (box, k-bin)
    ranks = (logP_sr < logP_hr[:, None]).sum(1)                       # 0..K
    thirds = np.array_split(np.where(ok)[0], 3)
    rank_hists = [np.bincount(ranks[:, sel].ravel(), minlength=K + 1) / ranks[:, sel].size for sel in thirds]
    F = {s: np.stack(v) for s, v in feats.items()}
    scat = {s: F[s].std(0) for s in F}

    def med_band(x):
        return np.nanmedian(x, 0), np.nanpercentile(x, 16, 0), np.nanpercentile(x, 84, 0)
    Tm, Tlo, Thi = med_band(T_sr.reshape(-1, nb)); Lm, Llo, Lhi = med_band(T_lr)
    print("\nk-bin      T_SR(med) [16,84]      T_LR(med)   r_SR    r_LR    SSR")
    for b in np.where(ok)[0][::3]:
        print(f"{k[b]:.3f}  {Tm[b]:.3f} [{Tlo[b]:.3f},{Thi[b]:.3f}]  {Lm[b]:.3f}   {np.median(r_sr[:, b]):.3f}  {np.median(r_lr[:, b]):.3f}  {ssr[b]:.2f}")
    lowk = ok & (k < 0.1)
    print(f"\nlarge-scale (k<0.1) T_SR median {np.nanmedian(T_sr[..., lowk]):.3f}  T_LR {np.nanmedian(T_lr[:, lowk]):.3f}")
    print(f"SSR mean over k: {np.nanmean(ssr[ok]):.3f}  (1 = calibrated; <1 under-dispersed)")
    print(f"per-voxel generation std across draws / HR field std: {genvar.mean():.4f}")
    print(f"rank-hist (low/mid/high k), ideal flat = {1/(K+1):.3f}:")
    for name, h in zip(["low", "mid", "high"], rank_hists):
        print(f"  {name}: " + " ".join(f"{x:.3f}" for x in h))
    print("box-to-box std of count-histogram features (bins " + ",".join(str(int(c)) if np.isfinite(c) else "inf" for c in CBINS[1:]) + " ; logmean):")
    for s in ("HR", "LR", "SR"):
        print(f"  {s}: " + " ".join(f"{x:.4f}" for x in scat[s]))
    np.savez(f"{a.npz_dir}/spread_{a.tag}.npz", k=k, ok=ok, T_sr=T_sr, T_lr=T_lr, r_sr=r_sr, r_lr=r_lr, ssr=ssr,
             ranks=ranks, genvar=genvar, feats_hr=F["HR"], feats_lr=F["LR"], feats_sr=F["SR"], ids=np.array(ids))

    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    ax[0, 0].fill_between(k[ok], Tlo[ok], Thi[ok], alpha=0.3, label="SR 16-84 (boxes x draws)")
    ax[0, 0].plot(k[ok], Tm[ok], label="SR median"); ax[0, 0].plot(k[ok], Lm[ok], "--", label="LF median")
    ax[0, 0].axhline(1, c="k", lw=0.8); ax[0, 0].set_xscale("log"); ax[0, 0].set_ylim(0.6, 1.4); ax[0, 0].set_title("transfer T(k)"); ax[0, 0].legend()
    ax[0, 1].plot(k[ok], np.median(r_sr, 0)[ok], label="SR draw0"); ax[0, 1].plot(k[ok], np.median(r_lr, 0)[ok], "--", label="LF")
    ax[0, 1].set_xscale("log"); ax[0, 1].set_ylim(0, 1.05); ax[0, 1].set_title("cross-coherence r(k)"); ax[0, 1].legend()
    ax[0, 2].plot(k[ok], ssr[ok]); ax[0, 2].axhline(1, c="k", lw=0.8); ax[0, 2].set_xscale("log"); ax[0, 2].set_title("spread-skill ratio (1 = calibrated)")
    for name, h in zip(["low k", "mid k", "high k"], rank_hists):
        ax[1, 0].plot(np.arange(K + 1), h, marker="o", label=name)
    ax[1, 0].axhline(1 / (K + 1), c="k", lw=0.8); ax[1, 0].set_title("rank of HR logP among draws"); ax[1, 0].legend()
    x = np.arange(len(CBINS))
    for s in ("HR", "LR", "SR"):
        ax[1, 1].plot(x, scat[s], marker="o", label=s)
    ax[1, 1].set_yscale("log"); ax[1, 1].set_title("box-to-box std of count-hist features"); ax[1, 1].legend()
    ax[1, 2].hist(genvar, bins=20); ax[1, 2].set_title("per-voxel generation std / HR std")
    fig.suptitle(f"{a.tag}: {len(ids)} {a.split} boxes x {K} draws"); fig.tight_layout()
    fig.savefig(f"{a.out_dir}/spread_{a.tag}.png", dpi=120)
    print(f"SPREAD_DONE -> {a.out_dir}/spread_{a.tag}.png", flush=True)


if __name__ == "__main__":
    main()
