"""Diagnostic plots for the best SR model:

  1) Power-spectrum panel (P(k) median+band, transfer Pred/Truth, cross-power).
  2) 2D density-projection panel for one held-out sim:
     LR input, HR truth, SR pred, residual (Truth-Pred)/sqrt(Pred).

Density is computed by CIC-painting the lattice + displacement field.
"""
import argparse
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from map2map.models.styled_srsgan import G_correct
from data.pair_dataset import PairDataset, denormalize
from analysis.power_spectrum import cic_density


def cross_pk(delta_a, delta_b, lbox, n_bins=64):
    N = delta_a.shape[-1]
    a_k = np.fft.fftn(delta_a) / N**3
    b_k = np.fft.fftn(delta_b) / N**3
    pk_aa = (np.abs(a_k) ** 2) * (lbox ** 3)
    pk_bb = (np.abs(b_k) ** 2) * (lbox ** 3)
    pk_ab = (a_k * np.conj(b_k)).real * (lbox ** 3)
    kx = np.fft.fftfreq(N, d=lbox / N) * 2 * np.pi
    kgrid = np.sqrt(kx[:, None, None] ** 2 + kx[None, :, None] ** 2 + kx[None, None, :] ** 2)
    k_nyq = np.pi * N / lbox
    k_min = 2 * np.pi / lbox
    bins = np.logspace(np.log10(k_min * 1.01), np.log10(k_nyq * 1.05), n_bins + 1)
    n_modes, _ = np.histogram(kgrid, bins=bins)
    k_sum, _ = np.histogram(kgrid, bins=bins, weights=kgrid)
    paa, _ = np.histogram(kgrid, bins=bins, weights=pk_aa)
    pbb, _ = np.histogram(kgrid, bins=bins, weights=pk_bb)
    pab, _ = np.histogram(kgrid, bins=bins, weights=pk_ab)
    mask = n_modes > 0
    k = np.zeros(n_bins); a = np.zeros(n_bins); b = np.zeros(n_bins); c = np.zeros(n_bins)
    k[mask] = k_sum[mask] / n_modes[mask]
    a[mask] = paa[mask] / n_modes[mask]
    b[mask] = pbb[mask] / n_modes[mask]
    c[mask] = pab[mask] / n_modes[mask]
    return k, a, b, c, mask


def run_inference(G, x_lr_norm, theta, device):
    G.eval()
    with torch.no_grad():
        x_in = torch.from_numpy(x_lr_norm).unsqueeze(0).to(device)
        th = torch.from_numpy(theta).unsqueeze(0).to(device)
        out = G(x_in, th).squeeze(0).cpu().numpy()
    return denormalize(out)


def make_pk_panel(results, k_nyq, outpath, title="Power Spectrum"):
    Ks = np.stack([r[0] for r in results])
    Ts = np.stack([r[1] for r in results])
    Ps = np.stack([r[2] for r in results])
    Rs = np.stack([r[3] for r in results])
    k_med = np.nanmedian(Ks, axis=0)
    valid = (k_med > 0)
    k_med = k_med[valid]
    Ts = Ts[:, valid]; Ps = Ps[:, valid]; Rs = Rs[:, valid]
    t_med = np.nanmedian(Ts, axis=0)
    p_med = np.nanmedian(Ps, axis=0); p_lo = np.nanpercentile(Ps, 16, axis=0); p_hi = np.nanpercentile(Ps, 84, axis=0)
    tf = Ps / np.maximum(Ts, 1e-30)
    tf_med = np.nanmedian(tf, axis=0); tf_lo = np.nanpercentile(tf, 16, axis=0); tf_hi = np.nanpercentile(tf, 84, axis=0)
    r_med = np.nanmedian(Rs, axis=0); r_lo = np.nanpercentile(Rs, 16, axis=0); r_hi = np.nanpercentile(Rs, 84, axis=0)

    fig, axes = plt.subplots(3, 1, figsize=(7.5, 11), sharex=True)

    ax = axes[0]
    ax.plot(k_med, t_med, color="red", label="Truth", lw=2)
    ax.fill_between(k_med, p_lo, p_hi, color="C0", alpha=0.3)
    ax.plot(k_med, p_med, color="C0", label="Pred median", lw=2)
    ax.axvline(k_nyq, color="gray", lw=1.5, label="Nyquist (approx)")
    ax.set_yscale("log"); ax.set_xscale("log")
    ax.set_ylabel("P(k)"); ax.set_title(title)
    ax.legend(loc="best", fontsize=10)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    ax.fill_between(k_med, tf_lo, tf_hi, color="C0", alpha=0.3)
    ax.plot(k_med, tf_med, color="C0", lw=2)
    ax.axhline(1.0, color="k", lw=1, ls="--")
    ax.axvline(k_nyq, color="gray", lw=1.5)
    ax.set_yscale("log"); ax.set_xscale("log")
    ax.set_ylim(1e-2, 1e2)
    ax.set_ylabel("Transfer (Pred/Truth)")
    ax.grid(alpha=0.3, which="both")

    ax = axes[2]
    ax.fill_between(k_med, r_lo, r_hi, color="C0", alpha=0.3)
    ax.plot(k_med, r_med, color="C0", lw=2)
    ax.axhline(1.0, color="k", lw=1, ls="--")
    ax.axhline(0.0, color="k", lw=1, ls="--")
    ax.axvline(k_nyq, color="gray", lw=1.5)
    ax.set_xscale("log")
    ax.set_ylim(-0.25, 1.1)
    ax.set_xlabel("k [1/Box]"); ax.set_ylabel("Cross-power")
    ax.grid(alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig(outpath, dpi=120)
    print(f"saved {outpath}")
    plt.close(fig)


def make_projection_panel(delta_lr, delta_hr, delta_sr, outpath):
    """4-panel image like the reference: LR / HR / SR / residual.
    Uses log(1+delta) projection summed along z for nice contrast."""
    def proj(d):
        s = np.log1p(np.maximum(d, -0.999)).sum(axis=2)
        return s

    p_lr = proj(delta_lr); p_hr = proj(delta_hr); p_sr = proj(delta_sr)
    # residual a la reference: (Truth - Pred) / sqrt(Pred)
    safe = np.maximum(p_sr - p_sr.min() + 1e-3, 1e-3)
    resid = (p_hr - p_sr) / np.sqrt(safe)

    vmin = min(p_lr.min(), p_hr.min(), p_sr.min())
    vmax = max(np.percentile(p_lr, 99.5),
               np.percentile(p_hr, 99.5),
               np.percentile(p_sr, 99.5))

    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5))
    titles = ["Input Histogram", "Ground Truth Histogram", "Predicted Histogram",
              r"Residual ${\rm(Truth - Pred)/\sqrt{Pred}}$"]
    for ax, img, t in zip(axes[:3], [p_lr, p_hr, p_sr], titles[:3]):
        ax.imshow(img, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(t, fontsize=16)
        ax.set_xticks([]); ax.set_yticks([])
    rabs = max(abs(np.percentile(resid, 1)), abs(np.percentile(resid, 99)))
    im = axes[3].imshow(resid, origin="lower", cmap="bwr", vmin=-rabs, vmax=rabs)
    axes[3].set_title(titles[3], fontsize=16)
    axes[3].set_xticks([]); axes[3].set_yticks([])
    plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(outpath, dpi=120)
    print(f"saved {outpath}")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/v2/best.pt")
    p.add_argument("--data-root", default="/data/group_data/universedata/lagrangian_output_64/stitched")
    p.add_argument("--out-dir", default="runs/baseline/plots/diagnostic_v2_best")
    p.add_argument("--n-sims", type=int, default=20)
    p.add_argument("--projection-sim-idx", type=int, default=0)
    p.add_argument("--lbox", type=float, default=1000.0)
    p.add_argument("--n-bins", type=int, default=64)
    p.add_argument("--tag", default="v2_best")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    sd = ck["model"]
    style_size = 5
    for k, v in sd.items():
        if k.endswith("style_block.0.weight"):
            style_size = int(v.shape[1])
            break
    G = G_correct(in_chan=6, out_chan=6, style_size=style_size, chan_base=256, num_blocks=4).to(device)
    G.load_state_dict(sd)
    G.eval()
    print(f"loaded {args.ckpt} (epoch {ck.get('epoch', '?')}) style_size={style_size} on {device}", flush=True)

    if style_size == 6:
        from data.pair_dataset_zaug import PairDatasetZAug
        ds = PairDatasetZAug(args.data_root, split="val", seed=0, fixed_z=0.0)
    else:
        ds = PairDataset(args.data_root, split="val", seed=0)
    N = 64
    lbox = args.lbox
    k_nyq = np.pi * N / lbox

    results = []
    cached = None
    for i in range(min(args.n_sims, len(ds))):
        x_lr_norm, x_hr_norm, theta, sid = ds[i]
        x_lr_norm_np = x_lr_norm.numpy() if hasattr(x_lr_norm, "numpy") else np.asarray(x_lr_norm)
        x_hr_norm_np = x_hr_norm.numpy() if hasattr(x_hr_norm, "numpy") else np.asarray(x_hr_norm)
        theta_np = theta.numpy() if hasattr(theta, "numpy") else np.asarray(theta)
        x_lr = denormalize(x_lr_norm_np)
        x_hr = denormalize(x_hr_norm_np)
        x_sr = run_inference(G, x_lr_norm_np, theta_np, device)

        d_hr = cic_density(x_hr[:3], lbox)
        d_sr = cic_density(x_sr[:3], lbox)
        d_lr = cic_density(x_lr[:3], lbox)

        k, Pa, Pb, Pab, mask = cross_pk(d_hr, d_sr, lbox, n_bins=args.n_bins)
        r_corr = np.zeros_like(k)
        denom = np.sqrt(np.maximum(Pa * Pb, 1e-60))
        r_corr[mask] = Pab[mask] / denom[mask]
        results.append((k, Pa, Pb, r_corr))

        if i == args.projection_sim_idx:
            cached = (d_lr, d_hr, d_sr, sid)
        print(f"  sim {i+1}/{args.n_sims} (set{sid}) done", flush=True)

    make_pk_panel(results, k_nyq, os.path.join(args.out_dir, f"pk_panel_{args.tag}.png"),
                  title=f"Power Spectrum ({args.tag})")

    if cached is not None:
        d_lr, d_hr, d_sr, sid = cached
        make_projection_panel(d_lr, d_hr, d_sr,
                              os.path.join(args.out_dir, f"projection_{args.tag}_set{sid}.png"))


if __name__ == "__main__":
    main()
