"""Quantify how well the stitched SR matches HR (count field), beyond the power spectrum.
Reports, averaged over N test boxes: total-count ratio, void fraction, real-space cell
std ratio (clustering amplitude), and the cross-correlation r(k) at large vs small scales.
Numbers are written to runs/patch_cmass/match_stats_<tag>.txt for FINAL.md."""
import argparse, os
import numpy as np
import torch

from map2map.models.styled_srsgan import G_correct
from data.patch_dataset_cmass import (
    PatchPairDatasetCmass, extract_patch, stitch_patches, crop_interior,
    to_counts, counts_to_delta, PATCH, N_PATCHES, N_FULL,
)
from analysis.plot_diagnostic import cross_pk


def build_noise_list(grid, nb, seed, dev):
    rng = np.random.default_rng(seed)
    return [torch.from_numpy(rng.standard_normal((1, grid, grid, grid)).astype(np.float32)).to(dev)
            for _ in range(2 * nb)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--mode", default="naive", choices=["naive", "overlap"])
    p.add_argument("--tag", default="cmassA")
    p.add_argument("--n-sims", type=int, default=16)
    p.add_argument("--lbox", type=float, default=1000.0)
    args = p.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    s = ck.get("args", {}) or {}
    cb, nb = s.get("chan_base_g", 128), s.get("num_blocks", 4)
    transform = s.get("transform", "log1p")
    pad = 0 if args.mode == "naive" else s.get("pad", 8)
    G = G_correct(1, 1, 5, chan_base=cb, num_blocks=nb).to(dev); G.load_state_dict(ck["model"]); G.eval()
    nl = build_noise_list(PATCH + 2 * pad, nb, 0, dev)
    ds = PatchPairDatasetCmass(split="test", pad=pad, transform=transform)
    ds_raw = PatchPairDatasetCmass(split="test", pad=0, normalize_inputs=False)
    ids = ds.ids[:args.n_sims]

    tot, voidH, voidS, stdR, rlo, rhi = [], [], [], [], [], []
    klo, khi = None, None
    with torch.no_grad():
        for idx in ids:
            lr_m, _ = ds.load_boxes(idx); _, hr = ds_raw.load_boxes(idx)
            patches = np.stack([extract_patch(lr_m, q, pad) for q in range(N_PATCHES)])
            xb = torch.from_numpy(patches).to(dev)
            th = torch.from_numpy(ds.theta[idx]).unsqueeze(0).expand(N_PATCHES, -1).to(dev)
            fake = crop_interior(G(xb, th, nl), pad).cpu().numpy()
            sr = stitch_patches(to_counts(fake, transform, ds.scale))
            h, sc = hr[0], sr[0]
            tot.append(sc.sum() / max(h.sum(), 1))
            voidH.append(float((h < 0.5).mean())); voidS.append(float((sc < 0.5).mean()))
            stdR.append(sc.std() / max(h.std(), 1e-6))
            k, Pa, Pb, Pab, m = cross_pk(counts_to_delta(h), counts_to_delta(sc), args.lbox, n_bins=24)
            r = np.zeros_like(k); r[m] = Pab[m] / np.sqrt(np.maximum(Pa * Pb, 1e-60))[m]
            rlo.append(np.nanmean(r[(k > 0) & (k < 0.05)])); rhi.append(np.nanmean(r[k > 0.2]))

    def ms(x): return float(np.mean(x)), float(np.std(x))
    lines = [f"# SR vs HR match stats ({args.tag}), {len(ids)} test boxes",
             f"total count ratio sum(SR)/sum(HR) : {ms(tot)[0]:.3f} +/- {ms(tot)[1]:.3f}",
             f"void fraction HR (cells <0.5)     : {ms(voidH)[0]:.3f}",
             f"void fraction SR (cells <0.5)     : {ms(voidS)[0]:.3f}",
             f"real-space std ratio std(SR)/std(HR): {ms(stdR)[0]:.3f} +/- {ms(stdR)[1]:.3f}",
             f"cross-corr r(k) large scales k<0.05: {ms(rlo)[0]:.3f}",
             f"cross-corr r(k) small scales k>0.2 : {ms(rhi)[0]:.3f}"]
    out = f"runs/patch_cmass/match_stats_{args.tag}.txt"
    os.makedirs("runs/patch_cmass", exist_ok=True)
    open(out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines)); print("wrote", out)


if __name__ == "__main__":
    main()
