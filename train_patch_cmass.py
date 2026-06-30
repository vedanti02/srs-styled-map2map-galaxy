"""CMASS patch GAN: 1-channel halo-count fields, LR(FastPM)->HR(Nbody) per 64^3 patch,
stitch to 128^3. Eulerian count data (no Lagrangian/displacement) — see patch_dataset_cmass.

  --pad 0   Arm A (mentor 2.1): full-patch loss, naive stitch.
  --pad P>0 Arm B (mentor 2.2): halo'd (64+2P)^3 input (periodic wrap), loss on kept 64^3.

Model learns in `--transform` space (default log1p). Reconstruction = plain L1 (optionally
Gaussian-smoothed). Pk loss + stitched-128 Pk-RMS validation are computed on the overdensity
delta = n/nbar-1 of the physical counts (is_density=True). Keeps the v2 GAN recipe (R1).
"""
import argparse, os, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.patch_dataset_cmass import (
    PatchPairDatasetCmass, extract_patch, stitch_patches, crop_interior,
    to_counts, counts_to_delta, PATCH, N_PATCHES, N_FULL,
)
from map2map.models.styled_srsgan import G_correct, D_const
from analysis.pk_torch import TorchPk


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", default="checkpoints/patch_cmass_A/")
    p.add_argument("--pad", type=int, default=0)
    p.add_argument("--transform", default="log1p", choices=["log1p", "scale", "delta"])
    p.add_argument("--nbar", default="perbox", choices=["perbox", "global"])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--sims-per-batch", type=int, default=1, help="sims/step; patch-batch = 8×this")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--lr-g", type=float, default=1e-4)
    p.add_argument("--lr-d", type=float, default=1e-4)
    p.add_argument("--beta1", type=float, default=0.0)
    p.add_argument("--beta2", type=float, default=0.99)
    p.add_argument("--lambda-adv", type=float, default=1.0, help="0 disables the GAN (Arm A pure)")
    p.add_argument("--lambda-rec", type=float, default=2.0)
    p.add_argument("--lambda-pk", type=float, default=2.0)
    p.add_argument("--rec-smooth-sigma", type=float, default=0.0)
    p.add_argument("--lambda-r1", type=float, default=10.0)
    p.add_argument("--r1-every", type=int, default=16)
    p.add_argument("--chan-base-g", type=int, default=128)
    p.add_argument("--chan-base-d", type=int, default=64)
    p.add_argument("--num-blocks", type=int, default=4)
    p.add_argument("--save-every", type=int, default=5)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--max-train-sets", type=int, default=0)
    p.add_argument("--val-max-sims", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--resume", default="")
    p.add_argument("--lbox", type=float, default=1000.0)
    p.add_argument("--n-pk-bins", type=int, default=32)
    p.add_argument("--n-pk-bins-patch", type=int, default=24)
    return p.parse_args()


def softplus_loss_real(l): return F.softplus(-l).mean()
def softplus_loss_fake(l): return F.softplus(l).mean()


def r1_penalty(D, x, theta):
    x = x.detach().requires_grad_(True)
    g = torch.autograd.grad(D(x, theta).sum(), x, create_graph=True, retain_graph=True)[0]
    return g.pow(2).sum(dim=(1, 2, 3, 4)).mean()


def gauss_blur3d(x, sigma):
    """Separable periodic Gaussian blur (1-channel)."""
    r = max(1, int(round(3 * sigma)))
    k = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    g = torch.exp(-0.5 * (k / sigma) ** 2); g = g / g.sum()
    for d in range(3):
        shp = [1, 1, 1, 1, 1]; shp[2 + d] = 2 * r + 1
        ker = g.view(shp)
        pad = [0, 0, 0, 0, 0, 0]; pad[(2 - d) * 2] = r; pad[(2 - d) * 2 + 1] = r
        x = F.conv3d(F.pad(x, pad, mode="circular"), ker)
    return x


def rec_loss(fake, hr, sigma=0.0):
    if sigma > 0:
        fake, hr = gauss_blur3d(fake, sigma), gauss_blur3d(hr, sigma)
    return (fake - hr).abs().mean()


def hann3d(n, device):
    h = torch.hann_window(n, periodic=True, device=device)
    return (h[:, None, None] * h[None, :, None] * h[None, None, :]).view(1, 1, n, n, n)


def build_noise_list(grid, num_blocks, seed, device):
    rng = np.random.default_rng(seed)
    return [torch.from_numpy(rng.standard_normal((1, grid, grid, grid)).astype(np.float32)).to(device)
            for _ in range(2 * num_blocks)]


def _pk_pair(fake_model, hr_model, pk, win, transform, scale, nbar):
    """log Pk of fake & hr (model space -> counts -> overdensity), is_density."""
    df = counts_to_delta(to_counts(fake_model, transform, scale), nbar)
    dh = counts_to_delta(to_counts(hr_model, transform, scale), nbar)
    if win is not None:
        df, dh = df * win, dh * win
    return pk(df, is_density=True), pk(dh, is_density=True)


@torch.no_grad()
def stitched_validation(G, val_ds, pad, pk128, dev, num_blocks, transform, scale, nbar, max_sims):
    G.eval()
    grid = PATCH + 2 * pad
    nl = build_noise_list(grid, num_blocks, 0, dev)
    ids = val_ds.ids if max_sims <= 0 else val_ds.ids[:max_sims]
    l1, nvox, pks, pkn = 0.0, 0, 0.0, 0
    for idx in ids:
        lr, hr = val_ds.load_boxes(idx)                      # model space (1,128,128,128)
        patches = np.stack([extract_patch(lr, p, pad) for p in range(N_PATCHES)])
        xb = torch.from_numpy(patches).to(dev)
        tb = torch.from_numpy(val_ds.theta[idx]).unsqueeze(0).expand(N_PATCHES, -1).to(dev)
        fake = crop_interior(G(xb, tb, nl), pad)
        sr_t = torch.from_numpy(stitch_patches(fake.cpu().numpy())).unsqueeze(0).to(dev)  # (1,1,128³)
        hr_t = torch.from_numpy(hr).unsqueeze(0).to(dev)
        l1 += (sr_t - hr_t).abs().sum().item(); nvox += hr_t.numel()
        lpf, lph = _pk_pair(sr_t, hr_t, pk128, None, transform, scale, nbar)
        m = (lph > -10).float()
        pks += ((lpf - lph) ** 2 * m).sum().item(); pkn += m.sum().item()
    return l1 / nvox, (pks / max(pkn, 1)) ** 0.5


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pad = args.pad
    arm = "A/baseline (naive stitch)" if pad == 0 else f"B/boundary-masked (pad={pad}, overlap-tile)"

    train_ds = PatchPairDatasetCmass(split="train", pad=pad, transform=args.transform)
    val_ds = PatchPairDatasetCmass(split="val", pad=pad, transform=args.transform)
    if args.max_train_sets > 0:
        train_ds.ids = train_ds.ids[:args.max_train_sets]
    loader = DataLoader(train_ds, batch_size=args.sims_per_batch, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True, drop_last=True)

    G = G_correct(1, 1, 5, chan_base=args.chan_base_g, num_blocks=args.num_blocks).to(dev)
    D = D_const(1, 5, chan_base=args.chan_base_d, num_blocks=args.num_blocks).to(dev)
    opt_g = torch.optim.Adam(G.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))
    opt_d = torch.optim.Adam(D.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))

    pk64 = TorchPk(N=PATCH, lbox=args.lbox * PATCH / N_FULL, n_bins=args.n_pk_bins_patch, device=dev)
    pk128 = TorchPk(N=N_FULL, lbox=args.lbox, n_bins=args.n_pk_bins, device=dev)
    win = hann3d(PATCH, dev)
    scale = train_ds.scale
    nbar = None if args.nbar == "perbox" else scale
    use_gan = args.lambda_adv > 0

    start, best = 0, float("inf")
    if args.resume:
        c = torch.load(args.resume, map_location=dev, weights_only=False)
        G.load_state_dict(c["model"]); D.load_state_dict(c["D"])
        opt_g.load_state_dict(c["opt_g"]); opt_d.load_state_dict(c["opt_d"])
        start = c["epoch"] + 1; best = c.get("best_pk", float("inf"))
        print(f"resumed at epoch {start}")

    print(f"ARM {arm}  transform={args.transform} nbar={args.nbar} GAN={'on' if use_gan else 'OFF'}")
    print(f"G {sum(p.numel() for p in G.parameters())/1e6:.2f}M  D {sum(p.numel() for p in D.parameters())/1e6:.2f}M")
    print(f"train sims {len(train_ds.ids)}  val sims {len(val_ds.ids)}  input {PATCH+2*pad}³  "
          f"patch-batch {args.sims_per_batch*N_PATCHES}  dev {dev}")

    dstep = 0
    for epoch in range(start, args.epochs):
        G.train(); D.train(); t0 = time.time()
        for it, (lr_in, hr_tgt, theta, _) in enumerate(loader):
            B = lr_in.shape[0]
            x_lr = lr_in.reshape(B * N_PATCHES, 1, *lr_in.shape[3:]).to(dev, non_blocking=True)
            x_hr = hr_tgt.reshape(B * N_PATCHES, 1, PATCH, PATCH, PATCH).to(dev, non_blocking=True)
            th = theta.repeat_interleave(N_PATCHES, 0).to(dev, non_blocking=True)

            dr = df_ = 0.0
            if use_gan:
                opt_d.zero_grad(set_to_none=True)
                with torch.no_grad():
                    fake = crop_interior(G(x_lr, th), pad)
                lr_real, lr_fake = D(x_hr, th), D(fake, th)
                loss_d = softplus_loss_real(lr_real) + softplus_loss_fake(lr_fake)
                if args.lambda_r1 > 0 and dstep % args.r1_every == 0:
                    loss_d = loss_d + 0.5 * args.lambda_r1 * args.r1_every * r1_penalty(D, x_hr, th)
                loss_d.backward()
                if args.grad_clip > 0: torch.nn.utils.clip_grad_norm_(D.parameters(), args.grad_clip)
                opt_d.step(); dstep += 1
                dr, df_ = lr_real.mean().item(), lr_fake.mean().item()

            opt_g.zero_grad(set_to_none=True)
            fake = crop_interior(G(x_lr, th), pad)
            adv = softplus_loss_real(D(fake, th)) if use_gan else torch.tensor(0.0, device=dev)
            rec = rec_loss(fake, x_hr, args.rec_smooth_sigma)
            if args.lambda_pk > 0:
                lpf, lph = _pk_pair(fake, x_hr, pk64, win, args.transform, scale, nbar)
                lph = lph.detach()
                m = (lph > -10).float(); pk = ((lpf - lph) ** 2 * m).sum() / m.sum().clamp_min(1.0)
            else:
                pk = torch.tensor(0.0, device=dev)
            loss_g = args.lambda_adv * adv + args.lambda_rec * rec + args.lambda_pk * pk
            loss_g.backward()
            if args.grad_clip > 0: torch.nn.utils.clip_grad_norm_(G.parameters(), args.grad_clip)
            opt_g.step()

            if (it + 1) % args.log_every == 0:
                print(f"e{epoch} it{it+1}/{len(loader)}  adv={float(adv):.3f} "
                      f"rec={rec.item():.4f} pk={float(pk):.4f}  D(r)={dr:+.2f} D(f)={df_:+.2f}",
                      flush=True)

        vl1, vpk = stitched_validation(G, val_ds, pad, pk128, dev, args.num_blocks,
                                       args.transform, scale, nbar, args.val_max_sims)
        print(f"epoch {epoch} {time.time()-t0:.1f}s  STITCHED-128 val_L1/vox={vl1:.4f} val_pkRMS={vpk:.4f}", flush=True)

        st = {"model": G.state_dict(), "D": D.state_dict(), "opt_g": opt_g.state_dict(),
              "opt_d": opt_d.state_dict(), "epoch": epoch, "args": vars(args), "pad": pad,
              "val_l1": vl1, "val_pk_rms": vpk, "best_pk": best}
        if (epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs:
            torch.save(st, os.path.join(args.ckpt_dir, f"epoch_{epoch+1}.pt"))
        if vpk < best:
            best = vpk; st["best_pk"] = best
            torch.save(st, os.path.join(args.ckpt_dir, "best.pt"))
            print(f"saved best.pt (stitched val_pk_rms={vpk:.4f})")


if __name__ == "__main__":
    main()
