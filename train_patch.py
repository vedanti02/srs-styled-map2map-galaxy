"""Patch-wise GAN training on 32^3 tiles of the stitched 64^3 cubes (mentor task 2).

One script, two arms (identical code path => fair A/B):

  --pad 0   Arm A (baseline, mentor 2.1): plain 32^3 tiles, losses on the full
            patch. Deployment = naive stitching of 8 patch outputs.
  --pad P>0 Arm B (mentor 2.2): input is a halo'd (32+2P)^3 tile (periodic-wrap
            halo = TRUE neighbouring data, the full box is periodic). ALL losses
            (L1, adversarial, Pk) are computed only on the central 32^3 interior
            — i.e. the boundary voxels that get cut out in post-processing are
            ignored, exactly as the mentor proposed. Deployment = overlap-tile
            stitching (crop halo, place interior).

Loss recipe = train_v2.py (our GAN SOTA): softplus GAN + lazy R1 + weighted L1
(w_disp/w_vel) + log-Pk MSE. The patch Pk term uses a 3D Hann window on both
fake and real (a 32^3 patch is not periodic; apodization suppresses the edge
leakage in the FFT — both sides windowed identically so the mismatch signal
survives).

Validation each epoch is on the STITCHED 64^3 cubes of the val split, in the
arm's own deployment mode, with a fixed noise list (deterministic). Best
checkpoint = lowest stitched val pkRMS. This directly implements "train ...
based on performance on the stitched result" for model selection.
"""
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.patch_dataset import (
    PatchPairDataset, extract_patch, stitch_patches, crop_interior,
    PATCH, N_PATCHES,
)
from map2map.models.styled_srsgan import G_correct, D_const
from analysis.pk_torch import TorchPk


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="/data/group_data/universedata/lagrangian_output_64/stitched/")
    p.add_argument("--ckpt-dir", default="checkpoints/patch_v1/")
    p.add_argument("--pad", type=int, default=0,
                   help="0: Arm A baseline (full-patch loss). >0: Arm B halo "
                        "input + boundary-masked loss (interior 32^3 only).")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--lr-g", type=float, default=1e-4)
    p.add_argument("--lr-d", type=float, default=1e-4)
    p.add_argument("--beta1", type=float, default=0.0)
    p.add_argument("--beta2", type=float, default=0.99)
    p.add_argument("--lambda-rec", type=float, default=10.0)
    p.add_argument("--lambda-pk", type=float, default=2.0)
    p.add_argument("--w-disp", type=float, default=2.0)
    p.add_argument("--w-vel", type=float, default=0.5)
    p.add_argument("--lambda-r1", type=float, default=10.0)
    p.add_argument("--r1-every", type=int, default=16)
    p.add_argument("--chan-base-g", type=int, default=256)
    p.add_argument("--chan-base-d", type=int, default=64)
    p.add_argument("--num-blocks", type=int, default=4)
    p.add_argument("--save-every", type=int, default=5)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--max-train-sets", type=int, default=0)
    p.add_argument("--val-max-sims", type=int, default=0, help="0 = all val sims")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--resume", default="")
    p.add_argument("--lbox", type=float, default=1000.0, help="full-box size, Mpc/h")
    p.add_argument("--n-pk-bins", type=int, default=32, help="bins for the 64^3 val Pk")
    p.add_argument("--n-pk-bins-patch", type=int, default=16, help="bins for the 32^3 loss Pk")
    return p.parse_args()


def softplus_loss_real(logits): return F.softplus(-logits).mean()
def softplus_loss_fake(logits): return F.softplus(logits).mean()


def r1_penalty(D, x_real, theta):
    x_real = x_real.detach().requires_grad_(True)
    logits = D(x_real, theta)
    grad = torch.autograd.grad(
        outputs=logits.sum(), inputs=x_real, create_graph=True, retain_graph=True,
    )[0]
    return grad.pow(2).sum(dim=(1, 2, 3, 4)).mean()


def weighted_l1(x_fake, x_hr, w_disp, w_vel):
    err = (x_fake - x_hr).abs()
    w = torch.tensor([w_disp, w_disp, w_disp, w_vel, w_vel, w_vel],
                     device=err.device, dtype=err.dtype).view(1, 6, 1, 1, 1)
    return (err * w).mean()


def hann3d(n, device):
    h = torch.hann_window(n, periodic=True, device=device)
    return (h[:, None, None] * h[None, :, None] * h[None, None, :]).view(1, 1, n, n, n)


def patch_pk_mse(pk_engine, window, fake_disp, hr_disp):
    """log-Pk MSE on Hann-windowed disp patches (both sides windowed identically)."""
    lpf = pk_engine(fake_disp * window)
    lph = pk_engine(hr_disp * window).detach()
    m = (lph > -10).float()
    return ((lpf - lph) ** 2 * m).sum() / m.sum().clamp_min(1.0)


def build_noise_list(grid, num_blocks, seed, device):
    rng = np.random.default_rng(seed)
    return [torch.from_numpy(rng.standard_normal((1, grid, grid, grid)).astype(np.float32)).to(device)
            for _ in range(2 * num_blocks)]


@torch.no_grad()
def stitched_validation(G, val_ds, pad, pk64, device, num_blocks, val_max_sims=0):
    """Patch-infer + stitch every val sim in the arm's deployment mode; return
    (stitched L1/voxel, stitched pkRMS_log10) vs the HR cube. Deterministic noise."""
    G.eval()
    grid = PATCH + 2 * pad
    noise_list = build_noise_list(grid, num_blocks, seed=0, device=device)
    ids = val_ds.ids if val_max_sims <= 0 else val_ds.ids[:val_max_sims]
    l1_sum, nvox = 0.0, 0
    pk_sum, pk_n = 0.0, 0
    for sid in ids:
        x_lr, x_hr, theta = val_ds.load_cubes(sid)
        patches = np.stack([extract_patch(x_lr, p, pad) for p in range(N_PATCHES)])
        xb = torch.from_numpy(patches).to(device)
        tb = torch.from_numpy(theta).unsqueeze(0).expand(N_PATCHES, -1).to(device)
        fake = G(xb, tb, noise_list)
        fake = crop_interior(fake, pad)
        sr = stitch_patches(fake.cpu().numpy())
        sr_t = torch.from_numpy(sr).unsqueeze(0).to(device)
        hr_t = torch.from_numpy(x_hr).unsqueeze(0).to(device)
        l1_sum += (sr_t - hr_t).abs().sum().item()
        nvox += hr_t.numel()
        lpf = pk64(sr_t[:, :3])
        lph = pk64(hr_t[:, :3])
        m = (lph > -10).float()
        pk_sum += ((lpf - lph) ** 2 * m).sum().item()
        pk_n += m.sum().item()
    return l1_sum / nvox, (pk_sum / max(pk_n, 1)) ** 0.5


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pad = args.pad
    arm = "A/baseline(full-patch loss, naive stitch)" if pad == 0 else \
          f"B/boundary-masked(pad={pad}, overlap-tile stitch)"

    train_ds = PatchPairDataset(args.data_root, split="train", pad=pad, seed=args.seed)
    val_ds = PatchPairDataset(args.data_root, split="val", pad=pad, seed=args.seed)
    if args.max_train_sets > 0:
        train_ds.ids = train_ds.ids[:args.max_train_sets]
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)

    G = G_correct(in_chan=6, out_chan=6, style_size=5,
                  chan_base=args.chan_base_g, num_blocks=args.num_blocks).to(device)
    D = D_const(in_chan=6, style_size=5,
                chan_base=args.chan_base_d, num_blocks=args.num_blocks).to(device)

    opt_g = torch.optim.Adam(G.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))
    opt_d = torch.optim.Adam(D.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))

    pk32 = TorchPk(N=PATCH, lbox=args.lbox * PATCH / 64.0,
                   n_bins=args.n_pk_bins_patch, device=device)
    pk64 = TorchPk(N=64, lbox=args.lbox, n_bins=args.n_pk_bins, device=device)
    win32 = hann3d(PATCH, device)

    start_epoch = 0
    best_pk = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        G.load_state_dict(ckpt["model"]); D.load_state_dict(ckpt["D"])
        opt_g.load_state_dict(ckpt["opt_g"]); opt_d.load_state_dict(ckpt["opt_d"])
        start_epoch = ckpt["epoch"] + 1
        best_pk = ckpt.get("best_pk", float("inf"))
        print(f"resumed from {args.resume} at epoch {start_epoch}")

    print(f"ARM {arm}")
    print(f"G params: {sum(p.numel() for p in G.parameters())/1e6:.2f}M  "
          f"D params: {sum(p.numel() for p in D.parameters())/1e6:.2f}M")
    print(f"train sims: {len(train_ds.ids)} ({len(train_ds)} patches)  "
          f"val sims: {len(val_ds.ids)}  input grid: {PATCH + 2*pad}^3  device: {device}")
    print(f"losses: w_disp={args.w_disp} w_vel={args.w_vel} lambda_pk={args.lambda_pk} "
          f"(Hann-windowed 32^3 Pk, lbox_patch={args.lbox * PATCH / 64.0:.0f})")

    d_step = 0
    for epoch in range(start_epoch, args.epochs):
        G.train(); D.train()
        t0 = time.time()
        for it, (x_lr, x_hr, theta, _, _) in enumerate(train_loader):
            x_lr = x_lr.to(device, non_blocking=True)
            x_hr = x_hr.to(device, non_blocking=True)
            theta = theta.to(device, non_blocking=True)
            # the loss region: interior 32^3 (== full patch when pad=0)
            hr_int = crop_interior(x_hr, pad)

            # --- D step (sees only the surviving interior region) ---
            opt_d.zero_grad(set_to_none=True)
            with torch.no_grad():
                fake_int = crop_interior(G(x_lr, theta), pad)
            logit_real = D(hr_int, theta); logit_fake = D(fake_int, theta)
            loss_d = softplus_loss_real(logit_real) + softplus_loss_fake(logit_fake)
            do_r1 = args.lambda_r1 > 0 and (d_step % args.r1_every == 0)
            if do_r1:
                r1 = r1_penalty(D, hr_int, theta)
                loss_d = loss_d + 0.5 * args.lambda_r1 * args.r1_every * r1
            loss_d.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(D.parameters(), args.grad_clip)
            opt_d.step()
            d_step += 1

            # --- G step ---
            opt_g.zero_grad(set_to_none=True)
            fake_int = crop_interior(G(x_lr, theta), pad)
            logit_fake = D(fake_int, theta)
            loss_g_adv = softplus_loss_real(logit_fake)
            loss_g_rec = weighted_l1(fake_int, hr_int, args.w_disp, args.w_vel)
            if args.lambda_pk > 0:
                loss_g_pk = patch_pk_mse(pk32, win32, fake_int[:, :3], hr_int[:, :3])
            else:
                loss_g_pk = torch.tensor(0.0, device=device)

            loss_g = loss_g_adv + args.lambda_rec * loss_g_rec + args.lambda_pk * loss_g_pk
            loss_g.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(G.parameters(), args.grad_clip)
            opt_g.step()

            if (it + 1) % args.log_every == 0:
                print(f"e{epoch} it{it+1}/{len(train_loader)}  "
                      f"loss_d={loss_d.item():.3f} adv={loss_g_adv.item():.3f} "
                      f"rec={loss_g_rec.item():.4f} pk={loss_g_pk.item():.4f}  "
                      f"D(real)={logit_real.mean().item():+.2f} D(fake)={logit_fake.mean().item():+.2f}",
                      flush=True)

        # --- validation on STITCHED cubes (the arm's own deployment mode) ---
        val_l1, val_pk_rms = stitched_validation(
            G, val_ds, pad, pk64, device, args.num_blocks, args.val_max_sims)
        print(f"epoch {epoch} done in {time.time()-t0:.1f}s  "
              f"STITCHED val_L1/voxel={val_l1:.4f}  val_pkRMS_log10={val_pk_rms:.4f}",
              flush=True)

        state = {
            "model": G.state_dict(), "D": D.state_dict(),
            "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(),
            "epoch": epoch, "args": vars(args), "pad": pad,
            "val_l1": val_l1, "val_pk_rms": val_pk_rms, "best_pk": best_pk,
        }
        if (epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs:
            path = os.path.join(args.ckpt_dir, f"epoch_{epoch+1}.pt")
            torch.save(state, path)
            print(f"saved {path}")

        if val_pk_rms < best_pk:
            best_pk = val_pk_rms
            state["best_pk"] = best_pk
            torch.save(state, os.path.join(args.ckpt_dir, "best.pt"))
            print(f"saved best.pt (stitched val_pk_rms={val_pk_rms:.4f})")


if __name__ == "__main__":
    main()
