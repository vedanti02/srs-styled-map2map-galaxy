"""Real-scale patch GAN: quijote-64 64³ patches → HR, stitch to 128³ (interpretation (b)).

Same v2 GAN recipe and two-arm design as train_patch.py, but on the REAL patches via
PatchPairDatasetReal (indexed by sim; the 8 patches per sim are flattened into the batch).

  --pad 0   Arm A baseline (2.1): full 64³-patch loss, naive stitch.
  --pad P>0 Arm B (2.2): halo'd (64+2P)³ input, loss only on the kept central 64³.

Validation = stitch all val sims to 128³ in the arm's deployment mode, score vs assembled HR.
"""
import argparse, os, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.patch_dataset_real import (
    PatchPairDatasetReal, assemble_box, extract_patch, stitch_patches, crop_interior,
    normalize, PATCH, N_PATCHES, N_FULL,
)
from map2map.models.styled_srsgan import G_correct, D_const
from analysis.pk_torch import TorchPk


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", default="checkpoints/patch_real_v1/")
    p.add_argument("--pad", type=int, default=0)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--sims-per-batch", type=int, default=1, help="sims/step; patch-batch = 8×this")
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


def weighted_l1(f, h, wd, wv):
    w = torch.tensor([wd, wd, wd, wv, wv, wv], device=f.device, dtype=f.dtype).view(1, 6, 1, 1, 1)
    return ((f - h).abs() * w).mean()


def hann3d(n, device):
    h = torch.hann_window(n, periodic=True, device=device)
    return (h[:, None, None] * h[None, :, None] * h[None, None, :]).view(1, 1, n, n, n)


def build_noise_list(grid, num_blocks, seed, device):
    rng = np.random.default_rng(seed)
    return [torch.from_numpy(rng.standard_normal((1, grid, grid, grid)).astype(np.float32)).to(device)
            for _ in range(2 * num_blocks)]


@torch.no_grad()
def stitched_validation(G, val_ds, pad, pk128, device, num_blocks, root, snap, max_sims):
    G.eval()
    grid = PATCH + 2 * pad
    nl = build_noise_list(grid, num_blocks, 0, device)
    ids = val_ds.ids if max_sims <= 0 else val_ds.ids[:max_sims]
    l1, nvox, pks, pkn = 0.0, 0, 0.0, 0
    for sid in ids:
        lr = normalize(assemble_box(root, "quijotelike-64", sid, snap))
        hr = normalize(assemble_box(root, "quijote-64", sid, snap))
        patches = np.stack([extract_patch(lr, p, pad) for p in range(N_PATCHES)])
        xb = torch.from_numpy(patches).to(device)
        tb = torch.from_numpy(np.load(os.path.join(root, "quijote-64", f"set{sid}_pos_0_0_0", snap,
             "style.npy")).astype(np.float32)).unsqueeze(0).expand(N_PATCHES, -1).to(device)
        fake = crop_interior(G(xb, tb, nl), pad)
        sr = stitch_patches(fake.cpu().numpy())
        sr_t = torch.from_numpy(sr).unsqueeze(0).to(device)
        hr_t = torch.from_numpy(hr).unsqueeze(0).to(device)
        l1 += (sr_t - hr_t).abs().sum().item(); nvox += hr_t.numel()
        lpf, lph = pk128(sr_t[:, :3]), pk128(hr_t[:, :3])
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

    train_ds = PatchPairDatasetReal(split="train", pad=pad, seed=args.seed)
    val_ds = PatchPairDatasetReal(split="val", pad=pad, seed=args.seed)
    if args.max_train_sets > 0:
        train_ds.ids = train_ds.ids[:args.max_train_sets]
    loader = DataLoader(train_ds, batch_size=args.sims_per_batch, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True, drop_last=True)

    G = G_correct(6, 6, 5, chan_base=args.chan_base_g, num_blocks=args.num_blocks).to(dev)
    D = D_const(6, 5, chan_base=args.chan_base_d, num_blocks=args.num_blocks).to(dev)
    opt_g = torch.optim.Adam(G.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))
    opt_d = torch.optim.Adam(D.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))

    pk64 = TorchPk(N=PATCH, lbox=args.lbox * PATCH / N_FULL, n_bins=args.n_pk_bins_patch, device=dev)
    pk128 = TorchPk(N=N_FULL, lbox=args.lbox, n_bins=args.n_pk_bins, device=dev)
    win = hann3d(PATCH, dev)
    root, snap = train_ds.root, train_ds.snap

    start, best = 0, float("inf")
    if args.resume:
        c = torch.load(args.resume, map_location=dev, weights_only=False)
        G.load_state_dict(c["model"]); D.load_state_dict(c["D"])
        opt_g.load_state_dict(c["opt_g"]); opt_d.load_state_dict(c["opt_d"])
        start = c["epoch"] + 1; best = c.get("best_pk", float("inf"))
        print(f"resumed at epoch {start}")

    print(f"ARM {arm}")
    print(f"G {sum(p.numel() for p in G.parameters())/1e6:.2f}M  D {sum(p.numel() for p in D.parameters())/1e6:.2f}M")
    print(f"train sims {len(train_ds.ids)}  val sims {len(val_ds.ids)}  input {PATCH+2*pad}³  "
          f"patch-batch {args.sims_per_batch*N_PATCHES}  dev {dev}")

    dstep = 0
    for epoch in range(start, args.epochs):
        G.train(); D.train(); t0 = time.time()
        for it, (lr_in, hr_tgt, theta, _) in enumerate(loader):
            # flatten (B, 8, ...) → (B*8, ...)
            B = lr_in.shape[0]
            x_lr = lr_in.reshape(B * N_PATCHES, 6, *lr_in.shape[3:]).to(dev, non_blocking=True)
            x_hr = hr_tgt.reshape(B * N_PATCHES, 6, PATCH, PATCH, PATCH).to(dev, non_blocking=True)
            th = theta.repeat_interleave(N_PATCHES, 0).to(dev, non_blocking=True)

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

            opt_g.zero_grad(set_to_none=True)
            fake = crop_interior(G(x_lr, th), pad)
            adv = softplus_loss_real(D(fake, th))
            rec = weighted_l1(fake, x_hr, args.w_disp, args.w_vel)
            if args.lambda_pk > 0:
                lpf = pk64(fake[:, :3] * win); lph = pk64(x_hr[:, :3] * win).detach()
                m = (lph > -10).float(); pk = ((lpf - lph) ** 2 * m).sum() / m.sum().clamp_min(1.0)
            else:
                pk = torch.tensor(0.0, device=dev)
            loss_g = adv + args.lambda_rec * rec + args.lambda_pk * pk
            loss_g.backward()
            if args.grad_clip > 0: torch.nn.utils.clip_grad_norm_(G.parameters(), args.grad_clip)
            opt_g.step()

            if (it + 1) % args.log_every == 0:
                print(f"e{epoch} it{it+1}/{len(loader)}  d={loss_d.item():.3f} adv={adv.item():.3f} "
                      f"rec={rec.item():.4f} pk={pk.item():.4f}  D(r)={lr_real.mean():+.2f} D(f)={lr_fake.mean():+.2f}",
                      flush=True)

        vl1, vpk = stitched_validation(G, val_ds, pad, pk128, dev, args.num_blocks, root, snap, args.val_max_sims)
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
