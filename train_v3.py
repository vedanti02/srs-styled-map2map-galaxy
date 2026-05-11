"""GAN training v3: extends v2 with multi-scale Pk loss and stronger weight.

Builds on v2 but adds:
  * ``--lambda-pk`` defaults to 10 (vs 2 in v2) — stronger emphasis on Pk match
  * Multi-scale Pk: compute Pk on both the original 64³ field AND a smoothed
    field, summing the two log-Pk MSE terms. The smoothed field stresses the
    largest scales the model tends to mis-predict.
  * Per-bin weighted Pk loss: weight low-k bins higher because cosmology is
    most sensitive to large-scale power.

Same architecture (G_correct + D_const) as v1/v2.
"""
import argparse
import os
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.pair_dataset import PairDataset
from map2map.models.styled_srsgan import G_correct, D_const
from analysis.pk_torch import TorchPk


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="/data/group_data/universedata/lagrangian_output_64/stitched/")
    p.add_argument("--ckpt-dir", default="checkpoints/v3/")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr-g", type=float, default=1e-4)
    p.add_argument("--lr-d", type=float, default=1e-4)
    p.add_argument("--beta1", type=float, default=0.0)
    p.add_argument("--beta2", type=float, default=0.99)
    p.add_argument("--lambda-rec", type=float, default=10.0)
    p.add_argument("--lambda-pk", type=float, default=10.0,
                   help="Total weight on Pk loss (split equally over scales).")
    p.add_argument("--w-disp", type=float, default=2.0)
    p.add_argument("--w-vel", type=float, default=0.5)
    p.add_argument("--lambda-r1", type=float, default=10.0)
    p.add_argument("--r1-every", type=int, default=16)
    p.add_argument("--chan-base-g", type=int, default=256)
    p.add_argument("--chan-base-d", type=int, default=64)
    p.add_argument("--num-blocks", type=int, default=4)
    p.add_argument("--save-every", type=int, default=5)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--max-train-sets", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--resume", default="")
    p.add_argument("--lbox", type=float, default=1000.0)
    p.add_argument("--n-pk-bins", type=int, default=32)
    p.add_argument("--init-from", default="",
                   help="Optional warm-start ckpt path (loads G+D state_dict).")
    p.add_argument("--low-k-weight", type=float, default=2.0,
                   help="Linear weight on lowest k-bins (gradient toward 1.0 at high k).")
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


def smooth_field(x, sigma_voxels=2.0):
    """Light Gaussian-like smoothing via average pooling. Returns same shape."""
    k = max(int(2 * sigma_voxels) | 1, 3)  # odd kernel
    pad = k // 2
    # 3D box-like blur (cheap; doesn't have to be a true Gaussian)
    return F.avg_pool3d(F.pad(x, (pad, pad, pad, pad, pad, pad), mode="circular"),
                         kernel_size=k, stride=1)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = PairDataset(args.data_root, split="train", seed=args.seed)
    val_ds = PairDataset(args.data_root, split="val", seed=args.seed)
    if args.max_train_sets > 0:
        train_ds.ids = train_ds.ids[:args.max_train_sets]
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    G = G_correct(in_chan=6, out_chan=6, style_size=5,
                  chan_base=args.chan_base_g, num_blocks=args.num_blocks).to(device)
    D = D_const(in_chan=6, style_size=5,
                chan_base=args.chan_base_d, num_blocks=args.num_blocks).to(device)

    if args.init_from:
        ck = torch.load(args.init_from, map_location=device, weights_only=False)
        if "model" in ck: G.load_state_dict(ck["model"])
        if "D" in ck: D.load_state_dict(ck["D"])
        print(f"warm-started G/D from {args.init_from}")

    opt_g = torch.optim.Adam(G.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))
    opt_d = torch.optim.Adam(D.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))

    pk_engine = TorchPk(N=64, lbox=args.lbox, n_bins=args.n_pk_bins, device=device)
    # Per-bin weight: linearly interpolate from low_k_weight at smallest k to 1.0 at largest k
    bin_weights = torch.linspace(args.low_k_weight, 1.0, args.n_pk_bins, device=device)

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        G.load_state_dict(ckpt["model"]); D.load_state_dict(ckpt["D"])
        opt_g.load_state_dict(ckpt["opt_g"]); opt_d.load_state_dict(ckpt["opt_d"])
        start_epoch = ckpt["epoch"] + 1
        print(f"resumed from {args.resume} at epoch {start_epoch}")

    print(f"G params: {sum(p.numel() for p in G.parameters())/1e6:.2f}M  "
          f"D params: {sum(p.numel() for p in D.parameters())/1e6:.2f}M")
    print(f"v3 losses: w_disp={args.w_disp} w_vel={args.w_vel} "
          f"lambda_pk={args.lambda_pk} low_k_weight={args.low_k_weight}")
    print(f"train sims: {len(train_ds)}  val sims: {len(val_ds)}  device: {device}")

    d_step = 0
    main._best_pk = float("inf")
    for epoch in range(start_epoch, args.epochs):
        G.train(); D.train()
        t0 = time.time()
        for it, (x_lr, x_hr, theta, _) in enumerate(train_loader):
            x_lr = x_lr.to(device, non_blocking=True)
            x_hr = x_hr.to(device, non_blocking=True)
            theta = theta.to(device, non_blocking=True)

            # --- D step ---
            opt_d.zero_grad(set_to_none=True)
            with torch.no_grad():
                x_fake = G(x_lr, theta)
            logit_real = D(x_hr, theta); logit_fake = D(x_fake, theta)
            loss_d = softplus_loss_real(logit_real) + softplus_loss_fake(logit_fake)
            do_r1 = args.lambda_r1 > 0 and (d_step % args.r1_every == 0)
            if do_r1:
                r1 = r1_penalty(D, x_hr, theta)
                loss_d = loss_d + 0.5 * args.lambda_r1 * args.r1_every * r1
            loss_d.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(D.parameters(), args.grad_clip)
            opt_d.step()
            d_step += 1

            # --- G step ---
            opt_g.zero_grad(set_to_none=True)
            x_fake = G(x_lr, theta)
            logit_fake = D(x_fake, theta)
            loss_g_adv = softplus_loss_real(logit_fake)
            loss_g_rec = weighted_l1(x_fake, x_hr, args.w_disp, args.w_vel)

            if args.lambda_pk > 0:
                # Scale 1: original 64³ disp
                lp_fake_full = pk_engine(x_fake[:, :3])
                lp_hr_full = pk_engine(x_hr[:, :3]).detach()
                # Scale 2: smoothed disp (suppresses small-scale, emphasizes large-scale)
                xs_fake = smooth_field(x_fake[:, :3], sigma_voxels=2.0)
                xs_hr = smooth_field(x_hr[:, :3], sigma_voxels=2.0)
                lp_fake_s = pk_engine(xs_fake)
                lp_hr_s = pk_engine(xs_hr).detach()

                m_full = (lp_hr_full > -10).float() * bin_weights.unsqueeze(0)
                m_s = (lp_hr_s > -10).float() * bin_weights.unsqueeze(0)

                loss_pk_full = ((lp_fake_full - lp_hr_full) ** 2 * m_full).sum() / m_full.sum().clamp_min(1.0)
                loss_pk_s = ((lp_fake_s - lp_hr_s) ** 2 * m_s).sum() / m_s.sum().clamp_min(1.0)
                loss_g_pk = 0.5 * (loss_pk_full + loss_pk_s)
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
                      f"D(r)={logit_real.mean().item():+.2f} D(f)={logit_fake.mean().item():+.2f}",
                      flush=True)

        # validation
        G.eval()
        with torch.no_grad():
            val_l1 = 0.0
            val_pk_mse = 0.0
            val_n = 0
            val_pk_n = 0
            for x_lr, x_hr, theta, _ in val_loader:
                x_lr = x_lr.to(device); x_hr = x_hr.to(device); theta = theta.to(device)
                x_fake = G(x_lr, theta)
                val_l1 += F.l1_loss(x_fake, x_hr, reduction="sum").item()
                val_n += x_hr.numel()
                lpf = pk_engine(x_fake[:, :3])
                lph = pk_engine(x_hr[:, :3])
                m = (lph > -10).float()
                val_pk_mse += ((lpf - lph) ** 2 * m).sum().item()
                val_pk_n += m.sum().item()
        val_l1_voxel = val_l1 / val_n
        val_pk_rms = (val_pk_mse / max(val_pk_n, 1)) ** 0.5
        print(f"epoch {epoch} done in {time.time()-t0:.1f}s  "
              f"val_L1/voxel={val_l1_voxel:.4f}  val_pkRMS_log10={val_pk_rms:.4f}", flush=True)

        if (epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs:
            path = os.path.join(args.ckpt_dir, f"epoch_{epoch+1}.pt")
            torch.save({
                "model": G.state_dict(), "D": D.state_dict(),
                "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(),
                "epoch": epoch, "args": vars(args),
            }, path)
            print(f"saved {path}")

        best_path = os.path.join(args.ckpt_dir, "best.pt")
        if val_pk_rms < main._best_pk:
            main._best_pk = val_pk_rms
            torch.save({
                "model": G.state_dict(),
                "epoch": epoch, "args": vars(args),
                "val_l1": val_l1_voxel, "val_pk_rms": val_pk_rms,
            }, best_path)
            print(f"saved {best_path} (best val_pk_rms={val_pk_rms:.4f})")


if __name__ == "__main__":
    main()
