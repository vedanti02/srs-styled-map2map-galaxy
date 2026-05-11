"""GAN training: cheap-sim → expensive-sim correction at fixed 64³ resolution.

Generator: G_correct (residual prediction, style-conditioned, custom-noise machinery).
Discriminator: D_const (style-conditioned, periodic-padded).
Loss: non-saturating GAN + L1 reconstruction (+ optional R1 regularization on D).
"""
import argparse
import os
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.pair_dataset import PairDataset
from map2map.models.styled_srsgan import G_correct, D_const


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="/data/group_data/universedata/lagrangian_output_64/stitched/")
    p.add_argument("--ckpt-dir", default="checkpoints/")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--lr-g", type=float, default=1e-4)
    p.add_argument("--lr-d", type=float, default=1e-4)
    p.add_argument("--beta1", type=float, default=0.0)
    p.add_argument("--beta2", type=float, default=0.99)
    p.add_argument("--lambda-rec", type=float, default=10.0)
    p.add_argument("--lambda-r1", type=float, default=10.0,
                   help="R1 grad-penalty coefficient on D (StyleGAN2). 0 disables.")
    p.add_argument("--r1-every", type=int, default=16,
                   help="Apply R1 penalty every k D updates (lazy regularization).")
    p.add_argument("--chan-base-g", type=int, default=256)
    p.add_argument("--chan-base-d", type=int, default=64)
    p.add_argument("--num-blocks", type=int, default=4)
    p.add_argument("--save-every", type=int, default=5)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--grad-clip", type=float, default=1.0,
                   help="Max grad norm for G and D (defends against GAN collapse). 0 disables.")
    p.add_argument("--max-train-sets", type=int, default=0,
                   help="If >0, only use this many sets (for smoke runs).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", default="")
    return p.parse_args()


def softplus_loss_real(logits):
    return F.softplus(-logits).mean()


def softplus_loss_fake(logits):
    return F.softplus(logits).mean()


def r1_penalty(D, x_real, theta):
    x_real = x_real.detach().requires_grad_(True)
    logits = D(x_real, theta)
    grad = torch.autograd.grad(
        outputs=logits.sum(), inputs=x_real, create_graph=True, retain_graph=True,
    )[0]
    return grad.pow(2).sum(dim=(1, 2, 3, 4)).mean()


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

    opt_g = torch.optim.Adam(G.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))
    opt_d = torch.optim.Adam(D.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        G.load_state_dict(ckpt["model"])
        D.load_state_dict(ckpt["D"])
        opt_g.load_state_dict(ckpt["opt_g"])
        opt_d.load_state_dict(ckpt["opt_d"])
        start_epoch = ckpt["epoch"] + 1
        print(f"resumed from {args.resume} at epoch {start_epoch}")

    print(f"G params: {sum(p.numel() for p in G.parameters())/1e6:.2f}M  "
          f"D params: {sum(p.numel() for p in D.parameters())/1e6:.2f}M")
    print(f"train sims: {len(train_ds)}  val sims: {len(val_ds)}  device: {device}")

    d_step = 0
    for epoch in range(start_epoch, args.epochs):
        G.train(); D.train()
        t0 = time.time()
        for it, (x_lr, x_hr, theta, _) in enumerate(train_loader):
            x_lr = x_lr.to(device, non_blocking=True)
            x_hr = x_hr.to(device, non_blocking=True)
            theta = theta.to(device, non_blocking=True)

            # --- D step
            opt_d.zero_grad(set_to_none=True)
            with torch.no_grad():
                x_fake = G(x_lr, theta)
            logit_real = D(x_hr, theta)
            logit_fake = D(x_fake, theta)
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

            # --- G step
            opt_g.zero_grad(set_to_none=True)
            x_fake = G(x_lr, theta)
            logit_fake = D(x_fake, theta)
            loss_g_adv = softplus_loss_real(logit_fake)
            loss_g_rec = F.l1_loss(x_fake, x_hr)
            loss_g = loss_g_adv + args.lambda_rec * loss_g_rec
            loss_g.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(G.parameters(), args.grad_clip)
            opt_g.step()

            if (it + 1) % args.log_every == 0:
                print(f"e{epoch} it{it+1}/{len(train_loader)}  "
                      f"loss_d={loss_d.item():.3f} loss_g_adv={loss_g_adv.item():.3f} "
                      f"loss_g_rec={loss_g_rec.item():.4f}  "
                      f"D(real)={logit_real.mean().item():+.2f} D(fake)={logit_fake.mean().item():+.2f}",
                      flush=True)

        # validation
        G.eval()
        with torch.no_grad():
            val_l1 = 0.0
            val_n = 0
            for x_lr, x_hr, theta, _ in val_loader:
                x_lr = x_lr.to(device); x_hr = x_hr.to(device); theta = theta.to(device)
                x_fake = G(x_lr, theta)
                val_l1 += F.l1_loss(x_fake, x_hr, reduction="sum").item()
                val_n += x_hr.numel()
        print(f"epoch {epoch} done in {time.time()-t0:.1f}s  val_L1/voxel={val_l1/val_n:.4f}", flush=True)

        if (epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs:
            path = os.path.join(args.ckpt_dir, f"epoch_{epoch+1}.pt")
            torch.save({
                "model": G.state_dict(),
                "D": D.state_dict(),
                "opt_g": opt_g.state_dict(),
                "opt_d": opt_d.state_dict(),
                "epoch": epoch,
                "args": vars(args),
            }, path)
            print(f"saved {path}")


if __name__ == "__main__":
    main()
