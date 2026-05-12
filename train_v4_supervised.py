"""Pure supervised training: L1 + Pk losses only, NO GAN.

Tests the hypothesis that the GAN discriminator is hurting Pk match. Same generator
architecture (G_correct) as v1/v2/v3 but trained without adversarial loss.

If v4 beats v1/v2 on posterior metrics, the discriminator is net-negative and we
should remove it. If v4 underperforms, the GAN is providing useful structure.
"""
import argparse
import os
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.pair_dataset import PairDataset
from map2map.models.styled_srsgan import G_correct
from analysis.pk_torch import TorchPk


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="/data/group_data/universedata/lagrangian_output_64/stitched/")
    p.add_argument("--ckpt-dir", default="checkpoints/v4/")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--lambda-rec", type=float, default=10.0)
    p.add_argument("--lambda-pk", type=float, default=10.0)
    p.add_argument("--w-disp", type=float, default=2.0)
    p.add_argument("--w-vel", type=float, default=0.5)
    p.add_argument("--chan-base-g", type=int, default=256)
    p.add_argument("--num-blocks", type=int, default=4)
    p.add_argument("--save-every", type=int, default=5)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--lbox", type=float, default=1000.0)
    p.add_argument("--n-pk-bins", type=int, default=32)
    p.add_argument("--init-from", default="",
                   help="Optional warm-start: load only G state_dict from this ckpt.")
    return p.parse_args()


def weighted_l1(x_fake, x_hr, w_disp, w_vel):
    err = (x_fake - x_hr).abs()
    w = torch.tensor([w_disp, w_disp, w_disp, w_vel, w_vel, w_vel],
                     device=err.device, dtype=err.dtype).view(1, 6, 1, 1, 1)
    return (err * w).mean()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = PairDataset(args.data_root, split="train", seed=args.seed)
    val_ds = PairDataset(args.data_root, split="val", seed=args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    G = G_correct(in_chan=6, out_chan=6, style_size=5,
                  chan_base=args.chan_base_g, num_blocks=args.num_blocks).to(device)

    if args.init_from:
        ck = torch.load(args.init_from, map_location=device, weights_only=False)
        sd = ck["model"] if "model" in ck else ck
        G.load_state_dict(sd)
        print(f"warm-started G from {args.init_from}", flush=True)

    opt = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    pk_engine = TorchPk(N=64, lbox=args.lbox, n_bins=args.n_pk_bins, device=device)

    print(f"G params: {sum(p.numel() for p in G.parameters())/1e6:.2f}M")
    print(f"v4 (no-GAN supervised) lambda_rec={args.lambda_rec} lambda_pk={args.lambda_pk}")
    print(f"train sims: {len(train_ds)}  val sims: {len(val_ds)}  device: {device}")

    main._best_pk = float("inf")
    for epoch in range(args.epochs):
        G.train()
        t0 = time.time()
        for it, (x_lr, x_hr, theta, _) in enumerate(train_loader):
            x_lr = x_lr.to(device, non_blocking=True)
            x_hr = x_hr.to(device, non_blocking=True)
            theta = theta.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            x_fake = G(x_lr, theta)
            loss_rec = weighted_l1(x_fake, x_hr, args.w_disp, args.w_vel)

            if args.lambda_pk > 0:
                lp_fake = pk_engine(x_fake[:, :3])
                lp_hr = pk_engine(x_hr[:, :3]).detach()
                m = (lp_hr > -10).float()
                loss_pk = ((lp_fake - lp_hr) ** 2 * m).sum() / m.sum().clamp_min(1.0)
            else:
                loss_pk = torch.tensor(0.0, device=device)

            loss = args.lambda_rec * loss_rec + args.lambda_pk * loss_pk
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(G.parameters(), args.grad_clip)
            opt.step()

            if (it + 1) % args.log_every == 0:
                print(f"e{epoch} it{it+1}/{len(train_loader)}  "
                      f"rec={loss_rec.item():.4f} pk={loss_pk.item():.4f}", flush=True)

        # validation
        G.eval()
        with torch.no_grad():
            val_l1 = 0.0; val_pk_mse = 0.0; val_n = 0; val_pk_n = 0
            for x_lr, x_hr, theta, _ in val_loader:
                x_lr = x_lr.to(device); x_hr = x_hr.to(device); theta = theta.to(device)
                x_fake = G(x_lr, theta)
                val_l1 += F.l1_loss(x_fake, x_hr, reduction="sum").item()
                val_n += x_hr.numel()
                lpf = pk_engine(x_fake[:, :3]); lph = pk_engine(x_hr[:, :3])
                m = (lph > -10).float()
                val_pk_mse += ((lpf - lph) ** 2 * m).sum().item()
                val_pk_n += m.sum().item()
        val_l1_voxel = val_l1 / val_n
        val_pk_rms = (val_pk_mse / max(val_pk_n, 1)) ** 0.5
        print(f"epoch {epoch} done in {time.time()-t0:.1f}s  "
              f"val_L1/voxel={val_l1_voxel:.4f}  val_pkRMS_log10={val_pk_rms:.4f}", flush=True)

        if (epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs:
            path = os.path.join(args.ckpt_dir, f"epoch_{epoch+1}.pt")
            torch.save({"model": G.state_dict(), "epoch": epoch, "args": vars(args)}, path)
            print(f"saved {path}")

        best_path = os.path.join(args.ckpt_dir, "best.pt")
        if val_pk_rms < main._best_pk:
            main._best_pk = val_pk_rms
            torch.save({"model": G.state_dict(), "epoch": epoch, "args": vars(args),
                        "val_l1": val_l1_voxel, "val_pk_rms": val_pk_rms}, best_path)
            print(f"saved {best_path} (best val_pk_rms={val_pk_rms:.4f})")


if __name__ == "__main__":
    main()
