"""v6: v5 architecture + redshift-augmented training data.

Same as train_v5_supervised.py (pure supervised, multi-scale Pk with low-k
bin weighting, no GAN) but:
  - dataset is PairDatasetZAug (linear-growth z augmentation)
  - style_size = 6 (Om, Ob, h, n_s, sigma_8, z)
  - validation runs at fixed_z = 0.0 so val metrics stay comparable to v5/v2

Goal: plumb a redshift dimension through the network. With single-snapshot
training data the z signal is only linear-theory exact, but if the model
learns to use the z slot consistently, it can be extended once real multi-z
data is available.
"""
import argparse
import os
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.pair_dataset_zaug import PairDatasetZAug
from map2map.models.styled_srsgan import G_correct
from analysis.pk_torch import TorchPk


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="/data/group_data/universedata/lagrangian_output_64/stitched/")
    p.add_argument("--ckpt-dir", default="checkpoints/v6/")
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
    p.add_argument("--low-k-weight", type=float, default=3.0)
    p.add_argument("--smooth-sigma", type=float, default=2.0)
    p.add_argument("--z-min", type=float, default=0.0)
    p.add_argument("--z-max", type=float, default=1.5)
    p.add_argument("--resume", default="")
    p.add_argument("--init-from", default="")
    return p.parse_args()


def weighted_l1(x_fake, x_hr, w_disp, w_vel):
    err = (x_fake - x_hr).abs()
    w = torch.tensor([w_disp, w_disp, w_disp, w_vel, w_vel, w_vel],
                     device=err.device, dtype=err.dtype).view(1, 6, 1, 1, 1)
    return (err * w).mean()


def smooth_field(x, sigma_voxels=2.0):
    k = max(int(2 * sigma_voxels) | 1, 3)
    pad = k // 2
    return F.avg_pool3d(F.pad(x, (pad, pad, pad, pad, pad, pad), mode="circular"),
                        kernel_size=k, stride=1)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = PairDatasetZAug(args.data_root, split="train", seed=args.seed,
                               z_min=args.z_min, z_max=args.z_max)
    val_ds = PairDatasetZAug(args.data_root, split="val", seed=args.seed,
                             fixed_z=0.0)  # eval at z=0 for direct v5/v2 comparison
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    G = G_correct(in_chan=6, out_chan=6, style_size=6,
                  chan_base=args.chan_base_g, num_blocks=args.num_blocks).to(device)

    if args.init_from:
        ck = torch.load(args.init_from, map_location=device, weights_only=False)
        sd = ck["model"] if "model" in ck else ck
        # Allow partial load if old ckpt had style_size=5 — drop mismatched keys
        new_sd = G.state_dict()
        loaded, skipped = 0, 0
        for k, v in sd.items():
            if k in new_sd and new_sd[k].shape == v.shape:
                new_sd[k] = v; loaded += 1
            else:
                skipped += 1
        G.load_state_dict(new_sd)
        print(f"warm-started G from {args.init_from} ({loaded} matched, {skipped} skipped due to shape)")

    opt = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    pk_engine = TorchPk(N=64, lbox=args.lbox, n_bins=args.n_pk_bins, device=device)
    bin_weights = torch.linspace(args.low_k_weight, 1.0, args.n_pk_bins, device=device)

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        G.load_state_dict(ckpt["model"])
        if "opt" in ckpt:
            opt.load_state_dict(ckpt["opt"])
        start_epoch = ckpt["epoch"] + 1
        print(f"resumed from {args.resume} at epoch {start_epoch}")

    print(f"G params: {sum(p.numel() for p in G.parameters())/1e6:.2f}M (style_size=6)")
    print(f"v6 (no-GAN, multi-scale Pk, z-aug linear) lambda_rec={args.lambda_rec} "
          f"lambda_pk={args.lambda_pk} low_k_weight={args.low_k_weight} "
          f"smooth_sigma={args.smooth_sigma} z_range=[{args.z_min},{args.z_max}]")
    print(f"train sims: {len(train_ds)}  val sims: {len(val_ds)} (val_z=0)  device: {device}")

    main._best_pk = float("inf")
    for epoch in range(start_epoch, args.epochs):
        G.train()
        t0 = time.time()
        for it, (x_lr, x_hr, theta6, _) in enumerate(train_loader):
            x_lr = x_lr.to(device, non_blocking=True)
            x_hr = x_hr.to(device, non_blocking=True)
            theta6 = theta6.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            x_fake = G(x_lr, theta6)
            loss_rec = weighted_l1(x_fake, x_hr, args.w_disp, args.w_vel)

            if args.lambda_pk > 0:
                lp_fake_full = pk_engine(x_fake[:, :3])
                lp_hr_full = pk_engine(x_hr[:, :3]).detach()
                xs_fake = smooth_field(x_fake[:, :3], sigma_voxels=args.smooth_sigma)
                xs_hr = smooth_field(x_hr[:, :3], sigma_voxels=args.smooth_sigma)
                lp_fake_s = pk_engine(xs_fake)
                lp_hr_s = pk_engine(xs_hr).detach()

                m_full = (lp_hr_full > -10).float() * bin_weights.unsqueeze(0)
                m_s = (lp_hr_s > -10).float() * bin_weights.unsqueeze(0)
                loss_pk_full = ((lp_fake_full - lp_hr_full) ** 2 * m_full).sum() / m_full.sum().clamp_min(1.0)
                loss_pk_s = ((lp_fake_s - lp_hr_s) ** 2 * m_s).sum() / m_s.sum().clamp_min(1.0)
                loss_pk = 0.5 * (loss_pk_full + loss_pk_s)
            else:
                loss_pk = torch.tensor(0.0, device=device)

            loss = args.lambda_rec * loss_rec + args.lambda_pk * loss_pk
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(G.parameters(), args.grad_clip)
            opt.step()

            if (it + 1) % args.log_every == 0:
                z_batch = theta6[:, 5].mean().item()
                print(f"e{epoch} it{it+1}/{len(train_loader)}  "
                      f"rec={loss_rec.item():.4f} pk={loss_pk.item():.4f} "
                      f"z_mean={z_batch:.2f}", flush=True)

        G.eval()
        with torch.no_grad():
            val_l1 = 0.0; val_pk_mse = 0.0; val_n = 0; val_pk_n = 0
            for x_lr, x_hr, theta6, _ in val_loader:
                x_lr = x_lr.to(device); x_hr = x_hr.to(device); theta6 = theta6.to(device)
                x_fake = G(x_lr, theta6)
                val_l1 += F.l1_loss(x_fake, x_hr, reduction="sum").item()
                val_n += x_hr.numel()
                lpf = pk_engine(x_fake[:, :3]); lph = pk_engine(x_hr[:, :3])
                m = (lph > -10).float()
                val_pk_mse += ((lpf - lph) ** 2 * m).sum().item()
                val_pk_n += m.sum().item()
        val_l1_voxel = val_l1 / val_n
        val_pk_rms = (val_pk_mse / max(val_pk_n, 1)) ** 0.5
        print(f"epoch {epoch} done in {time.time()-t0:.1f}s  "
              f"val_L1/voxel={val_l1_voxel:.4f}  val_pkRMS_log10={val_pk_rms:.4f}  (val_z=0)",
              flush=True)

        if (epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs:
            path = os.path.join(args.ckpt_dir, f"epoch_{epoch+1}.pt")
            torch.save({"model": G.state_dict(), "opt": opt.state_dict(),
                        "epoch": epoch, "args": vars(args)}, path)
            print(f"saved {path}")

        best_path = os.path.join(args.ckpt_dir, "best.pt")
        if val_pk_rms < main._best_pk:
            main._best_pk = val_pk_rms
            torch.save({"model": G.state_dict(), "epoch": epoch, "args": vars(args),
                        "val_l1": val_l1_voxel, "val_pk_rms": val_pk_rms}, best_path)
            print(f"saved {best_path} (best val_pk_rms={val_pk_rms:.4f})")


if __name__ == "__main__":
    main()
