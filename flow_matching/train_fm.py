"""Train the standalone conditional flow-matching corrector (no GAN, no theta).

  python -m flow_matching.train_fm --ckpt-dir /data/user_data/vkshirsa/cmass-ili/models/fm_gauss \
      --source gaussian --cache --epochs 60

Protocol parity with train_patch_cmass.py: same dataset/splits, 64^3 cores (pad 0, naive
stitch), model space log1p (here standardised + dequantised), validation on stitched 128^3
boxes, checkpoint selection on val L1/voxel in log1p space (--select l1), P(k) only ever a
held-out check. Differences: no theta anywhere; loss = MSE on the flow velocity; weight EMA;
cube-symmetry augmentation; bf16 autocast.
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from flow_matching.data import CountPatchDataset, extract_patch, N_PATCHES, PATCH
from flow_matching.space import ModelSpace
from flow_matching.interpolant import Interpolant, SOURCES, sample_t, residual_spectrum
from flow_matching.unet3d import build_unet
from flow_matching.augment import random_op, apply_op
from flow_matching.common import EMA, generate_box, box_metrics
from analysis.pk_torch import TorchPk


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--source", default="gaussian", choices=SOURCES)
    p.add_argument("--sigma-scale", type=float, default=1.0, help="multiplier on sigma_z / spectral amp")
    p.add_argument("--cond", action=argparse.BooleanOptionalAction, default=True,
                   help="feed the LF field as a conditioning channel to the velocity net")
    p.add_argument("--dequant", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--aug", default="full48", choices=["none", "los16", "full48"])
    p.add_argument("--t-dist", default="uniform", choices=["uniform", "logitnormal"])
    p.add_argument("--t-shift", type=float, default=0.0, help="logitnormal only: >0 emphasises late t")
    p.add_argument("--loss-weight", default="none", choices=["none", "lf"],
                   help="lf: per-voxel weight w = 1 + alpha * blur3(n_LF), mean-normalised. Depends only on the "
                        "conditioning, so the minimiser is unchanged (still E[v|x_t,LF]); it re-weights dense "
                        "regions where the under-produced high-count HF voxels live.")
    p.add_argument("--lw-alpha", type=float, default=2.0)
    # network
    p.add_argument("--base-ch", type=int, default=32)
    p.add_argument("--ch-mult", default="1,2,4,4", help="channel multipliers per level; 1,2,4,4 = 17.4M params (1,2,4,8 = 41.7M)")
    p.add_argument("--num-res", type=int, default=2)
    p.add_argument("--attn-levels", default="3", help="comma list of levels (0=64^3 .. 3=8^3) with attention")
    p.add_argument("--dropout", type=float, default=0.0)
    # optimisation
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--sims-per-batch", type=int, default=1, help="patch batch = 8 x this")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--amp", default="bf16", choices=["none", "bf16"])
    p.add_argument("--seed", type=int, default=0)
    # data
    p.add_argument("--cache", action="store_true", help="preload all boxes as uint8 in RAM")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-train-sets", type=int, default=0)
    p.add_argument("--stats-boxes", type=int, default=50, help="train boxes used for mean/std/sigma_z/spectrum")
    # validation / selection
    p.add_argument("--val-max-sims", type=int, default=16)
    p.add_argument("--val-draws", type=int, default=2)
    p.add_argument("--val-steps", type=int, default=16)
    p.add_argument("--val-method", default="heun", choices=["euler", "heun"])
    p.add_argument("--select", default="l1", choices=["l1", "crps", "l1_mean"])
    p.add_argument("--save-every", type=int, default=1)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--resume", default="", help="path to ckpt, or 'auto' = <ckpt-dir>/last.pt if present")
    p.add_argument("--lbox", type=float, default=1000.0)
    return p.parse_args()


@torch.no_grad()
def estimate_stats(ds, n_boxes, dequant, device, seed=0):
    """mean/std of log1p(n_hr+u); sigma_z and residual spectrum of (y_hf - y_lf) on 64^3 patches."""
    g = torch.Generator(device=device).manual_seed(seed)
    ids = ds.ids[:n_boxes]
    # pass 1: mean/std of the (dequantised) HR log field
    s1 = s2 = cnt = 0.0
    boxes = []
    for idx in ids:
        lr, hr = ds.load_boxes(idx)
        boxes.append((lr, hr))
        h = torch.from_numpy(hr).to(device)
        if dequant:
            h = h + torch.rand(h.shape, device=device, generator=g)
        lh = torch.log1p(h)
        s1 += lh.sum().item(); s2 += (lh ** 2).sum().item(); cnt += lh.numel()
    mean = s1 / cnt
    std = max((s2 / cnt - mean ** 2), 1e-12) ** 0.5
    space = ModelSpace(mean, std, dequant)
    # pass 2: residual stats in model space on 64^3 patches. The residual is measured on the
    # EXACT HF transform (no dequantisation), so sigma_z and the spectrum describe the physical
    # LF-HF mismatch rather than the uniform smear u (which the flow generates regardless).
    tab = None; r2 = 0.0; r2_dq = 0.0; n = 0
    for lr, hr in boxes:
        lr_p = torch.from_numpy(np.stack([extract_patch(lr, p, 0) for p in range(N_PATCHES)])).to(device)
        hr_p = torch.from_numpy(np.stack([extract_patch(hr, p, 0) for p in range(N_PATCHES)])).to(device)
        y_lf = space.forward(lr_p, dequant=False)
        r = space.forward(hr_p, dequant=False) - y_lf
        r2 += (r ** 2).sum().item(); n += r.numel()
        r2_dq += ((space.forward(hr_p, generator=g) - y_lf) ** 2).sum().item()
        t = residual_spectrum(r)
        tab = t if tab is None else tab + t
    sigma_z = (r2 / n) ** 0.5
    print(f"  residual RMSE in model space: exact HF {sigma_z:.4f} | dequantised HF {(r2_dq / n) ** 0.5:.4f}", flush=True)
    tab = tab / len(boxes)
    return space, sigma_z, tab.cpu()


def blur3(x):
    """Periodic 3x3x3 mean filter (B,1,D,H,W)."""
    xp = F.pad(x, (1,) * 6, mode="circular")
    return F.avg_pool3d(xp, 3, stride=1)


def loss_weights(lr_counts, alpha):
    w = 1.0 + alpha * blur3(lr_counts)
    return w / w.mean()


def make_loader(ds, args):
    return DataLoader(ds, batch_size=args.sims_per_batch, shuffle=True, num_workers=args.num_workers,
                      pin_memory=True, drop_last=True, persistent_workers=args.num_workers > 0)


@torch.no_grad()
def validate(net, space, interp, val_ds, args, pk128, device, amp_dtype):
    net.eval()
    ids = val_ds.ids[:args.val_max_sims] if args.val_max_sims > 0 else val_ds.ids
    acc = {}
    for idx in ids:
        lr, hr = val_ds.load_boxes(idx)
        srs = generate_box(net, space, interp, lr, device, n_draws=args.val_draws, steps=args.val_steps,
                           method=args.val_method, seed=idx, cond=args.cond, amp_dtype=amp_dtype)
        m = box_metrics(srs, hr, lr, pk128, device)
        for k, v in m.items():
            acc[k] = acc.get(k, 0.0) + v / len(ids)
    return acc


def main():
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    os.makedirs(args.ckpt_dir, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if (args.amp == "bf16" and dev.type == "cuda") else None

    train_ds = CountPatchDataset("train", pad=0, cache=args.cache, max_sets=args.max_train_sets,
                                 workers=max(args.num_workers, 4))
    val_ds = CountPatchDataset("val", pad=0, cache=False)
    loader = make_loader(train_ds, args)
    pk128 = TorchPk(N=128, lbox=args.lbox, n_bins=32, device=dev)

    resume_path = args.resume
    if resume_path == "auto":
        resume_path = os.path.join(args.ckpt_dir, "last.pt")
        if not os.path.exists(resume_path):
            resume_path = ""
    ck = torch.load(resume_path, map_location=dev, weights_only=False) if resume_path else None

    if ck is not None:
        space = ModelSpace.from_state(ck["space"]); interp = Interpolant.from_state(ck["interp"])
    else:
        t0 = time.time()
        space, sigma_z, spec_tab = estimate_stats(train_ds, min(args.stats_boxes, len(train_ds)), args.dequant, dev, args.seed)
        interp = Interpolant(args.source, sigma_z=sigma_z, sigma_scale=args.sigma_scale,
                             spec_table=spec_tab, grid=PATCH)
        print(f"stats from {min(args.stats_boxes, len(train_ds))} boxes in {time.time()-t0:.0f}s: {space}  sigma_z={sigma_z:.4f}", flush=True)

    net = build_unet(args, in_ch=2 if args.cond else 1).to(dev)
    ema = EMA(net, args.ema)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
    step, start_epoch, best = 0, 0, float("inf")
    if ck is not None:
        net.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"]); opt.load_state_dict(ck["opt"])
        step, start_epoch = ck["step"], ck["epoch"] + 1
        best = ck.get("best", float("inf")) if ck["args"].get("select") == args.select else float("inf")
        print(f"resumed {resume_path}: epoch {start_epoch} step {step} best={best:.4f}", flush=True)

    n_par = sum(p.numel() for p in net.parameters()) / 1e6
    print(f"FM source={args.source} cond={args.cond} dequant={args.dequant} aug={args.aug} t={args.t_dist}+{args.t_shift} "
          f"loss_weight={args.loss_weight}(alpha={args.lw_alpha}) "
          f"| UNet {n_par:.1f}M | train sims {len(train_ds)} val sims {len(val_ds)} | patch-batch {args.sims_per_batch*N_PATCHES} | {dev} amp={args.amp}",
          flush=True)
    print(interp, flush=True)
    with open(os.path.join(args.ckpt_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=1)

    g_aug = torch.Generator().manual_seed(args.seed + 1)
    for epoch in range(start_epoch, args.epochs):
        net.train(); t0 = time.time(); run = 0.0; nrun = 0
        for it, (lr_c, hr_c, _) in enumerate(loader):
            B = lr_c.shape[0]
            lr_c = lr_c.reshape(B * N_PATCHES, 1, PATCH, PATCH, PATCH).to(dev, non_blocking=True)
            hr_c = hr_c.reshape(B * N_PATCHES, 1, PATCH, PATCH, PATCH).to(dev, non_blocking=True)
            if args.aug != "none":
                op = random_op(args.aug, g_aug)
                lr_c, hr_c = apply_op(lr_c, op), apply_op(hr_c, op)
            y_hf = space.forward(hr_c)                    # dequantised target, fresh u each step
            y_lf = space.forward(lr_c, dequant=False)     # exact LF
            x0 = interp.sample_source(y_lf)
            t = sample_t(x0.shape[0], args.t_dist, dev, shift=args.t_shift)
            w = loss_weights(lr_c, args.lw_alpha) if args.loss_weight == "lf" else None
            x_t, v_star = interp.path(x0, y_hf, t)
            cond = y_lf if args.cond else None

            # linear LR warm-up
            lr_now = args.lr * min(1.0, (step + 1) / max(1, args.warmup))
            for gr in opt.param_groups: gr["lr"] = lr_now
            opt.zero_grad(set_to_none=True)
            if amp_dtype is not None:
                with torch.autocast("cuda", dtype=amp_dtype):
                    v = net(x_t, t, cond)
                v = v.float()
            else:
                v = net(x_t, t, cond)
            se = (v - v_star) ** 2
            loss = (se * w).mean() if w is not None else se.mean()
            loss.backward()
            if args.grad_clip > 0:
                gnorm = torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip).item()
            else:
                gnorm = 0.0
            opt.step(); ema.update(net); step += 1
            run += loss.item(); nrun += 1
            if (it + 1) % args.log_every == 0:
                print(f"e{epoch} it{it+1}/{len(loader)} loss={run/nrun:.4f} gnorm={gnorm:.2f} lr={lr_now:.2e} "
                      f"{(time.time()-t0)/(it+1):.2f}s/it", flush=True)
                run = 0.0; nrun = 0

        vm = validate(ema.shadow, space, interp, val_ds, args, pk128, dev, amp_dtype)
        print(f"epoch {epoch} {time.time()-t0:.0f}s STITCHED-128 val: L1/vox={vm['l1']:.4f} L1(mean)={vm['l1_mean']:.4f} "
              f"CRPS={vm['crps']:.4f} spread={vm['spread']:.4f} cnt_err={vm['cnt_err']:+.4f} pkRMS={vm['pk_rms']:.4f}", flush=True)

        st = {"model": net.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "step": step,
              "epoch": epoch, "args": vars(args), "space": space.state(), "interp": interp.state(),
              "val": vm, "best": best}
        torch.save(st, os.path.join(args.ckpt_dir, "last.pt"))
        if (epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs:
            torch.save(st, os.path.join(args.ckpt_dir, f"epoch_{epoch+1}.pt"))
        vsel = vm[args.select]
        if vsel < best:
            best = vsel; st["best"] = best
            torch.save(st, os.path.join(args.ckpt_dir, "best.pt"))
            print(f"saved best.pt (val_{args.select}={vsel:.4f})", flush=True)
        with open(os.path.join(args.ckpt_dir, "val_log.jsonl"), "a") as f:
            f.write(json.dumps({"epoch": epoch, "step": step, **vm}) + "\n")
    with open(os.path.join(args.ckpt_dir, "TRAIN_DONE"), "w") as f:
        f.write(f"epochs={args.epochs} step={step} best_{args.select}={best:.5f}\n")
    print("TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
