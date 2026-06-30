"""CMASS patch inference + stitching → 128³ count cubes (Eulerian; no Lagrangian).

For each sim: load LR 128³ count box, to model space, run G per 64³ patch (periodic-wrap
halo for overlap), crop, stitch, invert transform → physical-count SR 128³.
  --mode naive    pad=0, place 64³ outputs directly (mentor 2.1)
  --mode overlap  pad>0 halo'd input, crop, place interior (mentor 2.2)
Saves set{idx}_transformed.npy (1,128,128,128) physical counts — power_spectrum
--estimator counts reads these directly. --hr-only dumps the HR count boxes (no model).
"""
import argparse, os
import numpy as np
import torch

from data.patch_dataset_cmass import (
    PatchPairDatasetCmass, extract_patch, stitch_patches, crop_interior,
    to_counts, PATCH, N_PATCHES,
)
from map2map.models.styled_srsgan import G_correct

CMAX = 200.0  # physical-ish cap on per-cell counts (HR max ~12); guards rare blow-ups / fp16 inf


def build_noise_list(grid, num_blocks, seed, device):
    rng = np.random.default_rng(seed)
    return [torch.from_numpy(rng.standard_normal((1, grid, grid, grid)).astype(np.float32)).to(device)
            for _ in range(2 * num_blocks)]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default="")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    p.add_argument("--mode", default="naive", choices=["naive", "overlap"])
    p.add_argument("--pad", type=int, default=-1, help="overlap halo; -1=from ckpt")
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument("--max-sims", type=int, default=0, help="cap #sims (0=all; for smoke tests)")
    p.add_argument("--hr-only", action="store_true", help="dump HR count boxes only (no model)")
    p.add_argument("--lr-only", action="store_true", help="dump LR count boxes only (no model)")
    p.add_argument("--save-dtype", default="float32", choices=["float16", "float32"])
    return p.parse_args()


def main():
    args = parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dt = np.float16 if args.save_dtype == "float16" else np.float32
    os.makedirs(args.output_dir, exist_ok=True)

    if args.hr_only or args.lr_only:
        which = "LR" if args.lr_only else "HR"
        ds = PatchPairDatasetCmass(split=args.split, pad=0, normalize_inputs=False)  # raw counts
        if args.max_sims > 0: ds.ids = ds.ids[:args.max_sims]
        print(f"{which}-only: dumping {len(ds.ids)} {which} 128³ count cubes (split={args.split})")
        for n, idx in enumerate(ds.ids):
            lr, hr = ds.load_boxes(idx)                     # (1,128,128,128) physical counts
            box = lr if args.lr_only else hr
            np.save(os.path.join(args.output_dir, f"set{idx}_transformed.npy"), box.astype(dt))
            if (n + 1) % 50 == 0 or n + 1 == len(ds.ids):
                print(f"  {n+1}/{len(ds.ids)}", flush=True)
        return

    ck = torch.load(args.model_path, map_location=dev, weights_only=False)
    saved = ck.get("args", {}) or {}
    cb, nb = saved.get("chan_base_g", 128), saved.get("num_blocks", 4)
    transform = saved.get("transform", "log1p")
    pad = 0 if args.mode == "naive" else (args.pad if args.pad >= 0 else saved.get("pad", 8))

    G = G_correct(1, 1, 5, chan_base=cb, num_blocks=nb).to(dev)
    G.load_state_dict(ck["model"]); G.eval()

    ds = PatchPairDatasetCmass(split=args.split, pad=pad, transform=transform)  # model space
    if args.max_sims > 0: ds.ids = ds.ids[:args.max_sims]
    nl = build_noise_list(PATCH + 2 * pad, nb, args.base_seed, dev)
    print(f"mode={args.mode} pad={pad} transform={transform} sims={len(ds.ids)} "
          f"split={args.split} ckpt_epoch={ck.get('epoch')}")

    with torch.no_grad():
        for n, idx in enumerate(ds.ids):
            lr, _ = ds.load_boxes(idx)                      # model space (1,128,128,128)
            patches = np.stack([extract_patch(lr, p, pad) for p in range(N_PATCHES)])
            xb = torch.from_numpy(patches).to(dev)
            th = torch.from_numpy(ds.theta[idx]).unsqueeze(0).expand(N_PATCHES, -1).to(dev)
            fake = crop_interior(G(xb, th, nl), pad)        # model space cores
            sr_model = stitch_patches(fake.cpu().numpy())   # (1,128,128,128) model space
            sr = to_counts(sr_model, transform, ds.scale)   # -> physical counts
            # guard rare model blow-ups: HR counts are O(10); clamp to a generous physical cap
            # (also avoids float16 inf at save, which would poison Pk/seam with NaN).
            sr = np.clip(np.nan_to_num(sr, nan=0.0, posinf=CMAX, neginf=0.0), 0.0, CMAX)
            np.save(os.path.join(args.output_dir, f"set{idx}_transformed.npy"), sr.astype(dt))
            if (n + 1) % 25 == 0 or n + 1 == len(ds.ids):
                print(f"  {n+1}/{len(ds.ids)}", flush=True)


if __name__ == "__main__":
    main()
