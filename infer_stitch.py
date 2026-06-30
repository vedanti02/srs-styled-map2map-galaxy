"""Patch-wise inference + stitching: apply a patch-trained G to LR cubes,
patch-by-patch, and stitch the outputs back into 64^3 cubes.

Modes:
  naive   : plain 32^3 tiles in, outputs placed directly (mentor 2.1's
            "simple stitching"). Seams expected at planes x,y,z in {0,32}.
  overlap : halo'd (32+2*pad)^3 tiles in (periodic-wrap halo), the outer pad
            shell of each output is cut off, interiors placed (overlap-tile,
            Ronneberger et al. 2015). Deployment mode of the boundary-masked
            model (mentor 2.2).

Outputs are saved as ``set{sid}_transformed.npy`` (6,64,64,64), denormalized
by default — identical layout to transform.py, so the existing Pk/NDE/KL
pipeline runs on them unchanged.
"""
import argparse
import os
import numpy as np
import torch

from data.pair_dataset import denormalize
from data.patch_dataset import (
    PatchPairDataset, extract_patch, stitch_patches, crop_interior,
    PATCH, N_PATCHES,
)
from map2map.models.styled_srsgan import G_correct
from transform import build_noise_list


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--data-root", default="/data/group_data/universedata/lagrangian_output_64/stitched/")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    p.add_argument("--mode", required=True, choices=["naive", "overlap"])
    p.add_argument("--pad", type=int, default=-1,
                   help="halo width for overlap mode; -1 = take from checkpoint args")
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument("--denormalize", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--save-dtype", default="float32", choices=["float16", "float32"])
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    saved = ckpt.get("args", {}) or {}
    chan_base = saved.get("chan_base_g", 256)
    num_blocks = saved.get("num_blocks", 4)
    pad = 0 if args.mode == "naive" else (args.pad if args.pad >= 0 else saved.get("pad", 3))

    G = G_correct(in_chan=6, out_chan=6, style_size=5,
                  chan_base=chan_base, num_blocks=num_blocks).to(device)
    G.load_state_dict(ckpt["model"])
    G.eval()

    ds = PatchPairDataset(args.data_root, split=args.split, pad=pad,
                          seed=saved.get("seed", 0))
    grid = PATCH + 2 * pad
    noise_list = build_noise_list(grid, num_blocks, args.base_seed, device)
    print(f"mode={args.mode} pad={pad} grid={grid}^3  sims={len(ds.ids)} "
          f"split={args.split}  ckpt_epoch={ckpt.get('epoch')}")

    with torch.no_grad():
        for n, sid in enumerate(ds.ids):
            x_lr, _, theta = ds.load_cubes(sid)
            patches = np.stack([extract_patch(x_lr, p, pad) for p in range(N_PATCHES)])
            xb = torch.from_numpy(patches).to(device)
            tb = torch.from_numpy(theta).unsqueeze(0).expand(N_PATCHES, -1).to(device)
            fake = crop_interior(G(xb, tb, noise_list), pad)
            sr = stitch_patches(fake.cpu().numpy())
            if args.denormalize:
                sr = denormalize(sr)
            dtype = np.float16 if args.save_dtype == "float16" else np.float32
            np.save(os.path.join(args.output_dir, f"set{sid}_transformed.npy"),
                    sr.astype(dtype))
            if (n + 1) % 20 == 0 or n + 1 == len(ds.ids):
                print(f"  {n+1}/{len(ds.ids)} done", flush=True)


if __name__ == "__main__":
    main()
