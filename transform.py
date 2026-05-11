"""Inference: apply trained G to LR cubes from `stitched/`, save transformed cubes.

For posterior sampling, run with multiple --seed values (or --n-noise-samples K
to do all in one call). Each saved file is a (6, 64, 64, 64) float32 npy.
"""
import argparse
import os
import numpy as np
import torch

from data.pair_dataset import PairDataset, denormalize
from map2map.models.styled_srsgan import G_correct


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--data-root", default="/data/group_data/universedata/lagrangian_output_64/stitched/")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    p.add_argument("--n-noise-samples", type=int, default=1,
                   help="K posterior samples per simulation (different noise seeds).")
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument("--denormalize", action=argparse.BooleanOptionalAction, default=True,
                   help="Save outputs in original physical units (multiply by CHAN_STD). "
                        "Default True so Pk on transform.py outputs is comparable to Pk on stitched/.")
    p.add_argument("--save-dtype", default="float32", choices=["float16", "float32"],
                   help="float16 halves disk usage with negligible Pk error. Default float32 for safety.")
    p.add_argument("--eval-z", type=float, default=0.0,
                   help="Only used if the ckpt was trained with style_size=6 (z-aug). "
                        "Inference is done with this redshift in the 6th theta slot.")
    return p.parse_args()


def build_noise_list(grid, num_blocks, seed, device):
    rng = np.random.default_rng(seed)
    return [
        torch.from_numpy(
            rng.standard_normal((1, grid, grid, grid)).astype(np.float32)
        ).to(device)
        for _ in range(2 * num_blocks)
    ]


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    saved = ckpt.get("args", {}) or {}
    chan_base = saved.get("chan_base_g", 256)
    num_blocks = saved.get("num_blocks", 4)

    # Detect style_size from the saved state dict — robust to whether the trainer
    # recorded it in args. ConvStyled3d's style_block.0 is nn.Linear(style_size, in_chan)
    # so its weight has shape (in_chan, style_size).
    sd = ckpt["model"]
    style_size = 5
    for k, v in sd.items():
        if k.endswith("style_block.0.weight"):
            style_size = int(v.shape[1])
            break

    G = G_correct(in_chan=6, out_chan=6, style_size=style_size,
                  chan_base=chan_base, num_blocks=num_blocks).to(device)
    G.load_state_dict(sd)
    G.eval()

    if style_size == 6:
        from data.pair_dataset_zaug import PairDatasetZAug
        ds = PairDatasetZAug(args.data_root, split=args.split,
                             seed=saved.get("seed", 0), fixed_z=args.eval_z)
        print(f"loaded {len(ds)} sims for split={args.split}; "
              f"style_size=6 z-aug @ fixed_z={args.eval_z}; n_noise_samples={args.n_noise_samples}")
    else:
        ds = PairDataset(args.data_root, split=args.split, seed=saved.get("seed", 0))
        print(f"loaded {len(ds)} sims for split={args.split}; "
              f"style_size=5; n_noise_samples={args.n_noise_samples}")

    with torch.no_grad():
        for idx in range(len(ds)):
            x_lr, _, theta, sid = ds[idx]
            x_lr = x_lr.unsqueeze(0).to(device)
            theta = theta.unsqueeze(0).to(device)

            for k in range(args.n_noise_samples):
                seed = args.base_seed + k
                noise_list = build_noise_list(64, num_blocks, seed, device)
                x_fake = G(x_lr, theta, noise_list)
                arr = x_fake.squeeze(0).cpu().numpy()
                if args.denormalize:
                    arr = denormalize(arr)
                tag = f"_seed{seed}" if args.n_noise_samples > 1 else ""
                out = os.path.join(args.output_dir, f"set{sid}_transformed{tag}.npy")
                dtype = np.float16 if args.save_dtype == "float16" else np.float32
                np.save(out, arr.astype(dtype))
            if (idx + 1) % 10 == 0 or idx + 1 == len(ds):
                print(f"  {idx+1}/{len(ds)} done", flush=True)


if __name__ == "__main__":
    main()
