"""Real-scale patch inference + stitching → 128³ cubes (interpretation (b)).

For each sim: assemble LR 128³, run G per 64³ patch, stitch outputs → SR 128³.
  --mode naive    pad=0, place 64³ outputs directly (mentor 2.1)
  --mode overlap  pad>0 halo'd input, crop, place interior (mentor 2.2)
Saves set{sid}_transformed.npy (6,128,128,128), denormalized — power_spectrum
transformed-mode reads these directly. --save-hr also dumps assembled HR cubes.
"""
import argparse, os
import numpy as np
import torch

from data.patch_dataset_real import (
    PatchPairDatasetReal, assemble_box, extract_patch, stitch_patches, crop_interior,
    normalize, denormalize, PATCH, N_PATCHES,
)
from map2map.models.styled_srsgan import G_correct
from transform import build_noise_list


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default="")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    p.add_argument("--mode", default="naive", choices=["naive", "overlap"])
    p.add_argument("--pad", type=int, default=-1, help="overlap halo; -1=from ckpt")
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument("--save-hr", default="", help="if set, also dump assembled HR 128³ cubes here")
    p.add_argument("--hr-only", action="store_true", help="dump assembled HR cubes only (no model)")
    p.add_argument("--save-dtype", default="float32", choices=["float16", "float32"])
    return p.parse_args()


def main():
    args = parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dt = np.float16 if args.save_dtype == "float16" else np.float32

    if args.hr_only:
        os.makedirs(args.output_dir, exist_ok=True)
        ds = PatchPairDatasetReal(split=args.split, pad=0, seed=0)
        print(f"HR-only: dumping {len(ds.ids)} assembled HR 128³ cubes (split={args.split})")
        for n, sid in enumerate(ds.ids):
            hr = assemble_box(ds.root, "quijote-64", sid, ds.snap)
            np.save(os.path.join(args.output_dir, f"set{sid}_transformed.npy"), hr.astype(dt))
            if (n + 1) % 50 == 0 or n + 1 == len(ds.ids):
                print(f"  {n+1}/{len(ds.ids)}", flush=True)
        return

    os.makedirs(args.output_dir, exist_ok=True)
    if args.save_hr:
        os.makedirs(args.save_hr, exist_ok=True)

    ck = torch.load(args.model_path, map_location=dev, weights_only=False)
    saved = ck.get("args", {}) or {}
    cb, nb = saved.get("chan_base_g", 128), saved.get("num_blocks", 4)
    pad = 0 if args.mode == "naive" else (args.pad if args.pad >= 0 else saved.get("pad", 3))

    G = G_correct(6, 6, 5, chan_base=cb, num_blocks=nb).to(dev)
    G.load_state_dict(ck["model"]); G.eval()

    ds = PatchPairDatasetReal(split=args.split, pad=pad, seed=saved.get("seed", 0))
    nl = build_noise_list(PATCH + 2 * pad, nb, args.base_seed, dev)
    dt = np.float16 if args.save_dtype == "float16" else np.float32
    print(f"mode={args.mode} pad={pad} sims={len(ds.ids)} split={args.split} ckpt_epoch={ck.get('epoch')}")

    root, snap = ds.root, ds.snap
    with torch.no_grad():
        for n, sid in enumerate(ds.ids):
            lr = normalize(assemble_box(root, "quijotelike-64", sid, snap))
            patches = np.stack([extract_patch(lr, p, pad) for p in range(N_PATCHES)])
            xb = torch.from_numpy(patches).to(dev)
            th = torch.from_numpy(np.load(os.path.join(root, "quijote-64", f"set{sid}_pos_0_0_0",
                 snap, "style.npy")).astype(np.float32)).unsqueeze(0).expand(N_PATCHES, -1).to(dev)
            fake = crop_interior(G(xb, th, nl), pad)
            sr = denormalize(stitch_patches(fake.cpu().numpy()))
            np.save(os.path.join(args.output_dir, f"set{sid}_transformed.npy"), sr.astype(dt))
            if args.save_hr:
                hr = assemble_box(root, "quijote-64", sid, snap)  # already physical units
                np.save(os.path.join(args.save_hr, f"set{sid}_transformed.npy"), hr.astype(dt))
            if (n + 1) % 25 == 0 or n + 1 == len(ds.ids):
                print(f"  {n+1}/{len(ds.ids)}", flush=True)


if __name__ == "__main__":
    main()
