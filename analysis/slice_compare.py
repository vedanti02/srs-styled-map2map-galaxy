"""Compare HR, LR, and SR cube slices visually.

For a given simulation set ID, loads:
  - HR (quijote) and LR (quijotelike) from stitched/
  - SR from a transform.py output directory
and plots a 3-row grid of disp_x slice (z=32) for visual inspection.
"""
import argparse
import os
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stitched-root", default="/data/group_data/universedata/lagrangian_output_64/stitched")
    p.add_argument("--transformed-dir", required=True)
    p.add_argument("--sid", type=int, required=True)
    p.add_argument("--snap", default="PART_009")
    p.add_argument("--seed", type=int, default=0,
                   help="If transform.py was run with --n-noise-samples > 1.")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    hr_disp = np.load(os.path.join(args.stitched_root, f"set{args.sid}_quijote", args.snap, "disp.npy"))
    lr_disp = np.load(os.path.join(args.stitched_root, f"set{args.sid}_quijotelike", args.snap, "disp.npy"))

    # SR cube includes 6 channels. transform.py saves normalized; if --denormalize was
    # used the file would be in physical units. Try to detect.
    sr_path = os.path.join(args.transformed_dir, f"set{args.sid}_transformed.npy")
    if not os.path.exists(sr_path):
        sr_path = os.path.join(args.transformed_dir, f"set{args.sid}_transformed_seed{args.seed}.npy")
    sr_full = np.load(sr_path)
    sr_disp = sr_full[:3]
    if abs(sr_disp.std() - 1.0) < 0.5:  # normalized output → bring back to physical scale
        sr_disp = sr_disp * 18.0  # CHAN_STD[disp]

    z = 32
    sl_hr = hr_disp[0, z]
    sl_lr = lr_disp[0, z]
    sl_sr = sr_disp[0, z]
    print(f"set{args.sid} slice statistics @ z={z}:")
    for name, sl in [("HR", sl_hr), ("LR", sl_lr), ("SR", sl_sr)]:
        print(f"  {name}: shape={sl.shape}, mean={sl.mean():+.2f}, std={sl.std():.2f}, "
              f"min={sl.min():+.2f}, max={sl.max():+.2f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        vmax = max(abs(sl_hr).max(), abs(sl_sr).max(), abs(sl_lr).max())
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
        for ax, sl, ttl in zip(axes, [sl_lr, sl_sr, sl_hr], ["LR (quijotelike)", "SR (GAN output)", "HR (quijote)"]):
            im = ax.imshow(sl, cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower")
            ax.set_title(ttl)
            ax.set_xticks([]); ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(f"set{args.sid} disp_x slice z={z}")
        fig.tight_layout()
        fig.savefig(args.output, dpi=120, bbox_inches="tight")
        print(f"saved {args.output}")
    except ImportError:
        print("matplotlib not available; skipped PNG")


if __name__ == "__main__":
    main()
