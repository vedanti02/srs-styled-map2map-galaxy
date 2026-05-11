"""Generate fixed noise tensors for reproducible same-resolution inference.

Writes 2 * num_blocks tensors of shape (1, N, N, N) into noise_dir, named
`noise_const_{layer}_{id_inside}.npy` so that the inference loader can build
a noise_list keyed by (layer_id, id_inside) matching the model's H-blocks.
"""
import argparse
import os
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--noise-dir", required=True)
    p.add_argument("--grid", type=int, default=64)
    p.add_argument("--num-blocks", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    os.makedirs(args.noise_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    N = args.grid

    for layer in range(args.num_blocks):
        for id_inside in (0, 1):
            noise = rng.standard_normal((1, N, N, N)).astype(np.float32)
            path = os.path.join(args.noise_dir, f"noise_const_{layer}_{id_inside}.npy")
            np.save(path, noise)

    meta = dict(grid=N, num_blocks=args.num_blocks, seed=args.seed)
    np.save(os.path.join(args.noise_dir, "meta.npy"), np.array([meta], dtype=object))
    print(f"wrote {2*args.num_blocks} noise tensors to {args.noise_dir} (seed={args.seed})")


if __name__ == "__main__":
    main()
