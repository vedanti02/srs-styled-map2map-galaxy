"""Neural density estimator: P(k) → posterior over θ (5-D cosmological style).

Uses `sbi` (install: pip install sbi). Trains an NPE_C estimator with a Neural
Spline Flow density estimator. Two estimators are typically trained:
  - q_HR(θ | Pk) on (Pk(quijote), θ)
  - q_SR(θ | Pk) on (Pk(model_output), θ)
"""
import argparse
import glob
import os
import re
import pickle
import numpy as np
import torch

try:
    from sbi.inference import SNPE_C
    from sbi.utils import BoxUniform
    SBI_AVAILABLE = True
except ImportError:
    SBI_AVAILABLE = False


_SET_RE = re.compile(r"set(\d+)")


def _collect_pk_theta(pk_dir, stitched_root, snap="PART_009", theta_npz=None):
    """Pair each pk_*.npz with the matching θ. θ from cmass_theta.npz[idx] if
    theta_npz is given, else from stitched_root/set{sid}_{kind}/.../style.npy."""
    theta_table = np.load(theta_npz)["theta"] if theta_npz else None
    files = sorted(glob.glob(os.path.join(pk_dir, "pk_*.npz")))
    X, Y = [], []
    for f in files:
        m = _SET_RE.search(os.path.basename(f))
        if m is None:
            continue
        sid = int(m.group(1))
        if theta_table is not None:
            theta = theta_table[sid].astype(np.float32)
        else:
            theta_path = os.path.join(stitched_root, f"set{sid}_quijote", snap, "style.npy")
            if not os.path.exists(theta_path):
                theta_path = os.path.join(stitched_root, f"set{sid}_quijotelike", snap, "style.npy")
            theta = np.load(theta_path).astype(np.float32)
        pk = np.load(f)["pk"].astype(np.float32)
        # log P(k) is the conventional summary; clip to avoid -inf
        pk = np.log10(np.maximum(pk, 1e-12))
        X.append(pk); Y.append(theta)
    return np.stack(X), np.stack(Y)


def train_nde(pk_dir, stitched_root, out_path, n_train=None,
              theta_low=0.0, theta_high=1.5, hidden=64, num_transforms=5, theta_npz=None):
    if not SBI_AVAILABLE:
        raise RuntimeError("sbi not installed — pip install sbi")

    X, Y = _collect_pk_theta(pk_dir, stitched_root, theta_npz=theta_npz)
    if n_train is not None:
        X, Y = X[:n_train], Y[:n_train]
    print(f"train: X={X.shape} (log Pk), Y={Y.shape} (theta)")

    dim_t = Y.shape[1]
    prior = BoxUniform(
        low=torch.full((dim_t,), float(theta_low)),
        high=torch.full((dim_t,), float(theta_high)),
    )

    inferer = SNPE_C(prior=prior, density_estimator="nsf")
    inferer.append_simulations(torch.from_numpy(Y), torch.from_numpy(X))
    posterior_net = inferer.train(max_num_epochs=100, training_batch_size=64)
    posterior = inferer.build_posterior(posterior_net)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "wb") as fh:
        pickle.dump(posterior, fh)
    print(f"saved posterior to {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pk-dir", required=True, help="Directory of pk_*.npz files (training set).")
    p.add_argument("--stitched-root", default="",
                   help="Root of stitched/ (for looking up θ via set ID). Unused if --theta-npz given.")
    p.add_argument("--theta-npz", default="",
                   help="cmass_theta.npz with 'theta' (idx->5 params); overrides --stitched-root.")
    p.add_argument("--out", required=True, help="Path to save the trained posterior (.pkl).")
    p.add_argument("--n-train", type=int, default=None)
    p.add_argument("--theta-low", type=float, default=0.0)
    p.add_argument("--theta-high", type=float, default=1.5)
    args = p.parse_args()
    train_nde(args.pk_dir, args.stitched_root, args.out,
              n_train=args.n_train, theta_low=args.theta_low, theta_high=args.theta_high,
              theta_npz=args.theta_npz or None)


if __name__ == "__main__":
    main()
