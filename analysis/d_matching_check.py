"""Diagnostic: does D actually discriminate based on theta-correctness, or does it currently
carry no usable theta-consistency signal? DEBUG_HANDOFF.md Section 10.2 makes a STRUCTURAL
argument: D's loss only ever sees matched (real,theta)/(fake,theta) pairs, so it COULD
achieve its whole real/fake task while ignoring theta. That is a fact about the training
objective, not a measurement of D's learned weights -- this script tests it empirically on
an already-trained D.

Two experiments, both using REAL HR fields only (D's realism judgment is not at stake here,
only its theta-sensitivity):

1. MATCHED vs MISMATCHED: for each test sim, compare D(real_field_i, theta_i) [matched] to
   D(real_field_i, theta_j) [mismatched, j != i via a derangement]. If D currently has no
   usable theta-consistency signal, the paired difference should be ~0 relative to its own
   spread. This is the direct empirical test of the Section 10.2 structural claim.

2. PER-PARAMETER SWEEP: at one fixed real field, sweep each of the 5 params across its LH
   prior range (others held at the field's true theta) and report the induced range in
   D's output logit -- mirrors theta_sensitivity_check.py's Experiment 2 for G.
"""
import argparse
import numpy as np
import torch

from data.patch_dataset_cmass import PatchPairDatasetCmass, N_PATCHES
from map2map.models.styled_srsgan import D_const

# Quijote LH prior box, param order (Om, Ob, h, ns, s8) -- same bounds used elsewhere in
# this repo (train_patch_cmass.py, analysis/theta_sensitivity_check.py).
LO = np.array([0.10, 0.03, 0.50, 0.80, 0.60], dtype=np.float32)
HI = np.array([0.50, 0.07, 0.90, 1.20, 1.00], dtype=np.float32)
PARAM_NAMES = ["Om", "Ob", "h", "ns", "s8"]


def normalize_theta_np(theta, mode):
    """Must mirror train_patch_cmass.py's normalize_theta / theta_sensitivity_check.py's
    normalize_theta_np -- read from the checkpoint's saved args, not assumed."""
    if mode == "none":
        return theta
    return (theta - LO) / (HI - LO)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--n-sims", type=int, default=40,
                   help="number of test sims for the matched-vs-mismatched comparison")
    p.add_argument("--n-sweep", type=int, default=9)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    s = ck.get("args", {}) or {}
    cb_d = s.get("chan_base_d", 64)
    nb = s.get("num_blocks", 4)
    transform = s.get("transform", "log1p")
    use_encoder = bool(s.get("use_cosmo_encoder", False))
    embed_dim = s.get("cosmo_embed_dim", 32)
    theta_norm = "none" if use_encoder else s.get("theta_norm", "none")

    if "D" not in ck:
        raise RuntimeError(f"{args.ckpt} has no saved discriminator state ('D' key) -- "
                            f"was this checkpoint trained with the GAN on?")

    D = D_const(1, 5, chan_base=cb_d, num_blocks=nb,
               use_cosmo_encoder=use_encoder, cosmo_embed_dim=embed_dim).to(dev)
    D.load_state_dict(ck["D"]); D.eval()

    ds = PatchPairDatasetCmass(split=args.split, pad=0, transform=transform)
    n = min(args.n_sims, len(ds.ids))
    print(f"D matching-aware check: ckpt_epoch={ck.get('epoch')}  n_sims={n}  "
          f"use_cosmo_encoder={use_encoder}  theta_norm={theta_norm}")

    hr_patches, thetas = [], []
    for k in range(n):
        _, hr_tgt, theta, _ = ds[k]
        hr_patches.append(hr_tgt.numpy())              # (N_PATCHES,1,64,64,64)
        thetas.append(theta.numpy())
    hr_patches = np.stack(hr_patches)                   # (n, N_PATCHES,1,64,64,64)
    thetas = np.stack(thetas).astype(np.float32)         # (n, 5)

    @torch.no_grad()
    def d_logit(hr_np, theta_vec):
        x = torch.from_numpy(hr_np).to(dev)
        th_np = normalize_theta_np(theta_vec.astype(np.float32), theta_norm)
        th = torch.from_numpy(th_np).unsqueeze(0).expand(N_PATCHES, -1).to(dev)
        return D(x, th).mean().item()

    # --- Experiment 1: matched vs mismatched ---
    matched = np.array([d_logit(hr_patches[i], thetas[i]) for i in range(n)])
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    while np.any(perm == np.arange(n)):   # derangement: no sim paired with its own theta
        perm = rng.permutation(n)
    mismatched = np.array([d_logit(hr_patches[i], thetas[perm[i]]) for i in range(n)])

    diff = matched - mismatched
    print(f"\n=== EXPERIMENT 1: D(real_field, theta) -- matched vs mismatched theta, "
          f"{n} test sims ===")
    print(f"  matched    logit: mean={matched.mean():+.4f}  std={matched.std():.4f}")
    print(f"  mismatched logit: mean={mismatched.mean():+.4f}  std={mismatched.std():.4f}")
    print(f"  paired diff (matched-mismatched): mean={diff.mean():+.4f}  std={diff.std():.4f}  "
          f"effect size (mean/std)={diff.mean()/max(diff.std(),1e-12):.3f}")
    print(f"  (if this is ~0 relative to its own spread, D currently cannot distinguish "
          f"correct from incorrect theta for a real field -- direct empirical evidence for "
          f"--lambda-match, not just the structural argument in DEBUG_HANDOFF.md Section 10.2)")

    # --- Experiment 2: per-parameter sweep at one fixed real field ---
    print(f"\n=== EXPERIMENT 2: per-parameter sweep of D(real_field, theta), other 4 held at "
          f"true theta ({args.n_sweep} points, sim {ds.ids[0]}) ===")
    ranked = []
    for i, name in enumerate(PARAM_NAMES):
        grid = np.linspace(LO[i], HI[i], args.n_sweep, dtype=np.float32)
        logits = []
        for v in grid:
            th = thetas[0].copy(); th[i] = v
            logits.append(d_logit(hr_patches[0], th))
        logits = np.array(logits)
        rng_ = logits.max() - logits.min()
        ranked.append((name, rng_))
        print(f"  {name:3s} sweep [{LO[i]:.3f},{HI[i]:.3f}]: range(logit) = {rng_:.4f}  "
              f"logits={np.round(logits, 3).tolist()}")
    ranked.sort(key=lambda t: -t[1])
    print(f"\n  ranked by induced logit range (most to least sensitive): "
          f"{[n_ for n_, _ in ranked]}")


if __name__ == "__main__":
    main()
