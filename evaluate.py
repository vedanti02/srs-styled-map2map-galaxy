"""Compare posteriors q_HR(θ|Pk_HR) and q_SR(θ|Pk_SR) on a held-out test set.

For each test simulation, samples N posterior draws from each estimator and
reports per-parameter mean/std + a posterior-distance metric.
"""
import argparse
import glob
import os
import pickle
import re
import numpy as np
import torch


_SET_RE = re.compile(r"set(\d+)")


def _load_pk_set(pk_dir, prefix):
    """Returns {sid: log10 Pk vector} for filenames matching `prefix*set{sid}*.npz`."""
    out = {}
    for f in sorted(glob.glob(os.path.join(pk_dir, f"{prefix}*.npz"))):
        m = _SET_RE.search(os.path.basename(f))
        if m is None:
            continue
        sid = int(m.group(1))
        pk = np.load(f)["pk"].astype(np.float32)
        out[sid] = np.log10(np.maximum(pk, 1e-12))
    return out


def _load_theta(stitched_root, sid, snap="PART_009"):
    for kind in ("quijote", "quijotelike"):
        p = os.path.join(stitched_root, f"set{sid}_{kind}", snap, "style.npy")
        if os.path.exists(p):
            return np.load(p).astype(np.float32)
    raise FileNotFoundError(sid)


def _load_posterior(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _sample(post, x, n):
    s = post.sample((n,), x=torch.from_numpy(x), show_progress_bars=False)
    return s.cpu().numpy()


def kl_gauss(mu1, std1, mu2, std2, eps=1e-8):
    """Per-parameter KL(N(mu1,std1) || N(mu2,std2))."""
    s1 = std1 ** 2 + eps
    s2 = std2 ** 2 + eps
    return 0.5 * (np.log(s2 / s1) + (s1 + (mu1 - mu2) ** 2) / s2 - 1.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--posterior-hr", required=True, help="Pickled q_HR posterior.")
    p.add_argument("--posterior-sr", required=True, help="Pickled q_SR posterior.")
    p.add_argument("--pk-hr-dir", required=True, help="Test-set Pk dir (HR side).")
    p.add_argument("--pk-sr-dir", required=True, help="Test-set Pk dir (SR side).")
    p.add_argument("--pk-hr-prefix", default="pk_quijote_")
    p.add_argument("--pk-sr-prefix", default="pk_set",
                   help="Filename prefix for SR Pk files. Use 'pk_quijotelike_' for LR baseline.")
    p.add_argument("--stitched-root", required=True)
    p.add_argument("--n-samples", type=int, default=2000)
    p.add_argument("--output", required=True, help="Output .npz with per-sim metrics.")
    p.add_argument("--restrict-sids", default=None,
                   help="Path to a .npz with array 'val_sids' or 'test_sids' to restrict evaluation. "
                        "If omitted, intersect HR/SR sims and use all common ones.")
    args = p.parse_args()

    q_hr = _load_posterior(args.posterior_hr)
    q_sr = _load_posterior(args.posterior_sr)

    pk_hr = _load_pk_set(args.pk_hr_dir, prefix=args.pk_hr_prefix)
    pk_sr = _load_pk_set(args.pk_sr_dir, prefix=args.pk_sr_prefix)

    common = sorted(set(pk_hr) & set(pk_sr))
    if args.restrict_sids:
        z = np.load(args.restrict_sids)
        keep = set()
        for k in ("val_sids", "test_sids", "sids"):
            if k in z:
                keep.update(z[k].tolist())
        common = [s for s in common if s in keep]
    # Use the same train/val/test split as training to identify held-out sims.
    # PairDataset deterministic split with seed=0 — replicate here to filter.
    if args.restrict_sids is None:
        from data.pair_dataset import PairDataset
        val_ds = PairDataset(args.stitched_root, split="val", seed=0)
        common = [s for s in common if s in set(val_ds.ids)]
    print(f"evaluating on {len(common)} common held-out sims")

    rows = []
    for sid in common:
        x_hr = pk_hr[sid]; x_sr = pk_sr[sid]
        theta_true = _load_theta(args.stitched_root, sid)

        s_hr = _sample(q_hr, x_hr, args.n_samples)
        s_sr = _sample(q_sr, x_sr, args.n_samples)

        mu_hr, std_hr = s_hr.mean(0), s_hr.std(0)
        mu_sr, std_sr = s_sr.mean(0), s_sr.std(0)
        kl = kl_gauss(mu_hr, std_hr, mu_sr, std_sr)

        rows.append({
            "sid": sid, "theta_true": theta_true,
            "mu_hr": mu_hr, "std_hr": std_hr,
            "mu_sr": mu_sr, "std_sr": std_sr,
            "kl_hr_to_sr": kl,
        })

    sids = np.array([r["sid"] for r in rows])
    theta_true = np.stack([r["theta_true"] for r in rows])
    mu_hr = np.stack([r["mu_hr"] for r in rows])
    std_hr = np.stack([r["std_hr"] for r in rows])
    mu_sr = np.stack([r["mu_sr"] for r in rows])
    std_sr = np.stack([r["std_sr"] for r in rows])
    kls = np.stack([r["kl_hr_to_sr"] for r in rows])

    print("=== summary (per parameter, averaged over test sims) ===")
    print("|μ_HR − μ_SR|       :", np.abs(mu_hr - mu_sr).mean(0).round(4))
    print("mean σ_HR           :", std_hr.mean(0).round(4))
    print("mean σ_SR           :", std_sr.mean(0).round(4))
    print("KL(HR‖SR) Gauss-approx:", kls.mean(0).round(4))
    print("HR bias |μ−θ|       :", np.abs(mu_hr - theta_true).mean(0).round(4))
    print("SR bias |μ−θ|       :", np.abs(mu_sr - theta_true).mean(0).round(4))

    np.savez(args.output,
             sids=sids, theta_true=theta_true,
             mu_hr=mu_hr, std_hr=std_hr,
             mu_sr=mu_sr, std_sr=std_sr,
             kl_hr_to_sr=kls)
    print(f"saved metrics to {args.output}")


if __name__ == "__main__":
    main()
