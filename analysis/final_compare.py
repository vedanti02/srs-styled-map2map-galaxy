"""Compare v1, v2, v3 final results across multiple metrics.

Loads:
  - HR, LR, SR_v1, SR_v2, SR_v3 Pk per simulation
  - Their corresponding posterior comparison .npz files
and prints a unified ranking table.
"""
import argparse
import os
import numpy as np


def _load_pk_dir(d, prefix):
    import glob, re
    files = sorted(glob.glob(os.path.join(d, f"pk_{prefix}*.npz")))
    out = {}
    for f in files:
        m = re.search(r"set(\d+)", os.path.basename(f))
        if m is None: continue
        sid = int(m.group(1))
        z = np.load(f)
        out[sid] = z["pk"]
    return out


def _pk_stats(pk_sr, pk_hr, pk_lr=None, valid_mask=None):
    """Return median/mean/p95 fractional Pk error per sim, vs HR."""
    common = sorted(set(pk_sr) & set(pk_hr))
    sr = np.stack([pk_sr[s] for s in common])
    hr = np.stack([pk_hr[s] for s in common])
    bad = np.isnan(sr).any(1) | np.isnan(hr).any(1)
    sr = sr[~bad]; hr = hr[~bad]
    eps = 1e-12
    if valid_mask is None:
        valid_mask = (hr.mean(0) > 0)
    err = np.abs(sr[:, valid_mask] - hr[:, valid_mask]) / np.maximum(hr[:, valid_mask], eps)
    per_sim = err.mean(1)
    return {
        "n_sims": len(common) - bad.sum(),
        "median": float(np.median(per_sim)),
        "mean": float(per_sim.mean()),
        "p95": float(np.percentile(per_sim, 95)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pk-hr-dir", default="runs/baseline/pk/hr")
    p.add_argument("--pk-lr-dir", default="runs/baseline/pk/lr")
    p.add_argument("--versions", nargs="+", default=[
        "sr_e31_v1full",
    ], help="Subdirs of runs/baseline/pk/ to compare.")
    p.add_argument("--metrics-files", nargs="+", default=[
        "metrics_hr_vs_lr.npz",
        "metrics_v1_e31.npz",
    ], help="Posterior metric files to summarize.")
    p.add_argument("--metrics-dir", default="runs/baseline")
    args = p.parse_args()

    # ---------- Pk-level ----------
    pk_hr = _load_pk_dir(args.pk_hr_dir, "quijote_")
    pk_lr = _load_pk_dir(args.pk_lr_dir, "quijotelike_")

    print(f"=== Pk-level fractional error vs HR (over all sims) ===\n")
    print(f"{'version':<25} {'n':>5}  {'median':>8}  {'mean':>8}  {'p95':>8}")
    s = _pk_stats(pk_lr, pk_hr)
    print(f"{'LR (no GAN baseline)':<25} {s['n_sims']:>5}  {s['median']:>8.3f}  {s['mean']:>8.3f}  {s['p95']:>8.3f}")
    for ver in args.versions:
        d = os.path.join("runs/baseline/pk", ver)
        if not os.path.exists(d):
            print(f"{ver:<25}  (not found)"); continue
        pk_sr = _load_pk_dir(d, "set")
        if not pk_sr:
            print(f"{ver:<25}  (no Pk files)"); continue
        s = _pk_stats(pk_sr, pk_hr)
        print(f"{ver:<25} {s['n_sims']:>5}  {s['median']:>8.3f}  {s['mean']:>8.3f}  {s['p95']:>8.3f}")

    # ---------- Posterior-level ----------
    print(f"\n=== Posterior agreement vs HR (per-parameter, lower = better) ===\n")
    print(f"{'metric_file':<35}", end="")
    for p_idx in range(5):
        print(f" {'p'+str(p_idx)+'_KL':>11}", end="")
    print()
    print(f"{'metric_file':<35}", end="")
    for p_idx in range(5):
        print(f" {'p'+str(p_idx)+'_bias':>11}", end="")
    print()

    for mf in args.metrics_files:
        path = os.path.join(args.metrics_dir, mf)
        if not os.path.exists(path):
            print(f"{mf:<35}  (not found)"); continue
        z = np.load(path)
        kl_row = z["kl_hr_to_sr"].mean(0)
        bias_sr = np.abs(z["mu_sr"] - z["theta_true"]).mean(0)
        bias_hr = np.abs(z["mu_hr"] - z["theta_true"]).mean(0)
        print(f"{mf:<35}", end="")
        for p in range(5):
            print(f" {kl_row[p]:>11.3g}", end="")
        print()
        print(f"  -> SR bias                          ", end="")
        for p in range(5):
            print(f" {bias_sr[p]:>11.4f}", end="")
        print()
        print(f"  -> HR bias                          ", end="")
        for p in range(5):
            print(f" {bias_hr[p]:>11.4f}", end="")
        print()
    print()


if __name__ == "__main__":
    main()
