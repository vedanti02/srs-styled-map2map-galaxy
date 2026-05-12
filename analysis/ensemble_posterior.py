"""Posterior ensembling: combine per-sim Gaussian-approx posteriors across
multiple trained SR variants into a mixture-of-Gaussians, then approximate as
a single Gaussian for the same KL-vs-HR metric used in the rest of the
project.

Mixture moments:
    mu_mix  = mean_i(mu_i)
    var_mix = mean_i(var_i + (mu_i - mu_mix)^2)

This is cheap (no extra training, no extra GPU) and tends to beat any single
SR variant because each model captures a slightly different slice of the
Pk -> theta mapping that the NDE has fit to.
"""
from __future__ import annotations

import argparse
import os
import numpy as np


def kl_gauss(mu1, std1, mu2, std2, eps=1e-8):
    s1 = std1 ** 2 + eps
    s2 = std2 ** 2 + eps
    return 0.5 * (np.log(s2 / s1) + (s1 + (mu1 - mu2) ** 2) / s2 - 1.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metrics", nargs="+", required=True,
                   help="Paths to metrics_<tag>.npz files to ensemble.")
    p.add_argument("--output", required=True,
                   help="Output metrics_<tag>.npz for the ensemble.")
    args = p.parse_args()

    arrs = [np.load(p) for p in args.metrics]
    # All evals share the same HR side (seed=0 makes this exact).
    mu_hr = arrs[0]["mu_hr"]
    std_hr = arrs[0]["std_hr"]
    theta_true = arrs[0]["theta_true"]
    sids = arrs[0]["sids"]

    mus = np.stack([a["mu_sr"] for a in arrs], axis=0)
    stds = np.stack([a["std_sr"] for a in arrs], axis=0)

    mu_mix = mus.mean(0)
    var_mix = (stds ** 2).mean(0) + ((mus - mu_mix[None]) ** 2).mean(0)
    std_mix = np.sqrt(var_mix)

    kl = kl_gauss(mu_hr, std_hr, mu_mix, std_mix)

    print(f"ensemble of {len(arrs)} variants from:")
    for p in args.metrics:
        print(f"  - {p}")
    print(f"on {len(sids)} test sims")
    print(f"KL(HR || ensemble) per param: {kl.mean(0).round(4)}")
    print(f"HR bias  per param:           {np.abs(mu_hr - theta_true).mean(0).round(4)}")
    print(f"SR bias  per param (ens):     {np.abs(mu_mix - theta_true).mean(0).round(4)}")
    print(f"mean sigma_HR:                {std_hr.mean(0).round(4)}")
    print(f"mean sigma_SR (ens):          {std_mix.mean(0).round(4)}")

    np.savez(args.output,
             sids=sids, theta_true=theta_true,
             mu_hr=mu_hr, std_hr=std_hr,
             mu_sr=mu_mix, std_sr=std_mix,
             kl_hr_to_sr=kl)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
