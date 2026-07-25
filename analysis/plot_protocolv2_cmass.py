"""Stitched validation curves for the protocol-v2 runs (no Pk loss in training):
  - no-pad Arm A            (9250956 -> resumed as 9269632)
  - padding, trim-aware     (9252145 -> resumed as 9269633)
  - padding, trim-unaware   (9252146 -> resumed as 9269634)
Each run crashed once on a shared-storage NFS hiccup and was resumed from its
last checkpoint; this script stitches the pre-crash and post-resume segments
into one trajectory per run (keeping the resumed segment's value on any epoch
computed twice, since it is the more recent/authoritative one).

Also overlays the original (Pk-loss-on) Arm A run for the no-Pk-vs-Pk story.

Output: figures_cmass/protocolv2_cmass.png
"""
import argparse
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EPOCH_RE = re.compile(r"epoch (\d+) [\d.]+s\s+STITCHED-128 val_L1/vox=([-\d.]+) val_pkRMS=([-\d.]+)")
RESUME_RE = re.compile(r"resumed at epoch (\d+)")


def parse_log(path):
    epochs = {}
    resume_at = None
    try:
        with open(path) as f:
            for line in f:
                if resume_at is None and (m := RESUME_RE.search(line)):
                    resume_at = int(m.group(1))
                if m := EPOCH_RE.search(line):
                    epochs[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
    except FileNotFoundError:
        pass
    return epochs, resume_at


def stitched_trajectory(paths):
    """paths: chronological list of log files for one run (crash -> resume -> ...).
    Later files' epochs override earlier ones on overlap."""
    merged = {}
    for p in paths:
        epochs, _ = parse_log(p)
        merged.update(epochs)
    ep = sorted(merged)
    l1 = np.array([merged[e][0] for e in ep])
    pk = np.array([merged[e][1] for e in ep])
    return np.array(ep), l1, pk


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="figures_cmass/protocolv2_cmass.png")
    args = p.parse_args()

    runs = {
        "no-pad, Pk-loss ON (original A)": ["logs/8841137_patchcmass.log"],
        "no-pad, Pk-loss OFF": ["logs/9250956_patchcmass.log", "logs/9269632_patchcmass.log"],
        "pad 8, trim-aware, Pk OFF": ["logs/9252145_patchcmass.log", "logs/9269633_patchcmass.log"],
        "pad 8, trim-unaware, Pk OFF": ["logs/9252146_patchcmass.log", "logs/9269634_patchcmass.log"],
    }
    colors = {
        "no-pad, Pk-loss ON (original A)": "grey",
        "no-pad, Pk-loss OFF": "C0",
        "pad 8, trim-aware, Pk OFF": "C2",
        "pad 8, trim-unaware, Pk OFF": "C3",
    }
    styles = {
        "no-pad, Pk-loss ON (original A)": "--",
        "no-pad, Pk-loss OFF": "-",
        "pad 8, trim-aware, Pk OFF": "-",
        "pad 8, trim-unaware, Pk OFF": "-",
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name, paths in runs.items():
        ep, l1, pk = stitched_trajectory(paths)
        if len(ep) == 0:
            continue
        axes[0].plot(ep, l1, styles[name], color=colors[name], marker="o", ms=3, label=name)
        axes[1].plot(ep, pk, styles[name], color=colors[name], marker="o", ms=3, label=name)
        print(f"{name}: {len(ep)} epochs, last = L1 {l1[-1]:.4f} / PkRMS {pk[-1]:.4f}")

    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("stitched val L1 / voxel")
    axes[0].set_title("Real-space accuracy"); axes[0].grid(alpha=0.3); axes[0].legend(fontsize=8)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("stitched val Pk RMS (log10)")
    axes[1].set_title("Power-spectrum match"); axes[1].grid(alpha=0.3)
    axes[1].set_ylim(0.08, 0.20)

    plt.tight_layout()
    plt.savefig(args.out, dpi=130)
    print("saved", args.out)


if __name__ == "__main__":
    main()
