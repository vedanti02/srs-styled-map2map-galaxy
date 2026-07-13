"""Training/validation loss curves for the CMASS patch GAN, Arm A vs Arm B.

Parses the slurm training logs (train_patch_cmass.py output):
  per-iter : "e{ep} it{n}/{total}  adv=.. rec=.. pk=..  D(r)=.. D(f)=.."
  per-epoch: "epoch {ep} {sec}s  STITCHED-128 val_L1/vox=.. val_pkRMS=.."
Arm identity comes from the "ARM ..." header each log prints at startup.

Output: figures_cmass/losses_cmass.png
"""
import argparse
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ITER_RE = re.compile(
    r"e(\d+) it(\d+)/(\d+)\s+adv=([-\d.]+) rec=([-\d.]+) pk=([-\d.]+)")
EPOCH_RE = re.compile(
    r"epoch (\d+) [\d.]+s\s+STITCHED-128 val_L1/vox=([-\d.]+) val_pkRMS=([-\d.]+)")
ARM_RE = re.compile(r"ARM (\S+)")


def parse_log(path):
    arm, iters, epochs = None, [], []
    with open(path) as f:
        for line in f:
            if arm is None and (m := ARM_RE.search(line)):
                arm = m.group(1)
            if m := ITER_RE.search(line):
                ep, it, total = int(m.group(1)), int(m.group(2)), int(m.group(3))
                frac = ep + it / total          # fractional epoch for the x-axis
                iters.append((frac, float(m.group(4)), float(m.group(5)), float(m.group(6))))
            elif m := EPOCH_RE.search(line):
                epochs.append((int(m.group(1)), float(m.group(2)), float(m.group(3))))
    return arm, np.array(iters), np.array(epochs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--logs", nargs="+",
                   default=["logs/8841137_patchcmass.log", "logs/8841138_patchcmass.log"])
    p.add_argument("--out", default="figures_cmass/losses_cmass.png")
    args = p.parse_args()

    runs = []
    for path in args.logs:
        arm, iters, epochs = parse_log(path)
        label = "Arm A (pad 0)" if arm and arm.startswith("A") else "Arm B (pad 8)"
        runs.append((label, iters, epochs))
        print(f"{path}: {label}  iter-points={len(iters)}  epochs={len(epochs)}")

    colors = {"Arm A (pad 0)": "C0", "Arm B (pad 8)": "C2"}
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # top row: generator loss terms (per-iter, light) over fractional epoch
    for j, (name, col_idx) in enumerate([("adversarial", 1), ("L1 reconstruction", 2),
                                         ("Pk loss", 3)]):
        ax = axes[0, j]
        for label, iters, _ in runs:
            if len(iters):
                ax.plot(iters[:, 0], iters[:, col_idx], color=colors[label],
                        alpha=0.7, lw=1.2, label=label)
        ax.set_title(name); ax.set_xlabel("epoch"); ax.grid(alpha=0.3)
        if j == 2:
            ax.set_yscale("log")
        ax.legend(fontsize=9)

    # bottom row: stitched-128 validation metrics per epoch + best-epoch marker
    for j, (name, col_idx) in enumerate([("val L1 / voxel (stitched 128)", 1),
                                         ("val Pk RMS log10 (stitched 128)", 2)]):
        ax = axes[1, j]
        for label, _, epochs in runs:
            if len(epochs):
                ax.plot(epochs[:, 0], epochs[:, col_idx], "o-", color=colors[label],
                        lw=1.6, ms=4, label=label)
                if col_idx == 2:   # mark the checkpoint-selection minimum
                    b = np.argmin(epochs[:, 2])
                    ax.plot(epochs[b, 0], epochs[b, 2], "*", color=colors[label],
                            ms=16, mec="k", label=f"{label} best (ep {int(epochs[b,0])})")
        ax.set_title(name); ax.set_xlabel("epoch"); ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    # last panel: epochs completed (the runs stopped at different points)
    ax = axes[1, 2]
    for label, _, epochs in runs:
        ax.bar(label, len(epochs), color=colors[label], width=0.5)
    ax.set_title("epochs completed (30 configured)")
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(args.out, dpi=130)
    print("saved", args.out)

    for label, _, epochs in runs:
        if len(epochs):
            b = np.argmin(epochs[:, 2])
            print(f"{label}: best val_pkRMS={epochs[b,2]:.4f} at epoch {int(epochs[b,0])}"
                  f" (of {len(epochs)} completed)")


if __name__ == "__main__":
    main()
