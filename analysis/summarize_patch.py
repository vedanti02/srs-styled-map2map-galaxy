"""Aggregate patch-experiment results into one comparison table + combined figures.

Reads, for each run tag:
  runs/patch/seam_<tag>.npz     (from analysis.eval_seam)
  runs/patch/metrics_<tag>.npz  (from evaluate.py)
plus baseline references (LR, whole-box v2) for context.

Writes:
  runs/patch/SUMMARY.md            full comparison table
  runs/patch/pkratio_all.png       P_SR/P_HR curves, all arms overlaid
  runs/patch/seamprofile_all.png   error-vs-seam-distance, all arms overlaid
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PARAMS = ["Om", "Ob", "h", "ns", "s8"]
RUNDIR = "runs/patch"

# (tag, label, kind) — kind: 'patch' has seam+metrics, 'ref' metrics only
RUNS = [
    ("patchA_naive",   "Arm A · naive stitch (2.1)",       "patch"),
    ("patchA_overlap", "Arm A · overlap-tile pad3 (abl)",  "patch"),
    ("patchB_overlap", "Arm B · boundary-masked (2.2)",    "patch"),
]
# seam-only rows (no NDE/KL): controls + receptive-field ablation
SEAM_ONLY = [
    ("wholebox_v2",         "whole-box v2 (control, no patches)"),
    ("wholebox_v3",         "whole-box v3 (control, no patches)"),
    ("patchA_overlap_pad8", "Arm A · overlap-tile pad8 (RF abl)"),
    ("patchB_overlap_pad8", "Arm B · overlap-tile pad8 (RF abl)"),
]
REFS = [
    ("runs/baseline/metrics_hr_vs_lr.npz", "LR (no model)"),
    ("runs/baseline/metrics_v2_best.npz",  "whole-box v2 (ref)"),
]


def mean_kl(npz_path):
    if not os.path.exists(npz_path):
        return None
    d = np.load(npz_path, allow_pickle=True)
    return d["kl_hr_to_sr"].mean(axis=0)  # (5,) per-param mean over test sims


def load_seam(tag):
    p = os.path.join(RUNDIR, f"seam_{tag}.npz")
    return np.load(p, allow_pickle=True) if os.path.exists(p) else None


def fmt(x):
    if x is None:
        return "—"
    return f"{x:.3g}" if abs(x) < 100 else f"{x:.0f}"


def main():
    os.makedirs(RUNDIR, exist_ok=True)
    lines = ["# Patch experiment — summary\n"]

    # --- seam / Pk fidelity table ---
    lines.append("## Stitching fidelity & seam artifacts (test split)\n")
    lines.append("| run | seam MAE (d=0) | interior MAE (d≥8) | **seam ratio** | stitched Pk RMS (log10) |")
    lines.append("|---|---:|---:|---:|---:|")
    seam_data = {}
    for tag, label, _ in RUNS:
        s = load_seam(tag)
        seam_data[tag] = s
        if s is None:
            lines.append(f"| {label} | — | — | — | — |")
            continue
        lines.append(f"| {label} | {float(s['seam_err']):.4f} | {float(s['interior_err']):.4f} "
                     f"| **{float(s['seam_ratio']):.4f}** | {float(s['pk_rms']):.4f} |")
    for tag, label in SEAM_ONLY:
        s = load_seam(tag)
        seam_data[tag] = s
        if s is None:
            lines.append(f"| {label} | — | — | — | — |")
            continue
        lines.append(f"| {label} | {float(s['seam_err']):.4f} | {float(s['interior_err']):.4f} "
                     f"| **{float(s['seam_ratio']):.4f}** | {float(s['pk_rms']):.4f} |")
    lines.append("\n*seam ratio = 1.0 means no detectable seam; >1 means error concentrates at patch boundaries.*")
    lines.append("*Controls (whole-box, no patches) calibrate the metric's null. RF-ablation = overlap "
                 "inference with halo pad=8 (≥ receptive-field radius); tests whether inference context fixes seams.*\n")

    # --- KL table ---
    lines.append("## Posterior agreement KL(q_HR ‖ q_X), lower=better (mean over 200 test sims)\n")
    header = "| param | " + " | ".join(lbl for _, lbl in REFS) + " | " + \
             " | ".join(lbl for _, lbl, _ in RUNS) + " |"
    lines.append(header)
    lines.append("|---" * (1 + len(REFS) + len(RUNS)) + "|")
    kl_cols = {}
    for path, lbl in REFS:
        kl_cols[lbl] = mean_kl(path)
    for tag, lbl, _ in RUNS:
        kl_cols[lbl] = mean_kl(os.path.join(RUNDIR, f"metrics_{tag}.npz"))
    for i, pname in enumerate(PARAMS):
        row = [f"| {pname} "]
        for _, lbl in REFS:
            v = kl_cols[lbl]
            row.append(f"| {fmt(v[i]) if v is not None else '—'} ")
        for _, lbl, _ in RUNS:
            v = kl_cols[lbl]
            row.append(f"| {fmt(v[i]) if v is not None else '—'} ")
        lines.append("".join(row) + "|")
    # mean-KL row
    row = ["| **mean** "]
    for _, lbl in REFS + [(t, l) for t, l, _ in RUNS]:
        v = kl_cols[lbl]
        row.append(f"| {fmt(v.mean()) if v is not None else '—'} ")
    lines.append("".join(row) + "|")
    lines.append("")

    with open(os.path.join(RUNDIR, "SUMMARY.md"), "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))

    # --- combined Pk-ratio figure ---
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for tag, label, _ in RUNS:
        s = seam_data.get(tag)
        if s is None:
            continue
        pk_sr, pk_hr, k = s["pk_sr"], s["pk_hr"], s["k"]
        m = pk_hr.mean(0) > -10
        ratio = (10 ** (pk_sr - pk_hr)).mean(0)
        ax.semilogx(k[m], ratio[m], lw=2, marker="o", ms=3, label=label)
    ax.axhline(1.0, color="k", ls=":")
    ax.set_xlabel("k [h/Mpc]"); ax.set_ylabel("P_SR(k) / P_HR(k)")
    ax.set_title("Stitched power-spectrum ratio vs HR"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(RUNDIR, "pkratio_all.png"), dpi=140)
    plt.close(fig)

    # --- combined seam-profile figure ---
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for tag, label, _ in RUNS:
        s = seam_data.get(tag)
        if s is None:
            continue
        prof = s["profile"]
        ax.plot(range(len(prof)), prof, "o-", label=f"{label} (ratio {float(s['seam_ratio']):.3f})")
    ax.set_xlabel("voxel distance to nearest patch seam")
    ax.set_ylabel("displacement MAE [Mpc/h]")
    ax.set_title("Error vs distance to seam"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(RUNDIR, "seamprofile_all.png"), dpi=140)
    plt.close(fig)
    print(f"\nwrote {RUNDIR}/SUMMARY.md, pkratio_all.png, seamprofile_all.png")


if __name__ == "__main__":
    main()
