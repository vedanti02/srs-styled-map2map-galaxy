"""Aggregate real-scale (128³) patch results: seam + θ/KL into one table + figures.
Reads runs/patch_real/{seam_<tag>.npz, metrics_<tag>.npz}."""
import os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

PARAMS = ["Om", "Ob", "h", "ns", "s8"]
R = "runs/patch_real"
RUNS = [("realA_naive", "Arm A · naive stitch (2.1)"),
        ("realB_overlap", "Arm B · boundary-masked overlap (2.2)")]


def seam(tag):
    p = f"{R}/seam_{tag}.npz"
    return np.load(p, allow_pickle=True) if os.path.exists(p) else None


def kl(tag):
    p = f"{R}/metrics_{tag}.npz"
    return np.load(p, allow_pickle=True)["kl_hr_to_sr"].mean(0) if os.path.exists(p) else None


def fmt(x): return "—" if x is None else (f"{x:.3g}" if abs(x) < 100 else f"{x:.0f}")


def main():
    L = ["# Real-scale patch experiment (128³, 64³ patches) — summary\n"]
    L += ["## Seam artifacts & fidelity (test split, assembled-HR reference)\n",
          "| run | seam ratio (d0/d≥8) | **x=64 internal-seam excess** | x=0 periodic-edge (ctrl) | stitched Pk RMS |",
          "|---|---:|---:|---:|---:|"]
    sd = {}
    for tag, lab in RUNS:
        s = seam(tag); sd[tag] = s
        if s is None: L.append(f"| {lab} | — | — | — | — |"); continue
        L.append(f"| {lab} | {float(s['seam_ratio']):.4f} | **{float(s['x64_excess']):.4f}** | "
                 f"{float(s['x0_excess']):.4f} | {float(s['pk_rms']):.4f} |")
    L += ["\n*x=64 = pure internal patch seam (the 2.1/2.2 signature). x=0 = periodic box edge "
          "(present in any model; control). Excess = error at plane / interior mean.*\n"]

    L += ["## Posterior agreement KL(q_HR‖q_SR), mean over test sims (lower=better)\n",
          "| param | " + " | ".join(l for _, l in RUNS) + " |",
          "|---" * (1 + len(RUNS)) + "|"]
    kls = {t: kl(t) for t, _ in RUNS}
    for i, pn in enumerate(PARAMS):
        L.append(f"| {pn} | " + " | ".join(fmt(kls[t][i]) if kls[t] is not None else "—" for t, _ in RUNS) + " |")
    L.append("| **mean** | " + " | ".join(fmt(kls[t].mean()) if kls[t] is not None else "—" for t, _ in RUNS) + " |")
    with open(f"{R}/SUMMARY.md", "w") as f:
        f.write("\n".join(L))
    print("\n".join(L))

    # per-position overlay (the decisive figure)
    fig, ax = plt.subplots(figsize=(9, 5))
    for tag, lab in RUNS:
        s = sd.get(tag)
        if s is None: continue
        ax.plot(range(len(s["pos_profile"])), s["pos_profile"], lw=1.6, label=lab)
    for x in (0, 64): ax.axvline(x, color="grey", ls=":", lw=0.9)
    ax.set_xlabel("grid position"); ax.set_ylabel("disp MAE [Mpc/h]")
    ax.set_title("Per-position error (x=64 = internal patch seam)"); ax.legend()
    fig.tight_layout(); fig.savefig(f"{R}/posprofile_all.png", dpi=140); plt.close(fig)

    # Pk ratio overlay
    fig, ax = plt.subplots(figsize=(8, 5))
    for tag, lab in RUNS:
        s = sd.get(tag)
        if s is None: continue
        m = s["pk_hr"].mean(0) > -10
        ax.semilogx(s["k"][m], (10 ** (s["pk_sr"] - s["pk_hr"])).mean(0)[m], lw=2, marker="o", ms=3, label=lab)
    ax.axhline(1.0, color="k", ls=":"); ax.set_xlabel("k [h/Mpc]"); ax.set_ylabel("P_SR/P_HR")
    ax.set_title("Stitched Pk ratio vs HR (128³)"); ax.legend()
    fig.tight_layout(); fig.savefig(f"{R}/pkratio_all.png", dpi=140); plt.close(fig)
    print(f"\nwrote {R}/SUMMARY.md, posprofile_all.png, pkratio_all.png")


if __name__ == "__main__":
    main()
