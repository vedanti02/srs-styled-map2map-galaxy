"""SR vs HR power-spectrum agreement on the test split, built from saved per-box Pk.

The point of the experiment: from a cheap LR field the model produces SR that should
match the expensive HR field, so HR is never needed at inference. This figure shows how
well SR recovers HR's power spectrum, with LR (no model) as the control.

Inputs: runs/patch_cmass/pk_{hr,cmassA_naive,cmassB_overlap,LRbase}/pk_set{sid}_transformed.npz
Output: figures_cmass/sr_vs_hr_pk.png
"""
import glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "runs/patch_cmass"
SETS = {
    "HR": "pk_hr",
    "SR (Arm A)": "pk_cmassA_naive",
    "SR (Arm B)": "pk_cmassB_overlap",
    "LR (no model)": "pk_LRbase",
}


def load_set(subdir, sids):
    k0, rows = None, []
    for sid in sids:
        f = os.path.join(ROOT, subdir, f"pk_set{sid}_transformed.npz")
        if not os.path.exists(f):
            continue
        z = np.load(f)
        k0 = z["k"]
        rows.append(z["pk"])
    return k0, np.stack(rows)


def band(a):
    return (np.nanmedian(a, 0), np.nanpercentile(a, 16, 0), np.nanpercentile(a, 84, 0))


def main():
    sids = np.load(os.path.join(ROOT, "split_sids.npz"))["test_sids"]
    data = {name: load_set(sub, sids) for name, sub in SETS.items()}
    k = data["HR"][0]
    pk = {name: v[1] for name, v in data.items()}
    nboxes = pk["HR"].shape[0]

    # ratio of each field's Pk to HR, box-by-box (matched ICs -> per-box ratio is meaningful)
    ratio = {name: pk[name] / np.maximum(pk["HR"], 1e-30) for name in pk}

    colors = {"HR": "k", "SR (Arm A)": "C0", "SR (Arm B)": "C2", "LR (no model)": "C3"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    # --- left: P(k) overlay ---
    ax = axes[0]
    for name in ["HR", "SR (Arm A)", "LR (no model)"]:
        med, lo, hi = band(pk[name])
        ax.fill_between(k, lo, hi, color=colors[name], alpha=0.15)
        ax.plot(k, med, color=colors[name], lw=2,
                ls="--" if name == "HR" else "-", label=name)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$k\ [h/{\rm Mpc}]$"); ax.set_ylabel(r"$P(k)$")
    ax.set_title(f"Power spectrum, {nboxes} test boxes")
    ax.legend(); ax.grid(alpha=0.3, which="both")

    # --- right: transfer to HR ---
    ax = axes[1]
    for name in ["SR (Arm A)", "SR (Arm B)", "LR (no model)"]:
        med, lo, hi = band(ratio[name])
        ax.fill_between(k, lo, hi, color=colors[name], alpha=0.18)
        ax.plot(k, med, color=colors[name], lw=2, label=name)
    ax.axhline(1.0, color="k", lw=1.2, ls="--", label="perfect match to HR")
    ax.set_xscale("log")
    ax.set_ylim(0.6, 1.6)
    ax.set_xlabel(r"$k\ [h/{\rm Mpc}]$"); ax.set_ylabel(r"$P_X(k)\,/\,P_{\rm HR}(k)$")
    ax.set_title("Transfer to HR  (1.0 = recovers HR)")
    ax.legend(); ax.grid(alpha=0.3, which="both")

    plt.tight_layout()
    out = "figures_cmass/sr_vs_hr_pk.png"
    plt.savefig(out, dpi=120)
    print("saved", out)

    # print the numbers that go in the report
    def rms_log(name):
        return float(np.sqrt(np.nanmean((np.log10(np.maximum(pk[name], 1e-30))
                                         - np.log10(np.maximum(pk["HR"], 1e-30))) ** 2)))
    print(f"boxes={nboxes}")
    for name in ["SR (Arm A)", "SR (Arm B)", "LR (no model)"]:
        print(f"{name:16s} log10 Pk RMS vs HR = {rms_log(name):.4f} | "
              f"median transfer @ k<0.1: {np.nanmedian(ratio[name][:, k < 0.1]):.3f} | "
              f"@ k>0.3: {np.nanmedian(ratio[name][:, k > 0.3]):.3f}")


if __name__ == "__main__":
    main()
