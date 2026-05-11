"""Final comparison plots: KL/bias bar charts + Pk ratio panels."""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PARAM_NAMES = [r"$\Omega_m$", r"$\Omega_b$", r"$h$", r"$n_s$", r"$\sigma_8$"]


def plot_kl(metrics, outpath):
    versions = list(metrics.keys())
    n_params = 5
    kl_arr = np.zeros((len(versions), n_params))
    for i, k in enumerate(versions):
        kl_arr[i] = metrics[k]["kl_hr_to_sr"].mean(0)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(n_params)
    w = 0.16
    colors = ["#999999", "#cc8855", "#88aacc", "#5588aa", "#117755"]
    for i, ver in enumerate(versions):
        ax.bar(x + (i - len(versions) / 2 + 0.5) * w, kl_arr[i], w,
               label=ver, color=colors[i % len(colors)])
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(PARAM_NAMES)
    ax.set_ylabel(r"$\mathrm{KL}(q_{HR}\,\Vert\,q_{SR})$ — lower is better")
    ax.set_title("Posterior agreement with HR (per parameter)")
    ax.legend(fontsize=9, ncol=2)
    ax.grid(alpha=0.3, axis="y", which="both")
    plt.tight_layout()
    plt.savefig(outpath, dpi=120)
    print(f"saved {outpath}")
    plt.close(fig)


def plot_bias(metrics, outpath):
    versions = list(metrics.keys())
    n_params = 5
    bias_arr = np.zeros((len(versions) + 1, n_params))
    # First row = HR bias (constant baseline)
    bias_arr[0] = np.abs(metrics[versions[0]]["mu_hr"] - metrics[versions[0]]["theta_true"]).mean(0)
    for i, k in enumerate(versions):
        bias_arr[i + 1] = np.abs(metrics[k]["mu_sr"] - metrics[k]["theta_true"]).mean(0)

    labels = ["HR (ref)"] + versions
    colors = ["k", "#999999", "#cc8855", "#88aacc", "#5588aa", "#117755"]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(n_params)
    w = 0.13
    for i, lab in enumerate(labels):
        ax.bar(x + (i - len(labels) / 2 + 0.5) * w, bias_arr[i], w,
               label=lab, color=colors[i % len(colors)])
    ax.set_xticks(x); ax.set_xticklabels(PARAM_NAMES)
    ax.set_ylabel(r"$|\mu_{SR}-\theta_{\rm true}|$ — lower is better")
    ax.set_title("Posterior bias (averaged over 200 held-out sims)")
    ax.legend(fontsize=9, ncol=2)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(outpath, dpi=120)
    print(f"saved {outpath}")
    plt.close(fig)


def _load_pk_dir(d, prefix):
    import glob, re
    files = sorted(glob.glob(os.path.join(d, f"pk_{prefix}*.npz")))
    out = {}
    for f in files:
        m = re.search(r"set(\d+)", os.path.basename(f))
        if m is None: continue
        sid = int(m.group(1))
        z = np.load(f)
        out[sid] = (z["k"], z["pk"])
    return out


def plot_pk_ratio(pk_versions, pk_hr, outpath):
    """Plot mean Pk_X / Pk_HR per k bin."""
    fig, ax = plt.subplots(figsize=(10, 5))
    common = sorted(set.intersection(*[set(d) for d in pk_versions.values()]) & set(pk_hr))
    hr_pk = np.stack([pk_hr[s][1] for s in common])
    hr_k = pk_hr[common[0]][0]
    valid = (hr_pk.mean(0) > 0)

    colors = {"LR": "#cc8855", "v1_e31": "#cc66cc", "v2_e40": "#117755", "v4_e56": "#5588aa"}
    for ver, pk_dict in pk_versions.items():
        sr_pk = np.stack([pk_dict[s][1] for s in common])
        ratio = (sr_pk[:, valid] / hr_pk[:, valid]).mean(0)
        k = hr_k[valid]
        ax.plot(k, ratio, label=ver, color=colors.get(ver, "C0"), lw=2)
    ax.axhline(1.0, color="k", lw=0.8, ls="--", alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel(r"$k$ [h/Mpc]")
    ax.set_ylabel(r"$\langle P(k)_{\rm SR} / P(k)_{\rm HR}\rangle$")
    ax.set_title("Mean per-bin Pk ratio across 200 sims")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.set_ylim(0.7, 1.5)
    plt.tight_layout()
    plt.savefig(outpath, dpi=120)
    print(f"saved {outpath}")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="runs/baseline/plots")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    metric_files = {
        "LR":     "runs/baseline/metrics_hr_vs_lr.npz",
        "v1_e31": "runs/baseline/metrics_v1_e31.npz",
        "v2_e10": "runs/baseline/metrics_v2_e10.npz",
        "v2_e40": "runs/baseline/metrics_v2_best.npz",
        "v4_e56": "runs/baseline/metrics_v4_best.npz",
    }
    metrics = {k: np.load(v) for k, v in metric_files.items() if os.path.exists(v)}
    plot_kl(metrics, os.path.join(args.out_dir, "kl_comparison.png"))
    plot_bias(metrics, os.path.join(args.out_dir, "bias_comparison.png"))

    # Pk ratio panel
    pk_hr = _load_pk_dir("runs/baseline/pk/hr", "quijote_")
    pk_versions = {}
    pk_versions["LR"] = _load_pk_dir("runs/baseline/pk/lr", "quijotelike_")
    pk_versions["v1_e31"] = _load_pk_dir("runs/baseline/pk/sr_e31_v1full", "set")
    pk_versions["v2_e40"] = _load_pk_dir("runs/baseline/pk/sr_v2_best", "set")
    pk_versions["v4_e56"] = _load_pk_dir("runs/baseline/pk/sr_v4_best", "set")
    plot_pk_ratio(pk_versions, pk_hr, os.path.join(args.out_dir, "pk_ratio.png"))


if __name__ == "__main__":
    main()
