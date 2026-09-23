"""Cross-fidelity KL to HR, summary (P(k) NDE) and field (3D CNN) level, all corrector arms.
Numbers are taken from the evaluation logs (Tier-2 = analysis/padding_crossfid.py ensemble
protocol; Tier-3 = eval_field_crossfid.py) and, for the GAN field-level rows, from the paper
table / field eval logs. Summary error bars = std over test boxes; field error bars = std over
(HR-reference x replicate) combos. Log axis; lower whiskers clipped at 0.9*mean.
  python -m flow_matching.plot_crossfid_arms
"""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

arms = ["LR (raw)", "GAN deployed", "GAN Arm R", "GAN uncond.", "FM gauss\n(dequant)", "FM gauss\n(no-dequant)", "FM lf_spectral\n(no-dequant)"]
cols = ["C1", "C0", "C0", "C0", "C2", "C2", "C2"]
S = [(0.0306, 0.0219), (0.1373, 0.0661), (0.1363, 0.0809), (0.0870, 0.0463), (0.1170, 0.1215), (0.0575, 0.0637), (0.1439, 0.1612)]
S_floor = 0.0278
F = [(0.4711, 0.7824), (0.0369, 0.0144), None, (0.0223, 0.0102), (0.0277, 0.0135), (0.0293, 0.0081), (0.0248, 0.0112)]
F_floor = 0.0193


def asym(m, e):
    return [[min(e, 0.9 * m)], [e]]


fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
for panel, vals, floor, title in ((ax[0], S, S_floor, "summary level: NDE on log P(k)"), (ax[1], F, F_floor, "field level: 3D CNN on the 128$^3$ box")):
    x = np.arange(len(arms))
    for i, v in enumerate(vals):
        if v is None:
            panel.text(i, floor * 1.05, "n/a", ha="center", va="bottom", fontsize=9, color="gray"); continue
        m, e = v
        panel.bar(i, m, color=cols[i], alpha=0.85, width=0.7)
        panel.errorbar(i, m, yerr=asym(m, e), fmt="none", ecolor="k", capsize=4, lw=1.2)
        panel.text(i, m * 1.08 + 0.0, f"{m:.3f}", ha="center", va="bottom", fontsize=9)
    panel.axhline(floor, ls="--", c="k", lw=1); panel.text(len(arms) - 0.5, floor * 1.04, "HR-to-HR floor", ha="right", va="bottom", fontsize=9)
    panel.set_yscale("log"); panel.set_xticks(x); panel.set_xticklabels(arms, fontsize=9)
    panel.set_ylabel(r"KL$\,(q_{\rm HR}\,\|\,q_X)$ on HR test boxes"); panel.set_title(title)
    panel.set_ylim(0.012, 1.5)
fig.suptitle("Cross-fidelity: posterior trained on source X, applied to HR test fields (lower is better)")
fig.tight_layout()
fig.savefig("figures_cmass/crossfid_all_arms.png", dpi=130)
print("saved figures_cmass/crossfid_all_arms.png")
