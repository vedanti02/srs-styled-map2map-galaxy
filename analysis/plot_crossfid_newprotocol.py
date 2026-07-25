"""Cross-fidelity KL (q_SR trained on SR, tested on HR) for the new-protocol
(no-Pk) models, vs the old-protocol F6 references. The research goal metric.
Output: figures_cmass/crossfid_newprotocol_cmass.png
"""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
R="runs/patch_cmass"

# new-protocol arms (goal metric)
new = {
  "A_nopk\n(no pad)":     "metrics_crossfid_Anopk.npz",
  "A_lowlr\n(no pad, lr/2)":"metrics_crossfid_Alowlr.npz",
  "B_aware\n(pad, aware)":"metrics_crossfid_Baware.npz",
  "B_unaware\n(pad, unaware)":"metrics_crossfid_Bunaware.npz",
}
means = {k: float(np.load(f"{R}/{v}")["kl_hr_to_sr"].mean()) for k,v in new.items()}
# old-protocol + controls (F6)
ref = {"old SR-A":0.0029, "old SR-B":0.0033, "LR control":0.0047, "HR floor":0.0007}

fig, ax = plt.subplots(figsize=(9,5.2))
labels = list(means) + list(ref)
vals   = list(means.values()) + list(ref.values())
colors = ["C0","C0","C2","C3","0.5","0.5","C1","k"]
bars = ax.bar(range(len(vals)), vals, color=colors)
ax.set_yscale("log")
ax.set_ylabel("cross-fidelity KL (q_SR on HR)  [log]")
ax.set_title("Goal metric: cosmology recovery when trained on SR, tested on HR\n(new-protocol models vs old-protocol/controls; lower is better)")
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=8)
ax.axhline(0.0029, color="0.5", ls="--", lw=1)  # old SR-A reference line
for b,v in zip(bars, vals):
    ax.text(b.get_x()+b.get_width()/2, v*1.08, f"{v:.4f}", ha="center", fontsize=8)
ax.annotate("padding HURTS the\ngoal (0.0034 > 0.0026)\ndespite better L1",
            xy=(2,0.0034), xytext=(1.4,0.012), fontsize=7.5,
            arrowprops=dict(arrowstyle="->",color="C2"))
ax.annotate("trim-unaware:\ncatastrophic",
            xy=(3,1.05), xytext=(2.5,0.2), fontsize=7.5,
            arrowprops=dict(arrowstyle="->",color="C3"))
plt.tight_layout(); plt.savefig("figures_cmass/crossfid_newprotocol_cmass.png", dpi=130)
print("saved figures_cmass/crossfid_newprotocol_cmass.png")
for k,v in means.items(): print(f"  {k.split(chr(10))[0]:10s} {v:.4f}")
