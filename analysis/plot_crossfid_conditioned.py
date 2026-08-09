"""Cross-fidelity bar chart for the RETRAINED correct-conditioning generator,
deployment (fiducial) conditioning. Same format as the corrected (scrambled-SR)
chart, so the two are directly comparable."""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

# corrected-label cross-fid, retrained conditioned SR (fiducial = deployment)
lr   =[0.0306, 0.471]; lr_e=[0.0, 0.782]          # summary, field  (field firmed 8LR/6HR)
sr   =[0.1373, 0.037]; sr_e=[0.0, 0.014]          # conditioned SR fiducial
floor=[0.0278, 0.0193]
groups=["summary $P(k)$","field (CNN)"]; x=np.arange(2); w=0.36
def asym(v,e): return [[min(a,b*0.9) for b,a in zip(v,e)], e]

fig,ax=plt.subplots(figsize=(7.2,4.6))
b1=ax.bar(x-w/2, lr, w, yerr=asym(lr,lr_e), capsize=4, color="C1", label="LR (raw)")
b2=ax.bar(x+w/2, sr, w, yerr=asym(sr,sr_e), capsize=4, color="C0", label="SR (conditioned)")
for xi,fl in zip(x,floor): ax.hlines(fl, xi-0.5, xi+0.5, color="k", ls="--", lw=1.2)
ax.text(1.5, 0.021, "HR floor", fontsize=8, va="bottom", ha="right")
for b,v in zip(b1,lr): ax.text(b.get_x()+b.get_width()/2, v*1.12, f"{v:.3f}", ha="center", fontsize=8)
for b,v in zip(b2,sr): ax.text(b.get_x()+b.get_width()/2, v*1.12, f"{v:.3f}", ha="center", fontsize=8)
ax.set_yscale("log"); ax.set_ylabel("cross-fidelity KL to HR (lower better)")
ax.set_xticks(x); ax.set_xticklabels(groups); ax.legend(frameon=False, loc="upper left")
ax.set_title("Conditioned SR (deployment): helps at field level, worse at summary")
plt.tight_layout(); plt.savefig("figures_cmass/crossfid_conditioned_cmass.png", dpi=140)
print("saved figures_cmass/crossfid_conditioned_cmass.png")
