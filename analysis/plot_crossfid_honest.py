"""HONEST cross-fidelity bar chart: real seed/replicate error bars, broken NDE seed excluded.
Replaces plot_crossfid_conditioned.py, whose tiny error bars understated the true uncertainty.

SUMMARY (P(k)) level: per-seed KL from analysis/seed_xfid_check.py (job 10361507). NDE seed 4
produced a broken q_HR (HR self-floor exploded to ~0.53 for every pair involving it), so it is
EXCLUDED. Each bar = mean +/- std over the 4 healthy seeds; dots = the individual seeds.
FIELD (CNN) level: mean +/- std over (HR-ref x member) replicate combos, from the field eval
logs (9831621 fiducial, 9754743 scrambled). LR's field std exceeds its mean (bimodal, D-F15)
so its lower whisker is clipped on the log axis."""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

# ---- SUMMARY: raw per-seed values (seed 4 = broken, dropped) ----
S = {"LR":            [0.048,0.045,0.049,0.055],
     "SR scrambled":  [0.149,0.161,0.101,0.071],
     "SR conditioned":[0.157,0.129,0.133,0.220]}
S_floor_pairs = [0.031,0.034,0.022,0.034,0.033,0.025]     # HR-vs-HR pairs not involving seed 4
# ---- FIELD: mean, std over replicate combos (from logs) ----
F = {"LR":            (0.4711,0.7824),
     "SR scrambled":  (0.0223,0.0102),
     "SR conditioned":(0.0369,0.0144)}
F_floor = 0.0193

arms=["LR","SR scrambled","SR conditioned"]; cols=["C1","C2","C0"]
def asym(m,e): return [[min(e,0.9*m)],[e]]    # clip lower whisker for log axis

fig,ax=plt.subplots(1,2,figsize=(12.5,5),sharey=False)
# --- summary panel ---
x=np.arange(3); rng=np.random.default_rng(0)
for i,a in enumerate(arms):
    v=np.array(S[a]); m,e=v.mean(),v.std()
    ax[0].bar(i,m,0.6,yerr=asym(m,e),capsize=5,color=cols[i],alpha=0.85)
    ax[0].scatter(i+rng.uniform(-0.15,0.15,len(v)),v,color="k",s=22,zorder=5)
    ax[0].text(i,m*1.55,f"{m:.3f}\n±{e:.3f}",ha="center",fontsize=9)
fl=np.mean(S_floor_pairs)
ax[0].axhline(fl,color="k",ls="--",lw=1.2); ax[0].text(2.45,fl*1.08,"HR floor",fontsize=8,ha="right")
ax[0].set_yscale("log"); ax[0].set_ylim(0.015,0.6)
ax[0].set_xticks(x); ax[0].set_xticklabels(arms,fontsize=9)
ax[0].set_ylabel("cross-fidelity KL to HR (lower better)")
ax[0].set_title("Summary $P(k)$ level\n(bars = mean ± std over 4 healthy NDE seeds; dots = seeds)",fontsize=10)
# --- field panel ---
for i,a in enumerate(arms):
    m,e=F[a]
    ax[1].bar(i,m,0.6,yerr=asym(m,e),capsize=5,color=cols[i],alpha=0.85)
    ax[1].text(i,m*1.55,f"{m:.3f}\n±{e:.3f}",ha="center",fontsize=9)
ax[1].axhline(F_floor,color="k",ls="--",lw=1.2); ax[1].text(2.45,F_floor*1.08,"HR floor",fontsize=8,ha="right")
ax[1].set_yscale("log"); ax[1].set_ylim(0.01,2.0)
ax[1].set_xticks(x); ax[1].set_xticklabels(arms,fontsize=9)
ax[1].set_title("Field (3D CNN) level\n(bars = mean ± std over HR-ref × replicate combos)",fontsize=10)
fig.suptitle("Cross-fidelity with honest uncertainty (broken NDE seed 4 excluded)",fontsize=12)
plt.tight_layout(); out="figures_cmass/crossfid_honest_cmass.png"; plt.savefig(out,dpi=140)
print("saved",out)
for a in arms:
    v=np.array(S[a]); print(f"  summary {a:15s}: {v.mean():.3f} +/- {v.std():.3f}   | field {a:15s}: {F[a][0]:.3f} +/- {F[a][1]:.3f}")
print(f"  summary floor {fl:.3f}  | field floor {F_floor:.3f}")
