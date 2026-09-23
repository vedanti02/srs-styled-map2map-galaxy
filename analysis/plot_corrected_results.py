"""Corrected-results figures for section 6 (post label-bug fix).
 1. crossfid_corrected_cmass.png : two-level cross-fidelity KL (summary vs field,
    LR vs SR) with the HR floor. The headline corrected result.
 2. labelbug_count_vs_s8.png     : halo count vs S8, as-used (scrambled, corr~0)
    vs fixed (corr~0.93). Proves the indexing bug and its fix.
 3. fisher_corrected_cmass.png   : Fisher single-box P(k) posterior width / prior
    width per parameter (Om and s8 constrained, Ob/h/ns weak).
"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h5py, os, time

from paths import NBODY as NB
theta=np.load("data/cmass_theta_numeric_buggy.npz")["theta"]   # NUMERIC order = cosmo[sim s]
NAMES=["$\\Omega_m$","$\\Omega_b$","$h$","$n_s$","$\\sigma_8$"]

# ---------- figure 1: two-level cross-fidelity (corrected numbers) ----------
# summary P(k): LR 0.032, SR 0.088, floor 0.028 ; field: LR 0.49+-0.81, SR 0.024, floor 0.022
fig,ax=plt.subplots(figsize=(7.2,4.6))
groups=["summary $P(k)$","field (CNN)"]
lr=[0.032,0.488]; lr_e=[0.0,0.808]; sr=[0.088,0.0243]; sr_e=[0.0,0.0114]; floor=[0.028,0.022]
x=np.arange(2); w=0.36
# asymmetric whiskers clipped so the lower end stays positive on the log axis
# (LR field std 0.81 > mean 0.49 would otherwise draw a spike below zero)
def asym(vals,errs): return [[min(e,v*0.9) for v,e in zip(vals,errs)], errs]
b1=ax.bar(x-w/2, lr, w, yerr=asym(lr,lr_e), capsize=4, color="C1", label="LR (raw)")
b2=ax.bar(x+w/2, sr, w, yerr=asym(sr,sr_e), capsize=4, color="C0", label="SR")
for xi,fl in zip(x,floor):
    ax.hlines(fl, xi-0.5, xi+0.5, color="k", ls="--", lw=1.2)
ax.text(1.5, 0.026, "HR floor", fontsize=8, va="bottom", ha="right", color="k")
ax.set_yscale("log"); ax.set_ylabel("cross-fidelity KL to HR (lower better)")
ax.set_xticks(x); ax.set_xticklabels(groups)
for b,v in zip(b1,lr): ax.text(b.get_x()+b.get_width()/2, v*1.12, f"{v:.3f}", ha="center", fontsize=8)
for b,v in zip(b2,sr): ax.text(b.get_x()+b.get_width()/2, v*1.12, f"{v:.3f}", ha="center", fontsize=8)
ax.legend(frameon=False, loc="upper left")
ax.set_title("SR helps at the field level, not at the summary level")
plt.tight_layout(); plt.savefig("figures_cmass/crossfid_corrected_cmass.png", dpi=140); plt.close()
print("wrote crossfid_corrected_cmass.png")

# ---------- catalog counts (fast h5 metadata) ----------
M=600; t0=time.time()
sims=sorted(range(2000),key=str)[:M]        # true sim id of field index k=0..M-1
ycount=np.zeros(M)                          # ycount[k] = halo count of field k
for k,s in enumerate(sims):
    with h5py.File(f"{NB}/{s}/halos.h5","r") as f:
        g=list(f.keys())[0]; ycount[k]=f[g]["mass"].shape[0]
print(f"loaded {M} catalog counts {time.time()-t0:.0f}s")
S8=lambda th: th[:,4]*np.sqrt(th[:,0]/0.3)
sims=np.array(sims)
# as-used (buggy): field k count ycount[k], labeled theta[k] (numeric)
xb=S8(theta[:M]); yb=ycount
# fixed: field k count ycount[k], correct label theta[sims[k]]
xf=S8(theta[sims]); yf=ycount
cb=np.corrcoef(xb,yb)[0,1]; cf=np.corrcoef(xf,yf)[0,1]

fig,(a1,a2)=plt.subplots(1,2,figsize=(9,4.2),sharey=True)
a1.scatter(xb,yb,s=6,alpha=0.4,color="C3"); a1.set_title(f"as used (mislabeled): r={cb:+.2f}")
a2.scatter(xf,yf,s=6,alpha=0.4,color="C0"); a2.set_title(f"fixed (string-sort): r={cf:+.2f}")
for a in (a1,a2): a.set_xlabel("$S_8=\\sigma_8\\sqrt{\\Omega_m/0.3}$")
a1.set_ylabel("total halo count in box")
fig.suptitle("Halo count vs cosmology: the indexing bug destroyed a real 0.93 correlation")
plt.tight_layout(); plt.savefig("figures_cmass/labelbug_count_vs_s8.png", dpi=140); plt.close()
print(f"wrote labelbug_count_vs_s8.png (buggy r={cb:+.2f}, fixed r={cf:+.2f})")

# ---------- figure 3: Fisher per-param sigma/prior ----------
# linear (avg-slope) and quad (centre) from analysis/fisher_pk_cmass.py (fixed labels)
lin=[0.288,0.835,0.787,0.604,0.566]; quad=[0.210,0.768,0.746,0.542,0.341]
fig,ax=plt.subplots(figsize=(7,4.2))
xx=np.arange(5)
ax.bar(xx-0.2,lin,0.4,color="C0",label="linear emulator")
ax.bar(xx+0.2,quad,0.4,color="C2",label="quadratic emulator")
ax.axhline(1.0,color="k",ls="--",lw=1); ax.text(4.45,1.02,"prior (no info)",fontsize=8,va="bottom",ha="right")
ax.axhline(0.5,color="0.6",ls=":",lw=1); ax.text(0,0.44,"constrained",fontsize=7.5,color="0.4")
ax.set_xticks(xx); ax.set_xticklabels(NAMES); ax.set_ylabel("posterior width / prior width")
ax.set_ylim(0,1.18)
ax.set_title("Single-box $P(k)$ Fisher forecast (corrected labels)")
ax.legend(frameon=True, framealpha=1, loc="upper left")
plt.tight_layout(); plt.savefig("figures_cmass/fisher_corrected_cmass.png", dpi=140); plt.close()
print("wrote fisher_corrected_cmass.png")
print("DONE")
