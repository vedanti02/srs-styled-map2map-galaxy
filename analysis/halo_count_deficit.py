"""Halo-count deficit: is LR systematically short of halos vs HR, by how much, does the
deficit depend on cosmology, and how much of the missing count does SR restore?
Usage: python analysis/halo_count_deficit.py [SRDIR] [TAG]
  SRDIR: dir of sr_{idx}.npy count fields (default: fixcond deployment SR)
  TAG:   output suffix (default Afix_fiducial) -> runs/patch_cmass/halo_count_deficit_{TAG}.npz,
         figures_cmass/halo_count_deficit_{TAG}.png
The LR/HR totals over the 500-box subsample (every 4th box) are arm-independent, so they are
computed once and CACHED in runs/patch_cmass/halo_count_lrhr.npz and reused (the full 2000-box
read timed out at 30 min on the universedata NFS, job 10364325; even 500 takes ~14 min)."""
import numpy as np, os, sys, time
sys.path.insert(0,".")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
PROC="/data/group_data/universedata/cmass-ili/processed"
SRDIR=sys.argv[1] if len(sys.argv)>1 else "/data/user_data/vkshirsa/cmass-ili/sr_fields/Afix_fiducial"
TAG=sys.argv[2] if len(sys.argv)>2 else "Afix_fiducial"
theta=np.load("data/cmass_theta.npz")["theta"]
SUB=list(range(0,2000,4)); NSR_MAX=400; t0=time.time()
os.makedirs("runs/patch_cmass",exist_ok=True); os.makedirs("figures_cmass",exist_ok=True)
CACHE="runs/patch_cmass/halo_count_lrhr.npz"
if os.path.exists(CACHE):
    c=np.load(CACHE); n_lr,n_hr=c["n_lr"],c["n_hr"]; print(f"LR/HR totals loaded from cache ({len(SUB)} boxes)",flush=True)
else:
    n_lr=np.zeros(len(SUB)); n_hr=np.zeros(len(SUB))
    for j,i in enumerate(SUB):
        n_lr[j]=np.load(f"{PROC}/{i:04d}_input.npy").sum(); n_hr[j]=np.load(f"{PROC}/{i:04d}_label.npy").sum()
        if (j+1)%100==0: print(f"  {j+1}/{len(SUB)} boxes {time.time()-t0:.0f}s",flush=True)
    np.savez(CACHE,sub=np.array(SUB),n_lr=n_lr,n_hr=n_hr); print("LR/HR totals cached",flush=True)
d=(n_lr-n_hr)/n_hr; th=theta[SUB]
print(f"\nLR vs HR total halos, {len(SUB)} boxes (every 4th of 2000):")
print(f"  mean HR total {n_hr.mean():.0f}   mean LR total {n_lr.mean():.0f}")
print(f"  LR fractional deficit (LR-HR)/HR: mean {d.mean()*100:+.2f}%  std {d.std()*100:.2f}%  median {np.median(d)*100:+.2f}%")
print(f"  boxes with LR < HR: {np.mean(n_lr<n_hr)*100:.1f}%")
for k,nm in [(0,"Om"),(4,"s8")]:
    print(f"  corr(LR deficit, {nm:2s}) = {np.corrcoef(d,th[:,k])[0,1]:+.3f}")
pos={i:j for j,i in enumerate(SUB)}
ids=[i for i in SUB if os.path.exists(f"{SRDIR}/sr_{i}.npy")][:NSR_MAX]
if not ids: print(f"NO SR files found in {SRDIR} for the subsample; SR stats skipped"); sys.exit(0)
n_sr=np.array([np.load(f"{SRDIR}/sr_{i}.npy").astype(np.float64).sum() for i in ids])
jj=[pos[i] for i in ids]; hr_s,lr_s=n_hr[jj],n_lr[jj]
dsr=(n_sr-hr_s)/hr_s; gap=hr_s-lr_s; rec=(n_sr-lr_s)/np.where(gap!=0,gap,np.nan)
print(f"\nSR [{TAG}] vs HR, {len(ids)} boxes with SR fields:")
print(f"  SR fractional deficit (SR-HR)/HR: mean {dsr.mean()*100:+.2f}%  std {dsr.std()*100:.2f}%  median {np.median(dsr)*100:+.2f}%")
print(f"  LR deficit on the SAME boxes:      mean {((lr_s-hr_s)/hr_s).mean()*100:+.2f}%")
print(f"  share of missing halos SR restores, (SR-LR)/(HR-LR): median {np.nanmedian(rec)*100:.1f}%  mean {np.nanmean(rec)*100:.1f}%")
np.savez(f"runs/patch_cmass/halo_count_deficit_{TAG}.npz",sub=np.array(SUB),n_lr=n_lr,n_hr=n_hr,ids=np.array(ids),n_sr=n_sr)
fig,ax=plt.subplots(figsize=(7,4.4))
ax.hist(d*100,bins=40,alpha=0.55,color="C1",label=f"LR vs HR: mean {d.mean()*100:+.2f}%  ({len(SUB)} boxes)")
ax.hist(dsr*100,bins=40,alpha=0.55,color="C0",label=f"SR [{TAG}] vs HR: mean {dsr.mean()*100:+.2f}%  ({len(ids)} boxes)")
ax.axvline(0,color="k",ls="--",lw=1); ax.set_xlabel("total halos relative to HR: (X - HR)/HR  [%]"); ax.set_ylabel("boxes")
ax.set_title(f"Per-box halo-count deficit ({TAG})"); ax.legend(fontsize=9)
plt.tight_layout(); plt.savefig(f"figures_cmass/halo_count_deficit_{TAG}.png",dpi=140)
print(f"saved figures_cmass/halo_count_deficit_{TAG}.png\nHALODEF_DONE",flush=True)
