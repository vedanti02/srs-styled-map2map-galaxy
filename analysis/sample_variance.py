"""Generator SAMPLE variance: draw M stochastic noise realizations per box (fiducial theta),
and measure how much the SR field varies sample-to-sample. Outputs:
  1) power transfer function T(k)=P_SR/P_HF with two bands: the GENERATION band (scatter across
     the M noise draws of a box) and the BOX-TO-BOX band (scatter of the per-box mean across boxes),
  2) PQMass chi2/dof vs HF for each of the M samples (mean +/- std), with LF as the control,
  3) a headline generation-variance number: per-voxel std across samples / mean count.

Motivation: the learned noise std is ~0 (verified), so we expect the generation band to be tiny
(under-dispersion), which is what makes SR distinguishable under PQMass. This quantifies it and
tells us whether averaging multiple samples could help inference (only if the variance is large)."""
import numpy as np, os, sys, time, torch
sys.path.insert(0, ".")
from data.patch_dataset_cmass import (PatchPairDatasetCmass, extract_patch, stitch_patches,
    crop_interior, to_counts, PATCH, N_PATCHES)
from map2map.models.styled_srsgan import G_correct
from analysis.power_spectrum import cube_pk_counts
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

CKPT=sys.argv[1]; TAG=sys.argv[2] if len(sys.argv)>2 else "Afix_fiducial"
PROC="/data/group_data/universedata/cmass-ili/processed"
NBOX=60; M=12; L=1000.0
FIDUCIAL=np.array([0.30,0.05,0.70,1.00,0.80],np.float32)
CBINS=np.array([-0.5,0.5,1.5,2.5,3.5,4.5,5.5,7.5,11.5,20.5,np.inf])
def feat(n):
    n=np.asarray(n,np.float32).ravel(); h,_=np.histogram(n,bins=CBINS)
    return np.concatenate([h/h.sum(),[np.log1p(n.mean())]]).astype(np.float32)

dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
ck=torch.load(CKPT,map_location=dev,weights_only=False); s=ck.get("args",{}) or {}
cb,nb=s.get("chan_base_g",128),s.get("num_blocks",4); tf=s.get("transform","log1p")
G=G_correct(1,1,5,chan_base=cb,num_blocks=nb).to(dev); G.load_state_dict(ck["model"]); G.eval()
ds=PatchPairDatasetCmass(split="val",pad=0,transform=tf)
ids=[i for i in list(ds.ids) if os.path.exists(f"{PROC}/{i:04d}_label.npy")][:NBOX]
th=torch.from_numpy(FIDUCIAL).unsqueeze(0).expand(N_PATCHES,-1).to(dev)
print(f"sample-variance: {len(ids)} boxes x {M} noise draws | ckpt={CKPT}",flush=True)

def gen(lr, seed):
    rng=np.random.default_rng(seed)
    nl=[torch.from_numpy(rng.standard_normal((1,PATCH,PATCH,PATCH)).astype(np.float32)).to(dev) for _ in range(8)]
    xb=torch.from_numpy(np.stack([extract_patch(lr,p,0) for p in range(N_PATCHES)])).to(dev)
    with torch.no_grad():
        fake=crop_interior(G(xb,th,nl),0).cpu().numpy()
    sr=to_counts(stitch_patches(fake),tf,ds.scale)[0]
    return np.clip(np.nan_to_num(sr,nan=0.),0,200).round().astype(np.float32)

kk=None; Phf=[]; Fhf=[]; Plf=[]; Flf=[]; Psr=[]; Fsr=[]; relstd=[]
t0=time.time()
for bi,idx in enumerate(ids):
    lr,_=ds.load_boxes(idx)
    hf=np.load(f"{PROC}/{idx:04d}_label.npy").astype(np.float32)
    lf=np.load(f"{PROC}/{idx:04d}_input.npy").astype(np.float32)
    k,phf,_=cube_pk_counts(hf,L,n_bins=32); kk=k; Phf.append(phf); Fhf.append(feat(hf))
    k,plf,_=cube_pk_counts(lf,L,n_bins=32); Plf.append(plf); Flf.append(feat(lf))
    ps=[]; fs=[]; ssum=np.zeros_like(hf); ssq=np.zeros_like(hf)
    for m in range(M):
        sr=gen(lr, seed=1000*bi+m)
        k,p,_=cube_pk_counts(sr,L,n_bins=32); ps.append(p); fs.append(feat(sr))
        ssum+=sr; ssq+=sr*sr
    Psr.append(np.stack(ps)); Fsr.append(np.stack(fs))
    mean=ssum/M; var=np.maximum(ssq/M-mean*mean,0.0)
    relstd.append(float(np.sqrt(var).mean()/(mean.mean()+1e-8)))
    if (bi+1)%10==0: print(f"  {bi+1}/{len(ids)} boxes  {time.time()-t0:.0f}s",flush=True)

Phf=np.stack(Phf); Plf=np.stack(Plf)                     # (B,nk)
Psr=np.stack(Psr); Fsr=np.stack(Fsr)                     # (B,M,nk) , (B,M,11)
Fhf=np.stack(Fhf); Flf=np.stack(Flf)                     # (B,11)
good=(Phf>0).all(0); kk=kk[good]; Phf=Phf[:,good]; Plf=Plf[:,good]; Psr=Psr[:,:,good]

# ---- transfer function + bands ----
Tsr=Psr/Phf[:,None,:]                                    # (B,M,nk)
gen_mean=Tsr.mean(1)                                     # (B,nk) per-box mean over samples
T_mean=gen_mean.mean(0)                                  # (nk)
gen_band=Tsr.std(1).mean(0)                              # typical generation scatter (over samples)
box_band=gen_mean.std(0)                                 # box-to-box scatter of the mean
Tlf=(Plf/Phf).mean(0)
rel=float(np.mean(relstd))
print(f"\nGENERATION variance: per-voxel std/mean across {M} samples = {rel*100:.2f}%  (mean over boxes)",flush=True)
print(f"transfer-fn band widths (mean over k): generation={np.mean(gen_band):.4f}  box-to-box={np.mean(box_band):.4f}",flush=True)

# ---- PQMass per sample ----
NREG,NTESS,dof=10,200,9; rng=np.random.default_rng(0)
mu,sd=Fhf.mean(0),Fhf.std(0)+1e-8
def pqmass(A,B):
    N1,N2=len(A),len(B); pool=np.concatenate([A,B],0); chis=[]
    for _ in range(NTESS):
        ref=pool[rng.choice(len(pool),NREG,replace=False)]
        dA=((A[:,None,:]-ref[None])**2).sum(-1).argmin(1); dB=((B[:,None,:]-ref[None])**2).sum(-1).argmin(1)
        k1=np.bincount(dA,minlength=NREG).astype(float); k2=np.bincount(dB,minlength=NREG).astype(float)
        keep=(k1+k2)>0
        chis.append((((np.sqrt(N2/N1)*k1-np.sqrt(N1/N2)*k2)**2)[keep]/(k1+k2)[keep]).sum())
    return np.median(chis)/dof
Zhf=(Fhf-mu)/sd
pq_sr=np.array([pqmass((Fsr[:,m,:]-mu)/sd, Zhf) for m in range(M)])
pq_lf=pqmass((Flf-mu)/sd, Zhf)
# HF self-floor: random halves, few splits
fl=[]
for _ in range(6):
    p=rng.permutation(len(Zhf)); h=len(p)//2
    fl.append(pqmass(Zhf[p[:h]],Zhf[p[h:]]))
pq_floor=np.mean(fl)
print(f"\nPQMass chi2/dof vs HF:  SR = {pq_sr.mean():.2f} +/- {pq_sr.std():.2f} (over {M} samples)   "
      f"LF = {pq_lf:.2f}   HF-self floor ~ {pq_floor:.2f}",flush=True)

# ---- plot ----
fig,ax=plt.subplots(1,2,figsize=(12,4.6))
ax[0].fill_between(kk,T_mean-box_band,T_mean+box_band,color="C0",alpha=0.15,label="box-to-box band")
ax[0].fill_between(kk,T_mean-gen_band,T_mean+gen_band,color="C0",alpha=0.55,label="generation band (M draws)")
ax[0].plot(kk,T_mean,"C0-",lw=1.6,label="SR mean")
ax[0].plot(kk,Tlf,"C3--",lw=1.4,label="LF")
ax[0].axhline(1,color="k",ls=":",lw=1)
ax[0].set_xscale("log"); ax[0].set_xlabel("k [h/Mpc]"); ax[0].set_ylabel("T(k)=P/P_HF")
ax[0].set_title(f"power transfer function ({rel*100:.2f}% generation variance)"); ax[0].legend(fontsize=8)
bars=["SR","LF","HF floor"]; vals=[pq_sr.mean(),pq_lf,pq_floor]; errs=[pq_sr.std(),0,0]
ax[1].bar(range(3),vals,yerr=errs,capsize=5,color=["C0","C3","0.6"])
ax[1].axhline(1,color="k",ls="--",lw=1,label="matches HF")
ax[1].set_xticks(range(3)); ax[1].set_xticklabels(bars); ax[1].set_ylabel("PQMass chi2/dof")
ax[1].set_title("distributional match (SR errbar = across samples)"); ax[1].legend(fontsize=8)
plt.tight_layout(); out=f"figures_cmass/sample_variance_{TAG}.png"; plt.savefig(out,dpi=140)
print(f"\nsaved {out}\nSAMPLEVAR_DONE",flush=True)
