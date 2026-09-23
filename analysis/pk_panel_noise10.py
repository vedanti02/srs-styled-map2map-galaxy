"""P(k) panel for ONE LR box with 10 SR realisations, each from a DIFFERENT noise vector, for the
fixcond baseline and the Stage-0 Arm R model. Reports each checkpoint's learned noise amplitude and
the spread across draws in the field, transfer and coherence. Fiducial theta, pad 0, same estimator
as diag_cmass (full box, delta = n/nbar - 1, n_bins 40)."""
import sys, numpy as np, torch
sys.path.insert(0,".")
from map2map.models.styled_srsgan import G_correct
from data.patch_dataset_cmass import (PatchPairDatasetCmass, extract_patch, stitch_patches, crop_interior,
    to_counts, counts_to_delta, PATCH, N_PATCHES, N_FULL)
from analysis.plot_diagnostic import cross_pk
from analysis.diag_cmass import build_noise_list
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
M=10; NBINS=40; LBOX=1000.0
FID=np.array([0.30,0.05,0.70,1.00,0.80],np.float32)
from paths import MODELS as MODELS_DIR
MODELS={"fixcond (baseline)":f"{MODELS_DIR}/patch_cmass_A_nopk_fixcond/best.pt",
        "Arm R (stage0_R)":f"{MODELS_DIR}/stage0_R/best.pt"}
dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
ds=PatchPairDatasetCmass(split="test",pad=0,transform="log1p"); ds_raw=PatchPairDatasetCmass(split="test",pad=0,normalize_inputs=False)
idx=ds.ids[0]; lr_m,_=ds.load_boxes(idx); lr_c,hr_c=ds_raw.load_boxes(idx); print(f"box idx={idx}",flush=True)
xb=torch.from_numpy(np.stack([extract_patch(lr_m,q,0) for q in range(N_PATCHES)])).to(dev)
th=torch.from_numpy(FID).unsqueeze(0).expand(N_PATCHES,-1).to(dev)
_nz=[torch.cat([t.flatten() for t in build_noise_list(PATCH,4,sd,dev)]) for sd in range(M)]   # the exact draws used
NZ_MIN=min(t.min().item() for t in _nz); NZ_MAX=max(t.max().item() for t in _nz); NZ_STD=float(np.mean([t.std().item() for t in _nz]))
print(f"noise draws: seeds 0-{M-1}, 8 x (1,{PATCH},{PATCH},{PATCH}) ~ N(0,1) each; values in [{NZ_MIN:.3f}, {NZ_MAX:.3f}], std {NZ_STD:.3f}",flush=True)
d_hr=counts_to_delta(hr_c)[0]; d_lr=counts_to_delta(lr_c)[0]; k_nyq=np.pi*N_FULL/LBOX
def coh(k,Pa,Pb,Pab,m): r=np.zeros_like(k); r[m]=Pab[m]/np.sqrt(np.maximum(Pa*Pb,1e-60))[m]; return r
k,Phr,Plr,Pab,m=cross_pk(d_hr,d_lr,LBOX,n_bins=NBINS); r_lr=coh(k,Phr,Plr,Pab,m)
res={}
for name,ck in MODELS.items():
    c=torch.load(ck,map_location=dev,weights_only=False); s=c.get("args",{}) or {}; nb=s.get("num_blocks",4)
    G=G_correct(1,1,5,chan_base=s.get("chan_base_g",128),num_blocks=nb).to(dev); G.load_state_dict(c["model"]); G.eval()
    nstd=torch.cat([v.abs().flatten() for kk,v in c["model"].items() if kk.endswith(".std")])
    Ps=[]; Rs=[]; fields=[]
    with torch.no_grad():
        for seed in range(M):
            nl=build_noise_list(PATCH,nb,seed,dev)                       # a DIFFERENT noise vector per draw
            fake=crop_interior(G(xb,th,nl),0).cpu().numpy()
            sr=stitch_patches(to_counts(fake,"log1p",ds.scale)); fields.append(sr[0])
            d=counts_to_delta(sr)[0]; _,_,Psr,Pab_s,ms=cross_pk(d_hr,d,LBOX,n_bins=NBINS)
            Ps.append(Psr); Rs.append(coh(k,Phr,Psr,Pab_s,ms))
    Ps=np.array(Ps); Rs=np.array(Rs); F=np.array(fields); T=Ps/np.maximum(Phr,1e-30)
    vox=F.std(0).mean()/(F.mean(0).mean()+1e-8)
    print(f"{name}: learned noise |std| mean={nstd.mean():.2e} max={nstd.max():.2e} | field std across {M} draws / mean = {vox*100:.4f}% | "
          f"max_k std(T)={np.nanmax(np.nan_to_num(T.std(0))):.5f} | max_k std(r)={np.nanmax(Rs.std(0)):.5f}",flush=True)
    res[name]=(Ps,T,Rs,nstd,vox)
good=(Phr>0)&(k>0); kk=k[good]
ii=[np.argmin(np.abs(k-x)) for x in (0.02,0.1,0.3)]
print("HR and LR are computed ONCE and reused in both columns. P_HR at k~0.02/0.1/0.3:",np.round(Phr[ii],1),"P_LR:",np.round(Plr[ii],1),"LR/HR:",np.round((Plr/Phr)[ii],3),flush=True)
fig,ax=plt.subplots(3,2,figsize=(13,11),sharex=True,sharey="row")   # identical HR/LR must render identically
for j,(name,(Ps,T,Rs,nstd,vox)) in enumerate(res.items()):
    a=ax[:,j]
    a[0].plot(kk,Phr[good],color="red",lw=2,label="HR",zorder=5); a[0].plot(kk,Plr[good],color="C3",ls=":",lw=1.8,label="LR")
    for i in range(M): a[0].plot(kk,np.where(Ps[i]>0,Ps[i],np.nan)[good],color="C0",lw=0.9,alpha=0.7,label="SR, 10 noise draws" if i==0 else None)
    a[0].set_xscale("log"); a[0].set_yscale("log"); a[0].set_ylabel("P(k)"); a[0].legend(fontsize=8)
    a[0].set_title(f"{name}\nlearned noise |std| {nstd.mean():.1e}; field spread across draws {vox*100:.3f}%",fontsize=10)
    a[1].plot(kk,(Plr/np.maximum(Phr,1e-30))[good],color="C3",ls=":",lw=1.8,label="LR/HR")
    for i in range(M): a[1].plot(kk,T[i][good],color="C0",lw=0.9,alpha=0.7,label="SR/HR, 10 draws" if i==0 else None)
    a[1].axhline(1,color="k",ls="--",lw=1); a[1].set_ylim(0.6,1.5); a[1].set_ylabel("Transfer (X/HR), linear"); a[1].legend(fontsize=8)
    a[2].plot(kk,r_lr[good],color="C3",ls=":",lw=1.8,label="LR x HR")
    for i in range(M): a[2].plot(kk,Rs[i][good],color="C0",lw=0.9,alpha=0.7,label="SR x HR, 10 draws" if i==0 else None)
    a[2].axhline(1,color="k",ls="--",lw=1); a[2].axhline(0,color="k",ls="--",lw=1); a[2].set_ylim(-0.25,1.1)
    a[2].set_xlabel("k [1/Box]"); a[2].set_ylabel("Cross-power (X x HR)"); a[2].legend(fontsize=8)
    for q in a: q.axvline(k_nyq,color="gray",lw=1.2); q.grid(alpha=0.3,which="both")
fig.suptitle(f"One LR box (set{idx}): 10 SR realisations from 10 different noise vectors, fiducial theta\n"
             f"noise per draw: 8 x {PATCH}^3 cubes ~ N(0,1), seeds 0-{M-1}, values in [{NZ_MIN:.2f}, {NZ_MAX:.2f}] (std {NZ_STD:.2f}); injected as learned_std x noise",fontsize=11)
plt.tight_layout(); out=f"figures_cmass/pk_panel_noise10_set{idx}.png"; plt.savefig(out,dpi=130); print("saved",out,"\nNOISE10_DONE",flush=True)
