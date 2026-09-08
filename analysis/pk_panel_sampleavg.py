"""Per-patch P(k) panel: HR (paired) vs LR (single input patch) vs SR as
(a) ONE stochastic sample and (b) the MEAN over M=10 noise draws of the same patch.

Tests the posterior-mean-field question: does averaging SR samples move the power
spectrum, transfer, or coherence toward HR? Context: the generator is effectively
deterministic (noise std ~2e-4, D-F22), so the expectation is that the 10-sample mean
equals a single sample and averaging changes nothing. If it does move, the determinism
finding was wrong at the P(k) level.

Comparable to pk_panel_patch_cmassAfix.png by construction: the SAME per-patch estimator
as diag_cmass (cross_pk on delta=n/nbar-1, lbox_patch=500, n_bins=40), the SAME noise seed
0 for the single sample (so SR-single IS the existing panel's SR), and SR kept as the
CONTINUOUS to_counts output (no rounding) exactly as the existing panel does, so the ONLY
difference between SR-single and SR-avg is the averaging."""
import sys, numpy as np, torch
sys.path.insert(0, ".")
from map2map.models.styled_srsgan import G_correct
from data.patch_dataset_cmass import (PatchPairDatasetCmass, extract_patch, crop_interior,
    to_counts, counts_to_delta, PATCH, N_PATCHES, N_FULL)
from analysis.plot_diagnostic import cross_pk
from analysis.diag_cmass import build_noise_list
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

CKPT=sys.argv[1]; TAG=sys.argv[2] if len(sys.argv)>2 else "cmassAfix"
M=10; NSIMS=16; NBINS=40; LBOX=1000.0
FIDUCIAL=np.array([0.30,0.05,0.70,1.00,0.80],np.float32)
dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
ck=torch.load(CKPT,map_location=dev,weights_only=False); s=ck.get("args",{}) or {}
cb,nb=s.get("chan_base_g",128),s.get("num_blocks",4); tf=s.get("transform","log1p")
pad=0   # naive arm (cmassAfix)
G=G_correct(1,1,5,chan_base=cb,num_blocks=nb).to(dev); G.load_state_dict(ck["model"]); G.eval()
ds=PatchPairDatasetCmass(split="test",pad=pad,transform=tf)
ds_raw=PatchPairDatasetCmass(split="test",pad=0,normalize_inputs=False)
ids=ds.ids[:NSIMS]
lbox_patch=LBOX*PATCH/N_FULL; k_nyq=np.pi*PATCH/lbox_patch
th=torch.from_numpy(FIDUCIAL).unsqueeze(0).expand(N_PATCHES,-1).to(dev)
nls=[build_noise_list(PATCH+2*pad,nb,seed,dev) for seed in range(M)]   # M independent noise draws; seed 0 == diag's
print(f"sample-avg panel: {len(ids)} sims x {N_PATCHES} patches, M={M} draws, n_bins={NBINS}",flush=True)

def coh(k,Pa,Pb,Pab,m):
    r=np.zeros_like(k); r[m]=Pab[m]/np.sqrt(np.maximum(Pa*Pb,1e-60))[m]; return r

res=[]        # per patch: (k, P_hr, P_lr, P_single, P_avg, r_lr, r_single, r_avg)
field_rel=[]  # per patch mean|avg-single|/mean(single): the averaging effect on the FIELD itself
with torch.no_grad():
    for i,idx in enumerate(ids):
        lr_m,_=ds.load_boxes(idx); lr_c,hr_c=ds_raw.load_boxes(idx)
        xb=torch.from_numpy(np.stack([extract_patch(lr_m,q,pad) for q in range(N_PATCHES)])).to(dev)
        draws=np.stack([to_counts(crop_interior(G(xb,th,nls[m]),pad).cpu().numpy(),tf,ds.scale)
                        for m in range(M)])                     # (M,8,1,64,64,64) continuous counts
        s1=draws[0]; avg=draws.mean(0)
        for q in range(N_PATCHES):
            dh=counts_to_delta(extract_patch(hr_c,q,0)[0]); dl=counts_to_delta(extract_patch(lr_c,q,0)[0])
            d1=counts_to_delta(s1[q][0]); da=counts_to_delta(avg[q][0])
            k,Ph,P1,Pab1,m1=cross_pk(dh,d1,lbox_patch,n_bins=NBINS)
            _,_,Pa,Paba,ma=cross_pk(dh,da,lbox_patch,n_bins=NBINS)
            _,_,Pl,Pabl,ml=cross_pk(dh,dl,lbox_patch,n_bins=NBINS)
            res.append((k,Ph,Pl,P1,Pa,coh(k,Ph,Pl,Pabl,ml),coh(k,Ph,P1,Pab1,m1),coh(k,Ph,Pa,Paba,ma)))
            field_rel.append(np.abs(avg[q][0]-s1[q][0]).mean()/(s1[q][0].mean()+1e-8))
        print(f"  sim {i+1}/{len(ids)} (set{idx}) done",flush=True)

# ---- aggregate ----
K=np.stack([r[0] for r in res]); k=np.nanmedian(K,0); v=k>0; k=k[v]
def A(j): return np.stack([r[j] for r in res])[:,v]
Ph,Pl,P1,Pa,Rl,R1,Ra=[A(j) for j in range(1,8)]
def band(X): return np.nanmedian(X,0),np.nanpercentile(X,16,0),np.nanpercentile(X,84,0)
Tl=Pl/np.maximum(Ph,1e-30); T1=P1/np.maximum(Ph,1e-30); Ta=Pa/np.maximum(Ph,1e-30)
rms=lambda T: np.sqrt(np.nanmean((T-1.0)**2))
print(f"\nAVERAGING EFFECT on the field: mean |avg - single| / mean(single) = {np.mean(field_rel)*100:.4f}%",flush=True)
print(f"RMS err vs HR: LR {rms(Tl)*100:.1f}%  SR-single {rms(T1)*100:.1f}%  SR-avg{M} {rms(Ta)*100:.1f}%",flush=True)
print(f"max |T_avg - T_single| over k (median curves) = {np.nanmax(np.abs(band(Ta)[0]-band(T1)[0])):.5f}",flush=True)
print(f"max |r_avg - r_single| over k (median curves) = {np.nanmax(np.abs(band(Ra)[0]-band(R1)[0])):.5f}",flush=True)

# ---- plot: 3 panels, 4 series (SR-avg drawn dashed ON TOP of SR-single so overlap is visible) ----
fig,ax=plt.subplots(3,1,figsize=(7.5,11),sharex=True)
hm,_,_=band(Ph); lm,_,_=band(Pl); m1,lo1,hi1=band(P1); ma,loa,hia=band(Pa)
ax[0].plot(k,hm,color="red",lw=2,label="HR (paired)")
ax[0].plot(k,lm,color="C3",lw=1.8,ls=":",label="LR (single patch)")
ax[0].fill_between(k,lo1,hi1,color="C0",alpha=0.25); ax[0].plot(k,m1,color="C0",lw=2,label="SR, 1 sample")
ax[0].plot(k,ma,color="C2",lw=2,ls="--",label=f"SR, mean of {M} samples")
ax[0].axvline(k_nyq,color="gray",lw=1.5); ax[0].set_xscale("log"); ax[0].set_yscale("log")
ax[0].set_ylabel("P(k)"); ax[0].set_title(f"Per-patch P(k): 1 SR sample vs mean of {M} ({TAG})")
ax[0].legend(fontsize=9); ax[0].grid(alpha=0.3,which="both")
tl,tll,tlh=band(Tl); t1,t1l,t1h=band(T1); ta,tal,tah=band(Ta)
ax[1].fill_between(k,tll,tlh,color="C3",alpha=0.12); ax[1].plot(k,tl,color="C3",lw=1.8,ls=":",label=f"LR/HR (RMS {rms(Tl)*100:.1f}%)")
ax[1].fill_between(k,t1l,t1h,color="C0",alpha=0.25); ax[1].plot(k,t1,color="C0",lw=2,label=f"SR 1-sample/HR (RMS {rms(T1)*100:.1f}%)")
ax[1].plot(k,ta,color="C2",lw=2,ls="--",label=f"SR {M}-avg/HR (RMS {rms(Ta)*100:.1f}%)")
ax[1].axhline(1,color="k",lw=1,ls="--"); ax[1].axvline(k_nyq,color="gray",lw=1.5)
ax[1].set_xscale("log"); ax[1].set_ylim(0.6,1.5); ax[1].set_ylabel("Transfer (X/HR), linear")
ax[1].legend(fontsize=9); ax[1].grid(alpha=0.3,which="both")
rl,_,_=band(Rl); r1,r1l,r1h=band(R1); ra,_,_=band(Ra)
ax[2].plot(k,rl,color="C3",lw=1.8,ls=":",label="LR x HR")
ax[2].fill_between(k,r1l,r1h,color="C0",alpha=0.25); ax[2].plot(k,r1,color="C0",lw=2,label="SR 1-sample x HR")
ax[2].plot(k,ra,color="C2",lw=2,ls="--",label=f"SR {M}-avg x HR")
ax[2].axhline(1,color="k",lw=1,ls="--"); ax[2].axhline(0,color="k",lw=1,ls="--"); ax[2].axvline(k_nyq,color="gray",lw=1.5)
ax[2].set_xscale("log"); ax[2].set_ylim(-0.25,1.1); ax[2].set_xlabel("k [1/Box]"); ax[2].set_ylabel("Cross-power (X x HR)")
ax[2].legend(fontsize=9); ax[2].grid(alpha=0.3,which="both")
plt.tight_layout(); out=f"figures_cmass/pk_panel_patch_sampleavg_{TAG}.png"; plt.savefig(out,dpi=120)
print(f"saved {out}\nPKSAMPAVG_DONE",flush=True)
