"""Per-box field-level cross-fidelity KL, to test whether the LR halo-count catastrophe
(sparse tail in high-Om/high-s8 boxes, D-F27) explains LR's poor/bimodal field cross-fid.

Reuses eval_field_crossfid.py machinery (kl_gauss, FieldCNN loading, test ids, HR field
tensor) but SAVES the per-box KL that the eval computes for its ranking and then discards.
Separates two axes a naive test would conflate:
  SEED axis: D-F18's bimodality is across LR-trained estimators (~2 of 8 broken).
  BOX  axis: the hypothesis proper: for GOOD LR estimators, is per-box KL heavy-tailed, and
             are the high-KL boxes the high-Om*s8 / large-deficit boxes?  SR is the control."""
import glob, os, sys, time, numpy as np, torch
sys.path.insert(0, ".")
from models.field_cnn import FieldCNN
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from paths import PROCESSED as PROC
CKD="runs/patch_cmass/field_ckpts_fiducial"
dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def kl_gauss(m1,s1,m2,s2,eps=1e-8):
    s1=s1**2+eps; s2=s2**2+eps
    return 0.5*(np.log(s2/s1)+(s1+(m1-m2)**2)/s2-1.0)

theta=np.load("data/cmass_theta.npz")["theta"]
te=sorted(int(d["idx"]) for d in np.load(f"{PROC}/test_list.npy",allow_pickle=True))
n=len(te); print(f"{n} test boxes",flush=True); t0=time.time()
# --- HR test fields (log1p, what the CNN sees) + LR/HR halo totals for the SAME boxes ---
F=[]; n_lr=np.zeros(n); n_hr=np.zeros(n)
for j,i in enumerate(te):
    hr=np.load(f"{PROC}/{i:04d}_label.npy").astype(np.float32); n_hr[j]=hr.sum()
    n_lr[j]=np.load(f"{PROC}/{i:04d}_input.npy").astype(np.float32).sum()
    F.append(np.log1p(hr))
    if (j+1)%50==0: print(f"  loaded {j+1}/{n}  {time.time()-t0:.0f}s",flush=True)
F=torch.tensor(np.stack(F),device=dev).unsqueeze(1)            # (n,1,128,128,128)
deficit=(n_lr-n_hr)/n_hr; clust=theta[te,0]*theta[te,4]          # Om*s8 clustering proxy
print(f"test-box LR count deficit: median {np.median(deficit)*100:+.2f}%  min {deficit.min()*100:+.1f}%  "
      f"boxes < -20%: {(deficit<-0.2).sum()}",flush=True)

# --- Gaussian posterior (mu,sigma) per box from every checkpoint ---
posts={}
for ck in sorted(glob.glob(f"{CKD}/q*_s*.pt")):
    name=os.path.basename(ck)[:-3]
    d=torch.load(ck,map_location=dev,weights_only=False)
    net=FieldCNN().to(dev); net.load_state_dict(d["model"]); net.eval()
    mean,std=d["mean"],d["std"]; mus=[]; sigs=[]
    with torch.no_grad():
        for k in range(0,n,16):
            mu,sg=net((F[k:k+16]-mean)/std); mus.append(mu.cpu().numpy()); sigs.append(sg.cpu().numpy())
    posts[name]=(np.concatenate(mus),np.concatenate(sigs)); print(f"  {name} done",flush=True)
def members(src): return {k:v for k,v in posts.items() if k.startswith(f"q{src}_")}
HR,LR,SR=members("hr"),members("lr"),members("sr")

def perbox(mem):   # (n_x, n_hr, n): per-box KL of each X estimator vs each HR reference
    return np.array([[kl_gauss(rm,rs,xm,xs).mean(1) for (rm,rs) in HR.values()] for (xm,xs) in mem.values()])
KL_LR=perbox(LR); KL_SR=perbox(SR)
hk=list(HR.values())
KL_HR=np.array([kl_gauss(hk[a][0],hk[a][1],hk[b][0],hk[b][1]).mean(1)
                for a in range(len(hk)) for b in range(a+1,len(hk))])   # (15,n) HR self floor
lr_names=list(LR); sr_names=list(SR)
print(f"\nSANITY vs D-F19: LR {KL_LR.mean():.4f}   SR {KL_SR.mean():.4f}   HR floor {KL_HR.mean():.4f}   "
      f"(expect ~0.47 / ~0.037 / ~0.019)",flush=True)

# --- SEED axis: good vs bad LR estimators ---
seed_mean=KL_LR.mean(axis=(1,2))
print("\nLR seeds by mean KL:")
for j in np.argsort(seed_mean): print(f"  {lr_names[j]}: {seed_mean[j]:.4f}")
thr=3*np.median(seed_mean); bad=seed_mean>thr; good=~bad
print(f"bad LR seeds (> 3x median = {thr:.3f}): {[lr_names[j] for j in np.where(bad)[0]]}",flush=True)
kl_good=KL_LR[good].mean(axis=(0,1))
kl_bad=KL_LR[bad].mean(axis=(0,1)) if bad.any() else None
kl_sr=KL_SR.mean(axis=(0,1)); kl_hr=KL_HR.mean(0)

# --- BOX axis, good LR seeds (the hypothesis proper) ---
def corr(a,b): return np.corrcoef(a,b)[0,1]
print(f"\nBOX AXIS (good LR seeds), per-box KL over {n} boxes:")
print(f"  median {np.median(kl_good):.4f}  mean {kl_good.mean():.4f}  p95 {np.percentile(kl_good,95):.4f}  "
      f"max {kl_good.max():.4f}   (heavy tail if max >> p95)")
print(f"  corr(KL, LR count deficit) = {corr(kl_good,deficit):+.3f}   corr(KL, |deficit|) = {corr(kl_good,np.abs(deficit)):+.3f}")
print(f"  corr(KL, Om*s8)            = {corr(kl_good,clust):+.3f}")
print(f"  SR control: corr(KL_SR, deficit) = {corr(kl_sr,deficit):+.3f}   corr(KL_SR, Om*s8) = {corr(kl_sr,clust):+.3f}")
q=np.percentile(clust,[0,50,80,95,100])
bins=[(q[0],q[1],"bottom50"),(q[1],q[2],"50-80"),(q[2],q[3],"80-95"),(q[3],q[4],"top5")]
print("  mean per-box KL by Om*s8 bin:      LR-good   LR-bad    SR       HR-floor")
rows=[]
for lo,hi,nm in bins:
    m=(clust>=lo)&(clust<=hi)
    rows.append((nm,kl_good[m].mean(),(kl_bad[m].mean() if kl_bad is not None else np.nan),kl_sr[m].mean(),kl_hr[m].mean()))
    print(f"    {nm:9s} n={m.sum():3d}    {rows[-1][1]:.4f}   {rows[-1][2]:.4f}   {rows[-1][3]:.4f}   {rows[-1][4]:.4f}")
top=clust>=q[3]
print(f"  share of total LR-good KL carried by the top-5% Om*s8 boxes: {kl_good[top].sum()/kl_good.sum()*100:.1f}%   (5% if uniform)")
cat=deficit<-0.2
print(f"  catastrophic-count boxes (deficit < -20%): n={cat.sum()}   mean KL LR-good "
      f"{(kl_good[cat].mean() if cat.any() else float('nan')):.4f} vs other boxes {kl_good[~cat].mean():.4f}")
if kl_bad is not None:
    ex=kl_bad-kl_good
    print(f"\nSEED AXIS: is the bad seeds' excess KL concentrated on catastrophic boxes?")
    print(f"  mean excess on cat boxes {(ex[cat].mean() if cat.any() else float('nan')):.4f} vs others {ex[~cat].mean():.4f}")
    print(f"  corr(excess, deficit) = {corr(ex,deficit):+.3f}   corr(excess, Om*s8) = {corr(ex,clust):+.3f}",flush=True)

np.savez("runs/patch_cmass/field_kl_perbox.npz",te=np.array(te),KL_LR=KL_LR,KL_SR=KL_SR,KL_HR=KL_HR,
         lr_names=np.array(lr_names),sr_names=np.array(sr_names),deficit=deficit,clust=clust,n_lr=n_lr,n_hr=n_hr,bad=bad)

# --- figure ---
fig,ax=plt.subplots(1,3,figsize=(16,4.6))
sc=ax[0].scatter(deficit*100,kl_good,c=clust,cmap="viridis",s=18,label="LR (good seeds)")
ax[0].scatter(deficit*100,kl_sr,color="C0",s=12,alpha=0.6,marker="x",label="SR (control)")
ax[0].set_yscale("log"); ax[0].set_xlabel("LR halo-count deficit (LR - HR)/HR  [%]"); ax[0].set_ylabel("per-box cross-fid KL")
ax[0].set_title("per-box KL vs count deficit (color = Om*s8)"); ax[0].legend(fontsize=8); plt.colorbar(sc,ax=ax[0],label="Om*s8")
lo=max(min(kl_hr.min(),kl_sr.min(),kl_good.min())*0.8,1e-4); hi=max(kl_good.max(),(kl_bad.max() if kl_bad is not None else 0))*1.2
b=np.logspace(np.log10(lo),np.log10(hi),40)
ax[1].hist(kl_hr,bins=b,alpha=0.5,color="0.5",label="HR floor"); ax[1].hist(kl_sr,bins=b,alpha=0.5,color="C0",label="SR")
ax[1].hist(kl_good,bins=b,alpha=0.5,color="C2",label="LR good seeds")
if kl_bad is not None: ax[1].hist(kl_bad,bins=b,alpha=0.5,color="C3",label="LR bad seeds")
ax[1].set_xscale("log"); ax[1].set_xlabel("per-box KL"); ax[1].set_ylabel("boxes"); ax[1].set_title("per-box KL distributions"); ax[1].legend(fontsize=8)
x=np.arange(len(rows)); w=0.2
ax[2].bar(x-1.5*w,[r[1] for r in rows],w,color="C2",label="LR good"); ax[2].bar(x-0.5*w,[r[2] for r in rows],w,color="C3",label="LR bad")
ax[2].bar(x+0.5*w,[r[3] for r in rows],w,color="C0",label="SR"); ax[2].bar(x+1.5*w,[r[4] for r in rows],w,color="0.5",label="HR floor")
ax[2].set_yscale("log"); ax[2].set_xticks(x); ax[2].set_xticklabels([r[0] for r in rows]); ax[2].set_xlabel("Om*s8 quantile bin")
ax[2].set_ylabel("mean per-box KL"); ax[2].set_title("KL by clustering strength"); ax[2].legend(fontsize=8)
plt.tight_layout(); plt.savefig("figures_cmass/field_kl_perbox.png",dpi=140)
print("\nsaved figures_cmass/field_kl_perbox.png\nFIELD_KL_DONE",flush=True)
