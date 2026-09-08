"""Discretization / stochasticity probe for the conditioned SR (deployment, fiducial theta).

Question: is SR's one-point count-PDF mismatch (PQMass chi2/dof 2.2 vs LF 0.83) and its
worse summary cross-fidelity caused by (a) rounding to integers, (b) the generator's missing
Poisson scatter (its learned noise is ~0, so it outputs one smoothed rate per voxel instead of
a stochastic count), or (c) something intrinsic to the generator's rate distribution?

From ONE continuous generator output we build three count-field variants:
  cont  = clip(exp(y)-1, 0, 200)          raw rate, un-rounded (float)
  round = round(cont)                      the currently deployed SR
  pois  = Poisson(cont)                    rate + physical integer scatter (approx count-aware head)
and compare, on VAL boxes, against HF (label) and LF (input):
  1) one-point count PDF (mean over boxes) + L1 distance to HF,
  2) PQMass chi2/dof vs HF (same feature/method as pqmass_cmass.py),
  3) mean P(k) and transfer T(k)=P_var/P_HF at large/mid/small scales.

Note by construction the count-histogram PQMass feature is ~invariant to rounding (bin edges are
the rounding thresholds), so 'round' vs 'cont' isolates the P(k) quantization effect while 'pois'
isolates the missing-scatter effect. If 'pois' matches HF but 'cont'/'round' do not, the fix is a
count-aware probabilistic output head, not de-rounding."""
import numpy as np, os, sys, time
sys.path.insert(0, ".")
from scipy.stats import chi2 as chi2dist
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from analysis.power_spectrum import cube_pk_counts

PROC="/data/group_data/universedata/cmass-ili/processed"
CONT=sys.argv[1] if len(sys.argv)>1 else "/data/user_data/vkshirsa/cmass-ili/sr_fields/Afix_fiducial_cont"
TAG =sys.argv[2] if len(sys.argv)>2 else "Afix_fiducial"
NMAX=200; NPK=40; L=1000.0
rng=np.random.default_rng(0)

CBINS=np.array([-0.5,0.5,1.5,2.5,3.5,4.5,5.5,7.5,11.5,20.5,np.inf])
LBL=["0","1","2","3","4","5","6-7","8-11","12-20","21+"]
def hist(n):
    n=np.asarray(n,np.float32).ravel(); h,_=np.histogram(n,bins=CBINS); return h/h.sum()
def feat(n):  # 11-dim PQMass feature: 10 count-bin fractions + log-mean
    return np.concatenate([hist(n),[np.log1p(np.asarray(n,np.float32).mean())]]).astype(np.float32)

va=sorted(int(d["idx"]) for d in np.load(f"{PROC}/val_list.npy",allow_pickle=True))
va=[i for i in va if os.path.exists(f"{CONT}/sr_{i}.npy")][:NMAX]
print(f"discretization probe on {len(va)} VAL boxes | cont dir={CONT}",flush=True)
t0=time.time()

# variant builders from the continuous rate
def variants(cont):
    cont=np.clip(np.nan_to_num(cont,nan=0.),0,200).astype(np.float32)
    return {"cont":cont, "round":np.round(cont), "pois":rng.poisson(cont).astype(np.float32)}

# ---- 1) one-point PDF (mean over boxes) + PQMass features ----
srcs=["HF","LF","cont","round","pois"]
pdf={s:[] for s in srcs}; F={s:[] for s in srcs}
for i in va:
    hf=np.load(f"{PROC}/{i:04d}_label.npy"); lf=np.load(f"{PROC}/{i:04d}_input.npy")
    v=variants(np.load(f"{CONT}/sr_{i}.npy"))
    boxes={"HF":hf,"LF":lf,**v}
    for s in srcs: pdf[s].append(hist(boxes[s])); F[s].append(feat(boxes[s]))
for s in srcs: pdf[s]=np.mean(pdf[s],0); F[s]=np.stack(F[s])
print(f"featurized {time.time()-t0:.0f}s",flush=True)

# ---- 2) PQMass chi2/dof vs HF ----
NREG,NTESS,dof=10,200,9
mu,sd=F["HF"].mean(0),F["HF"].std(0)+1e-8
Z={s:(F[s]-mu)/sd for s in srcs}
def pqmass(A,B):
    N1,N2=len(A),len(B); pool=np.concatenate([A,B],0); chis=[]
    for _ in range(NTESS):
        ref=pool[rng.choice(len(pool),NREG,replace=False)]
        dA=((A[:,None,:]-ref[None])**2).sum(-1).argmin(1)
        dB=((B[:,None,:]-ref[None])**2).sum(-1).argmin(1)
        k1=np.bincount(dA,minlength=NREG).astype(float); k2=np.bincount(dB,minlength=NREG).astype(float)
        keep=(k1+k2)>0
        chis.append((((np.sqrt(N2/N1)*k1-np.sqrt(N1/N2)*k2)**2)[keep]/(k1+k2)[keep]).sum())
    return np.median(chis)/dof

print("\n=== one-point PDF L1 to HF  &  PQMass chi2/dof vs HF (match ~1) ===",flush=True)
chi2dof={}
for s in ["LF","cont","round","pois"]:
    l1=np.abs(pdf[s]-pdf["HF"]).sum(); c=pqmass(Z[s],Z["HF"]); chi2dof[s]=c
    print(f"  {s:6s}: PDF-L1={l1:.4f}   PQMass chi2/dof={c:.2f}",flush=True)

# ---- 3) mean P(k) + transfer to HF ----
print("\n=== P(k) transfer T(k)=P_var/P_HF (NPK boxes) ===",flush=True)
pk={s:[] for s in ["HF","cont","round","pois"]}; kk=None
for i in va[:NPK]:
    hf=np.load(f"{PROC}/{i:04d}_label.npy").astype(np.float32)
    v=variants(np.load(f"{CONT}/sr_{i}.npy"))
    for s,cube in [("HF",hf),("cont",v["cont"]),("round",v["round"]),("pois",v["pois"])]:
        k,p,_=cube_pk_counts(cube,L,n_bins=32); kk=k; pk[s].append(p)
mpk={s:np.mean(pk[s],0) for s in pk}
def Tat(kt):
    j=np.argmin(np.abs(kk-kt)); return {s:mpk[s][j]/mpk["HF"][j] for s in ["cont","round","pois"]},kk[j]
for kt in [0.05,0.2,0.5]:
    T,kv=Tat(kt); print(f"  k~{kv:.3f}: "+"  ".join(f"{s} T={T[s]:.3f}" for s in ["cont","round","pois"]),flush=True)

# ---- plot ----
fig,ax=plt.subplots(1,3,figsize=(15,4.4))
x=np.arange(len(LBL)); cols={"HF":"k","LF":"C3","cont":"C0","round":"C1","pois":"C2"}
for s in srcs: ax[0].plot(x,pdf[s],"o-",ms=4,color=cols[s],label=s)
ax[0].set_yscale("log"); ax[0].set_xticks(x); ax[0].set_xticklabels(LBL,rotation=45,fontsize=7)
ax[0].set_xlabel("voxel count"); ax[0].set_ylabel("fraction of voxels"); ax[0].legend(fontsize=8)
ax[0].set_title("one-point count PDF (mean over boxes)")
bars=["LF","cont","round","pois"]
ax[1].bar(range(len(bars)),[chi2dof[b] for b in bars],color=[cols[b] for b in bars])
ax[1].axhline(1,color="k",ls="--",lw=1); ax[1].set_xticks(range(len(bars))); ax[1].set_xticklabels(bars)
ax[1].set_ylabel("PQMass chi2/dof (vs HF)"); ax[1].set_title("distributional match (1 = matches HF)")
for s in ["cont","round","pois"]: ax[2].plot(kk,mpk[s]/mpk["HF"],color=cols[s],label=s)
ax[2].axhline(1,color="k",ls="--",lw=1); ax[2].set_xscale("log"); ax[2].set_xlabel("k [h/Mpc]")
ax[2].set_ylabel("T(k)=P_var/P_HF"); ax[2].set_title("P(k) transfer to HF"); ax[2].legend(fontsize=8)
plt.tight_layout(); out=f"figures_cmass/discretization_{TAG}.png"; plt.savefig(out,dpi=140)
print(f"\nsaved {out}\nDISCRETIZATION_DONE",flush=True)
