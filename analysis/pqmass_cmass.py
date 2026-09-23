"""PQMass two-sample test (Lemos et al. 2024, arXiv:2402.04355) for the CMASS
count fields: does the distribution of SR fields match HR? LR vs HR is the control.

Method (self-contained; the pip package is unavailable here): pool the two sample
sets, and for many random tessellations pick n_regions reference points from the
pool, assign every sample to its nearest reference (Voronoi), count how many of
each set land in each region, and form the two-sample chi^2
    chi2 = sum_i ( sqrt(N2/N1) k1_i - sqrt(N1/N2) k2_i )^2 / (k1_i + k2_i).
Under the null (same distribution) chi2 ~ chi2_{n_regions-1}. A distribution
mismatch pushes chi2 well above the dof and the tessellation p-values toward 0.

Field representation: each 128^3 count box is average-pooled to 16^3, log1p'd, and
standardized per feature by the HR statistics. This is a genuine FIELD-level test
(sensitive to structure beyond the power spectrum), unlike comparing P(k) directly.
Uses the VAL split (held out; all three sources available on disk)."""
import numpy as np, os, sys, time
from scipy.stats import chi2 as chi2dist
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

from paths import PROCESSED as PROC
from paths import SR_FIELDS
SRDIR=sys.argv[1] if len(sys.argv)>1 else f"{SR_FIELDS}/Afix_fiducial"
TAG  =sys.argv[2] if len(sys.argv)>2 else "Afix_fiducial"
NREG=10; NTESS=200; NMAX=200
rng=np.random.default_rng(0)
# Count-histogram (1-point PDF) representation: a genuine DISTRIBUTIONAL feature where
# clumpy HR and smooth LR differ, and which is not dominated by the shared (matched-IC)
# large-scale structure. Fractions of voxels in these count bins, + log mean count.
CBINS=np.array([-0.5,0.5,1.5,2.5,3.5,4.5,5.5,7.5,11.5,20.5,np.inf])
def feat(n):
    n=np.asarray(n,np.float32).ravel()
    h,_=np.histogram(n,bins=CBINS); h=h/h.sum()
    return np.concatenate([h,[np.log1p(n.mean())]]).astype(np.float32)   # 11-dim
pool16=feat  # keep call sites

va=sorted(int(d["idx"]) for d in np.load(f"{PROC}/val_list.npy",allow_pickle=True))
va=[i for i in va if os.path.exists(f"{SRDIR}/sr_{i}.npy")][:NMAX]
print(f"PQMass on {len(va)} VAL boxes | rep=1-point count histogram ({len(CBINS)-1} bins + logmean) | regions={NREG} tess={NTESS}",flush=True)
t0=time.time()
HR=np.stack([pool16(np.load(f"{PROC}/{i:04d}_label.npy")) for i in va])
LR=np.stack([pool16(np.load(f"{PROC}/{i:04d}_input.npy")) for i in va])
SR=np.stack([pool16(np.load(f"{SRDIR}/sr_{i}.npy")) for i in va])
mu,sd=HR.mean(0),HR.std(0)+1e-8
HR=(HR-mu)/sd; LR=(LR-mu)/sd; SR=(SR-mu)/sd
print(f"loaded+featurized {time.time()-t0:.0f}s  dim={HR.shape[1]}",flush=True)

def pqmass(A,B):
    """chi^2 per tessellation between sample sets A (N1,d), B (N2,d)."""
    N1,N2=len(A),len(B); pool=np.concatenate([A,B],0); chis=[]
    for _ in range(NTESS):
        ref=pool[rng.choice(len(pool),NREG,replace=False)]           # reference points
        # nearest-reference assignment (Voronoi)
        dA=((A[:,None,:]-ref[None])**2).sum(-1).argmin(1)
        dB=((B[:,None,:]-ref[None])**2).sum(-1).argmin(1)
        k1=np.bincount(dA,minlength=NREG).astype(float)
        k2=np.bincount(dB,minlength=NREG).astype(float)
        keep=(k1+k2)>0
        num=(np.sqrt(N2/N1)*k1-np.sqrt(N1/N2)*k2)**2
        chis.append((num[keep]/(k1+k2)[keep]).sum())
    return np.array(chis)

dof=NREG-1
res={}
for name,X in [("SR vs HR",SR),("LR vs HR",LR)]:
    c=pqmass(X,HR)
    p=chi2dist.sf(c,dof)                # p-value per tessellation
    res[name]=(c,p)
    print(f"  {name:9s}: chi2/dof median={np.median(c)/dof:.2f}  mean_p={p.mean():.3f}  "
          f"frac(p<0.05)={np.mean(p<0.05):.2f}   (match => chi2/dof~1, p~uniform)",flush=True)

# plot: chi2 histograms vs the chi2_dof null pdf
fig,ax=plt.subplots(1,2,figsize=(11,4.4))
xx=np.linspace(0,max(np.percentile(res['LR vs HR'][0],99),3*dof),300)
for nm,col in [("SR vs HR","C0"),("LR vs HR","C3")]:
    ax[0].hist(res[nm][0],bins=30,density=True,alpha=0.5,color=col,
               label=f"{nm} (median chi2/dof={np.median(res[nm][0])/dof:.1f})")
ax[0].plot(xx,chi2dist.pdf(xx,dof),"k--",lw=1.5,label=f"null chi2_{{{dof}}}")
ax[0].set_xlabel("PQMass chi2 per tessellation"); ax[0].set_ylabel("density")
ax[0].set_title(f"PQMass field-level test ({TAG})"); ax[0].legend(fontsize=8)
for nm,col in [("SR vs HR","C0"),("LR vs HR","C3")]:
    ax[1].hist(res[nm][1],bins=20,range=(0,1),density=True,alpha=0.5,color=col,label=nm)
ax[1].axhline(1.0,color="k",ls="--",lw=1,label="uniform (match)")
ax[1].set_xlabel("tessellation p-value"); ax[1].set_ylabel("density")
ax[1].set_title("p-values (uniform if distributions match)"); ax[1].legend(fontsize=8)
plt.tight_layout(); out=f"figures_cmass/pqmass_{TAG}.png"; plt.savefig(out,dpi=140)
print(f"saved {out}\nPQMASS_DONE",flush=True)
