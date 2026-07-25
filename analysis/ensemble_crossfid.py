"""DEFINITIVE resolution of the cross-fid robustness question (F18-F20).
Hermans et al. 2022 fix for NPE non-reproducibility: ENSEMBLE the NDEs.
Members = retrained seeds 0-3 (4 per condition) from the robust job's saved
posteriors. Three analyses:
 (1) NDE LEGITIMACY: per-member parameter recovery |mu-theta_true| on HR test.
     If all sensible (~0.10, cf FINAL.md), the retrained NDEs are valid and
     their disagreement is real (not a broken-training artifact).
 (2) REAL FLOOR: pairwise KL within the HR-ensemble members.
 (3) ENSEMBLED RANKING: pool samples across members -> ensemble posteriors;
     per box, is A_nopk closer to HR-ensemble than LR? Fraction + mean.
"""
import numpy as np, torch, sys, time
sys.path.insert(0,".")
from evaluate import _load_pk_set, _load_posterior, _sample, kl_gauss
R="runs/patch_cmass"; T=f"{R}/_robust_tmp"
theta_tab=np.load("data/cmass_theta.npz")["theta"]
test=set(np.load(f"{R}/split_sids.npz")["test_sids"].tolist())
pk_hr={s:v for s,v in _load_pk_set(f"{R}/pk_hr","pk_set").items() if s in test}
common=sorted(pk_hr)
SEEDS=[0,1,2,3]
load=lambda p:_load_posterior(p)
t0=time.time()

def members(tag): return [load(f"{T}/q{tag}_{s}.pkl") for s in SEEDS]
qHR=members("hr"); qA=members("Anopk"); qL=members("LR")

# (1) legitimacy: per-member theta recovery on HR test (posterior mean vs truth)
print("=== (1) NDE LEGITIMACY: per-member |posterior-mean - theta_true| on HR test ===",flush=True)
for name,mem in [("q_HR",qHR),("q_Anopk",qA),("q_LR",qL)]:
    recs=[]
    for m in mem:
        torch.manual_seed(0); errs=[]
        for sid in common[:60]:
            s=_sample(m,pk_hr[sid],800); errs.append(np.abs(s.mean(0)-theta_tab[sid]))
        recs.append(np.mean(errs))
    print(f"  {name}: per-seed recovery err = {np.round(recs,3).tolist()}  (all similar+sensible => legit)",flush=True)
print(f"  [t={time.time()-t0:.0f}s]",flush=True)

# ensemble sampler: pool equal shares across members
def ens_sample(mem,x,n=2000):
    per=n//len(mem); return np.concatenate([_sample(m,x,per) for m in mem],0)

# (2) real floor: pairwise KL within HR ensemble
print("\n=== (2) REAL FLOOR: pairwise KL within q_HR members ===",flush=True)
fl=[]
for i in range(len(qHR)):
    for j in range(len(qHR)):
        if i<j:
            torch.manual_seed(5); ks=[]
            for sid in common[:80]:
                a=_sample(qHR[i],pk_hr[sid],1500); b=_sample(qHR[j],pk_hr[sid],1500)
                ks.append(kl_gauss(a.mean(0),a.std(0),b.mean(0),b.std(0)).mean())
            fl.append(np.mean(ks))
print(f"  within-q_HR KL: {np.round(fl,4).tolist()}  mean={np.mean(fl):.4f}  (vs F15's 0.0007)",flush=True)
print(f"  [t={time.time()-t0:.0f}s]",flush=True)

# (3) ensembled ranking
print("\n=== (3) ENSEMBLED comparison (pooled across 4 NDE members each) ===",flush=True)
torch.manual_seed(9); dA=[]; dL=[]
for sid in common:
    x=pk_hr[sid]
    h=ens_sample(qHR,x); a=ens_sample(qA,x); l=ens_sample(qL,x)
    ka=kl_gauss(h.mean(0),h.std(0),a.mean(0),a.std(0)).mean()
    kl_=kl_gauss(h.mean(0),h.std(0),l.mean(0),l.std(0)).mean()
    dA.append(ka); dL.append(kl_)
dA=np.array(dA); dL=np.array(dL)
print(f"  ensembled KL(HR||Anopk) mean={dA.mean():.4f}  KL(HR||LR) mean={dL.mean():.4f}",flush=True)
print(f"  A_nopk closer than LR on {np.mean(dA<dL)*100:.0f}% of boxes  "
      f"mean(Anopk-LR)={ (dA-dL).mean():+.4f}",flush=True)
print("  (F16 single-NDE claim was 82%. Ensembled = the trustworthy answer.)",flush=True)
print("ENSEMBLE_DONE",flush=True)
