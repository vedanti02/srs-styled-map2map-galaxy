"""Corrected-label cross-fidelity for ALL conditions present in _robust_tmp
(hr, Anopk, LR, Baware, Bunaware). Each condition: up to 4 separate-process NDE
replicates trained on that source's P(k) with corrected labels, evaluated on the
SAME HR test P(k). Reports each condition's ensembled cross-fid KL to the HR
ensemble (mean over test boxes) plus the within-HR floor. Extends the earlier
hr/Anopk/LR-only cross-fid to the padding variants."""
import numpy as np, torch, sys, time, os
sys.path.insert(0,".")
from evaluate import _load_pk_set, _load_posterior, _sample, kl_gauss
R="runs/patch_cmass"; T=f"{R}/_robust_tmp"
test=set(np.load(f"{R}/split_sids.npz")["test_sids"].tolist())
pk_hr={s:v for s,v in _load_pk_set(f"{R}/pk_hr","pk_set").items() if s in test}
common=sorted(pk_hr); SEEDS=[0,1,2,3]; t0=time.time()

def members(tag):
    out=[]
    for s in SEEDS:
        p=f"{T}/q{tag}_{s}.pkl"
        if os.path.exists(p): out.append(_load_posterior(p))
    return out

TAGS=["hr","Anopk","LR","Baware","Bunaware"]
mem={t:members(t) for t in TAGS}
for t in TAGS: print(f"  {t}: {len(mem[t])} members",flush=True)
qHR=mem["hr"]
def ens(ms,x,n=2000):
    per=max(1,n//len(ms)); return np.concatenate([_sample(m,x,per) for m in ms],0)

print("\n=== corrected cross-fidelity KL to HR ensemble (mean over test boxes) ===",flush=True)
torch.manual_seed(9)
for t in TAGS:
    if t=="hr" or not mem[t]: continue
    ks=[]
    for sid in common:
        h=ens(qHR,pk_hr[sid]); x=ens(mem[t],pk_hr[sid])
        ks.append(kl_gauss(h.mean(0),h.std(0),x.mean(0),x.std(0)).mean())
    print(f"  {t:9s}: KL={np.mean(ks):.4f} +/- {np.std(ks):.4f}  [t={time.time()-t0:.0f}s]",flush=True)

fl=[]
for i in range(len(qHR)):
    for j in range(i+1,len(qHR)):
        torch.manual_seed(5); ks=[]
        for sid in common[:80]:
            a=_sample(qHR[i],pk_hr[sid],1500); b=_sample(qHR[j],pk_hr[sid],1500)
            ks.append(kl_gauss(a.mean(0),a.std(0),b.mean(0),b.std(0)).mean())
        fl.append(np.mean(ks))
print(f"\n  HR self floor: mean={np.mean(fl):.4f}",flush=True)
print("PADDING_XFID_DONE",flush=True)
