"""Per-seed spread of the SUMMARY (P(k)) cross-fidelity KL, to test whether the
0.137 (SR) vs 0.031 (LR) bar is robust or a fragile single-NDE artifact.
For each NDE seed s: KL between q_HR(seed s) and q_source(seed s), evaluated on HR
test P(k), meaned over test boxes. Reports the mean +/- spread over seeds, plus the
HR-self floor across seed pairs.
  python analysis/seed_xfid_check.py [NDE_DIR] [TAG ...]
NDE_DIR defaults to the restored pre-Delta posteriors (_robust_tmp); new-protocol ones are in nde_v2."""
import numpy as np, torch, sys, os
sys.path.insert(0, ".")
from evaluate import _load_pk_set, _load_posterior, _sample, kl_gauss
R="runs/patch_cmass"; T=sys.argv[1] if len(sys.argv)>1 else f"{R}/_robust_tmp"
TAGS=sys.argv[2:] or ["LR","Anopk","Afixfid"]
test=set(np.load(f"{R}/split_sids.npz")["test_sids"].tolist())
pk_hr={s:v for s,v in _load_pk_set(f"{R}/pk_hr","pk_set").items() if s in test}
common=sorted(pk_hr); print(f"{len(common)} test boxes",flush=True)
def q(tag,s):
    p=f"{T}/q{tag}_{s}.pkl"; return _load_posterior(p) if os.path.exists(p) else None
def xfid(qh,qx,n=1500):
    ks=[]
    for sid in common:
        a=_sample(qh,pk_hr[sid],n); b=_sample(qx,pk_hr[sid],n)
        ks.append(kl_gauss(a.mean(0),a.std(0),b.mean(0),b.std(0)).mean())
    return float(np.mean(ks))
print("\nper-seed summary cross-fid KL (q_source seed s vs q_HR seed s):",flush=True)
for tag in TAGS:
    vals=[]
    for s in range(5):
        qh,qx=q("hr",s),q(tag,s)
        if qh is None or qx is None: continue
        torch.manual_seed(1000+s); vals.append(xfid(qh,qx))
    vals=np.array(vals)
    print(f"  {tag:8s}: {np.round(vals,3).tolist()}  -> mean {vals.mean():.3f} +/- {vals.std():.3f}",flush=True)
fl=[]
for i in range(5):
    for j in range(i+1,5):
        qi,qj=q("hr",i),q("hr",j)
        if qi is None or qj is None: continue
        torch.manual_seed(5); fl.append(xfid(qi,qj))
fl=np.array(fl); print(f"  HR-floor : {np.round(fl,3).tolist()}  -> mean {fl.mean():.3f} +/- {fl.std():.3f}",flush=True)
print("SEED_XFID_DONE",flush=True)
