"""Compute per-box P(k) (count overdensity) for SR count fields (sr_{idx}.npy),
writing pk_set{idx}_transformed.npz (k, pk, n_modes) in the format the NDE reads.
Used to build the summary-level cross-fidelity for the retrained (conditioned) SR."""
import numpy as np, os, sys, time
sys.path.insert(0, ".")
from analysis.power_spectrum import cube_pk_counts
SRDIR=sys.argv[1]; OUT=sys.argv[2]; SPLIT=sys.argv[3] if len(sys.argv)>3 else "train"
from paths import PROCESSED as PROC
os.makedirs(OUT, exist_ok=True)
ids=sorted(int(d["idx"]) for d in np.load(f"{PROC}/{SPLIT}_list.npy", allow_pickle=True))
print(f"P(k) for {len(ids)} {SPLIT} SR boxes: {SRDIR} -> {OUT}", flush=True)
t0=time.time(); n_done=0
for j,idx in enumerate(ids):
    f=f"{SRDIR}/sr_{idx}.npy"; of=f"{OUT}/pk_set{idx}_transformed.npz"
    if not os.path.exists(f) or os.path.exists(of): continue
    cube=np.load(f).astype(np.float32)
    k,pk,nm=cube_pk_counts(cube, 1000.0, n_bins=32)
    np.savez(of, k=k, pk=pk, n_modes=nm); n_done+=1
    if (j+1)%200==0: print(f"  {j+1}/{len(ids)} {time.time()-t0:.0f}s", flush=True)
print(f"PK_SR_DONE {n_done} written", flush=True)
