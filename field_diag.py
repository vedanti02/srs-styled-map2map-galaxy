"""Overfit-vs-weak-signal diagnostic on a TRAINED checkpoint.
Recovery + corr(mu,theta) on TRAIN and TEST fields:
 - train good & test bad  -> OVERFIT (fixable: more data / reg / smaller net)
 - train bad & test bad    -> weak signal or optimization failure (field ~uninformative)
"""
import sys, numpy as np, torch
sys.path.insert(0,".")
from models.field_cnn import FieldCNN
PROC="/data/group_data/universedata/cmass-ili/processed"
dev=torch.device("cuda")
theta=np.load("data/cmass_theta.npz")["theta"]
LO=np.array([0.10,0.03,0.50,0.80,0.60]); HI=np.array([0.50,0.07,0.90,1.20,1.00])
CK=sys.argv[1] if len(sys.argv)>1 else "runs/patch_cmass/field_ckpts/qhr_s1.pt"
d=torch.load(CK,map_location=dev,weights_only=False)
net=FieldCNN().to(dev); net.load_state_dict(d["model"]); net.eval()
print(f"ckpt {CK} epoch {d['epoch']} mean={d['mean']:.3f} std={d['std']:.3f}")
names=["Om","Ob","h","ns","s8"]
def run(ids, tag):
    F=np.stack([np.log1p(np.load(f"{PROC}/{i:04d}_label.npy").astype(np.float32)) for i in ids])
    F=torch.tensor(F,device=dev).unsqueeze(1); Y=(theta[ids]-LO)/(HI-LO)
    mus=[]
    with torch.no_grad():
        for k in range(0,len(ids),16):
            mu,_=net((F[k:k+16]-d["mean"])/d["std"]); mus.append(mu.cpu().numpy())
    mu=np.concatenate(mus)
    print(f"--- {tag} (n={len(ids)}) ---")
    for i,nm in enumerate(names):
        r=np.abs(mu[:,i]-Y[:,i]).mean(); c=np.corrcoef(mu[:,i],Y[:,i])[0,1]
        print(f"  {nm:4} recov={r:.3f} corr={c:+.3f} std(mu)={mu[:,i].std():.3f}")
    print(f"  mean recov={np.abs(mu-Y).mean():.3f}  (marginal={np.abs(Y-Y.mean(0)).mean():.3f})")
tr=sorted(int(x["idx"]) for x in np.load(f"{PROC}/train_list.npy",allow_pickle=True))[:150]
te=sorted(int(x["idx"]) for x in np.load(f"{PROC}/test_list.npy",allow_pickle=True))
run(tr,"TRAIN subset"); run(te,"TEST")
print("DIAG2_DONE")
