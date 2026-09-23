"""Train a 3D CNN moment network (field -> posterior over 5 cosmo params) for a
given field source (hr/lr/sr). For the cross-fidelity test each model learns its
own input normalization; at eval it is applied to HR fields (distribution shift).
Loads all train fields into RAM (float16 log1p) for speed."""
import argparse, os, time, numpy as np, torch
from torch.utils.data import DataLoader, TensorDataset
from models.field_cnn import FieldCNN, moment_loss

from paths import PROCESSED as PROC
LO=np.array([0.10,0.03,0.50,0.80,0.60],np.float32)   # Quijote LH prior lows
HI=np.array([0.50,0.07,0.90,1.20,1.00],np.float32)   # highs

def load_field(source, idx, sr_dir):
    if source=="hr": n=np.load(f"{PROC}/{idx:04d}_label.npy")
    elif source=="lr": n=np.load(f"{PROC}/{idx:04d}_input.npy")
    else: n=np.load(f"{sr_dir}/sr_{idx}.npy")
    return np.log1p(n.astype(np.float32))            # model space, tames sparse tail

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",required=True,choices=["hr","lr","sr"])
    ap.add_argument("--sr-dir",default="runs/patch_cmass/sr_fields_Anopk")
    ap.add_argument("--out",required=True); ap.add_argument("--seed",type=int,default=0)
    ap.add_argument("--epochs",type=int,default=60); ap.add_argument("--bs",type=int,default=8)
    ap.add_argument("--lr",type=float,default=1e-3); ap.add_argument("--warmup",type=int,default=15)
    a=ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev=torch.device("cuda")
    theta=np.load("data/cmass_theta.npz")["theta"]
    tr=sorted(int(d["idx"]) for d in np.load(f"{PROC}/train_list.npy",allow_pickle=True))
    va=sorted(int(d["idx"]) for d in np.load(f"{PROC}/val_list.npy",allow_pickle=True))[:40]
    print(f"source={a.source} seed={a.seed}  train={len(tr)} val={len(va)}",flush=True)

    t0=time.time()
    X=np.empty((len(tr),128,128,128),np.float16)
    for k,idx in enumerate(tr):
        X[k]=load_field(a.source,idx,a.sr_dir).astype(np.float16)
        if (k+1)%400==0: print(f"  loaded {k+1}/{len(tr)} {time.time()-t0:.0f}s",flush=True)
    mean=float(X.mean(dtype=np.float64)); std=float(X.std(dtype=np.float64))+1e-8  # float64 accum (float16 sum overflows)
    Y=((theta[tr]-LO)/(HI-LO)).astype(np.float32)         # params -> [0,1]
    Xv=np.stack([load_field(a.source,i,a.sr_dir) for i in va]).astype(np.float32)
    Yv=torch.tensor(((theta[va]-LO)/(HI-LO)).astype(np.float32),device=dev)
    Xv=torch.tensor(((Xv-mean)/std),device=dev).unsqueeze(1)
    print(f"loaded+normed ({time.time()-t0:.0f}s) mean={mean:.3f} std={std:.3f}",flush=True)

    Xt=torch.from_numpy(X); Yt=torch.from_numpy(Y)
    dl=DataLoader(TensorDataset(Xt,Yt),batch_size=a.bs,shuffle=True,num_workers=4,pin_memory=True,drop_last=True)
    net=FieldCNN().to(dev); opt=torch.optim.Adam(net.parameters(),lr=a.lr,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.epochs)
    best=1e9
    for ep in range(a.epochs):
        net.train(); te=time.time()
        for xb,yb in dl:
            xb=xb.float().to(dev,non_blocking=True).unsqueeze(1); yb=yb.to(dev,non_blocking=True)
            # random axis flips (cheap symmetry augmentation) on spatial axes
            for ax in (2,3,4):
                if np.random.rand()<0.5: xb=torch.flip(xb,[ax])
            if np.random.rand()<0.5: xb=torch.rot90(xb,int(np.random.randint(1,4)),[2,3])
            if np.random.rand()<0.5: xb=torch.rot90(xb,int(np.random.randint(1,4)),[3,4])
            xb=(xb-mean)/std
            mu,sig=net(xb)
            # MSE warmup forces mu to track theta before the variance term
            # can take the 'predict marginal' shortcut (moment-net collapse fix)
            loss=((yb-mu)**2).mean() if ep<a.warmup else moment_loss(mu,sig,yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            mv,sv=net(Xv); vloss=float(moment_loss(mv,sv,Yv))
            verr=float((mv-Yv).abs().mean())
        if vloss<best:
            best=vloss
            torch.save({"model":net.state_dict(),"mean":mean,"std":std,
                        "LO":LO,"HI":HI,"source":a.source,"seed":a.seed,"epoch":ep},a.out)
        if (ep+1)%5==0 or ep==0:
            print(f"ep{ep} {time.time()-te:.0f}s vloss={vloss:.3f} verr(norm)={verr:.3f} best={best:.3f}",flush=True)
    print(f"DONE source={a.source} seed={a.seed} best_vloss={best:.3f} -> {a.out}",flush=True)
if __name__=="__main__": main()
