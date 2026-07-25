"""Can the CNN memorize 32 fields? (broken-net vs weak-signal test). No aug, no
dropout, MSE, 300 epochs. If train recovery -> ~0, the architecture can fit."""
import sys, numpy as np, torch
sys.path.insert(0,".")
from models.field_cnn import FieldCNN
PROC="/data/group_data/universedata/cmass-ili/processed"
dev=torch.device("cuda")
theta=np.load("data/cmass_theta.npz")["theta"]
LO=np.array([0.10,0.03,0.50,0.80,0.60],np.float32); HI=np.array([0.50,0.07,0.90,1.20,1.00],np.float32)
tr=sorted(int(x["idx"]) for x in np.load(f"{PROC}/train_list.npy",allow_pickle=True))[:32]
X=np.stack([np.log1p(np.load(f"{PROC}/{i:04d}_label.npy").astype(np.float32)) for i in tr])
mean,std=float(X.mean()),float(X.std())+1e-8
Xt=torch.tensor((X-mean)/std,device=dev).unsqueeze(1)
Yt=torch.tensor(((theta[tr]-LO)/(HI-LO)).astype(np.float32),device=dev)
net=FieldCNN().to(dev)
# disable dropout for pure memorization test
for m in net.modules():
    if isinstance(m,torch.nn.Dropout): m.p=0.0
opt=torch.optim.Adam(net.parameters(),lr=1e-3)
print(f"overfit test: 32 fields, marginal={float((Yt-Yt.mean(0)).abs().mean()):.3f}",flush=True)
for ep in range(300):
    net.train()
    for k in range(0,32,8):
        mu,_=net(Xt[k:k+8]); loss=((mu-Yt[k:k+8])**2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    if (ep+1)%25==0:
        net.eval()
        with torch.no_grad(): mu,_=net(Xt); tr_rec=float((mu-Yt).abs().mean()); ms=float(mu.std(0).mean())
        print(f"ep{ep} train_recov={tr_rec:.3f} std(mu)={ms:.3f} loss={float(loss):.4f}",flush=True)
print("OVERFIT_DONE",flush=True)
