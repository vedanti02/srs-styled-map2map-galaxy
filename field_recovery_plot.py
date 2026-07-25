"""Before/after the label fix: field-CNN predicted Omega_m vs true, on HR test
boxes. Buggy CNN (trained on scrambled labels) -> flat/uncorrelated; corrected
CNN -> diagonal, corr ~0.90. The clearest visual that the field carries cosmology
and the inference works once labels are right."""
import numpy as np, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from models.field_cnn import FieldCNN
PROC="/data/group_data/universedata/cmass-ili/processed"
dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
theta=np.load("data/cmass_theta.npz")["theta"]                 # corrected (field order)
LO=np.array([0.10,0.03,0.50,0.80,0.60],np.float32); HI=np.array([0.50,0.07,0.90,1.20,1.00],np.float32)
te=sorted(int(d["idx"]) for d in np.load(f"{PROC}/test_list.npy",allow_pickle=True))
Yt=(theta[te]-LO)/(HI-LO)                                       # normalized truth
F=np.stack([np.log1p(np.load(f"{PROC}/{i:04d}_label.npy").astype(np.float32)) for i in te])
F=torch.tensor(F,device=dev).unsqueeze(1)

def predict(ckpt):
    d=torch.load(ckpt,map_location=dev,weights_only=False)
    net=FieldCNN().to(dev); net.load_state_dict(d["model"]); net.eval()
    mus=[]
    with torch.no_grad():
        for k in range(0,len(te),16):
            mu,_=net((F[k:k+16]-d["mean"])/d["std"]); mus.append(mu.cpu().numpy())
    return np.concatenate(mus)

# param 0 = Omega_m ; de-normalize to physical for readable axes
def denorm(v,i): return v*(HI[i]-LO[i])+LO[i]
# before = GroupNorm scrambled-label ckpt (qhr_s7, loads into current arch);
# after  = GroupNorm corrected-label ckpt (qhr_s100).
cases=[("runs/patch_cmass/field_ckpts_buggy/qhr_s7.pt","before fix (mislabeled)","C3"),
       ("runs/patch_cmass/field_ckpts/qhr_s100.pt","after fix (corrected)","C0")]
fig,axs=plt.subplots(1,2,figsize=(9,4.8),sharex=True,sharey=True)
tru=denorm(Yt[:,0],0)
for ax,(ck,title,col) in zip(axs,cases):
    mu=predict(ck); pred=denorm(mu[:,0],0)      # let a load error raise loudly
    sd=float(pred.std()); sdt=float(tru.std())
    r=np.corrcoef(pred,tru)[0,1] if sd>1e-6 else 0.0
    print(f"{title}: std(pred)={sd:.4f} std(true)={sdt:.4f} ratio={sd/sdt:.3f} r={r:+.3f}",flush=True)
    # near-constant output -> correlation is numerically meaningless; label it honestly
    rlab=(f"predicts ~constant (spread {sd/sdt:.0%} of true)" if sd/sdt<0.2
          else f"r={r:+.2f}")
    ax.scatter(tru,pred,s=14,alpha=0.55,color=col,edgecolors="none")
    ax.plot([0.10,0.50],[0.10,0.50],"k--",lw=1)
    ax.set_title(f"{title}\n{rlab}",fontsize=10)
    ax.set_xlabel(r"true $\Omega_m$")
    ax.set_xlim(0.10,0.50); ax.set_ylim(0.08,0.52); ax.set_aspect("equal",adjustable="box")
axs[0].set_ylabel(r"predicted $\Omega_m$")
fig.suptitle(r"Field-CNN recovery of $\Omega_m$ on HR test boxes, before and after the label fix",
             y=1.02,fontsize=12)
plt.tight_layout()
plt.savefig("figures_cmass/field_recovery_Om_cmass.png",dpi=140,bbox_inches="tight")
print("wrote figures_cmass/field_recovery_Om_cmass.png")
print("RECOVERY_PLOT_DONE")
