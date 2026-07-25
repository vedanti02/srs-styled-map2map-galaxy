"""Field-level cross-fidelity evaluation. For every trained CNN posterior
(q_hr/q_lr/q_sr, several seeds), run it on the HR TEST fields to get a Gaussian
posterior (mu,sigma) per box, then:
 (1) parameter recovery |mu-theta_true| (is the CNN working?),
 (2) cross-fid KL: KL(q_HR_ref(.|HR) || q_X(.|HR)) per box (analytic Gaussian KL),
 (3) paired ranking: over (q_HR_r, q_SR_i, q_LR_j) legit replicate combos, fraction
     of boxes SR closer to the HR reference than LR. (F24-style rigor, but these
     CNNs are separate-process so no same-process NDE bug.)
"""
import argparse, glob, os, re, numpy as np, torch
from models.field_cnn import FieldCNN
PROC="/data/group_data/universedata/cmass-ili/processed"

def kl_gauss(m1,s1,m2,s2,eps=1e-8):
    s1=s1**2+eps; s2=s2**2+eps
    return 0.5*(np.log(s2/s1)+(s1+(m1-m2)**2)/s2-1.0)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir",default="runs/patch_cmass/field_ckpts")
    ap.add_argument("--out",default="runs/patch_cmass/field_crossfid.npz")
    a=ap.parse_args()
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    theta=np.load("data/cmass_theta.npz")["theta"]
    te=sorted(int(d["idx"]) for d in np.load(f"{PROC}/test_list.npy",allow_pickle=True))
    LO=np.array([0.10,0.03,0.50,0.80,0.60],np.float32); HI=np.array([0.50,0.07,0.90,1.20,1.00],np.float32)
    Yt=(theta[te]-LO)/(HI-LO)                                   # true params, normalized
    # load HR test fields once (log1p)
    F=np.stack([np.log1p(np.load(f"{PROC}/{i:04d}_label.npy").astype(np.float32)) for i in te])  # (200,128^3)
    F=torch.tensor(F,device=dev).unsqueeze(1)                   # (200,1,128,128,128)

    posts={}                                                    # name -> (mu,sigma) (200,5)
    for ck in sorted(glob.glob(f"{a.ckpt_dir}/q*_s*.pt")):
        name=os.path.basename(ck)[:-3]
        d=torch.load(ck,map_location=dev,weights_only=False)
        net=FieldCNN().to(dev); net.load_state_dict(d["model"]); net.eval()
        mean,std=d["mean"],d["std"]
        mus=[]; sigs=[]
        with torch.no_grad():
            for k in range(0,len(te),16):
                xb=(F[k:k+16]-mean)/std
                mu,sg=net(xb); mus.append(mu.cpu().numpy()); sigs.append(sg.cpu().numpy())
        posts[name]=(np.concatenate(mus),np.concatenate(sigs))
        rec=np.abs(posts[name][0]-Yt).mean()
        print(f"{name}: recovery |mu-theta|(norm)={rec:.3f}",flush=True)

    def members(src): return {k:v for k,v in posts.items() if k.startswith(f"q{src}_")}
    HR=members("hr"); SR=members("sr"); LR=members("lr")
    # (2) cross-fid KL means (each X vs each HR ref)
    print("\n=== cross-fid KL means (X on HR vs HR-ref on HR) ===",flush=True)
    for src,mem in [("SR",SR),("LR",LR),("HR",HR)]:
        vals=[]
        for rk,(rm,rs) in HR.items():
            for xk,(xm,xs) in mem.items():
                if src=="HR" and xk==rk: continue
                vals.append(kl_gauss(rm,rs,xm,xs).mean())
        print(f"  {src}: mean cross-fid KL over refs = {np.mean(vals):.4f} +/- {np.std(vals):.4f}",flush=True)
    # (3) paired ranking SR vs LR
    print("\n=== paired ranking: SR closer to HR-ref than LR, per box ===",flush=True)
    fr=[]
    for rk,(rm,rs) in HR.items():
        for sk,(sm,ss) in SR.items():
            for lk,(lm,ls) in LR.items():
                dS=kl_gauss(rm,rs,sm,ss).mean(1); dL=kl_gauss(rm,rs,lm,ls).mean(1)
                fr.append(np.mean(dS<dL))
    fr=np.array(fr)
    print(f"  SR<LR fraction over {len(fr)} (ref,SR,LR) combos: min={fr.min()*100:.0f}% "
          f"max={fr.max()*100:.0f}% mean={fr.mean()*100:.0f}%",flush=True)
    # HR self floor
    hr_pw=[]
    ks=list(HR)
    for i in range(len(ks)):
        for j in range(len(ks)):
            if i<j:
                a1=HR[ks[i]]; a2=HR[ks[j]]; hr_pw.append(kl_gauss(a1[0],a1[1],a2[0],a2[1]).mean())
    print(f"\n  HR self floor (pairwise CNN retrain KL): {np.round(hr_pw,4).tolist()} mean {np.mean(hr_pw):.4f}",flush=True)
    np.savez(a.out, frac=fr, hr_floor=np.array(hr_pw))
    print("FIELD_EVAL_DONE",flush=True)
if __name__=="__main__": main()
