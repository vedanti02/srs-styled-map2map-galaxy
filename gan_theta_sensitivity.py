"""Does the A_nopk GAN actually use its theta (style) conditioning? It was TRAINED
with scrambled (effectively random) theta labels, so it may have learned to ignore
theta. If G(LR, theta_correct) ~ G(LR, theta_scrambled), the existing SR is valid
as-is and no GAN retrain is needed for a correct-label cross-fidelity. Measures the
relative change in the SR count field under different theta conditionings."""
import numpy as np, torch, time
from data.patch_dataset_cmass import (PatchPairDatasetCmass, extract_patch,
    stitch_patches, crop_interior, to_counts, PATCH, N_PATCHES)
from map2map.models.styled_srsgan import G_correct

dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
ck=torch.load("checkpoints/patch_cmass_A_nopk/best.pt",map_location=dev,weights_only=False)
s=ck.get("args",{}) or {}
cb,nb=s.get("chan_base_g",128),s.get("num_blocks",4); tf=s.get("transform","log1p")
pad=0
G=G_correct(1,1,5,chan_base=cb,num_blocks=nb).to(dev); G.load_state_dict(ck["model"]); G.eval()
ds=PatchPairDatasetCmass(split="test",pad=pad,transform=tf)
theta_good=np.load("data/cmass_theta.npz")["theta"]              # corrected (field order)
theta_bug =np.load("data/cmass_theta_numeric_buggy.npz")["theta"] # scrambled (used in training)
rng=np.random.default_rng(0)
nl=[torch.from_numpy(rng.standard_normal((1,PATCH,PATCH,PATCH)).astype(np.float32)).to(dev) for _ in range(8)]

def gen(lr, th5):
    xb=torch.from_numpy(np.stack([extract_patch(lr,p,pad) for p in range(N_PATCHES)])).to(dev)
    th=torch.from_numpy(th5.astype(np.float32)).unsqueeze(0).expand(N_PATCHES,-1).to(dev)
    fake=crop_interior(G(xb,th,nl),pad).cpu().numpy()
    return to_counts(stitch_patches(fake),tf,ds.scale)[0]

def reldiff(a,b): return float(np.linalg.norm(a-b)/(np.linalg.norm(a)+1e-8))

ids=list(ds.ids)[:5]
print(f"boxes={ids}  (SR relative change under different theta conditioning)")
print(f"{'idx':>5} {'corr_vs_scram':>13} {'corr_vs_extreme':>15} {'corr_vs_zero':>12} {'scram_vs_LRpair':>15}")
with torch.no_grad():
    for idx in ids:
        lr,_=ds.load_boxes(idx)
        tg=theta_good[idx]; tb=theta_bug[idx]
        extreme=np.array([0.5,0.07,0.9,1.2,1.0],np.float32) if tg[0]<0.3 else np.array([0.1,0.03,0.5,0.8,0.6],np.float32)
        sr_g=gen(lr,tg); sr_b=gen(lr,tb); sr_e=gen(lr,extreme); sr_z=gen(lr,np.zeros(5,np.float32))
        print(f"{idx:>5} {reldiff(sr_g,sr_b):>13.4f} {reldiff(sr_g,sr_e):>15.4f} {reldiff(sr_g,sr_z):>12.4f}",flush=True)
print("\nINTERPRET: reldiff << 0.01 for all -> G IGNORES theta (trained on noise labels)")
print("           -> existing SR is valid, no retrain needed for correct-label cross-fid.")
print("           reldiff >> 0.05 -> G uses theta -> scrambled conditioning corrupted SR -> retrain.")
print("SENS_DONE")
