"""Generate SR count fields (uint8, clip 0-200) for the TRAIN split, for
field-level cosmology inference. q_SR is trained on these; it is evaluated on HR
test fields (which already exist), so we only need SR TRAIN fields here."""
import argparse, os, numpy as np, torch
from data.patch_dataset_cmass import (PatchPairDatasetCmass, extract_patch,
    stitch_patches, crop_interior, to_counts, PATCH, N_PATCHES)
from map2map.models.styled_srsgan import G_correct

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ckpt",required=True); ap.add_argument("--out",required=True)
    ap.add_argument("--split",default="train"); ap.add_argument("--mode",default="naive")
    ap.add_argument("--theta-mode",default="true",choices=["true","fiducial"],
                    help="true = per-box cosmology (oracle upper bound); "
                         "fiducial = prior-mean for ALL boxes (deployment-faithful, no theta leak)")
    a=ap.parse_args()
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(a.out,exist_ok=True)
    FIDUCIAL=np.array([0.30,0.05,0.70,1.00,0.80],np.float32)  # prior mean (Om,Ob,h,ns,s8)
    print(f"theta-mode={a.theta_mode}"+(f" fiducial={FIDUCIAL.tolist()}" if a.theta_mode=="fiducial" else ""),flush=True)
    ck=torch.load(a.ckpt,map_location=dev,weights_only=False); s=ck.get("args",{}) or {}
    cb,nb=s.get("chan_base_g",128),s.get("num_blocks",4); tf=s.get("transform","log1p")
    pad=0 if a.mode=="naive" else s.get("pad",8)
    G=G_correct(1,1,5,chan_base=cb,num_blocks=nb).to(dev); G.load_state_dict(ck["model"]); G.eval()
    ds=PatchPairDatasetCmass(split=a.split,pad=pad,transform=tf)
    rng=np.random.default_rng(0)
    nl=[torch.from_numpy(rng.standard_normal((1,PATCH+2*pad,)*1+(PATCH+2*pad,PATCH+2*pad)).astype(np.float32)).to(dev) for _ in range(8)]
    # fix noise shape
    nl=[torch.from_numpy(rng.standard_normal((1,PATCH+2*pad,PATCH+2*pad,PATCH+2*pad)).astype(np.float32)).to(dev) for _ in range(8)]
    import time; t0=time.time()
    with torch.no_grad():
        for n,idx in enumerate(ds.ids):
            of=f"{a.out}/sr_{idx}.npy"
            if os.path.exists(of): continue
            lr,_=ds.load_boxes(idx)
            xb=torch.from_numpy(np.stack([extract_patch(lr,p,pad) for p in range(N_PATCHES)])).to(dev)
            th_vec=ds.theta[idx] if a.theta_mode=="true" else FIDUCIAL
            th=torch.from_numpy(np.asarray(th_vec,np.float32)).unsqueeze(0).expand(N_PATCHES,-1).to(dev)
            fake=crop_interior(G(xb,th,nl),pad).cpu().numpy()
            sr=to_counts(stitch_patches(fake),tf,ds.scale)[0]  # (128,128,128)
            sr=np.clip(np.nan_to_num(sr,nan=0.,posinf=200.,neginf=0.),0,200).round().astype(np.uint8)
            np.save(of,sr)
            if (n+1)%100==0: print(f"{n+1}/{len(ds.ids)} {time.time()-t0:.0f}s",flush=True)
    print("GEN_SR_DONE",flush=True)
if __name__=="__main__": main()
