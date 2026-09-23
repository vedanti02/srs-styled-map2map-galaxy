"""Show EXACTLY what the model is fed: the data source, one real LR/HR box with actual voxel
values, the theta vector, and how tensor DIMENSIONS change: raw counts -> log1p -> 8 patches
-> generator -> stitched box -> integer counts. Real weights + real data (needs /data)."""
import sys, numpy as np, torch
sys.path.insert(0,".")
from data.patch_dataset_cmass import (PatchPairDatasetCmass, extract_patch, stitch_patches,
    to_counts, PATCH, N_PATCHES, PROCESSED, NBODY)
from map2map.models.styled_srsgan import G_correct
np.set_printoptions(precision=4, suppress=True, linewidth=120)
CKPT=sys.argv[1]; dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("="*72); print("1) WHERE THE DATA COMES FROM"); print("="*72)
print(f"count fields : {PROCESSED}/{{idx:04d}}_input.npy = LR (FastPM), _label.npy = HR (Quijote N-body)")
print(f"cosmology    : {NBODY}/{{idx}}/config.yaml -> 5 params, cached in data/cmass_theta.npz")
print("2000 paired sims, each a 128^3 halo-count grid on a periodic 1000 Mpc/h box (7.8 Mpc/h voxels)")

ds=PatchPairDatasetCmass(split="test",pad=0,transform="log1p")
ds_raw=PatchPairDatasetCmass(split="test",pad=0,normalize_inputs=False)
idx=ds.ids[0]; lr_raw,hr_raw=ds_raw.load_boxes(idx); theta=ds.theta[idx]

print("\n"+"="*72); print(f"2) ONE REAL BOX (sim idx={idx}): what a 'halo count field' actually is"); print("="*72)
for name,b in [("LR input (FastPM) ",lr_raw),("HR label (Quijote)",hr_raw)]:
    b0=b[0]; vals,cnt=np.unique(b0,return_counts=True)
    print(f"{name}: shape={b.shape} dtype={b.dtype}  min={b0.min():.0f} max={b0.max():.0f} mean={b0.mean():.4f}  total halos={b0.sum():.0f}")
    print("    how many voxels hold N halos: "+"  ".join(f"N={int(v)}:{c}" for v,c in zip(vals[:7],cnt[:7]))+" ...")
print("\nActual voxel values, a 6x6 slab at z=64, x,y=0..5 (each number = halos in that 7.8 Mpc/h cell):")
print("LR:\n"+str(lr_raw[0,64,0:6,0:6].astype(int))); print("HR:\n"+str(hr_raw[0,64,0:6,0:6].astype(int)))
print(f"\nTHETA for this box (Om, Ob, h, ns, s8) = {theta}   <- fed raw, no normalization")

print("\n"+"="*72); print("3) DIMENSION FLOW"); print("="*72)
lr_m,_=ds.load_boxes(idx)
v=lr_raw[0,64,0,0]; print(f"raw LR counts             {lr_raw.shape}    voxel[64,0,0] = {v:.0f} halos")
print(f"log1p -> model space      {lr_m.shape}    same voxel -> log1p({v:.0f}) = {lr_m[0,64,0,0]:.4f}")
patches=np.stack([extract_patch(lr_m,q,0) for q in range(N_PATCHES)])
print(f"cut into 2x2x2 patches    {patches.shape}   [patch, channel, x, y, z]  (8 cubes of 64^3)")
xb=torch.from_numpy(patches).to(dev)
th=torch.from_numpy(theta).unsqueeze(0).expand(N_PATCHES,-1).to(dev)
print(f"theta, one row per patch  {tuple(th.shape)}          (same 5 numbers copied 8 times)")
ck=torch.load(CKPT,map_location=dev,weights_only=False); s=ck.get("args",{}) or {}
G=G_correct(1,1,5,chan_base=s.get("chan_base_g",128),num_blocks=s.get("num_blocks",4)).to(dev)
G.load_state_dict(ck["model"]); G.eval()
rng=np.random.default_rng(0)
nl=[torch.from_numpy(rng.standard_normal((1,PATCH,PATCH,PATCH)).astype(np.float32)).to(dev) for _ in range(8)]
print(f"noise list                {len(nl)} x {tuple(nl[0].shape)}   (one per noise-inject layer)")
print("\n--- inside the generator ---")
def hook(name):
    def f(m,inp,out):
        o=out[0] if isinstance(out,tuple) else out; print(f"  {name:30s} -> {tuple(o.shape)}")
    return f
G.block0.register_forward_hook(hook("block0: 1 -> 128 channels"))
for b,blk in enumerate(G.blocks): blk.register_forward_hook(hook(f"HBlock_const[{b}] features"))
sc=G.block0[0].style_block(th)
print(f"  theta {tuple(th.shape)} --Linear(5->{sc.shape[1]})--> {tuple(sc.shape)} per-channel scales that multiply the first conv's kernel")
print(f"     scales, patch 0, first 6 of {sc.shape[1]}: {sc[0,:6].detach().cpu().numpy()}   (started at 1.0 before training)")
with torch.no_grad(): fake=G(xb,th,nl)
print(f"generator output          {tuple(fake.shape)}    (= input + predicted residual, model space)")
cp=to_counts(fake.cpu().numpy(),"log1p",ds.scale)
print(f"expm1 -> counts/patch     {cp.shape}    patch0 voxel[0,0,0] = {cp[0,0,0,0,0]:.3f} (continuous)")
sr=stitch_patches(cp); print(f"stitch 8 patches -> box   {sr.shape}")
sr_i=np.clip(np.nan_to_num(sr),0,200).round().astype(np.uint8)
print(f"round -> integer counts   {sr_i.shape} {sr_i.dtype}  min={sr_i.min()} max={sr_i.max()} mean={sr_i.mean():.4f} total={sr_i.sum()}")
print("\nSame 6x6 slab, SR output (z=64):\n"+str(sr_i[0,64,0:6,0:6].astype(int)))
print("\nSHOW_INPUT_DONE",flush=True)
