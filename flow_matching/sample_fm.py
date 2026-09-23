"""Generate SR count fields from a flow-matching checkpoint, in the layout every existing
evaluation script reads (gen_sr_fields.py convention):
    <out>/sr_{idx}.npy               uint8 (128,128,128) counts, draw 0
    <out>/sr_{idx}_d{k}.npy          draws k>=1 (only with --n-draws > 1)
    <out>/set{idx}_transformed.npy   symlink -> sr_{idx}.npy (bispectrum_cmass --mode dirs)
Skips existing files, so a preempted job resumes. Seeds are per (idx, draw) -> reproducible.

  python -m flow_matching.sample_fm --ckpt .../best.pt --out .../sr_fields/fm_gauss --split val --n-draws 8
"""
import argparse, os, time
import numpy as np
import torch

from flow_matching.data import CountPatchDataset
from flow_matching.common import build_from_ckpt, generate_box


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    p.add_argument("--n-draws", type=int, default=1)
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--method", default="heun", choices=["euler", "heun", "sde"])
    p.add_argument("--sde-gamma", type=float, default=1.0, help="sde only: noise scale s_t = gamma (1-t)")
    p.add_argument("--lowk-kc", type=int, default=0,
                   help="diagnostic: copy LF modes with |k| < kc (box fundamental units) into the output")
    p.add_argument("--max-sims", type=int, default=0)
    p.add_argument("--decode-shift", type=float, default=0.0,
                   help="dequantised models: n = floor(z - shift); calibrated ~0.07 for fm_gauss (see diag_decoder)")
    p.add_argument("--amp", default="bf16", choices=["none", "bf16"])
    p.add_argument("--no-ema", action="store_true", help="use raw (non-EMA) weights")
    return p.parse_args()


def main():
    a = parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(a.out, exist_ok=True)
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    if a.no_ema:
        ck["ema"] = None
    net, space, interp, args = build_from_ckpt(ck, dev)
    amp_dtype = torch.bfloat16 if (a.amp == "bf16" and dev.type == "cuda") else None
    ds = CountPatchDataset(a.split, pad=0, cache=False, max_sets=a.max_sims)
    print(f"sample: ckpt={a.ckpt} epoch={ck.get('epoch')} {interp} {space} cond={args['cond']} | "
          f"{len(ds)} {a.split} boxes x {a.n_draws} draws | {a.method} {a.steps} steps | lowk_kc={a.lowk_kc} decode_shift={a.decode_shift}", flush=True)
    t0 = time.time(); n_done = 0
    for n, idx in enumerate(ds.ids):
        names = [f"{a.out}/sr_{idx}.npy"] + [f"{a.out}/sr_{idx}_d{k}.npy" for k in range(1, a.n_draws)]
        if all(os.path.exists(f) for f in names):
            continue
        lr, _ = ds.load_boxes(idx)
        srs = generate_box(net, space, interp, lr, dev, n_draws=a.n_draws, steps=a.steps, method=a.method,
                           seed=a.base_seed * 7_919 + idx, cond=args["cond"], lowk_kc=a.lowk_kc, amp_dtype=amp_dtype,
                           decode_shift=a.decode_shift, sde_gamma=a.sde_gamma)
        for f, sr in zip(names, srs):
            np.save(f, np.clip(sr[0], 0, 255).astype(np.uint8))
        link = f"{a.out}/set{idx}_transformed.npy"
        if not os.path.lexists(link):
            os.symlink(os.path.basename(names[0]), link)
        n_done += 1
        if n_done % 50 == 0 or n + 1 == len(ds.ids):
            print(f"  {n+1}/{len(ds.ids)} boxes {time.time()-t0:.0f}s", flush=True)
    print(f"SAMPLE_DONE {n_done} boxes written ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
