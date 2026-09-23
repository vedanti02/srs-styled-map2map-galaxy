#!/bin/bash
#SBATCH --job-name=datacheck
#SBATCH --partition=ghx4
#SBATCH --account=bdne-dtai-gh
#SBATCH --output=logs/%j_datacheck.log
#SBATCH --error=logs/%j_datacheck.err
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"; source scripts/env_delta.sh
echo "node: $(hostname)"
echo "=== /data on compute node ==="; ls -ld /data /data/user_data/vkshirsa 2>&1 | head
echo "=== models intact? ==="
for p in $SRS_CMASS_ROOT/models/patch_cmass_A_nopk_fixcond/best.pt \
         $SRS_CMASS_ROOT/checkpoints/patch_cmass_A_nopk/best.pt \
         /data/group_data/universedata/cmass-ili/models/patch_cmass_A_nopk_fixcond/best.pt; do
  ls -la "$p" 2>&1 | awk '{print $5, $NF}'
done
echo "=== NOISE-STD verification (claim 1) ==="
export PYTHONPATH=$PWD
python3 - << 'PY'
import torch, os
paths=[("A_nopk (orig, scrambled)","$SRS_CMASS_ROOT/checkpoints/patch_cmass_A_nopk/best.pt"),
       ("A_nopk_fixcond (retrain, current)","$SRS_CMASS_ROOT/models/patch_cmass_A_nopk_fixcond/best.pt")]
for name,ck in paths:
    if not os.path.exists(ck): print(f"{name}: MISSING"); continue
    d=torch.load(ck,map_location="cpu",weights_only=False)["model"]
    nstd={k:v for k,v in d.items() if k.endswith(".std")}
    a=torch.cat([v.abs().flatten() for v in nstd.values()])
    w=torch.cat([v.abs().flatten() for k,v in d.items() if "weight" in k and v.dim()>1]).mean()
    print(f"{name}: {len(nstd)} noise-std params | |std| mean={a.mean():.5f} med={a.median():.5f} max={a.max():.5f} | conv|w| ref={w:.4f}")
PY
echo "DATACHECK_DONE"
