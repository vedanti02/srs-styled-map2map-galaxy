#!/bin/bash
#SBATCH --job-name=datacheck
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --account=bapoczos
#SBATCH --output=logs/%j_datacheck.log
#SBATCH --error=logs/%j_datacheck.err
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
set -uo pipefail
cd /home/vkshirsa/srs-styled-map2map-galaxy
echo "node: $(hostname)"
echo "=== /data on compute node ==="; ls -ld /data /data/user_data/vkshirsa 2>&1 | head
echo "=== models intact? ==="
for p in /data/user_data/vkshirsa/cmass-ili/models/patch_cmass_A_nopk_fixcond/best.pt \
         /data/user_data/vkshirsa/cmass-ili/checkpoints/patch_cmass_A_nopk/best.pt \
         /data/group_data/universedata/cmass-ili/models/patch_cmass_A_nopk_fixcond/best.pt; do
  ls -la "$p" 2>&1 | awk '{print $5, $NF}'
done
echo "=== NOISE-STD verification (claim 1) ==="
source /home/vkshirsa/venv/bin/activate; export PYTHONPATH=$PWD
python3 - << 'PY'
import torch, os
paths=[("A_nopk (orig, scrambled)","/data/user_data/vkshirsa/cmass-ili/checkpoints/patch_cmass_A_nopk/best.pt"),
       ("A_nopk_fixcond (retrain, current)","/data/user_data/vkshirsa/cmass-ili/models/patch_cmass_A_nopk_fixcond/best.pt")]
for name,ck in paths:
    if not os.path.exists(ck): print(f"{name}: MISSING"); continue
    d=torch.load(ck,map_location="cpu",weights_only=False)["model"]
    nstd={k:v for k,v in d.items() if k.endswith(".std")}
    a=torch.cat([v.abs().flatten() for v in nstd.values()])
    w=torch.cat([v.abs().flatten() for k,v in d.items() if "weight" in k and v.dim()>1]).mean()
    print(f"{name}: {len(nstd)} noise-std params | |std| mean={a.mean():.5f} med={a.median():.5f} max={a.max():.5f} | conv|w| ref={w:.4f}")
PY
echo "DATACHECK_DONE"
