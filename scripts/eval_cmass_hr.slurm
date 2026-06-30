#!/bin/bash
# CMASS HR reference (no model): HR count cubes → Pk(counts) → q_HR NDE + split sids.
# Runs in parallel with training.
#SBATCH --job-name=evalcmass_hr
#SBATCH --partition=general
#SBATCH --output=logs/%j_evalcmass_hr.log
#SBATCH --error=logs/%j_evalcmass_hr.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00

set -euo pipefail
cd /home/vkshirsa/srs-styled-map2map-galaxy
source /home/vkshirsa/venv/bin/activate
export PYTHONPATH=/home/vkshirsa/srs-styled-map2map-galaxy:${PYTHONPATH:-} OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
THETA=data/cmass_theta.npz
R=runs/patch_cmass; mkdir -p "$R" logs
SCR=/tmp/vkshirsa_cmass_${SLURM_JOB_ID}; mkdir -p "$SCR"
trap 'rm -rf "$SCR"' EXIT
df -h "$SCR" | tail -1 >&2

echo "=== HR count cubes (all sims) → scratch ==="
python infer_stitch_cmass.py --hr-only --output-dir "$SCR/hr_cubes" --split all --save-dtype float16
echo "=== Pk(HR) on overdensity (counts estimator) ==="
python -m analysis.power_spectrum --mode transformed --input "$SCR/hr_cubes" \
    --output "$R/pk_hr" --lbox 1000.0 --estimator counts --n-bins 32
rm -rf "$SCR/hr_cubes"

echo "=== split sids + q_HR train dir ==="
python - <<PY
import os, numpy as np
from data.patch_dataset_cmass import PatchPairDatasetCmass
tr=PatchPairDatasetCmass(split="train").ids; te=PatchPairDatasetCmass(split="test").ids
np.savez("$R/split_sids.npz", train_sids=np.array(tr), test_sids=np.array(te))
os.makedirs("$R/pk_hr_train", exist_ok=True)
for s in tr:
    src=os.path.abspath("$R/pk_hr/pk_set%d_transformed.npz"%s)
    dst="$R/pk_hr_train/pk_set%d_transformed.npz"%s
    if os.path.exists(src) and not os.path.exists(dst): os.symlink(src,dst)
print("train",len(tr),"test",len(te))
PY

echo "=== train q_HR ==="
python -m inference.nde --pk-dir "$R/pk_hr_train" --theta-npz "$THETA" \
    --out "$R/posterior_hr.pkl" --theta-low 0.0 --theta-high 1.5

echo "=== LR baseline (no model): Pk(LR) → q_LR → KL(q_HR‖q_LR) ==="
python infer_stitch_cmass.py --lr-only --output-dir "$SCR/lr_cubes" --split all --save-dtype float16
python -m analysis.power_spectrum --mode transformed --input "$SCR/lr_cubes" \
    --output "$R/pk_LRbase" --lbox 1000.0 --estimator counts --n-bins 32
rm -rf "$SCR/lr_cubes"
python - <<PY
import os, numpy as np
tr=np.load("$R/split_sids.npz")["train_sids"]
os.makedirs("$R/pk_LRbase_train", exist_ok=True)
for s in tr:
    src=os.path.abspath("$R/pk_LRbase/pk_set%d_transformed.npz"%s)
    dst="$R/pk_LRbase_train/pk_set%d_transformed.npz"%s
    if os.path.exists(src) and not os.path.exists(dst): os.symlink(src,dst)
PY
python -m inference.nde --pk-dir "$R/pk_LRbase_train" --theta-npz "$THETA" \
    --out "$R/posterior_LRbase.pkl" --theta-low 0.0 --theta-high 1.5 || echo "q_LR failed"
python evaluate.py --posterior-hr "$R/posterior_hr.pkl" --posterior-sr "$R/posterior_LRbase.pkl" \
    --pk-hr-dir "$R/pk_hr" --pk-sr-dir "$R/pk_LRbase" --pk-hr-prefix "pk_set" --pk-sr-prefix "pk_set" \
    --theta-npz "$THETA" --restrict-sids "$R/split_sids.npz" \
    --n-samples 2000 --output "$R/metrics_LRbase.npz" || echo "LR KL eval failed"
echo "=== HR reference + LR baseline done ==="
