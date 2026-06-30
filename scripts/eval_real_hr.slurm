#!/bin/bash
# Build the HR reference (no trained model needed): assembled HR 128³ cubes → Pk →
# q_HR NDE → split sids. Runs in parallel with training.
#SBATCH --job-name=evalreal_hr
#SBATCH --partition=general
#SBATCH --output=logs/%j_evalreal_hr.log
#SBATCH --error=logs/%j_evalreal_hr.err
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
STITCH=/data/group_data/universedata/lagrangian_output_64/stitched
R=runs/patch_real; mkdir -p "$R" logs
# big cubes go to node-local scratch (home NFS is small); only tiny Pk/metrics return to home
SCR=/tmp/vkshirsa_real_${SLURM_JOB_ID}; mkdir -p "$SCR"
trap 'rm -rf "$SCR"' EXIT
df -h "$SCR" | tail -1 >&2

echo "=== assemble HR 128³ cubes (all clean sims) → scratch ==="
python infer_stitch_real.py --hr-only --output-dir "$SCR/hr_cubes" --split all --save-dtype float16
echo "=== Pk(HR) ==="
python -m analysis.power_spectrum --mode transformed --input "$SCR/hr_cubes" \
    --output "$R/pk_hr" --lbox 1000.0 --estimator div --n-bins 32
rm -rf "$SCR/hr_cubes"
echo "=== split sids + q_HR train dir ==="
python - <<PY
import os, numpy as np
from data.patch_dataset_real import PatchPairDatasetReal
tr=PatchPairDatasetReal(split="train").ids; te=PatchPairDatasetReal(split="test").ids
np.savez("$R/split_sids.npz", train_sids=np.array(tr), test_sids=np.array(te))
os.makedirs("$R/pk_hr_train", exist_ok=True)
for s in tr:
    src=os.path.abspath("$R/pk_hr/pk_set%d_transformed.npz"%s)
    dst="$R/pk_hr_train/pk_set%d_transformed.npz"%s
    if os.path.exists(src) and not os.path.exists(dst): os.symlink(src,dst)
print("train",len(tr),"test",len(te))
PY
echo "=== train q_HR ==="
python -m inference.nde --pk-dir "$R/pk_hr_train" --stitched-root "$STITCH" \
    --out "$R/posterior_hr.pkl" --theta-low 0.0 --theta-high 1.5
echo "=== HR reference done ==="
