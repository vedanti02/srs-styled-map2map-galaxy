#!/bin/bash
# Per-arm CMASS eval. Usage:
#   sbatch --export=ALL,CKPT=checkpoints/patch_cmass_A/best.pt,MODE=naive,TAG=cmassA_naive   scripts/eval_cmass_wrapper.slurm
#   sbatch --export=ALL,CKPT=checkpoints/patch_cmass_B/best.pt,MODE=overlap,TAG=cmassB_overlap scripts/eval_cmass_wrapper.slurm
#SBATCH --job-name=evalcmass
#SBATCH --partition=general
#SBATCH --output=logs/%j_evalcmass.log
#SBATCH --error=logs/%j_evalcmass.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=05:00:00

set -euo pipefail
cd /home/vkshirsa/srs-styled-map2map-galaxy
source /home/vkshirsa/venv/bin/activate
: "${CKPT:?}"; : "${MODE:?}"; : "${TAG:?}"
export PYTHONPATH=/home/vkshirsa/srs-styled-map2map-galaxy:${PYTHONPATH:-} OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
THETA=data/cmass_theta.npz
R=runs/patch_cmass; mkdir -p "$R" logs
SCR=/tmp/vkshirsa_cmass_${SLURM_JOB_ID}; mkdir -p "$SCR"
trap 'rm -rf "$SCR"' EXIT
echo "ARM eval: CKPT=$CKPT MODE=$MODE TAG=$TAG  scratch=$SCR"

if [ ! -f "$R/posterior_hr.pkl" ]; then
  echo "ERROR: HR reference ($R/posterior_hr.pkl) missing — run scripts/eval_cmass_hr.slurm first"; exit 1
fi

echo "=== infer SR ($TAG) → scratch ==="
python infer_stitch_cmass.py --model-path "$CKPT" --output-dir "$SCR/sr_$TAG" \
    --split all --mode "$MODE" --save-dtype float16

echo "=== seam eval (test split) ==="
python -m analysis.eval_seam_cmass --sr-dir "$SCR/sr_$TAG" --out "$R/seam_$TAG" --label "$TAG" --split test

echo "=== Pk(SR) + q_SR + KL eval ==="
python -m analysis.power_spectrum --mode transformed --input "$SCR/sr_$TAG" \
    --output "$R/pk_$TAG" --lbox 1000.0 --estimator counts --n-bins 32
python - <<PY
import os, numpy as np
tr=np.load("$R/split_sids.npz")["train_sids"]
os.makedirs("$R/pk_${TAG}_train", exist_ok=True)
for s in tr:
    src=os.path.abspath("$R/pk_$TAG/pk_set%d_transformed.npz"%s)
    dst="$R/pk_${TAG}_train/pk_set%d_transformed.npz"%s
    if os.path.exists(src) and not os.path.exists(dst): os.symlink(src,dst)
PY
python -m inference.nde --pk-dir "$R/pk_${TAG}_train" --theta-npz "$THETA" \
    --out "$R/posterior_$TAG.pkl" --theta-low 0.0 --theta-high 1.5 || echo "q_SR train failed"
python evaluate.py --posterior-hr "$R/posterior_hr.pkl" --posterior-sr "$R/posterior_$TAG.pkl" \
    --pk-hr-dir "$R/pk_hr" --pk-sr-dir "$R/pk_$TAG" --pk-hr-prefix "pk_set" --pk-sr-prefix "pk_set" \
    --theta-npz "$THETA" --restrict-sids "$R/split_sids.npz" \
    --n-samples 2000 --output "$R/metrics_$TAG.npz" || echo "KL eval failed"

echo "=== eval $TAG done (scratch auto-cleaned) ==="
