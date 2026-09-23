#!/bin/bash
# Spread-only re-run of the last Tier-1 stage (8 draws x 60 val boxes -> eval_spread); resumes existing draws.
#   sbatch --export=ALL,TAG=fm_gauss_nodq flow_matching/slurm/spread_fm.slurm
#SBATCH --job-name=fm_spread
#SBATCH --partition=ghx4
#SBATCH --account=bdne-dtai-gh
#SBATCH --output=logs/%j_fm_spread.log
#SBATCH --error=logs/%j_fm_spread.err
#SBATCH --requeue
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; source scripts/env_delta.sh
: "${TAG:?}"
CKPT=${CKPT:-$SRS_CMASS_ROOT/models/${TAG}/best.pt}; STEPS=${STEPS:-32}
SPREAD=$SRS_CMASS_ROOT/sr_fields/${TAG}_draws
export PYTHONPATH=$PWD PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
python -m flow_matching.sample_fm --ckpt "$CKPT" --out "$SPREAD" --split val --n-draws 8 --steps "$STEPS" --max-sims 60
python -m flow_matching.eval_spread "$SPREAD" "$TAG" --draws 8 --n-boxes 60
echo "SPREAD_JOB_DONE $TAG"
