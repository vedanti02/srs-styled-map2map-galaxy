#!/bin/bash
# Spread-only re-run of the last Tier-1 stage (8 draws x 60 val boxes -> eval_spread); resumes existing draws.
#   sbatch --export=ALL,TAG=fm_gauss_nodq flow_matching/slurm/spread_fm.slurm
#SBATCH --job-name=fm_spread
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --account=bapoczos
#SBATCH --output=logs/%j_fm_spread.log
#SBATCH --error=logs/%j_fm_spread.err
#SBATCH --requeue
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
set -euo pipefail
cd /home/vkshirsa/srs-styled-map2map-galaxy
: "${TAG:?}"
CKPT=${CKPT:-/data/user_data/vkshirsa/cmass-ili/models/${TAG}/best.pt}; STEPS=${STEPS:-32}
SPREAD=/data/user_data/vkshirsa/cmass-ili/sr_fields/${TAG}_draws
source /home/vkshirsa/venv/bin/activate
export PYTHONPATH=$PWD PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
python -m flow_matching.sample_fm --ckpt "$CKPT" --out "$SPREAD" --split val --n-draws 8 --steps "$STEPS" --max-sims 60
python -m flow_matching.eval_spread "$SPREAD" "$TAG" --draws 8 --n-boxes 60
echo "SPREAD_JOB_DONE $TAG"
