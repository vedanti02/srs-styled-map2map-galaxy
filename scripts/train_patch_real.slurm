#!/bin/bash
# Usage: sbatch --export=ALL,PAD=0,CKPT=checkpoints/patch_real_v1,NAME=A scripts/train_patch_real.slurm
#SBATCH --job-name=patchreal
#SBATCH --partition=general
#SBATCH --output=logs/%j_patchreal.log
#SBATCH --error=logs/%j_patchreal.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=23:00:00

set -euo pipefail
cd /home/vkshirsa/srs-styled-map2map-galaxy
: "${PAD:?need PAD}"; : "${CKPT:?need CKPT}"; : "${NAME:?need NAME}"
mkdir -p logs "$CKPT"
source /home/vkshirsa/venv/bin/activate
echo "Job start $(date) on $(hostname); arm=$NAME pad=$PAD" >&2
nvidia-smi -L >&2
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
export PYTHONPATH=/home/vkshirsa/srs-styled-map2map-galaxy:${PYTHONPATH:-}

python train_patch_real.py \
    --pad "$PAD" --ckpt-dir "$CKPT" \
    --epochs 25 --sims-per-batch 1 --num-workers 8 \
    --chan-base-g 128 --chan-base-d 64 --num-blocks 4 \
    --lambda-rec 10.0 --lambda-pk 2.0 --w-disp 2.0 --w-vel 0.5 \
    --lambda-r1 10.0 --r1-every 16 \
    --save-every 5 --log-every 200 --grad-clip 1.0 --val-max-sims 40
