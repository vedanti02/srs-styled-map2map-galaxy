#!/bin/bash
#SBATCH --job-name=sr_train_full
#SBATCH --partition=general
#SBATCH --output=logs/%j_train.log
#SBATCH --error=logs/%j_error.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

set -euo pipefail
cd /home/vkshirsa/SR-styled-map2map
mkdir -p logs checkpoints/full

source /home/vkshirsa/venv/bin/activate

echo "Job start: $(date) on $(hostname)" >&2
nvidia-smi -L >&2

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export PYTHONPATH=/home/vkshirsa/SR-styled-map2map:${PYTHONPATH:-}

python train.py \
    --data-root /data/group_data/universedata/lagrangian_output_64/stitched \
    --ckpt-dir checkpoints/full \
    --epochs 50 \
    --batch-size 4 \
    --num-workers 8 \
    --chan-base-g 256 \
    --chan-base-d 64 \
    --num-blocks 4 \
    --lambda-rec 10.0 \
    --lambda-r1 10.0 \
    --r1-every 16 \
    --save-every 1 \
    --log-every 50 \
    --grad-clip 1.0
