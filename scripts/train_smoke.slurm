#!/bin/bash
#SBATCH --job-name=sr_smoke
#SBATCH --partition=general
#SBATCH --output=logs/%j_train.log
#SBATCH --error=logs/%j_error.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00

set -euo pipefail
cd /home/vkshirsa/SR-styled-map2map
mkdir -p logs checkpoints/smoke

source /home/vkshirsa/venv/bin/activate

echo "Job start: $(date) on $(hostname)" >&2
nvidia-smi -L >&2
echo "---" >&2

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export PYTHONPATH=/home/vkshirsa/SR-styled-map2map:${PYTHONPATH:-}

python train.py \
    --data-root /data/group_data/universedata/lagrangian_output_64/stitched \
    --ckpt-dir checkpoints/smoke \
    --epochs 3 \
    --max-train-sets 32 \
    --batch-size 2 \
    --num-workers 4 \
    --chan-base-g 128 \
    --chan-base-d 32 \
    --num-blocks 4 \
    --save-every 1 \
    --log-every 4
