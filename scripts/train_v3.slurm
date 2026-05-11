#!/bin/bash
#SBATCH --job-name=sr_train_v3
#SBATCH --partition=general
#SBATCH --output=logs/%j_train_v3.log
#SBATCH --error=logs/%j_train_v3.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=15:00:00

set -euo pipefail
cd /home/vkshirsa/SR-styled-map2map
mkdir -p logs checkpoints/v3

source /home/vkshirsa/venv/bin/activate

echo "Job start: $(date) on $(hostname)" >&2
nvidia-smi -L >&2

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export PYTHONPATH=/home/vkshirsa/SR-styled-map2map:${PYTHONPATH:-}

# Optionally warm-start from v2's best checkpoint when ready
INIT_FROM=""
if [ -f checkpoints/v2/best.pt ]; then INIT_FROM="--init-from checkpoints/v2/best.pt"; fi

python train_v3.py \
    --data-root /data/group_data/universedata/lagrangian_output_64/stitched \
    --ckpt-dir checkpoints/v3 \
    --epochs 60 \
    --batch-size 4 \
    --num-workers 8 \
    --chan-base-g 256 --chan-base-d 64 --num-blocks 4 \
    --lambda-rec 10.0 --lambda-pk 10.0 \
    --w-disp 2.0 --w-vel 0.5 \
    --lambda-r1 10.0 --r1-every 16 \
    --low-k-weight 3.0 \
    --save-every 5 --log-every 50 --grad-clip 1.0 \
    $INIT_FROM
