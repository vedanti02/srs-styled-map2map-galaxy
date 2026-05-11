#!/bin/bash
#SBATCH --job-name=sr_v2_smoke
#SBATCH --partition=general
#SBATCH --output=logs/%j_v2_smoke.log
#SBATCH --error=logs/%j_v2_smoke.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00

set -euo pipefail
cd /home/vkshirsa/SR-styled-map2map
mkdir -p logs checkpoints/v2_smoke
source /home/vkshirsa/venv/bin/activate

export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export PYTHONPATH=/home/vkshirsa/SR-styled-map2map:${PYTHONPATH:-}

python train_v2.py \
    --ckpt-dir checkpoints/v2_smoke \
    --epochs 2 --max-train-sets 16 \
    --batch-size 2 --num-workers 2 \
    --chan-base-g 64 --chan-base-d 32 --num-blocks 3 \
    --lambda-pk 2.0 --w-disp 2.0 --w-vel 0.5 \
    --save-every 1 --log-every 4
