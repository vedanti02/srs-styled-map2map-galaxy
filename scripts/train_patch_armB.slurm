#!/bin/bash
#SBATCH --job-name=patch_armB
#SBATCH --partition=general
#SBATCH --output=logs/%j_patch_armB.log
#SBATCH --error=logs/%j_patch_armB.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=18:00:00

set -euo pipefail
cd /home/vkshirsa/srs-styled-map2map-galaxy
mkdir -p logs checkpoints/patch_v2_boundary
source /home/vkshirsa/venv/bin/activate

echo "Job start: $(date) on $(hostname)" >&2
nvidia-smi -L >&2
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export PYTHONPATH=/home/vkshirsa/srs-styled-map2map-galaxy:${PYTHONPATH:-}

python train_patch.py \
    --pad 3 \
    --ckpt-dir checkpoints/patch_v2_boundary \
    --epochs 40 \
    --batch-size 16 \
    --num-workers 8 \
    --chan-base-g 256 --chan-base-d 64 --num-blocks 4 \
    --lambda-rec 10.0 --lambda-pk 2.0 --w-disp 2.0 --w-vel 0.5 \
    --lambda-r1 10.0 --r1-every 16 \
    --save-every 5 --log-every 100 --grad-clip 1.0
