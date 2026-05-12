#!/bin/bash
#SBATCH --job-name=sr_train_v7
#SBATCH --partition=general
#SBATCH --output=logs/%j_train_v7.log
#SBATCH --error=logs/%j_train_v7.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00

set -euo pipefail
cd /home/vkshirsa/SR-styled-map2map
mkdir -p logs checkpoints/v7

source /home/vkshirsa/venv/bin/activate

echo "Job start: $(date) on $(hostname)" >&2
nvidia-smi -L >&2

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export PYTHONPATH=/home/vkshirsa/SR-styled-map2map:${PYTHONPATH:-}

# v7 = v4's no-GAN single-scale Pk recipe, WARM-STARTED from v2's best.pt.
# Hypothesis: v4 already wins KL(Omega_m). Removing the GAN's adversarial
# pull while keeping v2's strong starting point should push Omega_m KL down
# further. v3 showed warm-starting from v2 gave the best val_pkRMS_log10
# (0.2088); v7 keeps that warm start but drops the GAN like v4 did.
python train_v4_supervised.py \
    --data-root /data/group_data/universedata/lagrangian_output_64/stitched \
    --ckpt-dir checkpoints/v7 \
    --init-from checkpoints/v2/best.pt \
    --epochs 60 \
    --batch-size 4 \
    --num-workers 8 \
    --chan-base-g 256 --num-blocks 4 \
    --lambda-rec 10.0 --lambda-pk 10.0 \
    --w-disp 2.0 --w-vel 0.5 \
    --lr 2e-4 \
    --save-every 5 --log-every 50 --grad-clip 1.0
