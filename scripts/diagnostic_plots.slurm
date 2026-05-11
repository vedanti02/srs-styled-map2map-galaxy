#!/bin/bash
#SBATCH --job-name=sr_diag_plots
#SBATCH --partition=general
#SBATCH --output=logs/%j_diag.log
#SBATCH --error=logs/%j_diag.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00

set -euo pipefail
cd /home/vkshirsa/SR-styled-map2map
source /home/vkshirsa/venv/bin/activate

export PYTHONPATH=/home/vkshirsa/SR-styled-map2map:${PYTHONPATH:-}
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1

CKPT="${CKPT:-checkpoints/v2/best.pt}"
TAG="${TAG:-v2_best}"

python -m analysis.plot_diagnostic \
    --ckpt "$CKPT" --tag "$TAG" \
    --out-dir runs/baseline/plots/diagnostic_${TAG} \
    --n-sims 20 --projection-sim-idx 0
