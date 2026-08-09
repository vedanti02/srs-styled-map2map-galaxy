#!/bin/bash
#SBATCH --job-name=qsrfix
#SBATCH --partition=general
#SBATCH --output=logs/%j_qsrfix.log
#SBATCH --error=logs/%j_qsrfix.err
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
set -euo pipefail
cd /home/vkshirsa/srs-styled-map2map-galaxy
source /home/vkshirsa/venv/bin/activate
export PYTHONPATH=$PWD PYTHONUNBUFFERED=1
: "${SRDIR:?}"; : "${OUT:?}"; : "${SEED:?}"
python train_field_cnn.py --source sr --sr-dir "$SRDIR" --seed "$SEED" --out "$OUT" --epochs 40
