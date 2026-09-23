#!/bin/bash
#SBATCH --job-name=qsrfix
#SBATCH --partition=ghx4
#SBATCH --account=bdne-dtai-gh
#SBATCH --output=logs/%j_qsrfix.log
#SBATCH --error=logs/%j_qsrfix.err
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; source scripts/env_delta.sh
export PYTHONPATH=$PWD PYTHONUNBUFFERED=1
: "${SRDIR:?}"; : "${OUT:?}"; : "${SEED:?}"
python train_field_cnn.py --source sr --sr-dir "$SRDIR" --seed "$SEED" --out "$OUT" --epochs 40
