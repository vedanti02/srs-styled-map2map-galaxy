#!/bin/bash
#SBATCH --job-name=fieldcnn
#SBATCH --partition=ghx4
#SBATCH --account=bdne-dtai-gh
#SBATCH --output=logs/%j_fieldcnn.log
#SBATCH --error=logs/%j_fieldcnn.err
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; source scripts/env_delta.sh
export PYTHONPATH=$PWD PYTHONUNBUFFERED=1
: "${SRC:?}"; : "${SEED:?}"
python train_field_cnn.py --source "$SRC" --seed "$SEED" \
   --out runs/patch_cmass/field_ckpts/q${SRC}_s${SEED}.pt --epochs 40
