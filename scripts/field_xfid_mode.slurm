#!/bin/bash
#SBATCH --job-name=fxmode
#SBATCH --partition=ghx4
#SBATCH --account=bdne-dtai-gh
#SBATCH --output=logs/%j_fxmode.log
#SBATCH --error=logs/%j_fxmode.err
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; source scripts/env_delta.sh
export PYTHONPATH=$PWD PYTHONUNBUFFERED=1
: "${CKPTDIR:?}"
python eval_field_crossfid.py --ckpt-dir "$CKPTDIR" --out "$CKPTDIR/field_crossfid.npz"
