#!/bin/bash
#SBATCH --job-name=fdiag2
#SBATCH --partition=ghx4
#SBATCH --account=bdne-dtai-gh
#SBATCH --output=logs/%j_fdiag.log
#SBATCH --error=logs/%j_fdiag.err
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; source scripts/env_delta.sh
export PYTHONPATH=$PWD PYTHONUNBUFFERED=1
python field_diag.py runs/patch_cmass/field_ckpts/qhr_s7.pt
