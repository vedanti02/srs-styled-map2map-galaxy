#!/bin/bash
#SBATCH --job-name=frecov
#SBATCH --partition=ghx4
#SBATCH --account=bdne-dtai-gh
#SBATCH --output=logs/%j_frecov.log
#SBATCH --error=logs/%j_frecov.err
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:20:00
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; source scripts/env_delta.sh
export PYTHONPATH=$PWD PYTHONUNBUFFERED=1
python field_recovery_plot.py
