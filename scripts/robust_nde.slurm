#!/bin/bash
#SBATCH --job-name=robustnde
#SBATCH --partition=ghx4
#SBATCH --account=bdne-dtai-gh
#SBATCH --output=logs/%j_robustnde.log
#SBATCH --error=logs/%j_robustnde.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:00:00
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; source scripts/env_delta.sh
export PYTHONPATH=$PWD:${PYTHONPATH:-} OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
python analysis/robust_nde.py
