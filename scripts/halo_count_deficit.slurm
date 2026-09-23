#!/bin/bash
#SBATCH --job-name=halodef
#SBATCH --partition=ghx4
#SBATCH --account=bdne-dtai-gh
#SBATCH --gpus-per-node=1   # DeltaAI rejects GPU-less jobs
#SBATCH --output=logs/%j_halodef.log
#SBATCH --error=logs/%j_halodef.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; source scripts/env_delta.sh
export PYTHONPATH=$PWD PYTHONUNBUFFERED=1
python analysis/halo_count_deficit.py
