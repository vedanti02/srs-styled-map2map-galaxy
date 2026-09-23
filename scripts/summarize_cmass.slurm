#!/bin/bash
#SBATCH --job-name=sumcmass
#SBATCH --partition=ghx4
#SBATCH --account=bdne-dtai-gh
#SBATCH --gpus-per-node=1   # DeltaAI rejects GPU-less jobs
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%j_sumcmass.log
#SBATCH --error=logs/%j_sumcmass.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; source scripts/env_delta.sh
export PYTHONPATH=$PWD:${PYTHONPATH:-} PYTHONUNBUFFERED=1
echo "=== inputs ==="; ls -la runs/patch_cmass/seam_*.npz runs/patch_cmass/metrics_*.npz 2>&1 || true
python -m analysis.summarize_cmass
echo "=== summary done ==="
