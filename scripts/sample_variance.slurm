#!/bin/bash
#SBATCH --job-name=samplevar
#SBATCH --partition=ghx4
#SBATCH --account=bdne-dtai-gh
#SBATCH --output=logs/%j_samplevar.log
#SBATCH --error=logs/%j_samplevar.err
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; source scripts/env_delta.sh
export PYTHONPATH=$PWD PYTHONUNBUFFERED=1
CKPT=$SRS_CMASS_ROOT/models/patch_cmass_A_nopk_fixcond/best.pt
echo "node: $(hostname)"; mkdir -p figures_cmass
python analysis/sample_variance.py "$CKPT" Afix_fiducial
echo "SAMPLEVAR_ALL_DONE"
