#!/bin/bash
#SBATCH --job-name=fm_hist2
#SBATCH --partition=ghx4
#SBATCH --account=bdne-dtai-gh
#SBATCH --gpus-per-node=1   # DeltaAI rejects GPU-less jobs
#SBATCH --output=logs/%j_fm_hist.log
#SBATCH --error=logs/%j_fm_hist.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:40:00
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; source scripts/env_delta.sh
export PYTHONPATH=$PWD PYTHONUNBUFFERED=1
python -m flow_matching.diag_hist fm_gauss_nodq=$SRS_CMASS_ROOT/sr_fields/fm_gauss_nodq fm_gauss_nodq_lw2=$SRS_CMASS_ROOT/sr_fields/fm_gauss_nodq_lw2 fm_gauss_nodq_tlate=$SRS_CMASS_ROOT/sr_fields/fm_gauss_nodq_tlate stage0_R=$SRS_CMASS_ROOT/sr_fields/stage0_R --n-boxes 100
echo "HIST_DONE"
