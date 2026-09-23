#!/bin/bash
#SBATCH --job-name=fm_hist
#SBATCH --partition=ghx4
#SBATCH --account=bdne-dtai-gh
#SBATCH --output=logs/%j_fm_hist.log
#SBATCH --error=logs/%j_fm_hist.err
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; source scripts/env_delta.sh
export PYTHONPATH=$PWD PYTHONUNBUFFERED=1
S=$SRS_CMASS_ROOT/sr_fields
python -m flow_matching.diag_hist fm_gauss_dq=$S/fm_gauss fm_gauss_s07=$S/fm_gauss_s07 fm_gauss_nodq=$S/fm_gauss_nodq fm_lfspec_nodq=$S/fm_lfspec_nodq --n-boxes 100
echo HIST_DONE
