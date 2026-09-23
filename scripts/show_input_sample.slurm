#!/bin/bash
#SBATCH --job-name=showinput
#SBATCH --partition=ghx4
#SBATCH --account=bdne-dtai-gh
#SBATCH --output=logs/%j_showinput.log
#SBATCH --error=logs/%j_showinput.err
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:10:00
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; source scripts/env_delta.sh
export PYTHONPATH=$PWD PYTHONUNBUFFERED=1
python analysis/show_input_sample.py $SRS_CMASS_ROOT/models/patch_cmass_A_nopk_fixcond/best.pt
