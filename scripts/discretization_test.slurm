#!/bin/bash
#SBATCH --job-name=disctest
#SBATCH --partition=ghx4
#SBATCH --account=bdne-dtai-gh
#SBATCH --output=logs/%j_disctest.log
#SBATCH --error=logs/%j_disctest.err
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; source scripts/env_delta.sh
export PYTHONPATH=$PWD PYTHONUNBUFFERED=1
CKPT=$SRS_CMASS_ROOT/models/patch_cmass_A_nopk_fixcond/best.pt
CONT=$SRS_CMASS_ROOT/sr_fields/Afix_fiducial_cont
echo "node: $(hostname)"; mkdir -p "$CONT" figures_cmass
# 1) continuous (un-rounded) SR for the VAL split, deployment (fiducial) conditioning
python gen_sr_fields.py --ckpt "$CKPT" --out "$CONT" --split val --theta-mode fiducial --no-round
# 2) discretization / stochasticity probe
python analysis/discretization_test.py "$CONT" Afix_fiducial
echo "DISCTEST_ALL_DONE"
