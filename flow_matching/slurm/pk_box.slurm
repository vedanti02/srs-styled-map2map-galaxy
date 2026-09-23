#!/bin/bash
#SBATCH --job-name=fm_pkbox
#SBATCH --partition=ghx4
#SBATCH --account=bdne-dtai-gh
#SBATCH --output=logs/%j_fm_pkbox.log
#SBATCH --error=logs/%j_fm_pkbox.err
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; source scripts/env_delta.sh
export PYTHONPATH=$PWD PYTHONUNBUFFERED=1
S=$SRS_CMASS_ROOT/sr_fields
python -m flow_matching.plot_pk_box LR=input "GAN deployed"=$S/Afix_fiducial "GAN Arm R"=$S/stage0_R "FM gauss dequant"=$S/fm_gauss "FM gauss no-dequant"=$S/fm_gauss_nodq "FM lf_spectral"=$S/fm_lfspec_nodq --n-boxes 100 --tag all_arms
echo PKBOX_JOB_DONE
