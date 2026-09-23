#!/bin/bash
#SBATCH --job-name=fm_pkbox
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --account=bapoczos
#SBATCH --output=logs/%j_fm_pkbox.log
#SBATCH --error=logs/%j_fm_pkbox.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
set -euo pipefail
cd /home/vkshirsa/srs-styled-map2map-galaxy
source /home/vkshirsa/venv/bin/activate
export PYTHONPATH=$PWD PYTHONUNBUFFERED=1
S=/data/user_data/vkshirsa/cmass-ili/sr_fields
python -m flow_matching.plot_pk_box LR=input "GAN deployed"=$S/Afix_fiducial "GAN Arm R"=$S/stage0_R "FM gauss dequant"=$S/fm_gauss "FM gauss no-dequant"=$S/fm_gauss_nodq "FM lf_spectral"=$S/fm_lfspec_nodq --n-boxes 100 --tag all_arms
echo PKBOX_JOB_DONE
