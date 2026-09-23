#!/bin/bash
#SBATCH --job-name=fm_edit
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --account=bapoczos
#SBATCH --output=logs/%j_fm_edit.log
#SBATCH --error=logs/%j_fm_edit.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
set -euo pipefail
cd /home/vkshirsa/srs-styled-map2map-galaxy
source /home/vkshirsa/venv/bin/activate
export PYTHONPATH=$PWD PYTHONUNBUFFERED=1
S=/data/user_data/vkshirsa/cmass-ili/sr_fields
python -m flow_matching.diag_edit_fraction GAN_fixcond=$S/Afix_fiducial GAN_armR=$S/stage0_R fm_gauss_dq=$S/fm_gauss fm_gauss_nodq=$S/fm_gauss_nodq --n-boxes 50
echo EDIT_DONE
