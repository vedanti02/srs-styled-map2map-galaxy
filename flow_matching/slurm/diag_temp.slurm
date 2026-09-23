#!/bin/bash
#SBATCH --job-name=fm_diagtemp
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --account=bapoczos
#SBATCH --output=logs/%j_fm_diagtemp.log
#SBATCH --error=logs/%j_fm_diagtemp.err
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
set -euo pipefail
cd /home/vkshirsa/srs-styled-map2map-galaxy
source /home/vkshirsa/venv/bin/activate
export PYTHONPATH=$PWD PYTHONUNBUFFERED=1
M=/data/user_data/vkshirsa/cmass-ili/models; S=/data/user_data/vkshirsa/cmass-ili/sr_fields
echo "=== GAN arms vs flows: count histogram ==="
python -m flow_matching.diag_hist GAN_fixcond=$S/Afix_fiducial GAN_armR=$S/stage0_R fm_gauss_dq=$S/fm_gauss fm_gauss_nodq=$S/fm_gauss_nodq fm_lfspec_nodq=$S/fm_lfspec_nodq --n-boxes 100
echo "=== temperature sweep, fm_gauss_nodq ==="; python -m flow_matching.diag_temp --ckpt $M/fm_gauss_nodq/best.pt --scales 1.0,1.1,1.2,1.35 --n-boxes 10
echo "=== temperature sweep, fm_lfspec_nodq ==="; python -m flow_matching.diag_temp --ckpt $M/fm_lfspec_nodq/best.pt --scales 1.0,1.25,1.5,2.0 --n-boxes 10
echo "ALL_DONE"
