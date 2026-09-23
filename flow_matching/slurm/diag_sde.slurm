#!/bin/bash
#SBATCH --job-name=fm_diagsde
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --account=bapoczos
#SBATCH --output=logs/%j_fm_diagsde.log
#SBATCH --error=logs/%j_fm_diagsde.err
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
set -euo pipefail
cd /home/vkshirsa/srs-styled-map2map-galaxy
source /home/vkshirsa/venv/bin/activate
export PYTHONPATH=$PWD PYTHONUNBUFFERED=1
M=/data/user_data/vkshirsa/cmass-ili/models
echo "=== SDE sampler sweep, fm_gauss_nodq (ODE scale 1.0 row first) ==="
python -m flow_matching.diag_temp --ckpt $M/fm_gauss_nodq/best.pt --scales 1.0 --sde-gammas 0.5,1.0,2.0,3.0 --n-boxes 10 --steps 64
echo "ALL_DONE"
