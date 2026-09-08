#!/bin/bash
#SBATCH --job-name=disctest
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --account=bapoczos
#SBATCH --output=logs/%j_disctest.log
#SBATCH --error=logs/%j_disctest.err
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
set -euo pipefail
cd /home/vkshirsa/srs-styled-map2map-galaxy
source /home/vkshirsa/venv/bin/activate
export PYTHONPATH=$PWD PYTHONUNBUFFERED=1
CKPT=/data/user_data/vkshirsa/cmass-ili/models/patch_cmass_A_nopk_fixcond/best.pt
CONT=/data/user_data/vkshirsa/cmass-ili/sr_fields/Afix_fiducial_cont
echo "node: $(hostname)"; mkdir -p "$CONT" figures_cmass
# 1) continuous (un-rounded) SR for the VAL split, deployment (fiducial) conditioning
python gen_sr_fields.py --ckpt "$CKPT" --out "$CONT" --split val --theta-mode fiducial --no-round
# 2) discretization / stochasticity probe
python analysis/discretization_test.py "$CONT" Afix_fiducial
echo "DISCTEST_ALL_DONE"
