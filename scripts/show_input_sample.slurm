#!/bin/bash
#SBATCH --job-name=showinput
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --account=bapoczos
#SBATCH --output=logs/%j_showinput.log
#SBATCH --error=logs/%j_showinput.err
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:10:00
set -euo pipefail
cd /home/vkshirsa/srs-styled-map2map-galaxy
source /home/vkshirsa/venv/bin/activate
export PYTHONPATH=$PWD PYTHONUNBUFFERED=1
python analysis/show_input_sample.py /data/user_data/vkshirsa/cmass-ili/models/patch_cmass_A_nopk_fixcond/best.pt
