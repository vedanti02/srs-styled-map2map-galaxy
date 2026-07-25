#!/bin/bash
#SBATCH --job-name=fxfid
#SBATCH --partition=general
#SBATCH --output=logs/%j_fxfid.log
#SBATCH --error=logs/%j_fxfid.err
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
set -euo pipefail
cd /home/vkshirsa/srs-styled-map2map-galaxy
source /home/vkshirsa/venv/bin/activate
export PYTHONPATH=$PWD PYTHONUNBUFFERED=1
python eval_field_crossfid.py
