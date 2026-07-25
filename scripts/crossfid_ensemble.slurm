#!/bin/bash
#SBATCH --job-name=xfens
#SBATCH --partition=general
#SBATCH --output=logs/%j_xfens.log
#SBATCH --error=logs/%j_xfens.err
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
set -euo pipefail
cd /home/vkshirsa/srs-styled-map2map-galaxy
source /home/vkshirsa/venv/bin/activate
export PYTHONPATH=$PWD PYTHONUNBUFFERED=1
python analysis/ensemble_crossfid.py
