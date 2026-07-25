#!/bin/bash
#SBATCH --job-name=frecov
#SBATCH --partition=debug
#SBATCH --output=logs/%j_frecov.log
#SBATCH --error=logs/%j_frecov.err
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:20:00
set -euo pipefail
cd /home/vkshirsa/srs-styled-map2map-galaxy
source /home/vkshirsa/venv/bin/activate
export PYTHONPATH=$PWD PYTHONUNBUFFERED=1
python field_recovery_plot.py
