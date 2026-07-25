#!/bin/bash
#SBATCH --job-name=ovf
#SBATCH --partition=debug
#SBATCH --output=logs/%j_ovf.log
#SBATCH --error=logs/%j_ovf.err
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
set -euo pipefail
cd /home/vkshirsa/srs-styled-map2map-galaxy
source /home/vkshirsa/venv/bin/activate
export PYTHONPATH=$PWD PYTHONUNBUFFERED=1
python overfit_test.py
