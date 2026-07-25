#!/bin/bash
#SBATCH --job-name=fdiag2
#SBATCH --partition=debug
#SBATCH --output=logs/%j_fdiag.log
#SBATCH --error=logs/%j_fdiag.err
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
set -euo pipefail
cd /home/vkshirsa/srs-styled-map2map-galaxy
source /home/vkshirsa/venv/bin/activate
export PYTHONPATH=$PWD PYTHONUNBUFFERED=1
python field_diag.py runs/patch_cmass/field_ckpts/qhr_s100.pt
