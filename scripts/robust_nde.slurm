#!/bin/bash
#SBATCH --job-name=robustnde
#SBATCH --partition=general
#SBATCH --output=logs/%j_robustnde.log
#SBATCH --error=logs/%j_robustnde.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:00:00
set -euo pipefail
cd /home/vkshirsa/srs-styled-map2map-galaxy
source /home/vkshirsa/venv/bin/activate
export PYTHONPATH=/home/vkshirsa/srs-styled-map2map-galaxy:${PYTHONPATH:-} OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
python analysis/robust_nde.py
