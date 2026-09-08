#!/bin/bash
#SBATCH --job-name=samplevar
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --account=bapoczos
#SBATCH --output=logs/%j_samplevar.log
#SBATCH --error=logs/%j_samplevar.err
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
set -euo pipefail
cd /home/vkshirsa/srs-styled-map2map-galaxy
source /home/vkshirsa/venv/bin/activate
export PYTHONPATH=$PWD PYTHONUNBUFFERED=1
CKPT=/data/user_data/vkshirsa/cmass-ili/models/patch_cmass_A_nopk_fixcond/best.pt
echo "node: $(hostname)"; mkdir -p figures_cmass
python analysis/sample_variance.py "$CKPT" Afix_fiducial
echo "SAMPLEVAR_ALL_DONE"
