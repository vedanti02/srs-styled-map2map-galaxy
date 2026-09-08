#!/bin/bash
# Per-patch P(k) panel comparing HR (paired) vs LR (single input patch) vs SR as
# (a) one stochastic sample and (b) the MEAN of 10 noise draws of the same patch.
#SBATCH --job-name=pksampavg
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --account=bapoczos
#SBATCH --output=logs/%j_pksampavg.log
#SBATCH --error=logs/%j_pksampavg.err
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:40:00
set -euo pipefail
cd /home/vkshirsa/srs-styled-map2map-galaxy
source /home/vkshirsa/venv/bin/activate
export PYTHONPATH=$PWD PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
CKPT=/data/user_data/vkshirsa/cmass-ili/models/patch_cmass_A_nopk_fixcond/best.pt
echo "node: $(hostname)"; mkdir -p figures_cmass
python analysis/pk_panel_sampleavg.py "$CKPT" cmassAfix
echo "PKSAMPAVG_ALL_DONE"
