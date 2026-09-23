#!/bin/bash
#SBATCH --job-name=fm_hist2
#SBATCH --partition=preempt
#SBATCH --qos=preempt_cpu_qos
#SBATCH --account=bapoczos
#SBATCH --output=logs/%j_fm_hist.log
#SBATCH --error=logs/%j_fm_hist.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:40:00
set -euo pipefail
cd /home/vkshirsa/srs-styled-map2map-galaxy
source /home/vkshirsa/venv/bin/activate
export PYTHONPATH=$PWD PYTHONUNBUFFERED=1
python -m flow_matching.diag_hist fm_gauss_nodq=/data/user_data/vkshirsa/cmass-ili/sr_fields/fm_gauss_nodq fm_gauss_nodq_lw2=/data/user_data/vkshirsa/cmass-ili/sr_fields/fm_gauss_nodq_lw2 fm_gauss_nodq_tlate=/data/user_data/vkshirsa/cmass-ili/sr_fields/fm_gauss_nodq_tlate stage0_R=/data/user_data/vkshirsa/cmass-ili/sr_fields/stage0_R --n-boxes 100
echo "HIST_DONE"
