#!/bin/bash
#SBATCH --job-name=fm_diagdec
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --account=bapoczos
#SBATCH --output=logs/%j_fm_diagdec.log
#SBATCH --error=logs/%j_fm_diagdec.err
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
set -euo pipefail
cd /home/vkshirsa/srs-styled-map2map-galaxy
: "${TAG:?}"
source /home/vkshirsa/venv/bin/activate
export PYTHONPATH=$PWD PYTHONUNBUFFERED=1
CK=/data/user_data/vkshirsa/cmass-ili/models/${TAG}
python -m flow_matching.diag_decoder --ckpt "$CK/best.pt" --n-boxes 10
