#!/bin/bash
#SBATCH --job-name=fm_diag
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --account=bapoczos
#SBATCH --output=logs/%j_fm_diag.log
#SBATCH --error=logs/%j_fm_diag.err
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
set -euo pipefail
cd /home/vkshirsa/srs-styled-map2map-galaxy
: "${TAG:?}"
source /home/vkshirsa/venv/bin/activate
export PYTHONPATH=$PWD PYTHONUNBUFFERED=1
CK=/data/user_data/vkshirsa/cmass-ili/models/${TAG}
python -m flow_matching.diag_sampler --ckpt "$CK/best.pt" --n-boxes 10 --draws 4 --configs "heun:16,heun:32,heun:64,heun:128,euler:50"
echo "=== last.pt (final epoch) vs best.pt, and raw vs EMA weights, at heun:32 ==="
python -m flow_matching.diag_sampler --ckpt "$CK/last.pt" --n-boxes 10 --draws 4 --configs "heun:32"
python -m flow_matching.diag_sampler --ckpt "$CK/best.pt" --n-boxes 10 --draws 4 --configs "heun:32" --no-ema
