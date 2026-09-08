#!/bin/bash
#SBATCH --job-name=seedxfid
#SBATCH --partition=preempt
#SBATCH --qos=preempt_cpu_qos
#SBATCH --account=bapoczos
#SBATCH --output=logs/%j_seedxfid.log
#SBATCH --error=logs/%j_seedxfid.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:25:00
set -euo pipefail
cd /home/vkshirsa/srs-styled-map2map-galaxy
source /home/vkshirsa/venv/bin/activate
export PYTHONPATH=$PWD PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
echo "node: $(hostname)"
python analysis/seed_xfid_check.py
echo "SEEDXFID_ALL_DONE"
