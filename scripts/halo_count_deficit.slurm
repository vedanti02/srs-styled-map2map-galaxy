#!/bin/bash
#SBATCH --job-name=halodef
#SBATCH --partition=preempt
#SBATCH --qos=preempt_cpu_qos
#SBATCH --account=bapoczos
#SBATCH --output=logs/%j_halodef.log
#SBATCH --error=logs/%j_halodef.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
set -euo pipefail
cd /home/vkshirsa/srs-styled-map2map-galaxy
source /home/vkshirsa/venv/bin/activate
export PYTHONPATH=$PWD PYTHONUNBUFFERED=1
python analysis/halo_count_deficit.py
