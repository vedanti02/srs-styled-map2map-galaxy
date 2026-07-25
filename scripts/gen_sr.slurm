#!/bin/bash
#SBATCH --job-name=gensr
#SBATCH --partition=general
#SBATCH --output=logs/%j_gensr.log
#SBATCH --error=logs/%j_gensr.err
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
set -euo pipefail
cd /home/vkshirsa/srs-styled-map2map-galaxy
source /home/vkshirsa/venv/bin/activate
export PYTHONPATH=$PWD PYTHONUNBUFFERED=1
python gen_sr_fields.py --ckpt checkpoints/patch_cmass_A_nopk/best.pt \
   --out runs/patch_cmass/sr_fields_Anopk --split train --mode naive
