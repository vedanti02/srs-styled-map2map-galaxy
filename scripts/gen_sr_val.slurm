#!/bin/bash
#SBATCH --job-name=gensrval
#SBATCH --partition=ghx4
#SBATCH --account=bdne-dtai-gh
#SBATCH --output=logs/%j_gensrval.log
#SBATCH --error=logs/%j_gensrval.err
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; source scripts/env_delta.sh
export PYTHONPATH=$PWD PYTHONUNBUFFERED=1
python gen_sr_fields.py --ckpt checkpoints/patch_cmass_A_nopk/best.pt \
   --out runs/patch_cmass/sr_fields_Anopk --split val --mode naive
