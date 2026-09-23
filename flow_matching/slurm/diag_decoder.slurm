#!/bin/bash
#SBATCH --job-name=fm_diagdec
#SBATCH --partition=ghx4
#SBATCH --account=bdne-dtai-gh
#SBATCH --output=logs/%j_fm_diagdec.log
#SBATCH --error=logs/%j_fm_diagdec.err
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; source scripts/env_delta.sh
: "${TAG:?}"
export PYTHONPATH=$PWD PYTHONUNBUFFERED=1
CK=$SRS_CMASS_ROOT/models/${TAG}
python -m flow_matching.diag_decoder --ckpt "$CK/best.pt" --n-boxes 10
