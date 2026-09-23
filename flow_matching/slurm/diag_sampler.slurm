#!/bin/bash
#SBATCH --job-name=fm_diag
#SBATCH --partition=ghx4
#SBATCH --account=bdne-dtai-gh
#SBATCH --output=logs/%j_fm_diag.log
#SBATCH --error=logs/%j_fm_diag.err
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; source scripts/env_delta.sh
: "${TAG:?}"
export PYTHONPATH=$PWD PYTHONUNBUFFERED=1
CK=$SRS_CMASS_ROOT/models/${TAG}
python -m flow_matching.diag_sampler --ckpt "$CK/best.pt" --n-boxes 10 --draws 4 --configs "heun:16,heun:32,heun:64,heun:128,euler:50"
echo "=== last.pt (final epoch) vs best.pt, and raw vs EMA weights, at heun:32 ==="
python -m flow_matching.diag_sampler --ckpt "$CK/last.pt" --n-boxes 10 --draws 4 --configs "heun:32"
python -m flow_matching.diag_sampler --ckpt "$CK/best.pt" --n-boxes 10 --draws 4 --configs "heun:32" --no-ema
