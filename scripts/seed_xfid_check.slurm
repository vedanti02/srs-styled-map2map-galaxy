#!/bin/bash
#SBATCH --job-name=seedxfid
#SBATCH --partition=ghx4
#SBATCH --account=bdne-dtai-gh
#SBATCH --gpus-per-node=1   # DeltaAI rejects GPU-less jobs
#SBATCH --output=logs/%j_seedxfid.log
#SBATCH --error=logs/%j_seedxfid.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:25:00
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; source scripts/env_delta.sh
export PYTHONPATH=$PWD PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
echo "node: $(hostname)"
python analysis/seed_xfid_check.py ${XFID_ARGS:-}   # e.g. XFID_ARGS="runs/patch_cmass/nde_v2 LR Anopk Afixfid"
echo "SEEDXFID_ALL_DONE"
