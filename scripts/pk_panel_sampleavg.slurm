#!/bin/bash
# Per-patch P(k) panel comparing HR (paired) vs LR (single input patch) vs SR as
# (a) one stochastic sample and (b) the MEAN of 10 noise draws of the same patch.
#SBATCH --job-name=pksampavg
#SBATCH --partition=ghx4
#SBATCH --account=bdne-dtai-gh
#SBATCH --output=logs/%j_pksampavg.log
#SBATCH --error=logs/%j_pksampavg.err
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:40:00
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; source scripts/env_delta.sh
export PYTHONPATH=$PWD PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
CKPT=$SRS_CMASS_ROOT/models/patch_cmass_A_nopk_fixcond/best.pt
echo "node: $(hostname)"; mkdir -p figures_cmass
python analysis/pk_panel_sampleavg.py "$CKPT" cmassAfix
echo "PKSAMPAVG_ALL_DONE"
