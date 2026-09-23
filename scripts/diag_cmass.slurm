#!/bin/bash
# Box + per-patch diagnostic panels (Pk: P(k)/transfer/cross-power, and projection) for a cmass arm.
#   sbatch --export=ALL,CKPT=checkpoints/patch_cmass_A/best.pt,MODE=naive,TAG=cmassA scripts/diag_cmass.slurm
#SBATCH --job-name=diagcmass
#SBATCH --partition=ghx4
#SBATCH --account=bdne-dtai-gh
#SBATCH --output=logs/%j_diagcmass.log
#SBATCH --error=logs/%j_diagcmass.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:40:00

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; source scripts/env_delta.sh
: "${CKPT:?}"; : "${MODE:?}"; : "${TAG:?}"
export PYTHONPATH=$PWD:${PYTHONPATH:-} OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
python -m analysis.diag_cmass --ckpt "$CKPT" --mode "$MODE" --tag "$TAG" --theta-mode "${TMODE:-fiducial}" --n-sims 16 --out-dir figures_cmass
echo "diag $TAG done"
