#!/bin/bash
#SBATCH --job-name=qsrrep
#SBATCH --partition=ghx4
#SBATCH --account=bdne-dtai-gh
#SBATCH --output=logs/%j_qsrrep.log
#SBATCH --error=logs/%j_qsrrep.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; source scripts/env_delta.sh
export PYTHONPATH=$PWD:${PYTHONPATH:-} PYTHONUNBUFFERED=1
mkdir -p runs/patch_cmass/qsr_reps
for arm in Anopk LRbase; do
  for i in 0 1 2 3; do
    echo "=== $arm replicate $i (fresh process) ==="
    python -m inference.nde --pk-dir runs/patch_cmass/pk_${arm}_train \
       --theta-npz data/cmass_theta.npz \
       --out runs/patch_cmass/qsr_reps/q${arm}_rep${i}.pkl --theta-low 0.0 --theta-high 1.5
  done
done
echo "QSR_REPS_DONE"
