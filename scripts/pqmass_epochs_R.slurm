#!/bin/bash
# Robustness: is Arm R's PQMass 0.75 a property of the model or a lucky val-selected epoch?
# Evaluate PQMass (val, 200 boxes) on EVERY saved epoch of stage0_R. (file epoch_N = log epoch N-1;
# best.pt = log ep21 = epoch_22.pt, which should reproduce 0.75 as a consistency check.)
#SBATCH --job-name=pqepochR
#SBATCH --partition=ghx4
#SBATCH --account=bdne-dtai-gh
#SBATCH --output=logs/%j_pqepochR.log
#SBATCH --error=logs/%j_pqepochR.err
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; source scripts/env_delta.sh
export PYTHONPATH=$PWD PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
M=$SRS_CMASS_ROOT/models/stage0_R
SCR=$SRS_CMASS_ROOT/sr_fields/_pq_epochs_R
echo "node $(hostname); checkpoints:"; ls -1 $M
for CK in $M/epoch_*.pt; do
  E=$(basename "$CK" .pt); D=$SCR/$E; mkdir -p "$D"
  python gen_sr_fields.py --ckpt "$CK" --out "$D" --split val --theta-mode fiducial > /dev/null
  echo "=== PQMass stage0_R $E (log epoch $(( ${E#epoch_} - 1 ))) ==="
  python analysis/pqmass_cmass.py "$D" "stage0_R_$E" | grep -E "SR vs HR|LR vs HR"
done
rm -rf "$SCR"
echo "PQEPOCH_DONE"
