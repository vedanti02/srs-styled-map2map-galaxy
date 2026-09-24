#!/bin/bash
#SBATCH --job-name=xfrep
#SBATCH --partition=ghx4
#SBATCH --account=bdne-dtai-gh
#SBATCH --output=logs/%j_xfrep.log
#SBATCH --error=logs/%j_xfrep.err
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=05:00:00
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; source scripts/env_delta.sh
export PYTHONPATH=$PWD PYTHONUNBUFFERED=1
: "${TAG:?}"; : "${PKDIR:?}"
# NDEDIR=runs/patch_cmass/nde_v2 (default): zero-mode k-bins dropped, seeded replicates.
# The restored pre-Delta posteriors (32-dim input, unseeded) live in runs/patch_cmass/_robust_tmp.
NDEDIR=${NDEDIR:-runs/patch_cmass/nde_v2}
mkdir -p "$NDEDIR"
for i in 0 1 2 3; do
  echo "=== $TAG replicate $i (fresh process, CORRECTED labels) ==="
  python -m inference.nde --pk-dir runs/patch_cmass/${PKDIR} \
     --theta-npz data/cmass_theta.npz \
     --out "$NDEDIR/q${TAG}_${i}.pkl" --seed "$i" --max-epochs "${MAXEP:-100}" \
     --theta-low 0.0 --theta-high 1.5
done
echo "XFREP_DONE_${TAG}"
