#!/bin/bash
#SBATCH --job-name=xfrep
#SBATCH --partition=general
#SBATCH --output=logs/%j_xfrep.log
#SBATCH --error=logs/%j_xfrep.err
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=05:00:00
set -euo pipefail
cd /home/vkshirsa/srs-styled-map2map-galaxy
source /home/vkshirsa/venv/bin/activate
export PYTHONPATH=$PWD PYTHONUNBUFFERED=1
: "${TAG:?}"; : "${PKDIR:?}"
mkdir -p runs/patch_cmass/_robust_tmp
for i in 0 1 2 3; do
  echo "=== $TAG replicate $i (fresh process, CORRECTED labels) ==="
  python -m inference.nde --pk-dir runs/patch_cmass/${PKDIR} \
     --theta-npz data/cmass_theta.npz \
     --out runs/patch_cmass/_robust_tmp/q${TAG}_${i}.pkl \
     --theta-low 0.0 --theta-high 1.5
done
echo "XFREP_DONE_${TAG}"
