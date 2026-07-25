#!/bin/bash
#SBATCH --job-name=qsrrep
#SBATCH --partition=general
#SBATCH --output=logs/%j_qsrrep.log
#SBATCH --error=logs/%j_qsrrep.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
set -euo pipefail
cd /home/vkshirsa/srs-styled-map2map-galaxy
source /home/vkshirsa/venv/bin/activate
export PYTHONPATH=/home/vkshirsa/srs-styled-map2map-galaxy:${PYTHONPATH:-} PYTHONUNBUFFERED=1
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
