#!/bin/bash
#SBATCH --job-name=qhrrep
#SBATCH --partition=general
#SBATCH --output=logs/%j_qhrrep.log
#SBATCH --error=logs/%j_qhrrep.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
set -euo pipefail
cd /home/vkshirsa/srs-styled-map2map-galaxy
source /home/vkshirsa/venv/bin/activate
export PYTHONPATH=/home/vkshirsa/srs-styled-map2map-galaxy:${PYTHONPATH:-} PYTHONUNBUFFERED=1
mkdir -p runs/patch_cmass/qhr_reps
# 4 q_HR each in a FRESH process (like the originals), no explicit seed -> each
# gets fresh random torch state. Tests whether separate-process NDEs cluster
# tight (~0.003, like originals) or scatter (~0.22, like same-process retrains).
for i in 0 1 2 3; do
  echo "=== replicate $i (fresh process) ==="
  python -m inference.nde --pk-dir runs/patch_cmass/pk_hr_train \
     --theta-npz data/cmass_theta.npz --out runs/patch_cmass/qhr_reps/qhr_rep${i}.pkl \
     --theta-low 0.0 --theta-high 1.5
done
echo "QHR_REPS_DONE"
