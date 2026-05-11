#!/bin/bash
#SBATCH --job-name=sr_nde
#SBATCH --partition=cpu
#SBATCH --output=logs/%j_nde.log
#SBATCH --error=logs/%j_nde.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -euo pipefail
cd /home/vkshirsa/SR-styled-map2map
source /home/vkshirsa/venv/bin/activate

PK_DIR="${PK_DIR:-runs/baseline/pk/hr}"
OUT="${OUT:-runs/baseline/nde/posterior_hr.pkl}"
N_TRAIN="${N_TRAIN:-1600}"

mkdir -p "$(dirname "$OUT")" logs

export PYTHONPATH=/home/vkshirsa/SR-styled-map2map:${PYTHONPATH:-}
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

echo "Training NDE on $PK_DIR -> $OUT (n_train=$N_TRAIN)"
python -m inference.nde \
    --pk-dir "$PK_DIR" \
    --stitched-root /data/group_data/universedata/lagrangian_output_64/stitched \
    --out "$OUT" \
    --n-train "$N_TRAIN" \
    --theta-low 0.0 --theta-high 1.5
