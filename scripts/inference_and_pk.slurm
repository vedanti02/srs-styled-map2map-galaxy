#!/bin/bash
#SBATCH --job-name=sr_infer_pk
#SBATCH --partition=general
#SBATCH --output=logs/%j_infer.log
#SBATCH --error=logs/%j_infer.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -euo pipefail
cd /home/vkshirsa/SR-styled-map2map
source /home/vkshirsa/venv/bin/activate

# Use the checkpoint path passed in via env var, or default to latest
CKPT="${CKPT:-checkpoints/full/epoch_17.pt}"
TAG="${TAG:-e17}"

OUT_TR=runs/baseline/transformed_${TAG}
OUT_PK=runs/baseline/pk/sr_${TAG}
mkdir -p "$OUT_TR" "$OUT_PK"

export PYTHONPATH=/home/vkshirsa/SR-styled-map2map:${PYTHONPATH:-}
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1

echo "=== inference: $CKPT -> $OUT_TR ==="
python transform.py \
    --model-path "$CKPT" \
    --data-root /data/group_data/universedata/lagrangian_output_64/stitched \
    --output-dir "$OUT_TR" \
    --split val \
    --n-noise-samples 1 --base-seed 0

echo "=== Pk on transformed: -> $OUT_PK ==="
python -m analysis.power_spectrum --mode transformed \
    --input "$OUT_TR" --output "$OUT_PK" \
    --lbox 1000.0 --estimator div --n-bins 32

echo "=== done ==="
