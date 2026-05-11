#!/bin/bash
#SBATCH --job-name=sr_infer_full
#SBATCH --partition=general
#SBATCH --output=logs/%j_infer_full.log
#SBATCH --error=logs/%j_infer_full.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# Inference on ALL 2000 sims; saves float16 to save disk.
# Args: CKPT (required), TAG (required)
set -euo pipefail
cd /home/vkshirsa/SR-styled-map2map
source /home/vkshirsa/venv/bin/activate

CKPT="${CKPT:?must set CKPT}"
TAG="${TAG:?must set TAG}"
OUT_TR=runs/baseline/transformed_${TAG}
OUT_PK=runs/baseline/pk/sr_${TAG}
mkdir -p "$OUT_TR" "$OUT_PK"

export PYTHONPATH=/home/vkshirsa/SR-styled-map2map:${PYTHONPATH:-}
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1

python transform.py \
    --model-path "$CKPT" \
    --data-root /data/group_data/universedata/lagrangian_output_64/stitched \
    --output-dir "$OUT_TR" \
    --split all --denormalize --save-dtype float16 \
    --n-noise-samples 1 --base-seed 0

python -m analysis.power_spectrum --mode transformed \
    --input "$OUT_TR" --output "$OUT_PK" \
    --lbox 1000.0 --estimator div --n-bins 32

echo "=== free space: keep Pk, delete cubes ==="
rm -rf "$OUT_TR"
echo "=== done ==="
