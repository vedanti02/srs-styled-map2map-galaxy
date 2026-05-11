#!/bin/bash
#SBATCH --job-name=sr_eval_v6
#SBATCH --partition=general
#SBATCH --output=logs/%j_eval_v6.log
#SBATCH --error=logs/%j_eval_v6.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --dependency=afterany:7880316

set -euo pipefail
cd /home/vkshirsa/SR-styled-map2map
source /home/vkshirsa/venv/bin/activate

# Evaluate v6 at z=0 so KL is directly comparable to v2/v3/v4/v5
EVAL_Z=${EVAL_Z:-0.0}
CKPT=checkpoints/v6/best.pt
if [ ! -f "$CKPT" ]; then
  CKPT=$(ls -1v checkpoints/v6/epoch_*.pt 2>/dev/null | tail -1)
fi
if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then echo "no v6 checkpoint found"; exit 1; fi
TAG="v6_$(basename "$CKPT" .pt)_z$(printf '%g' "$EVAL_Z")"
TAG=$(echo "$TAG" | tr -d '.')
# but keep the dot in z for clarity
TAG="v6_$(basename "$CKPT" .pt)"
if [ "$EVAL_Z" != "0.0" ]; then
  TAG="${TAG}_z$(printf '%g' "$EVAL_Z" | tr -d '.')"
fi
echo "Using checkpoint: $CKPT, tag: $TAG, eval_z=$EVAL_Z"

OUT_TR=runs/baseline/transformed_${TAG}
OUT_PK=runs/baseline/pk/sr_${TAG}
NDE_OUT=runs/baseline/nde/posterior_sr_${TAG}.pkl
EVAL_OUT=runs/baseline/metrics_${TAG}.npz

mkdir -p "$OUT_TR" "$OUT_PK" "$(dirname "$NDE_OUT")" logs

export PYTHONPATH=/home/vkshirsa/SR-styled-map2map:${PYTHONPATH:-}
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

echo "=== 1) Inference on ALL 2000 sims with $CKPT (z=$EVAL_Z) ==="
python transform.py \
    --model-path "$CKPT" \
    --data-root /data/group_data/universedata/lagrangian_output_64/stitched \
    --output-dir "$OUT_TR" \
    --split all --denormalize --save-dtype float16 \
    --n-noise-samples 1 --base-seed 0 \
    --eval-z "$EVAL_Z"

echo "=== 2) Pk on transformed cubes ==="
python -m analysis.power_spectrum --mode transformed \
    --input "$OUT_TR" --output "$OUT_PK" \
    --lbox 1000.0 --estimator div --n-bins 32

echo "=== 3) Diagnostic compare HR/LR/SR ==="
python -m analysis.compare_pk \
    --pk-hr-dir runs/baseline/pk/hr \
    --pk-lr-dir runs/baseline/pk/lr \
    --pk-sr-dir "$OUT_PK" \
    --output "runs/baseline/pk/compare_${TAG}"

echo "=== 4) Train q_SR NDE ==="
python -m inference.nde \
    --pk-dir "$OUT_PK" \
    --stitched-root /data/group_data/universedata/lagrangian_output_64/stitched \
    --out "$NDE_OUT" --n-train 1600 \
    --theta-low 0.0 --theta-high 1.5

echo "=== 5) Posterior eval q_HR vs q_SR ==="
python evaluate.py \
    --posterior-hr runs/baseline/nde/posterior_hr.pkl \
    --posterior-sr "$NDE_OUT" \
    --pk-hr-dir runs/baseline/pk/hr \
    --pk-sr-dir "$OUT_PK" \
    --pk-hr-prefix "pk_quijote_" --pk-sr-prefix "pk_set" \
    --stitched-root /data/group_data/universedata/lagrangian_output_64/stitched \
    --n-samples 2000 --output "$EVAL_OUT"

echo "=== free space: keep Pk arrays, delete cubes ==="
rm -rf "$OUT_TR"

echo "=== full v6 eval done (tag=$TAG) ==="
