#!/bin/bash
#SBATCH --job-name=sr_eval_wrapper
#SBATCH --partition=general
#SBATCH --output=logs/%j_eval_wrap.log
#SBATCH --error=logs/%j_eval_wrap.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --dependency=afterany:7839151

set -euo pipefail
cd /home/vkshirsa/SR-styled-map2map
source /home/vkshirsa/venv/bin/activate

# Use the LATEST epoch_*.pt
CKPT=$(ls -1v checkpoints/full/epoch_*.pt | tail -1)
TAG="final_$(basename "$CKPT" .pt)"
echo "Final checkpoint: $CKPT, tag: $TAG"

OUT_TR=runs/baseline/transformed_${TAG}
OUT_PK=runs/baseline/pk/sr_${TAG}
NDE_OUT=runs/baseline/nde/posterior_sr_${TAG}.pkl
EVAL_OUT=runs/baseline/metrics_${TAG}.npz

mkdir -p "$OUT_TR" "$OUT_PK" "$(dirname "$NDE_OUT")" logs

export PYTHONPATH=/home/vkshirsa/SR-styled-map2map:${PYTHONPATH:-}
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

echo "=== 1) Inference on ALL 2000 sims ==="
python transform.py \
    --model-path "$CKPT" \
    --data-root /data/group_data/universedata/lagrangian_output_64/stitched \
    --output-dir "$OUT_TR" \
    --split all --denormalize \
    --n-noise-samples 1 --base-seed 0

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

echo "=== 5) Posterior eval q_HR vs q_SR (val split held out from NDE) ==="
python evaluate.py \
    --posterior-hr runs/baseline/nde/posterior_hr.pkl \
    --posterior-sr "$NDE_OUT" \
    --pk-hr-dir runs/baseline/pk/hr \
    --pk-sr-dir "$OUT_PK" \
    --pk-hr-prefix "pk_quijote_" --pk-sr-prefix "pk_set" \
    --stitched-root /data/group_data/universedata/lagrangian_output_64/stitched \
    --n-samples 2000 --output "$EVAL_OUT"

echo "=== full eval done ==="
