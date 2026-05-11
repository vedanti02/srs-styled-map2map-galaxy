#!/bin/bash
# Full evaluation pipeline: runs after the final training checkpoint is ready.
# Submit with: CKPT=checkpoints/full/epoch_50.pt TAG=final sbatch --export=ALL,CKPT,TAG scripts/full_eval_pipeline.slurm
#SBATCH --job-name=sr_full_eval
#SBATCH --partition=general
#SBATCH --output=logs/%j_eval.log
#SBATCH --error=logs/%j_eval.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00

set -euo pipefail
cd /home/vkshirsa/SR-styled-map2map
source /home/vkshirsa/venv/bin/activate

CKPT="${CKPT:?must set CKPT}"
TAG="${TAG:?must set TAG}"

OUT_TR=runs/baseline/transformed_${TAG}
OUT_PK=runs/baseline/pk/sr_${TAG}
NDE_OUT=runs/baseline/nde/posterior_sr_${TAG}.pkl
EVAL_OUT=runs/baseline/metrics_${TAG}.npz

mkdir -p "$OUT_TR" "$OUT_PK" "$(dirname "$NDE_OUT")" logs

export PYTHONPATH=/home/vkshirsa/SR-styled-map2map:${PYTHONPATH:-}
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

echo "=== 1) Inference on ALL 2000 sims with $CKPT ==="
python transform.py \
    --model-path "$CKPT" \
    --data-root /data/group_data/universedata/lagrangian_output_64/stitched \
    --output-dir "$OUT_TR" \
    --split all \
    --denormalize \
    --n-noise-samples 1 --base-seed 0

echo "=== 2) Power spectrum on transformed cubes ==="
python -m analysis.power_spectrum --mode transformed \
    --input "$OUT_TR" --output "$OUT_PK" \
    --lbox 1000.0 --estimator div --n-bins 32

echo "=== 3) Diagnostic compare (HR vs LR vs SR) ==="
python -m analysis.compare_pk \
    --pk-hr-dir runs/baseline/pk/hr \
    --pk-lr-dir runs/baseline/pk/lr \
    --pk-sr-dir "$OUT_PK" \
    --output "runs/baseline/pk/compare_${TAG}"

echo "=== 4) Train q_SR NDE (using train-split SR Pk) ==="
python -m inference.nde \
    --pk-dir "$OUT_PK" \
    --stitched-root /data/group_data/universedata/lagrangian_output_64/stitched \
    --out "$NDE_OUT" \
    --n-train 1600 \
    --theta-low 0.0 --theta-high 1.5

echo "=== 5) Evaluate posteriors HR vs SR on val split (held out from NDE training) ==="
# Evaluate uses pk_quijote_set*.npz vs pk_set*.npz patterns
python evaluate.py \
    --posterior-hr runs/baseline/nde/posterior_hr.pkl \
    --posterior-sr "$NDE_OUT" \
    --pk-hr-dir runs/baseline/pk/hr \
    --pk-sr-dir "$OUT_PK" \
    --stitched-root /data/group_data/universedata/lagrangian_output_64/stitched \
    --n-samples 2000 \
    --output "$EVAL_OUT"

echo "=== 6) Also evaluate HR vs LR baseline (no GAN) ==="
python evaluate.py \
    --posterior-hr runs/baseline/nde/posterior_hr.pkl \
    --posterior-sr runs/baseline/nde/posterior_lr.pkl \
    --pk-hr-dir runs/baseline/pk/hr \
    --pk-sr-dir runs/baseline/pk/lr \
    --stitched-root /data/group_data/universedata/lagrangian_output_64/stitched \
    --n-samples 2000 \
    --output "runs/baseline/metrics_lr_baseline.npz"

echo "=== full eval done ==="
