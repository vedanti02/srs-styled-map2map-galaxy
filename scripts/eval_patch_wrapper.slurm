#!/bin/bash
#SBATCH --job-name=patch_eval
#SBATCH --partition=general
#SBATCH --output=logs/%j_patch_eval.log
#SBATCH --error=logs/%j_patch_eval.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

# Required env (pass via sbatch --export=ALL,CKPT=...,MODE=...,TAG=...):
#   CKPT : path to patch checkpoint (e.g. checkpoints/patch_v1/best.pt)
#   MODE : naive | overlap   (stitching mode for infer_stitch.py)
#   TAG  : output tag (e.g. patchA_naive, patchA_overlap, patchB_overlap)
set -euo pipefail
cd /home/vkshirsa/srs-styled-map2map-galaxy
source /home/vkshirsa/venv/bin/activate

: "${CKPT:?need CKPT}"; : "${MODE:?need MODE}"; : "${TAG:?need TAG}"
PAD="${PAD:--1}"   # -1 => infer_stitch takes pad from checkpoint (overlap) or forces 0 (naive)
echo "CKPT=$CKPT MODE=$MODE TAG=$TAG PAD=$PAD"

OUT_TR=runs/patch/transformed_${TAG}
OUT_PK=runs/patch/pk_${TAG}
NDE_OUT=runs/patch/posterior_sr_${TAG}.pkl
EVAL_OUT=runs/patch/metrics_${TAG}.npz
SEAM_OUT=runs/patch/seam_${TAG}
mkdir -p "$OUT_TR" "$OUT_PK" runs/patch logs

export PYTHONPATH=/home/vkshirsa/srs-styled-map2map-galaxy:${PYTHONPATH:-}
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

echo "=== 1) Patch inference + stitching on ALL 2000 sims ==="
python infer_stitch.py \
    --model-path "$CKPT" \
    --output-dir "$OUT_TR" \
    --split all --mode "$MODE" --pad "$PAD" \
    --denormalize --save-dtype float16 --base-seed 0

echo "=== 2) Seam evaluation (test split sims only are in metrics; here all saved sims) ==="
python -m analysis.eval_seam \
    --sr-dir "$OUT_TR" --out "$SEAM_OUT" --label "$TAG" --split test

echo "=== 3) Pk on stitched cubes ==="
python -m analysis.power_spectrum --mode transformed \
    --input "$OUT_TR" --output "$OUT_PK" \
    --lbox 1000.0 --estimator div --n-bins 32

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

echo "=== cleanup: delete cubes, keep Pk + metrics ==="
rm -rf "$OUT_TR"
echo "=== eval $TAG done ==="
