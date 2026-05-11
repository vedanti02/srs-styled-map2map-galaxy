#!/usr/bin/env bash
# End-to-end pipeline:
#   1. train GAN
#   2. transform LR test sims with multiple noise seeds
#   3. compute Pk on quijote (HR), quijotelike (LR), and transformed (SR)
#   4. train two NDE posteriors q_HR and q_SR on training set
#   5. evaluate posterior agreement on test set
#
# Edit the variables below before running.
set -euo pipefail

DATA_ROOT=/data/group_data/universedata/lagrangian_output_64/stitched
WORK=runs/baseline
CKPT_DIR=$WORK/checkpoints
TRANSFORM_DIR=$WORK/transformed
PK_HR_DIR=$WORK/pk/hr
PK_LR_DIR=$WORK/pk/lr
PK_SR_DIR=$WORK/pk/sr
NDE_DIR=$WORK/nde
LBOX_MPC=1000.0

mkdir -p "$WORK" "$CKPT_DIR" "$TRANSFORM_DIR" \
         "$PK_HR_DIR" "$PK_LR_DIR" "$PK_SR_DIR" "$NDE_DIR"

# 1. Train (single-GPU defaults; override on the CLI for production).
python train.py \
    --data-root "$DATA_ROOT" \
    --ckpt-dir "$CKPT_DIR" \
    --epochs 100 --batch-size 4 --num-workers 4 \
    --chan-base-g 256 --chan-base-d 64 --num-blocks 4 \
    --save-every 5 --log-every 50

CKPT=$(ls -t "$CKPT_DIR"/epoch_*.pt | head -1)
echo "using checkpoint $CKPT"

# 2. Transform test split — K=5 posterior samples per sim.
python transform.py \
    --model-path "$CKPT" \
    --data-root "$DATA_ROOT" \
    --output-dir "$TRANSFORM_DIR" \
    --split test \
    --n-noise-samples 5 --base-seed 0

# 3a. Pk on quijote (HR) — train + test.
python -m analysis.power_spectrum --mode stitched \
    --input "$DATA_ROOT" --output "$PK_HR_DIR" \
    --kind quijote --lbox $LBOX_MPC --estimator div

# 3b. Pk on quijotelike (LR) — for diagnostic comparison.
python -m analysis.power_spectrum --mode stitched \
    --input "$DATA_ROOT" --output "$PK_LR_DIR" \
    --kind quijotelike --lbox $LBOX_MPC --estimator div

# 3c. Pk on transformed cubes.
python -m analysis.power_spectrum --mode transformed \
    --input "$TRANSFORM_DIR" --output "$PK_SR_DIR" \
    --lbox $LBOX_MPC --estimator div

# 4. Train two NDEs (need pip install sbi).
python -m inference.nde \
    --pk-dir "$PK_HR_DIR" --stitched-root "$DATA_ROOT" \
    --out "$NDE_DIR/posterior_hr.pkl"
python -m inference.nde \
    --pk-dir "$PK_SR_DIR" --stitched-root "$DATA_ROOT" \
    --out "$NDE_DIR/posterior_sr.pkl"

# 5. Compare on test split.
python evaluate.py \
    --posterior-hr "$NDE_DIR/posterior_hr.pkl" \
    --posterior-sr "$NDE_DIR/posterior_sr.pkl" \
    --pk-hr-dir "$PK_HR_DIR" --pk-sr-dir "$PK_SR_DIR" \
    --stitched-root "$DATA_ROOT" \
    --output "$WORK/metrics.npz"
