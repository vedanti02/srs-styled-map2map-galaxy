#!/bin/bash
#SBATCH --job-name=gensr
#SBATCH --partition=ghx4
#SBATCH --account=bdne-dtai-gh
#SBATCH --output=logs/%j_gensr.log
#SBATCH --error=logs/%j_gensr.err
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; source scripts/env_delta.sh
export PYTHONPATH=$PWD PYTHONUNBUFFERED=1
: "${CKPT:?}"; : "${OUT:?}"; : "${TMODE:?}"
mkdir -p "$OUT"
python gen_sr_fields.py --ckpt "$CKPT" --out "$OUT" --split train --theta-mode "$TMODE"
python gen_sr_fields.py --ckpt "$CKPT" --out "$OUT" --split val   --theta-mode "$TMODE"
echo "GENSR_ALL_DONE tmode=$TMODE out=$OUT"
