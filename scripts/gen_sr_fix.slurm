#!/bin/bash
#SBATCH --job-name=gensr
#SBATCH --partition=general
#SBATCH --output=logs/%j_gensr.log
#SBATCH --error=logs/%j_gensr.err
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
set -euo pipefail
cd /home/vkshirsa/srs-styled-map2map-galaxy
source /home/vkshirsa/venv/bin/activate
export PYTHONPATH=$PWD PYTHONUNBUFFERED=1
: "${CKPT:?}"; : "${OUT:?}"; : "${TMODE:?}"
mkdir -p "$OUT"
python gen_sr_fields.py --ckpt "$CKPT" --out "$OUT" --split train --theta-mode "$TMODE"
python gen_sr_fields.py --ckpt "$CKPT" --out "$OUT" --split val   --theta-mode "$TMODE"
echo "GENSR_ALL_DONE tmode=$TMODE out=$OUT"
