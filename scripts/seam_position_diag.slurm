#!/bin/bash
#SBATCH --job-name=seam_pos
#SBATCH --partition=general
#SBATCH --output=logs/%j_seam_pos.log
#SBATCH --error=logs/%j_seam_pos.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00

set -euo pipefail
cd /home/vkshirsa/srs-styled-map2map-galaxy
source /home/vkshirsa/venv/bin/activate
export PYTHONPATH=/home/vkshirsa/srs-styled-map2map-galaxy:${PYTHONPATH:-}
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1

PA=runs/patch/diag_patchA_naive
WB=runs/patch/diag_wholebox_v2
mkdir -p "$PA" "$WB"

# 40 test sims each
python infer_stitch.py --model-path checkpoints/patch_v1/best.pt --output-dir "$PA" \
    --split test --mode naive --denormalize --save-dtype float32
python transform.py --model-path checkpoints/v2/best.pt --output-dir "$WB" \
    --split test --denormalize --save-dtype float32 --n-noise-samples 1 --base-seed 0

python -m analysis.seam_position_profile \
    --sr-dirs "patchA_naive=$PA" "wholebox_v2=$WB" \
    --out runs/patch/seam_position_profile --max-sims 40

rm -rf "$PA" "$WB"
echo "=== seam position diag done ==="
