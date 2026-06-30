#!/bin/bash
#SBATCH --job-name=wb_v2_seam
#SBATCH --partition=general
#SBATCH --output=logs/%j_wb_v2_seam.log
#SBATCH --error=logs/%j_wb_v2_seam.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00

# Control: whole-box v2 (the SOTA reference, never saw patches) under the SAME
# seam metric. Regenerates test-split cubes (deleted by prior eval), then eval_seam.
set -euo pipefail
cd /home/vkshirsa/srs-styled-map2map-galaxy
source /home/vkshirsa/venv/bin/activate
export PYTHONPATH=/home/vkshirsa/srs-styled-map2map-galaxy:${PYTHONPATH:-}
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1

TR=runs/patch/transformed_wholebox_v2
mkdir -p "$TR"
python transform.py --model-path checkpoints/v2/best.pt --output-dir "$TR" \
    --split test --denormalize --save-dtype float16 --n-noise-samples 1 --base-seed 0
python -m analysis.eval_seam --sr-dir "$TR" --out runs/patch/seam_wholebox_v2 \
    --label "whole-box v2 (NO patches, control)" --split test
rm -rf "$TR"
echo "=== wholebox v2 seam control done ==="
