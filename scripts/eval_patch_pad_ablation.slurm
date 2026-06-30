#!/bin/bash
#SBATCH --job-name=patch_pad
#SBATCH --partition=general
#SBATCH --output=logs/%j_patch_pad.log
#SBATCH --error=logs/%j_patch_pad.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00

# Receptive-field ablation: does overlap-tile inference with pad >= RF radius (8)
# eliminate the seam? Inference + seam eval only (KL shown insensitive). Test split.
set -euo pipefail
cd /home/vkshirsa/srs-styled-map2map-galaxy
source /home/vkshirsa/venv/bin/activate
export PYTHONPATH=/home/vkshirsa/srs-styled-map2map-galaxy:${PYTHONPATH:-}
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1

run() {  # ckpt tag pad
  local CKPT=$1 TAG=$2 PAD=$3
  local TR=runs/patch/transformed_${TAG}
  mkdir -p "$TR"
  echo "=== infer $TAG (pad=$PAD) ==="
  python infer_stitch.py --model-path "$CKPT" --output-dir "$TR" \
      --split test --mode overlap --pad "$PAD" --denormalize --save-dtype float16
  python -m analysis.eval_seam --sr-dir "$TR" --out runs/patch/seam_${TAG} \
      --label "$TAG" --split test
  rm -rf "$TR"
}

run checkpoints/patch_v1/best.pt          patchA_overlap_pad8 8
run checkpoints/patch_v2_boundary/best.pt patchB_overlap_pad8 8
echo "=== pad ablation done ==="
