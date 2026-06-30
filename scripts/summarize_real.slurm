#!/bin/bash
#SBATCH --job-name=sumreal
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%j_sumreal.log
#SBATCH --error=logs/%j_sumreal.err

set -euo pipefail
cd /home/vkshirsa/srs-styled-map2map-galaxy
source /home/vkshirsa/venv/bin/activate
export PYTHONPATH=/home/vkshirsa/srs-styled-map2map-galaxy:${PYTHONPATH:-} PYTHONUNBUFFERED=1

echo "=== inputs present? ==="
ls -la runs/patch_real/seam_realA_naive.md runs/patch_real/seam_realB_overlap.md \
       runs/patch_real/metrics_realA_naive.npz runs/patch_real/metrics_realB_overlap.npz 2>&1 || true

echo "=== summarize_patch_real ==="
python -m analysis.summarize_patch_real
echo "=== summary done ==="
