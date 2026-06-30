#!/bin/bash
#SBATCH --job-name=sumcmass
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%j_sumcmass.log
#SBATCH --error=logs/%j_sumcmass.err

set -euo pipefail
cd /home/vkshirsa/srs-styled-map2map-galaxy
source /home/vkshirsa/venv/bin/activate
export PYTHONPATH=/home/vkshirsa/srs-styled-map2map-galaxy:${PYTHONPATH:-} PYTHONUNBUFFERED=1
echo "=== inputs ==="; ls -la runs/patch_cmass/seam_*.npz runs/patch_cmass/metrics_*.npz 2>&1 || true
python -m analysis.summarize_cmass
echo "=== summary done ==="
