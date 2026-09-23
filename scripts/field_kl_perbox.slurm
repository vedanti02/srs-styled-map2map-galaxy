#!/bin/bash
# Per-box field cross-fid KL vs LR halo-count deficit (D-F28 test). Reuses the
# eval_field_crossfid machinery; needs /data (HR test fields + raw counts) => compute node.
#SBATCH --job-name=fieldklpb
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --account=bapoczos
#SBATCH --output=logs/%j_fieldklpb.log
#SBATCH --error=logs/%j_fieldklpb.err
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
set -euo pipefail
cd /home/vkshirsa/srs-styled-map2map-galaxy
source /home/vkshirsa/venv/bin/activate
export PYTHONPATH=$PWD PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
echo "node: $(hostname)"; mkdir -p figures_cmass runs/patch_cmass
python analysis/field_kl_perbox.py
echo "FIELDKLPB_ALL_DONE"
