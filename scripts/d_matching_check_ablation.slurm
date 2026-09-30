#!/bin/bash
#SBATCH --job-name=dmatchchk2
#SBATCH --account=bapoczos
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%j_dmatchchk2.log
#SBATCH --error=logs/%j_dmatchchk2.err

set -uo pipefail
cd /home/juliahul/projects/stuff/srs-styled-map2map-galaxy
mkdir -p logs
PY=/home/juliahul/miniconda3/envs/histreg/bin/python
export PYTHONPATH=/home/juliahul/projects/stuff/srs-styled-map2map-galaxy:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1

# Same diagnostic as scripts/d_matching_check.slurm, rerun on the theta_match_ablation
# checkpoint (DEBUG_HANDOFF.md Section 10.7) to compare against the best.pt baseline
# numbers in Section 10.6.
CKPT=/data/user_data/juliahul/cmass-ili-gan-debug/theta_match_ablation/best.pt

$PY -m analysis.d_matching_check --ckpt "$CKPT" --split test --n-sims 40 --n-sweep 9

echo "D_MATCHING_CHECK_ABLATION_DONE"
