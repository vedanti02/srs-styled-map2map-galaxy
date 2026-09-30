#!/bin/bash
#SBATCH --job-name=dmatchchk
#SBATCH --account=bapoczos
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%j_dmatchchk.log
#SBATCH --error=logs/%j_dmatchchk.err

set -uo pipefail
cd /home/juliahul/projects/stuff/srs-styled-map2map-galaxy
mkdir -p logs
PY=/home/juliahul/miniconda3/envs/histreg/bin/python
export PYTHONPATH=/home/juliahul/projects/stuff/srs-styled-map2map-galaxy:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1

CKPT=/data/group_data/universedata/cmass-ili/models/patch_cmass_A_nopk_fixcond/best.pt

$PY -m analysis.d_matching_check --ckpt "$CKPT" --split test --n-sims 40 --n-sweep 9

echo "D_MATCHING_CHECK_DONE"
