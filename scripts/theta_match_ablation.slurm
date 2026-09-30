#!/bin/bash
#SBATCH --job-name=thetamatch
#SBATCH --account=bapoczos
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%j_thetamatch.log
#SBATCH --error=logs/%j_thetamatch.err

set -uo pipefail
cd /home/juliahul/projects/stuff/srs-styled-map2map-galaxy
mkdir -p logs
PY=/home/juliahul/miniconda3/envs/histreg/bin/python
export PYTHONPATH=/home/juliahul/projects/stuff/srs-styled-map2map-galaxy:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1

# Cheap ablation from DEBUG_HANDOFF.md Section 10.5: --theta-norm prior + --lambda-match,
# warm-started from the SAME epoch_15.pt the lambda_rec sweep used (Section 4), with
# --lambda-rec 2.0 --lambda-pk 1.45 matching that sweep's lambda_rec=2 arm exactly --
# isolates the theta-conditioning axis against an already-documented reference point,
# so results are directly comparable to that arm rather than confounded by also changing
# lambda_rec/lambda_pk at the same time.
BASE_CKPT=/data/group_data/universedata/cmass-ili/models/patch_cmass_A_nopk_fixcond/epoch_15.pt
OUTDIR=/data/user_data/juliahul/cmass-ili-gan-debug/theta_match_ablation
mkdir -p "$OUTDIR"

# Resume-aware (see DEBUG_HANDOFF.md Section 2 gotcha #2): on a preemption requeue this
# script reruns from the top. Pick up the latest epoch_*.pt already written to OUTDIR by
# THIS run if one exists, instead of always restarting from the original warm start.
LATEST=$(ls -1 "$OUTDIR"/epoch_*.pt 2>/dev/null | sed -E 's/.*epoch_([0-9]+)\.pt/\1 &/' | sort -n | tail -1 | awk '{print $2}')
CKPT_IN="${LATEST:-$BASE_CKPT}"
echo "warm-start=$CKPT_IN  out=$OUTDIR  (restarts=${SLURM_RESTART_COUNT:-0})"

$PY train_patch_cmass.py \
    --resume "$CKPT_IN" --ckpt-dir "$OUTDIR" \
    --epochs 23 --save-every 1 --val-max-sims 40 \
    --lambda-adv 1.0 --lambda-rec 2.0 --lambda-pk 1.45 --lambda-match 1.0 \
    --theta-norm prior --select pk \
    --lr-g 1e-4 --lr-d 1e-4 --chan-base-g 128 --chan-base-d 64 --num-blocks 4 \
    --transform log1p --pad 0 --nbar perbox --seed 0

echo "THETA_MATCH_ABLATION_DONE"
