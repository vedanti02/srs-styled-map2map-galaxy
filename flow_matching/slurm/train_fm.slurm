#!/bin/bash
# Full training of one flow-matching arm. Auto-resumes from <ckpt>/last.pt (requeue-safe).
#   sbatch --job-name=fm_fm_gauss --export=ALL,TAG=fm_gauss,SOURCE=gaussian flow_matching/slurm/train_fm.slurm
#   (the script renames itself to fm_<TAG> anyway; --no-dequant is recommended, see README)
#   sbatch --export=ALL,TAG=fm_lfwhite,SOURCE=lf_white flow_matching/slurm/train_fm.slurm
#   sbatch --export=ALL,TAG=fm_lfspec,SOURCE=lf_spectral flow_matching/slurm/train_fm.slurm
#   sbatch --export=ALL,TAG=fm_gauss_nocond,SOURCE=gaussian,EXTRA="--no-cond" flow_matching/slurm/train_fm.slurm
# Set AUG=los16 if check_isotropy shows redshift-space anisotropy along z.
#SBATCH --job-name=fm_train
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --account=bapoczos
#SBATCH --output=logs/%j_fm_train.log
#SBATCH --error=logs/%j_fm_train.err
#SBATCH --open-mode=append
# (no --requeue: the afterany chain below restarts the job; requeue + chain ran two copies at once)
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=10:00:00
set -euo pipefail
cd /home/vkshirsa/srs-styled-map2map-galaxy
: "${TAG:?}"; : "${SOURCE:?}"
EPOCHS=${EPOCHS:-60}; AUG=${AUG:-full48}; EXTRA=${EXTRA:-}
CKPT=/data/user_data/vkshirsa/cmass-ili/models/${TAG}
mkdir -p "$CKPT" logs
# Self-chaining, idempotent. The preempt partition has PreemptMode=REQUEUE, so a preempted job
# restarts under the SAME job id and re-runs this script; SLURM also has no requeue on time
# limit. So: rename this job to fm_<TAG>, and queue ONE follow-up (afterany) with that name only
# if none is already pending. The follow-up exits at once when $CKPT/TRAIN_DONE exists.
MAX_CHAIN=${MAX_CHAIN:-6}
scontrol update JobId="$SLURM_JOB_ID" JobName="fm_${TAG}" || true
if [ -f "$CKPT/TRAIN_DONE" ]; then echo "TRAIN_DONE marker present for $TAG; nothing to do"; exit 0; fi
if squeue -u "$USER" -h -n "fm_${TAG}" -t PENDING -o %i | grep -q .; then
  echo "follow-up for $TAG already pending; not queuing another"
elif [ "$(cat "$CKPT/chain_count" 2>/dev/null || echo 0)" -ge "$MAX_CHAIN" ]; then
  echo "MAX_CHAIN=$MAX_CHAIN reached for $TAG; not queuing a follow-up"
else
  NEXT=$(sbatch --parsable --job-name="fm_${TAG}" --dependency=afterany:$SLURM_JOB_ID \
         --export=ALL,TAG=$TAG,SOURCE=$SOURCE,EPOCHS=$EPOCHS,AUG=$AUG,EXTRA="$EXTRA",MAX_CHAIN=$MAX_CHAIN \
         flow_matching/slurm/train_fm.slurm)
  echo "$(( $(cat "$CKPT/chain_count" 2>/dev/null || echo 0) + 1 ))" > "$CKPT/chain_count"
  echo "queued follow-up job $NEXT for $TAG (chain link $(cat "$CKPT/chain_count"))"
fi
source /home/vkshirsa/venv/bin/activate
export PYTHONPATH=$PWD PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
echo "train $TAG source=$SOURCE aug=$AUG epochs=$EPOCHS extra='$EXTRA' on $(hostname) $(date)"; nvidia-smi -L
python -m flow_matching.train_fm --ckpt-dir "$CKPT" --source "$SOURCE" --aug "$AUG" --epochs "$EPOCHS" \
    --cache --num-workers 4 --val-max-sims 16 --val-draws 2 --val-steps 16 --select l1 --resume auto $EXTRA
echo "TRAIN_DONE $TAG $(date)"
