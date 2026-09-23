#!/bin/bash
# Gate: run Tier-1 for fm_gauss_nodq_lw2 as soon as its training chain has written TRAIN_DONE.
# Fires after the current chain link; if training is not done (preempted again), re-gates itself
# on the newest training link and exits, so no slot is held idle. Needs /data => compute node.
#SBATCH --job-name=lw2_t1gate
#SBATCH --partition=ghx4
#SBATCH --account=bdne-dtai-gh
#SBATCH --gpus-per-node=1   # DeltaAI rejects GPU-less jobs
#SBATCH --output=logs/%j_lw2gate.log
#SBATCH --error=logs/%j_lw2gate.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; source scripts/env_delta.sh
TAG=fm_gauss_nodq_lw2; CK=$SRS_CMASS_ROOT/models/$TAG
if [ -f "$CK/TRAIN_DONE" ]; then
  J=$(sbatch --parsable --export=ALL,TAG=$TAG flow_matching/slurm/tier1_fm.slurm)
  echo "TRAIN_DONE present -> Tier-1 submitted: $J  $(date)"; exit 0
fi
NEXT=$(squeue -u "$USER" -h -n "fm_$TAG" -o "%i" 2>/dev/null | sort -n | tail -1)
if [ -z "$NEXT" ]; then echo "no training link in queue and no TRAIN_DONE: chain stalled, not re-gating  $(date)"; exit 0; fi
G=$(sbatch --parsable --dependency=afterany:$NEXT scripts/lw2_tier1_gate.slurm)
echo "TRAIN_DONE absent; re-gated as $G after training link $NEXT  $(date)"
