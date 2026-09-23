#!/bin/bash
# Tier-2 summary cross-fidelity gate for a flow-matching arm, mirroring scripts/stage0_tier2.slurm:
#   sample 1600 train boxes (1 draw) -> P(k) set -> 4 NDE seeds -> ensemble cross-fid vs HR.
#   sbatch --export=ALL,TAG=fm_gauss flow_matching/slurm/tier2_fm.slurm
#SBATCH --job-name=fm_tier2
#SBATCH --partition=ghx4
#SBATCH --account=bdne-dtai-gh
#SBATCH --output=logs/%j_fm_tier2.log
#SBATCH --error=logs/%j_fm_tier2.err
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; source scripts/env_delta.sh
: "${TAG:?}"
CKPT=${CKPT:-$SRS_CMASS_ROOT/models/${TAG}/best.pt}
STEPS=${STEPS:-32}
SRDIR=$SRS_CMASS_ROOT/sr_fields/${TAG}
PKDIR=runs/patch_cmass/pk_${TAG}_train
export PYTHONPATH=$PWD PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
echo "Tier-2 [$TAG] ckpt=$CKPT on $(hostname)"
echo "=== 1) train SR (1600 boxes) ==="; python -m flow_matching.sample_fm --ckpt "$CKPT" --out "$SRDIR" --split train --steps "$STEPS"
echo "=== 2) P(k) set ==="; python analysis/pk_from_srfields.py "$SRDIR" "$PKDIR" train
echo "=== 3) NDE seeds 0-3 on $PKDIR ==="
python - << PY
import sys, numpy as np, torch; sys.path.insert(0,".")
from inference.nde import train_nde
R="runs/patch_cmass"; tmp=f"{R}/_robust_tmp"
for s in range(4):
    torch.manual_seed(s); np.random.seed(s)
    train_nde(f"{R}/pk_${TAG}_train", "", f"{tmp}/q${TAG}_{s}.pkl", theta_npz="data/cmass_theta.npz")
    print(f"  q${TAG}_{s} trained", flush=True)
PY
echo "=== 4) ensemble summary cross-fid (vs HR ensemble; LR / Afixfid / stage0_R rows for comparison) ==="
python analysis/padding_crossfid.py stage0_R "$TAG"
echo "TIER2_DONE $TAG"
