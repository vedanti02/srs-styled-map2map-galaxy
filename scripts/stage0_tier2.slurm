#!/bin/bash
# Stage-0 TIER-2: summary cross-fidelity for one surviving arm (the GATE for a winner).
#   gen 1600 train SR (fiducial) -> P(k) set -> 4 NDE seeds -> ensemble cross-fid vs HR (padding_crossfid).
#   sbatch --export=ALL,TAG=stage0_R,CKPT=/data/user_data/vkshirsa/cmass-ili/models/stage0_R/best.pt scripts/stage0_tier2.slurm
# gen_sr_fields skips existing files, so a preempted/requeued job resumes the generation.
#SBATCH --job-name=stage0t2
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --account=bapoczos
#SBATCH --output=logs/%j_stage0t2.log
#SBATCH --error=logs/%j_stage0t2.err
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
set -euo pipefail
cd /home/vkshirsa/srs-styled-map2map-galaxy
: "${TAG:?}"; : "${CKPT:?}"
SRDIR=/data/user_data/vkshirsa/cmass-ili/sr_fields/${TAG}
PKDIR=runs/patch_cmass/pk_${TAG}_train
source /home/vkshirsa/venv/bin/activate
export PYTHONPATH=$PWD PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
echo "Tier-2 [$TAG] on $(hostname)  ckpt=$CKPT"
echo "=== 1) train SR (1600 boxes, fiducial) ==="; python gen_sr_fields.py --ckpt "$CKPT" --out "$SRDIR" --split train --theta-mode fiducial
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
echo "=== 4) ensemble summary cross-fid (vs HR ensemble; LR/Afixfid rows are the comparison) ==="
python analysis/padding_crossfid.py "$TAG"
echo "TIER2_DONE $TAG"
