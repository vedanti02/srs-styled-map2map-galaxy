#!/bin/bash
# Stage 0: short fine-tune of fixcond/best.pt with a rebalanced loss, P(k) loss OFF (protocol).
#   Arm R (rebalance):   sbatch --export=ALL,TAG=stage0_R,LREC=0.5,SIGMA=0.0 scripts/stage0_finetune.slurm
#   Arm S (smoothed L1): sbatch --export=ALL,TAG=stage0_S,LREC=2.0,SIGMA=1.5 scripts/stage0_finetune.slurm
# NOTE --epochs is the ABSOLUTE final epoch (loop is range(start, epochs)); computed on-node
# from the checkpoint's epoch so that EXTRA epochs are actually run.
#SBATCH --job-name=stage0ft
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --account=bapoczos
#SBATCH --output=logs/%j_stage0ft.log
#SBATCH --error=logs/%j_stage0ft.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=06:00:00
set -euo pipefail
cd /home/vkshirsa/srs-styled-map2map-galaxy
: "${TAG:?}"; : "${LREC:?}"; : "${SIGMA:?}"
RESUME=/data/user_data/vkshirsa/cmass-ili/models/patch_cmass_A_nopk_fixcond/best.pt
CKPT=/data/user_data/vkshirsa/cmass-ili/models/${TAG}
EXTRA=${EXTRA:-5}
mkdir -p logs "$CKPT"
source /home/vkshirsa/venv/bin/activate
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 PYTHONPATH=/home/vkshirsa/srs-styled-map2map-galaxy:${PYTHONPATH:-}
E0=$(python -c "import torch;print(torch.load('$RESUME',map_location='cpu',weights_only=False)['epoch'])")
EPOCHS=$((E0+1+EXTRA))
echo "Stage0 $TAG on $(hostname): resume epoch $E0 -> run epochs $((E0+1))..$((EPOCHS-1)) ($EXTRA extra) | lambda_rec=$LREC sigma=$SIGMA pk=0 adv=1 lr=5e-5 select=l1" >&2
nvidia-smi -L >&2
python train_patch_cmass.py \
    --pad 0 --ckpt-dir "$CKPT" --transform log1p --nbar perbox \
    --epochs "$EPOCHS" --sims-per-batch 1 --num-workers 8 \
    --chan-base-g 128 --chan-base-d 64 --num-blocks 4 \
    --lambda-adv 1.0 --lambda-rec "$LREC" --lambda-pk 0.0 --rec-smooth-sigma "$SIGMA" \
    --lambda-r1 10.0 --r1-every 16 \
    --pad-loss crop --select l1 \
    --lr-g 5e-5 --lr-d 5e-5 \
    --save-every 1 --log-every 200 --grad-clip 1.0 --val-max-sims 40 \
    --resume "$RESUME"
echo "STAGE0_DONE $TAG"
