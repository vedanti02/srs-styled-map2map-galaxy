#!/bin/bash
# Tier-3 FIELD-level cross-fidelity for a flow-matching arm (the paper's main metric), exactly
# as done for the GAN (research_scratch_4 D-F18/D-F25): reuse the existing HR-reference and LR
# field CNNs (symlinked), train 3 SR field CNNs (seeds 100-102) on this arm's SR train fields,
# evaluate all on HR test fields -> KL(q_HR || q_SR) per box, paired ranking vs LR, HR floor.
# Needs SR fields for the train split (tier2_fm) and the first 40 val boxes (tier1_fm).
#   sbatch --export=ALL,TAG=fm_gauss flow_matching/slurm/tier3_field_fm.slurm
#SBATCH --job-name=fm_tier3
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --account=bapoczos
#SBATCH --output=logs/%j_fm_tier3.log
#SBATCH --error=logs/%j_fm_tier3.err
#SBATCH --requeue
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00
set -euo pipefail
cd /home/vkshirsa/srs-styled-map2map-galaxy
: "${TAG:?}"
SRDIR=/data/user_data/vkshirsa/cmass-ili/sr_fields/${TAG}
CK=runs/patch_cmass/field_ckpts_${TAG}
REF=/home/vkshirsa/srs-styled-map2map-galaxy/runs/patch_cmass/field_ckpts
mkdir -p "$CK" logs
source /home/vkshirsa/venv/bin/activate
export PYTHONPATH=$PWD PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
echo "Tier-3 field-level [$TAG] on $(hostname); SR from $SRDIR"
n_tr=$(ls "$SRDIR"/sr_*.npy 2>/dev/null | grep -vc '_d[0-9]' || true)
echo "SR files present: $n_tr (need train 1600 + val 200 + test 200 = 2000)"
[ "$n_tr" -ge 1840 ] || { echo "not enough SR fields (train + first 40 val required)"; exit 1; }
for f in "$REF"/qhr_s*.pt "$REF"/qlr_s*.pt; do ln -sf "$f" "$CK/$(basename "$f")"; done
for SEED in 100 101 102; do
  OUT="$CK/qsr_s${SEED}.pt"
  if [ -f "$OUT" ]; then echo "have $OUT"; continue; fi
  echo "=== SR field CNN seed $SEED ==="
  python train_field_cnn.py --source sr --sr-dir "$SRDIR" --seed "$SEED" --out "$OUT" --epochs 40
done
echo "=== field cross-fid eval ==="
python eval_field_crossfid.py --ckpt-dir "$CK" --out "$CK/field_crossfid.npz"
echo "TIER3_DONE $TAG"
