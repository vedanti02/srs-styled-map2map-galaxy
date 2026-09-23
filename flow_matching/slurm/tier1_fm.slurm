#!/bin/bash
# Tier-1 held-out evaluation of a flow-matching arm, mirroring scripts/stage0_eval.slurm:
#   sample val+test (1 draw) -> PQMass (val) -> bispectrum (test) -> halo-count deficit
#   -> 8 draws on 60 val boxes -> spread/calibration eval (T(k), r(k), SSR, rank hist)
#   sbatch --export=ALL,TAG=fm_gauss flow_matching/slurm/tier1_fm.slurm
#SBATCH --job-name=fm_tier1
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --account=bapoczos
#SBATCH --output=logs/%j_fm_tier1.log
#SBATCH --error=logs/%j_fm_tier1.err
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
set -euo pipefail
cd /home/vkshirsa/srs-styled-map2map-galaxy
: "${TAG:?}"
CKPT=${CKPT:-/data/user_data/vkshirsa/cmass-ili/models/${TAG}/best.pt}
STEPS=${STEPS:-32}; SHIFT=${SHIFT:-0.0}
SRDIR=/data/user_data/vkshirsa/cmass-ili/sr_fields/${TAG}
SPREAD=/data/user_data/vkshirsa/cmass-ili/sr_fields/${TAG}_draws
mkdir -p "$SRDIR" "$SPREAD" figures_cmass runs/patch_cmass logs
source /home/vkshirsa/venv/bin/activate
export PYTHONPATH=$PWD PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
echo "Tier-1 [$TAG] ckpt=$CKPT steps=$STEPS shift=$SHIFT on $(hostname)"
python -m flow_matching.sample_fm --ckpt "$CKPT" --out "$SRDIR" --split val  --steps "$STEPS" --decode-shift "$SHIFT"
python -m flow_matching.sample_fm --ckpt "$CKPT" --out "$SRDIR" --split test --steps "$STEPS" --decode-shift "$SHIFT"
echo; echo "=== PQMass (val, 200 boxes) ==="; python analysis/pqmass_cmass.py "$SRDIR" "$TAG"
echo; echo "=== bispectrum (test, 32 boxes) ==="
python analysis/bispectrum_cmass.py --mode dirs --sources HR=label LR=input "SR=$SRDIR" --which-sids test_sids --max-sims 32 --tag "$TAG"
echo; echo "=== halo-count deficit ==="; python analysis/halo_count_deficit.py "$SRDIR" "$TAG"
echo; echo "=== 8 draws x 60 val boxes -> spread eval ==="
python -m flow_matching.sample_fm --ckpt "$CKPT" --out "$SPREAD" --split val --n-draws 8 --steps "$STEPS" --max-sims 60 --decode-shift "$SHIFT"
python -m flow_matching.eval_spread "$SPREAD" "$TAG" --draws 8 --n-boxes 60
echo "TIER1_DONE $TAG"
