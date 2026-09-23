#!/bin/bash
# Stage-0 TIER-1 held-out evaluation for one arm, or the fixcond baseline (identical protocol):
#   gen SR (val+test, fiducial) -> PQMass (val) -> diag panels (test) -> bispectrum (test) -> halo count
#   sbatch --export=ALL,TAG=stage0_R,CKPT=/data/user_data/vkshirsa/cmass-ili/models/stage0_R/best.pt scripts/stage0_eval.slurm
#   baseline: TAG=Afix_fiducial CKPT=.../models/patch_cmass_A_nopk_fixcond/best.pt
# P(k) appears only as a held-out CHECK (diag transfer panel); it is never trained on (protocol item 2).
#SBATCH --job-name=stage0eval
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --account=bapoczos
#SBATCH --output=logs/%j_stage0eval.log
#SBATCH --error=logs/%j_stage0eval.err
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
set -euo pipefail
cd /home/vkshirsa/srs-styled-map2map-galaxy
: "${TAG:?}"; : "${CKPT:?}"
SRDIR=/data/user_data/vkshirsa/cmass-ili/sr_fields/${TAG}
mkdir -p "$SRDIR" figures_cmass runs/patch_cmass logs
source /home/vkshirsa/venv/bin/activate
export PYTHONPATH=$PWD PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
echo "Tier-1 eval [$TAG] on $(hostname)  ckpt=$CKPT"
python gen_sr_fields.py --ckpt "$CKPT" --out "$SRDIR" --split val  --theta-mode fiducial
python gen_sr_fields.py --ckpt "$CKPT" --out "$SRDIR" --split test --theta-mode fiducial
# bispectrum_cmass --mode dirs expects set{sid}_transformed.npy; symlink alongside sr_{sid}.npy
for f in "$SRDIR"/sr_*.npy; do s=$(basename "$f" .npy); s=${s#sr_}; ln -sf "$f" "$SRDIR/set${s}_transformed.npy"; done
echo; echo "=== PQMass (val, 200 boxes) ==="; python analysis/pqmass_cmass.py "$SRDIR" "$TAG"
echo; echo "=== diag panels (test: transfer as held-out check, cross-coherence) ==="
python -m analysis.diag_cmass --ckpt "$CKPT" --mode naive --tag "$TAG" --theta-mode fiducial --n-sims 16 --out-dir figures_cmass
echo; echo "=== bispectrum (test, 32 boxes) ==="
python analysis/bispectrum_cmass.py --mode dirs --sources HR=label LR=input "SR=$SRDIR" --which-sids test_sids --max-sims 32 --tag "$TAG"
echo; echo "=== halo-count deficit ==="; python analysis/halo_count_deficit.py "$SRDIR" "$TAG"
echo "TIER1_DONE $TAG"
