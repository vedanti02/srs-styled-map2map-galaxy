#!/bin/bash
# SMOKE TEST of the flow-matching pipeline on real data (20 train sims, 2 epochs, 4 val boxes),
# then sample 4 val boxes x 2 draws, run the spread eval, PQMass path check, and the isotropy check.
#   sbatch flow_matching/slurm/smoke_fm.slurm
#SBATCH --job-name=fm_smoke
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --account=bapoczos
#SBATCH --output=logs/%j_fm_smoke.log
#SBATCH --error=logs/%j_fm_smoke.err
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
set -euo pipefail
cd /home/vkshirsa/srs-styled-map2map-galaxy
source /home/vkshirsa/venv/bin/activate
export PYTHONPATH=$PWD PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
CKPT=/data/user_data/vkshirsa/cmass-ili/models/fm_smoke
SRDIR=/data/user_data/vkshirsa/cmass-ili/sr_fields/fm_smoke
rm -rf "$CKPT" "$SRDIR"; mkdir -p "$CKPT" "$SRDIR" logs
echo "smoke on $(hostname)"; nvidia-smi -L
echo "=== 0) unit tests ==="; python -m flow_matching.tests.test_fm
echo "=== 1) isotropy check (decides --aug full48 vs los16) ==="; python -m flow_matching.check_isotropy --n-boxes 8
echo "=== 2) train: 20 sims, 2 epochs ==="
python -m flow_matching.train_fm --ckpt-dir "$CKPT" --source gaussian --cache --max-train-sets 20 --epochs 2 \
    --stats-boxes 20 --val-max-sims 4 --val-draws 2 --val-steps 8 --log-every 5 --num-workers 4
echo "=== 3) resume for 1 more epoch (checks last.pt round-trip) ==="
python -m flow_matching.train_fm --ckpt-dir "$CKPT" --source gaussian --cache --max-train-sets 20 --epochs 3 \
    --stats-boxes 20 --val-max-sims 4 --val-draws 2 --val-steps 8 --log-every 5 --num-workers 4 --resume auto
echo "=== 4) sample 4 val boxes x 2 draws ==="
python -m flow_matching.sample_fm --ckpt "$CKPT/best.pt" --out "$SRDIR" --split val --n-draws 2 --steps 8 --max-sims 4
ls -la "$SRDIR" | head
echo "=== 5) spread eval ==="; python -m flow_matching.eval_spread "$SRDIR" fm_smoke --draws 2 --n-boxes 4
echo "=== 6) existing eval scripts read the output? ==="; python analysis/pqmass_cmass.py "$SRDIR" fm_smoke | tail -5
echo "SMOKE_DONE"
