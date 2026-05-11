#!/bin/bash
#SBATCH --job-name=pk_lr
#SBATCH --partition=cpu
#SBATCH --output=logs/%j_pk_lr.log
#SBATCH --error=logs/%j_pk_lr.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00

set -euo pipefail
cd /home/vkshirsa/SR-styled-map2map
source /home/vkshirsa/venv/bin/activate
mkdir -p runs/baseline/pk/lr logs

export PYTHONPATH=/home/vkshirsa/SR-styled-map2map:${PYTHONPATH:-}
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

python -m analysis.power_spectrum --mode stitched \
    --input /data/group_data/universedata/lagrangian_output_64/stitched \
    --output runs/baseline/pk/lr \
    --kind quijotelike --lbox 1000.0 --estimator div --n-bins 32
