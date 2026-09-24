# Environment for NCSA DeltaAI (aarch64 GH200). Source from the repo root:  source scripts/env_delta.sh
# Slurm scripts `cd "$SLURM_SUBMIT_DIR"` first, so submit them from the repo root (sbatch scripts/x.slurm).
# The venv sits on top of the PyTorch module (--system-site-packages) and adds sbi; it was built with:
#   module load python/miniforge3_pytorch/2.10.0
#   python -m venv --system-site-packages $SRS_VENV && source $SRS_VENV/bin/activate && pip install sbi==0.27.0   # 0.28 drops SNPE_C (inference/nde.py)
if [ ! -f paths.py ]; then echo "env_delta.sh: run from the repo root (cwd=$PWD)" >&2; return 1 2>/dev/null || exit 1; fi
_srs_opts=$-; set +eu          # lmod and venv activate are not `set -u` safe
module load python/miniforge3_pytorch/2.10.0
export SRS_VENV=${SRS_VENV:-/work/nvme/bdne/vkshirsagar1/venv}
source "$SRS_VENV/bin/activate"
case $_srs_opts in *e*) set -e;; esac; case $_srs_opts in *u*) set -u;; esac; unset _srs_opts
export SRS_CMASS_ROOT=${SRS_CMASS_ROOT:-/work/nvme/bdne/vkshirsagar1/cmass-ili}   # read by paths.py
export PYTHONPATH=$PWD PYTHONUNBUFFERED=1
# torch otherwise starts one thread per node core (288 on GH200) inside an 8-cpu allocation: ~75x slower NDE training
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}
mkdir -p logs
