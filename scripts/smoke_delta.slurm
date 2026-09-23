#!/bin/bash
# DeltaAI port smoke test: env + data + theta + GAN + flow-matching + old sbi posteriors, each checked
# against a number already on record. Submit from the repo root:  sbatch scripts/smoke_delta.slurm
#   1. GAN fixcond, fiducial theta -> VAL SR fields (sr_fields/Afix_fiducial, reused by later evals)
#      -> PQMass; on record (main.tex): SR 2.2, LF 0.83
#   2. fm_gauss_nodq loads and samples 2 val boxes
#   3. restored NDE posteriors: seed-0 summary cross-fid KL; on record (plot_crossfid_honest.py): LR 0.048, SR cond 0.157
#SBATCH --job-name=smoke_delta
#SBATCH --partition=ghx4
#SBATCH --account=bdne-dtai-gh
#SBATCH --output=logs/%j_smoke.log
#SBATCH --error=logs/%j_smoke.err
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; source scripts/env_delta.sh
export OMP_NUM_THREADS=16
echo "node $(hostname) $(date)"; nvidia-smi -L
python -c "import torch,sbi,numpy;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),'sbi',sbi.__version__,'numpy',numpy.__version__)"

echo "=== 1. GAN -> val SR -> PQMass ==="
SRDIR=$SRS_CMASS_ROOT/sr_fields/Afix_fiducial
python gen_sr_fields.py --ckpt "$SRS_CMASS_ROOT/models/patch_cmass_A_nopk_fixcond/best.pt" \
    --out "$SRDIR" --split val --theta-mode fiducial
python analysis/pqmass_cmass.py "$SRDIR" Afix_fiducial_deltacheck   # new tag: don't overwrite the paper figure

echo "=== 2. flow matching ==="
python -m flow_matching.sample_fm --ckpt "$SRS_CMASS_ROOT/models/fm_gauss_nodq/best.pt" \
    --out "${TMPDIR:-/tmp}/smoke_fm_$SLURM_JOB_ID" --split val --max-sims 2 --steps 32
python -c "import numpy as np,glob;[print(f,np.load(f).shape,np.load(f).mean()) for f in sorted(glob.glob('${TMPDIR:-/tmp}/smoke_fm_$SLURM_JOB_ID/sr_*.npy'))]"

echo "=== 3. restored posteriors, seed 0 ==="
python - <<'PY'
import numpy as np, torch
from evaluate import _load_pk_set, _load_posterior, _sample, kl_gauss
R="runs/patch_cmass"; T=f"{R}/_robust_tmp"
test=set(np.load(f"{R}/split_sids.npz")["test_sids"].tolist())
pk={s:v for s,v in _load_pk_set(f"{R}/pk_hr","pk_set").items() if s in test}; common=sorted(pk)
def xfid(qh,qx,n=1500):
    ks=[]
    for s in common:
        a=_sample(qh,pk[s],n); b=_sample(qx,pk[s],n)
        ks.append(kl_gauss(a.mean(0),a.std(0),b.mean(0),b.std(0)).mean())
    return float(np.mean(ks))
qh=_load_posterior(f"{T}/qhr_0.pkl")
for tag,ref in [("LR",0.048),("Afixfid",0.157)]:
    torch.manual_seed(1000); print(f"{tag}: {xfid(qh,_load_posterior(f'{T}/q{tag}_0.pkl')):.3f}  (on record {ref})",flush=True)
PY
echo "SMOKE_DONE $(date)"
