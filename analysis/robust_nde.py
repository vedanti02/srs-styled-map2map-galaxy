"""NDE-training-seed robustness on the cross-fidelity goal metric.
Grounding: SBI posteriors (SNPE_C) have run-to-run variance from network init +
training stochasticity (Hermans et al. 2022; Lueckmann et al. 2021). A single
NDE per arm cannot support a 0.0026-vs-0.0029 comparison. Here we retrain q_HR
and each q_X together under matched seeds and report the spread of the
cross-fidelity KL (q_X trained on X-Pk, tested on HR-Pk, vs q_HR on HR-Pk).
"""
import os, sys, numpy as np, torch
sys.path.insert(0, ".")
from inference.nde import train_nde
from evaluate import _load_pk_set, _load_posterior, _sample, kl_gauss

R = "runs/patch_cmass"
THETA = "data/cmass_theta.npz"
SEEDS = [0, 1, 2, 3, 4]
# arm -> its TRAIN pk dir (for retraining q) ; all tested on HR pk
ARMS = {"Anopk": "pk_Anopk_train", "oldA": "pk_cmassA_naive_train", "LR": "pk_LRbase_train"}
HR_TRAIN = "pk_hr_train"

test_sids = set(np.load(f"{R}/split_sids.npz")["test_sids"].tolist())
pk_hr = {s: v for s, v in _load_pk_set(f"{R}/pk_hr", "pk_set").items() if s in test_sids}
common = sorted(pk_hr)
print(f"test sims: {len(common)}", flush=True)

def crossfid_kl(q_hr, q_x, n=2000):
    per = []
    for sid in common:
        x = pk_hr[sid]
        sh = _sample(q_hr, x, n); ss = _sample(q_x, x, n)
        per.append(kl_gauss(sh.mean(0), sh.std(0), ss.mean(0), ss.std(0)))
    return np.stack(per)  # (Ntest, 5)

results = {a: [] for a in ARMS}
tmp = f"{R}/_robust_tmp"; os.makedirs(tmp, exist_ok=True)
for s in SEEDS:
    print(f"=== seed {s} ===", flush=True)
    torch.manual_seed(s); np.random.seed(s)
    train_nde(f"{R}/{HR_TRAIN}", "", f"{tmp}/qhr_{s}.pkl", theta_npz=THETA)
    q_hr = _load_posterior(f"{tmp}/qhr_{s}.pkl")
    for a, d in ARMS.items():
        torch.manual_seed(s); np.random.seed(s)
        train_nde(f"{R}/{d}", "", f"{tmp}/q{a}_{s}.pkl", theta_npz=THETA)
        q_x = _load_posterior(f"{tmp}/q{a}_{s}.pkl")
        torch.manual_seed(1000 + s)  # sampling seed fixed per (arm,seed)
        kl = crossfid_kl(q_hr, q_x)
        m = float(kl.mean())
        results[a].append(m)
        print(f"  seed {s} {a}: cross-fid KL mean = {m:.4f}", flush=True)

print("\n=== ROBUSTNESS SUMMARY (cross-fid KL, mean over test; spread over 5 NDE seeds) ===")
summ = {}
for a, vals in results.items():
    v = np.array(vals); summ[a] = v
    print(f"  {a:6s}: {v.mean():.4f} +/- {v.std(ddof=1):.4f}  (seeds: {np.round(v,4).tolist()})")
# paired-by-seed new vs old
if len(results["Anopk"]) == len(results["oldA"]):
    d = summ["Anopk"] - summ["oldA"]
    print(f"\n  Anopk - oldA per seed: {np.round(d,4).tolist()}  mean {d.mean():+.4f} +/- {d.std(ddof=1):.4f}")
    dl = summ["Anopk"] - summ["LR"]
    print(f"  Anopk - LR   per seed: {np.round(dl,4).tolist()}  mean {dl.mean():+.4f} +/- {dl.std(ddof=1):.4f}")
np.savez(f"{R}/robust_nde_crossfid.npz", **{a: v for a, v in summ.items()})
print(f"\nsaved {R}/robust_nde_crossfid.npz")
