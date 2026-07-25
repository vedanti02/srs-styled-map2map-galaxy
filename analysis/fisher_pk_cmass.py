"""Fisher forecast for the 5 cosmo params from a single-box P(k) summary.
Answers the 'information ceiling' question: which of Om/Ob/h/ns/s8 does one
CMASS-ILI box constrain at all, independent of the inference method? Separates
'data has no info' (Fisher sigma ~ prior) from 'NDE/CNN failed to extract it'.

Method (emulator Fisher, standard for a Latin-hypercube design):
  d_b = log P(k_b)                          data vector (valid bins only)
  emulator d(theta) via least squares       -> derivatives D = dd/dtheta at fiducial
  residual covariance C = cov(d - d_emu)     = scatter at fixed cosmo
                                               (cosmic variance + shot noise,
                                                exactly the single-box noise)
  F_data = D^T C^-1 D   (Hartlap-corrected C^-1)
  F_post = F_data + F_prior   (uniform prior, var 1/12 in normalized units)
  sigma_i = sqrt((F_post^-1)_ii);  ratio_i = sigma_i / sigma_prior
  ratio ~ 1  -> parameter UNCONSTRAINED by one box;  ratio << 1 -> constrained.
Fisher is invariant to per-bin rescaling, so log + standardization only help
conditioning. Derivatives taken with both a linear emulator (average sensitivity
over the prior) and a quadratic emulator evaluated at the prior centre (local
sensitivity); agreement between the two makes the verdict robust."""
import numpy as np, os, sys, itertools

PK   = sys.argv[1] if len(sys.argv) > 1 else "runs/patch_cmass/pk_hr"
LABEL= sys.argv[2] if len(sys.argv) > 2 else "HR"
FIX  = (len(sys.argv) > 3 and sys.argv[3] == "fix")        # apply string-sort label fix
N    = 2000
LO = np.array([0.10,0.03,0.50,0.80,0.60]); HI = np.array([0.50,0.07,0.90,1.20,1.00])
NAMES = ["Om","Ob","h","ns","s8"]
theta = np.load("data/cmass_theta.npz")["theta"]           # (2000,5), numeric idx
# BUG: processed fields (and pk_set{k}) are ordered by STRING-sorted sim id, but
# theta is numeric. field/pk index k corresponds to true sim string_sorted[k].
string_sorted = sorted(range(N), key=str)

# ---- load P(k) ----
PKS=[]; idxs=[]; K=None
for i in range(N):
    f=f"{PK}/pk_set{i}_transformed.npz"
    if not os.path.exists(f): continue
    d=np.load(f)
    if K is None: K=d["k"]
    PKS.append(d["pk"]); idxs.append(i)
PKS=np.array(PKS); idxs=np.array(idxs)
th = theta[[string_sorted[i] for i in idxs]] if FIX else theta[idxs]
print(f"[{LABEL}] label mapping: {'STRING-SORT FIX' if FIX else 'numeric (current/buggy)'}")
print(f"[{LABEL}] loaded {len(idxs)} sims, {PKS.shape[1]} raw k-bins")

# valid bins: positive across ALL sims (drops k=0 / zero-mode bins, keeps log safe)
valid = np.all(PKS>0, axis=0)
Kv=K[valid]; P=PKS[:,valid]
D_ = np.log(P)                                              # log data vector (Nsim, nb)
nb = D_.shape[1]
print(f"[{LABEL}] {nb} valid bins, k=[{Kv.min():.3f},{Kv.max():.3f}] h/Mpc")

# normalized params in [0,1]
X=((th-LO)/(HI-LO)).astype(np.float64)                     # (Nsim,5)
# sanity: log-P amplitude must track s8 if indexing is right
amp=D_.mean(1); c_s8=np.corrcoef(amp,X[:,4])[0,1]; c_om=np.corrcoef(amp,X[:,0])[0,1]
print(f"[{LABEL}] sanity corr(mean logP, s8)={c_s8:+.3f} corr(,Om)={c_om:+.3f}  (expect s8>0 strong)")

sig_prior = 1/np.sqrt(12.0)                                # uniform[0,1] std = 0.2887
Fprior = np.eye(5)/sig_prior**2

def fisher_from(Dmat, resid):
    """Dmat: (nb,5) derivatives; resid: (Nsim,nb) emulator residuals."""
    C = np.cov(resid, rowvar=False)                        # (nb,nb)
    Ns=resid.shape[0]
    Cinv = np.linalg.inv(C) * (Ns-nb-2)/(Ns-1)             # Hartlap debias
    Fdata = Dmat.T @ Cinv @ Dmat
    Fpost = Fdata + Fprior
    sig = np.sqrt(np.diag(np.linalg.inv(Fpost)))
    sig_cond = 1/np.sqrt(np.clip(np.diag(Fdata),1e-12,None))  # others fixed
    return sig, sig_cond, Fdata

# ---- linear emulator: average slope over the prior ----
A = np.hstack([np.ones((len(X),1)), X])                    # (Nsim,6)
coef,_,_,_ = np.linalg.lstsq(A, D_, rcond=None)            # (6,nb)
Dlin = coef[1:].T                                          # (nb,5) slopes
res_lin = D_ - A@coef
R2lin = 1 - res_lin.var(0)/D_.var(0)

# ---- quadratic emulator: local sensitivity at prior centre ----
def design(Xm):
    cols=[np.ones((len(Xm),1)), Xm, Xm**2]
    for i,j in itertools.combinations(range(5),2):
        cols.append((Xm[:,i]*Xm[:,j])[:,None])
    return np.hstack(cols)
Aq=design(X)
cq,_,_,_=np.linalg.lstsq(Aq, D_, rcond=None)
res_q = D_ - Aq@cq
R2q = 1 - res_q.var(0)/D_.var(0)
Xf=np.full((1,5),0.5); eps=1e-2
Dquad=np.zeros((nb,5))
for i in range(5):
    Xp=Xf.copy(); Xp[0,i]+=eps; Xm=Xf.copy(); Xm[0,i]-=eps
    Dquad[:,i]=((design(Xp)@cq)-(design(Xm)@cq))[0]/(2*eps)

print(f"[{LABEL}] emulator median R^2: linear={np.median(R2lin):.3f} quad={np.median(R2q):.3f}")

for tag,Dmat,resid in [("linear(avg-slope)",Dlin,res_lin),
                       ("quad(centre)",Dquad,res_q)]:
    sig,sigc,Fdata=fisher_from(Dmat,resid)
    ratio=sig/sig_prior
    print(f"\n[{LABEL}] === {tag} ===")
    print(f"  {'param':4} {'sigma/prior':>11} {'marg_sigma':>11} {'cond_sigma':>11}  verdict")
    for i,nm in enumerate(NAMES):
        v = "CONSTRAINED" if ratio[i]<0.5 else ("weak" if ratio[i]<0.8 else "UNCONSTRAINED")
        print(f"  {nm:4} {ratio[i]:>11.3f} {sig[i]:>11.3f} {sigc[i]:>11.3f}  {v}")
