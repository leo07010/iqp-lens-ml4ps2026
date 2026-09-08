"""Q5: why doesn't DCGAN@11 collapse? diversity / memorisation diagnostics for all arms."""
import sys, os, json
import numpy as np
sys.path.insert(0,'.')
import common
from common import *
OUT="<HOME>/gsoc2026-deeplense-qml/experiments/13_iqp_turbulence_replication/outputs_d2_main"
GT=gt_fields(); A=[int(a) for a in ANCHORS]; H=[int(h) for h in HOLDOUT]
anch=GT[[T2I[a] for a in A]].reshape(11,-1).astype(np.float64)
def norm(x): 
    x=x-x.mean(1,keepdims=True); return x/np.maximum(x.std(1,keepdims=True),1e-12)
anchn=norm(anch)
def load(d):
    fs=sorted([f for f in os.listdir(os.path.join(OUT,d)) if f.startswith("generated_")])
    return np.stack([np.load(os.path.join(OUT,d,f)) for f in fs]).astype(np.float64)
print("%-24s %-7s %-9s %-9s %-9s %-9s"%("arm","n","selfcorr","std","maxcorr_anchor","corr_to_gt(paired)"))
rows=[("GROUND-TRUTH holdout", GT[[T2I[t] for t in H]].astype(np.float64))]
for d in ["iqp_K7_seed0","iqp_K6_seed0","rbm_seed0","dcgan_1.2m_n11_seed0","dcgan_1.2m_n11_seed2",
          "dcgan_1.2m_n100_seed0","dcgan_0.2m_n11_seed0","dcgan_51k_n11_seed0","ctrl_interp","ctrl_marginal_seed0"]:
    rows.append((d, load(d)))
gt_h = GT[[T2I[t] for t in H]].astype(np.float64)
for name,a in rows:
    n=a.shape[0]; F=norm(a.reshape(n,-1))
    C=(F@F.T)/F.shape[1]; iu=np.triu_indices(n,1)
    mx=((F@anchn.T)/F.shape[1]).max(axis=1)
    paired = np.nan
    if n==200:
        Gn=norm(gt_h.reshape(200,-1)); paired=float(np.mean(np.sum(F*Gn,axis=1)/F.shape[1]))
    print("%-24s %-7d %-9.4f %-9.4f %-9.4f %s"%(name,n,C[iu].mean(),a.std(),mx.mean(),
          ("%.4f"%paired) if paired==paired else "n/a"))
print("\nNOTE selfcorr = mean pairwise correlation among that arm's own 200 outputs;")
print("     maxcorr_anchor = mean over outputs of the best correlation to any of the 11 TRAINING anchors;")
print("     corr_to_gt(paired) = correlation to the ground truth AT THE SAME tau (the only time-aware number).")
