"""Alternative reading of the paper's PDF-JS: computed on the RAW sampled tuple z-values
(the marginal intensity pdf of the generated tuples) instead of on the cell-averaged
64x64 field. Same metrics.pdf_js, 50 bins."""
import sys, os, json, numpy as np, torch
sys.path.insert(0,'.')
import common; from common import *
import codec
from metrics import pdf_js
from iqp_exact import IQPExact
OUT="<REPO_ROOT>/code/outputs_d2_main"
GT=gt_fields(); P=pools(); A=[int(a) for a in ANCHORS]; H=[int(h) for h in HOLDOUT]
# data-side reference: dequantized z of the native pool, pooled over the 200 held-out tau
zdat=np.concatenate([common._ZLUT[P[T2I[t]].astype(np.int64)&63] for t in H[:40]])
print("data tuple-z reference: n=%d mean=%.4f std=%.4f"%(zdat.size,zdat.mean(),zdat.std()))
res={}
for arm,K in [("iqp_K7_seed0",7),("iqp_K6_seed0",6)]:
    sim=IQPExact(n=18,K=K,device="cpu",dtype=torch.complex64)
    tc=torch.as_tensor(np.load(os.path.join(OUT,arm,"theta_core.npy")))
    lat=np.load(os.path.join(OUT,arm,"theta_lat_anchors.npy"))
    m=json.load(open(os.path.join(OUT,arm,"meta.json")))
    def il(tau):
        return np.array([np.interp(float(tau),np.array(A,dtype=float),lat[:,i]) for i in range(lat.shape[1])])
    g=torch.Generator(); g.manual_seed(999); zs=[]
    for t in H[:40]:
        th=torch.cat((torch.as_tensor(il(t),dtype=torch.float32),tc))
        c=sim.sample(th,100000,generator=g).numpy()
        zs.append(common._ZLUT[c&63])
    z=np.concatenate(zs)
    res[arm]=float(pdf_js(zdat,z)); print("%-22s tuple-level PDF-JS = %.5f (field-level %.5f)"%(arm,res[arm], 0.2354 if K==7 else 0.2660))
# RBM / CTRL-marginal / DCGAN do not store tuples; approximate their tuple-z marginal is
# only available for CTRL-marginal analytically -> skip, but give the FIELD-level numbers for context.
print("\nFor comparison the paper reports IQP PDF-JS = 0.024 on D2 (Table 2).")
json.dump(res,open("q1b_tuple_js.json","w"),indent=1)
