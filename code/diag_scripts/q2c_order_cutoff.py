"""Decisive test of the 'objective is low-order-blind' hypothesis, with NO training:
project the DATA pmf onto Walsh order <= L, resample it through the identical
sample+aggregate pipeline, and score it. If order<=3 lands at IQP's score, the MMD
objective's kernel weights fully explain the failure."""
import sys, numpy as np, torch
sys.path.insert(0,'.')
import common; from common import *
import codec
from iqp_exact import fwht, popcount_table
GT=gt_fields(); P=pools(); H=[int(h) for h in HOLDOUT]
pc=popcount_table(18)
rng=np.random.default_rng(7)
gts=GT[[T2I[t] for t in H]].astype(np.float32)
print("%-8s %-9s %-9s %-9s %-9s"%("order L","PDF-JS","Conv-MMD2","corr(paired)","field std"))
for L in [1,2,3,4,5,6,7,9,12,18]:
    keep=(pc<=L)
    gen=[]
    for t in H:
        p=codec.codes_to_pmf(P[T2I[t]].astype(np.int64))
        e=fwht(torch.as_tensor(p)).numpy(); e=e*keep
        q=fwht(torch.as_tensor(e)).numpy()/(1<<18)
        q=np.clip(q,0,None); q=q/q.sum()
        c=rng.choice(1<<18,size=100000,p=q)
        gen.append(fast_aggregate(c))
    gen=np.stack(gen).astype(np.float32)
    js,mm=score(gts,gen)
    F=gen.reshape(200,-1).astype(np.float64); G=gts.reshape(200,-1).astype(np.float64)
    F=(F-F.mean(1,keepdims=True))/np.maximum(F.std(1,keepdims=True),1e-12)
    G=(G-G.mean(1,keepdims=True))/G.std(1,keepdims=True)
    print("%-8d %-9.5f %-9.5f %-9.4f %-9.4f"%(L,js,mm,float((F*G).sum(1).mean()/F.shape[1]),gen.std()))
print("\nreference: trained IQP-63k  PDF-JS 0.235  MMD2 0.834  corr 0.693  std 0.219")
print("reference: trained IQP-31k  PDF-JS 0.266  MMD2 0.819  corr 0.669  std 0.202")
print("reference: RBM-12k          PDF-JS 0.206  MMD2 0.706  corr 0.751  std 0.270")
print("reference: GROUND TRUTH     PDF-JS 0       MMD2 0      corr 1.000  std 0.446")
