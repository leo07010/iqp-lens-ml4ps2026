"""Calibrate the Conv-MMD^2 and PDF-JS scales: what do 'totally wrong' generators score?"""
import sys, numpy as np
sys.path.insert(0,'.')
import common; from common import *
GT=gt_fields(); H=[int(h) for h in HOLDOUT]
gts=GT[[T2I[t] for t in H]].astype(np.float32)
rng=np.random.default_rng(0)
refs={
 "identical to GT (perfect)": gts,
 "GT shuffled across tau":    GT[rng.permutation([T2I[t] for t in H])].astype(np.float32),
 "GT + N(0,0.05) noise":      (gts+rng.normal(0,0.05,gts.shape)).astype(np.float32),
 "GT spatially shuffled px":  np.stack([g.ravel()[rng.permutation(4096)].reshape(64,64) for g in gts]).astype(np.float32),
 "constant = global mean":    np.full_like(gts, gts.mean()),
 "iid uniform[a_z,b_z] noise":rng.uniform(0,2.0348,gts.shape).astype(np.float32),
 "iid N(mean,std of GT)":     rng.normal(gts.mean(),gts.std(),gts.shape).astype(np.float32),
}
print("%-30s %-10s %-10s"%("reference generator","PDF-JS","Conv-MMD2"))
for k,v in refs.items():
    js,mm=score(gts,v); print("%-30s %-10.5f %-10.5f"%(k,js,mm))
