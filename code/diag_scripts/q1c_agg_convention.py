"""The paper never specifies the aggregation rule (spec doc unknown #10). Test the two
obvious ones: MEAN over the samples landing in a cell (what we implemented) vs
LAST-WRITE-WINS (one sample per cell). Everything else identical."""
import sys, os, json, numpy as np, torch
sys.path.insert(0,'.')
import common; from common import *
import codec
from iqp_exact import IQPExact
OUT="<HOME>/gsoc2026-deeplense-qml/experiments/13_iqp_turbulence_replication/outputs_d2_main"
GT=gt_fields(); P=pools(); A=[int(a) for a in ANCHORS]; H=[int(h) for h in HOLDOUT][:60]
gts=GT[[T2I[t] for t in H]].astype(np.float32)

def agg_last(codes):
    c=np.asarray(codes,dtype=np.int64)
    lin=((c>>6)&63)*64+((c>>12)&63); z=common._ZLUT[c&63]
    f=np.full(4096,np.nan); f[lin]=z            # last write wins
    field=f.reshape(64,64)
    for _ in range(100):
        e=~np.isfinite(field)
        if not e.any(): break
        pad=np.pad(field,1,constant_values=np.nan)
        st=np.stack([pad[1+dy:65+dy,1+dx:65+dx] for dy in(-1,0,1) for dx in(-1,0,1) if not(dy==0 and dx==0)])
        with np.errstate(all="ignore"): nb=np.nanmean(st,axis=0)
        field=np.where(np.isfinite(nb)&e,nb,field)
    return np.where(np.isfinite(field),field,np.nanmean(field)).astype(np.float32)

def gen_codes_iqp(arm,K):
    sim=IQPExact(n=18,K=K,device="cpu",dtype=torch.complex64)
    tc=torch.as_tensor(np.load(os.path.join(OUT,arm,"theta_core.npy")))
    lat=np.load(os.path.join(OUT,arm,"theta_lat_anchors.npy")); g=torch.Generator(); g.manual_seed(999)
    for t in H:
        l=np.array([np.interp(float(t),np.array(A,dtype=float),lat[:,i]) for i in range(lat.shape[1])])
        yield sim.sample(torch.cat((torch.as_tensor(l,dtype=torch.float32),tc)),100000,generator=g).numpy()

rng=np.random.default_rng(3)
sources={"PERFECT (GT empirical pmf)": (codec.sample_codes(P[T2I[t]].astype(np.int64),100000,rng) for t in H),
         "IQP-63k": gen_codes_iqp("iqp_K7_seed0",7),
         "IQP-31k": gen_codes_iqp("iqp_K6_seed0",6)}
print("%-28s %-22s %-22s"%("tuple source","MEAN aggregation","LAST-WRITE aggregation"))
print("%-28s %-10s %-10s %-10s %-10s"%("","PDF-JS","MMD2","PDF-JS","MMD2"))
for name,it in sources.items():
    fm=[];fl=[]
    for c in it:
        fm.append(fast_aggregate(c)); fl.append(agg_last(c))
    fm=np.stack(fm).astype(np.float32); fl=np.stack(fl).astype(np.float32)
    j1,m1=score(gts,fm); j2,m2=score(gts,fl)
    print("%-28s %-10.5f %-10.5f %-10.5f %-10.5f   (std mean=%.3f last=%.3f)"%(name,j1,m1,j2,m2,fm.std(),fl.std()))
print("\nPaper Table 2 reports IQP-31k/63k PDF-JS ~ 0.024 on D2.")
