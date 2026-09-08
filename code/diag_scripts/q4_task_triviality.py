"""Q4: is 'unseen-time synthesis' on D2 a non-trivial task at all?"""
import sys, os, json
import numpy as np
sys.path.insert(0,'.')
import common
from common import *

GT=gt_fields(); A=[int(a) for a in ANCHORS]; H=[int(h) for h in HOLDOUT]
def f(t): return GT[T2I[t]].astype(np.float64)
def cc(a,b): return float(np.corrcoef(a.ravel(),b.ravel())[0,1])
def rms(a,b): return float(np.sqrt(((a-b)**2).mean()))

print("--- adjacent-anchor GT similarity (Delta t = 100, except 1->100 = 99) ---")
print("%-14s %-8s %-8s %-8s"%("pair","corr","RMSdiff","RMS/std"))
for i in range(len(A)-1):
    a,b=A[i],A[i+1]
    print("t=%-4d->%-6d %-8.5f %-8.5f %-8.4f"%(a,b,cc(f(a),f(b)),rms(f(a),f(b)),rms(f(a),f(b))/GT.std()))

print("\n--- GT decorrelation vs time lag (all pairs among anchors+holdout) ---")
ts=np.array(sorted(T2I.keys()))
flat=GT.reshape(len(ts),-1).astype(np.float64)
flat=(flat-flat.mean(1,keepdims=True))/flat.std(1,keepdims=True)
C=(flat@flat.T)/flat.shape[1]
lag=np.abs(ts[:,None]-ts[None,:])
for lo,hi in [(1,1),(2,5),(6,20),(21,50),(51,100),(101,200),(201,500),(501,1000)]:
    m=(lag>=lo)&(lag<=hi)
    if m.sum(): print("   lag in [%4d,%4d]: mean corr = %.4f  (n_pairs=%d)"%(lo,hi,C[m].mean(),m.sum()//2))

print("\n--- unseen tau vs its CTRL-interp prediction and vs nearest anchor ---")
ci=[];cn=[];ri=[];rn=[]
for t in H:
    j=np.searchsorted(A,t)
    lo,hi=A[max(j-1,0)],A[min(j,len(A)-1)]
    if hi==lo: pred=f(lo)
    else:
        wgt=(t-lo)/(hi-lo); pred=(1-wgt)*f(lo)+wgt*f(hi)
    near=f(lo) if abs(t-lo)<=abs(t-hi) else f(hi)
    ci.append(cc(pred,f(t))); ri.append(rms(pred,f(t)))
    cn.append(cc(near,f(t))); rn.append(rms(near,f(t)))
ci,cn,ri,rn=map(np.array,(ci,cn,ri,rn))
print("   linear interp of bracketing anchors : corr mean=%.5f min=%.5f  RMS=%.5f (%.1f%% of field std)"%(ci.mean(),ci.min(),ri.mean(),100*ri.mean()/GT.std()))
print("   nearest anchor (copy)               : corr mean=%.5f min=%.5f  RMS=%.5f (%.1f%% of field std)"%(cn.mean(),cn.min(),rn.mean(),100*rn.mean()/GT.std()))
print("   NOTE: a constant field (global mean) would give corr = 0, RMS = %.5f"%GT.std())

print("\n--- what would a 'copy one anchor for every tau' arm score on the paper's own metrics? ---")
gts=GT[[T2I[t] for t in H]].astype(np.float32)
for name,gen in [("copy anchor t=1 everywhere", np.repeat(f(1)[None].astype(np.float32),len(H),0)),
                 ("nearest-anchor copy", np.stack([ (f(A[max(np.searchsorted(A,t)-1,0)]) if abs(t-A[max(np.searchsorted(A,t)-1,0)])<=abs(t-A[min(np.searchsorted(A,t),len(A)-1)]) else f(A[min(np.searchsorted(A,t),len(A)-1)])).astype(np.float32) for t in H])),
                 ("GT shuffled across tau", GT[np.random.default_rng(0).permutation([T2I[t] for t in H])].astype(np.float32))]:
    js,mm=score(gts,gen); print("   %-28s PDF-JS=%.5f  Conv-MMD2=%.5f"%(name,js,mm))
