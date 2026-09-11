"""Q2b: where does the MMD^2 loss live? and what is a 'good' loss value?"""
import sys, os, json
import numpy as np, torch
sys.path.insert(0,'.')
import common
from common import *
import codec
from iqp_exact import IQPExact, popcount_table, fwht

OUT="<REPO_ROOT>/code/outputs_d2_main"
P=pools(); GT=gt_fields()
pc=popcount_table(18)
sim=IQPExact(n=18,K=7,device="cpu",dtype=torch.complex64)
meta=json.load(open(OUT+"/iqp_K7_seed0/meta.json"))
w=sim.kernel_weights(meta["sigmas"]).double().numpy()
print("kernel weight c_s by |s| (auto3, 3 sigmas averaged):")
for k in range(0,9):
    print("   |s|=%d  c=%.3e   n_subsets=%d   c*count=%.3e"%(k, w[pc==k][0], int((pc==k).sum()), w[pc==k][0]*(pc==k).sum()))

t=1
pool=P[T2I[t]].astype(np.int64)
p=codec.codes_to_pmf(pool)
ezd=fwht(torch.as_tensor(p)).numpy()
theta_core=torch.as_tensor(np.load(OUT+"/iqp_K7_seed0/theta_core.npy"))
lat=np.load(OUT+"/iqp_K7_seed0/theta_lat_anchors.npy")
th=torch.cat((torch.as_tensor(lat[0],dtype=torch.float32), theta_core))
ezm=sim.expvals(th).double().numpy()

# product-Bernoulli reference: <Z_s> = prod_j in s <Z_j>_data
zj=ezd[[1<<j for j in range(18)]]
ez_pb=np.ones(1<<18)
for j in range(18):
    mask=((np.arange(1<<18)>>j)&1).astype(bool)
    ez_pb[mask]*= zj[j]
# uniform reference: <Z_s>=0 for s!=0
ez_un=np.zeros(1<<18); ez_un[0]=1.0

def L(ez): 
    d=ez-ezd; return float((w*d*d).sum())
def Lby(ez):
    d=ez-ezd; e=w*d*d
    return {k: float(e[pc==k].sum()) for k in range(19)}

print("\nMMD^2 against t=1 data (same kernel weights as training):")
for name,ez in [("trained IQP-63k",ezm),("product-Bernoulli(exact 1-body)",ez_pb),("uniform (all <Z_s>=0)",ez_un)]:
    print("   %-32s  MMD^2 = %.4e"%(name,L(ez)))
lb=Lby(ezm); tot=sum(lb.values())
print("\n   trained-model loss split by |s| (fraction of total):")
print("   "+"  ".join("|s|=%d:%.1f%%"%(k,100*lb[k]/tot) for k in range(0,9) if lb[k]/tot>1e-4))
lbp=Lby(ez_un); totp=sum(lbp.values())
print("   uniform-model loss split by |s|:")
print("   "+"  ".join("|s|=%d:%.1f%%"%(k,100*lbp[k]/totp) for k in range(0,9) if lbp[k]/totp>1e-4))

# unweighted spectral energy: how much of the data's <Z_s>^2 mass sits above |s|=7?
en={k: float((ezd[pc==k]**2).sum()) for k in range(19)}
tote=sum(en[k] for k in range(1,19))
print("\n   data <Z_s>^2 energy by |s| (excluding s=empty), %% of total:")
print("   "+"  ".join("%d:%.1f%%"%(k,100*en[k]/tote) for k in range(1,13)))
print("   fraction of energy at |s|>7 (unreachable by K=7 params): %.1f%%"
      %(100*sum(en[k] for k in range(8,19))/tote))
# weighted version
enw={k: float((w[pc==k]*ezd[pc==k]**2).sum()) for k in range(19)}
totw=sum(enw[k] for k in range(1,19))
print("   KERNEL-WEIGHTED data energy by |s|, %% of total: "+"  ".join("%d:%.2f%%"%(k,100*enw[k]/totw) for k in range(1,10)))
print("   kernel-weighted energy at |s|>=4: %.3f%%"%(100*sum(enw[k] for k in range(4,19))/totw))

# ---- field-space collapse diagnostic
def stats(arr): return "std=%.4f mean=%.4f min=%.4f max=%.4f"%(arr.std(),arr.mean(),arr.min(),arr.max())
hold=[int(x) for x in HOLDOUT]; gts=GT[[T2I[x] for x in hold]]
print("\nField-value spread on the 200 held-out tau (PDF-JS is a histogram of THESE values):")
print("   %-24s %s"%("GROUND TRUTH",stats(gts)))
for arm in ["iqp_K7_seed0","iqp_K6_seed0","rbm_seed0","dcgan_1.2m_n11_seed0","ctrl_interp","ctrl_marginal_seed0"]:
    d=os.path.join(OUT,arm)
    fs=sorted([f for f in os.listdir(d) if f.startswith("generated_")])
    a=np.stack([np.load(os.path.join(d,f)) for f in fs]).astype(np.float32)
    print("   %-24s %s   (n=%d)"%(arm,stats(a),a.shape[0]))
