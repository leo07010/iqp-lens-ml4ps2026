"""Q3: sensitivity arms; plus optimizer-convergence and objective-noise-floor checks."""
import sys, os, json
import numpy as np, torch
sys.path.insert(0,'.')
import common
from common import *
import codec
from iqp_exact import IQPExact, fwht

REPO="<HOME>/gsoc2026-deeplense-qml/experiments/13_iqp_turbulence_replication"
OUT=REPO+"/outputs_d2_main"
rows=[json.loads(l) for l in open(REPO+"/results_d2_main.jsonl")]
by_dir={os.path.basename(r["run_dir"].rstrip("/")):r for r in rows}
print("%-34s %-9s %-9s %-8s %s"%("run","PDF-JS","MMD2","params","key config"))
names=["iqp_K6_seed0","iqp_K6_seed1","iqp_K6_seed2","iqp_K7_seed0","iqp_K7_seed1","iqp_K7_seed2"]+\
      sorted([d for d in os.listdir(OUT) if d.startswith("iqp_sens")])
for nm in names:
    r=by_dir.get(nm); m=json.load(open(os.path.join(OUT,nm,"meta.json")))
    c=m["config"]
    key="K=%d lr=%g init=%s lat_init=%s full_grid=%s sigmas=%s -> w=%s"%(
        c["K"],c["lr"],c["init"],c["lat_init"],c["full_grid"],c["sigmas"],
        ["%.2f"%x for x in m["mean_pauli_weights"]])
    print("%-34s %-9.5f %-9.5f %-8d %s"%(nm, r["pdf_js"], r["conv_mmd2"], r["param_count"], key))
    print("%-34s   final stage1 loss=%.3e  stage2 last=%.3e  wall=%.0fs"%("",m["final_loss_stage1"],m["final_loss_stage2"][-1],m["wallclock_s"]))

print("\n--- optimizer convergence (stage-1 loss curve) ---")
for nm in ["iqp_K6_seed0","iqp_K7_seed0","iqp_sens_lr1e-3","iqp_sens_sigma_ref38_16q"]:
    z=np.load(os.path.join(OUT,nm,"loss_curves.npz"))
    s1=z["stage1"]
    print("%-28s step0=%.3e  1k=%.3e  10k=%.3e  20k=%.3e  final=%.3e   last-5k rel drop=%.2f%%"
          %(nm,s1[0],s1[1000],s1[10000],s1[20000],s1[-1],100*(s1[-5000]-s1[-1])/s1[-5000]))

print("\n--- objective noise floor: MMD^2 between two independent 50k-tuple empirical pmfs of the SAME snapshot ---")
sim=IQPExact(n=18,K=7,device="cpu",dtype=torch.complex64)
m=json.load(open(OUT+"/iqp_K7_seed0/meta.json")); w=sim.kernel_weights(m["sigmas"]).double().numpy()
P=pools(); pool=P[T2I[1]].astype(np.int64)
rng=np.random.default_rng(0)
ez_full=fwht(torch.as_tensor(codec.codes_to_pmf(pool))).numpy()
for n in [50000, 200000]:
    ds=[]
    for k in range(3):
        e=fwht(torch.as_tensor(codec.codes_to_pmf(codec.sample_codes(pool,n,rng)))).numpy()
        d=e-ez_full; ds.append(float((w*d*d).sum()))
    print("   n_tuples=%-7d  MMD^2(empirical, full pool) = %.3e  (3 draws: %s)"%(n,np.mean(ds),["%.2e"%x for x in ds]))
print("   trained IQP-63k achieved MMD^2 vs its own 50k target ~ 7.7e-5 (meta final_loss_stage1)")
