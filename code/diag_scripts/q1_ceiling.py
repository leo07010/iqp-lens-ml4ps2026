"""Q1: pipeline ceiling. Perfect model = GT empirical pmf; same sample+aggregate path."""
import sys, time, json
import numpy as np
sys.path.insert(0,'.')
from common import *
import codec

GT = gt_fields(); P = pools()
hold = [int(t) for t in HOLDOUT]
idx  = np.array([T2I[t] for t in hold])
gt_stack = GT[idx]

rows=[]
for n in [10**4, 10**5, 10**6, 10**7]:
    t0=time.time()
    rng = np.random.default_rng(1234)
    gen=[]
    for t in hold:
        pool = P[T2I[t]].astype(np.int64)
        c = codec.sample_codes(pool, n, rng)
        gen.append(fast_aggregate(c))
    gen=np.stack(gen).astype(np.float32)
    js, mmd = score(gt_stack, gen)
    # per-field fidelity
    rmse = float(np.sqrt(((gen-gt_stack)**2).mean()))
    cors = [float(np.corrcoef(gen[i].ravel(), gt_stack[i].ravel())[0,1]) for i in range(len(hold))]
    rows.append(dict(n=n, pdf_js=js, conv_mmd2=mmd, rmse=rmse, corr_mean=float(np.mean(cors)),
                     gen_std=float(gen.std()), gt_std=float(gt_stack.std()), secs=time.time()-t0))
    print("n=%-9d  PDF-JS=%.5f  MMD2=%.5f  RMSE=%.5f  corr=%.5f  gen_std=%.4f gt_std=%.4f  (%.0fs)"
          % (n, js, mmd, rmse, np.mean(cors), gen.std(), gt_stack.std(), time.time()-t0), flush=True)
    if n==10**5:
        np.save("ceiling_1e5_fields.npy", gen)

# sanity: exact GT vs itself
print("sanity GT-vs-GT:", score(gt_stack, gt_stack))
# occupancy statistics of the aggregation grid
pool = P[T2I[hold[0]]].astype(np.int64)
lin = (((pool>>6)&63)*64 + ((pool>>12)&63))
cnt = np.bincount(lin, minlength=4096)
print("native pool per (kx,ky) cell: n_cells_covered=%d/4096 min=%d max=%d mean=%.2f"
      % (int((cnt>0).sum()), int(cnt[cnt>0].min()), int(cnt.max()), cnt.mean()))
# within-cell spread of z (drives the sampling noise)
kz = pool & 63; z=_ZLUT[kz]
ssum=np.bincount(lin,weights=z,minlength=4096); ssq=np.bincount(lin,weights=z*z,minlength=4096)
m=ssum/np.maximum(cnt,1); v=ssq/np.maximum(cnt,1)-m*m
print("within-cell z std (mean over cells) = %.5f ; between-cell z std = %.5f"
      % (float(np.sqrt(np.maximum(v[cnt>0],0)).mean()), float(m[cnt>0].std())))
json.dump(rows, open("q1_ceiling.json","w"), indent=1)
