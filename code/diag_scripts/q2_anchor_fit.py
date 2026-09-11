import numpy as _np
def _js(p,q,eps=1e-12):
    p=_np.asarray(p,dtype=_np.float64)+eps; q=_np.asarray(q,dtype=_np.float64)+eps
    p/=p.sum(); q/=q.sum(); m=0.5*(p+q)
    return float(0.5*(p*(_np.log2(p)-_np.log2(m))).sum()+0.5*(q*(_np.log2(q)-_np.log2(m))).sum())
"""Q2: does the trained IQP fit its own TRAINING anchors? Exact-pmf diagnostics."""
import sys, os, json, math, time
import numpy as np, torch
sys.path.insert(0,'.')
import common
from common import *
import codec
from iqp_exact import IQPExact, popcount_table

OUT="<REPO_ROOT>/code/outputs_d2_main"
GT=gt_fields(); P=pools()
pc = popcount_table(18)
pc_t = torch.as_tensor(pc.astype(np.int64))
res={}

for arm, K in [("iqp_K7_seed0",7), ("iqp_K6_seed0",6)]:
    d=os.path.join(OUT,arm)
    theta_core=torch.as_tensor(np.load(d+"/theta_core.npy"))
    lat=np.load(d+"/theta_lat_anchors.npy")
    meta=json.load(open(d+"/meta.json"))
    sim=IQPExact(n=18,K=K,device="cpu",dtype=torch.complex64)
    sigmas=meta["sigmas"]; w=sim.kernel_weights(sigmas)
    print("\n=== %s  params=%d  sigmas=%s ==="%(arm,sim.n_params,["%.3f"%s for s in sigmas]))
    anchors=meta["anchors"]
    gen=[]; per=[]
    gg=torch.Generator(); gg.manual_seed(999)
    for j,t in enumerate(anchors):
        th=torch.cat((torch.as_tensor(lat[j],dtype=torch.float32), theta_core))
        q=sim.probs(th).double(); q=torch.clamp(q,min=0); q=q/q.sum()
        pool=P[T2I[t]].astype(np.int64)
        p=torch.as_tensor(codec.codes_to_pmf(pool))
        # --- sampled field (same 1e5 path as the real run)
        codes=sim.sample(th,100000,generator=gg).numpy()
        f=fast_aggregate(codes); gen.append(f)
        gtf=GT[T2I[t]]
        corr=float(np.corrcoef(f.ravel(),gtf.ravel())[0,1]); rmse=float(np.sqrt(((f-gtf)**2).mean()))
        # --- exact conditional mean E[z|x,y] from the model pmf (no sampling noise)
        Q=q.numpy().reshape(64,64,64)         # [kx,ky,kz]  (code = kx<<12|ky<<6|kz)
        pxy=Q.sum(axis=2)                     # model (kx,ky) marginal
        Ez=(Q*common._ZLUT[None,None,:]).sum(axis=2)/np.maximum(pxy,1e-300)
        Ez_field=Ez.T                          # -> (ky,kx) = (row=y,col=x)
        corr_exact=float(np.corrcoef(Ez_field.ravel(), gtf.ravel())[0,1])
        # --- data-side same quantities
        Pd=p.numpy().reshape(64,64,64); pxy_d=Pd.sum(axis=2)
        # --- Walsh spectra
        ez_m=sim.expvals(th).double().numpy(); ez_d=fwht_np=torch.as_tensor(p).clone()
        ez_d=torch.from_numpy(np.asarray(p)); 
        from iqp_exact import fwht
        ez_d=fwht(torch.as_tensor(p.numpy())).numpy()
        rms={k: (float(np.sqrt(((ez_m[pc==k]-ez_d[pc==k])**2).mean())),
                 float(np.sqrt((ez_d[pc==k]**2).mean()))) for k in (1,2,3,4,6)}
        mmd=float(sim.mmd2(th, torch.as_tensor(ez_d,dtype=torch.float32), w))
        per.append(dict(t=t,corr_sampled=corr,rmse=rmse,corr_exact_cond=corr_exact,
                        mmd2=mmd,
                        pxy_min=float(pxy.min()*4096),pxy_max=float(pxy.max()*4096),
                        pxy_cv=float(pxy.std()/pxy.mean()),
                        pxy_d_cv=float(pxy_d.std()/pxy_d.mean()),
                        rms=rms))
        if j in (0,5,10):
            print(" t=%-5d corr(sampled)=%.4f rmse=%.4f | corr(exact E[z|x,y])=%.4f | MMD2=%.3e"
                  %(t,corr,rmse,corr_exact,mmd))
            print("        model p(x,y)*4096: min=%.3f max=%.3f cv=%.3f (data cv=%.3f)"
                  %(pxy.min()*4096,pxy.max()*4096,pxy.std()/pxy.mean(),pxy_d.std()/pxy_d.mean()))
            print("        <Z_s> RMS err / data RMS by |s|: "+"  ".join("|s|=%d %.4f/%.4f"%(k,rms[k][0],rms[k][1]) for k in (1,2,3,4,6)))
            # z-marginal
            zm=Q.sum(axis=(0,1)); zd=Pd.sum(axis=(0,1))
            print("        kz hist JS(model,data)=%.4f  mean kz model=%.2f data=%.2f"
                  %(_js(zm,zd), (zm*np.arange(64)).sum(), (zd*np.arange(64)).sum()))
            print("        <Z_j> j=0..5 (z bits) model=%s"%np.array2string(ez_m[[1<<i for i in range(6)]],precision=3))
            print("        <Z_j> j=0..5 (z bits) data =%s"%np.array2string(ez_d[[1<<i for i in range(6)]],precision=3))
            print("        <Z_j> j=6..11(y bits) model=%s data=%s"%(np.array2string(ez_m[[1<<i for i in range(6,12)]],precision=3),np.array2string(ez_d[[1<<i for i in range(6,12)]],precision=3)))
            print("        <Z_j> j=12..17(x bits) model=%s data=%s"%(np.array2string(ez_m[[1<<i for i in range(12,18)]],precision=3),np.array2string(ez_d[[1<<i for i in range(12,18)]],precision=3)))
    gen=np.stack(gen).astype(np.float32)
    gtA=np.stack([GT[T2I[t]] for t in anchors]).astype(np.float32)
    js,mmd2=score(gtA,gen)
    print(" ANCHOR-SET metrics (11 anchors, sampled 1e5): PDF-JS=%.5f  Conv-MMD2=%.5f"%(js,mmd2))
    print(" mean corr(sampled)=%.4f  mean corr(exact cond mean)=%.4f"
          %(np.mean([r['corr_sampled'] for r in per]), np.mean([r['corr_exact_cond'] for r in per])))
    res[arm]=dict(anchor_js=js,anchor_mmd2=mmd2,per=per)
    np.save("q2_%s_anchor_gen.npy"%arm, gen)
json.dump(res, open("q2_anchor_fit.json","w"), indent=1, default=float)
