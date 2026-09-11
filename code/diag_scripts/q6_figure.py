import sys, os, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0,'.')
import common; from common import *

OUT="<REPO_ROOT>/code/outputs_d2_main"
FIG="<REPO_ROOT>/docs/figures/iqp_turbulence_replication.png"
GT=gt_fields(); H=[int(h) for h in HOLDOUT]; TAUS=[101,502,947]
R=[json.loads(l) for l in open("<REPO_ROOT>/code/results_d2_main.jsonl")]
bydir={os.path.basename(r["run_dir"].rstrip("/")):r for r in R}
def cond(d,tau): return np.load(os.path.join(OUT,d,"generated_t%d.npy"%tau))
def uncond(d,i):
    fs=sorted(f for f in os.listdir(os.path.join(OUT,d)) if f.startswith("generated_"))
    return np.load(os.path.join(OUT,d,fs[i]))

ROWS=[("Ground truth",  None,                  "D2, plane=mid",       None),
      ("IQP-63k",       "iqp_K7_seed0",        "63.0k par., 11 snaps","cond"),
      ("RBM-12k",       "rbm_seed0",           "12.0k par., 11 snaps","cond"),
      ("DCGAN-1.2M",    "dcgan_1.2m_n11_seed0","1.21M par., 11 snaps","uncond"),
      ("CTRL-interp",   "ctrl_interp",         "0 par. (interp.)",    "cond"),
      ("CTRL-marginal", "ctrl_marginal_seed0", "198 par. (Bernoulli)","cond")]
imgs=[]
for label,d,sub,kind in ROWS:
    if d is None: imgs.append([GT[T2I[t]] for t in TAUS])
    elif kind=="cond": imgs.append([cond(d,t) for t in TAUS])
    else: imgs.append([uncond(d,i) for i in range(3)])
vmin, vmax = 0.0, 1.70

fig=plt.figure(figsize=(15.0,14.6))
gs=fig.add_gridspec(6,3,left=0.135,right=0.505,top=0.895,bottom=0.085,hspace=0.16,wspace=0.05)
for r,(label,d,sub,kind) in enumerate(ROWS):
    for c in range(3):
        ax=fig.add_subplot(gs[r,c])
        im=ax.imshow(imgs[r][c],origin="lower",cmap="turbo",vmin=vmin,vmax=vmax,interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_linewidth(0.6)
        if r==0: ax.set_title(r"$\tau$ = %d"%TAUS[c],fontsize=12,pad=6)
        elif kind=="uncond": ax.set_title("sample %d (no $\\tau$)"%(c+1),fontsize=8.5,color="0.4",pad=3)
        if c==0:
            met=""
            if d is not None and d in bydir:
                met="\nJS %.3f  MMD$^2$ %.3f"%(bydir[d]["pdf_js"],bydir[d]["conv_mmd2"])
            ax.text(-0.09,0.5,"%s\n%s%s"%(label,sub,met),transform=ax.transAxes,
                    fontsize=9.5,ha="right",va="center")
cax=fig.add_axes([0.135,0.048,0.37,0.013])
cb=fig.colorbar(im,cax=cax,orientation="horizontal")
cb.set_label(r"velocity magnitude $|u|$  (shared colour scale, %.2f - %.2f)"%(vmin,vmax),fontsize=9.5)
cb.ax.tick_params(labelsize=8.5)

# ---------------- histogram panel
axh=fig.add_axes([0.655,0.585,0.325,0.295])
edges=np.linspace(0.0,1.75,61); ctr=0.5*(edges[1:]+edges[:-1])
sets=[("Ground truth","k","-",2.2,np.stack([GT[T2I[t]] for t in H])),
      ("IQP-63k","tab:red","--",1.5,np.stack([cond("iqp_K7_seed0",t) for t in H])),
      ("RBM-12k","tab:blue","--",1.5,np.stack([cond("rbm_seed0",t) for t in H])),
      ("DCGAN-1.2M@11","tab:green","--",1.5,np.stack([uncond("dcgan_1.2m_n11_seed0",i) for i in range(200)])),
      ("CTRL-marginal","tab:orange",":",1.5,np.stack([cond("ctrl_marginal_seed0",t) for t in H]))]
for k,col,ls,lw,a in sets:
    h,_=np.histogram(a.ravel(),bins=edges,density=True)
    axh.step(ctr,h,where="mid",color=col,lw=lw,ls=ls,label="%s  (std %.3f)"%(k,a.std()))
axh.set_xlabel(r"pixel value $|u|$",fontsize=10); axh.set_ylabel("density",fontsize=10)
axh.set_xlim(0,1.75); axh.tick_params(labelsize=9)
axh.set_title("Pixel-intensity histogram over all 200 held-out $\\tau$\n(exactly what PDF--JS scores)",fontsize=11)
axh.legend(fontsize=8.5,frameon=False,loc="upper right")
axh.grid(alpha=0.25)
fig.text(0.655,0.545,"Tuple-sampling arms (IQP, RBM, CTRL-marginal) lose ~50-75% of the field's\n"
                     "contrast (std 0.22 / 0.27 / 0.12 vs GT 0.446); DCGAN works in image space\n"
                     "and keeps it (0.445). PDF--JS is essentially a contrast test.",
         fontsize=8.8,color="0.25",va="top")

# ---------------- metric-scale panel
axb=fig.add_axes([0.735,0.145,0.245,0.30])
lab=["CTRL-interp (0 par.)","DCGAN-1.2M @11","RBM-12k","IQP-31k","IQP-63k","CTRL-marginal"]
dirs=["ctrl_interp","dcgan_1.2m_n11_seed0","rbm_seed0","iqp_K6_seed0","iqp_K7_seed0","ctrl_marginal_seed0"]
mm=[bydir[d]["conv_mmd2"] for d in dirs]; js=[bydir[d]["pdf_js"] for d in dirs]
y=np.arange(len(lab))
axb.barh(y-0.19,mm,height=0.36,color="tab:purple",label=r"Conv2D-MMD$^2$")
axb.barh(y+0.19,js,height=0.36,color="tab:cyan",label="PDF--JS")
axb.axvline(0.704,color="crimson",ls=":",lw=1.8)
axb.set_yticks(y); axb.set_yticklabels(lab,fontsize=9)
axb.set_ylim(5.7,-1.15)
axb.text(0.704,-1.02,"0.704",color="crimson",fontsize=8.5,ha="center",va="bottom")
axb.set_xlabel("metric value (lower = better)",fontsize=10); axb.set_xlim(0,1.0)
axb.tick_params(axis="x",labelsize=9)
axb.set_title("Metric-scale calibration",fontsize=11,pad=8)
axb.legend(fontsize=8.5,frameon=False,loc="lower right")
axb.grid(alpha=0.25,axis="x")
fig.text(0.655,0.095,
    "Red line: Conv2D-MMD$^2$ = 0.704 is what the ground truth itself scores when its own\n"
    "pixels are spatially shuffled. IQP (0.83), CTRL-marginal (0.82) and RBM (0.71) all sit\n"
    "at or past that saturation point, so their ordering carries no information.\n"
    "Pipeline ceiling (perfect tuple model, $10^5$ samples): PDF--JS 4e-5, MMD$^2$ 2e-4 --\n"
    "sampling/aggregation noise is NOT what is limiting the quantum arm.",
    fontsize=8.8,color="0.25",va="top")

fig.suptitle("arXiv 2603.06440 replication on turbulence D2  (18 qubits, 11 training snapshots, plane=mid, 200 held-out $\\tau$)\n"
             "IQP does NOT beat RBM/DCGAN; a zero-parameter interpolation of the training anchors beats every model",
             fontsize=13.5,y=0.965)
os.makedirs(os.path.dirname(FIG),exist_ok=True)
fig.savefig(FIG,dpi=120)
print("wrote",FIG,os.path.getsize(FIG),"bytes")
