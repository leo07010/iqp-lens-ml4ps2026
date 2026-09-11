import os, sys, json
import numpy as np, torch
torch.set_num_threads(4)
REPO="<REPO_ROOT>/code"
sys.path.insert(0, REPO)
import codec
from metrics import pdf_js, conv_mmd2
D=os.path.dirname(os.path.abspath(__file__))
M=json.load(open(D+"/meta_prep.json"))
A_Z, B_Z = M["a_z"], M["b_z"]
ANCHORS, HOLDOUT = M["anchors"], M["holdout"]
TS = np.load(D+"/ts.npy")
T2I = {int(t):i for i,t in enumerate(TS)}
def gt_fields(): return np.load(D+"/gt_fields.npy")
def pools():     return np.load(D+"/pools.npy")

_ZLUT = codec.dequantize_z(np.arange(64), A_Z, B_Z)   # bin centres

def fast_aggregate(codes, max_iter=100):
    """bincount version of codec.aggregate (verified equal in q1.py)."""
    c = np.asarray(codes, dtype=np.int64)
    cell = (c >> 6) & 0xFFF          # ky*? no: code=(kx<<12)|(ky<<6)|kz -> (kx,ky)
    kx = (c >> 12) & 63; ky = (c >> 6) & 63; kz = c & 63
    lin = ky*64 + kx
    z = _ZLUT[kz]
    ssum = np.bincount(lin, weights=z, minlength=4096)
    cnt  = np.bincount(lin, minlength=4096).astype(np.float64)
    field = np.where(cnt>0, ssum/np.maximum(cnt,1), np.nan).reshape(64,64)
    gmean = np.nanmean(field) if np.isfinite(field).any() else 0.5*(A_Z+B_Z)
    for _ in range(max_iter):
        empty = ~np.isfinite(field)
        if not empty.any(): break
        padded = np.pad(field,1,mode="constant",constant_values=np.nan)
        stack = np.stack([padded[1+dy:65+dy, 1+dx:65+dx] for dy in (-1,0,1) for dx in (-1,0,1) if not(dy==0 and dx==0)])
        with np.errstate(all="ignore"): neigh=np.nanmean(stack,axis=0)
        fill = np.isfinite(neigh)&empty
        field = np.where(fill, neigh, field)
    field = np.where(np.isfinite(field), field, gmean)
    return field.astype(np.float32)

def score(gt_stack, gen_stack):
    return float(pdf_js(gt_stack, gen_stack)), float(conv_mmd2(gt_stack, gen_stack, seed=0))
