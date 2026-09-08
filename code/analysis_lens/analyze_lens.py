"""Quantify DeepLense (model_II) image characteristics for IQP circuit design.

Every measurement here maps to one circuit-design decision:
  (1) intensity levels      -> z-encoding / number of intensity qubits
  (2) Walsh order-spectrum  -> generator set order K
  (3) coefficient variance  -> which 50 subsets to use as the latent block
  (4) pixel PCA             -> intrinsic dimensionality / latent width
  (5) separability          -> is a class-conditional prior worth it
  (6) arc geometry          -> what structure the generators must express
  (7) bit layout            -> how to assign (x,y) bits to qubits
  (8) 18-bit tuple pmf      -> cost of the tuple encoding vs image-as-pmf

CPU only, numpy + matplotlib. Python 3.9.
"""
import json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lens_common import (fwht, popcount_table, order_spectrum, layout_index,
                         image_to_pmf12, image_to_pmf18, quantize_z)

SCRATCH = os.environ.get("LENS_SCRATCH", "/tmp")
NPZ = "<HOME>/mae-lensing/model_II.npz"
CLASSES = ["axion", "cdm", "no_sub"]
R = {}          # results dict -> results.json
CACHE = {}      # intermediates -> intermediates.npz (for the figure script)
T0 = time.time()


def log(msg):
    print("[%6.1fs] %s" % (time.time() - T0, msg), flush=True)


def ent_bits(counts):
    p = np.asarray(counts, dtype=np.float64)
    p = p / p.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


# =====================================================================
# load
# =====================================================================
val_sub = np.load(os.path.join(SCRATCH, "val_sub.npy"))      # (3,200,64,64)
train_sub = np.load(os.path.join(SCRATCH, "train_sub.npy"))  # (3,2000,64,64)
log("subsets loaded %s %s" % (val_sub.shape, train_sub.shape))

PC12 = popcount_table(12)
CODE = {k: layout_index(k) for k in ("standard", "morton", "gray")}

# =====================================================================
# 1. intensity levels
# =====================================================================
log("=== 1. intensity levels ===")
d = np.load(NPZ)
val_all = d["val_x"]                       # (8910,64,64)
px = val_all.reshape(-1).astype(np.float64)
del d
R["n1_intensity"] = {}
hist, edges = np.histogram(px, bins=256, range=(0.0, 1.0))
CACHE["n1_hist"] = hist
CACHE["n1_edges"] = edges
R["n1_intensity"]["n_pixels"] = int(px.size)
R["n1_intensity"]["pixel_stats"] = {
    "mean": float(px.mean()), "std": float(px.std()),
    "median": float(np.median(px)),
    "p90": float(np.percentile(px, 90)), "p99": float(np.percentile(px, 99)),
    "p99.9": float(np.percentile(px, 99.9)),
    "frac_exact_zero": float((px == 0).mean()),
    "frac_gt_1_64": float((px > 1.0 / 64).mean()),
    "frac_gt_0.1": float((px > 0.1).mean()),
}

# four 6-bit quantizers
q_edges = np.quantile(px, np.linspace(0, 1, 65)[1:-1])
CACHE["n1_qedges"] = q_edges
schemes = {}
schemes["raw"] = np.clip(np.floor(64 * px), 0, 63).astype(np.int64)
schemes["sqrt"] = np.clip(np.floor(64 * np.sqrt(px)), 0, 63).astype(np.int64)
schemes["log1p100"] = np.clip(
    np.floor(64 * np.log1p(100 * px) / np.log1p(100.0)), 0, 63).astype(np.int64)
schemes["quantile"] = np.searchsorted(q_edges, px, side="right").astype(np.int64)
occ = {}
for name, lv in schemes.items():
    c = np.bincount(lv, minlength=64).astype(np.float64)
    frac = c / c.sum()
    occ[name] = frac
    R["n1_intensity"].setdefault("quantizers", {})[name] = {
        "entropy_bits": ent_bits(c),
        "n_levels_used": int((c > 0).sum()),
        "frac_in_level0": float(frac[0]),
        "frac_in_top8_levels": float(frac[56:].sum()),
        "eff_levels_2^H": float(2 ** ent_bits(c)),
    }
    del lv
CACHE["n1_occ"] = np.stack([occ[k] for k in ["raw", "sqrt", "log1p100", "quantile"]])
del px, schemes

# per-image active fractions, per class
act = {}
for ci, cn in enumerate(CLASSES):
    im = val_sub[ci].reshape(200, -1)
    a1 = (im > 1.0 / 64).mean(axis=1)
    a2 = (im > 0.1).mean(axis=1)
    act[cn] = {
        "frac_gt_1_64_mean": float(a1.mean()), "frac_gt_1_64_std": float(a1.std()),
        "frac_gt_0.1_mean": float(a2.mean()), "frac_gt_0.1_std": float(a2.std()),
        "n_pix_gt_1_64_mean": float(a1.mean() * 4096),
        "n_pix_gt_0.1_mean": float(a2.mean() * 4096),
    }
R["n1_intensity"]["active_per_class"] = act
log("  entropies: " + ", ".join(
    "%s=%.2fb" % (k, v["entropy_bits"]) for k, v in R["n1_intensity"]["quantizers"].items()))

# =====================================================================
# 2. 12-bit pmf Walsh order-spectrum
# =====================================================================
log("=== 2. 12-bit Walsh order-spectrum ===")
pmf12 = {}   # class -> (200,4096) standard layout
W12 = {}
for ci, cn in enumerate(CLASSES):
    pmf12[cn] = image_to_pmf12(val_sub[ci], CODE["standard"])
    W12[cn] = fwht(pmf12[cn])
pooled_W = np.concatenate([W12[c] for c in CLASSES], axis=0)   # (600,4096)
pooled_pmf = np.concatenate([pmf12[c] for c in CLASSES], axis=0)

R["n2_walsh12"] = {}
os_per = {}
for cn in CLASSES:
    sp = order_spectrum(W12[cn], PC12, 12)      # (200,13)
    os_per[cn] = sp
    R["n2_walsh12"].setdefault("per_image", {})[cn] = {
        "mean": sp.mean(0).tolist(), "std": sp.std(0).tolist()}
    # class-mean image
    mimg = val_sub[ci if False else CLASSES.index(cn)].mean(0)
    mp = image_to_pmf12(mimg[None], CODE["standard"])
    R["n2_walsh12"].setdefault("class_mean_image", {})[cn] = \
        order_spectrum(fwht(mp), PC12, 12)[0].tolist()
CACHE["n2_os_per"] = np.stack([os_per[c] for c in CLASSES])   # (3,200,13)

# support mismatch + participation ratio
for cn in CLASSES:
    sp = os_per[cn]
    R["n2_walsh12"].setdefault("support_mismatch", {})[cn] = {
        "energy_outside_k<=4_mean": float(sp[:, 5:].sum(1).mean()),
        "energy_outside_k<=6_mean": float(sp[:, 7:].sum(1).mean()),
        "energy_outside_k<=4_std": float(sp[:, 5:].sum(1).std()),
        "energy_outside_k<=6_std": float(sp[:, 7:].sum(1).std()),
        "mean_order_|s|": float((sp * np.arange(13)).sum(1).mean()),
    }
    pr = 1.0 / (pmf12[cn] ** 2).sum(1)
    R["n2_walsh12"].setdefault("participation_ratio", {})[cn] = {
        "eff_n_pixels_mean": float(pr.mean()), "eff_n_pixels_std": float(pr.std()),
        "frac_of_4096": float(pr.mean() / 4096)}

# data-adaptive support curve (energy ranked)
Ms = [50, 100, 200, 500, 1000, 2510, 4096]
R["n2_walsh12"]["data_adaptive_support"] = {"M_list": Ms}
for name, Wm in [("pooled", pooled_W)] + [(c, W12[c]) for c in CLASSES]:
    meanE = (Wm ** 2).mean(0)
    order = np.argsort(meanE)[::-1]
    cum = np.cumsum(meanE[order]) / meanE.sum()
    R["n2_walsh12"]["data_adaptive_support"][name] = {
        "cum_energy_at_M": [float(cum[m - 1]) for m in Ms]}
    if name == "pooled":
        CACHE["n2_cum_pooled"] = cum
        CACHE["n2_rank_pooled"] = order
# Hamming |s|<=6 set for comparison (same size as M=2510)
mask_le6 = (PC12 <= 6)
meanE_p = (pooled_W ** 2).mean(0)
R["n2_walsh12"]["hamming_le6_energy_pooled"] = float(
    meanE_p[mask_le6].sum() / meanE_p.sum())
R["n2_walsh12"]["hamming_le4_energy_pooled"] = float(
    meanE_p[PC12 <= 4].sum() / meanE_p.sum())
R["n2_walsh12"]["n_subsets_le6"] = int(mask_le6.sum())
log("  pooled cum-energy @M: " + str(
    [round(x, 4) for x in R["n2_walsh12"]["data_adaptive_support"]["pooled"]["cum_energy_at_M"]]))

# =====================================================================
# 3. cross-image variance of Walsh coefficients -> latent candidates
# =====================================================================
log("=== 3. cross-image Walsh variance (latent candidates) ===")
R["n3_latent"] = {}
# paper's Hamming-first 50: empty + 12 singletons + 37 two-body
hf = [0] + [1 << b for b in range(12)]
pairs = [(i, j) for i in range(12) for j in range(i + 1, 12)]
hf += [(1 << i) | (1 << j) for (i, j) in pairs[:37]]
hf = np.array(hf, dtype=np.int64)
assert len(hf) == 50 and len(np.unique(hf)) == 50


def bits_of(m):
    return [b for b in range(12) if (m >> b) & 1]


def describe(m):
    """standard layout: bits 0-5 = ky (row/y), bits 6-11 = kx (col/x)."""
    bs = bits_of(m)
    if not bs:
        return "empty"
    return "+".join(("y%d" % b) if b < 6 else ("x%d" % (b - 6)) for b in bs)


for name, Wm in [("pooled", pooled_W)] + [(c, W12[c]) for c in CLASSES]:
    var = Wm.var(axis=0)
    order = np.argsort(var)[::-1]
    top50 = order[:50].astype(np.int64)
    tot = var.sum()
    R["n3_latent"][name] = {
        "top50_var_coverage": float(var[top50].sum() / tot),
        "hamming_first50_var_coverage": float(var[hf].sum() / tot),
        "top50_mean_|s|": float(PC12[top50].mean()),
        "top50_order_histogram": np.bincount(PC12[top50], minlength=13).tolist(),
        "top20": [{"mask": int(m), "bits": bits_of(int(m)), "order": int(PC12[m]),
                   "desc": describe(int(m)), "var_frac": float(var[m] / tot)}
                  for m in order[:20]],
        "var_at_empty": float(var[0]),
    }
    np.save(os.path.join(HERE, "latent_candidates_top50_%s.npy" % name), top50)
    if name == "pooled":
        CACHE["n3_var_pooled"] = var
        CACHE["n3_top50_pooled"] = top50

# PCA of the 4096-d Walsh coefficient vectors
def pca_evals(X):
    """eigenvalues of the covariance of X (n,d), descending, via the cheap side."""
    Xc = X - X.mean(0, keepdims=True)
    n, dd = Xc.shape
    if n <= dd:
        G = Xc @ Xc.T / (n - 1)
        w = np.linalg.eigvalsh(G)[::-1]
    else:
        C = Xc.T @ Xc / (n - 1)
        w = np.linalg.eigvalsh(C)[::-1]
    return np.clip(w, 0, None)


def n_for(evals, fracs=(0.90, 0.95, 0.99)):
    cs = np.cumsum(evals) / evals.sum()
    return {("%.2f" % f): int(np.searchsorted(cs, f) + 1) for f in fracs}


for name, Wm in [("pooled", pooled_W)] + [(c, W12[c]) for c in CLASSES]:
    ev = pca_evals(Wm)
    R["n3_latent"][name]["coeff_pca_ncomp"] = n_for(ev)
    R["n3_latent"][name]["coeff_pca_top1_frac"] = float(ev[0] / ev.sum())
log("  pooled top50 var coverage %.4f vs Hamming-first-50 %.4f" % (
    R["n3_latent"]["pooled"]["top50_var_coverage"],
    R["n3_latent"]["pooled"]["hamming_first50_var_coverage"]))

# =====================================================================
# 4. pixel-space PCA
# =====================================================================
log("=== 4. pixel-space PCA ===")
R["n4_pixel_pca"] = {}


def pca_full(X, k=6):
    Xc = X - X.mean(0, keepdims=True)
    n, dd = Xc.shape
    if n <= dd:
        G = Xc @ Xc.T / (n - 1)
        w, U = np.linalg.eigh(G)
        w = w[::-1]; U = U[:, ::-1]
        w = np.clip(w, 1e-30, None)
        V = (Xc.T @ U[:, :k]) / np.sqrt(w[:k] * (n - 1))
    else:
        C = Xc.T @ Xc / (n - 1)
        w, Vf = np.linalg.eigh(C)
        w = w[::-1]; Vf = Vf[:, ::-1]
        V = Vf[:, :k]
    return np.clip(w, 0, None), V, X.mean(0)


pcs_store = {}
for name, X in [("pooled", train_sub.reshape(-1, 4096).astype(np.float64))] + \
               [(c, train_sub[i].reshape(-1, 4096).astype(np.float64))
                for i, c in enumerate(CLASSES)]:
    ev, V, mu = pca_full(X, 6)
    R["n4_pixel_pca"][name] = {
        "ncomp": n_for(ev),
        "top6_frac": [float(x) for x in (ev[:6] / ev.sum())],
        "cum_top6": float(ev[:6].sum() / ev.sum()),
        "n_images": int(X.shape[0]),
    }
    pcs_store[name] = V
    if name == "pooled":
        CACHE["n4_ev_pooled"] = ev[:200]
        CACHE["n4_pc_pooled"] = V.T.reshape(6, 64, 64)
        CACHE["n4_mu_pooled"] = mu.reshape(64, 64)
        POOL_V, POOL_MU, POOL_EV = V, mu, ev
    log("  %s: 90%%=%d 95%%=%d 99%%=%d comps" % (
        name, R["n4_pixel_pca"][name]["ncomp"]["0.90"],
        R["n4_pixel_pca"][name]["ncomp"]["0.95"],
        R["n4_pixel_pca"][name]["ncomp"]["0.99"]))

# describe PCs quantitatively (radial vs azimuthal character)
yy, xx = np.mgrid[0:64, 0:64]
rr = np.sqrt((xx - 31.5) ** 2 + (yy - 31.5) ** 2)
th = np.arctan2(yy - 31.5, xx - 31.5)
pc_desc = []
for i in range(6):
    p = CACHE["n4_pc_pooled"][i]
    w = np.abs(p)
    pc_desc.append({
        "pc": i + 1,
        "explained_frac": float(POOL_EV[i] / POOL_EV.sum()),
        "weighted_mean_radius": float((w * rr).sum() / w.sum()),
        "cos2theta_corr": float(np.corrcoef(p.ravel(), np.cos(2 * th).ravel())[0, 1]),
        "sin2theta_corr": float(np.corrcoef(p.ravel(), np.sin(2 * th).ravel())[0, 1]),
        "cos1theta_corr": float(np.corrcoef(p.ravel(), np.cos(th).ravel())[0, 1]),
        "radial_monotone_corr": float(np.corrcoef(p.ravel(), rr.ravel())[0, 1]),
        "mean_sign": float(np.sign(p).mean()),
    })
R["n4_pixel_pca"]["pc_descriptors_pooled"] = pc_desc

# =====================================================================
# 5. class separability
# =====================================================================
log("=== 5. class separability ===")
R["n5_separability"] = {}
tr_pix = train_sub[:, :1000].reshape(3, 1000, 4096).astype(np.float64)
va_pix = val_sub.reshape(3, 200, 4096).astype(np.float64)

# (a) nearest centroid, pixel space
cent_p = tr_pix.mean(axis=1)                       # (3,4096)
Xv = va_pix.reshape(-1, 4096)
yv = np.repeat(np.arange(3), 200)
dist = ((Xv[:, None, :] - cent_p[None]) ** 2).sum(-1)
acc_pix = float((dist.argmin(1) == yv).mean())

# (a) nearest centroid, 12-bit Walsh space
tr_W = np.stack([fwht(image_to_pmf12(train_sub[c, :1000], CODE["standard"]))
                 for c in range(3)])               # (3,1000,4096)
cent_w = tr_W.mean(axis=1)
XvW = np.concatenate([W12[c] for c in CLASSES], axis=0)
distw = ((XvW[:, None, :] - cent_w[None]) ** 2).sum(-1)
acc_w = float((distw.argmin(1) == yv).mean())
R["n5_separability"]["nearest_centroid"] = {
    "pixel_space_acc": acc_pix, "walsh12_space_acc": acc_w, "chance": 1.0 / 3}

# (b) multinomial logistic regression on the top-50 pooled pixel PCs
Vp = POOL_V if POOL_V.shape[1] >= 6 else None
ev_all, V50, mu50 = pca_full(train_sub.reshape(-1, 4096).astype(np.float64), 50)
Ztr = (tr_pix.reshape(-1, 4096) - mu50) @ V50
Zva = (Xv - mu50) @ V50
sd = Ztr.std(0); sd[sd == 0] = 1
Ztr = Ztr / sd; Zva = Zva / sd
ytr = np.repeat(np.arange(3), 1000)
Ytr = np.eye(3)[ytr]
Wl = np.zeros((50, 3)); bl = np.zeros(3)
lr, lam = 0.5, 1e-4
for it in range(4000):
    lo = Ztr @ Wl + bl
    lo -= lo.max(1, keepdims=True)
    P = np.exp(lo); P /= P.sum(1, keepdims=True)
    G = (P - Ytr) / Ztr.shape[0]
    Wl -= lr * (Ztr.T @ G + lam * Wl)
    bl -= lr * G.sum(0)
tr_acc = float(((Ztr @ Wl + bl).argmax(1) == ytr).mean())
va_acc = float(((Zva @ Wl + bl).argmax(1) == yv).mean())
R["n5_separability"]["logreg_top50_pixel_pcs"] = {
    "train_acc": tr_acc, "val_acc": va_acc, "chance": 1.0 / 3,
    "n_train": int(Ztr.shape[0]), "n_val": int(Zva.shape[0])}
log("  NC pixel %.3f | NC walsh %.3f | logreg50 val %.3f" % (acc_pix, acc_w, va_acc))

# =====================================================================
# 6. arc geometry
# =====================================================================
log("=== 6. arc geometry ===")
R["n6_geometry"] = {}
rbin = np.clip(rr.astype(int), 0, 44)
nr = 45
radial = {}
azim = {}
nth = 36
tbin = ((th + np.pi) / (2 * np.pi) * nth).astype(int) % nth
annulus = (rr >= 5) & (rr <= 25)
for ci, cn in enumerate(CLASSES):
    im = val_sub[ci].astype(np.float64)            # (200,64,64)
    prof = np.zeros(nr); cnts = np.bincount(rbin.ravel(), minlength=nr)
    acc = np.bincount(rbin.ravel(), weights=im.mean(0).ravel(), minlength=nr)
    prof = acc / np.maximum(cnts, 1)
    radial[cn] = prof
    m = im.mean(0)
    a = np.bincount(tbin[annulus], weights=m[annulus], minlength=nth)
    c = np.bincount(tbin[annulus], minlength=nth)
    azim[cn] = a / np.maximum(c, 1)
    # per-image light centroid offset
    w = im.reshape(200, -1)
    tot = w.sum(1)
    cx = (w * xx.ravel()).sum(1) / tot
    cy = (w * yy.ravel()).sum(1) / tot
    off = np.sqrt((cx - 31.5) ** 2 + (cy - 31.5) ** 2)
    # intensity-weighted rms radius (arc size proxy)
    r2 = (w * (rr.ravel() ** 2)).sum(1) / tot
    npx = (w > 1.0 / 64).sum(1)
    R["n6_geometry"][cn] = {
        "centroid_offset_px_mean": float(off.mean()), "centroid_offset_px_std": float(off.std()),
        "centroid_offset_px_p95": float(np.percentile(off, 95)),
        "rms_radius_px_mean": float(np.sqrt(r2).mean()), "rms_radius_px_std": float(np.sqrt(r2).std()),
        "n_active_gt_1_64_mean": float(npx.mean()), "n_active_gt_1_64_std": float(npx.std()),
        "peak_radius_px": int(np.argmax(prof[:40])),
        "azimuthal_contrast": float((azim[cn].max() - azim[cn].min()) / azim[cn].mean()),
    }
CACHE["n6_radial"] = np.stack([radial[c] for c in CLASSES])
CACHE["n6_azim"] = np.stack([azim[c] for c in CLASSES])
CACHE["n6_meanimg"] = np.stack([val_sub[i].mean(0) for i in range(3)])

# =====================================================================
# 7. bit-layout sensitivity
# =====================================================================
log("=== 7. bit-layout sensitivity ===")
R["n7_layout"] = {}
allimg = val_sub.reshape(-1, 64, 64)
lay_os = {}
for k in ("standard", "morton", "gray"):
    Wl_ = fwht(image_to_pmf12(allimg, CODE[k]))
    sp = order_spectrum(Wl_, PC12, 12)
    lay_os[k] = sp
    meanE = (Wl_ ** 2).mean(0)
    order = np.argsort(meanE)[::-1]
    cum = np.cumsum(meanE[order]) / meanE.sum()
    R["n7_layout"][k] = {
        "order_spectrum_mean": sp.mean(0).tolist(),
        "mean_|s|": float((sp * np.arange(13)).sum(1).mean()),
        "energy_k<=4": float(sp[:, :5].sum(1).mean()),
        "energy_k<=6": float(sp[:, :7].sum(1).mean()),
        "energy_k>=8": float(sp[:, 8:].sum(1).mean()),
        "cum_energy_top50": float(cum[49]),
        "cum_energy_top2510": float(cum[2509]),
    }
CACHE["n7_os"] = np.stack([lay_os[k].mean(0) for k in ("standard", "morton", "gray")])
log("  mean|s| " + ", ".join("%s=%.2f" % (k, R["n7_layout"][k]["mean_|s|"])
                             for k in ("standard", "morton", "gray")))

# =====================================================================
# 8. 18-bit tuple pmf order-spectrum
# =====================================================================
log("=== 8. 18-bit tuple pmf ===")
PC18 = popcount_table(18)
R["n8_tuple18"] = {}
os18 = {}
for mode in ("raw", "sqrt"):
    for ci, cn in enumerate(CLASSES):
        sp_acc = []
        for b0 in range(0, 50, 25):
            p18 = image_to_pmf18(val_sub[ci, b0:b0 + 25], mode)
            W18 = fwht(p18)
            sp_acc.append(order_spectrum(W18, PC18, 18))
            del p18, W18
        sp = np.concatenate(sp_acc, 0)
        os18["%s_%s" % (mode, cn)] = sp.mean(0)
        R["n8_tuple18"]["%s_%s" % (mode, cn)] = {
            "order_spectrum_mean": sp.mean(0).tolist(),
            "mean_|s|": float((sp * np.arange(19)).sum(1).mean()),
            "energy_k<=4": float(sp[:, :5].sum(1).mean()),
            "energy_k<=6": float(sp[:, :7].sum(1).mean()),
            "energy_k>=10": float(sp[:, 10:].sum(1).mean()),
            "energy_at_empty": float(sp[:, 0].mean()),
            "n_images": int(sp.shape[0]),
        }
        log("  %s %s mean|s|=%.2f  k<=6:%.4f" % (
            mode, cn, R["n8_tuple18"]["%s_%s" % (mode, cn)]["mean_|s|"],
            R["n8_tuple18"]["%s_%s" % (mode, cn)]["energy_k<=6"]))
CACHE["n8_os18"] = np.stack([os18[k] for k in sorted(os18)])
CACHE["n8_os18_keys"] = np.array(sorted(os18))
# number of |s|<=K subsets in 18 bits, for the support-size comparison
from math import comb
R["n8_tuple18"]["n_subsets_le_K_18bit"] = {str(K): int(sum(comb(18, k) for k in range(K + 1)))
                                           for K in (2, 4, 6, 8)}
R["n8_tuple18"]["n_subsets_le_K_12bit"] = {str(K): int(sum(comb(12, k) for k in range(K + 1)))
                                           for K in (2, 4, 6, 8)}

# =====================================================================
np.savez_compressed(os.path.join(HERE, "intermediates.npz"), **CACHE)
with open(os.path.join(HERE, "results.json"), "w") as f:
    json.dump(R, f, indent=2)
log("DONE -> results.json, intermediates.npz")
