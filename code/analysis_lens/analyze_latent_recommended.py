"""Characterise the best axis-aligned 50-coefficient latent found:
low-order (|s|<=2) subsets ranked by cross-image variance.
Reports its composition, its physics recovery R^2, and saves the mask array.
Also sweeps the order cut K so the choice of |s|<=2 is justified, not assumed.
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lens_common import fwht, layout_index, image_to_pmf12, popcount_table

SCRATCH = os.environ.get("LENS_SCRATCH", "/tmp")
tr = np.load(os.path.join(SCRATCH, "train_sub.npy"))
va = np.load(os.path.join(SCRATCH, "val_sub.npy"))
code = layout_index("standard")
ytr = np.repeat(np.arange(3), 1000); yva = np.repeat(np.arange(3), 200)
PC12 = popcount_table(12)
Wtr = np.concatenate([fwht(image_to_pmf12(tr[c, :1000], code)) for c in range(3)])
Wva = np.concatenate([fwht(image_to_pmf12(va[c], code)) for c in range(3)])
Xtr = tr[:, :1000].reshape(-1, 4096).astype(np.float64)
Xva = va.reshape(-1, 4096).astype(np.float64)
var = Wtr.var(0); order = np.argsort(var)[::-1]


def std_fit(A, B):
    m, s = A.mean(0), A.std(0); s[s < 1e-12] = 1.0
    return (A - m) / s, (B - m) / s


def lda(A, B, reg=1e-4):
    A, B = std_fit(A, B); d = A.shape[1]
    Sw = np.zeros((d, d)); mus = []
    for c in range(3):
        Z = A[ytr == c]; mu = Z.mean(0); mus.append(mu); Sw += (Z - mu).T @ (Z - mu)
    Sw /= A.shape[0]; Sw += reg * np.trace(Sw) / d * np.eye(d)
    Si = np.linalg.inv(Sw); mus = np.array(mus)
    sc = B @ Si @ mus.T - 0.5 * np.einsum("ij,jk,ik->i", mus, Si, mus)[None, :]
    return float((sc.argmax(1) == yva).mean())


def geom(X):
    yy, xx = np.mgrid[0:64, 0:64]
    xr, yr = xx.ravel(), yy.ravel()
    rr = np.sqrt((xr - 31.5) ** 2 + (yr - 31.5) ** 2)
    th = np.arctan2(yr - 31.5, xr - 31.5)
    t = X.sum(1); w2 = (X * rr ** 2).sum(1)
    return np.stack([(X * xr).sum(1) / t - 31.5, (X * yr).sum(1) / t - 31.5, t,
                     (X * (rr ** 2 * np.cos(2 * th))).sum(1) / w2,
                     (X * (rr ** 2 * np.sin(2 * th))).sum(1) / w2], 1)


names = ["centroid_x", "centroid_y", "total_flux", "ellip_cos2t", "ellip_sin2t"]
Gtr, Gva = geom(Xtr), geom(Xva)
mu_g, sd_g = Gtr.mean(0), Gtr.std(0)
Gtr = (Gtr - mu_g) / sd_g; Gva = (Gva - mu_g) / sd_g


def phys_r2(idx):
    A, B = std_fit(Wtr[:, idx], Wva[:, idx])
    A = np.hstack([A, np.ones((len(A), 1))]); B = np.hstack([B, np.ones((len(B), 1))])
    beta = np.linalg.solve(A.T @ A + 1e-3 * np.eye(A.shape[1]), A.T @ Gtr)
    pred = B @ beta
    r2 = {names[i]: float(1 - ((Gva[:, i] - pred[:, i]) ** 2).sum() /
                          ((Gva[:, i] - Gva[:, i].mean()) ** 2).sum()) for i in range(5)}
    r2["MEAN"] = float(np.mean(list(r2.values())))
    return r2


out = {}
# sweep the order cut K
print("%-28s %6s %8s %8s" % ("latent rule", "n", "classACC", "physR2"))
sweep = {}
for K in (1, 2, 3, 4, 6, 12):
    idx = np.array([m for m in order if PC12[m] <= K][:50])
    a = lda(Wtr[:, idx], Wva[:, idx]); r = phys_r2(idx)
    sweep["|s|<=%d, top50 by variance" % K] = {
        "n": int(len(idx)), "class_acc_lda": a, "phys_R2_mean": r["MEAN"],
        "order_hist": np.bincount(PC12[idx], minlength=13).tolist()}
    print("%-28s %6d %8.3f %8.3f" % ("|s|<=%d top50 by var" % K, len(idx), a, r["MEAN"]))
hf = [0] + [1 << b for b in range(12)]
pairs = [(i, j) for i in range(12) for j in range(i + 1, 12)]
hf += [(1 << i) | (1 << j) for (i, j) in pairs[:37]]
hf = np.array(hf)
a = lda(Wtr[:, hf], Wva[:, hf]); r = phys_r2(hf)
sweep["hamming_first50_paper"] = {"n": 50, "class_acc_lda": a, "phys_R2_mean": r["MEAN"],
                                  "order_hist": np.bincount(PC12[hf], minlength=13).tolist()}
print("%-28s %6d %8.3f %8.3f" % ("hamming-first-50 (paper)", 50, a, r["MEAN"]))
out["order_cut_sweep"] = sweep

# the recommended set
best = np.array([m for m in order if PC12[m] <= 2][:50], dtype=np.int64)
np.save(os.path.join(HERE, "latent_recommended_lowvar50.npy"), best)


def bits_of(m):
    return [b for b in range(12) if (m >> b) & 1]


def desc(m):
    b = bits_of(m)
    return "empty" if not b else "+".join(("y%d" % x) if x < 6 else ("x%d" % (x - 6)) for x in b)


out["recommended_set"] = {
    "rule": "subsets with |s|<=2, ranked by cross-image variance, top 50",
    "n": 50,
    "order_histogram": np.bincount(PC12[best], minlength=13).tolist(),
    "class_acc_lda": lda(Wtr[:, best], Wva[:, best]),
    "phys_R2": phys_r2(best),
    "n_subsets_with_|s|<=2_available": int((PC12 <= 2).sum()),
    "members": [{"mask": int(m), "order": int(PC12[m]), "desc": desc(int(m))} for m in best],
    "file": "latent_recommended_lowvar50.npy",
}
print("\nrecommended set: order histogram (|s|=0..3):",
      out["recommended_set"]["order_histogram"][:4])
print("class acc %.3f | phys R2 mean %.4f" % (
    out["recommended_set"]["class_acc_lda"], out["recommended_set"]["phys_R2"]["MEAN"]))
print("phys R2 per target:", {k: round(v, 4) for k, v in out["recommended_set"]["phys_R2"].items()})
print("first 12 members:", [m["desc"] for m in out["recommended_set"]["members"][:12]])
print("note: only %d subsets have |s|<=2 in total" % out["recommended_set"]["n_subsets_with_|s|<=2_available"])
with open(os.path.join(HERE, "results_latent_recommended.json"), "w") as f:
    json.dump(out, f, indent=2)
