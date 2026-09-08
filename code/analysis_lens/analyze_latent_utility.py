"""Closing test: is the data-adaptive top-50 Walsh set actually a USEFUL latent?

Fit a ridge regression from the 50 chosen Walsh coefficients to the five
physical parameters that explain 89.4% of pixel variance (x, y, flux, e1, e2),
on 3000 train images, and score on 600 held-out val images. Compare against the
paper's Hamming-first 50 (empty + 12 singletons + 37 two-body).

Also builds the generator-budget table used in the report.
"""
import json, os, sys
import numpy as np
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lens_common import fwht, popcount_table, layout_index, image_to_pmf12

SCRATCH = os.environ.get("LENS_SCRATCH", "/tmp")
CLASSES = ["axion", "cdm", "no_sub"]
val_sub = np.load(os.path.join(SCRATCH, "val_sub.npy"))
train_sub = np.load(os.path.join(SCRATCH, "train_sub.npy"))
R = json.load(open(os.path.join(HERE, "results.json")))
out = {}

code = layout_index("standard")
Wtr = np.concatenate([fwht(image_to_pmf12(train_sub[c, :1000], code)) for c in range(3)])
Wva = np.concatenate([fwht(image_to_pmf12(val_sub[c], code)) for c in range(3)])
Xtr = train_sub[:, :1000].reshape(-1, 4096).astype(np.float64)
Xva = val_sub.reshape(-1, 4096).astype(np.float64)


def geom(X):
    yy, xx = np.mgrid[0:64, 0:64]
    xr, yr = xx.ravel(), yy.ravel()
    rr = np.sqrt((xr - 31.5) ** 2 + (yr - 31.5) ** 2)
    th = np.arctan2(yr - 31.5, xr - 31.5)
    t = X.sum(1)
    w2 = (X * rr ** 2).sum(1)
    return np.stack([(X * xr).sum(1) / t - 31.5, (X * yr).sum(1) / t - 31.5, t,
                     (X * (rr ** 2 * np.cos(2 * th))).sum(1) / w2,
                     (X * (rr ** 2 * np.sin(2 * th))).sum(1) / w2], 1)


names = ["centroid_x", "centroid_y", "total_flux", "ellip_cos2t", "ellip_sin2t"]
Gtr, Gva = geom(Xtr), geom(Xva)
mu_g, sd_g = Gtr.mean(0), Gtr.std(0)
Gtr = (Gtr - mu_g) / sd_g
Gva = (Gva - mu_g) / sd_g

hf = [0] + [1 << b for b in range(12)]
pairs = [(i, j) for i in range(12) for j in range(i + 1, 12)]
hf += [(1 << i) | (1 << j) for (i, j) in pairs[:37]]
sets = {
    "data_adaptive_top50": np.load(os.path.join(HERE, "latent_candidates_top50_pooled.npy")),
    "hamming_first50_paper": np.array(hf, dtype=np.int64),
}
out["latent_to_physics_R2"] = {}
for kname, idx in sets.items():
    A = Wtr[:, idx]; B = Wva[:, idx]
    m, s = A.mean(0), A.std(0); s[s < 1e-12] = 1.0
    A = np.hstack([(A - m) / s, np.ones((A.shape[0], 1))])
    B = np.hstack([(B - m) / s, np.ones((B.shape[0], 1))])
    lam = 1e-3
    beta = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ Gtr)
    pred = B @ beta
    r2 = {names[i]: float(1 - ((Gva[:, i] - pred[:, i]) ** 2).sum() /
                          ((Gva[:, i] - Gva[:, i].mean()) ** 2).sum()) for i in range(5)}
    r2["MEAN"] = float(np.mean(list(r2.values())))
    out["latent_to_physics_R2"][kname] = r2

# ---- generator-budget table ---------------------------------------
sp_std = np.array(R["n7_layout"]["standard"]["order_spectrum_mean"])
sp_gray = np.array(R["n7_layout"]["gray"]["order_spectrum_mean"])
cum = None
Wall = fwht(image_to_pmf12(val_sub.reshape(-1, 64, 64), code))
mE = (Wall ** 2).mean(0)
cum = np.cumsum(np.sort(mE)[::-1]) / mE.sum()
tbl = []


def n_le(n, K):
    return sum(comb(n, k) for k in range(K + 1))


raw18 = np.mean([R["n8_tuple18"]["raw_%s" % c]["energy_k<=6"] for c in CLASSES])
raw18_4 = np.mean([R["n8_tuple18"]["raw_%s" % c]["energy_k<=4"] for c in CLASSES])
sq18 = np.mean([R["n8_tuple18"]["sqrt_%s" % c]["energy_k<=6"] for c in CLASSES])
tbl.append(["18-bit tuple (paper), raw z, |s|<=6", n_le(18, 6), float(raw18)])
tbl.append(["18-bit tuple (paper), sqrt z, |s|<=6", n_le(18, 6), float(sq18)])
tbl.append(["18-bit tuple, raw z, |s|<=4", n_le(18, 4), float(raw18_4)])
tbl.append(["12-bit image-as-pmf, standard, |s|<=4", n_le(12, 4), float(sp_std[:5].sum())])
tbl.append(["12-bit image-as-pmf, standard, |s|<=6", n_le(12, 6), float(sp_std[:7].sum())])
tbl.append(["12-bit image-as-pmf, gray, |s|<=2", n_le(12, 2), float(sp_gray[:3].sum())])
tbl.append(["12-bit image-as-pmf, gray, |s|<=4", n_le(12, 4), float(sp_gray[:5].sum())])
tbl.append(["12-bit image-as-pmf, gray, |s|<=6", n_le(12, 6), float(sp_gray[:7].sum())])
for M in (50, 100, 200, 500, 1000, 2510):
    tbl.append(["12-bit, data-adaptive top-%d" % M, M, float(cum[M - 1])])
RBb = json.load(open(os.path.join(HERE, "results_bits.json")))
ds = RBb["downsample32"]
tbl.append(["10-bit 32x32 image-as-pmf, |s|<=4", n_le(10, 4), float(ds["energy_k<=4"])])
tbl.append(["10-bit 32x32 image-as-pmf, |s|<=6", n_le(10, 6), float(ds["energy_k<=6"])])
out["generator_budget_table"] = [{"scheme": a, "n_generators": int(b),
                                  "walsh_energy_covered": c} for a, b, c in tbl]
with open(os.path.join(HERE, "results_latent_utility.json"), "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out["latent_to_physics_R2"], indent=2))
print("\n%-42s %10s %10s" % ("scheme", "n_gen", "energy"))
for r in out["generator_budget_table"]:
    print("%-42s %10d %10.4f" % (r["scheme"], r["n_generators"], r["walsh_energy_covered"]))
