"""Supplement: what do the leading pixel PCs physically encode, and is the
azimuthal structure real or finite-sample noise?

Test 1: correlate each PC score with interpretable geometry (centroid x/y,
        rms radius, total flux, ellipticity components) on held-out val images.
Test 2: split-half replication of the azimuthal profile.
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCRATCH = os.environ.get("LENS_SCRATCH", "/tmp")
CLASSES = ["axion", "cdm", "no_sub"]
val_sub = np.load(os.path.join(SCRATCH, "val_sub.npy"))
train_sub = np.load(os.path.join(SCRATCH, "train_sub.npy"))
C = np.load(os.path.join(HERE, "intermediates.npz"))
out = {}

X = train_sub.reshape(-1, 4096).astype(np.float64)
mu = X.mean(0)
Xc = X - mu
cov = Xc.T @ Xc / (Xc.shape[0] - 1)
w, V = np.linalg.eigh(cov)
V = V[:, ::-1][:, :6]
w = np.clip(w[::-1][:6], 0, None)

Xv = val_sub.reshape(-1, 4096).astype(np.float64)
Z = (Xv - mu) @ V                                   # (600,6)

yy, xx = np.mgrid[0:64, 0:64]
xr, yr = xx.ravel(), yy.ravel()
rr = np.sqrt((xr - 31.5) ** 2 + (yr - 31.5) ** 2)
th = np.arctan2(yr - 31.5, xr - 31.5)
tot = Xv.sum(1)
cx = (Xv * xr).sum(1) / tot - 31.5
cy = (Xv * yr).sum(1) / tot - 31.5
rms = np.sqrt((Xv * rr ** 2).sum(1) / tot)
e1 = (Xv * (rr ** 2 * np.cos(2 * th))).sum(1) / (Xv * rr ** 2).sum(1)
e2 = (Xv * (rr ** 2 * np.sin(2 * th))).sum(1) / (Xv * rr ** 2).sum(1)
feats = {"centroid_x": cx, "centroid_y": cy, "rms_radius": rms,
         "total_flux": tot, "ellip_cos2t": e1, "ellip_sin2t": e2}
out["pc_vs_geometry_corr"] = {}
for i in range(6):
    out["pc_vs_geometry_corr"]["PC%d" % (i + 1)] = {
        "explained_frac": float(w[i] / np.linalg.eigvalsh(cov).sum()),
        **{k: float(np.corrcoef(Z[:, i], v)[0, 1]) for k, v in feats.items()}}
# how much of the top-6 PC subspace is explained by the 6 geometry features
F = np.stack(list(feats.values()), 1)
F = (F - F.mean(0)) / F.std(0)
F = np.hstack([F, np.ones((F.shape[0], 1))])
r2 = {}
for i in range(6):
    z = Z[:, i]
    beta, *_ = np.linalg.lstsq(F, z, rcond=None)
    r2["PC%d" % (i + 1)] = float(1 - ((z - F @ beta) ** 2).sum() / ((z - z.mean()) ** 2).sum())
out["pc_R2_from_6_geometry_features"] = r2

# ---- Test 2: split-half replication of the azimuthal profile -------
nth = 36
tb = ((th + np.pi) / (2 * np.pi) * nth).astype(int) % nth
ann = (rr >= 5) & (rr <= 25)
res = {}
rng = np.random.RandomState(0)
for ci, cn in enumerate(CLASSES):
    im = val_sub[ci].reshape(200, -1).astype(np.float64)
    perm = rng.permutation(200)
    profs = []
    for half in (perm[:100], perm[100:]):
        m = im[half].mean(0)
        a = np.bincount(tb[ann], weights=m[ann], minlength=nth)
        c = np.bincount(tb[ann], minlength=nth)
        profs.append(a / np.maximum(c, 1))
    p1, p2 = profs
    res[cn] = {
        "split_half_corr": float(np.corrcoef(p1, p2)[0, 1]),
        "contrast_half1": float((p1.max() - p1.min()) / p1.mean()),
        "contrast_half2": float((p2.max() - p2.min()) / p2.mean()),
        "contrast_full": float((C["n6_azim"][ci].max() - C["n6_azim"][ci].min())
                               / C["n6_azim"][ci].mean()),
    }
out["azimuthal_split_half"] = res
with open(os.path.join(HERE, "results_pc_semantics.json"), "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
