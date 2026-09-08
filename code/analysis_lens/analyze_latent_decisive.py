"""Decisive latent test.

(A) Is the near-perfect physics recovery an exact identity? For an image-as-pmf,
    the singleton Walsh coefficient of bit b of x satisfies
        W[e_b] = 1 - 2 P(bit_b(x)=1)
    so centroid_x = sum_b 2^b (1 - W[e_b]) / 2 EXACTLY. Test it.

(B) Which 50-subset latent carries the CLASS signal (axion/cdm/no_sub)?
    That is the science target; the smooth geometry is nuisance.
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lens_common import fwht, popcount_table, layout_index, image_to_pmf12

SCRATCH = os.environ.get("LENS_SCRATCH", "/tmp")
val_sub = np.load(os.path.join(SCRATCH, "val_sub.npy"))
train_sub = np.load(os.path.join(SCRATCH, "train_sub.npy"))
out = {}
code = layout_index("standard")

# ---------- (A) exact identity check --------------------------------
im = val_sub.reshape(-1, 64, 64).astype(np.float64)
W = fwht(image_to_pmf12(im, code))
yy, xx = np.mgrid[0:64, 0:64]
X = im.reshape(-1, 4096)
t = X.sum(1)
cx_true = (X * xx.ravel()).sum(1) / t
# bits 6..11 are kx in the standard layout
cx_from_walsh = sum(2 ** b * (1 - W[:, 1 << (6 + b)]) / 2 for b in range(6))
out["exact_identity_centroid_x"] = {
    "max_abs_error": float(np.abs(cx_true - cx_from_walsh).max()),
    "note": "centroid is EXACTLY linear in the 6 singleton Walsh coefficients"}
# second moment from singletons+pairs
r2_true = (X * ((xx.ravel() - 31.5) ** 2 + (yy.ravel() - 31.5) ** 2)).sum(1) / t
acc = np.zeros(im.shape[0])
for ax, off in ((xx, 6), (yy, 0)):
    for b1 in range(6):
        for b2 in range(6):
            s = (1 << (off + b1)) if b1 == b2 else ((1 << (off + b1)) | (1 << (off + b2)))
            v = W[:, 0] if b1 == b2 else W[:, s]
            p11 = (1 - W[:, 1 << (off + b1)] - W[:, 1 << (off + b2)] + W[:, s]) / 4 \
                if b1 != b2 else (1 - W[:, 1 << (off + b1)]) / 2
            acc += 2 ** (b1 + b2) * p11
mean_lin = np.zeros(im.shape[0])
for ax, off in ((xx, 6), (yy, 0)):
    for b in range(6):
        mean_lin += 2 * 31.5 * 2 ** b * (1 - W[:, 1 << (off + b)]) / 2
r2_from_walsh = acc - mean_lin + 2 * 31.5 ** 2
out["exact_identity_second_moment"] = {
    "max_abs_error": float(np.abs(r2_true - r2_from_walsh).max()),
    "note": "2nd moments (size, ellipticity) are exactly linear in singleton+pair coeffs"}

# ---------- (B) which latent carries the class signal ---------------
Wtr = np.concatenate([fwht(image_to_pmf12(train_sub[c, :1000], code)) for c in range(3)])
Wva = np.concatenate([fwht(image_to_pmf12(val_sub[c], code)) for c in range(3)])
ytr = np.repeat(np.arange(3), 1000); yva = np.repeat(np.arange(3), 200)
hf = [0] + [1 << b for b in range(12)]
pairs = [(i, j) for i in range(12) for j in range(i + 1, 12)]
hf += [(1 << i) | (1 << j) for (i, j) in pairs[:37]]

var = Wtr.var(0)
sets = {
    "hamming_first50_paper": np.array(hf, dtype=np.int64),
    "data_adaptive_top50_by_variance": np.argsort(var)[::-1][:50],
    "all_4096": np.arange(4096),
    "high_order_only_|s|>=5_top50": np.array(
        [m for m in np.argsort(var)[::-1] if popcount_table(12)[m] >= 5][:50]),
}


def logreg(A, ytr_, B, yva_, iters=6000, lr=0.5, lam=1e-3):
    m, s = A.mean(0), A.std(0); s[s < 1e-12] = 1.0
    A = (A - m) / s; B = (B - m) / s
    Wl = np.zeros((A.shape[1], 3)); bl = np.zeros(3)
    Y = np.eye(3)[ytr_]
    for _ in range(iters):
        lo = A @ Wl + bl; lo -= lo.max(1, keepdims=True)
        P = np.exp(lo); P /= P.sum(1, keepdims=True)
        G = (P - Y) / A.shape[0]
        Wl -= lr * (A.T @ G + lam * Wl); bl -= lr * G.sum(0)
    return (float(((A @ Wl + bl).argmax(1) == ytr_).mean()),
            float(((B @ Wl + bl).argmax(1) == yva_).mean()))


out["class_accuracy_from_latent"] = {"chance": 1 / 3}
for k, idx in sets.items():
    tr, va = logreg(Wtr[:, idx], ytr, Wva[:, idx], yva)
    out["class_accuracy_from_latent"][k] = {"n_features": int(len(idx)),
                                            "train_acc": tr, "val_acc": va}
    print("%-34s n=%4d train %.3f  val %.3f" % (k, len(idx), tr, va))

# pixel-space reference (top-50 PCs) from results.json
Rj = json.load(open(os.path.join(HERE, "results.json")))
out["class_accuracy_from_latent"]["reference_top50_pixel_PCs"] = \
    Rj["n5_separability"]["logreg_top50_pixel_pcs"]
with open(os.path.join(HERE, "results_latent_decisive.json"), "w") as f:
    json.dump(out, f, indent=2)
print("\nidentity checks:", json.dumps(
    {k: out[k]["max_abs_error"] for k in out if k.startswith("exact")}, indent=1))
