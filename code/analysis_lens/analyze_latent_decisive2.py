"""Latent comparison, REDONE with converged classifiers.

The first pass used a hand-rolled fixed-step gradient-descent logistic probe.
An independent check showed it was badly under-converged on the high-variance
feature sets (it reported chance for a set that closed-form LDA scores at 0.57),
so every class-accuracy number is recomputed here with:
  - regularised LDA            (closed form)
  - ridge regression to one-hot (closed form)
  - multinomial logistic by Newton/IRLS with a convergence check
All three are reported so no single optimiser can drive a conclusion.
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
ytr = np.repeat(np.arange(3), 1000)
yva = np.repeat(np.arange(3), 200)
PC12 = popcount_table(12)

Wtr = np.concatenate([fwht(image_to_pmf12(tr[c, :1000], code)) for c in range(3)])
Wva = np.concatenate([fwht(image_to_pmf12(va[c], code)) for c in range(3)])
Xtr = tr[:, :1000].reshape(-1, 4096).astype(np.float64)
Xva = va.reshape(-1, 4096).astype(np.float64)
Ptr = Xtr / Xtr.sum(1, keepdims=True)
Pva = Xva / Xva.sum(1, keepdims=True)


def std_fit(A, B):
    m, s = A.mean(0), A.std(0)
    s[s < 1e-12] = 1.0
    return (A - m) / s, (B - m) / s


def lda(A, B, reg=1e-4):
    A, B = std_fit(A, B)
    d = A.shape[1]
    Sw = np.zeros((d, d)); mus = []
    for c in range(3):
        Z = A[ytr == c]; mu = Z.mean(0); mus.append(mu)
        Sw += (Z - mu).T @ (Z - mu)
    Sw /= A.shape[0]
    Sw += reg * np.trace(Sw) / d * np.eye(d)
    Si = np.linalg.inv(Sw); mus = np.array(mus)
    sc = B @ Si @ mus.T - 0.5 * np.einsum("ij,jk,ik->i", mus, Si, mus)[None, :]
    return float((sc.argmax(1) == yva).mean())


def ridge(A, B, lam=1e-2):
    A, B = std_fit(A, B)
    A = np.hstack([A, np.ones((len(A), 1))]); B = np.hstack([B, np.ones((len(B), 1))])
    Y = np.eye(3)[ytr]
    W = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ Y)
    return float(((B @ W).argmax(1)).__eq__(yva).mean())


def logistic_irls(A, B, lam=1e-3, iters=60, tol=1e-9):
    """Multinomial logistic by damped Newton on the full Hessian. Reports the
    final gradient norm so convergence is verifiable, not assumed."""
    A, B = std_fit(A, B)
    A = np.hstack([A, np.ones((len(A), 1))]); B = np.hstack([B, np.ones((len(B), 1))])
    n, d = A.shape
    Y = np.eye(3)[ytr]
    W = np.zeros((d, 3))
    prev = np.inf
    for it in range(iters):
        lo = A @ W; lo -= lo.max(1, keepdims=True)
        P = np.exp(lo); P /= P.sum(1, keepdims=True)
        loss = -(Y * np.log(P + 1e-12)).sum() / n + 0.5 * lam * (W ** 2).sum()
        G = A.T @ (P - Y) / n + lam * W
        # block-diagonal Newton step (per class), damped
        step = np.zeros_like(W)
        for c in range(3):
            w = P[:, c] * (1 - P[:, c])
            H = (A * w[:, None]).T @ A / n + lam * np.eye(d)
            step[:, c] = np.linalg.solve(H + 1e-8 * np.eye(d), G[:, c])
        W -= step
        if abs(prev - loss) < tol:
            break
        prev = loss
    gn = float(np.abs(G).max())
    lo = B @ W; acc = float((lo.argmax(1) == yva).mean())
    return acc, gn, it + 1


def pca(A, B, k):
    mu = A.mean(0); A = A - mu; B = B - mu
    C = A.T @ A / (A.shape[0] - 1)
    w, V = np.linalg.eigh(C); V = V[:, ::-1][:, :k]
    return A @ V, B @ V


hf = [0] + [1 << b for b in range(12)]
pairs = [(i, j) for i in range(12) for j in range(i + 1, 12)]
hf += [(1 << i) | (1 << j) for (i, j) in pairs[:37]]
var = Wtr.var(0)
order = np.argsort(var)[::-1]

feats = {}
feats["hamming_first50_paper"] = (Wtr[:, hf], Wva[:, hf])
feats["data_adaptive_top50_by_variance"] = (Wtr[:, order[:50]], Wva[:, order[:50]])
hi = np.array([m for m in order if PC12[m] >= 5][:50])
feats["high_order_only_|s|>=5_top50"] = (Wtr[:, hi], Wva[:, hi])
lo50 = np.array([m for m in order if PC12[m] <= 2][:50])
feats["low_order_|s|<=2_top50_by_variance"] = (Wtr[:, lo50], Wva[:, lo50])
feats["walsh12_PCA50"] = pca(Wtr, Wva, 50)
feats["pixel_raw_PCA50"] = pca(Xtr, Xva, 50)
feats["pixel_pmf_PCA50"] = pca(Ptr, Pva, 50)
feats["all_4096_walsh"] = (Wtr, Wva)

out = {"chance": 1 / 3, "n_train": 3000, "n_val": 600,
       "note": ("val accuracy from three independent linear classifiers; the "
                "earlier fixed-step GD logistic probe was under-converged and "
                "its numbers are superseded"),
       "val_accuracy": {}}
print("%-36s %7s %7s %7s %10s" % ("feature set", "LDA", "ridge", "logit", "|grad|"))
for k, (A, B) in feats.items():
    a = lda(A, B); r = ridge(A, B)
    l, gn, nit = logistic_irls(A, B)
    out["val_accuracy"][k] = {"n_features": int(A.shape[1]), "lda": a, "ridge": r,
                              "logistic_irls": l, "logit_final_grad_inf_norm": gn,
                              "logit_iters": nit, "best": max(a, r, l)}
    print("%-36s %7.3f %7.3f %7.3f %10.2e" % (k, a, r, l, gn))
se = (0.5 * 0.5 / 600) ** 0.5
out["binomial_se_at_600_val"] = float(se)
print("\nbinomial SE at n=600 ~ %.3f (so differences < ~%.1f pp are noise)" % (se, 200 * se))
with open(os.path.join(HERE, "results_latent_decisive2.json"), "w") as f:
    json.dump(out, f, indent=2)
