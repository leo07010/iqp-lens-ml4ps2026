"""Memorisation / distribution check at the M=2 working point (same seed & recipe as fig.py)."""
from __future__ import annotations
import json, math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
_src = open(os.path.join(HERE, "affine_latent.py")).read()
exec(_src.split("# --------------------------------------------------------------------------- data")[0], globals())
import data_lens_r3, codec10, evaluate_r3
d = data_lens_r3.load_r3(40, seed=0, n_holdout=20)
tr_pmf = np.asarray(d.train_pmfs[:20], dtype=np.float64)
ho_pmf = np.stack([codec10.image_to_pmf(d.holdout_crops[i]) for i in range(20)], axis=0)
tr_t = torch.as_tensor(tr_pmf, dtype=DT); NTR = 20
w0 = torch.zeros(D, dtype=DT)
TH0 = torch.load(os.path.join(HERE, "states.pt"))["KL_0"]["theta0"]
torch.manual_seed(0)
th0 = TH0.clone(); Phi = torch.randn(2, D, dtype=DT); z = 0.001 * torch.randn(NTR, 2, dtype=DT)
lt = make_loss("KL", tr_t, w0)
_b, bs = run_adam([th0, Phi, z], lambda: lt(probs(theta_of(th0, Phi, z))) + 1e-3 * (z ** 2).sum(1).mean(),
                  4000, 0.02)
th0, Phi, z2 = bs
zn = z2.numpy(); mu = zn.mean(0)
L = np.linalg.cholesky(np.cov((zn - mu).T).reshape(2, 2) + 1e-12 * np.eye(2))
zs = mu[None, :] + np.random.default_rng(7).normal(size=(50, 2)) @ L.T
with torch.no_grad():
    qs = probs(theta_of(th0, Phi, torch.as_tensor(zs, dtype=DT))).numpy()


def nn(p, bank, skip=-1):
    return min(float(np.sqrt(((p - bank[j]) ** 2).sum())) for j in range(len(bank)) if j != skip)


rings = np.asarray([ringiness(img_of(qs[i])) for i in range(50)])
qq = np.asarray([quants(qs[i]) for i in range(50)])
tq = np.asarray([quants(tr_pmf[i]) for i in range(20)])
hq = np.asarray([quants(ho_pmf[i]) for i in range(20)])
trr = np.asarray([ringiness(img_of(tr_pmf[i])) for i in range(20)])
nn_tr = np.asarray([nn(qs[i], tr_pmf) for i in range(50)])
nn_ho = np.asarray([nn(qs[i], ho_pmf) for i in range(50)])
nn_self = np.asarray([nn(qs[i], qs, skip=i) for i in range(50)])
ho_ho = np.asarray([nn(ho_pmf[i], ho_pmf, skip=i) for i in range(20)])
ho_tr = np.asarray([nn(ho_pmf[i], tr_pmf) for i in range(20)])
out = {"M2_gen": {"ring": [float(rings.mean()), float(rings.min())],
                  "frac_ge_train_min": float((rings >= trr.min()).mean()),
                  "radius": [float(qq[:, 0].mean()), float(qq[:, 0].std())],
                  "e1": [float(qq[:, 1].mean()), float(qq[:, 1].std())],
                  "e2": [float(qq[:, 2].mean()), float(qq[:, 2].std())],
                  "nn_l2_train": [float(nn_tr.mean()), float(nn_tr.min())],
                  "nn_l2_holdout": [float(nn_ho.mean()), float(nn_ho.min())],
                  "nn_l2_within_samples": [float(nn_self.mean()), float(nn_self.min())]},
       "real": {"train_ring": [float(trr.mean()), float(trr.min())],
                "train_radius": [float(tq[:, 0].mean()), float(tq[:, 0].std())],
                "train_e1": [float(tq[:, 1].mean()), float(tq[:, 1].std())],
                "train_e2": [float(tq[:, 2].mean()), float(tq[:, 2].std())],
                "holdout_radius": [float(hq[:, 0].mean()), float(hq[:, 0].std())],
                "holdout_e1_std": float(hq[:, 1].std()),
                "ho_nn_ho": [float(ho_ho.mean()), float(ho_ho.min())],
                "ho_nn_tr": [float(ho_tr.mean()), float(ho_tr.min())]}}
json.dump(out, open(os.path.join(HERE, "diag4.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
