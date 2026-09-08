"""Falsification control for the M=4 result: does a RANDOM (untrained) 4-d phase direction set
generalise to held-out images just as well?  If yes, 'the directions are learned' is not what
is doing the work.  Also quantifies how many M=4 prior samples are actually ring-like.
"""
from __future__ import annotations
import json, math, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
_src = open(os.path.join(HERE, "affine_latent.py")).read()
exec(_src.split("# --------------------------------------------------------------------------- data")[0],
     globals())                                     # defs only: no fitting is re-run

import data_lens_r3, codec10, evaluate_r3
d = data_lens_r3.load_r3(40, seed=0, n_holdout=20)
tr_pmf = np.asarray(d.train_pmfs[:20], dtype=np.float64)
ho_pmf = np.stack([codec10.image_to_pmf(d.holdout_crops[i]) for i in range(20)], axis=0)
ho_t = torch.as_tensor(ho_pmf, dtype=DT)
tr_t = torch.as_tensor(tr_pmf, dtype=DT)
NHO = 20
S = torch.load(os.path.join(HERE, "states.pt"))
w_auto = torch.zeros(D, dtype=DT)                   # unused: KL only here
out = {}

st4 = S["KL_4"]
th0_tr, Phi_tr = st4["theta0"], st4["Phi_raw"]
th0_seed_like = S["KL_0"]["theta0"]                 # theta0 fitted to all 20 images, no latent
loss_ho = make_loss("KL", ho_t, w_auto)

print("[ctrl] held-out projection onto a 4-d subspace: trained vs random directions", flush=True)
rows = {}
cfgs = [("trained Phi + trained theta0", th0_tr, Phi_tr)]
for s in (99, 123):
    torch.manual_seed(s)
    cfgs.append(("random Phi(seed %d) + trained theta0" % s, th0_tr, torch.randn(4, D, dtype=DT)))
torch.manual_seed(99)
cfgs.append(("random Phi(seed 99) + M=0 theta0", th0_seed_like, torch.randn(4, D, dtype=DT)))
cfgs.append(("no latent (theta0 only)", th0_tr, None))
for name, th0, Phi in cfgs:
    if Phi is None:
        with torch.no_grad():
            q = probs(th0.unsqueeze(0).expand(NHO, D)).numpy()
    else:
        best = None
        for lr in (0.02, 0.005):
            z = torch.zeros(NHO, 4, dtype=DT)
            clo = lambda: loss_ho(probs(theta_of(th0, Phi, z))) + 1e-3 * (z ** 2).sum(dim=1).mean()
            _bv, bs = run_adam([z], clo, 1000, lr)
            with torch.no_grad():
                qq = probs(theta_of(th0, Phi, bs[0])).numpy()
            kl = float(kl_rows(ho_t, torch.as_tensor(qq)).mean())
            if best is None or kl < best[0]:
                best = (kl, qq)
        q = best[1]
    cs = np.asarray([corr(img_of(q[i]), img_of(ho_pmf[i])) for i in range(NHO)])
    kl = float(kl_rows(ho_t, torch.as_tensor(q)).mean())
    rows[name] = {"corr_mean": float(cs.mean()), "corr_min": float(cs.min()), "kl": kl}
    print("   %-38s corr %.4f (min %.4f) kl %.4f" % (name, cs.mean(), cs.min(), kl), flush=True)
out["holdout_random_vs_trained"] = rows

# same question on the TRAINING images (z free, everything else frozen)
print("[ctrl] same on the 20 training images", flush=True)
rows2 = {}
loss_tr = make_loss("KL", tr_t, w_auto)
for name, th0, Phi in cfgs:
    if Phi is None:
        with torch.no_grad():
            q = probs(th0.unsqueeze(0).expand(20, D)).numpy()
    else:
        best = None
        for lr in (0.02, 0.005):
            z = torch.zeros(20, 4, dtype=DT)
            clo = lambda: loss_tr(probs(theta_of(th0, Phi, z))) + 1e-3 * (z ** 2).sum(dim=1).mean()
            _bv, bs = run_adam([z], clo, 1000, lr)
            with torch.no_grad():
                qq = probs(theta_of(th0, Phi, bs[0])).numpy()
            kl = float(kl_rows(tr_t, torch.as_tensor(qq)).mean())
            if best is None or kl < best[0]:
                best = (kl, qq)
        q = best[1]
    cs = np.asarray([corr(img_of(q[i]), img_of(tr_pmf[i])) for i in range(20)])
    rows2[name] = {"corr_mean": float(cs.mean()), "corr_min": float(cs.min())}
    print("   %-38s corr %.4f (min %.4f)" % (name, cs.mean(), cs.min()), flush=True)
out["train_random_vs_trained"] = rows2

# how many M=4 prior samples are ring-like?
print("[ctrl] M=4 prior sample quality breakdown", flush=True)
zs_np = st4["z"].numpy()
mu = zs_np.mean(0)
cov = np.cov((zs_np - mu).T) + 1e-9 * np.eye(4)
L = np.linalg.cholesky(cov)
g = np.random.default_rng(7).normal(size=(50, 4))
zs = mu[None, :] + g @ L.T
with torch.no_grad():
    qs = probs(theta_of(th0_tr, Phi_tr, torch.as_tensor(zs, dtype=DT))).numpy()
rings = np.asarray([ringiness(img_of(qs[i])) for i in range(50)])
qq = np.asarray([quants(qs[i]) for i in range(50)])
tr_ring = np.asarray([ringiness(img_of(tr_pmf[i])) for i in range(20)])
thr = float(tr_ring.min())
ok = rings >= thr
out["M4_samples"] = {"train_ring_min": thr, "train_ring_mean": float(tr_ring.mean()),
                     "frac_ring_ge_train_min": float(ok.mean()),
                     "frac_ring_ge_0.7": float((rings >= 0.7).mean()),
                     "radius_ringlike": [float(qq[ok, 0].mean()), float(qq[ok, 0].std())],
                     "radius_all": [float(qq[:, 0].mean()), float(qq[:, 0].std())],
                     "e1_ringlike": [float(qq[ok, 1].mean()), float(qq[ok, 1].std())],
                     "e2_ringlike": [float(qq[ok, 2].mean()), float(qq[ok, 2].std())]}
print("   train ring min %.3f mean %.3f | samples >= train-min: %.0f%%  >=0.70: %.0f%% | "
      "ring-like radius %.2f+-%.2f (train 9.21+-0.15) e1 %.3f+-%.3f"
      % (thr, tr_ring.mean(), 100 * ok.mean(), 100 * (rings >= 0.7).mean(),
         qq[ok, 0].mean(), qq[ok, 0].std(), qq[ok, 1].mean(), qq[ok, 1].std()), flush=True)

with open(os.path.join(HERE, "diag2.json"), "w") as f:
    json.dump(out, f, indent=1, default=float)
print("[diag2] -> diag2.json", flush=True)
