"""Fill in the M sweep around the working point (M=2,6) and run the paper-kernel (MMD-auto3)
control at M=4, i.e. at the setting that actually generalises."""
from __future__ import annotations
import json, math, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
_src = open(os.path.join(HERE, "affine_latent.py")).read()
exec(_src.split("# --------------------------------------------------------------------------- data")[0],
     globals())

import data_lens_r3, codec10, evaluate_r3, iqp_exact, train_iqp
d = data_lens_r3.load_r3(40, seed=0, n_holdout=20)
tr_pmf = np.asarray(d.train_pmfs[:20], dtype=np.float64)
ho_pmf = np.stack([codec10.image_to_pmf(d.holdout_crops[i]) for i in range(20)], axis=0)
tr_t, ho_t = torch.as_tensor(tr_pmf, dtype=DT), torch.as_tensor(ho_pmf, dtype=DT)
mean_pmf = tr_pmf.mean(0); mean_pmf = mean_pmf / mean_pmf.sum()
NTR = NHO = 20
codes = codec10.sample_codes_from_pmf(mean_pmf, 4096, np.random.default_rng(0))
sigmas, _ = train_iqp.resolve_sigmas("auto3", 10, codes, 0)
w_auto = iqp_exact.IQPExact(10, 10, dtype=torch.complex128).kernel_weights(sigmas).to(DT)
TH0 = torch.load(os.path.join(HERE, "states.pt"))["KL_0"]["theta0"]   # reuse a sane theta0 start
tr_q = np.asarray([quants(tr_pmf[i]) for i in range(NTR)])
ho_q = np.asarray([quants(ho_pmf[i]) for i in range(NHO)])
tr_ring = np.asarray([ringiness(img_of(tr_pmf[i])) for i in range(NTR)])
out = {}


def joint(M, kind, steps=4000, lr=0.02, seed=0):
    torch.manual_seed(seed)
    th0 = TH0.clone(); Phi = torch.randn(M, D, dtype=DT); z = 0.001 * torch.randn(NTR, M, dtype=DT)
    lt = make_loss(kind, tr_t, w_auto)
    _b, bs = run_adam([th0, Phi, z], lambda: lt(probs(theta_of(th0, Phi, z)))
                      + 1e-3 * (z ** 2).sum(1).mean(), steps, lr)
    th0, Phi, z = bs
    with torch.no_grad():
        q_tr = probs(theta_of(th0, Phi, z)).numpy()
    lh = make_loss(kind, ho_t, w_auto)
    best = None
    for lrp in (0.02, 0.005):
        zz = torch.zeros(NHO, M, dtype=DT)
        _b2, bs2 = run_adam([zz], lambda: lh(probs(theta_of(th0, Phi, zz)))
                            + 1e-3 * (zz ** 2).sum(1).mean(), 1000, lrp)
        with torch.no_grad():
            qh = probs(theta_of(th0, Phi, bs2[0])).numpy()
        k = float(kl_rows(ho_t, torch.as_tensor(qh)).mean())
        if best is None or k < best[0]:
            best = (k, qh)
    q_ho = best[1]
    ct = np.asarray([corr(img_of(q_tr[i]), img_of(tr_pmf[i])) for i in range(NTR)])
    ch = np.asarray([corr(img_of(q_ho[i]), img_of(ho_pmf[i])) for i in range(NHO)])
    qt = np.asarray([quants(q_tr[i]) for i in range(NTR)])
    # generation
    zn = z.numpy(); mu = zn.mean(0)
    L = np.linalg.cholesky(np.cov((zn - mu).T).reshape(M, M) + 1e-9 * np.eye(M))
    zs = mu[None, :] + np.random.default_rng(7).normal(size=(50, M)) @ L.T
    with torch.no_grad():
        qs = probs(theta_of(th0, Phi, torch.as_tensor(zs, dtype=DT))).numpy()
    rings = np.asarray([ringiness(img_of(qs[i])) for i in range(50)])
    qq = np.asarray([quants(qs[i]) for i in range(50)])
    ok = rings >= float(tr_ring.min())
    r = {"M": M, "objective": kind,
         "train": {"corr_mean": float(ct.mean()), "corr_min": float(ct.min()),
                   "kl": float(kl_rows(tr_t, torch.as_tensor(q_tr)).mean()),
                   "rmse_radius": float(np.sqrt(((qt[:, 0] - tr_q[:, 0]) ** 2).mean())),
                   "rmse_e1": float(np.sqrt(((qt[:, 1] - tr_q[:, 1]) ** 2).mean()))},
         "holdout": {"corr_mean": float(ch.mean()), "corr_min": float(ch.min()), "kl": best[0]},
         "gen": {"ring_mean": float(rings.mean()), "frac_ok": float(ok.mean()),
                 "radius": [float(qq[:, 0].mean()), float(qq[:, 0].std())],
                 "e1_std": float(qq[:, 1].std())},
         "znorm": float(z.norm(dim=1).mean())}
    print("  [M=%2d %-3s] train corr %.4f (min %.4f) kl %.4f | holdout corr %.4f (min %.4f) | "
          "gen ring %.3f frac_ok %.0f%% radius %.2f+-%.2f | ||z|| %.3f"
          % (M, kind, ct.mean(), ct.min(), r["train"]["kl"], ch.mean(), ch.min(),
             rings.mean(), 100 * ok.mean(), qq[:, 0].mean(), qq[:, 0].std(), r["znorm"]), flush=True)
    return r


print("[M sweep around the working point]", flush=True)
for M in (2, 6):
    out["KL_M%d" % M] = joint(M, "KL")
print("[paper kernel at the working point]", flush=True)
out["MMD_M4"] = joint(4, "MMD")
out["KL_M4_seed1"] = joint(4, "KL", seed=1)
with open(os.path.join(HERE, "diag3.json"), "w") as f:
    json.dump(out, f, indent=1, default=float)
print("[diag3] -> diag3.json", flush=True)
