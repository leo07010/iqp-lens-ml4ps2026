"""Final figure: the M=2 working point (fit / project / sample / interpolate) against the
M=4 partial and M=8 collapsed regimes."""
from __future__ import annotations
import json, math, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
_src = open(os.path.join(HERE, "affine_latent.py")).read()
exec(_src.split("# --------------------------------------------------------------------------- data")[0],
     globals())
import data_lens_r3, codec10, evaluate_r3
d = data_lens_r3.load_r3(40, seed=0, n_holdout=20)
tr_pmf = np.asarray(d.train_pmfs[:20], dtype=np.float64)
ho_pmf = np.stack([codec10.image_to_pmf(d.holdout_crops[i]) for i in range(20)], axis=0)
tr_t, ho_t = torch.as_tensor(tr_pmf, dtype=DT), torch.as_tensor(ho_pmf, dtype=DT)
mean_pmf = tr_pmf.mean(0); mean_pmf = mean_pmf / mean_pmf.sum()
NTR = NHO = 20
w0 = torch.zeros(D, dtype=DT)
S = torch.load(os.path.join(HERE, "states.pt"))
TH0 = S["KL_0"]["theta0"]
tr_ring = np.asarray([ringiness(img_of(tr_pmf[i])) for i in range(NTR)])
ho_ring = np.asarray([ringiness(img_of(ho_pmf[i])) for i in range(NHO)])
pairs = [(0, 11), (3, 17), (5, 9), (2, 14), (7, 19)]


def project(th0, Phi, M, lrs=(0.02, 0.005)):
    lh = make_loss("KL", ho_t, w0)
    best = None
    for lrp in lrs:
        z = torch.zeros(NHO, M, dtype=DT)
        _b, bs = run_adam([z], lambda: lh(probs(theta_of(th0, Phi, z)))
                          + 1e-3 * (z ** 2).sum(1).mean(), 1000, lrp)
        with torch.no_grad():
            q = probs(theta_of(th0, Phi, bs[0])).numpy()
        k = float(kl_rows(ho_t, torch.as_tensor(q)).mean())
        if best is None or k < best[0]:
            best = (k, q)
    return best[1]


# --- M=2 refit (same recipe as diag3)
torch.manual_seed(0)
M = 2
th0 = TH0.clone(); Phi = torch.randn(M, D, dtype=DT); z = 0.001 * torch.randn(NTR, M, dtype=DT)
lt = make_loss("KL", tr_t, w0)
_b, bs = run_adam([th0, Phi, z], lambda: lt(probs(theta_of(th0, Phi, z)))
                  + 1e-3 * (z ** 2).sum(1).mean(), 4000, 0.02)
th0, Phi, z2 = bs
with torch.no_grad():
    q_tr2 = probs(theta_of(th0, Phi, z2)).numpy()
q_ho2 = project(th0, Phi, 2)
zn = z2.numpy(); mu = zn.mean(0)
L = np.linalg.cholesky(np.cov((zn - mu).T).reshape(2, 2) + 1e-12 * np.eye(2))
zs = mu[None, :] + np.random.default_rng(7).normal(size=(50, 2)) @ L.T
with torch.no_grad():
    qs2 = probs(theta_of(th0, Phi, torch.as_tensor(zs, dtype=DT))).numpy()
    zmid = torch.stack([(z2[a] + z2[b]) / 2.0 for a, b in pairs], 0)
    qm2 = probs(theta_of(th0, Phi, zmid)).numpy()
c_tr2 = [corr(img_of(q_tr2[i]), img_of(tr_pmf[i])) for i in range(NTR)]
c_ho2 = [corr(img_of(q_ho2[i]), img_of(ho_pmf[i])) for i in range(NHO)]
r_s2 = [ringiness(img_of(qs2[i])) for i in range(50)]
mid2 = [{"ring": ringiness(img_of(qm2[k])),
         "ca": corr(img_of(qm2[k]), img_of(tr_pmf[a])),
         "cb": corr(img_of(qm2[k]), img_of(tr_pmf[b]))} for k, (a, b) in enumerate(pairs)]
print("M=2 train %.4f holdout %.4f gen ring %.3f" % (np.mean(c_tr2), np.mean(c_ho2), np.mean(r_s2)),
      flush=True)
for k, (a, b) in enumerate(pairs):
    print("  mid %d-%d ring %.3f corr %.3f/%.3f  (data A-B %.3f)"
          % (a, b, mid2[k]["ring"], mid2[k]["ca"], mid2[k]["cb"],
             corr(img_of(tr_pmf[a]), img_of(tr_pmf[b]))), flush=True)
json.dump({"M2_train_corr": float(np.mean(c_tr2)), "M2_train_min": float(np.min(c_tr2)),
           "M2_holdout_corr": float(np.mean(c_ho2)), "M2_holdout_min": float(np.min(c_ho2)),
           "M2_gen_ring_mean": float(np.mean(r_s2)), "M2_gen_ring_min": float(np.min(r_s2)),
           "M2_midpoints": mid2}, open(os.path.join(HERE, "fig_stats.json"), "w"), indent=1)

# --- M=4 and M=8 contrast rows from the saved states
q_ho4 = project(S["KL_4"]["theta0"], S["KL_4"]["Phi_raw"], 4)
q_ho8 = project(S["KL_8"]["theta0"], S["KL_8"]["Phi_raw"], 8)
z4 = S["KL_4"]["z"]; mu4 = z4.numpy().mean(0)
L4 = np.linalg.cholesky(np.cov((z4.numpy() - mu4).T) + 1e-9 * np.eye(4))
with torch.no_grad():
    qs4 = probs(theta_of(S["KL_4"]["theta0"], S["KL_4"]["Phi_raw"],
                         torch.as_tensor(mu4[None, :] + np.random.default_rng(7).normal(size=(50, 4)) @ L4.T,
                                         dtype=DT))).numpy()
z8 = S["KL_8"]["z"]; mu8 = z8.numpy().mean(0)
L8 = np.linalg.cholesky(np.cov((z8.numpy() - mu8).T) + 1e-9 * np.eye(8))
with torch.no_grad():
    qs8 = probs(theta_of(S["KL_8"]["theta0"], S["KL_8"]["Phi_raw"],
                         torch.as_tensor(mu8[None, :] + np.random.default_rng(7).normal(size=(50, 8)) @ L8.T,
                                         dtype=DT))).numpy()

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
NC = 6
vmax = float(max(np.max(img_of(x)) for x in [tr_pmf[i] for i in range(NC)] +
                 [q_tr2[i] for i in range(NC)] + [ho_pmf[i] for i in range(NC)]))
fig, ax = plt.subplots(8, NC, figsize=(2.0 * NC, 16.8))


def show(a, im, t, c="k"):
    a.imshow(im, cmap="magma", vmin=0.0, vmax=vmax)
    a.set_title(t, fontsize=7, color=c); a.set_xticks([]); a.set_yticks([])


for j in range(NC):
    show(ax[0, j], img_of(tr_pmf[j]), "train #%d  ring %.2f" % (j, tr_ring[j]))
    show(ax[1, j], img_of(q_tr2[j]), "M=2 recon  corr %.4f" % c_tr2[j], "darkgreen")
    show(ax[2, j], img_of(ho_pmf[j]), "held-out #%d  ring %.2f" % (j, ho_ring[j]))
    show(ax[3, j], img_of(q_ho2[j]), "M=2 projection  corr %.4f" % c_ho2[j], "darkgreen")
    show(ax[4, j], img_of(qs2[j]), "M=2 prior sample  ring %.2f" % r_s2[j], "darkgreen")
for j in range(3):
    show(ax[5, j], img_of(qm2[j]), "M=2 midpoint %d-%d  ring %.2f\ncorr %.2f/%.2f"
         % (pairs[j][0], pairs[j][1], mid2[j]["ring"], mid2[j]["ca"], mid2[j]["cb"]), "darkgreen")
show(ax[5, 3], img_of(mean_pmf), "mean training image")
show(ax[5, 4], img_of(np.asarray(S["KL_0"]["q_tr"][0])), "M=0: one shared theta0")
show(ax[5, 5], img_of(qs2[6]), "M=2 prior sample  ring %.2f" % r_s2[6], "darkgreen")
for j in range(3):
    show(ax[6, j], img_of(q_ho4[j]), "M=4 projection  corr %.4f"
         % corr(img_of(q_ho4[j]), img_of(ho_pmf[j])), "darkorange")
    show(ax[6, 3 + j], img_of(qs4[j]), "M=4 prior sample  ring %.2f"
         % ringiness(img_of(qs4[j])), "darkorange")
for j in range(3):
    show(ax[7, j], img_of(q_ho8[j]), "M=8 projection  corr %.4f"
         % corr(img_of(q_ho8[j]), img_of(ho_pmf[j])), "firebrick")
    show(ax[7, 3 + j], img_of(qs8[j]), "M=8 prior sample  ring %.2f"
         % ringiness(img_of(qs8[j])), "firebrick")
fig.suptitle("Learned affine phase latent:  theta_i = theta0 + Phi^T z_i   (exact 10-qubit IQP, all 1024 phases)\n"
             "green M=2 (works: fit / held-out projection / Gaussian prior samples / midpoints)   "
             "orange M=4 (partial)   red M=8 (memorised, collapses)", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.968))
fig.savefig(os.path.join(HERE, "affine_latent.png"), dpi=110)
print("[fig] %s" % os.path.join(HERE, "affine_latent.png"), flush=True)
