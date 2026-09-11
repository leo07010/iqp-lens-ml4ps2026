"""Follow-up diagnostics: the trained ||z|| is ~0.03-0.3, so Adam lr=0.02 on a FROZEN subspace
moves z by ~its entire useful range every 5 steps. Redo experiment 2 with a proper lr sweep,
measure how sharp the latent is (corr vs perturbation radius), and rebuild the figure.
"""
from __future__ import annotations
import json, math, os, sys, time
sys.path.insert(0, "<REPO_ROOT>/code")
import numpy as np, torch
torch.set_num_threads(4)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from affine_latent import (probs, corr, ringiness, img_of, quants, kl_rows, theta_of, make_loss,
                           run_adam, tr_pmf, ho_pmf, tr_t, ho_t, mean_pmf, tr_q, ho_q, tr_ring,
                           ho_ring, w_auto, DT, D, OUT, NTR, NHO, STATES, GEN, RES, gaussian_sample,
                           nn_l2, pairs)

t0 = time.time()
OUTJ = {}

# ---------------------------------------------------------------- A. projection lr sweep
print("[A] experiment-2 redo: frozen subspace, z-only fit, lr sweep", flush=True)
A = {}
for M in (4, 8, 16):
    st = STATES[("KL", M)]
    th0, Phi = st["theta0"], st["Phi_raw"]
    zmu = st["z"].mean(dim=0, keepdim=True)
    loss_ho = make_loss("KL", ho_t, w_auto)
    rows = {}
    for lr in (0.02, 0.005, 0.001):
        for init_name, z0 in (("zero", torch.zeros(NHO, M, dtype=DT)),
                              ("zmean", zmu.repeat(NHO, 1))):
            z = z0.clone()
            clo = lambda: loss_ho(probs(theta_of(th0, Phi, z))) + 1e-3 * (z ** 2).sum(dim=1).mean()
            _bv, bs = run_adam([z], clo, 1000, lr)
            with torch.no_grad():
                q = probs(theta_of(th0, Phi, bs[0])).numpy()
            cs = np.asarray([corr(img_of(q[i]), img_of(ho_pmf[i])) for i in range(NHO)])
            kl = float(kl_rows(ho_t, torch.as_tensor(q)).mean())
            rows["lr%.3f_%s" % (lr, init_name)] = {
                "corr_mean": float(cs.mean()), "corr_min": float(cs.min()), "kl": kl,
                "znorm_mean": float(bs[0].norm(dim=1).mean())}
            print("   M=%2d lr=%.3f init=%-5s -> corr %.4f (min %.4f) kl %.4f ||z|| %.3f"
                  % (M, lr, init_name, cs.mean(), cs.min(), kl, bs[0].norm(dim=1).mean()), flush=True)
            if rows["lr%.3f_%s" % (lr, init_name)]["corr_mean"] == max(
                    r["corr_mean"] for r in rows.values()):
                A.setdefault("_best_q", {})[M] = q
    best = max((k for k in rows), key=lambda k: rows[k]["corr_mean"])
    A["M%d" % M] = {"sweep": rows, "best_cfg": best, "best": rows[best]}
OUTJ["exp2_redo"] = {k: v for k, v in A.items() if k != "_best_q"}

# ---------------------------------------------------------------- B. latent sharpness
print("[B] sharpness: corr( recon(z_i + eps*u), recon(z_i) ) vs eps", flush=True)
B = {}
g = torch.Generator().manual_seed(5)
for M in (4, 8, 16):
    st = STATES[("KL", M)]
    row = {}
    for eps in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5):
        cs = []
        for i in range(5):
            u = torch.randn(3, M, dtype=DT, generator=g)
            u = u / u.norm(dim=1, keepdim=True)
            zz = st["z"][i:i + 1] + eps * u
            with torch.no_grad():
                q = probs(theta_of(st["theta0"], st["Phi_raw"], zz)).numpy()
            for k in range(3):
                cs.append(corr(img_of(q[k]), img_of(np.asarray(st["q_tr"][i]))))
        row["%.2f" % eps] = float(np.mean(cs))
    sv = np.linalg.svd(st["z"].numpy() - st["z"].numpy().mean(0), compute_uv=False)
    row["z_spread_sv1"] = float(sv[0])
    B["M%d" % M] = row
    print("   M=%2d  " % M + "  ".join("eps=%s:%.3f" % (k, v) for k, v in row.items()
                                       if k != "z_spread_sv1")
          + "   (z spread sv1=%.3f)" % sv[0], flush=True)
OUTJ["sharpness"] = B

# ---------------------------------------------------------------- C. M=4 midpoints + samples
print("[C] M=4 midpoints and prior samples", flush=True)
st4 = STATES[("KL", 4)]
z4 = st4["z"]
with torch.no_grad():
    zmid = torch.stack([(z4[a] + z4[b]) / 2.0 for a, b in pairs], dim=0)
    q_mid4 = probs(theta_of(st4["theta0"], st4["Phi_raw"], zmid)).numpy()
mid4 = []
for k, (a, b) in enumerate(pairs):
    r = ringiness(img_of(q_mid4[k]))
    ca = corr(img_of(q_mid4[k]), img_of(tr_pmf[a]))
    cb = corr(img_of(q_mid4[k]), img_of(tr_pmf[b]))
    rad, e1, e2 = quants(q_mid4[k])
    mid4.append({"pair": [a, b], "ringiness": r, "corr_a": ca, "corr_b": cb, "radius": rad,
                 "corr_ab_data": corr(img_of(tr_pmf[a]), img_of(tr_pmf[b]))})
    print("   mid %d-%d ring %.3f corrA %.4f corrB %.4f (data A-B %.4f) radius %.2f"
          % (a, b, r, ca, cb, mid4[-1]["corr_ab_data"], rad), flush=True)
OUTJ["midpoints_M4"] = mid4

with open(os.path.join(OUT, "diag.json"), "w") as f:
    json.dump(OUTJ, f, indent=1, default=float)
print("[diag] %.0fs -> diag.json" % (time.time() - t0), flush=True)

# ---------------------------------------------------------------- figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

q_tr4 = np.asarray(st4["q_tr"])
q_ho4 = A["_best_q"][4]
qs4 = GEN[4][0]
rings4 = GEN[4][1]
q_ho8 = A["_best_q"][8]
qs8 = GEN[8][0]
NC = 6
vmax = float(max(np.max(img_of(x)) for x in
                 [tr_pmf[i] for i in range(NC)] + [q_tr4[i] for i in range(NC)] +
                 [ho_pmf[i] for i in range(NC)] + [q_ho4[i] for i in range(NC)]))
c_tr4 = RES["configs"]["KL_M4"]["train"]["corrs"]
c_ho4 = [corr(img_of(q_ho4[i]), img_of(ho_pmf[i])) for i in range(NHO)]
c_ho8 = [corr(img_of(q_ho8[i]), img_of(ho_pmf[i])) for i in range(NHO)]

fig, ax = plt.subplots(7, NC, figsize=(2.0 * NC, 14.8))


def show(a, im, title, color="k"):
    a.imshow(im, cmap="magma", vmin=0.0, vmax=vmax)
    a.set_title(title, fontsize=7, color=color)
    a.set_xticks([]); a.set_yticks([])


for j in range(NC):
    show(ax[0, j], img_of(tr_pmf[j]), "train #%d  ring %.2f" % (j, tr_ring[j]))
    show(ax[1, j], img_of(q_tr4[j]), "recon M=4  corr %.4f" % c_tr4[j])
    show(ax[2, j], img_of(ho_pmf[j]), "held-out #%d  ring %.2f" % (j, ho_ring[j]))
    show(ax[3, j], img_of(q_ho4[j]), "project M=4  corr %.4f" % c_ho4[j])
    show(ax[4, j], img_of(qs4[j]), "M=4 prior sample  ring %.2f" % rings4[j])
for j in range(3):
    show(ax[5, j], img_of(q_mid4[j]), "M=4 mid %d-%d  ring %.2f\ncorr %.2f/%.2f"
         % (pairs[j][0], pairs[j][1], mid4[j]["ringiness"], mid4[j]["corr_a"], mid4[j]["corr_b"]))
show(ax[5, 3], img_of(mean_pmf), "mean train image")
show(ax[5, 4], img_of(np.asarray(STATES[("KL", 0)]["q_tr"][0])), "M=0 (one shared theta0)")
show(ax[5, 5], img_of(qs4[6]), "M=4 prior sample  ring %.2f" % rings4[6])
for j in range(3):
    show(ax[6, j], img_of(q_ho8[j]), "project M=8  corr %.4f" % c_ho8[j], color="firebrick")
for j in range(3):
    show(ax[6, 3 + j], img_of(qs8[j]), "M=8 prior sample  ring %.2f" % GEN[8][1][j],
         color="firebrick")
fig.suptitle("Learned affine phase latent  theta_i = theta0 + Phi^T z_i  (exact 10-qubit IQP, 1024 phases)\n"
             "rows: train / M=4 recon / held-out / M=4 projection / M=4 prior samples / "
             "M=4 midpoints+controls / M=8 failure (red)", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.965))
fig.savefig(os.path.join(OUT, "affine_latent.png"), dpi=110)
print("[fig] %s" % os.path.join(OUT, "affine_latent.png"), flush=True)
