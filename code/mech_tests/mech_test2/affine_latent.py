"""Mechanism test 2: is the latent direction LEARNABLE in phase space?

Model: theta_i = theta0 + Phi^T z_i,  theta0 in R^1024, Phi in R^{M x 1024}, z_i in R^M.
Joint training of {theta0, Phi, z}. Read-only w.r.t. the repo.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

REPO = "<REPO_ROOT>/code"
sys.path.insert(0, REPO)

import numpy as np
import torch

torch.set_num_threads(4)

import codec10
import data_lens_r3
import evaluate_r3
import iqp_exact
import train_iqp
from iqp_exact import fwht

OUT = os.path.dirname(os.path.abspath(__file__))
DT = torch.float64
NQ = 10
D = 1 << NQ
SCALE = math.sqrt(float(D))          # ||Phi_m||_2 = sqrt(D)  =>  z_m = 1 shifts each phase by ~1 rad rms
RES = {}


# --------------------------------------------------------------------------- simulator
def probs(theta):
    """theta (..., 1024) in Walsh/mask order -> probs (..., 1024). Batched, autograd-safe."""
    phi = fwht(theta)
    re = fwht(torch.cos(phi))
    im = fwht(torch.sin(phi))
    return (re * re + im * im) / float(D * D)


# --------------------------------------------------------------------------- metrics
def corr(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    dd = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    return float((a * b).sum() / dd) if dd > 0 else 0.0


_RING_R0 = np.arange(2.0, 14.01, 0.5)
_RING_SIG = np.arange(0.5, 6.01, 0.5)


def ringiness(img):
    """max over (r0,sigma) of corr(image, Gaussian annulus) about the image's own centroid."""
    w = np.clip(np.asarray(img, dtype=np.float64), 0.0, None)
    tot = float(w.sum())
    if not (tot > 0):
        return 0.0
    ny, nx = w.shape
    ys = np.arange(ny, dtype=np.float64)[:, None]
    xs = np.arange(nx, dtype=np.float64)[None, :]
    cy = float((w * ys).sum() / tot)
    cx = float((w * xs).sum() / tot)
    r = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
    wf = w.ravel() - w.mean()
    dw = math.sqrt(float((wf * wf).sum()))
    if dw <= 0:
        return 0.0
    best = -1.0
    for r0 in _RING_R0:
        for sg in _RING_SIG:
            t = np.exp(-0.5 * ((r - r0) / sg) ** 2).ravel()
            t = t - t.mean()
            dt = math.sqrt(float((t * t).sum()))
            if dt <= 0:
                continue
            c = float((wf * t).sum() / (dt * dw))
            if c > best:
                best = c
    return best


def img_of(pmf):
    return codec10.pmf_to_image(np.asarray(pmf, dtype=np.float64), 1.0)


def quants(pmf):
    q = evaluate_r3.image_quantities(img_of(pmf))
    return q["radius"], q["e1"], q["e2"]


def kl_rows(p, q):
    ce = -(p * torch.log(q.clamp_min(1e-30))).sum(dim=-1)
    ent = -(p * torch.log(p.clamp_min(1e-30))).sum(dim=-1)
    return ce - ent


# --------------------------------------------------------------------------- model
def theta_of(theta0, Phi_raw, z):
    if Phi_raw is None:
        return theta0.unsqueeze(0).expand(z.shape[0] if z is not None else 1, D)
    Phi = Phi_raw / Phi_raw.norm(dim=1, keepdim=True).clamp_min(1e-12) * SCALE
    return theta0.unsqueeze(0) + z @ Phi


def make_loss(kind, p_t, w_auto):
    ez_data = fwht(p_t)
    if kind == "KL":
        return lambda q: kl_rows(p_t, q).mean()
    return lambda q: (((fwht(q) - ez_data) ** 2) * w_auto.unsqueeze(0)).sum(dim=-1).mean()


def run_adam(params, closure, steps, lr):
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.05)
    best, best_state = float("inf"), [p.detach().clone() for p in params]
    for it in range(steps):
        opt.zero_grad()
        L = closure()
        L.backward()
        opt.step()
        sch.step()
        v = float(L.detach())
        if v < best:
            best, best_state = v, [p.detach().clone() for p in params]
    return best, best_state


# --------------------------------------------------------------------------- data
t_start = time.time()
d = data_lens_r3.load_r3(40, seed=0, n_holdout=20)
tr_pmf = np.asarray(d.train_pmfs[:20], dtype=np.float64)          # 20 training images
ho_pmf = np.stack([codec10.image_to_pmf(d.holdout_crops[i]) for i in range(20)], axis=0)
tr_t = torch.as_tensor(tr_pmf, dtype=DT)
ho_t = torch.as_tensor(ho_pmf, dtype=DT)
mean_pmf = tr_pmf.mean(axis=0)
mean_pmf = mean_pmf / mean_pmf.sum()
mean_t = torch.as_tensor(mean_pmf, dtype=DT).unsqueeze(0)
NTR, NHO = tr_pmf.shape[0], ho_pmf.shape[0]

rng = np.random.default_rng(0)
codes = codec10.sample_codes_from_pmf(mean_pmf, 4096, rng)
sigmas, mpw = train_iqp.resolve_sigmas("auto3", NQ, codes, 0)
w_auto = iqp_exact.IQPExact(NQ, NQ, dtype=torch.complex128).kernel_weights(sigmas).to(DT)
RES["sigmas_auto3"] = sigmas
RES["mean_pauli_weights"] = mpw
print("[data] train %d, holdout %d, sigmas=%s (%.2fs)" %
      (NTR, NHO, ["%.3f" % s for s in sigmas], time.time() - t_start), flush=True)

# reference quantities
tr_q = np.asarray([quants(tr_pmf[i]) for i in range(NTR)])
ho_q = np.asarray([quants(ho_pmf[i]) for i in range(NHO)])
tr_ring = np.asarray([ringiness(img_of(tr_pmf[i])) for i in range(NTR)])
ho_ring = np.asarray([ringiness(img_of(ho_pmf[i])) for i in range(NHO)])


# --------------------------------------------------------------------------- theta0 seed
torch.manual_seed(0)
th0_seed = (0.01 * torch.randn(D, dtype=DT))
loss_mean = make_loss("KL", mean_t, w_auto)
_bv, _bs = run_adam([th0_seed], lambda: loss_mean(probs(th0_seed.unsqueeze(0))), 1000, 0.05)
TH0_SEED = _bs[0]
q_mean = probs(TH0_SEED.unsqueeze(0)).detach().numpy()[0]
print("[init] mean-image single-shot fit: KL %.4f corr %.4f (%.1fs)"
      % (_bv, corr(img_of(q_mean), img_of(mean_pmf)), time.time() - t_start), flush=True)
RES["theta0_seed"] = {"kl": _bv, "corr": corr(img_of(q_mean), img_of(mean_pmf))}


# --------------------------------------------------------------------------- experiment 1+2
def eval_rows(qs, pmfs, refq):
    cs = np.asarray([corr(img_of(qs[i]), img_of(pmfs[i])) for i in range(len(pmfs))])
    qq = np.asarray([quants(qs[i]) for i in range(len(pmfs))])
    return {"corr_mean": float(cs.mean()), "corr_min": float(cs.min()),
            "corr_med": float(np.median(cs)),
            "rmse_radius": float(np.sqrt(((qq[:, 0] - refq[:, 0]) ** 2).mean())),
            "rmse_e1": float(np.sqrt(((qq[:, 1] - refq[:, 1]) ** 2).mean())),
            "corrs": cs.tolist()}


def run_config(M, kind, steps, lr=0.02, seed=0, zpen=1e-3, ho_steps=500):
    t0 = time.time()
    torch.manual_seed(seed)
    theta0 = TH0_SEED.clone()
    loss_tr = make_loss(kind, tr_t, w_auto)
    if M == 0:
        params = [theta0]
        clo = lambda: loss_tr(probs(theta0.unsqueeze(0).expand(NTR, D)))
    else:
        Phi_raw = torch.randn(M, D, dtype=DT)
        z = 0.001 * torch.randn(NTR, M, dtype=DT)
        params = [theta0, Phi_raw, z]
        clo = lambda: loss_tr(probs(theta_of(theta0, Phi_raw, z))) + zpen * (z ** 2).sum(dim=1).mean()
    bv, bs = run_adam(params, clo, steps, lr)
    theta0 = bs[0]
    Phi_raw = bs[1] if M > 0 else None
    z = bs[2] if M > 0 else None
    with torch.no_grad():
        th = theta_of(theta0, Phi_raw, z) if M > 0 else theta0.unsqueeze(0).expand(NTR, D)
        q_tr = probs(th).numpy()
        kl_tr = float(kl_rows(tr_t, torch.as_tensor(q_tr)).mean())
    rec = {"M": M, "objective": kind, "steps": steps, "loss": bv, "kl_train": kl_tr,
           "train": eval_rows(q_tr, tr_pmf, tr_q), "secs": time.time() - t0}
    # --- experiment 2: project held-out onto the frozen subspace (z only)
    if M > 0:
        theta0_f = theta0.detach().clone()
        Phi_f = Phi_raw.detach().clone()
        z_ho = torch.zeros(NHO, M, dtype=DT)
        loss_ho = make_loss(kind, ho_t, w_auto)
        clo2 = lambda: loss_ho(probs(theta_of(theta0_f, Phi_f, z_ho))) + zpen * (z_ho ** 2).sum(dim=1).mean()
        bv2, bs2 = run_adam([z_ho], clo2, ho_steps, lr)
        z_ho = bs2[0]
        with torch.no_grad():
            q_ho = probs(theta_of(theta0_f, Phi_f, z_ho)).numpy()
            kl_ho = float(kl_rows(ho_t, torch.as_tensor(q_ho)).mean())
    else:
        with torch.no_grad():
            q_ho = probs(theta0.unsqueeze(0).expand(NHO, D)).numpy()
            kl_ho = float(kl_rows(ho_t, torch.as_tensor(q_ho)).mean())
        z_ho = None
    rec["kl_holdout"] = kl_ho
    rec["holdout"] = eval_rows(q_ho, ho_pmf, ho_q)
    rec["secs"] = time.time() - t0
    print("  [M=%2d %s] train corr %.4f (min %.4f) kl %.4f | holdout corr %.4f (min %.4f) kl %.4f "
          "| rmse rad %.3f e1 %.3f | %.0fs"
          % (M, kind, rec["train"]["corr_mean"], rec["train"]["corr_min"], kl_tr,
             rec["holdout"]["corr_mean"], rec["holdout"]["corr_min"], kl_ho,
             rec["train"]["rmse_radius"], rec["train"]["rmse_e1"], rec["secs"]), flush=True)
    state = {"theta0": theta0.detach(), "Phi_raw": (Phi_raw.detach() if M > 0 else None),
             "z": (z.detach() if M > 0 else None), "z_ho": (z_ho.detach() if z_ho is not None else None),
             "q_tr": q_tr, "q_ho": q_ho}
    return rec, state


print("[exp1+2] joint fits", flush=True)
RES["configs"] = {}
STATES = {}
for M, st in [(0, 2000), (4, 4000), (8, 4000), (16, 4000)]:
    r, s = run_config(M, "KL", st)
    RES["configs"]["KL_M%d" % M] = r
    STATES[("KL", M)] = s

print("[exp4] MMD-auto3 control at M=8", flush=True)
r, s = run_config(8, "MMD", 4000)
RES["configs"]["MMD_M8"] = r
STATES[("MMD", 8)] = s

# --------------------------------------------------------------------------- controls
copy_ub = np.asarray([max(corr(img_of(ho_pmf[i]), img_of(tr_pmf[j])) for j in range(NTR))
                      for i in range(NHO)])
mean_ho = np.asarray([corr(img_of(ho_pmf[i]), img_of(mean_pmf)) for i in range(NHO)])
mean_tr = np.asarray([corr(img_of(tr_pmf[i]), img_of(mean_pmf)) for i in range(NTR)])
RES["controls"] = {
    "holdout_vs_nearest_train_corr": {"mean": float(copy_ub.mean()), "min": float(copy_ub.min())},
    "holdout_vs_mean_image_corr": {"mean": float(mean_ho.mean()), "min": float(mean_ho.min())},
    "train_vs_mean_image_corr": {"mean": float(mean_tr.mean()), "min": float(mean_tr.min())},
}
print("[ctrl] holdout vs nearest-train corr %.4f (min %.4f); vs mean image %.4f; train vs mean %.4f"
      % (copy_ub.mean(), copy_ub.min(), mean_ho.mean(), mean_tr.mean()), flush=True)


# --------------------------------------------------------------------------- experiment 3
def l2_pmf(a, b):
    return float(np.sqrt(((np.asarray(a) - np.asarray(b)) ** 2).sum()))


def nn_l2(p, bank, skip=-1):
    best = float("inf")
    for j in range(len(bank)):
        if j == skip:
            continue
        v = l2_pmf(p, bank[j])
        if v < best:
            best = v
    return best


def gaussian_sample(z_np, n, rank, seed):
    mu = z_np.mean(axis=0)
    Xc = z_np - mu
    if rank is None or rank >= z_np.shape[1]:
        cov = np.cov(Xc.T) + 1e-9 * np.eye(z_np.shape[1])
        L = np.linalg.cholesky(cov)
        g = np.random.default_rng(seed).normal(size=(n, z_np.shape[1]))
        return mu[None, :] + g @ L.T, {"kind": "full-cov", "rank": z_np.shape[1]}
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    k = int(rank)
    sd = S[:k] / math.sqrt(max(z_np.shape[0] - 1, 1))
    g = np.random.default_rng(seed).normal(size=(n, k))
    return mu[None, :] + (g * sd[None, :]) @ Vt[:k], {"kind": "rank-%d PCA" % k, "rank": k,
                                                      "var_kept": float((S[:k] ** 2).sum() / (S ** 2).sum())}


def generation_block(M, tag, n_samples=50, seed=7):
    st = STATES[("KL", M)]
    z_np = st["z"].numpy()
    rank = None if M <= 8 else 8
    zs, meta = gaussian_sample(z_np, n_samples, rank, seed)
    with torch.no_grad():
        qs = probs(theta_of(st["theta0"], st["Phi_raw"], torch.as_tensor(zs, dtype=DT))).numpy()
    rings = np.asarray([ringiness(img_of(qs[i])) for i in range(n_samples)])
    qq = np.asarray([quants(qs[i]) for i in range(n_samples)])
    nn_tr = np.asarray([nn_l2(qs[i], tr_pmf) for i in range(n_samples)])
    nn_ho = np.asarray([nn_l2(qs[i], ho_pmf) for i in range(n_samples)])
    out = {"prior": meta, "n": n_samples,
           "ringiness": {"mean": float(rings.mean()), "min": float(rings.min()),
                         "max": float(rings.max()), "p25": float(np.percentile(rings, 25))},
           "radius": {"mean": float(qq[:, 0].mean()), "std": float(qq[:, 0].std()),
                      "min": float(qq[:, 0].min()), "max": float(qq[:, 0].max())},
           "e1": {"mean": float(qq[:, 1].mean()), "std": float(qq[:, 1].std())},
           "e2": {"mean": float(qq[:, 2].mean()), "std": float(qq[:, 2].std())},
           "nn_l2_train": {"mean": float(nn_tr.mean()), "min": float(nn_tr.min())},
           "nn_l2_holdout": {"mean": float(nn_ho.mean()), "min": float(nn_ho.min())}}
    print("  [gen M=%d %s] ring %.3f (min %.3f) | radius %.2f+-%.2f | e1 %.3f+-%.3f | "
          "nnL2 train %.4f holdout %.4f"
          % (M, meta["kind"], rings.mean(), rings.min(), qq[:, 0].mean(), qq[:, 0].std(),
             qq[:, 1].mean(), qq[:, 1].std(), nn_tr.mean(), nn_ho.mean()), flush=True)
    return out, qs, rings, qq


print("[exp3] generation from a Gaussian prior on the training z", flush=True)
RES["generation"] = {}
GEN = {}
for M in (4, 8, 16):
    o, qs, rings, qq = generation_block(M, "KL")
    RES["generation"]["M%d" % M] = o
    GEN[M] = (qs, rings, qq)

# real-data reference scales for the L2 numbers
ho_nn_ho = np.asarray([nn_l2(ho_pmf[i], ho_pmf, skip=i) for i in range(NHO)])
ho_nn_tr = np.asarray([nn_l2(ho_pmf[i], tr_pmf) for i in range(NHO)])
mean_nn_tr = nn_l2(mean_pmf, tr_pmf)
RES["generation"]["reference_l2"] = {
    "holdout_to_nearest_other_holdout": {"mean": float(ho_nn_ho.mean()), "min": float(ho_nn_ho.min())},
    "holdout_to_nearest_train": {"mean": float(ho_nn_tr.mean()), "min": float(ho_nn_tr.min())},
    "mean_image_to_nearest_train": mean_nn_tr,
    "train_ringiness": {"mean": float(tr_ring.mean()), "min": float(tr_ring.min())},
    "holdout_ringiness": {"mean": float(ho_ring.mean()), "min": float(ho_ring.min())},
    "train_radius": {"mean": float(tr_q[:, 0].mean()), "std": float(tr_q[:, 0].std())},
    "train_e1": {"mean": float(tr_q[:, 1].mean()), "std": float(tr_q[:, 1].std())},
    "train_e2": {"mean": float(tr_q[:, 2].mean()), "std": float(tr_q[:, 2].std())},
}
print("  [ref] holdout nnL2 to other holdout %.4f, to train %.4f; mean-image to train %.4f; "
      "train ring %.3f radius %.2f+-%.2f e1sd %.3f"
      % (ho_nn_ho.mean(), ho_nn_tr.mean(), mean_nn_tr, tr_ring.mean(),
         tr_q[:, 0].mean(), tr_q[:, 0].std(), tr_q[:, 1].std()), flush=True)

# midpoints of 5 training pairs (M=8)
st8 = STATES[("KL", 8)]
z8 = st8["z"]
pairs = [(0, 11), (3, 17), (5, 9), (2, 14), (7, 19)]
mid_rows = []
with torch.no_grad():
    zmid = torch.stack([(z8[a] + z8[b]) / 2.0 for a, b in pairs], dim=0)
    q_mid = probs(theta_of(st8["theta0"], st8["Phi_raw"], zmid)).numpy()
for k, (a, b) in enumerate(pairs):
    r = ringiness(img_of(q_mid[k]))
    ca = corr(img_of(q_mid[k]), img_of(tr_pmf[a]))
    cb = corr(img_of(q_mid[k]), img_of(tr_pmf[b]))
    rad, e1, e2 = quants(q_mid[k])
    mid_rows.append({"pair": [a, b], "ringiness": r, "corr_a": ca, "corr_b": cb,
                     "radius": rad, "e1": e1, "corr_ab_data": corr(img_of(tr_pmf[a]), img_of(tr_pmf[b]))})
    print("  [mid %d-%d] ring %.3f corrA %.4f corrB %.4f (data A-B %.4f) radius %.2f"
          % (a, b, r, ca, cb, mid_rows[-1]["corr_ab_data"], rad), flush=True)
RES["midpoints"] = mid_rows

RES["params"] = {"shared_M8": D + 8 * D, "per_image_M8": 8,
                 "shared_M4": D + 4 * D, "shared_M16": D + 16 * D,
                 "paper_31k_50": "paper: ~31k shared gate params, 50 latent phases per sample"}

with open(os.path.join(OUT, "results.json"), "w") as f:
    json.dump(RES, f, indent=1, default=float)
print("[done] %.1fs -> results.json" % (time.time() - t_start), flush=True)
torch.save({("%s_%d" % (k[0], k[1])): {kk: (vv if vv is None else torch.as_tensor(vv))
                                       for kk, vv in v.items()}
            for k, v in STATES.items()}, os.path.join(OUT, "states.pt"))
_zs = {("%s_M%d" % (k[0], k[1])): {"norm_mean": float(v["z"].norm(dim=1).mean()),
                                   "norm_max": float(v["z"].norm(dim=1).max()),
                                   "sv": np.linalg.svd(v["z"].numpy() - v["z"].numpy().mean(0),
                                                       compute_uv=False).tolist()}
       for k, v in STATES.items() if v["z"] is not None}
RES["z_stats"] = _zs
for k, v in _zs.items():
    print("  [z] %s ||z|| mean %.3f max %.3f  sv=%s" %
          (k, v["norm_mean"], v["norm_max"], ["%.2f" % x for x in v["sv"][:6]]), flush=True)
with open(os.path.join(OUT, "results.json"), "w") as f:
    json.dump(RES, f, indent=1, default=float)


# --------------------------------------------------------------------------- figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NC = 6
fig, ax = plt.subplots(6, NC, figsize=(2.0 * NC, 12.6))
st = STATES[("KL", 8)]
allimg = [img_of(tr_pmf[i]) for i in range(NC)] + [img_of(st["q_tr"][i]) for i in range(NC)] + \
         [img_of(ho_pmf[i]) for i in range(NC)] + [img_of(st["q_ho"][i]) for i in range(NC)]
vmax = float(max(np.max(x) for x in allimg))
qs8 = GEN[8][0]


def show(a, im, title):
    a.imshow(im, cmap="magma", vmin=0.0, vmax=vmax)
    a.set_title(title, fontsize=7)
    a.set_xticks([]); a.set_yticks([])


for j in range(NC):
    show(ax[0, j], img_of(tr_pmf[j]), "train #%d  ring %.2f" % (j, tr_ring[j]))
    show(ax[1, j], img_of(st["q_tr"][j]), "recon M=8  corr %.4f" % RES["configs"]["KL_M8"]["train"]["corrs"][j])
    show(ax[2, j], img_of(ho_pmf[j]), "held-out #%d  ring %.2f" % (j, ho_ring[j]))
    show(ax[3, j], img_of(st["q_ho"][j]), "project M=8  corr %.4f" % RES["configs"]["KL_M8"]["holdout"]["corrs"][j])
    show(ax[4, j], img_of(qs8[j]), "prior sample %d  ring %.2f" % (j, GEN[8][1][j]))
for j in range(3):
    show(ax[5, j], img_of(q_mid[j]), "mid %d-%d  ring %.2f\ncorr %.2f/%.2f"
         % (pairs[j][0], pairs[j][1], mid_rows[j]["ringiness"], mid_rows[j]["corr_a"], mid_rows[j]["corr_b"]))
show(ax[5, 3], img_of(mean_pmf), "mean train image")
show(ax[5, 4], img_of(STATES[("KL", 0)]["q_tr"][0]), "M=0 (shared theta0)")
show(ax[5, 5], img_of(qs8[6]), "prior sample 6  ring %.2f" % GEN[8][1][6])
fig.suptitle("Learned affine phase-space latent (theta = theta0 + Phi^T z), M=8, exact 10-qubit IQP",
             fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(os.path.join(OUT, "affine_latent.png"), dpi=110)
print("[fig] %s" % os.path.join(OUT, "affine_latent.png"), flush=True)
