"""Mechanism test: can an ancilla qubit + ancilla-touching low-order generators act as a
smooth 'mixing angle' latent between two lens images, where the paper-style low-order latent
(which only interferes a template with its bit-flip copies) cannot?

Read-only w.r.t. the repo: imports iqp_exact / codec10 / data_lens_r3 / evaluate_r3 / train_iqp
but writes nothing outside this scratchpad directory.
"""
from __future__ import annotations

import itertools
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
DTYPE = torch.float64
RESULTS = {}


# ----------------------------------------------------------------------------- simulator
class IQPFull(object):
    """Complete-basis IQP on n qubits, parameterised DIRECTLY by subset bitmask.

    theta[m] is the coefficient of Z_s with mask m, so theta_full is the identity and the
    whole thing batches: theta of shape (B, 2^n) -> probs of shape (B, 2^n).
    Identical maths to iqp_exact.IQPExact(n, K=n); verified against it in _selftest().
    """

    def __init__(self, n):
        self.n = int(n)
        self.size = 1 << self.n

    def probs(self, theta):
        phi = fwht(theta)
        re = fwht(torch.cos(phi))
        im = fwht(torch.sin(phi))
        return (re * re + im * im) / float(self.size * self.size)


def _selftest():
    """theta in subsets_upto order (IQPExact) == theta in mask order (IQPFull) after scatter."""
    n = 6
    ref = iqp_exact.IQPExact(n, n, dtype=torch.complex128)
    g = torch.Generator().manual_seed(0)
    th = torch.randn(ref.n_params, generator=g, dtype=DTYPE)
    q_ref = ref.probs(th)
    masks = iqp_exact.subset_masks(n, n)
    th_mask = torch.zeros(1 << n, dtype=DTYPE)
    th_mask[torch.as_tensor(masks)] = th
    q_mine = IQPFull(n).probs(th_mask)
    err = float((q_ref - q_mine).abs().max())
    assert err < 1e-12, "IQPFull disagrees with IQPExact: %g" % err
    # batching
    q_b = IQPFull(n).probs(torch.stack([th_mask, th_mask * 0.5]))
    assert float((q_b[0] - q_mine).abs().max()) < 1e-14
    return err


# ----------------------------------------------------------------------------- metrics
def corr(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    d = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    return float((a * b).sum() / d) if d > 0 else 0.0


_RING_R0 = np.arange(2.0, 14.01, 0.5)
_RING_SIG = np.arange(0.5, 6.01, 0.5)


def ringiness(img):
    """max over (r0, sigma) of Pearson corr( image , Gaussian annulus exp(-(r-r0)^2/2sig^2) )
    with r measured from the image's own intensity centroid. 1.0 = a perfect circular ring,
    ~0 = speckle / no ring structure. Rotationally symmetric on purpose: it scores 'is the
    light on one narrow ring', not 'which ring'."""
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


def quants(pmf, S):
    img = codec10.pmf_to_image(np.asarray(pmf, dtype=np.float64), S)
    q = evaluate_r3.image_quantities(img)
    return img, q


# ----------------------------------------------------------------------------- objectives
def kl_from_ce(p_t, q_t):
    """KL(p||q); p rows are pmfs (possibly with exact zeros)."""
    ce = -(p_t * torch.log(q_t.clamp_min(1e-30))).sum(dim=-1)
    ent = -(p_t * torch.log(p_t.clamp_min(1e-30))).sum(dim=-1)
    return ce - ent


def fit(loss_fn, params, steps, lr, seed, log_every=0, tag=""):
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.05)
    best = float("inf")
    best_state = [p.detach().clone() for p in params]
    for it in range(steps):
        opt.zero_grad()
        L = loss_fn(params)
        L.backward()
        opt.step()
        sched.step()
        v = float(L.detach())
        if v < best:
            best = v
            best_state = [p.detach().clone() for p in params]
        if log_every and (it % log_every == 0 or it == steps - 1):
            print("    [%s] step %5d  loss %.6f" % (tag, it, v), flush=True)
    return best, [b.clone() for b in best_state]


# ============================================================================= data
def load_pair():
    d = data_lens_r3.load_r3(20, seed=0, n_holdout=0)
    qt = np.asarray([evaluate_r3.image_quantities(d.train_crops[i]) for i in range(20)])
    e1 = np.asarray([q["e1"] for q in qt])
    rad = np.asarray([q["radius"] for q in qt])
    e2 = np.asarray([q["e2"] for q in qt])
    ia, ib = max(itertools.combinations(range(20), 2), key=lambda ij: abs(e1[ij[0]] - e1[ij[1]]))
    return d, e1, e2, rad, int(ia), int(ib)


# ============================================================================= (a)
def part_a(d, idx):
    n = 10
    sim = IQPFull(n)
    p = np.asarray(d.train_pmfs[idx], dtype=np.float64)
    p_t = torch.as_tensor(p, dtype=DTYPE)
    S = float(d.train_totals[idx])
    _img_p, q_p = quants(p, S)

    rng = np.random.default_rng(0)
    codes = codec10.sample_codes_from_pmf(p, 4096, rng)
    sigmas, mpw = train_iqp.resolve_sigmas("auto3", n, codes, 0)
    ref = iqp_exact.IQPExact(n, n, device="cpu", dtype=torch.complex128)
    w_auto = ref.kernel_weights(sigmas).to(DTYPE)
    w_flat = torch.ones(sim.size, dtype=DTYPE) / float(sim.size)
    ez_data = fwht(p_t)

    objs = {}
    objs["KL"] = lambda th: kl_from_ce(p_t, sim.probs(th))
    objs["MMD-flat"] = lambda th: ((fwht(sim.probs(th)) - ez_data) ** 2 * w_flat).sum()
    objs["MMD-auto3"] = lambda th: ((fwht(sim.probs(th)) - ez_data) ** 2 * w_auto).sum()

    out = {"target": {"radius": q_p["radius"], "e1": q_p["e1"], "e2": q_p["e2"],
                      "ringiness": ringiness(_img_p)},
           "image_index": idx, "sigmas_auto3": sigmas, "mean_pauli_weights": mpw, "fits": {}}
    print("  (a) target radius %.3f e1 %+.4f e2 %+.4f ring %.3f" %
          (q_p["radius"], q_p["e1"], q_p["e2"], out["target"]["ringiness"]), flush=True)

    for name, fn in objs.items():
        best_v, best_th = float("inf"), None
        for r, lr in enumerate([0.03, 0.012]):
            g = torch.Generator().manual_seed(100 + r)
            th0 = (torch.randn(sim.size, generator=g, dtype=DTYPE) * 0.05).requires_grad_(True)
            v, st = fit(lambda ps: fn(ps[0]), [th0], 3000, lr, seed=r, tag="%s r%d" % (name, r))
            if v < best_v:
                best_v, best_th = v, st[0]
        with torch.no_grad():
            q = sim.probs(best_th).numpy()
        img, qq = quants(q, S)
        rec = {"objective_value": best_v, "corr": corr(q, p), "radius": qq["radius"],
               "e1": qq["e1"], "e2": qq["e2"], "ringiness": ringiness(img),
               "kl": float(kl_from_ce(p_t, torch.as_tensor(q, dtype=DTYPE)))}
        out["fits"][name] = rec
        print("  (a) %-10s corr %.4f  radius %.3f  e1 %+.4f  e2 %+.4f  ring %.3f  KL %.4f"
              % (name, rec["corr"], rec["radius"], rec["e1"], rec["e2"], rec["ringiness"],
                 rec["kl"]), flush=True)
    return out


# ============================================================================= b / b'' / b'
def latent_masks_ancilla(n_img, n_anc, max_img_bits=1, spec_23=False):
    """masks that touch >=1 ancilla bit and <= max_img_bits image bits."""
    anc_bits = list(range(n_img, n_img + n_anc))
    out = []
    for k in range(1, n_anc + 1):
        for anc in itertools.combinations(anc_bits, k):
            am = 0
            for b in anc:
                am |= (1 << b)
            out.append(am)
            if spec_23 and k > 1:
                continue
            for j in range(n_img):
                out.append(am | (1 << j))
    return sorted(set(out))


def latent_masks_lowvar(n_img):
    """paper-style low-order latent: {} plus the n_img singletons (image bits only)."""
    return [0] + [1 << j for j in range(n_img)]


def marginal(q, n_img):
    """sum out the high (ancilla) bits: index = a * 2^n_img + x."""
    return q.reshape(q.shape[:-1] + (-1, 1 << n_img)).sum(dim=-2)


def joint_fit_and_scan(name, n_img, n_anc, lat_masks, pmfs, S, steps=4000, lrs=(0.03, 0.012),
                       n_t=11, mix_points=None):
    n = n_img + n_anc
    sim = IQPFull(n)
    P = sim.size
    lat_idx = torch.as_tensor(np.asarray(lat_masks, dtype=np.int64))
    core_mask = np.ones(P, dtype=bool)
    core_mask[np.asarray(lat_masks, dtype=np.int64)] = False
    core_idx = torch.as_tensor(np.nonzero(core_mask)[0])
    B = len(pmfs)
    p_t = torch.as_tensor(np.stack(pmfs), dtype=DTYPE)

    def assemble(core, lam):
        base = torch.zeros(B, P, dtype=DTYPE)
        base = base.index_copy(1, core_idx, core.unsqueeze(0).expand(B, -1))
        base = base.index_copy(1, lat_idx, lam)
        return base

    def loss_fn(ps):
        core, lam = ps
        q = sim.probs(assemble(core, lam))
        if n_anc > 0:
            q = marginal(q, n_img)
        return kl_from_ce(p_t, q).sum()

    best_v, best_state = float("inf"), None
    for r, lr in enumerate(lrs):
        g = torch.Generator().manual_seed(200 + r)
        core0 = (torch.randn(core_idx.numel(), generator=g, dtype=DTYPE) * 0.05)
        lam0 = (torch.randn(B, lat_idx.numel(), generator=g, dtype=DTYPE) * 0.5)
        v, st = fit(loss_fn, [core0, lam0], steps, lr, seed=r,
                    log_every=steps // 2, tag="%s r%d" % (name, r))
        if v < best_v:
            best_v, best_state = v, st
    core, lam = best_state

    def eval_lam(l_vec):
        with torch.no_grad():
            base = torch.zeros(1, P, dtype=DTYPE)
            base = base.index_copy(1, core_idx, core.unsqueeze(0))
            base = base.index_copy(1, lat_idx, l_vec.unsqueeze(0))
            q = sim.probs(base)
            if n_anc > 0:
                q = marginal(q, n_img)
        return q[0].numpy()

    rows = []
    imgs = []
    if mix_points is None:
        mix_points = [(float(i) / (n_t - 1),) for i in range(n_t)]
    for mp in mix_points:
        wts = np.zeros(B, dtype=np.float64)
        if len(mp) == 1:
            t = mp[0]
            wts[0] = 1.0 - t
            wts[1] = t
            label = "t=%.2f" % t
        else:
            wts[:] = np.asarray(mp, dtype=np.float64)
            label = "w=" + ",".join("%.2f" % v for v in wts)
        l_vec = (torch.as_tensor(wts, dtype=DTYPE).unsqueeze(1) * lam).sum(dim=0)
        q = eval_lam(l_vec)
        img, qq = quants(q, S)
        imgs.append(img)
        blend = (wts[:, None] * np.stack(pmfs)).sum(axis=0)
        row = {"label": label, "weights": wts.tolist(),
               "corr_to": [corr(q, pmfs[k]) for k in range(B)],
               "corr_blend": corr(q, blend),
               "radius": qq["radius"], "e1": qq["e1"], "e2": qq["e2"],
               "ringiness": ringiness(img)}
        rows.append(row)
    return {"name": name, "n": n, "n_img": n_img, "n_anc": n_anc,
            "n_latent": int(lat_idx.numel()), "n_core": int(core_idx.numel()),
            "n_params_total": P, "final_loss_sum_KL": best_v, "rows": rows}, imgs


# ============================================================================= main
def main():
    t0 = time.time()
    err = _selftest()
    print("selftest: IQPFull vs IQPExact max|dq| = %.2e" % err, flush=True)

    d, e1, e2, rad, ia, ib = load_pair()
    S = float(0.5 * (d.train_totals[ia] + d.train_totals[ib]))
    print("pair A=%d B=%d  e1 %+.4f / %+.4f (|d|=%.4f)  radius %.3f / %.3f (d=%.3f)  e2 %+.4f / %+.4f"
          % (ia, ib, e1[ia], e1[ib], abs(e1[ia] - e1[ib]), rad[ia], rad[ib],
             rad[ia] - rad[ib], e2[ia], e2[ib]), flush=True)
    RESULTS["pair"] = {"A": ia, "B": ib, "e1": [e1[ia], e1[ib]], "e2": [e2[ia], e2[ib]],
                       "radius": [rad[ia], rad[ib]], "S": S,
                       "ring": [ringiness(d.train_crops[ia]), ringiness(d.train_crops[ib])]}

    print("\n=== (a) single-image ceiling, n=10 ===", flush=True)
    RESULTS["a"] = part_a(d, ia)
    print("  (a) done, %.1f s" % (time.time() - t0), flush=True)

    pA = np.asarray(d.train_pmfs[ia], dtype=np.float64)
    pB = np.asarray(d.train_pmfs[ib], dtype=np.float64)

    print("\n=== (b) ancilla mixing-angle latent, n=11 ===", flush=True)
    lat_b = latent_masks_ancilla(10, 1)
    print("  latent masks (%d): %s" % (len(lat_b), lat_b), flush=True)
    res_b, imgs_b = joint_fit_and_scan("b", 10, 1, lat_b, [pA, pB], S, steps=4000)
    RESULTS["b"] = res_b

    print("\n=== (b'') param-matched control: n=11 ancilla present, latent = image-only low order ===",
          flush=True)
    lat_b2 = latent_masks_lowvar(10)
    res_b2, imgs_b2 = joint_fit_and_scan("b2", 10, 1, lat_b2, [pA, pB], S, steps=4000)
    RESULTS["b2"] = res_b2

    print("\n=== (b') no ancilla, n=10, paper-style low-order latent ===", flush=True)
    res_bp, imgs_bp = joint_fit_and_scan("bp", 10, 0, lat_b2, [pA, pB], S, steps=4000)
    RESULTS["bp"] = res_bp

    for tag in ("b", "b2", "bp"):
        r = RESULTS[tag]
        print("\n  [%s] n=%d latent=%d core=%d  joint KL sum %.4f" %
              (tag, r["n"], r["n_latent"], r["n_core"], r["final_loss_sum_KL"]), flush=True)
        print("   t      corrA   corrB   corr_blend  radius   e1       e2      ring", flush=True)
        for row in r["rows"]:
            print("   %-6s %+.4f %+.4f  %+.4f     %6.3f  %+.4f %+.4f %+.4f" %
                  (row["label"], row["corr_to"][0], row["corr_to"][1], row["corr_blend"],
                   row["radius"], row["e1"], row["e2"], row["ringiness"]), flush=True)

    def verdict(r):
        rows = r["rows"]
        cA = [x["corr_to"][0] for x in rows]
        cB = [x["corr_to"][1] for x in rows]
        rad_ = [x["radius"] for x in rows]
        e1_ = [x["e1"] for x in rows]
        mono = lambda v, sign: all(sign * (v[i + 1] - v[i]) >= -1e-9 for i in range(len(v) - 1))
        floor = min(max(a, b) for a, b in zip(cA, cB))
        target_r = (RESULTS["pair"]["radius"][0], RESULTS["pair"]["radius"][1])
        target_e1 = (RESULTS["pair"]["e1"][0], RESULTS["pair"]["e1"][1])
        return {"corrA_monotone_down": mono(cA, -1), "corrB_monotone_up": mono(cB, +1),
                "radius_monotone": mono(rad_, 1 if target_r[1] > target_r[0] else -1),
                "e1_monotone": mono(e1_, 1 if target_e1[1] > target_e1[0] else -1),
                "min_max_corr": floor, "no_dropout_below_0.8": floor >= 0.8,
                "endpoint_corrA": cA[0], "endpoint_corrB": cB[-1],
                "mean_corr_blend": float(np.mean([x["corr_blend"] for x in rows])),
                "min_ringiness": float(min(x["ringiness"] for x in rows))}

    for tag in ("b", "b2", "bp"):
        RESULTS[tag]["verdict"] = verdict(RESULTS[tag])
        print("  [%s] verdict: %s" % (tag, json.dumps(RESULTS[tag]["verdict"])), flush=True)

    vb = RESULTS["b"]["verdict"]
    b_ok = (vb["corrA_monotone_down"] and vb["corrB_monotone_up"] and vb["no_dropout_below_0.8"]
            and vb["radius_monotone"] and vb["e1_monotone"])
    print("\n(b) success criteria passed: %s" % b_ok, flush=True)

    imgs_c = None
    if b_ok:
        print("\n=== (c) k=2 ancillas, n=12, three images ===", flush=True)
        # three images with spread-out e1 AND radius
        order = np.argsort(e1)
        cand = [int(order[0]), int(order[len(order) // 2]), int(order[-1])]
        ic = max(range(20), key=lambda k: (abs(rad[k] - rad[cand[0]]) + abs(rad[k] - rad[cand[2]]))
                 if k not in (cand[0], cand[2]) else -1)
        trio = [cand[0], ic, cand[2]]
        print("  trio %s  e1 %s  radius %s" % (trio, [round(e1[k], 4) for k in trio],
                                               [round(rad[k], 3) for k in trio]), flush=True)
        RESULTS["c_trio"] = {"idx": trio, "e1": [float(e1[k]) for k in trio],
                             "radius": [float(rad[k]) for k in trio]}
        pmfs3 = [np.asarray(d.train_pmfs[k], dtype=np.float64) for k in trio]
        S3 = float(np.mean([d.train_totals[k] for k in trio]))
        lat_c = latent_masks_ancilla(10, 2)
        print("  latent masks (%d)" % len(lat_c), flush=True)
        mix = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (.5, .5, 0), (0, .5, .5), (.5, 0, .5),
               (1 / 3., 1 / 3., 1 / 3.)]
        res_c, imgs_c = joint_fit_and_scan("c", 10, 2, lat_c, pmfs3, S3, steps=5000,
                                           mix_points=mix)
        RESULTS["c"] = res_c
        print("   mix                corrA   corrB   corrC   corr_blend radius   e1      ring",
              flush=True)
        for row in res_c["rows"]:
            print("   %-18s %+.4f %+.4f %+.4f  %+.4f    %6.3f  %+.4f %+.4f" %
                  (row["label"], row["corr_to"][0], row["corr_to"][1], row["corr_to"][2],
                   row["corr_blend"], row["radius"], row["e1"], row["ringiness"]), flush=True)
    else:
        RESULTS["c"] = {"skipped": "part (b) did not pass its success criteria"}

    # -------------------------------------------------------------------- figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows_fig = [("(b) ancilla latent n=11", res_b, imgs_b),
                ("(b'') n=11, image-only latent", res_b2, imgs_b2),
                ("(b') no ancilla n=10", res_bp, imgs_bp)]
    if imgs_c is not None:
        rows_fig.append(("(c) 2 ancillas n=12", RESULTS["c"], imgs_c))
    ncol = 11
    vmax = max(float(np.max(im)) for _, _, ims in rows_fig for im in ims)
    vmax = max(vmax, float(np.max(codec10.pmf_to_image(pA, S))))
    fig, axes = plt.subplots(len(rows_fig), ncol, figsize=(1.5 * ncol, 1.85 * len(rows_fig)))
    axes = np.atleast_2d(axes)
    for ri, (title, res, ims) in enumerate(rows_fig):
        for ci in range(ncol):
            ax = axes[ri, ci]
            ax.set_xticks([])
            ax.set_yticks([])
            if ci >= len(ims):
                ax.axis("off")
                continue
            ax.imshow(ims[ci], cmap="inferno", vmin=0.0, vmax=vmax, origin="lower")
            r = res["rows"][ci]
            if len(r["corr_to"]) == 2:
                ax.set_title("%s\nA %.3f B %.3f" % (r["label"], r["corr_to"][0], r["corr_to"][1]),
                             fontsize=6.5)
            else:
                ax.set_title("%s\n%.2f/%.2f/%.2f" % (r["label"], r["corr_to"][0],
                                                     r["corr_to"][1], r["corr_to"][2]),
                             fontsize=6.0)
            if ci == 0:
                ax.set_ylabel(title, fontsize=7)
    fig.suptitle("Latent interpolation A->B: ancilla mixing-angle vs paper-style low-order latent "
                 "(same colour scale)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig_path = os.path.join(OUT, "interp.png")
    fig.savefig(fig_path, dpi=130)
    plt.close(fig)
    RESULTS["figure"] = fig_path
    RESULTS["runtime_s"] = time.time() - t0
    RESULTS["b_success"] = bool(b_ok)

    with open(os.path.join(OUT, "results.json"), "w") as fh:
        json.dump(RESULTS, fh, indent=2, default=float)
    print("\nwrote %s  (%.1f s total)" % (os.path.join(OUT, "results.json"),
                                          RESULTS["runtime_s"]), flush=True)


if __name__ == "__main__":
    main()
