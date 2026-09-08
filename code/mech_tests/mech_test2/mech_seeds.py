#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Seed replication of the M sweep behind Fig. 2(a): held-out projection correlation vs the
number of learned latent directions M, for M in {0, 2, 4, 6, 8, 16}.

The seed-0 sweep is spread over three scripts, each with its own held-out projection protocol.
This file reproduces ALL THREE protocols in one process so that a seed is a single number, and
it applies each protocol to exactly the M values it produced at seed 0 (fig2_mechanism_scaling.
M_SWEEP is the authority on which file each point came from):

    M = 0, 4    affine_latent.py  run_config(M, "KL", steps): joint Adam on
                {theta0, Phi, z}, steps = 2000 (M=0) / 4000, lr 0.02, z-penalty 1e-3;
                held-out projection = z-only Adam, 500 steps, lr 0.02, z init 0.
                -> results.json configs.KL_M{0,4}.holdout.corr_mean
    M = 2, 6    diag3.py joint(M, "KL"): same joint fit at 4000 steps but starting from the
                TRAINED M = 0 theta0 (diag3 read it from states.pt; here it is the M = 0 fit of
                the same seed, which is what states.pt held);
                held-out projection = z-only Adam, 1000 steps, lr in {0.02, 0.005}, the arm
                with the lower held-out KL wins.  -> diag3.json KL_M{2,6}.holdout.corr_mean
    M = 8, 16   affine_latent.py joint fit (4000 steps) + diag.py's "exp2_redo" projection:
                z-only Adam, 1000 steps, lr in {0.02, 0.005, 0.001} x z init in {0, zbar},
                the arm with the HIGHEST held-out corr wins (the choice most favourable to the
                model, which is what makes the collapse claim conservative).
                -> diag.json exp2_redo.M{8,16}.best.corr_mean

Reference levels of the panel:
    nearest training image / mean training image -- functions of the FIXED data split only
        (data_lens_r3 pins its split at R3_SPLIT_SEED and decouples it from any arm seed), so
        they are constants, recomputed here per seed only to prove they do not move.
    random direction -- diag2.py: the trained M = 4 theta0 with an UNTRAINED Phi drawn at
        torch seeds 99 and 123, projected the diag2 way (1000 steps, lr in {0.02, 0.005},
        lower held-out KL wins), averaged over the two draws.  The two Phi draws are part of
        the fixed protocol; what moves with the seed is the trained theta0 they sit on.

`--seed 0` must reproduce the published numbers exactly; the aggregator checks that.
Read-only w.r.t. every existing artefact: output goes to mech_seeds_s<seed>.json.
CPU only, ~4 min per seed.  Python 3.9 compatible.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
AFF = os.path.join(HERE, "affine_latent.py")

# affine_latent.py trains at module level, so it cannot be imported.  Execute only the part
# above its "data" banner -- the definitions -- exactly as diag2.py and diag3.py do.
_src = open(AFF).read()
_m = re.search(r"^# -+ data$", _src, flags=re.M)
if _m is None:
    raise RuntimeError("could not locate the 'data' banner in %s" % AFF)
exec(compile(_src[:_m.start()], AFF, "exec"), globals())      # noqa: S102

# --smoke divides every optimiser budget by this factor (an end-to-end shape check only;
# the numbers it produces are meaningless and it never writes to the default output path).
STEPS_DIV = [1]


def _ns(n):
    """optimiser steps, honouring --smoke"""
    return max(5, int(n) // STEPS_DIV[0])


# names now in scope from affine_latent.py's definition block:
#   np, torch, math, time (re-imported below), codec10, data_lens_r3, evaluate_r3, iqp_exact,
#   train_iqp, fwht, DT, NQ, D, SCALE, probs, corr, ringiness, img_of, quants, kl_rows,
#   theta_of, make_loss, run_adam


# =========================================================================== data (fixed)
def build_data():
    d = data_lens_r3.load_r3(40, seed=0, n_holdout=20)
    tr_pmf = np.asarray(d.train_pmfs[:20], dtype=np.float64)
    ho_pmf = np.stack([codec10.image_to_pmf(d.holdout_crops[i]) for i in range(20)], axis=0)
    mean_pmf = tr_pmf.mean(axis=0)
    mean_pmf = mean_pmf / mean_pmf.sum()
    codes = codec10.sample_codes_from_pmf(mean_pmf, 4096, np.random.default_rng(0))
    sigmas, _mpw = train_iqp.resolve_sigmas("auto3", NQ, codes, 0)
    w_auto = iqp_exact.IQPExact(NQ, NQ, dtype=torch.complex128).kernel_weights(sigmas).to(DT)
    return {"tr_pmf": tr_pmf, "ho_pmf": ho_pmf, "mean_pmf": mean_pmf,
            "tr_t": torch.as_tensor(tr_pmf, dtype=DT), "ho_t": torch.as_tensor(ho_pmf, dtype=DT),
            "mean_t": torch.as_tensor(mean_pmf, dtype=DT).unsqueeze(0),
            "w_auto": w_auto, "sigmas": [float(s) for s in sigmas],
            "NTR": tr_pmf.shape[0], "NHO": ho_pmf.shape[0],
            "tr_q": np.asarray([quants(tr_pmf[i]) for i in range(tr_pmf.shape[0])]),
            "ho_q": np.asarray([quants(ho_pmf[i]) for i in range(ho_pmf.shape[0])])}


def eval_rows(qs, pmfs, refq):
    cs = np.asarray([corr(img_of(qs[i]), img_of(pmfs[i])) for i in range(len(pmfs))])
    qq = np.asarray([quants(qs[i]) for i in range(len(pmfs))])
    return {"corr_mean": float(cs.mean()), "corr_min": float(cs.min()),
            "corr_med": float(np.median(cs)),
            "rmse_radius": float(np.sqrt(((qq[:, 0] - refq[:, 0]) ** 2).mean())),
            "rmse_e1": float(np.sqrt(((qq[:, 1] - refq[:, 1]) ** 2).mean())),
            "corrs": cs.tolist()}


# ============================================================ protocol A: affine_latent.py
def theta0_seed(DATA, seed, log):
    """affine_latent.py's TH0_SEED: a single shared theta0 fitted to the mean training image."""
    torch.manual_seed(int(seed))
    th0 = 0.01 * torch.randn(D, dtype=DT)
    loss_mean = make_loss("KL", DATA["mean_t"], DATA["w_auto"])
    bv, bs = run_adam([th0], lambda: loss_mean(probs(th0.unsqueeze(0))), _ns(1000), 0.05)
    q = probs(bs[0].unsqueeze(0)).detach().numpy()[0]
    c = corr(img_of(q), img_of(DATA["mean_pmf"]))
    log("[init] mean-image single-shot fit: KL %.4f corr %.4f" % (bv, c))
    return bs[0], {"kl": float(bv), "corr": float(c)}


def run_config(DATA, TH0, M, kind, steps, seed, lr=0.02, zpen=1e-3, ho_steps=500):
    """Verbatim affine_latent.run_config, with theta0 seed and TH0_SEED passed in."""
    t0 = time.time()
    NTR, NHO = DATA["NTR"], DATA["NHO"]
    torch.manual_seed(int(seed))
    theta0 = TH0.clone()
    loss_tr = make_loss(kind, DATA["tr_t"], DATA["w_auto"])
    if M == 0:
        params = [theta0]
        clo = lambda: loss_tr(probs(theta0.unsqueeze(0).expand(NTR, D)))         # noqa: E731
    else:
        Phi_raw = torch.randn(M, D, dtype=DT)
        z = 0.001 * torch.randn(NTR, M, dtype=DT)
        params = [theta0, Phi_raw, z]
        clo = lambda: (loss_tr(probs(theta_of(theta0, Phi_raw, z)))              # noqa: E731
                       + zpen * (z ** 2).sum(dim=1).mean())
    bv, bs = run_adam(params, clo, _ns(steps), lr)
    theta0 = bs[0]
    Phi_raw = bs[1] if M > 0 else None
    z = bs[2] if M > 0 else None
    with torch.no_grad():
        th = theta_of(theta0, Phi_raw, z) if M > 0 else theta0.unsqueeze(0).expand(NTR, D)
        q_tr = probs(th).numpy()
        kl_tr = float(kl_rows(DATA["tr_t"], torch.as_tensor(q_tr)).mean())
    rec = {"M": M, "objective": kind, "steps": steps, "loss": float(bv), "kl_train": kl_tr,
           "train": eval_rows(q_tr, DATA["tr_pmf"], DATA["tr_q"])}
    if M > 0:
        theta0_f, Phi_f = theta0.detach().clone(), Phi_raw.detach().clone()
        z_ho = torch.zeros(NHO, M, dtype=DT)
        loss_ho = make_loss(kind, DATA["ho_t"], DATA["w_auto"])
        clo2 = lambda: (loss_ho(probs(theta_of(theta0_f, Phi_f, z_ho)))          # noqa: E731
                        + zpen * (z_ho ** 2).sum(dim=1).mean())
        _bv2, bs2 = run_adam([z_ho], clo2, _ns(ho_steps), lr)
        z_ho = bs2[0]
        with torch.no_grad():
            q_ho = probs(theta_of(theta0_f, Phi_f, z_ho)).numpy()
    else:
        with torch.no_grad():
            q_ho = probs(theta0.unsqueeze(0).expand(NHO, D)).numpy()
        z_ho = None
    rec["kl_holdout"] = float(kl_rows(DATA["ho_t"], torch.as_tensor(q_ho)).mean())
    rec["holdout"] = eval_rows(q_ho, DATA["ho_pmf"], DATA["ho_q"])
    rec["secs"] = time.time() - t0
    state = {"theta0": theta0.detach(), "Phi_raw": (Phi_raw.detach() if M > 0 else None),
             "z": (z.detach() if M > 0 else None), "q_tr": q_tr, "q_ho": q_ho}
    return rec, state


# ==================================================================== protocol B: diag3.py
def joint_diag3(DATA, TH0_trained, M, kind, seed, steps=4000, lr=0.02):
    """Verbatim diag3.joint: start from the TRAINED M=0 theta0; held-out lr sweep on KL."""
    t0 = time.time()
    NTR, NHO = DATA["NTR"], DATA["NHO"]
    torch.manual_seed(int(seed))
    th0 = TH0_trained.clone()
    Phi = torch.randn(M, D, dtype=DT)
    z = 0.001 * torch.randn(NTR, M, dtype=DT)
    lt = make_loss(kind, DATA["tr_t"], DATA["w_auto"])
    _b, bs = run_adam([th0, Phi, z], lambda: lt(probs(theta_of(th0, Phi, z)))
                      + 1e-3 * (z ** 2).sum(1).mean(), _ns(steps), lr)
    th0, Phi, z = bs
    with torch.no_grad():
        q_tr = probs(theta_of(th0, Phi, z)).numpy()
    lh = make_loss(kind, DATA["ho_t"], DATA["w_auto"])
    best = None
    for lrp in (0.02, 0.005):
        zz = torch.zeros(NHO, M, dtype=DT)
        _b2, bs2 = run_adam([zz], lambda: lh(probs(theta_of(th0, Phi, zz)))
                            + 1e-3 * (zz ** 2).sum(1).mean(), _ns(1000), lrp)
        with torch.no_grad():
            qh = probs(theta_of(th0, Phi, bs2[0])).numpy()
        k = float(kl_rows(DATA["ho_t"], torch.as_tensor(qh)).mean())
        if best is None or k < best[0]:
            best = (k, qh, lrp)
    q_ho = best[1]
    rec = {"M": M, "objective": kind, "steps": steps,
           "train": eval_rows(q_tr, DATA["tr_pmf"], DATA["tr_q"]),
           "holdout": eval_rows(q_ho, DATA["ho_pmf"], DATA["ho_q"]),
           "kl_holdout": best[0], "holdout_lr": best[2],
           "znorm": float(z.norm(dim=1).mean()), "secs": time.time() - t0}
    rec["kl_train"] = float(kl_rows(DATA["tr_t"], torch.as_tensor(q_tr)).mean())
    return rec


# ===================================================================== protocol C: diag.py
def project_sweep(DATA, state, M):
    """Verbatim diag.py exp2_redo: frozen (theta0, Phi), z-only fit, lr x init sweep,
    the arm with the highest held-out corr wins."""
    NHO = DATA["NHO"]
    th0, Phi = state["theta0"], state["Phi_raw"]
    zmu = state["z"].mean(dim=0, keepdim=True)
    loss_ho = make_loss("KL", DATA["ho_t"], DATA["w_auto"])
    rows, best_q = {}, {}
    for lr in (0.02, 0.005, 0.001):
        for init_name, z0 in (("zero", torch.zeros(NHO, M, dtype=DT)),
                              ("zmean", zmu.repeat(NHO, 1))):
            z = z0.clone()
            clo = lambda: (loss_ho(probs(theta_of(th0, Phi, z)))                 # noqa: E731
                           + 1e-3 * (z ** 2).sum(dim=1).mean())
            _bv, bs = run_adam([z], clo, _ns(1000), lr)
            with torch.no_grad():
                q = probs(theta_of(th0, Phi, bs[0])).numpy()
            cs = np.asarray([corr(img_of(q[i]), img_of(DATA["ho_pmf"][i])) for i in range(NHO)])
            key = "lr%.3f_%s" % (lr, init_name)
            rows[key] = {"corr_mean": float(cs.mean()), "corr_min": float(cs.min()),
                         "kl": float(kl_rows(DATA["ho_t"], torch.as_tensor(q)).mean()),
                         "znorm_mean": float(bs[0].norm(dim=1).mean())}
            best_q[key] = q
    bk = max(rows, key=lambda k: rows[k]["corr_mean"])
    return {"sweep": rows, "best_cfg": bk, "best": rows[bk]}


# ================================================================== protocol D: diag2.py
def _project_random_phi(DATA, th0, Phi, M):
    """diag2.py's projection: z-only, 1000 steps, lr in {0.02, 0.005}, lower held-out KL wins."""
    NHO = DATA["NHO"]
    loss_ho = make_loss("KL", DATA["ho_t"], DATA["w_auto"])
    best = None
    for lr in (0.02, 0.005):
        z = torch.zeros(NHO, M, dtype=DT)
        clo = lambda: (loss_ho(probs(theta_of(th0, Phi, z)))                     # noqa: E731
                       + 1e-3 * (z ** 2).sum(dim=1).mean())
        _bv, bs = run_adam([z], clo, _ns(1000), lr)
        with torch.no_grad():
            qq = probs(theta_of(th0, Phi, bs[0])).numpy()
        kl = float(kl_rows(DATA["ho_t"], torch.as_tensor(qq)).mean())
        if best is None or kl < best[0]:
            best = (kl, qq)
    q = best[1]
    cs = np.asarray([corr(img_of(q[i]), img_of(DATA["ho_pmf"][i])) for i in range(NHO)])
    return {"corr_mean": float(cs.mean()), "corr_min": float(cs.min()), "kl": best[0]}


def random_direction_control(DATA, th0_trained_M4, th0_M0, phi_seeds=(99, 123), M=4):
    """diag2.py's falsification control, in BOTH of its theta0 variants.

    The two answer different questions and diag2.json reports both:
      * "+ trained theta0"  puts an untrained Phi on the theta0 that the M = 4 JOINT fit
        produced.  That theta0 was optimised together with a trained Phi, so on its own it is
        not a fair baseline offset -- it is a mismatched pairing, and it scores 0.672.
      * "+ M=0 theta0"      puts the untrained Phi on the theta0 that was fitted WITHOUT any
        latent (the M = 0 arm).  Direction and offset are then both untrained-with-each-other
        in the same sense, which is the correctly paired control; it scores 0.746 at seed 0.
    diag2.py computes the paired variant for Phi seed 99 only; both draws are computed here so
    the reference has the same two-draw average as the mismatched one.
    """
    out = {}
    for s in phi_seeds:
        torch.manual_seed(int(s))
        Phi = torch.randn(M, D, dtype=DT)
        out["random Phi(seed %d) + trained theta0" % s] = _project_random_phi(
            DATA, th0_trained_M4, Phi, M)
        out["random Phi(seed %d) + M=0 theta0" % s] = _project_random_phi(DATA, th0_M0, Phi, M)
    out["mean_over_draws"] = float(np.mean(
        [out["random Phi(seed %d) + trained theta0" % s]["corr_mean"] for s in phi_seeds]))
    out["mean_over_draws_M0theta0"] = float(np.mean(
        [out["random Phi(seed %d) + M=0 theta0" % s]["corr_mean"] for s in phi_seeds]))
    return out


# =========================================================================================
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--smoke", action="store_true",
                   help="shape check only: every optimiser budget divided by 100, output "
                        "forced to a throwaway path")
    a = p.parse_args(argv)
    if a.smoke:
        STEPS_DIV[0] = 100
    torch.set_num_threads(int(a.threads))
    out_path = (a.out or os.path.join(HERE, "mech_seeds_s%d.json" % a.seed))
    if a.smoke:
        out_path = os.path.join("/tmp", "mech_smoke_s%d_%d.json" % (a.seed, os.getpid()))

    t_all = time.time()
    log = lambda s: print("[s%d] %s" % (a.seed, s), flush=True)                   # noqa: E731
    DATA = build_data()
    log("data: train %d, holdout %d, sigmas=%s"
        % (DATA["NTR"], DATA["NHO"], ["%.3f" % s for s in DATA["sigmas"]]))

    TH0, th0_info = theta0_seed(DATA, a.seed, log)

    sweep, configs, states = {}, {}, {}
    # ---- M = 0, 4 : affine_latent protocol
    for M, steps in ((0, 2000), (4, 4000)):
        rec, st = run_config(DATA, TH0, M, "KL", steps, a.seed)
        configs["KL_M%d" % M] = rec
        states[M] = st
        sweep[M] = rec["holdout"]["corr_mean"]
        log("M=%2d [affine_latent]  train %.4f | holdout %.4f  (%.0fs)"
            % (M, rec["train"]["corr_mean"], rec["holdout"]["corr_mean"], rec["secs"]))
    # ---- M = 8, 16 : affine_latent joint fit + diag.py projection sweep
    for M in (8, 16):
        rec, st = run_config(DATA, TH0, M, "KL", 4000, a.seed)
        configs["KL_M%d" % M] = rec
        states[M] = st
        redo = project_sweep(DATA, st, M)
        configs["KL_M%d" % M]["exp2_redo"] = redo
        sweep[M] = redo["best"]["corr_mean"]
        log("M=%2d [diag redo]      train %.4f | holdout(best %s) %.4f | plain-500 %.4f"
            % (M, rec["train"]["corr_mean"], redo["best_cfg"], redo["best"]["corr_mean"],
               rec["holdout"]["corr_mean"]))
    # ---- M = 2, 6 : diag3 protocol, starting from this seed's trained M = 0 theta0
    for M in (2, 6):
        rec = joint_diag3(DATA, states[0]["theta0"], M, "KL", a.seed)
        configs["KL_M%d_diag3" % M] = rec
        sweep[M] = rec["holdout"]["corr_mean"]
        log("M=%2d [diag3]          train %.4f | holdout %.4f (lr %.3f)  (%.0fs)"
            % (M, rec["train"]["corr_mean"], rec["holdout"]["corr_mean"], rec["holdout_lr"],
               rec["secs"]))

    # ---- reference levels
    tr_pmf, ho_pmf, mean_pmf = DATA["tr_pmf"], DATA["ho_pmf"], DATA["mean_pmf"]
    NTR, NHO = DATA["NTR"], DATA["NHO"]
    copy_ub = np.asarray([max(corr(img_of(ho_pmf[i]), img_of(tr_pmf[j])) for j in range(NTR))
                          for i in range(NHO)])
    mean_ho = np.asarray([corr(img_of(ho_pmf[i]), img_of(mean_pmf)) for i in range(NHO)])
    mean_tr = np.asarray([corr(img_of(tr_pmf[i]), img_of(mean_pmf)) for i in range(NTR)])
    controls = {"holdout_vs_nearest_train_corr": {"mean": float(copy_ub.mean()),
                                                  "min": float(copy_ub.min())},
                "holdout_vs_mean_image_corr": {"mean": float(mean_ho.mean()),
                                               "min": float(mean_ho.min())},
                "train_vs_mean_image_corr": {"mean": float(mean_tr.mean()),
                                             "min": float(mean_tr.min())}}
    rand = random_direction_control(DATA, states[4]["theta0"], states[0]["theta0"])
    log("refs: nearest %.4f | mean-image %.4f | random-dir (paired, M=0 theta0) %.4f | "
        "random-dir (mismatched, trained theta0) %.4f"
        % (controls["holdout_vs_nearest_train_corr"]["mean"],
           controls["holdout_vs_mean_image_corr"]["mean"],
           rand["mean_over_draws_M0theta0"], rand["mean_over_draws"]))

    out = {"seed": int(a.seed),
           "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
           "script": os.path.abspath(__file__),
           "wallclock_s": time.time() - t_all,
           "theta0_seed": th0_info,
           "sigmas_auto3": DATA["sigmas"],
           "M_sweep_holdout_corr": {str(k): float(v) for k, v in sorted(sweep.items())},
           "M_sweep_protocol": {"0": "affine_latent.run_config", "4": "affine_latent.run_config",
                                "2": "diag3.joint", "6": "diag3.joint",
                                "8": "diag.exp2_redo", "16": "diag.exp2_redo"},
           "configs": configs, "controls": controls, "holdout_random_vs_trained": rand}
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    log("M sweep " + "  ".join("M=%s:%.4f" % (k, v)
                               for k, v in sorted(sweep.items(), key=lambda x: x[0])))
    log("wrote %s (%.0fs)" % (out_path, time.time() - t_all))
    return 0


if __name__ == "__main__":
    sys.exit(main())
