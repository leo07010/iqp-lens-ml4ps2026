#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per-restart held-out projection correlation -- the reviewer-facing decomposition.

The paper reports the held-out projection correlation as the BEST over 8 restarts of the
z-only search, and restart 1 is initialised at the nearest TRAINING image's trained z
(train_iqp_affine_r3.py:321-347).  A reviewer will say that restart starts next to the
answer, and the run's own `projection_restart_hist` shows the objection has teeth: at
N = 100 restart 1 wins for 60 of the 200 held-out images, more than any other single start.
This script splits the number apart:

  restart 0 only      z initialised at ZERO -- no information from the training set at all
  restart 1 only      z initialised at the nearest training image's trained z
  best excluding 1    best over restarts {0, 2..R-1} -- the honest "no answer-adjacent
                      start" number
  best over all       what the paper currently reports, selected by CORR
  best by loss        the same 8 restarts, but the restart is chosen by the training
                      objective's LOSS instead of by the correlation to the target

The last one matters on its own.  Selecting the restart by correlation to the held-out image
is selection on the metric being reported: the search is told which of its 8 answers scored
best against the answer.  Selecting by loss is what a model without access to the target
could actually do, so the gap between the two is the size of that advantage.

It is a POST-HOC script, not a fork of the trainer.  It reloads a finished run's frozen
theta0 / Phi / z_train, rebuilds the restart initialisations from the recorded seed by the
trainer's own `build_restart_inits`, and re-runs the trainer's own `infer_z_batched`.  Two
consequences matter:

  * no existing script and no existing seed-0 output is modified, and
  * seed 0 needs no retraining -- its run directories are scored the same way as seeds 1-4,
    so the decomposition covers all five seeds.

Because the inference is deterministic given (frozen model, seed, n_holdout), "best over all"
must come back equal to the run's recorded `projection_corr_mean`.  The script checks that
and says so in its output; a mismatch means the replay is not the original computation and
the row is flagged rather than quietly reported.

The full (n_holdout, R) correlation matrix is saved next to the run as
`projection_restart_corrs.npy` -- a new sidecar file, so the run's own meta.json stays
byte-identical to what its training job wrote.

    python3 seeds/diag_projection_restarts.py --run-dir <dir> [<dir> ...] \
        --results results_seeds_restarts.jsonl

Rows are APPENDED, and they live in their own file rather than in
results_seeds_diag.jsonl so that this script and the section-3.2 diagnostics can
run in either order without one truncating the other's output.

Python 3.9 compatible.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

REPO = "<REPO_ROOT>/code"
sys.path.insert(0, REPO)

import numpy as np                                      # noqa: E402
import torch                                            # noqa: E402

import codec10                                          # noqa: E402
import data_lens_r3                                     # noqa: E402
import data_lens_class                                  # noqa: E402
import train_iqp_affine_r3 as TA                        # noqa: E402
from iqp_exact import IQPExact, fwht                    # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", nargs="+", required=True)
    p.add_argument("--results", default=os.path.join(REPO, "results_seeds_restarts.jsonl"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--infer-chunk", type=int, default=50)
    p.add_argument("--threads", type=int, default=4)
    return p.parse_args(argv)


def score_run(run_dir, device, chunk):
    with open(os.path.join(run_dir, "meta.json")) as fh:
        meta = json.load(fh)
    if meta.get("script") != "train_iqp_affine_r3.py":
        raise SystemExit("diag_projection_restarts: %s was written by %r, expected "
                         "train_iqp_affine_r3.py" % (run_dir, meta.get("script")))

    seed = int(meta["seed"])
    n_train = int(meta["n_train"])
    n_ho_want = int(meta["n_holdout"])
    M = int(meta["M"])
    R = max(int(meta["projection_restarts"]), 1)
    objective = meta["objective"]
    zpen = float(meta["zpen"])
    L0 = float(meta["zpen_L0"])
    steps = int(meta["projection_steps"])
    lr = float(meta["projection_lr"])
    cls = meta.get("class_name")

    # ---- data: the per-class runs are scored against their own class's held-out set
    if cls:
        data_lens_class.install(cls)
    d = data_lens_r3.load_r3(n_train, seed=seed, n_holdout=n_ho_want)
    tr_pmf = np.asarray(d.train_pmfs, dtype=np.float64)
    n_ho = int(d.holdout_crops.shape[0])
    ho_pmf = np.stack([codec10.image_to_pmf(d.holdout_crops[i]) for i in range(n_ho)], axis=0)

    # ---- the frozen model
    theta0 = torch.as_tensor(np.load(os.path.join(run_dir, "theta0.npy")).astype(np.float64),
                             dtype=TA.DT, device=device)
    # Phi.npy holds the NORMALISED Phi; normalise_phi rescales rows to a fixed norm and is
    # therefore idempotent, so passing it as phi_raw reproduces the training-time Phi exactly.
    phi = torch.as_tensor(np.load(os.path.join(run_dir, "Phi.npy")).astype(np.float64),
                          dtype=TA.DT, device=device)
    z_np = np.load(os.path.join(run_dir, "z_train.npy")).astype(np.float64)

    sim = IQPExact(TA.N_QUBITS, TA.K_MAX, device=device, dtype=torch.complex128)
    if objective == "mmdflat":
        weights = TA.flat_kernel_weights(sim.size).to(device=device, dtype=TA.DT)
    elif objective == "auto3":
        sig = meta.get("sigmas") or []
        if not sig:
            raise SystemExit("diag_projection_restarts: %s is objective=auto3 but its "
                             "meta.json records no sigmas" % run_dir)
        weights = sim.kernel_weights(list(sig)).to(device=device, dtype=TA.DT)
    else:
        weights = torch.ones(sim.size, dtype=TA.DT, device=device)   # KL path ignores them

    # ---- the restart initialisations, rebuilt by the trainer's own function
    z_prior = TA.fit_z_gaussian(z_np, "full" if n_train >= 100 else "diag")
    nearest_train = TA.nearest_train_index(ho_pmf, tr_pmf)
    rng_r = np.random.default_rng(seed + 4321)
    rand_all = (TA.sample_z_gaussian(z_prior, n_ho * (R - 2), rng_r).reshape(n_ho, R - 2, M)
                if R > 2 else None)

    c_all = np.zeros((n_ho, R), dtype=np.float64)
    l_all = np.zeros((n_ho, R), dtype=np.float64)
    for lo in range(0, n_ho, max(int(chunk), 1)):
        hi = min(lo + max(int(chunk), 1), n_ho)
        m = hi - lo
        blk = ho_pmf[lo:hi]
        z0 = TA.build_restart_inits(z_np, nearest_train[lo:hi], R,
                                    None if rand_all is None else rand_all[lo:hi])
        flat_z = torch.as_tensor(z0.reshape(m * R, M), dtype=TA.DT, device=device)
        rep = torch.as_tensor(np.repeat(blk, R, axis=0), dtype=TA.DT, device=device)
        best_z, best_row = TA.infer_z_batched(sim, theta0, phi, rep, fwht(rep), weights,
                                              objective, flat_z, steps, lr, zpen, L0)
        with torch.no_grad():
            q = TA.probs_batch(sim, TA.theta_of(theta0, phi, best_z)).cpu().numpy()
        c_all[lo:hi] = TA.corr_rows(q, np.repeat(blk, R, axis=0)).reshape(m, R)
        l_all[lo:hi] = best_row.detach().cpu().numpy().reshape(m, R)

    np.save(os.path.join(run_dir, "projection_restart_corrs.npy"), c_all)
    np.save(os.path.join(run_dir, "projection_restart_losses.npy"), l_all)

    others = [i for i in range(R) if i != 1]
    best_all = c_all.max(axis=1)
    best_excl1 = c_all[:, others].max(axis=1) if others else np.zeros(n_ho)
    rand_only = c_all[:, 2:].max(axis=1) if R > 2 else np.zeros(n_ho)
    # selection by the OBJECTIVE rather than by the reported metric: the restart a model with
    # no access to the target would have to pick.
    pick_loss = l_all.argmin(axis=1)
    by_loss = c_all[np.arange(n_ho), pick_loss]
    recorded = meta.get("projection_corr_mean")
    replay_ok = (recorded is not None and abs(float(best_all.mean()) - float(recorded)) < 1e-6)

    row = {
        "diag": "projection_restarts",
        "script": "seeds/diag_projection_restarts.py",
        "run_dir": os.path.abspath(run_dir),
        "arm": meta.get("arm"), "seed": seed, "n_train": n_train, "M": M,
        "objective": objective, "class_name": cls, "restarts": R, "n_holdout": n_ho,
        "proj_corr_restart0_zero_init": float(c_all[:, 0].mean()),
        "proj_corr_restart1_nearest_train_init": float(c_all[:, 1].mean()) if R > 1 else None,
        "proj_corr_best_excl_restart1": float(best_excl1.mean()) if others else None,
        "proj_corr_best_random_restarts": float(rand_only.mean()) if R > 2 else None,
        "proj_corr_best_all_restarts": float(best_all.mean()),
        "proj_corr_best_by_loss": float(by_loss.mean()),
        "proj_corr_selection_gap": float(best_all.mean() - by_loss.mean()),
        "loss_pick_equals_corr_pick_share": float((pick_loss == c_all.argmax(axis=1)).mean()),
        "proj_corr_recorded_in_meta": None if recorded is None else float(recorded),
        "replay_matches_recorded": bool(replay_ok),
        "restart1_win_share": float((c_all.argmax(axis=1) == 1).mean()) if R > 1 else None,
        "corrs_npy": os.path.join(os.path.abspath(run_dir), "projection_restart_corrs.npy"),
        "losses_npy": os.path.join(os.path.abspath(run_dir), "projection_restart_losses.npy"),
        "S": meta.get("S"),
    }
    print("[diag-c] %-46s seed=%d N=%-3d  r0 %.4f | r1 %.4f | best-excl-r1 %.4f | "
          "by-loss %.4f | best-all %.4f (meta %.4f, replay %s) | r1 wins %.0f%%"
          % (meta.get("arm"), seed, n_train,
             row["proj_corr_restart0_zero_init"],
             row["proj_corr_restart1_nearest_train_init"],
             row["proj_corr_best_excl_restart1"], row["proj_corr_best_by_loss"],
             row["proj_corr_best_all_restarts"],
             float(recorded if recorded is not None else float("nan")),
             "OK" if replay_ok else "**MISMATCH**",
             100.0 * (row["restart1_win_share"] or 0.0)), flush=True)
    return row


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(int(args.threads))
    t0 = time.time()
    rows = []
    for rd in args.run_dir:
        if not os.path.isfile(os.path.join(rd, "meta.json")):
            print("[diag-c] SKIP %s (no meta.json)" % rd, flush=True)
            continue
        rows.append(score_run(rd, args.device, args.infer_chunk))
    with open(args.results, "a") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    bad = [r["run_dir"] for r in rows if not r["replay_matches_recorded"]]
    print("[diag-c] appended %d rows to %s (%.1f s); replay mismatches: %d"
          % (len(rows), args.results, time.time() - t0, len(bad)), flush=True)
    for b in bad:
        print("[diag-c] MISMATCH %s" % b, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
