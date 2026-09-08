#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Seed replicate of the SINGLE-CROP full-parameter fit of paper section 3.2.

The paper says the full 1,024-phase circuit fits ONE aligned crop to correlation 0.99997
under the flat MMD and 0.9962 under the paper kernel.  Those two numbers come from
`mech_tests/mech_test/mech_test.py`, part (a), at a hard-coded seed.  This script re-runs
EXACTLY that computation with a `--seed` flag so the claim can be quoted as mean +- sd over
five seeds.

It does not fork mech_test.py: it imports it and calls its own IQPFull / fit / corr /
quants / ringiness, so the arithmetic is bit-for-bit part (a)'s.  The only things `--seed`
moves are

  * the restart initialisation, torch.Generator().manual_seed(100 + r + 1000 * seed)
    -- at seed 0 this is manual_seed(100 + r), i.e. part (a) unchanged, and the script
    reproduces mech_tests/mech_test/results.json's `a.fits` to the last digit; and
  * the Monte-Carlo code draw and sigma resolution behind the auto3 ("paper") kernel,
    np.random.default_rng(seed) / resolve_sigmas(..., seed) -- the same convention every
    training arm uses (train_iqp_r3.py:432 feeds it args.seed).

The DATA does not move: data_lens_r3.load_r3 fixes its split at R3_SPLIT_SEED regardless of
any arm's seed (data_lens_r3.py:265-266), and the crop that is fitted is chosen by the same
deterministic max-|de1| pair rule as part (a).

    python3 seeds/diag_single_crop_seeds.py --seed 3 --results results_seeds_diag.jsonl

Rows are appended, one per (seed, objective).  Python 3.9 compatible.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

REPO = "<HOME>/gsoc2026-deeplense-qml/experiments/13_iqp_turbulence_replication"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "mech_tests", "mech_test"))

import numpy as np                                      # noqa: E402
import torch                                            # noqa: E402

import codec10                                          # noqa: E402
import iqp_exact                                        # noqa: E402
import train_iqp                                        # noqa: E402
from iqp_exact import fwht                              # noqa: E402

import mech_test as MT                                  # noqa: E402

N_QUBITS = 10
STEPS = 3000
LRS = (0.03, 0.012)
N_CODES_FOR_SIGMA = 4096


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--results", default=os.path.join(REPO, "results_seeds_diag.jsonl"))
    p.add_argument("--steps", type=int, default=STEPS)
    p.add_argument("--threads", type=int, default=4)
    return p.parse_args(argv)


def single_crop_fits(seed, steps):
    """part_a of mech_test.py with the seed threaded through.  Returns {objective: row}."""
    d, e1, _e2, _rad, ia, _ib = MT.load_pair()
    idx = int(ia)

    sim = MT.IQPFull(N_QUBITS)
    p = np.asarray(d.train_pmfs[idx], dtype=np.float64)
    p_t = torch.as_tensor(p, dtype=MT.DTYPE)
    S = float(d.train_totals[idx])
    img_p, q_p = MT.quants(p, S)

    rng = np.random.default_rng(seed)
    codes = codec10.sample_codes_from_pmf(p, N_CODES_FOR_SIGMA, rng)
    sigmas, mpw = train_iqp.resolve_sigmas("auto3", N_QUBITS, codes, seed)
    ref = iqp_exact.IQPExact(N_QUBITS, N_QUBITS, device="cpu", dtype=torch.complex128)
    w_auto = ref.kernel_weights(sigmas).to(MT.DTYPE)
    w_flat = torch.ones(sim.size, dtype=MT.DTYPE) / float(sim.size)
    ez_data = fwht(p_t)

    objs = {
        "KL": lambda th: MT.kl_from_ce(p_t, sim.probs(th)),
        "MMD-flat": lambda th: ((fwht(sim.probs(th)) - ez_data) ** 2 * w_flat).sum(),
        "MMD-auto3": lambda th: ((fwht(sim.probs(th)) - ez_data) ** 2 * w_auto).sum(),
    }

    print("[diag-a seed %d] target image %d  radius %.3f e1 %+.4f e2 %+.4f ring %.3f"
          % (seed, idx, q_p["radius"], q_p["e1"], q_p["e2"], MT.ringiness(img_p)), flush=True)

    out = {}
    for name, fn in objs.items():
        best_v, best_th = float("inf"), None
        for r, lr in enumerate(LRS):
            g = torch.Generator().manual_seed(100 + r + 1000 * int(seed))
            th0 = (torch.randn(sim.size, generator=g, dtype=MT.DTYPE) * 0.05).requires_grad_(True)
            v, st = MT.fit(lambda ps: fn(ps[0]), [th0], int(steps), lr, seed=r,
                           tag="%s r%d" % (name, r))
            if v < best_v:
                best_v, best_th = v, st[0]
        with torch.no_grad():
            q = sim.probs(best_th).numpy()
        img, qq = MT.quants(q, S)
        out[name] = {
            "objective_value": float(best_v), "corr": float(MT.corr(q, p)),
            "radius": float(qq["radius"]), "e1": float(qq["e1"]), "e2": float(qq["e2"]),
            "ringiness": float(MT.ringiness(img)),
            "kl": float(MT.kl_from_ce(p_t, torch.as_tensor(q, dtype=MT.DTYPE))),
        }
        print("  [diag-a seed %d] %-10s corr %.6f  radius %.3f  e1 %+.4f  ring %.3f  KL %.4f"
              % (seed, name, out[name]["corr"], out[name]["radius"], out[name]["e1"],
                 out[name]["ringiness"], out[name]["kl"]), flush=True)

    meta = {"image_index": idx, "target_radius": float(q_p["radius"]),
            "target_e1": float(q_p["e1"]), "sigmas_auto3": [float(s) for s in sigmas],
            "mean_pauli_weights": [float(x) for x in np.asarray(mpw).ravel()],
            "n_params": int(sim.size), "steps": int(steps), "lrs": list(LRS)}
    return out, meta


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(int(args.threads))
    t0 = time.time()
    fits, meta = single_crop_fits(int(args.seed), int(args.steps))
    rows = []
    for name, rec in fits.items():
        row = {"diag": "single_crop_fit", "seed": int(args.seed), "objective": name,
               "script": "seeds/diag_single_crop_seeds.py"}
        row.update(rec)
        row.update(meta)
        rows.append(row)
    with open(args.results, "a") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print("[diag-a seed %d] appended %d rows to %s (%.1f s)"
          % (args.seed, len(rows), args.results, time.time() - t0), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
