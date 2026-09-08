#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate mech_seeds.py runs into results_seeds.json (the seeds version of results.json).

Also checks that --seed 0 of mech_seeds.py reproduces the three published seed-0 sources that
fig2_mechanism_scaling.M_SWEEP reads (results.json, diag3.json, diag.json) and the two diag2
random-direction draws.  A non-zero difference there means the unified script drifted from one
of the original protocols and the whole seed table is suspect, so it is reported, never hidden.

Python 3.9 compatible.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MS = [0, 2, 4, 6, 8, 16]

# the published seed-0 values, and where fig2_mechanism_scaling.py reads each of them
PUBLISHED = {
    0:  ("results.json", ("configs", "KL_M0", "holdout", "corr_mean")),
    2:  ("diag3.json",   ("KL_M2", "holdout", "corr_mean")),
    4:  ("results.json", ("configs", "KL_M4", "holdout", "corr_mean")),
    6:  ("diag3.json",   ("KL_M6", "holdout", "corr_mean")),
    8:  ("diag.json",    ("exp2_redo", "M8", "best", "corr_mean")),
    16: ("diag.json",    ("exp2_redo", "M16", "best", "corr_mean")),
}
PUBLISHED_REFS = {
    "nearest":  ("results.json", ("controls", "holdout_vs_nearest_train_corr", "mean")),
    "mean_img": ("results.json", ("controls", "holdout_vs_mean_image_corr", "mean")),
}


def dig(d, path):
    for k in path:
        d = d[k]
    return float(d)


def ms(v):
    a = np.asarray(v, dtype=np.float64)
    return float(a.mean()), (float(a.std(ddof=1)) if a.size > 1 else 0.0)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--glob", default=os.path.join(HERE, "mech_seeds_s*.json"))
    p.add_argument("--out", default=os.path.join(HERE, "results_seeds.json"))
    a = p.parse_args(argv)

    files = sorted(glob.glob(a.glob))
    if not files:
        raise SystemExit("no files matched %s" % a.glob)
    runs = []
    for fn in files:
        with open(fn) as fh:
            d = json.load(fh)
        runs.append(d)
    runs.sort(key=lambda r: r["seed"])
    seeds = [int(r["seed"]) for r in runs]
    print("[agg] %d seeds: %s" % (len(runs), seeds))

    # ---- reproduction check for seed 0 --------------------------------------------------
    repro = None
    r0 = [r for r in runs if int(r["seed"]) == 0]
    if r0:
        cache, diffs = {}, {}
        for M, (fn, path) in PUBLISHED.items():
            if fn not in cache:
                with open(os.path.join(HERE, fn)) as fh:
                    cache[fn] = json.load(fh)
            pub = dig(cache[fn], path)
            got = float(r0[0]["M_sweep_holdout_corr"][str(M)])
            diffs["M=%d" % M] = {"published": pub, "reproduced": got, "abs_diff": abs(pub - got),
                                 "source": fn}
        for name, (fn, path) in PUBLISHED_REFS.items():
            pub = dig(cache[fn], path)
            key = ("holdout_vs_nearest_train_corr" if name == "nearest"
                   else "holdout_vs_mean_image_corr")
            got = float(r0[0]["controls"][key]["mean"])
            diffs[name] = {"published": pub, "reproduced": got, "abs_diff": abs(pub - got),
                           "source": fn}
        with open(os.path.join(HERE, "diag2.json")) as fh:
            d2 = json.load(fh)
        pub_keys = [("random_phi_%d" % s, "random Phi(seed %d) + trained theta0" % s)
                    for s in (99, 123)]
        # diag2.py computes the correctly PAIRED variant (untrained Phi on the M=0 theta0)
        # for Phi seed 99 only, so only that one can be checked against it.
        pub_keys.append(("random_phi_99_M0theta0", "random Phi(seed 99) + M=0 theta0"))
        for tag, k in pub_keys:
            pub = float(d2["holdout_random_vs_trained"][k]["corr_mean"])
            got = float(r0[0]["holdout_random_vs_trained"][k]["corr_mean"])
            diffs[tag] = {"published": pub, "reproduced": got,
                          "abs_diff": abs(pub - got), "source": "diag2.json"}
        worst = max(v["abs_diff"] for v in diffs.values())
        repro = {"max_abs_diff": worst, "per_item": diffs}
        print("[agg] seed-0 reproduction vs published: max |diff| = %.3e" % worst)
        for k, v in sorted(diffs.items()):
            flag = "OK " if v["abs_diff"] < 1e-9 else "!! "
            print("   %s%-16s published %.6f  reproduced %.6f  |diff| %.2e  (%s)"
                  % (flag, k, v["published"], v["reproduced"], v["abs_diff"], v["source"]))

    # ---- aggregation --------------------------------------------------------------------
    sweep = {}
    for M in MS:
        vals = [float(r["M_sweep_holdout_corr"][str(M)]) for r in runs]
        m, s = ms(vals)
        sweep[str(M)] = {"per_seed": vals, "mean": m, "sd": s,
                         "range": [float(np.min(vals)), float(np.max(vals))],
                         "protocol": runs[0]["M_sweep_protocol"][str(M)]}
    refs = {}
    for name, key in (("nearest", "holdout_vs_nearest_train_corr"),
                      ("mean_img", "holdout_vs_mean_image_corr")):
        vals = [float(r["controls"][key]["mean"]) for r in runs]
        m, s = ms(vals)
        refs[name] = {"per_seed": vals, "mean": m, "sd": s,
                      "note": "function of the fixed data split only; sd must be 0"}
    vals = [float(r["holdout_random_vs_trained"]["mean_over_draws"]) for r in runs]
    m, s = ms(vals)
    refs["random_mismatched"] = {"per_seed": vals, "mean": m, "sd": s,
                                 "note": "untrained Phi on each seed's TRAINED M=4 theta0 -- a "
                                         "mismatched pairing (that theta0 was optimised jointly "
                                         "with a trained Phi); kept because it is what the "
                                         "seed-0 figure used"}
    vals = [float(r["holdout_random_vs_trained"]["mean_over_draws_M0theta0"]) for r in runs]
    m, s = ms(vals)
    refs["random"] = {"per_seed": vals, "mean": m, "sd": s,
                      "note": "PAIRED control, the one the figure draws: untrained Phi on each "
                              "seed's M=0 theta0 (fitted without any latent), mean over the two "
                              "fixed Phi draws (torch seeds 99, 123)"}
    # train-side correlation, useful for the 'memorises but does not generalise' reading
    train = {}
    for M in MS:
        key = ("KL_M%d_diag3" % M) if M in (2, 6) else ("KL_M%d" % M)
        vals = [float(r["configs"][key]["train"]["corr_mean"]) for r in runs]
        m, s = ms(vals)
        train[str(M)] = {"per_seed": vals, "mean": m, "sd": s}

    out = {"meta": {"n_seeds": len(runs), "seeds": seeds,
                    "files": [os.path.basename(f) for f in files],
                    "script": os.path.abspath(__file__),
                    "note": "sd is the sample sd (ddof=1) over model seeds; the data split is "
                            "fixed at data_lens_r3.R3_SPLIT_SEED for every seed",
                    "reproduction_check_seed0": repro},
           "M_sweep_holdout_corr": sweep, "M_sweep_train_corr": train, "references": refs}
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1, default=float)

    print()
    print("| M | held-out projection corr (mean +- sd) | train corr (mean +- sd) | protocol |")
    print("|---|---|---|---|")
    for M in MS:
        print("| %d | %.4f $\\pm$ %.4f | %.4f $\\pm$ %.4f | %s |"
              % (M, sweep[str(M)]["mean"], sweep[str(M)]["sd"],
                 train[str(M)]["mean"], train[str(M)]["sd"], sweep[str(M)]["protocol"]))
    print()
    for k in ("nearest", "mean_img", "random", "random_mismatched"):
        print("| reference %-9s | %.4f $\\pm$ %.4f |" % (k, refs[k]["mean"], refs[k]["sd"]))
    print("\n[agg] wrote %s" % a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
