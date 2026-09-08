#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate run_map.py outputs over seeds into results_map_seeds.json + a mean +- sd table.

Table 1 of the ML4PS submission quotes three rows of the correlation-complexity map, each with
its structure-matched permutation null:

    lens,          18-bit (x,y,z) tuples, per-image      K = 4,096
    turbulence D2, 18-bit tuples, per-snapshot           K = 4,096
    lens,          10-bit crop pmf, per-image            exact pmf (no sampling)

`run_map.py --seed s --with-d2` is the ONLY thing that changes between seeds.  What the seed
actually moves is documented in the output (`varies_with_seed` per row), because the three rows
are not equally stochastic:

  * the lens 18-bit per-image I_QCLI/I_CCI are an EXACT statistic of a deterministic code pool
    (a 64x64 image contributes each of the 4,096 (kx,ky) cells exactly once), reused verbatim
    from the round-1 json -- they cannot move with the seed;
  * the lens 10-bit exact-pmf I_QCLI/I_CCI are computed from the population pmf of a FIXED
    holdout split (data_lens_r3.R3_SPLIT_SEED, decoupled from any arm seed by design) -- they
    cannot move either;
  * both permutation nulls, and the whole turbulence per-snapshot row (which subsamples the
    native grid to K = 4,096), are Monte-Carlo and do move.

Python 3.9 compatible.  No SLURM resources needed beyond the job that produced the inputs.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# (paper row label, dataset, representation substring, how K was obtained) in Table 1 order.
# The K note matters: "K = 4,096" means two different things on the two 18-bit rows.  A 64x64
# lens image contributes each of the 4,096 (k_x, k_y) cells exactly ONCE, with that cell's own
# k_z -- the pool is an exhaustive sweep, not a sample, so K = 4,096 is the whole population
# and the row carries no sampling noise at all.  The turbulence row at the same K is a random
# SUBSAMPLE of the snapshot's native grid, chosen to match the lens K, and it does carry
# sampling noise.  The native turbulence row (K = 5x10^4, the reference paper's own N_t) is
# included so the comparison can be read at the paper's sample count as well.
ROWS = [
    ("Lens, 18-bit (x,y,z) tuples", "lens", "18-bit (x,y,z) tuples, per-image",
     "exhaustive: one k_z per (k_x,k_y) cell of the 64x64 image"),
    ("Turbulence D2, 18-bit tuples", "turbulence D2", "18-bit tuples, per-snapshot (M=4096)",
     "subsampled from the native grid to match the lens K"),
    ("Turbulence D2, 18-bit tuples (native K)", "turbulence D2",
     "18-bit tuples, per-snapshot (N_t=5e4)",
     "the reference paper's own N_t = 5x10^4 draws per snapshot"),
    ("Lens, 10-bit crop pmf", "lens", "10-bit crop pmf, per-image (exact pmf)",
     "no sampling: the exact population pmf of the 32x32 crop"),
]
FIELDS = ["I_QCLI", "I_CCI", "perm_QCLI", "perm_CCI"]


def find_row(points, dataset, rep):
    for r in points:
        if r["dataset"] == dataset and r["representation"] == rep:
            return r
    raise KeyError("no row %r / %r" % (dataset, rep))


def ms(v):
    a = np.asarray(v, dtype=np.float64)
    return float(a.mean()), float(a.std(ddof=1)) if a.size > 1 else 0.0


def fmt(m, s):
    return "%.4f $\\pm$ %.4f" % (m, s)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--glob", default=os.path.join(HERE, "results_map_seed[0-9]*.json"),
                   help="per-seed run_map.py outputs; must not match the aggregate itself")
    p.add_argument("--seed0", default=os.path.join(HERE, "results_map.json"),
                   help="the committed seed-0 artefact, used only as a reproduction check")
    p.add_argument("--out", default=os.path.join(HERE, "results_map_seeds.json"))
    a = p.parse_args(argv)

    files = sorted(glob.glob(a.glob))
    if not files:
        raise SystemExit("no files matched %s" % a.glob)
    runs = []
    for fn in files:
        with open(fn) as fh:
            d = json.load(fh)
        m = re.search(r"seed(\d+)\.json$", fn)
        seed = int(m.group(1)) if m else int(d["meta"]["seed"])
        runs.append({"seed": seed, "file": fn, "data": d})
    runs.sort(key=lambda r: r["seed"])
    print("[agg] %d seed runs: %s" % (len(runs), [r["seed"] for r in runs]))

    # ---- reproduction check against the committed seed-0 artefact -----------------------
    repro = None
    if os.path.exists(a.seed0):
        with open(a.seed0) as fh:
            d0 = json.load(fh)
        r0 = [r for r in runs if r["seed"] == 0]
        if r0:
            diffs = {}
            for lab, ds, rep, _kn in ROWS:
                x = find_row(r0[0]["data"]["points"], ds, rep)
                y = find_row(d0["points"], ds, rep)
                for f in FIELDS:
                    if f in x and f in y:
                        diffs["%s/%s" % (lab, f)] = abs(float(x[f]) - float(y[f]))
            worst = max(diffs.values()) if diffs else float("nan")
            repro = {"max_abs_diff_vs_results_map_json": worst, "per_field": diffs}
            print("[agg] seed-0 reproduction vs %s: max |diff| = %.3e"
                  % (os.path.basename(a.seed0), worst))

    # ---- per-row aggregation -----------------------------------------------------------
    out_rows = []
    for lab, ds, rep, knote in ROWS:
        vals = {f: [] for f in FIELDS}
        K = None
        for r in runs:
            row = find_row(r["data"]["points"], ds, rep)
            K = row["M"]
            for f in FIELDS:
                vals[f].append(float(row[f]))
        rec = {"label": lab, "dataset": ds, "representation": rep, "K": K,
               "K_note": knote,
               "seeds": [r["seed"] for r in runs],
               "per_seed": {f: vals[f] for f in FIELDS}}
        for f in FIELDS:
            m, s = ms(vals[f])
            rec["%s_mean" % f] = m
            rec["%s_sd" % f] = s
            rec["%s_range" % f] = [float(np.min(vals[f])), float(np.max(vals[f]))]
        rec["varies_with_seed"] = {f: bool(rec["%s_sd" % f] > 0) for f in FIELDS}
        out_rows.append(rec)

    out = {"meta": {"n_seeds": len(runs), "seeds": [r["seed"] for r in runs],
                    "files": [r["file"] for r in runs],
                    "script": os.path.abspath(__file__),
                    "note": ("sd is the sample sd (ddof=1) over seeds; a row whose sd is "
                             "exactly 0 is an exact statistic of a fixed split, not a "
                             "Monte-Carlo estimate"),
                    "reproduction_check": repro},
           "rows": out_rows}
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2, default=float)

    print()
    print("| Dataset and encoding | K | how K is obtained | I_QCLI | I_CCI |")
    print("|---|---|---|---|---|")
    for rec in out_rows:
        print("| %s | %s | %s | %s | %s |"
              % (rec["label"], rec["K"], rec["K_note"],
                 fmt(rec["I_QCLI_mean"], rec["I_QCLI_sd"]),
                 fmt(rec["I_CCI_mean"], rec["I_CCI_sd"])))
        print("| &nbsp;&nbsp;permutation null | %s | | %s | %s |"
              % (rec["K"], fmt(rec["perm_QCLI_mean"], rec["perm_QCLI_sd"]),
                 fmt(rec["perm_CCI_mean"], rec["perm_CCI_sd"])))
    print("\n[agg] wrote %s" % a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
