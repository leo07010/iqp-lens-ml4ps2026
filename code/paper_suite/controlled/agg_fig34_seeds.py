#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Seed aggregation for App. C (Fig. 3) and App. D (Fig. 4) of the ML4PS submission.

App. C -- 5 LEARNER seeds on ONE fixed target family (140 circuits).  Inputs:
    outputs_controlled/results_fig3_v2.jsonl          the published run  = learner seed 0
    outputs_controlled/results_fig3_v2_ls{0..4}.jsonl fig3_learner_seeds.py
The ls0 file is checked target-by-target against the published one; if it does not agree to
machine precision the seed table is not trustworthy and the check says so.

Statistics recomputed per learner seed (the ones App. C quotes):
    median achieved MMD^2 per learner gate count      paper: 9.57e-3 / 2.15e-3 / 1.45e-3
    Spearman(final MMD^2, target exact I_QCLI)        paper: +0.98 / +0.90 / +0.81
    the same for the exact-pmf-trained control        ("a trend that also holds for a control")
    Spearman(Prop. II.2 floor proxy, exact I_QCLI)    paper: "rho <= -0.98"
        Two proxies are recorded because they answer different questions and only one of them
        supports the <= -0.98 wording at G_n = 150: `mmd2_floor_proxy` uses the 10^4-sample
        target spectrum, `mmd2_floor_proxy_exact` the exact one.  Neither depends on the
        learner's initial angles, so both have sd exactly 0 over learner seeds -- that is a
        property of the estimator, not a lucky run, and is reported as such.

App. D -- 5 SPSA seeds per system size.  Inputs:
    outputs_controlled/results_fig4.jsonl          seeds 0-2 (published, master seed 20260907)
    outputs_controlled/results_fig4_seeds34.jsonl  seeds 3-4 (master seed 20260908); its
        seed_idx 0,1 are relabelled 3,4 here.
Statistics per seed (pooling that seed's two SPSA gain arms), then mean +- sd over the 5:
    Spearman(I_CCI, I_QCLI) globally                  paper: -0.53 (n=12), -0.57 (n=16)
    the same restricted to I_QCLI <= 0.3              paper: +0.83 / +0.76 / +0.95
    exact i.i.d.-uniform calibration I_QCLI           paper: 0.982 / 0.998 / 1.000

Python 3.9 compatible.  Pure post-processing; no GPU, seconds.
"""
from __future__ import annotations

import argparse
import collections
import json
import os

import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "outputs_controlled"))
GATES = [50, 100, 150]
SIZES = [8, 12, 16]
QCLI_LOW = 0.3


def ms(v):
    a = np.asarray(v, dtype=np.float64)
    return float(a.mean()), (float(a.std(ddof=1)) if a.size > 1 else 0.0)


def sp(x, y):
    return float(spearmanr(np.asarray(x), np.asarray(y)).correlation)


# ============================================================================== App. C
def load_fig3(path):
    """{learner_gates: {target_id: row}} for one learner seed."""
    out = collections.defaultdict(dict)
    with open(path) as fh:
        for line in fh:
            if '"fit"' not in line:
                continue
            r = json.loads(line)
            if r.get("record") != "fit":
                continue
            out[int(r["learner_gates"])][int(r["target_id"])] = r
    return out


def fig3_stats(byg):
    st = {}
    for g in GATES:
        rows = [byg[g][i] for i in sorted(byg[g])]
        x = np.array([r["I_QCLI_exact"] for r in rows])
        y = np.array([r["final_mmd2"] for r in rows])
        ye = np.array([r.get("final_mmd2_exacttrained", np.nan) for r in rows])
        fp = np.array([r["mmd2_floor_proxy"] for r in rows])
        fpe = np.array([r.get("mmd2_floor_proxy_exact", np.nan) for r in rows])
        st[g] = {"n_targets": len(rows),
                 "median_mmd2": float(np.median(y)),
                 "spearman_mmd2_vs_qcli": sp(x, y),
                 "median_mmd2_exacttrained": float(np.median(ye)),
                 "spearman_mmd2_exacttrained_vs_qcli": sp(x, ye),
                 "spearman_floorproxy_vs_qcli": sp(x, fp),
                 "spearman_floorproxy_exact_vs_qcli": sp(x, fpe)}
    return st


# ============================================================================== App. D
def load_fig4(path, seed_offset=0):
    xs = collections.defaultdict(lambda: collections.defaultdict(list))   # n -> seed -> QCLI
    ys = collections.defaultdict(lambda: collections.defaultdict(list))
    calib = {}
    with open(path) as fh:
        for line in fh:
            if '"traj"' in line:
                r = json.loads(line)
                s = int(r["seed_idx"]) + seed_offset
                xs[int(r["n"])][s].append(r["I_QCLI"])
                ys[int(r["n"])][s].append(r["I_CCI"])
            elif '"calibration"' in line:
                r = json.loads(line)
                calib[int(r["n"])] = (float(r["I_QCLI"]), float(r["I_CCI"]))
    return xs, ys, calib


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", default=OUT_DIR)
    p.add_argument("--fig3-published", default=None)
    p.add_argument("--fig3-seeds", default="0,1,2,3,4")
    p.add_argument("--fig4-published", default=None)
    p.add_argument("--fig4-extra", default=None)
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    od = os.path.abspath(a.out_dir)
    f3_pub = a.fig3_published or os.path.join(od, "results_fig3_v2.jsonl")
    f4_pub = a.fig4_published or os.path.join(od, "results_fig4.jsonl")
    f4_ext = a.fig4_extra or os.path.join(od, "results_fig4_seeds34.jsonl")
    out_path = a.out or os.path.join(od, "results_fig34_seeds.json")
    ls_list = [int(x) for x in a.fig3_seeds.split(",") if x.strip()]

    res = {"meta": {"script": os.path.abspath(__file__), "out_dir": od}}

    # ---------------------------------------------------------------- App. C
    print("=" * 88)
    print("App. C (Fig. 3) -- %d learner seeds, one fixed target family" % len(ls_list))
    per_seed, missing = {}, []
    for ls in ls_list:
        fn = os.path.join(od, "results_fig3_v2_ls%d.jsonl" % ls)
        if not os.path.exists(fn):
            missing.append(fn)
            continue
        per_seed[ls] = load_fig3(fn)
    if missing:
        print("[warn] missing: %s" % ", ".join(os.path.basename(m) for m in missing))

    repro3 = None
    if 0 in per_seed and os.path.exists(f3_pub):
        pub = load_fig3(f3_pub)
        d = 0.0
        n_cmp = 0
        for g in GATES:
            for i in sorted(pub[g]):
                for k in ("final_mmd2", "final_mmd2_exacttrained", "mmd2_floor_proxy"):
                    if k in pub[g][i] and k in per_seed[0][g][i]:
                        d = max(d, abs(float(pub[g][i][k]) - float(per_seed[0][g][i][k])))
                        n_cmp += 1
        repro3 = {"max_abs_diff": d, "n_values_compared": n_cmp,
                  "published": os.path.basename(f3_pub)}
        print("[check] learner seed 0 vs %s: max |diff| = %.3e over %d values"
              % (os.path.basename(f3_pub), d, n_cmp))

    stats3 = {ls: fig3_stats(byg) for ls, byg in sorted(per_seed.items())}
    agg3 = {}
    keys3 = ["median_mmd2", "spearman_mmd2_vs_qcli", "median_mmd2_exacttrained",
             "spearman_mmd2_exacttrained_vs_qcli", "spearman_floorproxy_vs_qcli",
             "spearman_floorproxy_exact_vs_qcli"]
    for g in GATES:
        agg3[g] = {}
        for k in keys3:
            vals = [stats3[ls][g][k] for ls in sorted(stats3)]
            m, s = ms(vals)
            agg3[g][k] = {"mean": m, "sd": s, "per_seed": vals}
    print()
    print("| $G_n$ | median MMD$^2$ | Spearman($\\mathrm{MMD}^2$, $I_{QCLI}$) | "
          "median MMD$^2$ (exact-trained) | Spearman (exact-trained) | "
          "Spearman(floor proxy) | Spearman(floor proxy, exact) |")
    print("|---|---|---|---|---|---|---|")
    for g in GATES:
        r = agg3[g]
        print("| %d | %.3e $\\pm$ %.1e | %+.4f $\\pm$ %.4f | %.3e $\\pm$ %.1e | "
              "%+.4f $\\pm$ %.4f | %+.4f $\\pm$ %.4f | %+.4f $\\pm$ %.4f |"
              % (g, r["median_mmd2"]["mean"], r["median_mmd2"]["sd"],
                 r["spearman_mmd2_vs_qcli"]["mean"], r["spearman_mmd2_vs_qcli"]["sd"],
                 r["median_mmd2_exacttrained"]["mean"], r["median_mmd2_exacttrained"]["sd"],
                 r["spearman_mmd2_exacttrained_vs_qcli"]["mean"],
                 r["spearman_mmd2_exacttrained_vs_qcli"]["sd"],
                 r["spearman_floorproxy_vs_qcli"]["mean"],
                 r["spearman_floorproxy_vs_qcli"]["sd"],
                 r["spearman_floorproxy_exact_vs_qcli"]["mean"],
                 r["spearman_floorproxy_exact_vs_qcli"]["sd"]))
    res["app_c"] = {"learner_seeds": sorted(stats3), "per_seed": stats3, "aggregate": agg3,
                    "reproduction_check_ls0": repro3,
                    "note": "the floor proxies are functions of the data spectrum and the "
                            "learner's fixed Pauli support only, so they cannot move with the "
                            "learner's initial angles; their sd over seeds is exactly 0 by "
                            "construction"}

    # ---------------------------------------------------------------- App. D
    print()
    print("=" * 88)
    print("App. D (Fig. 4) -- SPSA seeds")
    xs, ys, calib = load_fig4(f4_pub, seed_offset=0)
    if os.path.exists(f4_ext):
        x2, y2, calib2 = load_fig4(f4_ext, seed_offset=3)
        for n in x2:
            for s in x2[n]:
                xs[n][s] = x2[n][s]
                ys[n][s] = y2[n][s]
        for n, v in calib2.items():
            if n in calib and max(abs(calib[n][0] - v[0]), abs(calib[n][1] - v[1])) > 1e-9:
                print("[warn] calibration mismatch at n=%d: %s vs %s" % (n, calib[n], v))
    else:
        print("[warn] missing %s -- App. D stays at 3 seeds" % os.path.basename(f4_ext))

    stats4, agg4 = {}, {}
    for n in SIZES:
        seeds = sorted(xs[n])
        stats4[n] = {}
        for s in seeds:
            x = np.asarray(xs[n][s]); y = np.asarray(ys[n][s])
            m = x <= QCLI_LOW
            stats4[n][s] = {"n_points": int(x.size),
                            "spearman_global": sp(x, y),
                            "spearman_qcli_le_0.3": (sp(x[m], y[m]) if m.sum() > 10 else float("nan")),
                            "n_points_le_0.3": int(m.sum()),
                            "max_I_QCLI": float(x.max())}
        agg4[n] = {}
        for k in ("spearman_global", "spearman_qcli_le_0.3"):
            vals = [stats4[n][s][k] for s in seeds]
            mm, sd = ms(vals)
            agg4[n][k] = {"mean": mm, "sd": sd, "per_seed": vals}
        agg4[n]["n_seeds"] = len(seeds)
        agg4[n]["uniform_calibration"] = {"I_QCLI": calib[n][0], "I_CCI": calib[n][1]}
        # POOLED across seeds: this is the statistic App. D currently quotes (it was computed
        # on the 3 published seeds pooled into one cloud).  It is NOT the mean of the per-seed
        # values -- pooling trajectories with different QCLI offsets can and does change the
        # answer -- so both are reported and the text has to say which one it means.
        xp = np.concatenate([np.asarray(xs[n][s]) for s in seeds])
        yp = np.concatenate([np.asarray(ys[n][s]) for s in seeds])
        mp = xp <= QCLI_LOW
        agg4[n]["pooled"] = {"n_points": int(xp.size),
                             "spearman_global": sp(xp, yp),
                             "spearman_qcli_le_0.3": sp(xp[mp], yp[mp]),
                             "n_points_le_0.3": int(mp.sum()),
                             "max_I_QCLI": float(xp.max())}
    print()
    print("| $n$ | seeds | global $\\rho$, per-seed mean $\\pm$ sd | global $\\rho$, pooled | "
          "$\\qcli \\leq 0.3$, per-seed mean $\\pm$ sd | $\\qcli \\leq 0.3$, pooled | "
          "uniform calib. $\\qcli$ |")
    print("|---|---|---|---|---|---|---|")
    for n in SIZES:
        r = agg4[n]
        print("| %d | %d | %+.4f $\\pm$ %.4f | %+.4f | %+.4f $\\pm$ %.4f | %+.4f | %.4f |"
              % (n, r["n_seeds"], r["spearman_global"]["mean"], r["spearman_global"]["sd"],
                 r["pooled"]["spearman_global"],
                 r["spearman_qcli_le_0.3"]["mean"], r["spearman_qcli_le_0.3"]["sd"],
                 r["pooled"]["spearman_qcli_le_0.3"],
                 r["uniform_calibration"]["I_QCLI"]))
    print()
    print("per-seed global rho (sign is not stable at n = 8, 12):")
    for n in SIZES:
        print("  n=%2d  %s" % (n, "  ".join("s%d %+.3f" % (s, stats4[n][s]["spearman_global"])
                                            for s in sorted(stats4[n]))))
    res["app_d"] = {"per_seed": {str(n): {str(s): v for s, v in stats4[n].items()}
                                 for n in SIZES},
                    "aggregate": {str(n): agg4[n] for n in SIZES},
                    "qcli_low_cut": QCLI_LOW,
                    "sources": {"published": os.path.basename(f4_pub),
                                "extra": os.path.basename(f4_ext)}}

    with open(out_path, "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print("\n[agg] wrote %s" % out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
