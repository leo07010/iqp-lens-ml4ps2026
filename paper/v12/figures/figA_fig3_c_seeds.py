#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""figA_fig3_c_seeds.pdf -- figA_fig3_c averaged over 5 LEARNER seeds.

The 140 target circuits, their 10^4-sample datasets and the three <=2-local learner
architectures are IDENTICAL in all five runs (fig3_learner_seeds.py holds --seed at the
published 20260907, which is what generates them); only the learner's initial angles move,
via np.random.default_rng(seed + 991 + G_n + 1e6 * learner_seed).  Learner seed 0 re-derives
results_fig3_v2.jsonl, so the seed-0 curve of this figure is the published one.

Because the x-axis (each target's exact I_QCLI) is therefore the SAME in every seed, the two
reductions are done per target and per bin rather than by pooling clouds:

    points    the geometric mean over seeds of that target's achieved MMD^2 (geometric, because
              the axis is logarithmic and the quantity spans three decades)
    line      the upper envelope of the seed-geometric-mean, i.e. figA_fig3.envelope applied to
              the averaged curve -- the same 8-bin per-bin-maximum reduction as the 1-seed
              figure
    band      the same envelope computed INSIDE each seed, then filled between the per-bin
              minimum and maximum across seeds; it is the seed-to-seed spread of the envelope,
              not a percentile of the point cloud

Python 3.9 compatible.
"""
from __future__ import annotations

import collections
import json
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

import fig_style as S
import figA_fig3 as F3

OUTC = os.path.join(F3.EXP, "outputs_controlled")
SEEDS = [0, 1, 2, 3, 4]
NBINS = F3.NBINS
TARGET_DRIFT_TOL = 1e-4


def load_seeds(seeds=SEEDS):
    """{G: (x (T,), Y (S, T))} with the targets in a single common order."""
    per, drift = {}, 0.0
    for ls in seeds:
        fn = os.path.join(OUTC, "results_fig3_v2_ls%d.jsonl" % ls)
        if not os.path.exists(fn):
            raise SystemExit("missing %s -- run submit_fig3_seeds.sbatch first" % fn)
        d = collections.defaultdict(dict)
        with open(fn) as f:
            for line in f:
                if '"fit"' not in line:
                    continue
                r = json.loads(line)
                if r.get("record") != "fit":
                    continue
                d[int(r["learner_gates"])][int(r["target_id"])] = (
                    float(r["I_QCLI_exact"]), float(r["final_mmd2"]))
        per[ls] = d

    out = {}
    for g, _c, _m in F3.GATES:
        ids = sorted(per[seeds[0]][g])
        x = np.asarray([per[seeds[0]][g][i][0] for i in ids])
        for ls in seeds[1:]:
            xi = np.asarray([per[ls][g][i][0] for i in ids])
            drift = max(drift, float(np.abs(xi - x).max()))
            # The targets are rebuilt from the same rng stream in every run, so they agree to
            # float64 GPU reproducibility, not bit-exactly: runs that landed on different nodes
            # differ by ~2e-6 in exact I_QCLI (5e-6 of the 0..0.408 axis).  Anything larger
            # would mean a real target change and must stop the figure.
            if np.abs(xi - x).max() > TARGET_DRIFT_TOL:
                raise RuntimeError("target I_QCLI moved between learner seeds at G_n=%d "
                                   "(max |diff| %.3e > %.0e) -- the targets were supposed to "
                                   "be fixed" % (g, np.abs(xi - x).max(), TARGET_DRIFT_TOL))
        Y = np.asarray([[per[ls][g][i][1] for i in ids] for ls in seeds])
        keep = np.isfinite(Y).all(axis=0) & (Y > 0).all(axis=0)
        o = np.argsort(x[keep])
        out[g] = (x[keep][o], Y[:, keep][:, o])
    print("[figA_fig3_c_seeds] %d learner seeds; max target I_QCLI drift across runs %.2e"
          % (len(seeds), drift))
    return out, list(seeds)


def envelope_seeds(x, Y, xmax):
    """Per-bin maxima: mean curve (geometric over seeds) plus the across-seed min/max band."""
    gm = np.exp(np.log(Y).mean(axis=0))                       # (T,) geometric mean over seeds
    e = np.linspace(0.0, xmax, NBINS + 1)
    cx, cy, lo, hi = [], [], [], []
    for i in range(NBINS):
        m = (x >= e[i]) & (x <= e[i + 1]) if i == NBINS - 1 else (x >= e[i]) & (x < e[i + 1])
        if not m.any():
            continue
        j = int(np.argmax(gm[m]))
        per_seed_max = Y[:, m].max(axis=1)                    # (S,)
        cx.append(float(x[m][j]))
        cy.append(float(np.exp(np.log(per_seed_max).mean())))
        lo.append(float(per_seed_max.min()))
        hi.append(float(per_seed_max.max()))
    return (np.asarray(cx), np.asarray(cy), np.asarray(lo), np.asarray(hi), gm)


def make(outdir):
    S.apply_style()
    data, seeds = load_seeds()
    xmax = max(x.max() for x, _Y in data.values())

    fig = plt.figure(figsize=(2.09, 1.88))
    ax = fig.add_axes([0.255, 0.215, 0.725, 0.745])
    for g, col, mk in F3.GATES:
        x, Y = data[g]
        ex, ey, elo, ehi, gm = envelope_seeds(x, Y, xmax)
        ax.plot(x, gm, ls="none", marker=mk, ms=2.2, color=col, alpha=0.5, mew=0,
                zorder=3, label="$G_n = %d$" % g)
        ax.fill_between(ex, elo, ehi, color=col, alpha=0.20, lw=0, zorder=2)
        ax.plot(ex, ey, "-", color=col, lw=1.2, zorder=4)
    ax.set_yscale("log")
    ax.set_xlim(-0.018, 0.445)
    ax.set_ylim(7e-5, 4.0e-1)
    ax.set_xlabel(r"target $I_{\mathrm{QCLI}}$")
    ax.set_ylabel(r"achieved MMD$^2$")
    ax.set_xticks([0, 0.2, 0.4])
    S.tidy(ax)
    ax.text(0.985, 0.012, "$n = 16$, $\\leq$4-local targets;\n"
                          "$\\leq$2-local learners, %d seeds" % len(seeds),
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.0,
            color=S.INK_MUTED, linespacing=1.25, zorder=6)
    ax.legend(loc="upper left", bbox_to_anchor=(-0.02, 1.04), fontsize=7.0,
              handletextpad=0.25, labelspacing=0.2, borderpad=0.15, borderaxespad=0.0,
              markerscale=1.9)
    return S.save(fig, outdir, "figA_fig3_c_seeds")


if __name__ == "__main__":
    make(os.path.dirname(os.path.abspath(__file__)))
