#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""figA_fig4_c_seeds.pdf -- figA_fig4_c over 5 SPSA seeds instead of 3.

Same panel, same binned reduction (`figA_fig4.binned`: 18 equal I_QCLI bins, bins with fewer
than 20 circuits dropped, line = per-bin median I_CCI, band = 10th-90th percentile), same
axes and same annotations; only the input cloud grows.

    outputs_controlled/results_fig4.jsonl          seeds 0-2, master seed 20260907 (published)
    outputs_controlled/results_fig4_seeds34.jsonl  seeds 3-4, master seed 20260908

A "seed" is one full SPSA trajectory: its own random <=2-local gate mask set, its own
N(0, 0.05) initial angles and its own perturbation stream, run at both gain settings
(a = 0.5, 2.0).  The two files carry the same exact i.i.d.-uniform calibration points, which
the loader checks rather than assumes.

The band therefore now spans 5 seeds x 2 gains x 3,334 recorded steps ~ 33,340 circuits per
system size.  Python 3.9 compatible.
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
import figA_fig4 as F4

EXTRA = os.path.join(F4.EXP, "outputs_controlled", "results_fig4_seeds34.jsonl")
LABEL_AT = {8: (0.755, 1.055), 12: (0.74, 0.665), 16: (0.42, 0.175)}


def _read(path, xs, ys, calib, seed_offset, seen):
    with open(path) as f:
        for line in f:
            if '"traj"' in line:
                r = json.loads(line)
                xs[r["n"]].append(r["I_QCLI"])
                ys[r["n"]].append(r["I_CCI"])
                seen[r["n"]].add(int(r["seed_idx"]) + seed_offset)
            elif '"calibration"' in line:
                r = json.loads(line)
                n = r["n"]
                if n in calib:
                    if max(abs(calib[n][0] - r["I_QCLI"]), abs(calib[n][1] - r["I_CCI"])) > 1e-9:
                        raise RuntimeError("calibration mismatch at n=%d between the two "
                                           "fig4 files: %s vs %s"
                                           % (n, calib[n], (r["I_QCLI"], r["I_CCI"])))
                else:
                    calib[n] = (r["I_QCLI"], r["I_CCI"])


def load():
    xs, ys = collections.defaultdict(list), collections.defaultdict(list)
    calib, seen = {}, collections.defaultdict(set)
    _read(F4.JSONL, xs, ys, calib, 0, seen)
    if not os.path.exists(EXTRA):
        raise SystemExit("missing %s -- run submit_fig4_seeds.sbatch first" % EXTRA)
    _read(EXTRA, xs, ys, calib, 3, seen)
    n_seeds = sorted(set(len(v) for v in seen.values()))
    return ({n: np.asarray(v) for n, v in xs.items()},
            {n: np.asarray(v) for n, v in ys.items()}, calib,
            n_seeds[0] if len(n_seeds) == 1 else n_seeds)


def make(outdir):
    S.apply_style()
    xs, ys, calib, n_seeds = load()

    fig = plt.figure(figsize=(2.09, 1.88))
    ax = fig.add_axes([0.245, 0.215, 0.735, 0.745])
    ax.axvspan(-0.02, F4.PAPER_XMAX, color="#e8ebee", zorder=0, lw=0)
    ax.axvline(F4.PAPER_XMAX, color="#9aa5ad", lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.text(F4.PAPER_XMAX - 0.03, 1.14, "paper's range", fontsize=7.0, color="#5d6b76",
            ha="right", va="top", zorder=5)

    for n in sorted(xs):
        cx, cy, lo, hi = F4.binned(xs[n], ys[n])
        c = F4.COLS[n]
        ax.fill_between(cx, lo, hi, color=c, alpha=0.16, lw=0, zorder=2)
        ax.plot(cx, cy, "-", marker=F4.MARKS[n], color=c, lw=1.2, ms=2.6, mec="white",
                mew=0.35, zorder=4)
        lx, ly = LABEL_AT[n]
        ax.text(lx, ly, "$n{=}%d$" % n, fontsize=7.0, color=c, ha="left", va="center",
                zorder=6)
        qx, qy = calib[n]
        ax.plot([qx], [qy], marker="*", ms=7.5, color=c, ls="none", mec="white", mew=0.4,
                zorder=7)
    ax.text(0.035, 0.40, "uniform\ncalib. ($\\star$)", fontsize=7.0, color=S.INK_MUTED,
            ha="left", va="top", zorder=6, linespacing=1.2)
    ax.text(0.035, 0.09, "%d seeds" % n_seeds, fontsize=6.6, color=S.INK_MUTED,
            ha="left", va="top", zorder=6)

    ax.set_xlim(-0.02, 1.045)
    ax.set_ylim(-0.05, 1.18)
    ax.set_xlabel(r"$I_{\mathrm{QCLI}}$")
    ax.set_ylabel(r"$I_{\mathrm{CCI}}$")
    ax.set_xticks([0, 0.5, 1.0]); ax.set_yticks([0, 0.5, 1.0])
    S.tidy(ax)
    return S.save(fig, outdir, "figA_fig4_c_seeds")


if __name__ == "__main__":
    make(os.path.dirname(os.path.abspath(__file__)))
