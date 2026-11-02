#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""figA_fig4.pdf -- the Fig. 4 (QCLI vs CCI) control, extended past the paper's range.

Every circuit visited along the QCLI-only optimisation is binned by its I_QCLI (18 equal
bins on [0, 1], bins with fewer than 20 circuits dropped); the line is the per-bin median
I_CCI and the band the 10th-90th percentile.  The stars are the exact i.i.d.-uniform
calibration points (I_QCLI ~ 1, I_CCI = 0) -- the ceiling of the indicator, and the place
a monotone QCLI/CCI coupling has to break.

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

EXP = "<HOME>/gsoc2026-deeplense-qml/experiments/13_iqp_turbulence_replication"
JSONL = os.path.join(EXP, "outputs_controlled", "results_fig4.jsonl")

NBINS, MIN_PER_BIN = 18, 20
PAPER_XMAX = 0.7
COLS = {8: S.OKABE_ITO["blue"], 12: S.OKABE_ITO["vermilion"], 16: S.OKABE_ITO["green"]}
MARKS = {8: "o", 12: "s", 16: "^"}
# where the direct "n = ..." label sits, in data coordinates
LABEL_AT = {8: (0.845, 1.055), 12: (0.865, 0.60), 16: (0.60, 0.235)}


def load():
    xs, ys = collections.defaultdict(list), collections.defaultdict(list)
    calib = {}
    with open(JSONL) as f:
        for line in f:
            if '"traj"' in line:
                r = json.loads(line)
                xs[r["n"]].append(r["I_QCLI"]); ys[r["n"]].append(r["I_CCI"])
            elif '"calibration"' in line:
                r = json.loads(line)
                calib[r["n"]] = (r["I_QCLI"], r["I_CCI"])
    return ({n: np.asarray(v) for n, v in xs.items()},
            {n: np.asarray(v) for n, v in ys.items()}, calib)


def binned(x, y):
    e = np.linspace(0.0, 1.0, NBINS + 1)
    cx, cy, lo, hi = [], [], [], []
    for i in range(NBINS):
        m = (x >= e[i]) & (x < e[i + 1]) if i < NBINS - 1 else (x >= e[i]) & (x <= e[i + 1])
        if m.sum() < MIN_PER_BIN:
            continue
        cx.append(float(np.median(x[m])))
        cy.append(float(np.median(y[m])))
        lo.append(float(np.percentile(y[m], 10)))
        hi.append(float(np.percentile(y[m], 90)))
    return map(np.asarray, (cx, cy, lo, hi))


def make(outdir):
    S.apply_style()
    xs, ys, calib = load()

    fig = plt.figure(figsize=(S.COLWIDTH, 2.25))
    ax = fig.add_axes([0.175, 0.175, 0.795, 0.775])

    ax.axvspan(-0.02, PAPER_XMAX, color="#e8ebee", zorder=0, lw=0)
    ax.axvline(PAPER_XMAX, color="#9aa5ad", lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.text(PAPER_XMAX - 0.03, 1.115, "paper's plotted range", fontsize=7.0,
            color="#5d6b76", ha="right", va="top", zorder=5)

    for n in sorted(xs):
        cx, cy, lo, hi = binned(xs[n], ys[n])
        c = COLS[n]
        ax.fill_between(cx, lo, hi, color=c, alpha=0.16, lw=0, zorder=2)
        ax.plot(cx, cy, "-", marker=MARKS[n], color=c, lw=1.3, ms=3.2, mec="white",
                mew=0.4, zorder=4)
        lx, ly = LABEL_AT[n]
        ax.text(lx, ly, "$n = %d$" % n, fontsize=7.5, color=c, ha="left", va="center",
                zorder=6)
        qx, qy = calib[n]
        ax.plot([qx], [qy], marker="*", ms=8.5, color=c, ls="none", mec="white",
                mew=0.5, zorder=7)

    ax.text(0.035, 0.34, "exact i.i.d.-uniform\ncalibration ($\\star$)", fontsize=7.0,
            color=S.INK_MUTED, ha="left", va="top", zorder=6, linespacing=1.2)

    ax.set_xlim(-0.02, 1.045)
    ax.set_ylim(-0.05, 1.16)
    ax.set_xlabel(r"$I_{\mathrm{QCLI}}$  (0–1)")
    ax.set_ylabel(r"$I_{\mathrm{CCI}}$  (0–1)")
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    S.tidy(ax)
    return S.save(fig, outdir, "figA_fig4")


if __name__ == "__main__":
    make(os.path.dirname(os.path.abspath(__file__)))
