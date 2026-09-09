#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""figA_fig4_c.pdf -- figA_fig4 at its native body size (0.38\\textwidth = 2.09 in).

Same data and same binned reduction as `figA_fig4.py` (loader and `binned` imported); only
the two in-panel annotations are shortened so 7 pt text fits without scaling.
Python 3.9 compatible.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

import fig_style as S
import figA_fig4 as F4

LABEL_AT = {8: (0.70, 1.055), 12: (0.72, 0.60), 16: (0.44, 0.20)}


def make(outdir):
    S.apply_style()
    xs, ys, calib = F4.load()

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

    ax.set_xlim(-0.02, 1.045)
    ax.set_ylim(-0.05, 1.18)
    ax.set_xlabel(r"$I_{\mathrm{QCLI}}$")
    ax.set_ylabel(r"$I_{\mathrm{CCI}}$")
    ax.set_xticks([0, 0.5, 1.0]); ax.set_yticks([0, 0.5, 1.0])
    S.tidy(ax)
    return S.save(fig, outdir, "figA_fig4_c")


if __name__ == "__main__":
    make(os.path.dirname(os.path.abspath(__file__)))
