#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""figA_fig3_c.pdf -- figA_fig3 at its native body size (0.38\\textwidth = 2.09 in).

Same data and same reduction as `figA_fig3.py` (its loader and envelope are imported, so the
two cannot drift); only the annotation is condensed so 7 pt text fits without scaling.
Python 3.9 compatible.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

import fig_style as S
import figA_fig3 as F3


def make(outdir):
    S.apply_style()
    data, _meta = F3.load()
    xmax = max(x.max() for x, _y in data.values())

    fig = plt.figure(figsize=(2.09, 1.88))
    ax = fig.add_axes([0.255, 0.215, 0.725, 0.745])
    for g, col, mk in F3.GATES:
        x, y = data[g]
        ax.plot(x, y, ls="none", marker=mk, ms=2.2, color=col, alpha=0.5, mew=0,
                zorder=3, label="$G_n = %d$" % g)
        ex, ey = F3.envelope(x, y, xmax)
        ax.plot(ex, ey, "-", color=col, lw=1.2, zorder=4)
    ax.set_yscale("log")
    ax.set_xlim(-0.018, 0.445)
    ax.set_ylim(7e-5, 4.0e-1)
    ax.set_xlabel(r"target $I_{\mathrm{QCLI}}$")
    ax.set_ylabel(r"achieved MMD$^2$")
    ax.set_xticks([0, 0.2, 0.4])
    S.tidy(ax)
    ax.text(0.985, 0.012, "$n = 16$, $\\leq$4-local targets;\n$\\leq$2-local learners",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.0,
            color=S.INK_MUTED, linespacing=1.25, zorder=6)
    ax.legend(loc="upper left", bbox_to_anchor=(-0.02, 1.04), fontsize=7.0,
              handletextpad=0.25, labelspacing=0.2, borderpad=0.15, borderaxespad=0.0,
              markerscale=1.9)
    return S.save(fig, outdir, "figA_fig3_c")


if __name__ == "__main__":
    make(os.path.dirname(os.path.abspath(__file__)))
