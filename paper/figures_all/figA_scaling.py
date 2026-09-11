#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""figA_scaling.pdf -- the scaling panel of fig2_mechanism_scaling, on its own (half column).

Conv2D-MMD^2 (solid) and its flux-aligned variant (dashed, open markers) against the
training-set size, for the IQP arm and the two reference baselines, read against the held-out
split-half floor.  Data and arm list are imported from `fig2_mechanism_scaling.py`.
Nothing is below 7 pt.

Python 3.9 compatible.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402
from matplotlib.lines import Line2D                    # noqa: E402

import fig_style as S
import fig2_mechanism_scaling as F2


def make(outdir):
    S.apply_style()
    data, floor, floor_al = F2.load_scaling()
    fig = plt.figure(figsize=(S.COLWIDTH, 2.10))
    ax = fig.add_axes([0.215, 0.185, 0.755, 0.775])

    n_vals = sorted({int(v) for n, _a, _b in data.values() for v in n})
    lo, hi = min(floor, floor_al), max(floor, floor_al)
    ax.axhspan(lo, hi, color="#c3ccd3", alpha=0.9, zorder=0, lw=0)
    ax.text(104, hi * 1.16, "held-out split-half floor", fontsize=7.0, color="#4a5a66",
            ha="right", va="bottom", zorder=4)
    for arm, lab, col, mk in F2.SCALING_ARMS:
        n, m2, m2a = data[arm]
        ax.plot(n, m2, "-", marker=mk, color=col, lw=1.3, ms=3.8, mec="white", mew=0.4,
                zorder=4, label=lab)
        ax.plot(n, m2a, ls=(0, (3.2, 1.8)), marker=mk, color=col, lw=1.0, ms=3.0,
                mfc="white", mew=0.8, zorder=3)
    ax.set_yscale("log")
    ax.set_xlabel("training images $N$")
    ax.set_ylabel("Conv2D-MMD$^2$")
    ax.set_xlim(5, 105)
    ax.set_ylim(2.6e-3, 1.6)
    ax.set_xticks(n_vals)
    ax.tick_params(axis="x", pad=1.5)
    S.tidy(ax)
    h, _l = ax.get_legend_handles_labels()
    h = h + [Line2D([], [], color=S.INK_MUTED, lw=1.0, ls=(0, (3.2, 1.8)),
                    label="flux-aligned")]
    ax.legend(handles=h, loc="upper left", bbox_to_anchor=(-0.015, 1.04), ncol=2,
              fontsize=7.0, handletextpad=0.4, columnspacing=0.8, labelspacing=0.25,
              borderpad=0.2, borderaxespad=0.0)
    return S.save(fig, outdir, "figA_scaling")


if __name__ == "__main__":
    make(os.path.dirname(os.path.abspath(__file__)))
