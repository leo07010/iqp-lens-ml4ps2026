#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fig_pipeline.pdf -- the method schematic (pure vector, matplotlib only).

`dot`/graphviz is not installed on this cluster, so the diagram is drawn with matplotlib.
Boxes are AUTO-SIZED to their text (text bbox patches measured after a draw pass), so no
label can overflow its box; the connectors are then routed between the measured edges.
Nothing here reads data; every number quoted in a box is sourced in FIGURE_PROVENANCE.md.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib.patches import FancyArrowPatch       # noqa: E402

import fig_style as S

GREY_FC, GREY_EC = "#f2f5f8", "#8fa6b8"
Q_FC, Q_EC = "#e3eef8", S.OKABE_ITO["blue"]
BAD_FC, BAD_EC = "#fdece2", S.OKABE_ITO["vermilion"]
LINK = "#5a6b78"
FS = 7.0


def node(ax, xc, yc, text, fc=GREY_FC, ec=GREY_EC, fs=FS, ha="center"):
    return ax.text(xc, yc, text, transform=ax.transAxes, ha=ha, va="center",
                   fontsize=fs, color=S.INK, linespacing=1.35, zorder=3,
                   bbox=dict(boxstyle="round,pad=0.42,rounding_size=0.25",
                             facecolor=fc, edgecolor=ec, linewidth=0.8))


def extent(fig, ax, t):
    """(x0, y0, x1, y1) of a text's bbox patch, in axes coordinates."""
    r = fig.canvas.get_renderer()
    bb = t.get_bbox_patch().get_window_extent(r) if t.get_bbox_patch() is not None \
        else t.get_window_extent(r)
    bb = bb.transformed(ax.transAxes.inverted())
    return bb.x0, bb.y0, bb.x1, bb.y1


def arrow(ax, p0, p1, color=LINK, lw=0.9, ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, transform=ax.transAxes, arrowstyle="-|>",
                                 mutation_scale=6.5, linewidth=lw, color=color,
                                 linestyle=ls, shrinkA=0, shrinkB=0, zorder=1,
                                 clip_on=False))


def make(outdir):
    S.apply_style()
    fig = plt.figure(figsize=(S.TEXTWIDTH - 0.04, 1.78))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    Y1, Y2, YB = 0.845, 0.480, 0.115

    r1 = [
        node(ax, 0.092, Y1, "64$\\times$64 lens\nimage (DeepLense\nsimulation)"),
        node(ax, 0.292, Y1, "centroid align\n+ 32$\\times$32 crop\n(99.4% of light)"),
        node(ax, 0.520, Y1, "pmf over 1024\npixels $=$ 10 qubits\n"
                            "code $(k_x\\!\\ll\\!5)\\,|\\,k_y$"),
        node(ax, 0.806, Y1, "IQP:  $H^{\\otimes n}\\,e^{\\,i\\,\\Sigma_s\\,\\theta_s Z_s}\\,H^{\\otimes n}$\n"
                            "all 1024 generators $s$\n"
                            "$\\theta=\\theta^{0}+\\Phi^{\\!\\top}\\! z$,  $M=2$–4",
             fc=Q_FC, ec=Q_EC),
    ]
    r2 = [
        node(ax, 0.158, Y2, "exact FWHT simulation\nMMD (flat / paper kernel)\n"
                            "or KL, joint over $N$", fc=Q_FC, ec=Q_EC),
        node(ax, 0.490, Y2, "latent prior (resample /\nGaussian) $\\rightarrow q(x,y)\\times S$\n"
                            "$\\rightarrow$ paste at offset"),
        node(ax, 0.812, Y2, "metrics: PDF–JS,\nConv2D-MMD$^2$, held-out\nprojection corr."),
    ]
    bad = node(ax, 0.255, YB, "paper protocol: fixed Hamming-first latent,\n"
                              "sequential stage 1 / stage 2", fc=BAD_FC, ec=BAD_EC)

    fig.canvas.draw()

    # ---- row 1 and row 2 straight connectors
    for row in (r1, r2):
        for a, b in zip(row[:-1], row[1:]):
            ea, eb = extent(fig, ax, a), extent(fig, ax, b)
            arrow(ax, (ea[2] + 0.004, Y1 if row is r1 else Y2),
                      (eb[0] - 0.004, Y1 if row is r1 else Y2))

    # ---- elbow: right end of row 1 -> left end of row 2
    e_last, e_first = extent(fig, ax, r1[-1]), extent(fig, ax, r2[0])
    xr = min(0.972, e_last[2] + 0.010)
    xl = max(0.020, e_first[0] - 0.012)
    ymid = (e_last[1] + e_first[3]) / 2.0
    ax.plot([e_last[2], xr, xr, xl, xl], [Y1, Y1, ymid, ymid, Y2],
            color=LINK, lw=0.9, solid_joinstyle="round", transform=ax.transAxes,
            clip_on=False, zorder=1)
    arrow(ax, (xl, Y2), (e_first[0] - 0.004, Y2))

    # ---- refuted branch
    e_bad, e_tr = extent(fig, ax, bad), extent(fig, ax, r2[0])
    arrow(ax, (0.158, e_tr[1] - 0.004), (0.158, e_bad[3] + 0.004),
          color=BAD_EC, ls=(0, (2.2, 1.6)), lw=0.8)
    ax.plot([e_bad[2] + 0.030], [YB], marker="x", ms=6.0, mew=1.6,
            color=BAD_EC, transform=ax.transAxes, clip_on=False, zorder=3)
    ax.text(e_bad[2] + 0.055, YB, "latent carries no image identity\n"
                                  "$\\Rightarrow$ output is noise",
            transform=ax.transAxes, ha="left", va="center", fontsize=FS,
            color=BAD_EC, linespacing=1.35)

    return S.save(fig, outdir, "fig_pipeline")


if __name__ == "__main__":
    make(os.path.dirname(os.path.abspath(__file__)))
