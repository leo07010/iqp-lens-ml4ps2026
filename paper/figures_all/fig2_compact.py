#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fig2_compact.pdf -- the body-column version of fig2_mechanism_scaling, panels (a) and (b).

The scaling panel moves out to `figA_scaling.pdf`; what is left is the M sweep and the
sample grid.  Data, arm list and reference levels are imported from
`fig2_mechanism_scaling.py` so the two figures cannot drift.  Nothing is below 7 pt.

Python 3.9 compatible.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

import fig_style as S
import fig2_mechanism_scaling as F2

FIG_W, FIG_H = 5.50, 2.00


def rect(x, y, w, h):
    return [x / FIG_W, y / FIG_H, w / FIG_W, h / FIG_H]


def panel_a(fig, ms, ys, refs):
    ax = fig.add_axes(rect(0.62, 0.44, 2.02, 1.34))
    grey = S.INK_MUTED
    ax.axhline(refs["nearest"], color=grey, lw=0.9, ls=(0, (4, 2)), zorder=1)
    ax.text(17.3, refs["nearest"] + 0.015, "nearest training image", fontsize=7.0,
            color=grey, ha="right", va="bottom", zorder=4)
    ax.axhline(refs["mean_img"], color=S.OKABE_ITO["orange"], lw=0.9, ls=(0, (4, 2)),
               zorder=1)
    ax.text(17.3, refs["mean_img"] + 0.015, "mean train. image", fontsize=7.0,
            color=S.OKABE_ITO["orange"], ha="right", va="bottom", zorder=4)
    ax.axhline(refs["random"], color=S.OKABE_ITO["vermilion"], lw=0.9,
               ls=(0, (1.5, 1.8)), zorder=1)
    ax.text(-0.7, refs["random"] - 0.025, "random dir.", fontsize=7.0,
            color=S.OKABE_ITO["vermilion"], ha="left", va="top", zorder=4)
    ax.plot(ms, ys, "-o", color=S.OKABE_ITO["blue"], lw=1.4, ms=4.2, zorder=5,
            mec="white", mew=0.5)
    ax.set_xlabel("latent directions $M$")
    ax.set_ylabel("held-out\nprojection corr.", linespacing=1.2)
    ax.set_xticks([0, 2, 4, 6, 8, 16])
    ax.set_xlim(-1.1, 17.6); ax.set_ylim(-0.05, 1.12)
    ax.set_yticks([0, 0.5, 1.0])
    S.tidy(ax)


def panel_b(fig, rows):
    cell, gap, x0, ytop = 0.40, 0.028, 3.66, 1.90
    ncol, nrow = F2.NCOL, len(rows)
    im = None
    for i, (lab, imgs) in enumerate(rows):
        y = ytop - (i + 1) * cell - i * gap
        for j in range(ncol):
            ax = fig.add_axes(rect(x0 + j * (cell + gap), y, cell, cell))
            im = ax.imshow(imgs[j], cmap=S.IMG_CMAP, vmin=S.IMG_VMIN, vmax=S.IMG_VMAX,
                           interpolation="nearest", origin="lower")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.4); sp.set_color("#b8b8b8")
        fig.text((x0 - 0.06) / FIG_W, (y + cell / 2.0) / FIG_H, lab, fontsize=7.0,
                 ha="right", va="center", linespacing=1.25)
    h_all = nrow * cell + (nrow - 1) * gap
    cax = fig.add_axes(rect(x0 + ncol * (cell + gap) + 0.02, ytop - h_all, 0.075, h_all))
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("pixel intensity (a.u.)", fontsize=7.0, labelpad=2)
    cb.set_ticks([0, 0.5, 1.0])
    cb.ax.tick_params(labelsize=7.0, width=0.5, length=2, pad=1)
    cb.outline.set_linewidth(0.4)


def make(outdir):
    S.apply_style()
    ms, ys, refs = F2.load_mech()
    rows = F2.load_images()
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    panel_a(fig, ms, ys, refs)
    panel_b(fig, rows)
    fig.text(0.06 / FIG_W, 1.90 / FIG_H, "(a)", fontsize=9, fontweight="bold",
             ha="left", va="bottom")
    fig.text(2.86 / FIG_W, 1.90 / FIG_H, "(b)", fontsize=9, fontweight="bold",
             ha="left", va="bottom")
    return S.save(fig, outdir, "fig2_compact")


if __name__ == "__main__":
    make(os.path.dirname(os.path.abspath(__file__)))
