#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""figB_classes_v12.pdf -- figB_classes at its native body size (0.70\\textwidth = 3.85 in).

Same rows, same selection rule and same colour scale as `figB_classes.py` (its `load_rows`
is imported, so the two cannot drift); the layout is placed in inches and the right-hand
group header is shortened so 7 pt text fits without scaling.
Python 3.9 compatible.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

import fig_style as S
import figB_classes as FB
FB.OUT_TREE = "<WORK>/gsoc_iqp_mlpsuite/phase2/s0"
FB.RUN_FMT = "%s_mlp_M2_H32_n100"

FIG_W, FIG_H = 3.85, 1.60
CELL, GAP, GROUP_GAP = 0.40, 0.07, 0.20
X0, YTOP = 0.60, 1.42
LAB_X = 0.55


def rect(x, y, w, h):
    return [x / FIG_W, y / FIG_H, w / FIG_W, h / FIG_H]


def col_x(j):
    """left edge of image column j (0,1 = held-out; 2,3,4 = generated)."""
    x = X0 + j * (CELL + GAP)
    if j >= FB.N_REAL:
        x += GROUP_GAP
    return x


def make(outdir):
    S.apply_style()
    rows = FB.load_rows()
    fig = plt.figure(figsize=(FIG_W, FIG_H))

    im = None
    for i, (label, real, gen) in enumerate(rows):
        y = YTOP - (i + 1) * CELL - i * GAP
        imgs = list(real) + list(gen)
        for j, img in enumerate(imgs):
            ax = fig.add_axes(rect(col_x(j), y, CELL, CELL))
            im = ax.imshow(img, cmap=S.IMG_CMAP, vmin=S.IMG_VMIN, vmax=S.IMG_VMAX,
                           interpolation="nearest", origin="lower")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.4); sp.set_color("#b8b8b8")
        fig.text(LAB_X / FIG_W, (y + CELL / 2.0) / FIG_H, label, fontsize=7.0,
                 ha="right", va="center")

    nrow = len(rows)
    h_all = nrow * CELL + (nrow - 1) * GAP
    x_cb = col_x(FB.N_REAL + FB.N_GEN - 1) + CELL + 0.10
    cax = fig.add_axes(rect(x_cb, YTOP - h_all, 0.085, h_all))
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("intensity (a.u.)", fontsize=7.0, labelpad=2)
    cb.set_ticks([0, 0.5, 1.0])
    cb.ax.tick_params(labelsize=7.0, width=0.5, length=2, pad=1)
    cb.outline.set_linewidth(0.4)

    x_real = 0.5 * (col_x(0) + col_x(FB.N_REAL - 1) + CELL)
    x_gen = 0.5 * (col_x(FB.N_REAL) + col_x(FB.N_REAL + FB.N_GEN - 1) + CELL)
    fig.text(x_real / FIG_W, (YTOP + 0.035) / FIG_H, "held-out (real)", fontsize=7.0,
             ha="center", va="bottom")
    fig.text(x_gen / FIG_W, (YTOP + 0.035) / FIG_H, "generated (this class)",
             fontsize=7.0, ha="center", va="bottom")
    return S.save(fig, outdir, "figB_classes_v12")


if __name__ == "__main__":
    make(os.path.dirname(os.path.abspath(__file__)))
