#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fig1_compact.pdf -- the body-column version of fig1_map_samples.

Same data and same conventions as `fig1_map_samples.py` (whose loaders are imported here so
the two cannot drift), re-laid-out to survive `width=\\linewidth` in a two-column body:
the map keeps a full 2.30 in panel, the dataset legend moves to a thin column beside it, the
sample grid uses 0.33 in cells with one-line row labels, and the colour bar is horizontal
under the grid.  Nothing is below 7 pt.

Python 3.9 compatible.
"""
from __future__ import annotations

import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402
from matplotlib.lines import Line2D                    # noqa: E402

import fig_style as S
import fig1_map_samples as F1

FIG_W, FIG_H = 5.50, 2.30

# fig1_map_samples' legend labels, shortened to fit a 0.95 in column
SHORT = {
    "lens 18-bit": "lens 18b",
    "lens 18-bit, pooled": "lens 18b pool",
    "lens 10-bit pmf": "lens 10b",
    "turbulence D2": "turb. D2",
    "turb. D2, pooled": "D2 pooled",
    "MNIST-16": "MNIST-16",
    "MNIST-784": "MNIST-784",
    "BinaryBlobs": "Blobs",
}
ROW_LABELS = ["held-out", "IQP ref.", "IQP $M{=}4$", "RBM", "DCGAN"]


def rect(x, y, w, h):
    return [x / FIG_W, y / FIG_H, w / FIG_W, h / FIG_H]


def panel_a(fig, pts):
    ax = fig.add_axes(rect(0.48, 0.44, 2.26, 1.70))
    gy, gx = np.mgrid[0:200, 0:200] / 199.0
    alpha = np.clip((gx - 0.30) / 0.70, 0, 1) * np.clip((gy - 0.40) / 0.60, 0, 1)
    wash = np.zeros(gx.shape + (4,))
    wash[..., 0], wash[..., 1], wash[..., 2] = 0.31, 0.55, 0.72
    wash[..., 3] = 0.32 * alpha
    ax.imshow(wash, extent=[0, 1, 0, 1], origin="lower", aspect="auto", zorder=0,
              interpolation="bilinear")
    ax.text(0.975, 0.975, "paper's\n\"IQP-compatible\"\ncorner", transform=ax.transAxes,
            ha="right", va="top", fontsize=7.0, color="#2c5a78", linespacing=1.25,
            zorder=5)

    handles = []
    for r, col, mk, lab, want_null in pts:
        x, y = r["I_QCLI"], r["I_CCI"]
        ms = 8.0 if mk == "*" else 5.0
        if want_null and r.get("null_QCLI") is not None:
            ax.plot([x, r["null_QCLI"]], [y, r["null_CCI"]], "-", color=col, lw=0.6,
                    alpha=0.6, zorder=2)
            ax.plot([r["null_QCLI"]], [r["null_CCI"]], marker=mk, ms=ms * 0.85,
                    mfc="white", mec=col, mew=0.9, ls="none", zorder=3)
        if r.get("perm_QCLI") is not None:
            ax.plot([x, r["perm_QCLI"]], [y, r["perm_CCI"]], ls=(0, (1.2, 1.4)),
                    color=col, lw=0.6, alpha=0.75, zorder=2)
            ax.plot([r["perm_QCLI"]], [r["perm_CCI"]], marker="x", ms=3.4, mew=1.0,
                    color=col, ls="none", zorder=3)
        ax.plot([x], [y], marker=mk, ms=ms, color=col, ls="none", zorder=4, mec="white",
                mew=0.5)
        handles.append(Line2D([], [], marker=mk, ls="none", ms=ms * 0.85, color=col,
                              mec="white", mew=0.4, label=SHORT.get(lab, lab)))

    r10 = [r for r, _c, _m, _l, w in pts if not w][0]
    ax.annotate("saturated at $n\\!=\\!10$:\nnull $=(%.2f,\\ %.2f)$"
                % (r10["null_QCLI"], r10["null_CCI"]),
                (r10["I_QCLI"], r10["I_CCI"]), textcoords="offset points",
                xytext=(0, -11), fontsize=7.0, ha="center", va="top",
                color=S.OKABE_ITO["purple"], linespacing=1.2, zorder=5)

    ax.set_xlim(-0.035, 1.05); ax.set_ylim(-0.05, 1.10)
    ax.set_xlabel(r"$I_{\mathrm{QCLI}}$  (0–1)")
    ax.set_ylabel(r"$I_{\mathrm{CCI}}$  (0–1)")
    ax.set_xticks([0, 0.5, 1.0]); ax.set_yticks([0, 0.5, 1.0])
    S.tidy(ax)
    return handles


def legend_column(fig, handles):
    ax = fig.add_axes(rect(2.80, 0.44, 0.95, 1.70))
    ax.axis("off")
    conv = [
        Line2D([], [], marker="o", ls="none", ms=4.4, color=S.INK_MUTED, mec="white",
               mew=0.4, label="dataset"),
        Line2D([], [], marker="o", ls="none", ms=3.8, mfc="white", mec=S.INK_MUTED,
               mew=0.9, label="null"),
        Line2D([], [], marker="x", ls="none", ms=3.4, mew=1.0, color=S.INK_MUTED,
               label="perm. null"),
    ]
    ax.legend(handles=handles + conv, loc="center left", bbox_to_anchor=(-0.06, 0.5),
              fontsize=7.0, handletextpad=0.35, labelspacing=0.42, borderpad=0.0,
              borderaxespad=0.0, frameon=False)


def panel_b(fig, rows):
    cell, gap, x0 = 0.33, 0.025, 4.34
    ncol, nrow = F1.NCOL, len(rows)
    ytop = 2.14
    im = None
    for i, (_lab, imgs) in enumerate(rows):
        y = ytop - (i + 1) * cell - i * gap
        for j in range(ncol):
            ax = fig.add_axes(rect(x0 + j * (cell + gap), y, cell, cell))
            im = ax.imshow(imgs[j], cmap=S.IMG_CMAP, vmin=S.IMG_VMIN, vmax=S.IMG_VMAX,
                           interpolation="nearest", origin="lower")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.4); sp.set_color("#b8b8b8")
        fig.text((x0 - 0.056) / FIG_W, (y + cell / 2.0) / FIG_H, ROW_LABELS[i],
                 fontsize=7.0, ha="right", va="center")
    w_all = ncol * cell + (ncol - 1) * gap
    ybot = ytop - nrow * cell - (nrow - 1) * gap
    cax = fig.add_axes(rect(x0, ybot - 0.115, w_all, 0.062))
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("intensity (a.u.)", fontsize=7.0, labelpad=1.5)
    cb.set_ticks([0, 0.5, 1.0])
    cb.ax.tick_params(labelsize=7.0, width=0.5, length=2, pad=1)
    cb.outline.set_linewidth(0.4)


def make(outdir):
    S.apply_style()
    pts, _raw = F1.load_map()
    rows = F1.load_images()
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    handles = panel_a(fig, pts)
    legend_column(fig, handles)
    panel_b(fig, rows)
    fig.text(0.06 / FIG_W, 2.145 / FIG_H, "(a)", fontsize=9, fontweight="bold",
             ha="left", va="bottom")
    fig.text(3.80 / FIG_W, 2.145 / FIG_H, "(b)", fontsize=9, fontweight="bold",
             ha="left", va="bottom")
    return S.save(fig, outdir, "fig1_compact")


if __name__ == "__main__":
    make(os.path.dirname(os.path.abspath(__file__)))
