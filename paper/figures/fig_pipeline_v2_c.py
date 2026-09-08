#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fig_pipeline_v2_c.pdf -- fig_pipeline_v2 at its native body size (0.92\\textwidth).

Same seven panels, same thumbnails and same numbers as `fig_pipeline_v2.py` -- its `load()`
is imported so the two cannot drift -- re-laid-out for a 5.06 in column so that 7 pt text
survives with no scaling.  Compared with the full-size version the thumbnails shrink from
0.66 to 0.58 in, the circuit from 4.92 to 3.74 in, and the three metric minis lose their
axis decoration (the Conv2D-MMD^2 bars carry rotated in-panel labels instead of a y axis).

Python 3.9 compatible.
"""
from __future__ import annotations

import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                              # noqa: E402
from matplotlib.patches import (FancyArrowPatch, FancyBboxPatch, Rectangle,  # noqa: E402
                                Arc)

import fig_style as S
import fig_pipeline_v2 as V

FIG_W, FIG_H = 5.06, 3.30
BLUE, BAD, GOOD, LINK = V.BLUE, V.BAD, V.GOOD, V.LINK
FS = 7.0
KX, KY = V.KX, V.KY


def rect(x, y, w, h):
    return [x / FIG_W, y / FIG_H, w / FIG_W, h / FIG_H]


def fx(x):
    return x / FIG_W


def fy(y):
    return y / FIG_H


def thumb(fig, x, y, w, img, caption=None, zorder=3):
    ax = fig.add_axes(rect(x, y, w, w))
    ax.set_zorder(zorder)
    ax.imshow(np.asarray(img, float), cmap=S.IMG_CMAP, vmin=S.IMG_VMIN, vmax=S.IMG_VMAX,
              interpolation="nearest", origin="lower")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.5); sp.set_color("#b8b8b8")
    if caption:
        fig.text(fx(x + w / 2.0), fy(y - 0.035), caption, fontsize=FS, ha="center",
                 va="top", zorder=6)
    return ax


def arrow(fig, p0, p1, color=LINK, lw=1.0, ls="-", ms=6.5, rad=0.0):
    fig.patches.append(FancyArrowPatch(
        (fx(p0[0]), fy(p0[1])), (fx(p1[0]), fy(p1[1])), transform=fig.transFigure,
        arrowstyle="-|>", mutation_scale=ms, linewidth=lw, color=color, linestyle=ls,
        shrinkA=0, shrinkB=0, zorder=2, connectionstyle="arc3,rad=%.3f" % rad))


def label(fig, txt, x, y):
    fig.text(fx(x), fy(y), txt, fontsize=9, fontweight="bold", ha="left", va="bottom",
             zorder=7)


# --------------------------------------------------------------------------- row 1
def row1(fig, D):
    yt, w, pitch = 2.58, 0.58, 0.78
    ax = thumb(fig, 0.28, yt, w, D["img"], "held-out")
    ax.plot([D["cx"]], [D["cy"]], marker="+", ms=5.0, mew=1.0, color="white", zorder=5)
    ax.add_patch(Rectangle((D["dx"] - 0.5, D["dy"] - 0.5), 32, 32, fill=False,
                           edgecolor="white", lw=0.8, ls=(0, (2.2, 1.6)), zorder=4))
    arrow(fig, (0.28 + w + 0.03, yt + w / 2), (0.28 + pitch - 0.03, yt + w / 2))
    thumb(fig, 0.28 + pitch, yt, w, D["crop"], "32$\\times$32 crop")
    arrow(fig, (0.28 + pitch + w + 0.03, yt + w / 2), (0.28 + 2 * pitch - 0.03, yt + w / 2))
    axp = thumb(fig, 0.28 + 2 * pitch, yt, w, D["pmf"] / D["pmf"].max(), "1024-code pmf")
    for t in (7.5, 15.5, 23.5):
        axp.axhline(t, color="white", lw=0.4, alpha=0.5)
        axp.axvline(t, color="white", lw=0.4, alpha=0.5)
    axp.add_patch(Rectangle((KX - 0.5, KY - 0.5), 1, 1, fill=False, edgecolor="white",
                            lw=0.9, zorder=5))
    axp.annotate("", xy=(KX + 0.7, KY + 0.7), xytext=(KX + 8, KY + 8),
                 arrowprops=dict(arrowstyle="-", lw=0.55, color="white"), zorder=5)

    fig.text(fx(2.62), fy(yt + w / 2),
             "pixel $(k_x, k_y) = (%d, %d)$\n"
             "$\\Rightarrow$ code $= (k_x{\\ll}5)\\,|\\,k_y$\n"
             "$\\quad = %d = %s_2$\n"
             "(bit $j$ $\\to$ qubit $j$)" % (KX, KY, D["code"], format(D["code"], "010b")),
             fontsize=FS, ha="left", va="center", linespacing=1.35)

    axz = fig.add_axes(rect(4.36, yt + 0.05, 0.50, 0.50))
    z = D["z"]
    axz.plot(z[:, 0], z[:, 1], ls="none", marker="o", ms=1.5, color=BLUE, alpha=0.7, mew=0)
    axz.set_xticks([]); axz.set_yticks([])
    for sp in axz.spines.values():
        sp.set_linewidth(0.5); sp.set_color("#b8b8b8")
    axz.set_xlabel("$z_1$", fontsize=FS, labelpad=0.5)
    axz.set_ylabel("$z_2$", fontsize=FS, labelpad=0.5)
    axz.text(0.05, 0.95, "$M = 4$", transform=axz.transAxes, fontsize=FS, ha="left",
             va="top")
    fig.text(fx(4.61), fy(3.18), "$\\theta = \\theta^{0} + \\Phi^{\\!\\top} z$",
             fontsize=FS, ha="center", va="bottom")
    label(fig, "(a)", 0.05, 3.18)
    label(fig, "(b)", 1.02, 3.18)
    label(fig, "(c)", 3.92, 3.18)


# --------------------------------------------------------------------------- row 2
def row2(fig, D):
    ax = fig.add_axes(rect(0.40, 1.68, 3.74, 0.70))
    ax.set_zorder(3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ys = [0.90, 0.75, 0.60, 0.24, 0.09]
    names = ["$q_0$", "$q_1$", "$q_2$", "$q_8$", "$q_9$"]
    x0, x1 = 0.070, 0.805
    for y, nm in zip(ys, names):
        ax.plot([x0, x1], [y, y], color=S.INK, lw=0.7, zorder=1)
        ax.text(x0 - 0.010, y, nm, fontsize=FS, ha="right", va="center")
    ax.text(-0.045, 0.495, "$n = 10$", fontsize=FS, ha="center", va="center", rotation=90)

    def hlayer(xc):
        for y in ys:
            ax.add_patch(FancyBboxPatch((xc - 0.024, y - 0.058), 0.048, 0.116,
                                        boxstyle="round,pad=0.003,rounding_size=0.008",
                                        fc="white", ec=S.INK, lw=0.7, zorder=3))
            ax.text(xc, y, "$H$", fontsize=FS, ha="center", va="center", zorder=4)
        ax.text(xc, 0.42, "$\\vdots$", fontsize=8, ha="center", va="center", zorder=4)

    hlayer(0.125)
    bx0, bx1 = 0.215, 0.590
    ax.add_patch(FancyBboxPatch((bx0, 0.015), bx1 - bx0, 0.945,
                                boxstyle="round,pad=0.003,rounding_size=0.010",
                                fc="#e3eef8", ec=BLUE, lw=1.0, zorder=2))
    ax.text(0.5 * (bx0 + bx1), 0.64, "$\\exp\\!\\left(i\\sum_{s}\\theta_s Z_s\\right)$",
            fontsize=8.5, ha="center", va="center", zorder=4)
    ax.text(0.5 * (bx0 + bx1), 0.23, "1024 generators,\nall subsets $s$",
            fontsize=FS, ha="center", va="center", linespacing=1.25, zorder=4)
    hlayer(0.665)
    for y in ys:
        mx = 0.760
        ax.add_patch(Rectangle((mx - 0.024, y - 0.058), 0.048, 0.116, fill=False,
                               edgecolor=S.INK, lw=0.7, zorder=3))
        ax.add_patch(Arc((mx, y - 0.030), 0.030, 0.095, theta1=15, theta2=165,
                         color=S.INK, lw=0.6, zorder=4))
        ax.plot([mx, mx + 0.013], [y - 0.030, y + 0.032], color=S.INK, lw=0.6, zorder=4)
    ax.text(0.760, 0.42, "$\\vdots$", fontsize=8, ha="center", va="center")
    ax.text(0.825, 0.495, "$x \\in \\{0,1\\}^{10}$", fontsize=FS, ha="left", va="center")

    arrow(fig, (4.61, 2.57), (4.61, 2.47), color=BLUE, lw=0.9)
    arrow(fig, (4.61, 2.47), (2.00, 2.42), color=BLUE, lw=0.9, rad=0.10)
    label(fig, "(d)", 0.05, 2.44)


# --------------------------------------------------------------------------- row 3
def row3(fig, D):
    fig.patches.append(FancyBboxPatch(
        (fx(0.24), fy(0.82)), fx(1.30), fy(0.70),
        boxstyle="round,pad=0.003,rounding_size=0.010", transform=fig.transFigure,
        fc="#e3eef8", ec=BLUE, lw=0.9, zorder=1))
    fig.text(fx(0.89), fy(1.17),
             "exact FWHT simulation\n"
             "$q_\\theta = \\left|\\mathcal{W}e^{i\\theta}\\right|^{2}\\!/\\,2^{n}$\n"
             "MMD (flat / reference\nkernel) or KL, joint\nover $N$ images",
             fontsize=FS, ha="center", va="center", linespacing=1.35, zorder=3)
    arrow(fig, (1.05, 1.53), (1.34, 1.66), color=BLUE, lw=0.9, ls=(0, (2.2, 1.5)),
          rad=-0.30)
    fig.text(fx(0.28), fy(1.53), "update $\\theta^{0}, \\Phi, z$", fontsize=FS,
             ha="left", va="bottom", color=BLUE)
    label(fig, "(e)", 0.05, 1.53)

    thumb(fig, 1.70, 0.92, 0.44, D["q_scaled"], "$q(x,y)\\times S$")
    arrow(fig, (2.18, 1.12), (2.62, 1.12))
    fig.text(fx(2.40), fy(1.18), "$\\times S$,\npaste at\noffset", fontsize=FS,
             ha="center", va="bottom", linespacing=1.25)
    thumb(fig, 2.66, 0.90, 0.50, D["gen"], "generated")
    label(fig, "(f)", 1.66, 1.53)


def row3_metrics(fig, D):
    y0, h, w = 0.90, 0.40, 0.38
    xs = [3.40, 3.92, 4.44]
    meta = D["meta"]
    ttl = fy(y0 + h + 0.03)

    ax = fig.add_axes(rect(xs[0], y0, w, h)); ax.set_zorder(3)
    x = D["holdout_all"].ravel(); y = D["gen_all"].ravel()
    lo, hi = float(min(x.min(), y.min())), float(max(x.max(), y.max()))
    edges = np.linspace(lo, hi, 51)
    ctr = 0.5 * (edges[1:] + edges[:-1])
    for v, c in ((x, S.INK_MUTED), (y, BLUE)):
        p, _ = np.histogram(v, bins=edges)
        ax.step(ctr, p.astype(float) / p.sum() + 1e-9, where="mid", color=c, lw=0.9)
    ax.set_yscale("log"); ax.set_xlim(lo, 0.55); ax.set_ylim(2e-6, 6.0)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.text(0.96, 0.97, "real", transform=ax.transAxes, fontsize=FS, ha="right",
            va="top", color=S.INK_MUTED)
    ax.text(0.96, 0.62, "IQP", transform=ax.transAxes, fontsize=FS, ha="right",
            va="top", color=BLUE)
    fig.text(fx(xs[0] + w / 2), ttl, "PDF–JS\n$0.0024$", fontsize=FS, ha="center",
             va="bottom", linespacing=1.2)

    ax = fig.add_axes(rect(xs[1], y0, w, h)); ax.set_zorder(3)
    names, vals = ["IQP", "RBM", "GAN"], [0.0861, 0.0752, 0.0371]
    cols = [BLUE, BAD, GOOD]
    ax.bar([0, 1, 2], vals, color=cols, width=0.62, lw=0)
    for i, (nm, v) in enumerate(zip(names, vals)):
        ax.text(i, v + 0.008, nm, fontsize=FS, ha="center", va="bottom", rotation=90,
                color=cols[i])
    ax.set_xticks([]); ax.set_yticks([]); ax.set_ylim(0, 0.205)
    ax.set_xlim(-0.7, 2.7)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    fig.text(fx(xs[1] + w / 2), ttl, "Conv2D\nMMD$^2$", fontsize=FS, ha="center",
             va="bottom", linespacing=1.2)

    ax = fig.add_axes(rect(xs[2], y0, w, h)); ax.set_zorder(3)
    pc = np.asarray(meta["projection_corrs"], float)
    ax.hist(pc, bins=18, range=(0.70, 1.0), color=BLUE, lw=0)
    ax.axvline(float(meta["ref_projection_mean_crop_corr"]), color=S.INK_MUTED, lw=0.8,
               ls=(0, (2.2, 1.6)))
    ax.axvline(float(meta["ref_projection_nearest_train_corr"]), color=BAD, lw=0.8,
               ls=(0, (2.2, 1.6)))
    ax.set_xlim(0.70, 1.01); ax.set_yticks([]); ax.set_xticks([0.8, 1.0])
    ax.tick_params(axis="x", labelsize=FS, pad=1, length=2)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    fig.text(fx(xs[2] + w / 2), ttl, "projection\n$%.3f$" % float(np.mean(pc)),
             fontsize=FS, ha="center", va="bottom", linespacing=1.2)
    label(fig, "(g)", 3.18, 1.55)


# --------------------------------------------------------------------------- row 4
def row4(fig, D):
    fig.patches.append(FancyBboxPatch(
        (fx(0.24), fy(0.06)), fx(4.66), fy(0.66),
        boxstyle="round,pad=0.003,rounding_size=0.010", transform=fig.transFigure,
        fc="#f5f7f8", ec="#ccd3d8", lw=0.7, zorder=0))
    thumb(fig, 0.32, 0.11, 0.50, D["ref_noise"], None, zorder=4)
    fig.text(fx(0.90), fy(0.39),
             "reference protocol (fixed\nlow-variance latent,\nsequential stage 1 / 2)",
             fontsize=FS, ha="left", va="center", linespacing=1.35, color=BAD, zorder=5)
    fig.text(fx(2.46), fy(0.39), "$\\times$", fontsize=11, ha="center", va="center",
             color=BAD, zorder=5)
    fig.text(fx(2.72), fy(0.39), "$\\checkmark$", fontsize=10, ha="center", va="center",
             color=GOOD, zorder=5)
    fig.text(fx(2.88), fy(0.39),
             "this work (learned\ndirections, $M = 4$,\njoint over $N$ images)",
             fontsize=FS, ha="left", va="center", linespacing=1.35, color=GOOD, zorder=5)
    thumb(fig, 4.30, 0.11, 0.50, D["gen"], None, zorder=4)


def make(outdir):
    S.apply_style()
    D = V.load()
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    row1(fig, D); row2(fig, D); row3(fig, D); row3_metrics(fig, D); row4(fig, D)
    return S.save(fig, outdir, "fig_pipeline_v2_c")


if __name__ == "__main__":
    make(os.path.dirname(os.path.abspath(__file__)))
