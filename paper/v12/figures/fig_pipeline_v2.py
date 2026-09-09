#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fig_pipeline_v2.pdf -- the illustrated method figure.

Unlike `fig_pipeline.py` (plain boxes) every stage here carries a real thumbnail rendered
from the actual arrays: the held-out frame with its centroid and crop window, the aligned
crop, the crop as a 1024-code pmf, the learned latent cloud, the trained model's own
q(x, y), the pasted 64x64 sample, and the three evaluation metrics recomputed from those
same arrays.  The circuit is hand-drawn (no pennylane dependency).  Every axes is placed in
inches so nothing can overlap.

Python 3.9 compatible.  CPU, a few seconds.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                              # noqa: E402
from matplotlib.patches import (FancyArrowPatch, FancyBboxPatch, Rectangle,  # noqa: E402
                                Arc)

import fig_style as S

EXP = "<HOME>/gsoc2026-deeplense-qml/experiments/13_iqp_turbulence_replication"
ARM = os.path.join(EXP, "outputs_lens_r3b", "iqp_m4_mmdflat_resample_n100")
REF = os.path.join(EXP, "outputs_lens_r3", "iqp_paper_n100")

FIG_W, FIG_H = 5.50, 3.58
HOLDOUT_IDX = 55      # most azimuthally complete ring among held-out[:60]
GEN_IDX = 43          # most azimuthally complete ring among this arm's samples[:60]
REF_IDX = 0           # reference-protocol sample (all of them are noise)
KX, KY = 12, 20       # the pixel whose 10-bit code is spelled out in panel (b)

BLUE = S.OKABE_ITO["blue"]
BAD = S.OKABE_ITO["vermilion"]
GOOD = S.OKABE_ITO["green"]
LINK = "#5a6b78"
FS = 7.0
FSS = 7.0             # secondary text -- nothing in this figure is below 7 pt


def rect(x, y, w, h):
    return [x / FIG_W, y / FIG_H, w / FIG_W, h / FIG_H]


def fx(x):
    return x / FIG_W


def fy(y):
    return y / FIG_H


def thumb(fig, x, y, w, img, caption=None, cap_dy=0.035, zorder=3):
    ax = fig.add_axes(rect(x, y, w, w))
    ax.set_zorder(zorder)
    ax.imshow(np.asarray(img, float), cmap=S.IMG_CMAP, vmin=S.IMG_VMIN, vmax=S.IMG_VMAX,
              interpolation="nearest", origin="lower")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.5); sp.set_color("#b8b8b8")
    if caption:
        fig.text(fx(x + w / 2.0), fy(y - cap_dy), caption, fontsize=FSS, ha="center",
                 va="top", linespacing=1.2, zorder=6)
    return ax


def arrow(fig, p0, p1, color=LINK, lw=1.0, ls="-", ms=7, rad=0.0):
    fig.patches.append(FancyArrowPatch(
        (fx(p0[0]), fy(p0[1])), (fx(p1[0]), fy(p1[1])), transform=fig.transFigure,
        arrowstyle="-|>", mutation_scale=ms, linewidth=lw, color=color, linestyle=ls,
        shrinkA=0, shrinkB=0, zorder=2,
        connectionstyle="arc3,rad=%.3f" % rad))


def label(fig, txt, x, y):
    fig.text(fx(x), fy(y), txt, fontsize=9, fontweight="bold", ha="left", va="bottom",
             zorder=7)


# --------------------------------------------------------------------------------- data
def load():
    sys.path.insert(0, EXP)
    import data_lens_r3, codec10                              # noqa: E402
    d = data_lens_r3.load_r3(100, n_holdout=200)
    img = np.asarray(d.holdout_images[HOLDOUT_IDX], float)
    crop, dy, dx = data_lens_r3.align_crop(img)
    cy, cx = data_lens_r3.centroid(img)
    crop = np.asarray(crop, float)
    return dict(
        img=img, crop=crop, cy=cy, cx=cx, dy=dy, dx=dx,
        pmf=crop / crop.sum(),
        z=np.load(os.path.join(ARM, "z_train.npy")),
        q_scaled=np.asarray(np.load(os.path.join(ARM, "crops",
                                                 "generated_%03d.npy" % GEN_IDX)), float),
        gen=np.asarray(np.load(os.path.join(ARM, "generated_%03d.npy" % GEN_IDX)), float),
        ref_noise=np.asarray(np.load(os.path.join(REF,
                                                  "generated_%03d.npy" % REF_IDX)), float),
        meta=json.load(open(os.path.join(ARM, "meta.json"))),
        holdout_all=np.asarray(d.holdout_images, np.float64),
        gen_all=np.stack([np.load(os.path.join(ARM, "generated_%03d.npy" % i))
                          for i in range(200)]).astype(np.float64),
        code=int(codec10.encode(KX, KY)), S=d.S)


# ------------------------------------------------------------------ row 1: encode
def row1(fig, D):
    yt, w, pitch = 2.80, 0.66, 0.89
    ax = thumb(fig, 0.30, yt, w, D["img"], "held-out 64$\\times$64")
    ax.plot([D["cx"]], [D["cy"]], marker="+", ms=5.5, mew=1.1, color="white", zorder=5)
    ax.add_patch(Rectangle((D["dx"] - 0.5, D["dy"] - 0.5), 32, 32, fill=False,
                           edgecolor="white", lw=0.9, ls=(0, (2.4, 1.7)), zorder=4))
    arrow(fig, (0.30 + w + 0.035, yt + w / 2), (0.30 + pitch - 0.035, yt + w / 2))

    thumb(fig, 0.30 + pitch, yt, w, D["crop"], "32$\\times$32 crop")
    arrow(fig, (0.30 + pitch + w + 0.035, yt + w / 2),
               (0.30 + 2 * pitch - 0.035, yt + w / 2))

    axp = thumb(fig, 0.30 + 2 * pitch, yt, w, D["pmf"] / D["pmf"].max(),
                "1024-code pmf")
    for t in (7.5, 15.5, 23.5):
        axp.axhline(t, color="white", lw=0.45, alpha=0.5)
        axp.axvline(t, color="white", lw=0.45, alpha=0.5)
    axp.add_patch(Rectangle((KX - 0.5, KY - 0.5), 1, 1, fill=False, edgecolor="white",
                            lw=1.0, zorder=5))
    axp.annotate("", xy=(KX + 0.7, KY + 0.7), xytext=(KX + 8, KY + 8),
                 arrowprops=dict(arrowstyle="-", lw=0.6, color="white"), zorder=5)

    fig.text(fx(2.86), fy(yt + w / 2),
             "pixel $(k_x, k_y) = (%d, %d)$\n"
             "$\\Rightarrow$ code $= (k_x{\\ll}5)\\,|\\,k_y$\n"
             "$\\quad = %d = %s_2$\n"
             "(bit $j$ $\\to$ qubit $j$)"
             % (KX, KY, D["code"], format(D["code"], "010b")),
             fontsize=FSS, ha="left", va="center", linespacing=1.35)

    axz = fig.add_axes(rect(4.63, yt + 0.05, 0.56, 0.56))
    z = D["z"]
    axz.plot(z[:, 0], z[:, 1], ls="none", marker="o", ms=1.7, color=BLUE, alpha=0.7, mew=0)
    axz.set_xticks([]); axz.set_yticks([])
    for sp in axz.spines.values():
        sp.set_linewidth(0.5); sp.set_color("#b8b8b8")
    axz.set_xlabel("$z_1$", fontsize=FSS, labelpad=0.5)
    axz.set_ylabel("$z_2$", fontsize=FSS, labelpad=0.5)
    fig.text(fx(4.91), fy(3.47), "$\\theta = \\theta^{0} + \\Phi^{\\!\\top} z$",
             fontsize=FS, ha="center", va="bottom")
    axz.text(0.05, 0.95, "$M = 4$", transform=axz.transAxes, fontsize=FSS,
             ha="left", va="top")
    label(fig, "(a)", 0.06, 3.47)
    label(fig, "(b)", 1.13, 3.47)
    label(fig, "(c)", 4.26, 3.47)


# ------------------------------------------------------------------ row 2: the circuit
def row2(fig, D):
    ax = fig.add_axes(rect(0.44, 1.80, 4.92, 0.76))
    ax.set_zorder(3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ys = [0.90, 0.75, 0.60, 0.24, 0.09]
    names = ["$q_0$", "$q_1$", "$q_2$", "$q_8$", "$q_9$"]
    x0, x1 = 0.055, 0.805
    for y, nm in zip(ys, names):
        ax.plot([x0, x1], [y, y], color=S.INK, lw=0.7, zorder=1)
        ax.text(x0 - 0.008, y, nm, fontsize=FSS, ha="right", va="center")
    ax.text(-0.028, 0.495, "$n = 10$", fontsize=FS, ha="center", va="center", rotation=90)

    def hlayer(xc):
        for y in ys:
            ax.add_patch(FancyBboxPatch((xc - 0.019, y - 0.058), 0.038, 0.116,
                                        boxstyle="round,pad=0.003,rounding_size=0.007",
                                        fc="white", ec=S.INK, lw=0.7, zorder=3))
            ax.text(xc, y, "$H$", fontsize=FSS, ha="center", va="center", zorder=4)
        ax.text(xc, 0.42, "$\\vdots$", fontsize=8, ha="center", va="center", zorder=4)

    hlayer(0.105)
    bx0, bx1 = 0.185, 0.585
    ax.add_patch(FancyBboxPatch((bx0, 0.015), bx1 - bx0, 0.945,
                                boxstyle="round,pad=0.003,rounding_size=0.010",
                                fc="#e3eef8", ec=BLUE, lw=1.0, zorder=2))
    ax.text(0.5 * (bx0 + bx1), 0.63,
            "$\\exp\\!\\left(i\\sum_{s}\\theta_s Z_s\\right)$",
            fontsize=9, ha="center", va="center", zorder=4)
    ax.text(0.5 * (bx0 + bx1), 0.24,
            "diagonal in $Z$, 1024 generators,\nall subsets $s \\subseteq \\{0,\\dots,9\\}$",
            fontsize=FSS, ha="center", va="center", linespacing=1.25, zorder=4)
    hlayer(0.665)
    for y in ys:
        mx = 0.755
        ax.add_patch(Rectangle((mx - 0.019, y - 0.058), 0.038, 0.116, fill=False,
                               edgecolor=S.INK, lw=0.7, zorder=3, facecolor="white"))
        ax.add_patch(Arc((mx, y - 0.030), 0.024, 0.095, theta1=15, theta2=165,
                         color=S.INK, lw=0.6, zorder=4))
        ax.plot([mx, mx + 0.010], [y - 0.030, y + 0.032], color=S.INK, lw=0.6, zorder=4)
    ax.text(0.755, 0.42, "$\\vdots$", fontsize=8, ha="center", va="center")
    ax.text(0.825, 0.495, "$x \\in \\{0,1\\}^{10}$", fontsize=FSS, ha="left",
            va="center")

    # z feeds the phase block
    arrow(fig, (4.91, 2.78), (4.91, 2.66), color=BLUE, lw=0.9)
    arrow(fig, (4.91, 2.66), (2.30, 2.60), color=BLUE, lw=0.9, rad=0.10)
    label(fig, "(d)", 0.06, 2.42)


# ------------------------------------------------------------------ row 3: train, sample
def row3(fig, D):
    fig.patches.append(FancyBboxPatch(
        (fx(0.26), fy(0.94)), fx(1.46), fy(0.68),
        boxstyle="round,pad=0.003,rounding_size=0.010", transform=fig.transFigure,
        fc="#e3eef8", ec=BLUE, lw=0.9, zorder=1))
    fig.text(fx(0.99), fy(1.28),
             "exact FWHT simulation\n"
             "$q_\\theta = \\left|\\mathcal{W}e^{i\\theta}\\right|^{2}\\!/\\,2^{n}$\n"
             "MMD (flat / reference\nkernel) or KL, joint\nover $N$ images",
             fontsize=FSS, ha="center", va="center", linespacing=1.35, zorder=3)
    arrow(fig, (1.16, 1.64), (1.56, 1.78), color=BLUE, lw=0.9, ls=(0, (2.4, 1.6)),
          rad=-0.30)
    fig.text(fx(0.30), fy(1.67), "update $\\theta^{0}, \\Phi, z$", fontsize=FSS,
             ha="left", va="bottom", color=BLUE)
    label(fig, "(e)", 0.06, 1.66)

    thumb(fig, 1.92, 1.06, 0.48, D["q_scaled"], "$q(x,y)\\times S$")
    arrow(fig, (2.44, 1.30), (3.04, 1.30))
    fig.text(fx(2.74), fy(1.36), "$\\times S$,\npaste at\noffset", fontsize=FSS,
             ha="center", va="bottom", linespacing=1.25)
    thumb(fig, 3.10, 1.02, 0.54, D["gen"], "generated")
    label(fig, "(f)", 1.88, 1.66)


# ------------------------------------------------------------------ row 3 right: metrics
def row3_metrics(fig, D):
    y0, h = 1.00, 0.42
    xs = [3.72, 4.46, 4.98]
    ws = [0.44, 0.46, 0.40]
    meta = D["meta"]
    ttl = fy(y0 + h + 0.03)

    ax = fig.add_axes(rect(xs[0], y0, ws[0], h)); ax.set_zorder(3)
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
    ax.text(0.96, 0.97, "real", transform=ax.transAxes, fontsize=FSS, ha="right",
            va="top", color=S.INK_MUTED)
    ax.text(0.96, 0.79, "IQP", transform=ax.transAxes, fontsize=FSS, ha="right",
            va="top", color=BLUE)
    fig.text(fx(xs[0] + ws[0] / 2), ttl, "PDF–JS\n$0.0024$", fontsize=FSS, ha="center",
             va="bottom", linespacing=1.2)

    ax = fig.add_axes(rect(xs[1], y0, ws[1], h)); ax.set_zorder(3)
    names, vals = ["IQP", "RBM", "GAN"], [0.0861, 0.0752, 0.0371]
    ax.barh([2, 1, 0], vals, color=[BLUE, BAD, GOOD], height=0.66, lw=0)
    ax.set_yticks([2, 1, 0]); ax.set_yticklabels(names, fontsize=FSS)
    ax.tick_params(axis="y", length=0, pad=1.5)
    ax.set_xticks([]); ax.set_xlim(0, 0.098); ax.set_ylim(-0.6, 2.6)
    for sp in ("top", "right", "bottom"):
        ax.spines[sp].set_visible(False)
    fig.text(fx(xs[1] + ws[1] / 2), ttl, "Conv2D\nMMD$^2$", fontsize=FSS, ha="center",
             va="bottom", linespacing=1.2)

    ax = fig.add_axes(rect(xs[2], y0, ws[2], h)); ax.set_zorder(3)
    pc = np.asarray(meta["projection_corrs"], float)
    ax.hist(pc, bins=18, range=(0.70, 1.0), color=BLUE, lw=0)
    ax.axvline(float(meta["ref_projection_mean_crop_corr"]), color=S.INK_MUTED, lw=0.8,
               ls=(0, (2.2, 1.6)))
    ax.axvline(float(meta["ref_projection_nearest_train_corr"]), color=BAD, lw=0.8,
               ls=(0, (2.2, 1.6)))
    ax.set_xlim(0.70, 1.01); ax.set_yticks([]); ax.set_xticks([0.8, 1.0])
    ax.tick_params(axis="x", labelsize=FSS, pad=1, length=2)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    fig.text(fx(xs[2] + ws[2] / 2), ttl, "projection\nmean $%.3f$"
             % float(np.mean(pc)), fontsize=FSS, ha="center", va="bottom",
             linespacing=1.2)
    label(fig, "(g)", 3.50, 1.70)


# ------------------------------------------------------------------ row 4: comparison
def row4(fig, D):
    fig.patches.append(FancyBboxPatch(
        (fx(0.26), fy(0.06)), fx(5.16), fy(0.68),
        boxstyle="round,pad=0.003,rounding_size=0.010", transform=fig.transFigure,
        fc="#f5f7f8", ec="#ccd3d8", lw=0.7, zorder=0))
    thumb(fig, 0.36, 0.10, 0.60, D["ref_noise"], None, zorder=4)
    fig.text(fx(1.04), fy(0.40),
             "reference protocol (fixed\nlow-variance latent,\nsequential stage 1 / 2)",
             fontsize=FSS, ha="left", va="center", linespacing=1.35, color=BAD, zorder=5)
    fig.text(fx(2.62), fy(0.40), "$\\times$", fontsize=12, ha="center", va="center",
             color=BAD, zorder=5)
    fig.text(fx(2.95), fy(0.40), "$\\checkmark$", fontsize=11, ha="center", va="center",
             color=GOOD, zorder=5)
    fig.text(fx(3.12), fy(0.40),
             "this work (learned\ndirections, $M = 4$,\njoint over $N$ images)",
             fontsize=FSS, ha="left", va="center", linespacing=1.35, color=GOOD, zorder=5)
    thumb(fig, 4.72, 0.10, 0.60, D["gen"], None, zorder=4)


def make(outdir):
    S.apply_style()
    D = load()
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    row1(fig, D)
    row2(fig, D)
    row3(fig, D)
    row3_metrics(fig, D)
    row4(fig, D)
    return S.save(fig, outdir, "fig_pipeline_v2")


if __name__ == "__main__":
    make(os.path.dirname(os.path.abspath(__file__)))
