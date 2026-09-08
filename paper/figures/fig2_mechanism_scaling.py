#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fig2_mechanism_scaling.pdf

(a) held-out projection correlation vs the number of learned latent directions M
    (controlled mechanism test, mech_tests/mech_test2, N = 20 training images), against
    three reference levels: random directions, the mean training image, and the nearest
    training image (the memorisation ceiling).
(b) held-out lens images and samples from the M = 2 / 4 / 8 affine-latent arms of round 3b
    (N = 100), one linear magma scale, vmin = 0, vmax = 1.
(c) sample quality vs training-set size: Conv2D-MMD^2 (solid) and its flux-aligned
    variant (dashed) for the IQP arm and the two paper baselines, against the held-out
    split-half floor.

Axes are placed in inches so that no label can run into a neighbouring panel.
Python 3.9 compatible.  Reads only committed artefacts.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402
from matplotlib.lines import Line2D                    # noqa: E402

import fig_style as S

EXP = "<HOME>/gsoc2026-deeplense-qml/experiments/13_iqp_turbulence_replication"
MECH = os.path.join(EXP, "mech_tests", "mech_test2")

FIG_W, FIG_H = 5.46, 3.30

# ---- panel (a): (M, held-out projection corr) with the file each value comes from.
#      M = 8 / 16 take the best over the learning-rate sweep -- the value most favourable
#      to the model -- so the collapse claim is the conservative one.
M_SWEEP = [
    (0,  "results.json", ("configs", "KL_M0", "holdout", "corr_mean")),
    (2,  "diag3.json",   ("KL_M2", "holdout", "corr_mean")),
    (4,  "results.json", ("configs", "KL_M4", "holdout", "corr_mean")),
    (6,  "diag3.json",   ("KL_M6", "holdout", "corr_mean")),
    (8,  "diag.json",    ("exp2_redo", "M8", "best", "corr_mean")),
    (16, "diag.json",    ("exp2_redo", "M16", "best", "corr_mean")),
]

ARMS_B = [
    ("held-out\n(real)",              None),
    ("IQP, $M{=}2$\n(KL)",            "iqp_m2_kl_resample_n100"),
    ("IQP, $M{=}4$\n(MMD-flat)",      "iqp_m4_mmdflat_resample_n100"),
    ("IQP, $M{=}8$\n(KL + $z$-pen.)", "iqp_m8_kl_zpen_resample_n100"),
]
NCOL = 3

SCALING_ARMS = [
    ("IQP-affine-M4-MMDflat-resample", "IQP, $M{=}4$", S.OKABE_ITO["blue"],      "o"),
    ("RBM-10bit",                      "RBM",          S.OKABE_ITO["vermilion"], "s"),
    ("DCGAN-32",                       "DCGAN",        S.OKABE_ITO["green"],     "^"),
]


def dig(d, path):
    for k in path:
        d = d[k]
    return float(d)


def rect(x, y, w, h):
    """inches -> figure fraction"""
    return [x / FIG_W, y / FIG_H, w / FIG_W, h / FIG_H]


def load_mech():
    cache = {}
    for _m, fn, _p in M_SWEEP:
        if fn not in cache:
            with open(os.path.join(MECH, fn)) as f:
                cache[fn] = json.load(f)
    ys = [dig(cache[fn], p) for _m, fn, p in M_SWEEP]
    res = cache["results.json"]
    with open(os.path.join(MECH, "diag2.json")) as f:
        diag2 = json.load(f)
    refs = {
        "nearest":  dig(res, ("controls", "holdout_vs_nearest_train_corr", "mean")),
        "mean_img": dig(res, ("controls", "holdout_vs_mean_image_corr", "mean")),
        "random":   0.5 * (dig(diag2, ("holdout_random_vs_trained",
                                       "random Phi(seed 99) + trained theta0", "corr_mean"))
                           + dig(diag2, ("holdout_random_vs_trained",
                                         "random Phi(seed 123) + trained theta0", "corr_mean"))),
    }
    return [m for m, _f, _p in M_SWEEP], ys, refs


def load_scaling():
    rows = [json.loads(l) for l in open(os.path.join(EXP, "results_scaling.jsonl"))]
    out = {}
    for arm, _lab, _c, _mk in SCALING_ARMS:
        r = sorted([x for x in rows if x["arm"] == arm], key=lambda x: x["n_train"])
        out[arm] = (np.array([x["n_train"] for x in r], float),
                    np.array([x["conv_mmd2"] for x in r], float),
                    np.array([x["conv_mmd2_aligned"] for x in r], float))
    fl = [x for x in rows if x["arm"] == "HOLDOUT split-half"][0]
    return out, float(fl["conv_mmd2"]), float(fl["conv_mmd2_aligned"])


def load_images():
    sys.path.insert(0, EXP)
    import data_lens_r3                                 # noqa: E402
    d = data_lens_r3.load_r3(100, n_holdout=200)
    rows = []
    for label, tag in ARMS_B:
        if tag is None:
            imgs = [d.holdout_images[i] for i in range(NCOL)]
        else:
            p = os.path.join(EXP, "outputs_lens_r3b", tag)
            imgs = [np.load(os.path.join(p, "generated_%03d.npy" % i)) for i in range(NCOL)]
        rows.append((label, np.asarray(imgs, dtype=np.float64)))
    return rows


def panel_a(ax, ms, ys, refs):
    grey = S.INK_MUTED
    ax.axhline(refs["nearest"], color=grey, lw=0.9, ls=(0, (4, 2)), zorder=1)
    ax.text(17.3, refs["nearest"] + 0.012, "nearest training image", fontsize=7.0,
            color=grey, ha="right", va="bottom", zorder=4)
    ax.axhline(refs["mean_img"], color=S.OKABE_ITO["orange"], lw=0.9, ls=(0, (4, 2)),
               zorder=1)
    ax.text(17.3, refs["mean_img"] + 0.012, "mean train. image", fontsize=7.0,
            color=S.OKABE_ITO["orange"], ha="right", va="bottom", zorder=4)
    ax.axhline(refs["random"], color=S.OKABE_ITO["vermilion"], lw=0.9, ls=(0, (1.5, 1.8)),
               zorder=1)
    ax.text(-0.7, refs["random"] - 0.020, "random dir.", fontsize=7.0,
            color=S.OKABE_ITO["vermilion"], ha="left", va="top", zorder=4)

    ax.plot(ms, ys, "-o", color=S.OKABE_ITO["blue"], lw=1.4, ms=4.2, zorder=5,
            mec="white", mew=0.5)
    ax.set_xlabel("latent directions $M$")
    ax.set_ylabel("held-out\nprojection corr.", linespacing=1.2)
    ax.set_xticks([0, 2, 4, 6, 8, 16])
    ax.set_xlim(-1.1, 17.6)
    ax.set_ylim(-0.05, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    S.tidy(ax)


def panel_c(ax, data, floor, floor_al):
    n_vals = sorted({int(v) for n, _a, _b in data.values() for v in n})
    lo, hi = min(floor, floor_al), max(floor, floor_al)
    ax.axhspan(lo, hi, color="#c3ccd3", alpha=0.9, zorder=0, lw=0)
    ax.text(105, hi * 1.15, "held-out split-half floor", fontsize=7.0, color="#4a5a66",
            ha="right", va="bottom", zorder=4)
    for arm, lab, col, mk in SCALING_ARMS:
        n, m2, m2a = data[arm]
        ax.plot(n, m2, "-", marker=mk, color=col, lw=1.3, ms=3.8, mec="white", mew=0.4,
                zorder=4, label=lab)
        ax.plot(n, m2a, ls=(0, (3.2, 1.8)), marker=mk, color=col, lw=1.0, ms=3.0,
                mfc="white", mew=0.8, zorder=3)
    ax.set_yscale("log")
    ax.set_xlabel("training images $N$")
    ax.set_ylabel("Conv2D-MMD$^2$")
    ax.set_xlim(5, 105)
    ax.set_ylim(2.6e-3, 1.15)
    ax.set_xticks(n_vals)
    ax.tick_params(axis="x", pad=1.5)
    S.tidy(ax)
    h, _l = ax.get_legend_handles_labels()
    h = h + [Line2D([], [], color=S.INK_MUTED, lw=1.0, ls=(0, (3.2, 1.8)),
                    label="flux-aligned")]
    ax.legend(handles=h, loc="upper left", bbox_to_anchor=(-0.015, 1.05), ncol=2,
              fontsize=7.0, handletextpad=0.4, columnspacing=0.8, labelspacing=0.25,
              borderpad=0.2, borderaxespad=0.0)


def panel_b(fig, rows, x0, y0, cell, gap, lab_x):
    im = None
    nrow = len(rows)
    for i, (label, imgs) in enumerate(rows):
        y = y0 + (nrow - 1 - i) * (cell + gap)
        for j in range(NCOL):
            ax = fig.add_axes(rect(x0 + j * (cell + gap), y, cell, cell))
            im = ax.imshow(imgs[j], cmap=S.IMG_CMAP, vmin=S.IMG_VMIN, vmax=S.IMG_VMAX,
                           interpolation="nearest", origin="lower")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.4); sp.set_color("#b8b8b8")
        fig.text(lab_x / FIG_W, (y + cell / 2.0) / FIG_H, label, fontsize=7.0,
                 ha="right", va="center", linespacing=1.3)
    x_cb = x0 + NCOL * (cell + gap) + 0.03
    h_all = nrow * cell + (nrow - 1) * gap
    cax = fig.add_axes(rect(x_cb, y0, 0.085, h_all))
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("pixel intensity (a.u.)", fontsize=7.0, labelpad=2)
    cb.ax.tick_params(labelsize=7.0, width=0.5, length=2)
    cb.outline.set_linewidth(0.4)


def make(outdir):
    S.apply_style()
    ms, ys, refs = load_mech()
    scal, floor, floor_al = load_scaling()
    rows = load_images()

    fig = plt.figure(figsize=(FIG_W, FIG_H))

    ax_a = fig.add_axes(rect(0.63, 1.98, 1.62, 1.10))
    panel_a(ax_a, ms, ys, refs)

    ax_c = fig.add_axes(rect(0.63, 0.44, 1.62, 1.10))
    panel_c(ax_c, scal, floor, floor_al)

    cell, gap, x0, y0 = 0.573, 0.045, 3.13, 0.44
    panel_b(fig, rows, x0, y0, cell, gap, lab_x=3.05)
    for txt, xi, yi in (("(a)", 0.06, 3.10), ("(c)", 0.06, 1.56), ("(b)", 2.34, 3.10)):
        fig.text(xi / FIG_W, yi / FIG_H, txt, fontsize=9, fontweight="bold",
                 ha="left", va="bottom")

    return S.save(fig, outdir, "fig2_mechanism_scaling")


if __name__ == "__main__":
    make(os.path.dirname(os.path.abspath(__file__)))
