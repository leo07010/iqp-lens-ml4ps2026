#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fig1_map_samples.pdf

(a) the correlation-complexity map (Fig. 5 analogue of arXiv 2603.06440), redrawn from
    paper_suite/map/results_map.json.  Filled marker = dataset; hollow marker of the same
    colour = its i.i.d.-uniform null at the SAME (n, M), joined by a thin line; a small
    cross on a dotted line = the structure-matched permutation null where one exists.
(b) held-out lens images and one sample from each generative arm, all on one linear
    magma scale, vmin = 0, vmax = 1.

Reads only committed artefacts; writes only into this directory.  Python 3.9 compatible.
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
import matplotlib.gridspec as gridspec                 # noqa: E402

import fig_style as S

EXP = "<HOME>/gsoc2026-deeplense-qml/experiments/13_iqp_turbulence_replication"
MAP_JSON = os.path.join(EXP, "paper_suite", "map", "results_map.json")

# representation-substring -> (colour, marker, legend label, draw its i.i.d. null?)
POINTS = [
    ("18-bit (x,y,z) tuples, per-image",       S.OKABE_ITO["blue"],      "o",
     "lens 18-bit", True),
    ("18-bit (x,y,z) tuples, pooled",          S.OKABE_ITO["blue"],      "s",
     "lens 18-bit, pooled", True),
    ("10-bit crop pmf, per-image (exact",      S.OKABE_ITO["purple"],    "D",
     "lens 10-bit pmf", False),
    ("per-snapshot (M=4096)",                  S.OKABE_ITO["green"],     "^",
     "turbulence D2", True),
    ("pooled 11 anchors",                      S.OKABE_ITO["green"],     "v",
     "turb. D2, pooled", True),
    ("16-bit (7x7 mean-pool to 4x4, thr 0.5)", S.OKABE_ITO["orange"],    "P",
     "MNIST-16", True),
    ("784-bit",                                S.OKABE_ITO["orange"],    "X",
     "MNIST-784", True),
    ("flip p=0.05",                            S.OKABE_ITO["vermilion"], "*",
     "BinaryBlobs", True),
]

ARMS = [
    ("held-out\n(real)",             None),
    ("IQP\n(paper protocol)",       ("outputs_lens_r3",  "iqp_paper_n100")),
    ("IQP, learned $\\Phi$\n($M{=}4$, MMD-flat)", ("outputs_lens_r3b", "iqp_m4_mmdflat_resample_n100")),
    ("RBM",                                   ("outputs_lens_r3",  "rbm_n100")),
    ("DCGAN",                                 ("outputs_lens_r3",  "dcgan_n100")),
]
NCOL = 3


def load_map():
    with open(MAP_JSON) as f:
        d = json.load(f)
    out = []
    for key, col, mk, lab, want_null in POINTS:
        hit = [r for r in d["points"] if key in r["representation"]]
        if not hit:
            raise SystemExit("map point not found: %s" % key)
        out.append((hit[0], col, mk, lab, want_null))
    return out, d


def load_images():
    sys.path.insert(0, EXP)
    import data_lens_r3                                 # noqa: E402
    d = data_lens_r3.load_r3(100, n_holdout=200)
    rows = []
    for label, src in ARMS:
        if src is None:
            imgs = [d.holdout_images[i] for i in range(NCOL)]
        else:
            p = os.path.join(EXP, src[0], src[1])
            imgs = [np.load(os.path.join(p, "generated_%03d.npy" % i)) for i in range(NCOL)]
        rows.append((label, np.asarray(imgs, dtype=np.float64)))
    return rows


def panel_a(ax, pts):
    # soft upper-right wash: the paper states no numeric threshold, so no hard edge here
    gy, gx = np.mgrid[0:200, 0:200] / 199.0
    alpha = np.clip((gx - 0.30) / 0.70, 0, 1) * np.clip((gy - 0.40) / 0.60, 0, 1)
    wash = np.zeros(gx.shape + (4,))
    wash[..., 0], wash[..., 1], wash[..., 2] = 0.31, 0.55, 0.72
    wash[..., 3] = 0.32 * alpha
    ax.imshow(wash, extent=[0, 1, 0, 1], origin="lower", aspect="auto", zorder=0,
              interpolation="bilinear")
    ax.text(0.982, 0.985, "paper's\n\"IQP-compatible\"\ncorner",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.0,
            color="#2c5a78", linespacing=1.25, zorder=5)

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
        ax.plot([x], [y], marker=mk, ms=ms, color=col, ls="none", zorder=4,
                mec="white", mew=0.5)
        handles.append(Line2D([], [], marker=mk, ls="none", ms=ms * 0.85, color=col,
                              mec="white", mew=0.4, label=lab))

    # the 10-bit point's own null does not fit the picture: state it instead of drawing it
    r10 = [r for r, _c, _m, _l, w in pts if not w][0]
    ax.annotate("saturated at $n\\!=\\!10$:\nnull $=(%.2f,\\ %.2f)$" % (r10["null_QCLI"], r10["null_CCI"]),
                (r10["I_QCLI"], r10["I_CCI"]), textcoords="offset points",
                xytext=(0, -12), fontsize=7.0, ha="center", va="top",
                color=S.OKABE_ITO["purple"], linespacing=1.2, zorder=5)

    ax.set_xlim(-0.035, 1.05)
    ax.set_ylim(-0.05, 1.10)
    ax.set_xlabel(r"$I_{\mathrm{QCLI}}$  (0–1)")
    ax.set_ylabel(r"$I_{\mathrm{CCI}}$  (0–1)")
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    S.tidy(ax)

    conv = [
        Line2D([], [], marker="o", ls="none", ms=4.4, color=S.INK_MUTED, mec="white",
               mew=0.4, label="dataset"),
        Line2D([], [], marker="o", ls="none", ms=3.8, mfc="white", mec=S.INK_MUTED,
               mew=0.9, label="uniform null"),
        Line2D([], [], marker="x", ls="none", ms=3.4, mew=1.0, color=S.INK_MUTED,
               label="permutation null"),
    ]
    return handles + conv


def panel_b(fig, gs_right, rows):
    inner = gridspec.GridSpecFromSubplotSpec(len(rows), NCOL + 1, subplot_spec=gs_right,
                                             wspace=0.06, hspace=0.06,
                                             width_ratios=[1] * NCOL + [0.085])
    im = None
    for i, (label, imgs) in enumerate(rows):
        for j in range(NCOL):
            ax = fig.add_subplot(inner[i, j])
            im = ax.imshow(imgs[j], cmap=S.IMG_CMAP, vmin=S.IMG_VMIN, vmax=S.IMG_VMAX,
                           interpolation="nearest", origin="lower")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.4)
                sp.set_color("#b8b8b8")
            if j == 0:
                ax.set_ylabel(label, fontsize=7.0, rotation=0, ha="right", va="center",
                              labelpad=4, linespacing=1.25)
    cax = fig.add_subplot(inner[:, NCOL])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("pixel intensity (a.u.)", fontsize=7.0, labelpad=2)
    cb.ax.tick_params(labelsize=7.0, width=0.5, length=2)
    cb.outline.set_linewidth(0.4)
    return inner


def make(outdir):
    S.apply_style()
    pts, _raw = load_map()
    rows = load_images()

    fig = plt.figure(figsize=(S.TEXTWIDTH - 0.037, 3.20))
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.02, 1.0], wspace=0.42,
                           left=0.088, right=0.936, bottom=0.235, top=0.955)
    ax_a = fig.add_subplot(gs[0, 0])
    handles = panel_a(ax_a, pts)
    S.panel_label(ax_a, "(a)", x=-0.165, y=1.00)

    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.005),
               ncol=4, fontsize=7.0, handletextpad=0.4, columnspacing=1.4,
               labelspacing=0.35, borderpad=0.2, borderaxespad=0.0)

    inner = panel_b(fig, gs[0, 1], rows)
    ax_first = fig.add_subplot(inner[0, 0])
    ax_first.set_visible(False)
    fig.text(0.530, 0.945, "(b)", fontsize=9, fontweight="bold", ha="left", va="bottom")

    return S.save(fig, outdir, "fig1_map_samples")


if __name__ == "__main__":
    make(os.path.dirname(os.path.abspath(__file__)))
