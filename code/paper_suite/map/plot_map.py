#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render results_map.json as docs/figures/iqp_lens_map.png (the Fig. 5 analogue).

Panel A -- the correlation-complexity map. Filled marker = dataset; hollow marker of the same
           colour = its i.i.d.-uniform null at the SAME (n, M), joined by a thin line; a cross
           on a dotted line = the structure-matched permutation null (identical value
           histogram, spatial layout destroyed).
Panel B -- zoom on the QCLI < 0.11 cluster, where every reference dataset lives.
Panel C -- the background every x-value has to be read against: I_QCLI of pure i.i.d.-uniform
           draws vs M, for n = 10, 16, 18, with M = C(n, n/2) marked.

Python 3.9 compatible. No network, no seaborn.
"""
from __future__ import annotations

import argparse
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
from matplotlib.lines import Line2D                 # noqa: E402
from matplotlib.patches import Rectangle            # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
IN_DEFAULT = os.path.join(HERE, "results_map.json")
OUT_DEFAULT = "<HOME>/gsoc2026-deeplense-qml/docs/figures/iqp_lens_map.png"

FAMILY = {                                          # Okabe-Ito, colourblind safe
    "lens-18": ("#0072B2", "o", "lens, 18-bit (x,y,z) tuples"),
    "lens-10": ("#7B3FA0", "s", "lens, 10-bit crop pmf"),
    "turbulence D2": ("#009E73", "^", "turbulence D2 (paper's dataset)"),
    "binarized MNIST": ("#E69F00", "D", "binarized MNIST (Ref [38])"),
    "BinaryBlobs": ("#D55E00", "*", "BinaryBlobs (Ref [38])"),
}

# representation substring -> family, panel-A label + offset, panel-B label + offset
#   None as a label means "do not annotate in this panel"
SPEC = [
    ("18-bit (x,y,z) tuples, per-image",       "lens-18", ("lens 18b per-image\nM=4k", (11, -11)),  (None, None)),
    ("18-bit (x,y,z) tuples, pooled",          "lens-18", ("lens 18b pooled\nM=410k", (11, -4)),     (None, None)),
    ("10-bit crop pmf, per-image (exact",      "lens-10", ("lens 10b per-image\n(exact pmf)", (11, -13)), (None, None)),
    ("10-bit crop pmf, per-image (M=",         "lens-10", ("lens 10b per-image\nM=4k", (-104, 5)),  (None, None)),
    ("10-bit crop pmf, pooled 100 train (exact", "lens-10", ("lens 10b pooled\n(exact pmf, and\nM=410k on top)", (11, 12)), (None, None)),
    ("10-bit crop pmf, pooled 100 train (M=",  "lens-10", (None, None),                            (None, None)),
    ("pooled 11 anchors",                      "turbulence D2", ("D2 pooled\nM=550k", (11, -12)),   (None, None)),
    ("per-snapshot (N_t=5e4)",                 "turbulence D2", (None, None),  ("D2 snapshot, M=50k", (-30, -17))),
    ("per-snapshot (M=4096)",                  "turbulence D2", (None, None),  ("D2 snapshot, M=4k", (8, 6))),
    ("16-bit (7x7 mean-pool to 4x4, thr 0.5)", "binarized MNIST", (None, None), ("MNIST 16b\n(4x4 mean-pooled)", (11, 2))),
    ("484-bit",                                "binarized MNIST", (None, None), ("MNIST 484b", (-24, 21))),
    ("784-bit",                                "binarized MNIST", (None, None), ("MNIST 784b", (-26, -19))),
    ("flip p=0.05",                            "BinaryBlobs", (None, None),    ("Blobs p=0.05", (11, 3))),
    ("flip p=0.01",                            "BinaryBlobs", (None, None),    ("Blobs p=0.01", (10, -2))),
]

ZOOM = (-0.004, 0.108, 0.05, 1.03)                  # x0, x1, y0, y1 of panel B


def spec_for(row):
    for key, fam, a_lab, b_lab in SPEC:
        if key in row["representation"]:
            return fam, a_lab, b_lab
    return None, (None, None), (None, None)


def draw_points(ax, rows, which, nulls=True):
    """which = 0 for panel A labels, 1 for panel B labels."""
    drawn = set()                       # coincident points must not be annotated twice
    for r in rows:
        fam, a_lab, b_lab = spec_for(r)
        if fam is None:
            continue
        colour, marker, _ = FAMILY[fam]
        x, y = r["I_QCLI"], r["I_CCI"]
        ms = 14 if marker == "*" else 8
        if nulls and r["null_QCLI"] is not None:
            ax.plot([x, r["null_QCLI"]], [y, r["null_CCI"]], "-", color=colour, lw=0.7,
                    alpha=0.4, zorder=1)
            ax.plot([r["null_QCLI"]], [r["null_CCI"]], marker=marker, mfc="none", mec=colour,
                    ms=ms, mew=1.1, ls="none", zorder=2, alpha=0.85)
        if "perm_QCLI" in r:
            ax.plot([x, r["perm_QCLI"]], [y, r["perm_CCI"]], ":", color=colour, lw=1.2,
                    alpha=0.95, zorder=1)
            ax.plot([r["perm_QCLI"]], [r["perm_CCI"]], marker="x", color=colour, ms=7,
                    mew=1.7, ls="none", zorder=3)
        ax.plot([x], [y], marker=marker, color=colour, ms=ms, ls="none", zorder=4,
                mec="white", mew=0.6)
        lab, off = (a_lab if which == 0 else b_lab)
        if lab in drawn:
            lab = None
        if lab:
            drawn.add(lab)
            ax.annotate(lab, (x, y), textcoords="offset points", xytext=off, fontsize=7.4,
                        color=colour, ha="left", va="center", zorder=5)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inp", default=IN_DEFAULT)
    ap.add_argument("--out", default=OUT_DEFAULT)
    a = ap.parse_args(argv)

    with open(a.inp) as fh:
        res = json.load(fh)
    rows = res["points"]

    fig, (ax, bx, cx) = plt.subplots(
        1, 3, figsize=(17.6, 6.4), gridspec_kw={"width_ratios": [1.45, 1.0, 1.15]})

    # -------------------------------------------------------------- A: the whole map
    ax.axhspan(0.5, 1.06, xmin=(0.2 + 0.04) / 1.08, xmax=1.0, color="#009E73", alpha=0.05, lw=0)
    ax.annotate("paper's \"IQP-compatible\" corner\n(Fig. 5 puts turbulence D1/D2 here;\n"
                "boundaries illustrative -- the paper gives none)",
                xy=(1.045, 0.60), fontsize=7.6, color="#00674c", ha="right", va="center",
                style="italic", zorder=6)
    draw_points(ax, rows, which=0)
    ax.add_patch(Rectangle((ZOOM[0], ZOOM[2]), ZOOM[1] - ZOOM[0], ZOOM[3] - ZOOM[2],
                           fill=False, ec="0.35", lw=0.9, ls="-", zorder=5))
    ax.annotate("panel B", xy=(ZOOM[1], 0.30), xytext=(0.22, 0.22), fontsize=8, color="0.35",
                arrowprops=dict(arrowstyle="->", color="0.35", lw=0.9))
    ax.axvline(0.2, color="0.55", lw=0.9, ls="--", zorder=0)
    ax.text(0.208, 0.035, "paper: no real / benchmark\ndataset it measured exceeded\n"
                          r"$I_{\mathrm{QCLI}} \approx 0.2$",
            fontsize=7.4, color="0.35", ha="left", va="bottom")
    ax.set_xlim(-0.04, 1.06)
    ax.set_ylim(-0.04, 1.08)
    ax.set_xlabel(r"$I_{\mathrm{QCLI}}$   (Walsh order spectrum vs binomial, $\log_2$ JS)")
    ax.set_ylabel(r"$I_{\mathrm{CCI}} = 1 - I_{\mathrm{TC}}^{\mathrm{tree}} / I_{\mathrm{TC}}$")
    ax.set_title("A.  Correlation-complexity map, recomputed on our data", fontsize=11)
    ax.grid(alpha=0.2, lw=0.6)

    # -------------------------------------------------------------- B: the cluster
    draw_points(bx, rows, which=1)
    bx.set_xlim(ZOOM[0], ZOOM[1])
    bx.set_ylim(ZOOM[2], ZOOM[3])
    bx.set_xlabel(r"$I_{\mathrm{QCLI}}$")
    bx.set_ylabel(r"$I_{\mathrm{CCI}}$")
    bx.set_title("B.  Zoom: every reference dataset,\nand D2 at small M", fontsize=11)
    bx.grid(alpha=0.2, lw=0.6)
    bx.text(0.5, 0.70, "hollow markers along the top = each point's i.i.d.-uniform null:\n"
                       "the plug-in CCI of pure noise is ~1 at every (n, M) here,\n"
                       "so a LOW CCI is what marks a structured dataset",
            transform=bx.transAxes, fontsize=7.4, color="0.35", ha="center", va="center",
            bbox=dict(fc="white", ec="0.85", lw=0.5, alpha=0.9, boxstyle="round,pad=0.4"))

    # -------------------------------------------------------------- C: null vs M
    curve_col = {"10": "#7B3FA0", "16": "#E69F00", "18": "#0072B2"}
    for n_str, pts in sorted(res["uniform_null_curves"].items(), key=lambda kv: int(kv[0])):
        n = int(n_str)
        cx.plot([p["M"] for p in pts], [p["I_QCLI"] for p in pts], "-o",
                color=curve_col[n_str], ms=4.5, lw=1.7, label="n = %d" % n)
        cross = math.comb(n, n // 2)
        cx.axvline(cross, color=curve_col[n_str], ls="--", lw=0.9, alpha=0.5)
        cx.text(cross, 1.035, r"$\binom{%d}{%d}$" % (n, n // 2), color=curve_col[n_str],
                fontsize=8.5, ha="center", va="bottom")
    cx.set_xscale("log")
    cx.set_xlim(6e2, 2e6)
    cx.set_ylim(-0.03, 1.12)
    cx.set_xlabel("M (number of sampled bitstrings)")
    cx.set_ylabel(r"$I_{\mathrm{QCLI}}$ of i.i.d.-uniform draws")
    cx.set_title("C.  The null is not zero:\nQCLI of pure noise vs sample count", fontsize=11)
    cx.grid(alpha=0.2, lw=0.6)
    cx.legend(fontsize=9, loc="upper left")
    cx.text(0.70, 0.12, "above M ~ C(n, n/2) the empirical order spectrum\n"
                        "of pure noise collapses onto k = 0, so QCLI -> 1",
            transform=cx.transAxes, fontsize=7.6, color="0.35", ha="center", va="center",
            bbox=dict(fc="white", ec="0.85", lw=0.5, alpha=0.9, boxstyle="round,pad=0.4"))

    # -------------------------------------------------------------- shared legend + caption
    handles = [Line2D([], [], color=c, marker=m, ls="none", ms=12 if m == "*" else 7.5,
                      mec="white", mew=0.6, label=lab) for c, m, lab in FAMILY.values()]
    handles += [
        Line2D([], [], color="0.35", marker="o", mfc="none", ls="-", lw=0.7, ms=7,
               label="i.i.d.-uniform null at the same (n, M)"),
        Line2D([], [], color="0.35", marker="x", ls=":", lw=1.2, ms=7,
               label="structure-matched null (value histogram kept, spatial layout permuted)"),
    ]
    fig.legend(handles=handles, fontsize=8.6, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.005), handletextpad=0.6, columnspacing=1.6)
    fig.suptitle("arXiv 2603.06440 Fig. 5 recomputed on the DeepLense lens images "
                 "(two training representations), turbulence D2, and the Ref [38] reference "
                 "datasets\n"
                 "D-Wave annealer and Google-RCS points are omitted: we have no access to "
                 "either dataset.  Every point is paired with a null drawn at its own (n, M).",
                 fontsize=10.5, y=0.985)
    fig.tight_layout(rect=(0, 0.115, 1, 0.925))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=165)
    print("[plot_map] wrote %s" % a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
