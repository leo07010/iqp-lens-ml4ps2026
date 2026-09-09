#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared matplotlib style for the ML4PS-2026 figures.

NeurIPS body text is 10 pt; figure text is 8-9 pt, never below 7 pt.
Palette: Okabe-Ito (colour-blind safe), used in a FIXED order, and every series also
carries its own marker shape so identity never rests on colour alone.
Image panels: one shared sequential colour map (magma), vmin=0, vmax=1.

Python 3.9 compatible.  No seaborn, no network.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ------------------------------------------------------------------ geometry (inches)
TEXTWIDTH = 5.5      # NeurIPS \textwidth
COLWIDTH = 2.7       # half of it, for single-column figures

# ------------------------------------------------------------------ colour
OKABE_ITO = {
    "blue":      "#0072B2",
    "vermilion": "#D55E00",
    "green":     "#009E73",
    "orange":    "#E69F00",
    "skyblue":   "#56B4E9",
    "purple":    "#7B3FA0",
    "yellow":    "#F0E442",
    "black":     "#000000",
}
INK = "#1a1a1a"
INK_MUTED = "#6b6b6b"
GRID = "#dcdcdc"

IMG_CMAP = "magma"
IMG_VMIN, IMG_VMAX = 0.0, 1.0


def apply_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "legend.handletextpad": 0.5,
        "legend.labelspacing": 0.3,
        "legend.borderaxespad": 0.3,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "lines.linewidth": 1.2,
        "lines.markersize": 4,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,      # TrueType -> selectable/searchable text in the PDF
        "ps.fonttype": 42,
        "pdf.compression": 6,
    })


def panel_label(ax, text, x=-0.02, y=1.06, **kw):
    """(a)/(b)/(c) in the panel's top-left, in axes coordinates."""
    ax.text(x, y, text, transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="bottom", ha="left", color=INK, **kw)


def tidy(ax, grid=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if grid:
        ax.grid(True, which="major", alpha=0.6, zorder=0)
        ax.set_axisbelow(True)


def save(fig, outdir, stem):
    """Write <stem>.pdf (vector) and <stem>.png (300 dpi).  Returns the two paths."""
    os.makedirs(outdir, exist_ok=True)
    pdf = os.path.join(outdir, stem + ".pdf")
    png = os.path.join(outdir, stem + ".png")
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(png, bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(fig)
    w, h = fig.get_size_inches()
    print("[fig] %-28s %.2f x %.2f in   -> %s (+ .png)" % (stem, w, h, pdf), flush=True)
    return pdf, png
