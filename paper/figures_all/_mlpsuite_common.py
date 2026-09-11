# -*- coding: utf-8 -*-
"""Shared loading + the paper's figure style for the part-F figures.

Uses figures/fig_style.py -- the same module fig2_mechanism_scaling.py and figA_scaling.py
use -- so these panels sit next to the published ones: S.apply_style() for the rcParams,
S.OKABE_ITO for the palette, S.tidy() for the spines, S.save() for the pdf+png pair.  The
font sizes are then pinned to 7 pt on top of apply_style()'s 8 pt.  The published colour
assignment is kept (IQP affine = blue, RBM = vermilion, DCGAN = green) and the new MLP arm
takes a colour none of them use, so a panel of this suite can be read against a panel of the
paper without re-learning the legend.  If fig_style cannot be imported the module falls back
to an equivalent explicit rcParams block rather than failing.
"""
from __future__ import annotations

import json
import os
import re

import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402

REPO = "<HOME>/gsoc2026-deeplense-qml"
EXP = os.path.join(REPO, "experiments/13_iqp_turbulence_replication")
OUT = os.path.join(REPO, "paper/ml4ps2026/figures")
SNAP_RE = re.compile(r"\s+s(\d+)k$")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import fig_style as S                                            # noqa: E402
    S.apply_style()
    _HAVE_STYLE = True
except Exception as _e:                                              # pragma: no cover
    S = None
    _HAVE_STYLE = False
    print("[style] fig_style unavailable (%s); using the explicit fallback" % _e)
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    })

# 7 pt on top of whatever the style set, as asked for these panels
plt.rcParams.update({
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6,
    "savefig.dpi": 400,
})

_OK = getattr(S, "OKABE_ITO", None) or {"blue": "#0072B2", "vermilion": "#D55E00",
                                        "green": "#009E73", "purple": "#7B3FA0"}
# published assignment kept; the new arm gets a colour the paper does not already use
COLOURS = {"mlp": _OK.get("purple", "#7B3FA0"), "affine": _OK["blue"],
           "rbm": _OK["vermilion"], "dcgan": _OK["green"]}
TEXTWIDTH = getattr(S, "TEXTWIDTH", 5.5)
COLWIDTH = getattr(S, "COLWIDTH", 2.7)


def tidy(ax, grid=True):
    if _HAVE_STYLE and hasattr(S, "tidy"):
        S.tidy(ax, grid=grid)
        return
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if grid:
        ax.grid(alpha=0.18, lw=0.4)


def read(path):
    out = []
    if os.path.isfile(path):
        for line in open(path):
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def meta_of(rd, _c={}):
    if rd not in _c:
        p = os.path.join(rd or "", "meta.json")
        _c[rd] = json.load(open(p)) if os.path.isfile(p) else {}
    return _c[rd]


def is_snap(arm):
    return bool(SNAP_RE.search(str(arm or "")))


def mean_sd(vals):
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], dtype=np.float64)
    if v.size == 0:
        return np.nan, np.nan, 0
    return float(v.mean()), (float(v.std(ddof=1)) if v.size > 1 else 0.0), int(v.size)


def as_mean_sd(v):
    """Accept a float, a list of per-seed values, or a dict with mean/sd-ish keys."""
    if v is None:
        return np.nan, np.nan
    if isinstance(v, (int, float)):
        return float(v), 0.0
    if isinstance(v, (list, tuple)):
        m, s, _n = mean_sd(list(v))
        return m, s
    if isinstance(v, dict):
        for mk in ("mean", "avg", "value", "corr", "m"):
            if mk in v:
                m = float(v[mk])
                for sk in ("sd", "std", "stdev", "s"):
                    if sk in v:
                        return m, float(v[sk])
                return m, 0.0
        for k in ("values", "seeds", "per_seed"):
            if k in v and isinstance(v[k], (list, tuple)):
                m, s, _n = mean_sd(list(v[k]))
                return m, s
    return np.nan, np.nan


def save(fig, stem, also_pdf=True):
    os.makedirs(OUT, exist_ok=True)
    made = []
    if _HAVE_STYLE and hasattr(S, "save") and also_pdf:
        S.save(fig, OUT, stem)                       # the paper's own pdf + png writer
        made = [os.path.join(OUT, stem + ext) for ext in (".pdf", ".png")]
        made = [p for p in made if os.path.isfile(p)]
    else:
        png = os.path.join(OUT, stem + ".png")
        fig.savefig(png, bbox_inches="tight")
        made = [png]
        if also_pdf:
            pdf = os.path.join(OUT, stem + ".pdf")
            fig.savefig(pdf, bbox_inches="tight")
            made.append(pdf)
    plt.close(fig)
    for p in made:
        print("[fig] wrote %s (%d bytes)" % (p, os.path.getsize(p)))
    return made
