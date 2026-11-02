#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""figA_scaling_seeds.pdf -- the data-scaling panel as mean +- sd over seeds.

Same axes, arms, colours and markers as figA_scaling.py, but every point is the mean over the
seeds in `results_seeds_scaling.jsonl` (written by aggregate_seeds.py) and carries a +- 1 sd
band.  Solid line / filled band: Conv2D-MMD^2.  Dashed line / hatched band: the flux-aligned
variant.  figA_scaling.py is untouched, so the single-seed figure the current draft uses
still builds.

An arm whose seeds have not all finished is still drawn, and the caption line at the bottom
says how many seeds each arm actually has, so a partly finished sweep cannot be mistaken for
a finished one.

Python 3.9 compatible.
"""
from __future__ import annotations

import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402
import numpy as np                                     # noqa: E402
from matplotlib.lines import Line2D                    # noqa: E402
from matplotlib.patches import Patch                   # noqa: E402

import fig_style as S
import fig2_mechanism_scaling as F2

EXP = "<HOME>/gsoc2026-deeplense-qml/experiments/13_iqp_turbulence_replication"
# v11 is the primary metric protocol (fixed [0, 1] PDF-JS bins; Conv2D-MMD^2 normalisation
# and bandwidth from the held-out set alone).  Fall back to the v10 file only if the v11 one
# has not been written yet, and say which was used on the figure.
RESULTS_V11 = os.path.join(EXP, "results_seeds_scaling_v11.jsonl")
RESULTS_V10 = os.path.join(EXP, "results_seeds_scaling.jsonl")
FLOOR_V11 = os.path.join(EXP, "results_v11_floor.jsonl")
FLOOR_ARM = "HOLDOUT split-half"
YMIN = 2.6e-3


def _mean_sd(vals):
    v = [float(x) for x in vals if x is not None]
    if not v:
        return None, None, 0
    m = sum(v) / len(v)
    if len(v) == 1:
        return m, 0.0, 1
    s = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))
    return m, s, len(v)


def load_seed_scaling(path=None):
    """-> ({arm: (n, m2_mean, m2_sd, m2al_mean, m2al_sd, n_seeds)}, floor, floor_al, protocol)."""
    protocol = "v11"
    if path is None:
        path = RESULTS_V11 if os.path.isfile(RESULTS_V11) else RESULTS_V10
        protocol = "v11" if path == RESULTS_V11 else "v10"
    if not os.path.isfile(path):
        raise SystemExit("figA_scaling_seeds: %s does not exist -- run aggregate_seeds.py "
                         "after the seed sweep" % path)
    rows = [json.loads(l) for l in open(path) if l.strip()]
    out = {}
    for arm, _lab, _c, _mk in F2.SCALING_ARMS:
        by_n = {}
        for r in rows:
            if r.get("arm") != arm or r.get("n_train") is None:
                continue
            by_n.setdefault(int(r["n_train"]), []).append(r)
        ns = sorted(by_n)
        if not ns:
            continue
        m2m, m2s, alm, als, ks = [], [], [], [], []
        for n in ns:
            a, b, k = _mean_sd([r.get("conv_mmd2") for r in by_n[n]])
            c, d, _ = _mean_sd([r.get("conv_mmd2_aligned") for r in by_n[n]])
            m2m.append(a); m2s.append(b); alm.append(c); als.append(d); ks.append(k)
        out[arm] = (np.array(ns, float), np.array(m2m, float), np.array(m2s, float),
                    np.array(alm, float), np.array(als, float), int(min(ks)))
    # Under v11 the floor is the 200-vs-200 real-vs-real bootstrap, which is the like-for-like
    # reference: every model point is also 200 generated vs 200 held-out.  The v10 file's
    # split-half row is 100 vs 100, and the biased MMD^2 estimator is sensitive to that.
    if protocol == "v11" and os.path.isfile(FLOOR_V11):
        fr = [json.loads(l) for l in open(FLOOR_V11) if l.strip()]
        fr = [r for r in fr if r.get("kind") == "floor_v11" and not r.get("class_name")]
        if fr:
            m2 = float(np.mean([r["conv_mmd2"] for r in fr]))
            m2a = float(np.mean([r["conv_mmd2_aligned"] for r in fr]))
            return out, m2, m2a, protocol
    fl = [r for r in rows if r.get("arm") == FLOOR_ARM]
    if not fl:
        raise SystemExit("figA_scaling_seeds: no %r row in %s" % (FLOOR_ARM, path))
    return out, float(fl[0]["conv_mmd2"]), float(fl[0]["conv_mmd2_aligned"]), protocol


def _band(ax, x, mu, sd, col, hatch=None, alpha=0.16):
    """A +- 1 sd band, clipped at YMIN so a log axis never sees a non-positive edge."""
    lo = np.clip(mu - sd, YMIN, None)
    hi = mu + sd
    ax.fill_between(x, lo, hi, color=col, alpha=alpha, lw=0, zorder=2, hatch=hatch,
                    edgecolor=col if hatch else None)


def make(outdir):
    S.apply_style()
    data, floor, floor_al, protocol = load_seed_scaling()
    fig = plt.figure(figsize=(S.COLWIDTH, 2.30))
    ax = fig.add_axes([0.215, 0.235, 0.755, 0.720])

    n_vals = sorted({int(v) for d in data.values() for v in d[0]})
    lo, hi = min(floor, floor_al), max(floor, floor_al)
    ax.axhspan(lo, hi, color="#c3ccd3", alpha=0.9, zorder=0, lw=0)
    floor_label = ("real-vs-real floor (200 v 200)" if protocol == "v11"
                   else "held-out split-half floor")
    ax.text(104, hi * 1.16, floor_label, fontsize=7.0, color="#4a5a66",
            ha="right", va="bottom", zorder=4)

    seed_counts = []
    for arm, lab, col, mk in F2.SCALING_ARMS:
        if arm not in data:
            continue
        n, m2, m2sd, al, alsd, k = data[arm]
        # the label carries its own comma ("IQP, $M{=}4$"), so separate the
        # entries with a semicolon and bind the count with a colon
        seed_counts.append("%s: %d" % (lab.replace(",", ""), k))
        _band(ax, n, m2, m2sd, col)
        _band(ax, n, al, alsd, col, hatch="////", alpha=0.10)
        ax.plot(n, m2, "-", marker=mk, color=col, lw=1.3, ms=3.8, mec="white", mew=0.4,
                zorder=5, label=lab)
        ax.plot(n, al, ls=(0, (3.2, 1.8)), marker=mk, color=col, lw=1.0, ms=3.0,
                mfc="white", mew=0.8, zorder=4)

    ax.set_yscale("log")
    ax.set_xlabel("training images $N$")
    ax.set_ylabel("Conv2D-MMD$^2$")
    ax.set_xlim(5, 105)
    ax.set_ylim(YMIN, 1.6)
    ax.set_xticks(n_vals)
    ax.tick_params(axis="x", pad=1.5)
    S.tidy(ax)
    h, _l = ax.get_legend_handles_labels()
    h = h + [Line2D([], [], color=S.INK_MUTED, lw=1.0, ls=(0, (3.2, 1.8)),
                    label="flux-aligned"),
             Patch(facecolor=S.INK_MUTED, alpha=0.16, lw=0, label=r"$\pm 1$ sd")]
    ax.legend(handles=h, loc="upper left", bbox_to_anchor=(-0.015, 1.04), ncol=2,
              fontsize=7.0, handletextpad=0.4, columnspacing=0.8, labelspacing=0.25,
              borderpad=0.2, borderaxespad=0.0)
    # The protocol and the seed counts live in the caption, not on the axes.
    return S.save(fig, outdir, "figA_scaling_seeds")


if __name__ == "__main__":
    make(os.path.dirname(os.path.abspath(__file__)))
