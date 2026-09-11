#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""figA_convergence.pdf -- were the classical baselines simply stopped too early?

Table 2 trains the RBM and the DCGAN for a fixed 20,000 steps.  This figure re-runs both for
longer (RBM to 160k, DCGAN to 60k), scores every snapshot with the SAME evaluation used for
Table 2, and so lets each baseline be read at its best step count rather than at an arbitrary
one -- an advantage no IQP arm in Table 2 is given.

Two panels, N = 11 and N = 100.  x is the training step count (log).  Colour is the model.
Solid lines with filled markers are PDF-JS and are read on the LEFT axis; dashed lines with
open markers are the flux-aligned Conv2D-MMD^2 and are read on the RIGHT axis.

The two axes are deliberately locked: the right axis is the left axis times a single constant
(29.6, the ratio of the two Table-2 IQP reference values), so the ONE grey horizontal is the
best IQP entry of Table 2 in BOTH metrics at once -- 0.00095 / 0.00092 read on the left,
0.0279 / 0.0275 read on the right, which coincide to within 1 % of a decade.  Anything below
the grey line is a baseline that has overtaken every IQP arm of Table 2 in that metric.
The vertical dotted line is Table 2's 20k budget.

Data: experiments/13_iqp_turbulence_replication/results_convergence.jsonl (the sweep) and
results_r3.jsonl / results_r3b.jsonl (the IQP reference, recomputed by
aggregate_convergence.iqp_reference, which asserts it still matches the published Table 2).

Nothing is below 7 pt.  Python 3.9 compatible.
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402
from matplotlib.lines import Line2D                    # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator   # noqa: E402

import fig_style as S

EXP = "<HOME>/gsoc2026-deeplense-qml/experiments/13_iqp_turbulence_replication"
SWEEP = os.path.join(EXP, "results_convergence.jsonl")

# (arm_base in results_convergence.jsonl, legend label, colour, marker)
ARMS = [
    ("RBM-10bit", "RBM", S.OKABE_ITO["blue"], "o"),
    ("DCGAN-32", "DCGAN", S.OKABE_ITO["vermilion"], "s"),
]
N_PANELS = [11, 100]
DASH = (0, (3.2, 1.8))
XTICKS = [5000, 10000, 20000, 40000, 80000, 160000]
XLABELS = ["5k", "10k", "20k", "40k", "80k", "160k"]


def load():
    """{(arm_base, N): (steps, pdf_js, mmd2_aligned)} and the Table-2 IQP reference."""
    rows = [json.loads(l) for l in open(SWEEP) if l.strip()]
    acc = {}
    for r in rows:
        acc.setdefault((r["arm_base"], int(r["n_train"])), []).append(
            (int(r["steps"]), r["pdf_js"], r["conv_mmd2_aligned"]))
    data = {}
    for key, pts in acc.items():
        pts.sort()
        data[key] = ([p[0] for p in pts], [p[1] for p in pts], [p[2] for p in pts])
    sys.path.insert(0, EXP)
    import aggregate_convergence                       # noqa: E402  (asserts against Table 2)
    return data, aggregate_convergence.iqp_reference(EXP)


def _log_axis(ax):
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0), numticks=14))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: "%g" % v))


def make(outdir):
    S.apply_style()
    data, ref = load()

    # one constant relating the two axes, so a single grey line is the IQP reference for both
    ratios = [ref[n]["conv_mmd2_aligned"]["value"] / ref[n]["pdf_js"]["value"] for n in N_PANELS]
    R = (ratios[0] * ratios[1]) ** 0.5
    pj_all = [v for key in data for v in data[key][1]] + \
             [ref[n]["pdf_js"]["value"] for n in N_PANELS] + \
             [v / R for key in data for v in data[key][2]] + \
             [ref[n]["conv_mmd2_aligned"]["value"] / R for n in N_PANELS]
    lo, hi = min(pj_all) / 1.55, max(pj_all) * 1.55

    fig = plt.figure(figsize=(S.TEXTWIDTH, 2.15))
    geom = [(0.093, 0.360), (0.548, 0.360)]        # (left, width) as figure fractions
    for k, n in enumerate(N_PANELS):
        left, width = geom[k]
        ax = fig.add_axes([left, 0.200, width, 0.600])
        tw = ax.twinx()
        iqp = ref[n]["pdf_js"]["value"]

        ax.axhspan(lo, iqp, color="#eaeef1", zorder=0, lw=0)
        ax.axvline(20000, color=S.INK_MUTED, lw=0.7, ls=(0, (1.2, 1.6)), zorder=1)
        ax.axhline(iqp, color=S.INK_MUTED, lw=1.0, zorder=2)

        for arm, _lab, col, mk in ARMS:
            steps, pj, ma = data[(arm, n)]
            ax.plot(steps, pj, "-", marker=mk, color=col, lw=1.3, ms=3.4, mec="white",
                    mew=0.4, zorder=5)
            ax.plot(steps, [v / R for v in ma], ls=DASH, marker=mk, color=col, lw=1.0,
                    ms=3.0, mfc="white", mew=0.8, zorder=4)

        ax.set_xscale("log")
        ax.set_xlim(3.8e3, 2.3e5)
        ax.set_xticks(XTICKS)
        ax.set_xticklabels(XLABELS)
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_ylim(lo, hi)
        tw.set_ylim(lo * R, hi * R)
        _log_axis(ax)
        _log_axis(tw)
        ax.set_xlabel("training steps", labelpad=1.5)
        ax.set_title("$N = %d$" % n, pad=3.0)
        S.panel_label(ax, "(%s)" % "ab"[k], x=-0.245, y=0.99)
        ax.spines["top"].set_visible(False)
        tw.spines["top"].set_visible(False)
        ax.grid(True, axis="x", alpha=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", pad=1.5)
        ax.tick_params(axis="y", pad=1.5)
        tw.tick_params(axis="y", pad=1.5)
        if k == 0:
            ax.set_ylabel("PDF--JS", labelpad=1.0)
            tw.tick_params(axis="y", labelright=False)
            ax.text(4.4e3, iqp / 1.30, "below: beats every IQP arm of Table 2", fontsize=7.0,
                    color="#5a6b78", ha="left", va="top", zorder=6)
            ax.text(20000 / 1.16, hi / 1.06, "Table 2\nbudget", fontsize=7.0,
                    color=S.INK_MUTED, ha="right", va="top", linespacing=1.05, zorder=6)
        else:
            ax.tick_params(axis="y", labelleft=False)
            tw.set_ylabel("MMD$^2_{\\mathrm{al}}$", labelpad=2.0)
            ax.text(2.1e5, iqp / 1.30, "beats best IQP", fontsize=7.0, color="#5a6b78",
                    ha="right", va="top", zorder=6)

    handles = [Line2D([], [], color=c, lw=1.3, marker=m, ms=3.4, mec="white", mew=0.4, label=l)
               for _a, l, c, m in ARMS]
    handles.append(Line2D([], [], color=S.INK_MUTED, lw=1.0, label="best IQP (Table 2)"))
    handles.append(Line2D([], [], color=S.INK, lw=1.1, ls="-", label="PDF--JS (left)"))
    handles.append(Line2D([], [], color=S.INK, lw=1.0, ls=DASH,
                          label="MMD$^2_{\\mathrm{al}}$ (right)"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.52, 1.045), ncol=5,
               fontsize=7.0, handletextpad=0.4, columnspacing=1.0, borderpad=0.15,
               borderaxespad=0.0)
    return S.save(fig, outdir, "figA_convergence")


if __name__ == "__main__":
    make(os.path.dirname(os.path.abspath(__file__)))
