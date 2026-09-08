#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""figA_fig3.pdf -- the Fig. 3 support-mismatch control.

x  the target distribution's EXACT I_QCLI (targets are <=4-local IQP circuits sampled along
   an Adam trajectory; the reachable range tops out at 0.41)
y  the MMD^2 a <=2-local IQP student with G_n gates achieves after training on that target
Per-gate-count upper envelopes are the per-bin maximum over 8 equal bins in x.

No i.i.d.-uniform null line is drawn: that null (0.0696) lives on the SAMPLE-QCLI scale,
while this axis is the exact I_QCLI of the target pmf, so the two are not comparable.

Python 3.9 compatible.
"""
from __future__ import annotations

import json
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

import fig_style as S

EXP = "<HOME>/gsoc2026-deeplense-qml/experiments/13_iqp_turbulence_replication"
JSONL = os.path.join(EXP, "outputs_controlled", "results_fig3_v2.jsonl")

GATES = [(50, S.OKABE_ITO["blue"], "o"),
         (100, S.OKABE_ITO["vermilion"], "s"),
         (150, S.OKABE_ITO["green"], "^")]
NBINS = 8


def load():
    out, meta = {}, None
    rows = [json.loads(l) for l in open(JSONL)]
    for r in rows:
        if r.get("record") == "meta":
            meta = r
    for g, _c, _m in GATES:
        sel = [r for r in rows if r.get("record") == "fit" and r["learner_gates"] == g
               and np.isfinite(r["final_mmd2"]) and r["final_mmd2"] > 0]
        x = np.asarray([r["I_QCLI_exact"] for r in sel], float)
        y = np.asarray([r["final_mmd2"] for r in sel], float)
        o = np.argsort(x)
        out[g] = (x[o], y[o])
    return out, meta


def envelope(x, y, xmax):
    e = np.linspace(0.0, xmax, NBINS + 1)
    cx, cy = [], []
    for i in range(NBINS):
        m = (x >= e[i]) & (x <= e[i + 1]) if i == NBINS - 1 else (x >= e[i]) & (x < e[i + 1])
        if not m.any():
            continue
        j = int(np.argmax(y[m]))
        cx.append(float(x[m][j])); cy.append(float(y[m][j]))
    return np.asarray(cx), np.asarray(cy)


def make(outdir):
    S.apply_style()
    data, meta = load()
    xmax = max(x.max() for x, _y in data.values())

    fig = plt.figure(figsize=(S.COLWIDTH, 2.25))
    ax = fig.add_axes([0.205, 0.175, 0.775, 0.775])

    for g, col, mk in GATES:
        x, y = data[g]
        ax.plot(x, y, ls="none", marker=mk, ms=2.4, color=col, alpha=0.5, mew=0,
                zorder=3, label="$G_n = %d$" % g)
        ex, ey = envelope(x, y, xmax)
        ax.plot(ex, ey, "-", color=col, lw=1.2, zorder=4)

    ax.set_yscale("log")
    ax.set_xlim(-0.018, 0.445)
    ax.set_ylim(7e-5, 1.6e-1)
    ax.set_xlabel(r"target $I_{\mathrm{QCLI}}$ (exact, 0–1)")
    ax.set_ylabel(r"achieved MMD$^2$")
    ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4])
    S.tidy(ax)

    ax.text(0.985, 0.012, "targets: $n = 16$, $\\leq$4-local;\n"
                          "learners: $\\leq$2-local, $G_n$ gates;\n"
                          "reachable $I_{\\mathrm{QCLI}} \\leq %.2f$" % xmax,
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.0,
            color=S.INK_MUTED, linespacing=1.25, zorder=6)
    ax.legend(loc="upper left", bbox_to_anchor=(-0.015, 1.02), fontsize=7.0,
              handletextpad=0.3, labelspacing=0.25, borderpad=0.2, borderaxespad=0.0,
              markerscale=1.9)
    return S.save(fig, outdir, "figA_fig3")


if __name__ == "__main__":
    make(os.path.dirname(os.path.abspath(__file__)))
