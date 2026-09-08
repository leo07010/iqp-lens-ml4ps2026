#!/usr/bin/env python3
"""DCGAN / IQP-affine / RBM data-scaling figure: N in {10, 30, 50, 70, 90, 100}.

docs/IQP_LENS_PAPER_SUITE_PLAN.md, row "DCGAN data-scaling N in {10, 30, 50, 70, 90, 100}" ->
"DCGAN-32 on the crops, plus IQP-affine-M4-MMDflat-resample and RBM-10bit at the same N".
Prediction Q4: IQP's MMD^2 is flat in N while DCGAN improves with N and overtakes IQP between
N = 30 and 50 (raw MMD^2).

Reads results_scaling.jsonl (evaluate_r3b.py's schema, written by submit_scaling.sbatch after
all 18 -- or fewer, if some N=100 runs were reused from round 3/3b, see the sbatch header --
arm x N runs finish) and plots, for each of the three scaling arms, three panels vs N:
  conv_mmd2          "MMD^2"            (evaluate_r3.conv_mmd2, raw intensity scale)
  conv_mmd2_aligned  "MMD^2 (aligned)"  (each image individually rescaled to the held-out mean
                                         total intensity first -- removes the flux-scale part of
                                         the distance, see evaluate_r3.align_to_target)
  pdf_js             "PDF-JS"           (evaluate_r3.pdf_js)

NEW file, reads but does not modify evaluate_r3b.py, evaluate_r3.py or results_scaling.jsonl.

Python 3.9 compatible (no `X | None`).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))          # .../paper_suite/scaling -> .../13_iqp_turbulence_replication
sys.path.insert(0, ROOT)

# ---------------------------------------------------------------------------------------
# Palette: same validated categorical slots / chart chrome as plot_pseudotime.py and
# analysis_lens/make_figures.py. Fixed arm -> slot assignment (never re-cycled if an arm is
# dropped): IQP = slot 1 (blue, the paper's own generative method), DCGAN = slot 2 (orange),
# RBM = slot 3 (aqua).
# ---------------------------------------------------------------------------------------
S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SURF = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": BASE, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID,
    "axes.grid": True, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "axes.titlesize": 10.5, "legend.frameon": False,
    "lines.linewidth": 2.0, "axes.titlelocation": "left",
})

# arm -> (display label, palette slot). Order fixed regardless of which arms are present in a
# given results file (a smoke run may carry only a subset).
ARMS = [
    ("IQP-affine-M4-MMDflat-resample", "IQP-affine-M4-MMDflat-resample", S[0]),
    ("DCGAN-32", "DCGAN-32", S[1]),
    ("RBM-10bit", "RBM-10bit", S[2]),
]
METRICS = [
    ("conv_mmd2", u"MMD²"),
    ("conv_mmd2_aligned", u"MMD² (aligned)"),
    ("pdf_js", "PDF–JS"),
]

DEFAULT_RESULTS = os.path.join(ROOT, "results_scaling.jsonl")
DEFAULT_OUT = "<HOME>/gsoc2026-deeplense-qml/docs/figures/iqp_lens_scaling.png"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="DCGAN / IQP-affine / RBM N-scaling figure.")
    p.add_argument("--results", default=DEFAULT_RESULTS)
    p.add_argument("--out", default=DEFAULT_OUT)
    return p.parse_args(argv)


def load_rows(path):
    if not os.path.isfile(path):
        raise SystemExit("plot_scaling.py: no such results file %r" % path)
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def series_for(rows, arm_name, metric_key):
    """(N array, value array), sorted by N, one point per N (last row wins if duplicated)."""
    by_n = {}
    for r in rows:
        if r.get("arm") != arm_name:
            continue
        v = r.get(metric_key)
        if v is None:
            continue
        try:
            n = int(r["n_train"])
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(fv):
            continue
        by_n[n] = fv                        # last write wins; a duplicate N is a data problem
    ns = sorted(by_n)
    return np.asarray(ns, dtype=np.int64), np.asarray([by_n[n] for n in ns], dtype=np.float64)


def main(argv=None):
    args = parse_args(argv)
    rows = load_rows(args.results)
    scored = [r for r in rows if not str(r.get("arm", "")).startswith(("HOLDOUT", "RANDOM"))]
    print("[plot_scaling] %d rows total, %d scored (non-reference) rows in %r"
          % (len(rows), len(scored), args.results))

    present_arms = [(key, label, color) for key, label, color in ARMS
                    if any(r.get("arm") == key for r in rows)]
    if not present_arms:
        raise SystemExit("plot_scaling.py: none of the %d expected arms %r were found in %r "
                          "(arms present: %s)"
                          % (len(ARMS), [a[0] for a in ARMS], args.results,
                             sorted(set(str(r.get("arm")) for r in rows))))
    missing = [a[0] for a in ARMS if a not in present_arms]
    if missing:
        print("[plot_scaling] NOTE: arm(s) %s not found in %r -- plotting the rest" % (missing, args.results))

    fig, axes = plt.subplots(1, len(METRICS), figsize=(5.1 * len(METRICS), 4.0))
    if len(METRICS) == 1:
        axes = [axes]

    n_all = sorted(set(int(r["n_train"]) for r in scored if r.get("n_train") is not None))
    for ax, (metric_key, metric_label) in zip(axes, METRICS):
        any_line = False
        for arm_key, arm_label, color in present_arms:
            ns, vs = series_for(rows, arm_key, metric_key)
            if ns.size == 0:
                print("[plot_scaling] NOTE: arm %r has no finite %r values -- skipped in this panel"
                      % (arm_key, metric_key))
                continue
            marker = "o" if ns.size > 1 else "D"
            ax.plot(ns, vs, color=color, marker=marker, ms=5.5, lw=2.0, label=arm_label)
            any_line = True
            for n, v in zip(ns.tolist(), vs.tolist()):
                print("[plot_scaling]   %-32s N=%-4d %s=%.6g" % (arm_key, n, metric_key, v))
        ax.set_xlabel("N (training crops)")
        ax.set_ylabel(metric_label)
        ax.set_title(metric_label, color=INK)
        if n_all:
            ax.set_xticks(n_all)
        if not any_line:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", color=MUTED,
                    transform=ax.transAxes)
    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle("Data-scaling: DCGAN-32 vs IQP-affine-M4-MMDflat-resample vs RBM-10bit "
                "(lens crops, seed 0)", fontsize=11, color=INK, y=1.02)
    fig.tight_layout()

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(args.out, dpi=140, facecolor=SURF, bbox_inches="tight")
    plt.close(fig)
    print("[plot_scaling] wrote %s (%d KB)" % (args.out, os.path.getsize(args.out) // 1024))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
