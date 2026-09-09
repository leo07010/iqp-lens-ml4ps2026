# -*- coding: utf-8 -*-
"""figA_scaling_mlp -- the data-scaling panel with the MLP arm added to the three published
curves.  Solid = Conv2D-MMD^2, dashed = its flux-aligned variant, band = +- 1 sd over seeds.
The three published curves are read from results_seeds_scaling_v11.jsonl and not recomputed.
"""
from __future__ import annotations

import os
import re
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _mlpsuite_common import EXP, COLOURS, read, is_snap, mean_sd, save   # noqa: E402

SCALE = os.path.join(EXP, "results_seeds_scaling_v11.jsonl")
SUITE = os.path.join(EXP, "results_seeds_mlp_suite.jsonl")
MLP_RE = re.compile(r"^IQP-mlpH(\d+)-M(\d+)-MMDflat-resample$")
NS = (10, 30, 50, 70, 90, 100)


def curve(rows, arm, key):
    xs, mu, sd = [], [], []
    for n in NS:
        vals = [r.get(key) for r in rows
                if r.get("arm") == arm and int(r.get("n_train", -1)) == n
                and r.get("class_name") is None]
        m, s, k = mean_sd(vals)
        if k:
            xs.append(n); mu.append(m); sd.append(s)
    return np.asarray(xs), np.asarray(mu), np.asarray(sd)


def main():
    pub = read(SCALE)
    suite = [r for r in read(SUITE) if not is_snap(r.get("arm"))]
    counts = {}
    for r in suite:
        m = MLP_RE.match(str(r.get("arm") or ""))
        if m and int(r.get("n_train", -1)) in NS and r.get("class_name") is None:
            counts[r["arm"]] = counts.get(r["arm"], 0) + 1
    mlp_arm = max(counts, key=counts.get) if counts else None
    series = [("affine IQP", pub, "IQP-affine-M4-MMDflat-resample", COLOURS["affine"]),
              ("RBM-10bit", pub, "RBM-10bit", COLOURS["rbm"]),
              ("DCGAN-32", pub, "DCGAN-32", COLOURS["dcgan"])]
    if mlp_arm:
        series.insert(0, ("MLP $g(z)$ (%s)" % mlp_arm.replace("IQP-mlp", "").replace("-MMDflat-resample", ""),
                          suite, mlp_arm, COLOURS["mlp"]))
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    nseeds = []
    for label, rows, arm, col in series:
        x, m, s = curve(rows, arm, "conv_mmd2")
        if x.size == 0:
            print("[figA] no points for %s" % arm)
            continue
        ax.plot(x, m, "-o", ms=2.5, lw=1.0, color=col, label=label)
        ax.fill_between(x, m - s, m + s, color=col, alpha=0.15, lw=0)
        xa, ma, sa = curve(rows, arm, "conv_mmd2_aligned")
        if xa.size:
            ax.plot(xa, ma, "--", lw=0.9, color=col)
            ax.fill_between(xa, ma - sa, ma + sa, color=col, alpha=0.08, lw=0, hatch="///")
        nseeds.append("%s n=%d" % (label.split(" (")[0], len(
            [r for r in rows if r.get("arm") == arm and int(r.get("n_train", -1)) == x[-1]])))
    ax.set_xlabel("training images $N$")
    ax.set_ylabel("Conv2D-MMD$^2$")
    ax.set_yscale("log")
    handles, labels = ax.get_legend_handles_labels()
    handles += [Line2D([], [], color="0.3", lw=1.0, ls="-"),
                Line2D([], [], color="0.3", lw=0.9, ls="--")]
    labels += ["raw", "flux-aligned"]
    ax.legend(handles, labels, loc="upper right", frameon=False, ncol=2)
    fig.text(0.0, -0.12, "mean $\\pm$ 1 sd over seeds; " + ", ".join(nseeds), fontsize=5.2,
             color="0.35")
    fig.tight_layout()
    save(fig, "figA_scaling_mlp")
    print("[figA] MLP arm: %s" % mlp_arm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
