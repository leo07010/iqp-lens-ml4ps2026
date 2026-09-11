# -*- coding: utf-8 -*-
"""figA_scaling_v12 -- data scaling, v12: four arms, raw and flux-aligned in separate panels.

Curves: the MLP latent map (H = 32, M = 2, flat MMD), the published affine IQP arm
(M = 4, flat MMD), the RBM, and the DCGAN.  The first three are drawn under the RESAMPLED-S
convention (results_v12_fluxres.jsonl); their fixed-S curves are drawn faintly behind so the
size of the convention change is visible rather than asserted.

DCGAN is the exception and is labelled as such on the figure: it learns a per-image flux, so
it has no fixed S to swap and only one curve exists for it.

Python 3.9 compatible.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _mlpsuite_common import (EXP, COLOURS, read, mean_sd, save, tidy,
                              TEXTWIDTH)                                 # noqa: E402

FLUX = os.path.join(EXP, "results_v12_fluxres.jsonl")
SCALE = os.path.join(EXP, "results_seeds_scaling_v11.jsonl")
NS = (10, 30, 50, 70, 90, 100)

SERIES = [("non-linear map, M=2", "IQP-mlpH32-M2-MMDflat-resample", COLOURS["mlp"], "o"),
          ("linear map, M=4", "IQP-affine-M4-MMDflat-resample", COLOURS["affine"], "s"),
          ("RBM", "RBM-10bit", COLOURS["rbm"], "^")]


def curve(rows, arm, key, mode=None):
    xs, mu, sd = [], [], []
    for n in NS:
        vals = [r.get(key) for r in rows
                if r.get("arm") == arm and int(r.get("n_train", -1)) == n
                and r.get("class_name") is None
                and (mode is None or r.get("flux_mode") == mode)]
        m, s, k = mean_sd(vals)
        if k:
            xs.append(n); mu.append(m); sd.append(s)
    return np.asarray(xs), np.asarray(mu), np.asarray(sd)


def main():
    flux = read(FLUX)
    if not flux:
        raise SystemExit("figA_scaling_v12: %s missing -- run evaluate_v12_flux.py" % FLUX)
    pub = read(SCALE)

    plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7})
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.75), sharex=True)
    counts = []
    for ax, key, title in ((axes[0], "conv_mmd2", "raw Conv2D-MMD$^2$"),
                           (axes[1], "conv_mmd2_aligned", "flux-aligned Conv2D-MMD$^2$")):
        for label, arm, col, mk in SERIES:
            xf, mf, _sf = curve(flux, arm, key, mode="fixed")
            if xf.size and key == "conv_mmd2":
                # the published convention, faint, so the size of the change is visible
                ax.plot(xf, mf, ":", lw=0.8, color=col, alpha=0.55, zorder=2)
            x, m, s = curve(flux, arm, key, mode="resample")
            if x.size == 0:
                continue
            ax.plot(x, m, "-", marker=mk, ms=2.6, lw=1.0, color=col, label=label, zorder=3)
            ax.fill_between(x, m - s, m + s, color=col, alpha=0.13, lw=0, zorder=1)
            if key == "conv_mmd2":
                counts.append("%s n=%d" % (label.split(",")[0], len(
                    [r for r in flux if r.get("arm") == arm and r.get("flux_mode") == "resample"
                     and int(r.get("n_train", -1)) == x[-1]
                     and r.get("class_name") is None])))
        xd, md, sd_ = curve(pub, "DCGAN-32", key)
        if xd.size:
            ax.plot(xd, md, "-", marker="D", ms=2.4, lw=1.0, color=COLOURS["dcgan"],
                    label="DCGAN (learns its own flux)", zorder=3)
            ax.fill_between(xd, md - sd_, md + sd_, color=COLOURS["dcgan"], alpha=0.13, lw=0,
                            zorder=1)
        ax.set_yscale("log")
        ax.set_xlabel("training images $N$")
        ax.set_title(title, fontsize=7)
        tidy(ax, grid=False)
    axes[0].set_ylabel("Conv2D-MMD$^2$")
    h, l = axes[0].get_legend_handles_labels()
    h.append(Line2D([], [], color="0.4", lw=0.8, ls=":"))
    l.append("same arm, published fixed $S$")
    # one legend for both panels: the series are identical and a second copy sat on top of
    # the MLP curve in (b)
    # one legend for both panels, placed under the axes so it cannot sit on the curves
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=7.0,
               bbox_to_anchor=(0.5, -0.015), columnspacing=1.4, handlelength=1.8)
    for ax, lab in zip(axes, ("(a)", "(b)")):
        ax.text(-0.02, 1.06, lab, transform=ax.transAxes, fontsize=8, fontweight="bold",
                va="bottom", ha="left")
    # The former in-figure footnote lived at 5 pt and stretched the saved bounding box to
    # 7.9 in, so LaTeX shrank the whole panel.  Its content now lives in the LaTeX caption.
    print("[figA] counts: %s" % ", ".join(counts))
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    save(fig, "figA_scaling_v12")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
