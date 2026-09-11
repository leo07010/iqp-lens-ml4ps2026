#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fig2_compact_seeds.pdf -- fig2_compact with panel (a) redrawn over 5 model seeds.

Panel (a) is the same M sweep, but each point is the mean over 5 model seeds of the held-out
projection correlation with a +-1 sd band, from
    experiments/13_iqp_turbulence_replication/mech_tests/mech_test2/results_seeds.json
(mech_seeds.py + agg_mech_seeds.py; seed 0 of that file reproduces the published
results.json / diag3.json / diag.json values to 0.0e+00).  The seed moves the model only:
theta0's random start, Phi's draw and z's draw.  The data split is pinned at
data_lens_r3.R3_SPLIT_SEED for every seed, so all five see the same 20 training and 20
held-out images.

The three reference levels are drawn the way their own uncertainty warrants:
    nearest training image, mean training image -- functions of the fixed split alone, so they
        are exact constants (sd 0.0000 across the five seeds, verified, not assumed);
    random direction -- an untrained Phi projected onto held-out images, drawn as a mean line
        with a +-1 sd band.  The variant drawn is the correctly PAIRED one: the untrained Phi
        sits on the M = 0 theta0, i.e. the offset that was itself fitted without any latent
        (diag2.json "random Phi(seed 99) + M=0 theta0", 0.746 at seed 0).  The seed-0 figure
        used "random Phi + TRAINED M=4 theta0" (0.672) instead, which is a mismatched pairing,
        because that theta0 was optimised jointly with a trained Phi and is not a fair offset
        for an untrained direction.  The mismatched level is still available in
        results_seeds.json as references.random_mismatched.

Panel (b) is unchanged and is imported from fig2_compact so the two cannot drift.
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
import fig2_compact as F2C
import fig2_mechanism_scaling as F2

SEEDS_JSON = os.path.join(F2.MECH, "results_seeds.json")
FIG_W, FIG_H = F2C.FIG_W, F2C.FIG_H      # reuse panel_b's geometry verbatim
MS = [0, 2, 4, 6, 8, 16]


def rect(x, y, w, h):
    return [x / FIG_W, y / FIG_H, w / FIG_W, h / FIG_H]


def load_mech_seeds():
    with open(SEEDS_JSON) as f:
        d = json.load(f)
    sw = d["M_sweep_holdout_corr"]
    ms = np.asarray(MS, dtype=float)
    mean = np.asarray([sw[str(m)]["mean"] for m in MS])
    sd = np.asarray([sw[str(m)]["sd"] for m in MS])
    r = d["references"]
    refs = {k: (float(r[k]["mean"]), float(r[k]["sd"]))
            for k in ("nearest", "mean_img", "random", "random_mismatched")}
    return ms, mean, sd, refs, int(d["meta"]["n_seeds"])


def panel_a(fig, ms, mean, sd, refs, n_seeds):
    ax = fig.add_axes(rect(0.62, 0.44, 2.02, 1.34))
    grey = S.INK_MUTED

    m_near, s_near = refs["nearest"]
    ax.axhline(m_near, color=grey, lw=0.9, ls=(0, (4, 2)), zorder=1)
    ax.text(17.3, m_near + 0.015, "nearest training image", fontsize=7.0, color=grey,
            ha="right", va="bottom", zorder=4)

    # The paired random-direction level and the mean-image level COINCIDE to ~0.002 -- an
    # untrained direction on a latent-free theta0 buys exactly what the mean training image
    # buys, and M = 0 sits there too.  Drawing them as two full-width lines would read as one
    # line, so each is drawn on the half of the axis where its own label sits and they overlap
    # only in the middle; the values are in the table, the point of the panel is the level.
    m_mean, s_mean = refs["mean_img"]
    ax.axhline(m_mean, color=S.OKABE_ITO["orange"], lw=0.9, ls=(0, (4, 2)), zorder=1)
    ax.text(17.3, m_mean + 0.018, "mean train. image", fontsize=7.0,
            color=S.OKABE_ITO["orange"], ha="right", va="bottom", zorder=4)

    m_rnd, s_rnd = refs["random"]
    ax.axhspan(m_rnd - s_rnd, m_rnd + s_rnd, color=S.OKABE_ITO["vermilion"], alpha=0.13,
               lw=0, zorder=1)
    ax.plot([-1.1, 7.4], [m_rnd, m_rnd], color=S.OKABE_ITO["vermilion"], lw=1.1,
            ls=(0, (1.4, 1.6)), zorder=2, solid_capstyle="butt")
    ax.text(-0.7, m_rnd - 0.055, "random dir.", fontsize=7.0,
            color=S.OKABE_ITO["vermilion"], ha="left", va="top", zorder=4)
    ax.annotate("", xy=(0.35, m_rnd - 0.006), xytext=(0.35, m_rnd - 0.052),
                arrowprops=dict(arrowstyle="-", lw=0.6, color=S.OKABE_ITO["vermilion"]),
                zorder=4)

    c = S.OKABE_ITO["blue"]
    ax.fill_between(ms, mean - sd, mean + sd, color=c, alpha=0.20, lw=0, zorder=3)
    ax.plot(ms, mean, "-o", color=c, lw=1.4, ms=4.2, zorder=5, mec="white", mew=0.5)

    ax.set_xlabel("latent directions $M$")
    ax.set_ylabel("held-out\nprojection corr.", linespacing=1.2)
    ax.set_xticks([0, 2, 4, 6, 8, 16])
    ax.set_xlim(-1.1, 17.6)
    ax.set_ylim(-0.05, 1.12)
    ax.set_yticks([0, 0.5, 1.0])
    ax.text(0.5, 0.02, "mean $\\pm$ sd, %d seeds" % n_seeds, fontsize=6.6, color=grey,
            ha="left", va="bottom", zorder=6)
    S.tidy(ax)


def make(outdir):
    S.apply_style()
    ms, mean, sd, refs, n_seeds = load_mech_seeds()
    rows = F2.load_images()
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    panel_a(fig, ms, mean, sd, refs, n_seeds)
    F2C.panel_b(fig, rows)
    fig.text(0.06 / FIG_W, 1.90 / FIG_H, "(a)", fontsize=9, fontweight="bold",
             ha="left", va="bottom")
    fig.text(2.86 / FIG_W, 1.90 / FIG_H, "(b)", fontsize=9, fontweight="bold",
             ha="left", va="bottom")
    return S.save(fig, outdir, "fig2_compact_seeds")


if __name__ == "__main__":
    make(os.path.dirname(os.path.abspath(__file__)))
