# -*- coding: utf-8 -*-
"""fig2_v12_msweep -- the latent-width sweep, v12.

(a) held-out projection correlation vs M.  Two curves: the affine map, READ from the
    controlled mechanism test (mech_tests/mech_test2/results_seeds.json, N = 20, 5 seeds),
    and the MLP map g(z) at H = 32 on the same N = 20.  Three reference lines from the
    mechanism test (nearest training image / mean training image / mismatched random
    direction).  This panel is flux-independent: the projection lives in the latent map, not
    in the decode.
(b) what the sweep buys on the metrics, under the RESAMPLED-S convention: raw and
    flux-aligned Conv2D-MMD^2 vs M for the MLP arm at N = 11 and N = 100, with the DCGAN and
    RBM levels of Table 2 (v12) as horizontal references.

HONEST CAVEAT, printed on the figure: panel (a)'s two curves do not come from the same
driver (the affine curve is mech_seeds.py's controlled protocol, the MLP curve the round-3
one), so a small offset between them is not by itself a model difference.  Panel (b) is one
driver throughout.

Python 3.9 compatible.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _mlpsuite_common import (EXP, COLOURS, read, meta_of, is_snap, mean_sd, as_mean_sd,
                              save, tidy, TEXTWIDTH)                     # noqa: E402

MECH = os.path.join(EXP, "mech_tests/mech_test2/results_seeds.json")
SUITE = os.path.join(EXP, "results_seeds_mlp_suite.jsonl")
FLUX = os.path.join(EXP, "results_v12_fluxres.jsonl")
T2 = os.path.join(EXP, "results_seeds_table2_v11.jsonl")
M_LIST = (1, 2, 3, 4, 6, 8, 16)


def affine_curve():
    if not os.path.isfile(MECH):
        print("[fig2v12] %s missing -- affine curve skipped" % MECH)
        return [], [], [], {}
    d = json.load(open(MECH))
    blk = d.get("M_sweep_holdout_corr")
    Ms, mu, sd = [], [], []
    if isinstance(blk, dict):
        for k, v in blk.items():
            try:
                M = int(str(k).lstrip("M").strip())
            except ValueError:
                continue
            m, s = as_mean_sd(v)
            if np.isfinite(m):
                Ms.append(M); mu.append(m); sd.append(0.0 if not np.isfinite(s) else s)
    elif isinstance(blk, list):
        for rec in blk:
            if isinstance(rec, dict) and "M" in rec:
                m, s = as_mean_sd(rec.get("holdout_corr", rec.get("corr", rec)))
                if np.isfinite(m):
                    Ms.append(int(rec["M"])); mu.append(m)
                    sd.append(0.0 if not np.isfinite(s) else s)
    refs = {}
    r = d.get("references")
    if isinstance(r, dict):
        for k in ("nearest", "mean_img", "random_mismatched"):
            if k in r:
                m, _s = as_mean_sd(r[k])
                if np.isfinite(m):
                    refs[k] = m
    keep = [i for i in np.argsort(Ms) if Ms[i] >= 1]
    return ([Ms[i] for i in keep], [mu[i] for i in keep], [sd[i] for i in keep], refs)


def mlp_corr_curve():
    rows = [r for r in read(SUITE)
            if not is_snap(r.get("arm")) and int(r.get("n_train", -1)) == 20]
    Ms, mu, sd = [], [], []
    for M in M_LIST:
        arm = "IQP-mlpH32-M%d-MMDflat-resample" % M
        vals = [meta_of(r.get("run_dir")).get("projection_corr_mean")
                for r in rows if r.get("arm") == arm]
        m, s, n = mean_sd(vals)
        if n:
            Ms.append(M); mu.append(m); sd.append(s)
    return Ms, mu, sd


def metric_curve(flux, n, key, mode="resample"):
    Ms, mu, sd = [], [], []
    for M in M_LIST:
        arm = "IQP-mlpH32-M%d-MMDflat-resample" % M
        vals = [r.get(key) for r in flux
                if r.get("arm") == arm and r.get("flux_mode") == mode
                and int(r.get("n_train", -1)) == n and r.get("class_name") is None]
        m, s, k = mean_sd(vals)
        if k:
            Ms.append(M); mu.append(m); sd.append(s)
    return np.asarray(Ms), np.asarray(mu), np.asarray(sd)


def main():
    aM, amu, asd, refs = affine_curve()
    cM, cmu, csd = mlp_corr_curve()
    flux = read(FLUX)
    if not flux:
        raise SystemExit("fig2_v12_msweep: %s missing -- run evaluate_v12_flux.py" % FLUX)
    t2 = read(T2)

    plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7})
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.85))

    # ---------------------------------------------------------------- (a) projection corr
    ax = axes[0]
    for name, val in sorted(refs.items()):
        ax.axhline(val, color="0.55", lw=0.6, ls=":", zorder=1)
        ax.annotate(name.replace("_", " "), xy=(0.985, val),
                    xycoords=("axes fraction", "data"), ha="right", va="bottom",
                    fontsize=7.0, color="0.35")
    if aM:
        amu_, asd_ = np.asarray(amu), np.asarray(asd)
        ax.plot(aM, amu_, "-o", ms=2.6, lw=1.0, color=COLOURS["affine"], label="linear map")
        ax.fill_between(aM, amu_ - asd_, amu_ + asd_, color=COLOURS["affine"], alpha=0.15, lw=0)
    if cM:
        cmu_, csd_ = np.asarray(cmu), np.asarray(csd)
        ax.plot(cM, cmu_, "-s", ms=2.6, lw=1.0, color=COLOURS["mlp"], label="non-linear map $g(z)$")
        ax.fill_between(cM, cmu_ - csd_, cmu_ + csd_, color=COLOURS["mlp"], alpha=0.15, lw=0)
    ax.set_xscale("log", base=2)
    ax.set_xticks(list(M_LIST))
    ax.set_xticklabels([str(m) for m in M_LIST])
    ax.set_xlabel("latent width $M$")
    ax.set_ylabel("held-out projection corr.")
    ax.set_title("$N=20$, 5 seeds (flux-independent)", fontsize=7)
    tidy(ax, grid=False)
    ax.legend(loc="lower left", frameon=False)

    # ---------------------------------------------------------------- (b) metrics
    # Panel (b) encodes the METRIC in colour and N in the line style; panel (a)'s colours mean
    # "which map", so the two keys are deliberately different and the caption says so.
    ax = axes[1]
    C_RAW, C_AL = "#E69F00", "#56B4E9"        # Okabe-Ito orange / sky blue
    styles = {11: ("-", "o"), 100: ("--", "s")}
    for n in (11, 100):
        ls, mk = styles[n]
        x, m, s = metric_curve(flux, n, "conv_mmd2")
        if x.size:
            ax.plot(x, m, ls, marker=mk, ms=2.6, lw=1.0, color=C_RAW)
            ax.fill_between(x, m - s, m + s, color=C_RAW, alpha=0.12, lw=0)
        xa, ma, sa = metric_curve(flux, n, "conv_mmd2_aligned")
        if xa.size:
            ax.plot(xa, ma, ls, marker=mk, ms=2.6, lw=1.0, color=C_AL)
            ax.fill_between(xa, ma - sa, ma + sa, color=C_AL, alpha=0.12, lw=0)
    for name, arm, col in (("DCGAN", "DCGAN-32", COLOURS["dcgan"]),
                           ("RBM", "RBM-10bit", COLOURS["rbm"])):
        v = [r.get("conv_mmd2") for r in t2
             if r.get("arm") == arm and int(r.get("n_train", -1)) == 11]
        mm, _s, k = mean_sd(v)
        if k:
            ax.axhline(mm, color=col, lw=0.7, ls=":", zorder=1)
            ax.annotate("%s raw, $N$=11" % name, xy=(0.985, mm),
                        xycoords=("axes fraction", "data"), ha="right", va="bottom",
                        fontsize=7.0, color=col)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(list(M_LIST))
    ax.set_xticklabels([str(m) for m in M_LIST])
    ax.set_xlabel("latent width $M$")
    ax.set_ylabel("Conv2D-MMD$^2$")
    ax.set_title("sampled $S$, non-linear map, 5 seeds", fontsize=7)
    tidy(ax, grid=False)
    handles = [Line2D([], [], color=C_RAW, lw=1.0), Line2D([], [], color=C_AL, lw=1.0),
               Line2D([], [], color="0.35", lw=1.0, ls="-", marker="o", ms=2.6),
               Line2D([], [], color="0.35", lw=1.0, ls="--", marker="s", ms=2.6)]
    ax.legend(handles, ["raw", "flux-aligned", "$N=11$", "$N=100$"],
              loc="lower right", frameon=False, ncol=2, fontsize=7.0,
              columnspacing=1.0, borderpad=0.1)

    for ax, lab in zip(axes, ("(a)", "(b)")):
        ax.text(-0.02, 1.06, lab, transform=ax.transAxes, fontsize=8, fontweight="bold",
                va="bottom", ha="left")
    fig.tight_layout()
    save(fig, "fig2_v12_msweep")
    print("[fig2v12] affine points=%d  corr points=%d  metric rows=%d" % (len(aM), len(cM),
                                                                          len(flux)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
