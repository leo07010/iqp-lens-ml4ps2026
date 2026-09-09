# -*- coding: utf-8 -*-
"""fig2_mlp_sweep -- the MLP counterpart of Fig. 2(a): held-out projection vs latent width M.

Two curves.  The affine one is READ from mech_tests/mech_test2/results_seeds.json (the
controlled N = 20 mechanism test, 5 seeds) and is not recomputed.  The MLP one is this
suite's own N = 20 runs of train_iqp_affine_mlp.py, H = 32, 5 seeds.

HONEST CAVEAT, printed on the figure: the two curves do not come from the same driver.  The
affine curve is mech_seeds.py's controlled protocol; the MLP curve is the round-3 protocol
(data_lens_r3, 4,000 steps, resample prior) at the same N = 20.  They share the reference
lines and the axis, not the pipeline, so a small offset between them is not by itself a
model difference.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _mlpsuite_common import (EXP, COLOURS, read, meta_of, is_snap, mean_sd, as_mean_sd,
                              save, tidy)                              # noqa: E402

MECH = os.path.join(EXP, "mech_tests/mech_test2/results_seeds.json")
SUITE = os.path.join(EXP, "results_seeds_mlp_suite.jsonl")
M_LIST = (1, 2, 3, 4, 6, 8, 16)


def affine_curve():
    """(Ms, means, sds, refs) from the mechanism test, or empty if it cannot be parsed."""
    if not os.path.isfile(MECH):
        print("[fig2] %s missing -- affine curve skipped" % MECH)
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
                    Ms.append(int(rec["M"])); mu.append(m); sd.append(0.0 if not np.isfinite(s) else s)
    # the published panel draws three lines: nearest training image, mean training image, and
    # the random-direction control.  results_seeds.json carries a fourth ("random", the paired
    # variant); it is left out so this panel and Fig. 2(a) show the same three.
    WANT = ("nearest", "mean_img", "random_mismatched")
    refs = {}
    r = d.get("references")
    if isinstance(r, dict):
        for k in WANT:
            if k in r:
                m, _s = as_mean_sd(r[k])
                if np.isfinite(m):
                    refs[str(k)] = m
    # M = 0 (no latent directions at all) exists in the mechanism test but cannot be placed on
    # a log-spaced M axis, so it is dropped here rather than silently clipped to the left edge.
    keep = [i for i in np.argsort(Ms) if Ms[i] >= 1]
    return ([Ms[i] for i in keep], [mu[i] for i in keep], [sd[i] for i in keep], refs)


def mlp_curve():
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


def main():
    aM, amu, asd, refs = affine_curve()
    mM, mmu, msd = mlp_curve()
    if not mM and not aM:
        raise SystemExit("fig2_mlp_sweep: neither curve has data")
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    for name, val in sorted(refs.items()):
        ax.axhline(val, color="0.55", lw=0.6, ls=":", zorder=1)
        ax.annotate(name.replace("_", " "), xy=(0.985, val), xycoords=("axes fraction", "data"),
                    ha="right", va="bottom", fontsize=5.5, color="0.35")
    if aM:
        amu, asd = np.asarray(amu), np.asarray(asd)
        ax.plot(aM, amu, "-o", ms=2.6, lw=1.0, color=COLOURS["affine"],
                label="affine (mech. test, N=20)")
        ax.fill_between(aM, amu - asd, amu + asd, color=COLOURS["affine"], alpha=0.15, lw=0)
    if mM:
        mmu, msd = np.asarray(mmu), np.asarray(msd)
        ax.plot(mM, mmu, "-s", ms=2.6, lw=1.0, color=COLOURS["mlp"],
                label="MLP $g(z)$, H=32 (round-3, N=20)")
        ax.fill_between(mM, mmu - msd, mmu + msd, color=COLOURS["mlp"], alpha=0.15, lw=0)
    ax.set_xscale("log", base=2)
    ax.set_xticks(list(M_LIST))
    ax.set_xticklabels([str(m) for m in M_LIST])
    ax.set_xlabel("latent width $M$")
    ax.set_ylabel("held-out projection correlation")
    tidy(ax, grid=False)
    ax.legend(loc="lower left", frameon=False)
    ax.set_title("N = 20, mean $\\pm$ sd over 5 seeds (M = 0 omitted: log axis)", fontsize=7)
    fig.text(0.0, -0.15, "Different drivers: the affine curve is mech_seeds.py's controlled "
             "protocol, the MLP curve the round-3 one, both on the same 20 training and 20 "
             "held-out frames.\nThe reference lines are shared.", fontsize=5.2, color="0.35")
    fig.tight_layout()
    save(fig, "fig2_mlp_sweep")
    print("[fig2] affine points=%d, MLP points=%d, references=%d" % (len(aM), len(mM), len(refs)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
