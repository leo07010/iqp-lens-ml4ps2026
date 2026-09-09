#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sample grids for the v12 grid: held-out images next to the best v12 combination's output.

NEW file.  Writes docs/figures/diag_v12_*.png only.  It picks the arm to draw from
results_seeds_v12.jsonl by the same rank rule the aggregator's verdict uses (mean rank over
PDF-JS, raw MMD^2 and aligned MMD^2 at the requested N), or takes an explicit --arm.

    python3 v12_figs.py                      # best arm at N = 11, seed 0
    python3 v12_figs.py --n-train 11 --arm IQP-v12-M4-multi-fluxres-resample

Python 3.9 compatible.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
import data_lens_r3 as D                 # noqa: E402

RESULTS = os.path.join(REPO, "results_seeds_v12.jsonl")
FIGDIR = os.path.join(os.path.dirname(os.path.dirname(REPO)), "docs", "figures")
RANK_COLS = ("pdf_js", "conv_mmd2", "conv_mmd2_aligned")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default=RESULTS)
    p.add_argument("--n-train", type=int, default=11)
    p.add_argument("--arm", default=None, help="explicit arm name (default: best by mean rank)")
    p.add_argument("--seed", type=int, default=0, help="which seed's run dir to draw")
    p.add_argument("--n-show", type=int, default=8, help="images per row")
    p.add_argument("--out-prefix", default=None)
    return p.parse_args(argv)


def pick_arm(rows, n):
    """Mean rank over the three verdict metrics, on the seed-averaged means."""
    agg = collections.defaultdict(list)
    for r in rows:
        if int(r.get("n_train", -1)) == n and str(r.get("arm", "")).startswith("IQP-v12"):
            agg[r["arm"]].append(r)
    if not agg:
        raise SystemExit("v12_figs: no IQP-v12 rows at N=%d in the results file" % n)
    means = {a: {c: float(np.mean([x[c] for x in v])) for c in RANK_COLS} for a, v in agg.items()}
    ranks = collections.defaultdict(float)
    for c in RANK_COLS:
        for i, a in enumerate(sorted(means, key=lambda k: means[k][c])):
            ranks[a] += i + 1
    best = min(ranks, key=lambda a: ranks[a])
    print("[v12-figs] best arm at N=%d by mean rank: %s (mean rank %.2f over %d arms)"
          % (n, best, ranks[best] / len(RANK_COLS), len(means)))
    return best


def load_gen(run_dir, k):
    out = []
    for i in range(k):
        p = os.path.join(run_dir, "generated_%03d.npy" % i)
        if os.path.isfile(p):
            out.append(np.load(p))
    return np.stack(out, axis=0) if out else None


def grid(fig_path, rows_of_images, row_titles, vmax, suptitle):
    nrow = len(rows_of_images)
    ncol = max(len(r) for r in rows_of_images)
    fig, axes = plt.subplots(nrow, ncol, figsize=(1.35 * ncol, 1.5 * nrow), squeeze=False)
    for i, imgs in enumerate(rows_of_images):
        for j in range(ncol):
            ax = axes[i][j]
            ax.set_xticks([]); ax.set_yticks([])
            if j < len(imgs):
                ax.imshow(imgs[j], cmap="inferno", vmin=0.0, vmax=vmax, interpolation="nearest")
                if j == 0:
                    ax.set_ylabel(row_titles[i], fontsize=7, rotation=90, labelpad=2)
                ax.set_title("%.0f" % float(imgs[j].sum()), fontsize=6, pad=1.5)
            else:
                ax.axis("off")
    fig.suptitle(suptitle, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(fig_path, dpi=170)
    plt.close(fig)
    print("[v12-figs] wrote %s" % fig_path)


def main(argv=None):
    a = parse_args(argv)
    os.makedirs(FIGDIR, exist_ok=True)
    rows = [json.loads(l) for l in open(a.results) if l.strip()]
    arm = a.arm or pick_arm(rows, a.n_train)
    cand = [r for r in rows if r.get("arm") == arm and int(r.get("n_train", -1)) == a.n_train]
    if not cand:
        raise SystemExit("v12_figs: arm %r not found at N=%d" % (arm, a.n_train))
    pick = sorted(cand, key=lambda r: abs(int(r.get("seed", 0)) - a.seed))[0]
    run_dir = pick["run_dir"]
    meta = json.load(open(os.path.join(run_dir, "meta.json")))
    # the published-protocol arm at the same N and seed, for a side-by-side
    ref_dir = os.path.join(REPO, "outputs_lens_r3b", "iqp_m4_mmdflat_resample_n%d" % a.n_train)

    d = D.load_r3(D.N_TRAIN_MAX, n_holdout=200)
    ho = d.holdout_images[:a.n_show]
    gen = load_gen(run_dir, a.n_show)
    vmax = float(np.percentile(np.concatenate([ho.ravel(), gen.ravel()]), 99.5))

    rows_img = [ho, gen]
    titles = ["held-out", "v12"]
    if os.path.isdir(ref_dir):
        old = load_gen(ref_dir, a.n_show)
        if old is not None:
            rows_img.append(old)
            titles.append("v11 flat")

    pref = a.out_prefix or os.path.join(FIGDIR, "diag_v12_samples_n%d" % a.n_train)
    grid(pref + ".png", rows_img, titles, vmax,
         "%s  (N=%d, seed %s)  |  PDF-JS %.2e  MMD2 %.4f  MMD2_al %.4f  gen_max %.2f\n"
         "titles = each image's total intensity; held-out mean %.1f"
         % (arm, a.n_train, pick.get("seed"), pick["pdf_js"], pick["conv_mmd2"],
            pick["conv_mmd2_aligned"], pick["gen_max"], pick["holdout_mean_total"]))

    # flux histogram: the mechanism lever C attacks
    ho_tot = d.holdout_images.reshape(d.holdout_images.shape[0], -1).sum(axis=1)
    g_all = load_gen(run_dir, int(pick["n_gen"]))
    g_tot = g_all.reshape(g_all.shape[0], -1).sum(axis=1)
    fig, ax = plt.subplots(figsize=(6.0, 3.1))
    bins = np.linspace(min(ho_tot.min(), g_tot.min()) * 0.9,
                       max(ho_tot.max(), g_tot.max()) * 1.05, 40)
    ax.hist(ho_tot, bins=bins, alpha=0.55, label="held-out (200)", color="#3b6ea5")
    ax.hist(g_tot, bins=bins, alpha=0.55, label="generated (%d)" % g_tot.size, color="#c1663a")
    ax.axvline(float(meta.get("S", np.nan)), color="k", ls="--", lw=1,
               label="training mean S = %.1f" % float(meta.get("S", np.nan)))
    ax.set_xlabel("total intensity of the frame")
    ax.set_ylabel("count")
    ax.set_title("flux distribution -- N=%d, flux=%s\n%s"
                 % (a.n_train, meta.get("flux_mode"), arm), fontsize=7)
    ax.legend(fontsize=6)
    fig.tight_layout()
    fp = os.path.join(FIGDIR, "diag_v12_flux_n%d.png" % a.n_train)
    fig.savefig(fp, dpi=170)
    plt.close(fig)
    print("[v12-figs] wrote %s" % fp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
