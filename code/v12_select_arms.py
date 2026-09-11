#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rank the v12 arms at one N and print the top few as ready-to-run CLI option strings.

NEW file.  It exists so that the follow-up job (N = 5 / N = 100, and the 16,000-step schedule)
picks its arms from the measured grid instead of from a hand-copied list -- one less place a
configuration can be mistyped.  The CLI it prints is REBUILT from each run's own meta.json, so
what the follow-up runs is what the grid ran.

Rule: mean rank over the three columns the verdict is defined on (PDF-JS, raw MMD^2, aligned
MMD^2), on the seed-averaged means, ties broken toward FEWER parameters.  The selection is
made on held-out scores and is therefore optimistic -- that is stated in seed_tables_e.md and
is why the follow-up reports the mu = 0 / scale = 1 rows too.

    python3 v12_select_arms.py --top 6 --n-train 11
    python3 v12_select_arms.py --top 1 --n-train 11 --quiet     # just the winner

Output: one `tag|cli options` line per arm.

Python 3.9 compatible.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
RANK_COLS = ("pdf_js", "conv_mmd2", "conv_mmd2_aligned")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default=os.path.join(REPO, "results_seeds_v12.jsonl"))
    p.add_argument("--n-train", type=int, default=11)
    p.add_argument("--top", type=int, default=6)
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def cli_of(meta):
    """Rebuild the training CLI options of one run from its meta.json."""
    o = []
    if meta.get("encoding", "crop10") != "crop10":
        o += ["--encoding", str(meta["encoding"])]
    if meta.get("latent_map", "affine") == "mlp":
        o += ["--latent-map", "mlp", "--hidden", str(int(meta.get("hidden") or 8))]
    o += ["--M", str(int(meta["M"]))]
    obj = meta.get("objective", "mmdflat")
    o += ["--objective", obj]
    if obj == "multi" and float(meta.get("lam_scale", 1.0)) != 1.0:
        o += ["--lam-scale", "%g" % float(meta["lam_scale"])]
    if obj == "spatial" and float(meta.get("spatial_lam_flat") or 0.0) != 0.0:
        o += ["--spatial-flat", "auto",
              "--spatial-flat-scale", "%g" % float(meta.get("spatial_flat_scale", 1.0))]
    if float(meta.get("mu", 0.0)) != 0.0:
        o += ["--mu", "%g" % float(meta["mu"]), "--walsh-k0", str(int(meta["walsh_k0"]))]
    o += ["--flux", str(meta.get("flux_mode", "mean"))]
    return " ".join(o)


def tag_of(meta):
    """A filesystem-safe tag mirroring the grid's directory names."""
    bits = []
    if meta.get("encoding", "crop10") != "crop10":
        bits.append(str(meta["encoding"]))
    if meta.get("latent_map", "affine") == "mlp":
        bits.append("mlpH%d" % int(meta.get("hidden") or 8))
    bits.append("M%d" % int(meta["M"]))
    obj = meta.get("objective", "mmdflat")
    if obj == "spatial" and float(meta.get("spatial_lam_flat") or 0.0) != 0.0:
        bits.append("spatialflat%g" % float(meta.get("spatial_flat_scale", 1.0)))
    elif obj == "multi" and float(meta.get("lam_scale", 1.0)) != 1.0:
        bits.append("multix%g" % float(meta["lam_scale"]))
    else:
        bits.append(obj)
    if float(meta.get("mu", 0.0)) != 0.0:
        bits.append("mu%gk%d" % (float(meta["mu"]), int(meta["walsh_k0"])))
    bits.append("flux%s" % meta.get("flux_mode", "mean"))
    return "_".join(bits).replace("-", "m")


def main(argv=None):
    a = parse_args(argv)
    rows = [json.loads(l) for l in open(a.results) if l.strip()]
    agg = collections.defaultdict(list)
    for r in rows:
        if int(r.get("n_train", -1)) == a.n_train and str(r.get("arm", "")).startswith("IQP-v12"):
            agg[r["arm"]].append(r)
    if not agg:
        raise SystemExit("v12_select_arms: no IQP-v12 rows at N=%d" % a.n_train)
    means = {k: {c: float(np.mean([x[c] for x in v])) for c in RANK_COLS} for k, v in agg.items()}
    params = {k: int(v[0].get("param_count") or 0) for k, v in agg.items()}
    rank = collections.defaultdict(float)
    for c in RANK_COLS:
        for i, k in enumerate(sorted(means, key=lambda z: means[z][c])):
            rank[k] += i + 1
    order = sorted(means, key=lambda k: (rank[k], params[k]))
    for k in order[: max(int(a.top), 1)]:
        meta = json.load(open(os.path.join(agg[k][0]["run_dir"], "meta.json")))
        if not a.quiet:
            sys.stderr.write("# %-52s rank %5.1f  js %.6f  mmd2 %.5f  mmd2al %.5f  params %d\n"
                             % (k, rank[k] / len(RANK_COLS), means[k]["pdf_js"],
                                means[k]["conv_mmd2"], means[k]["conv_mmd2_aligned"], params[k]))
        print("%s|%s" % (tag_of(meta), cli_of(meta)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
