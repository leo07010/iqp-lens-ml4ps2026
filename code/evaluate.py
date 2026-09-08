#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Table 2 evaluation for the arXiv 2603.06440 replication.

Scans arm output directories (outputs/*/ by default, or --runs), reads each arm's
generated_*.npy + meta.json, compares against outputs/ground_truth/gt_t{tau}.npy (written
by make_ground_truth.py for --data synthetic, by data_d2.py's split for --data d2), computes
metrics.pdf_js and metrics.conv_mmd2, writes results.jsonl (one row per arm-seed run), and
prints a markdown Table 2 (mean +/- std across seeds when more than one seed shares an
(arm, train_snapshots) group).

Time-conditioned arms (IQP, RBM, CTRL-marginal, CTRL-interp) write generated_t{tau}.npy and
are compared tau-for-tau against the matching ground-truth file. Time-unconditional arms
(DCGAN: no time-conditioning mechanism, per the paper) write generated_{i:03d}.npy and are
compared distributionally against the full held-out ground-truth set.

The Conv2D encoder used by metrics.conv_mmd2 is re-initialized from the SAME fixed seed
(CONV_SEED = 0) for every arm, so every arm is scored in the same feature space (metrics.py
re-derives identical weights from that seed every call -- there is no cross-call state to
share, so a fixed seed is equivalent to "one shared instance").

Python 3.9 compatible (no `X | None`).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np

from metrics import conv_mmd2, pdf_js

GT_DEFAULT = "outputs/ground_truth"
CONV_SEED = 0  # fixed Conv2D encoder seed -- same effective encoder weights for every arm

_TAU_RE = re.compile(r"generated_t(-?\d+)\.npy$")
_GT_RE = re.compile(r"gt_t(-?\d+)\.npy$")
_UNCOND_RE = re.compile(r"generated_(\d+)\.npy$")


def _load_meta(run_dir):
    path = os.path.join(run_dir, "meta.json")
    if not os.path.isfile(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def _train_snapshots_count(meta):
    """meta['train_snapshots'] is a list of times in some arms (RBM, DCGAN), a bare count
    in others (train_iqp.py, controls.py) -- normalize to a count."""
    ts = meta.get("train_snapshots")
    if isinstance(ts, (list, tuple)):
        return len(ts)
    if isinstance(ts, int):
        return ts
    anchors = meta.get("anchors")
    if isinstance(anchors, (list, tuple)):
        return len(anchors)
    return None


def _tau_conditioned_files(run_dir):
    """{tau: path} for this run's generated_t{tau}.npy files."""
    out = {}
    for path in glob.glob(os.path.join(run_dir, "generated_t*.npy")):
        m = _TAU_RE.match(os.path.basename(path))
        if m:
            out[int(m.group(1))] = path
    return out


def _unconditional_files(run_dir):
    """this run's generated_{i:03d}.npy files (DCGAN: unconditional samples), sorted by i."""
    found = []
    for path in glob.glob(os.path.join(run_dir, "generated_*.npy")):
        m = _UNCOND_RE.match(os.path.basename(path))
        if m:
            found.append((int(m.group(1)), path))
    return [p for _, p in sorted(found)]


def load_ground_truth(gt_dir):
    """{tau: (64,64) float32} for every outputs/ground_truth/gt_t{tau}.npy."""
    out = {}
    for path in glob.glob(os.path.join(gt_dir, "gt_t*.npy")):
        m = _GT_RE.match(os.path.basename(path))
        if m:
            out[int(m.group(1))] = np.load(path)
    return out


def evaluate_run(run_dir, gt_by_tau):
    """One arm-seed run -> a results.jsonl row, or None if this directory has no usable
    meta.json / generated files (e.g. outputs/ground_truth itself, or a still-running arm)."""
    meta = _load_meta(run_dir)
    if meta is None:
        return None

    tau_files = _tau_conditioned_files(run_dir)
    if tau_files:
        common = sorted(set(tau_files) & set(gt_by_tau))
        if not common:
            print("[evaluate] %s: generated_t*.npy present but none match ground_truth tau" % run_dir)
            return None
        gt = np.stack([gt_by_tau[t] for t in common], axis=0)
        gen = np.stack([np.load(tau_files[t]) for t in common], axis=0)
    else:
        uncond = _unconditional_files(run_dir)
        if not uncond or not gt_by_tau:
            return None
        gt = np.stack([gt_by_tau[t] for t in sorted(gt_by_tau)], axis=0)
        gen = np.stack([np.load(p) for p in uncond], axis=0)

    js = float(pdf_js(gt, gen))
    mmd2 = float(conv_mmd2(gt, gen, seed=CONV_SEED))

    return {
        "run_dir": run_dir,
        "arm": meta.get("arm", os.path.basename(os.path.normpath(run_dir))),
        "param_count": meta.get("param_count"),
        "seed": meta.get("seed"),
        "train_snapshots": _train_snapshots_count(meta),
        "n_compared": int(gt.shape[0]),
        "pdf_js": js,
        "conv_mmd2": mmd2,
    }


def render_table2(rows):
    """Group rows by (arm, train_snapshots); mean +/- std across seeds within a group."""
    groups = {}
    for row in rows:
        if row["pdf_js"] is None:
            continue
        key = (row["arm"], row["train_snapshots"])
        groups.setdefault(key, []).append(row)

    def sort_key(k):
        arm, n_train = k
        return (arm, n_train if n_train is not None else -1)

    lines = [
        "| Model (Params)(# Training Snapshots) | PDF–JS ↓ | MMD² (Conv2D) ↓ |",
        "|---|---|---|",
    ]
    for key in sorted(groups.keys(), key=sort_key):
        arm, n_train = key
        group_rows = groups[key]
        params = group_rows[0].get("param_count")
        js_vals = np.array([r["pdf_js"] for r in group_rows], dtype=np.float64)
        mmd_vals = np.array([r["conv_mmd2"] for r in group_rows], dtype=np.float64)
        label = "{} ({}) ({})".format(arm, params, n_train)
        if len(group_rows) > 1:
            js_str = "{:.4g} ± {:.4g}".format(js_vals.mean(), js_vals.std())
            mmd_str = "{:.4g} ± {:.4g}".format(mmd_vals.mean(), mmd_vals.std())
        else:
            js_str = "{:.4g}".format(js_vals[0])
            mmd_str = "{:.4g}".format(mmd_vals[0])
        lines.append("| {} | {} | {} |".format(label, js_str, mmd_str))
    return "\n".join(lines)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Table 2 evaluation for the IQP turbulence replication.")
    p.add_argument("--runs", nargs="*", default=None,
                   help="explicit list of arm output directories; default: every outputs/*/ "
                        "except outputs/ground_truth")
    p.add_argument("--outputs-dir", default="outputs")
    p.add_argument("--ground-truth", default=GT_DEFAULT)
    p.add_argument("--results", default="results.jsonl")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    gt_by_tau = load_ground_truth(args.ground_truth)
    if not gt_by_tau:
        raise SystemExit(
            "evaluate.py: no ground truth found in %r -- run make_ground_truth.py "
            "(--data synthetic) or data_d2.py's split (--data d2) first" % args.ground_truth
        )

    if args.runs:
        run_dirs = list(args.runs)
    else:
        run_dirs = sorted(
            d for d in glob.glob(os.path.join(args.outputs_dir, "*"))
            if os.path.isdir(d) and os.path.basename(os.path.normpath(d)) != "ground_truth"
        )

    rows = []
    for run_dir in run_dirs:
        row = evaluate_run(run_dir, gt_by_tau)
        if row is not None:
            rows.append(row)
        else:
            print("[evaluate] skipped %s (no meta.json / no matching generated files)" % run_dir)

    with open(args.results, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    print("[evaluate] wrote %d rows to %s" % (len(rows), args.results))

    table = render_table2(rows)
    print()
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
