#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Score the per-class runs, each against ITS OWN class's 200 held-out images.

A WRAPPER around evaluate_r3b.main (which is a wrapper around evaluate_r3): for each
class it installs that class's split, runs the unchanged scorer over that class's run
directories, tags every row with `class_name`, and appends the rows to one merged
results file.  Metrics, seeds and the held-out protocol are the pooled ones; only the
200 held-out images the arm is scored against change.

    python3 evaluate_class.py --outputs-dir outputs_lens_class --results results_class.jsonl

Rows carry `class_name` so the per-class HOLDOUT split-half / RANDOM-PAIRING reference
rows -- which have the same `arm` string for all three classes -- stay distinguishable.

Python 3.9 compatible.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import tempfile

import data_lens_class
import evaluate_r3b


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outputs-dir", default="outputs_lens_class",
                   help="tree holding the per-class run dirs, named <class>_*")
    p.add_argument("--classes", nargs="+", default=list(data_lens_class.CLASSES))
    p.add_argument("--results", default="results_class.jsonl")
    p.add_argument("--n-holdout", type=int, default=200)
    p.add_argument("--n-train-max", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def run_dirs_for(outputs_dir, class_name):
    """<outputs_dir>/<class>_* -- the naming the sbatch writes."""
    pat = os.path.join(outputs_dir, "%s_*" % class_name)
    return sorted(x for x in glob.glob(pat) if os.path.isdir(x))


def arm_of(run_dir):
    path = os.path.join(run_dir, "meta.json")
    if not os.path.isfile(path):
        return None
    with open(path) as fh:
        return json.load(fh).get("arm")


def main(argv=None):
    args = parse_args(argv)
    all_rows = []
    for class_name in args.classes:
        dirs = run_dirs_for(args.outputs_dir, class_name)
        if not dirs:
            print("[evaluate_class] no run dirs for class %r under %r -- skipping"
                  % (class_name, args.outputs_dir))
            continue
        meta = data_lens_class.install(class_name)
        print("\n" + "=" * 88)
        print("[evaluate_class] class=%s  runs=%d  held-out=%d images of this class only "
              "(cache holds %d)"
              % (class_name, len(dirs), args.n_holdout, meta["n_holdout"]))
        print("=" * 88, flush=True)

        pair_arm = arm_of(dirs[0]) or "IQP-affine-M4-MMDflat-resample-%s" % class_name
        fd, tmp = tempfile.mkstemp(suffix=".jsonl", prefix="evalclass_%s_" % class_name)
        os.close(fd)
        try:
            evaluate_r3b.main(["--runs"] + dirs
                              + ["--n-holdout", str(args.n_holdout),
                                 "--n-train-max", str(args.n_train_max),
                                 "--random-pair-arm", pair_arm,
                                 "--seed", str(args.seed),
                                 "--results", tmp])
            with open(tmp) as fh:
                for ln in fh:
                    ln = ln.strip()
                    if not ln:
                        continue
                    r = json.loads(ln)
                    r["class_name"] = class_name
                    r["classes_pooled"] = False
                    all_rows.append(r)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    if not all_rows:
        raise SystemExit("evaluate_class.py: nothing scored")
    with open(args.results, "w") as fh:
        for r in all_rows:
            fh.write(json.dumps(r) + "\n")
    print("\n[evaluate_class] wrote %d rows to %s" % (len(all_rows), args.results))

    # the one table the appendix quotes
    cols = [("pdf_js", "PDF–JS"), ("conv_mmd2", "MMD²"), ("conv_mmd2_aligned", "MMD²_al"),
            ("projection_corr_mean", "proj corr"), ("nn_holdout", "NN-holdout")]
    print("\n| class | arm | N | " + " | ".join(c[1] for c in cols) + " |")
    print("|---|---|---|" + "|".join(["---"] * len(cols)) + "|")
    for r in all_rows:
        if not str(r.get("arm", "")).startswith("IQP-affine"):
            continue
        vals = []
        for k, _lab in cols:
            v = r.get(k)
            vals.append("—" if v is None else "%.4g" % float(v))
        print("| %s | %s | %s | %s |"
              % (r["class_name"], r["arm"], r["n_train"], " | ".join(vals)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
