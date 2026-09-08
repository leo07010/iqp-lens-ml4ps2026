#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluation table for the DeepLense Model II pilot (docs/IQP_LENS_PILOT_PREREGISTRATION.md).

Same two distribution metrics as evaluate.py (metrics.pdf_js, metrics.conv_mmd2 with the
fixed seed-0 Conv2D encoder, so every arm is scored in one feature space), PLUS the two
nearest-neighbour L2 diagnostics the pre-registration adds, because on image data the two
distribution metrics can be maximized by copying:

  NN-train    mean over generated images of min_j ||gen_i - train_j||_2   (memorisation:
              small = the arm is reproducing its training set)
  NN-holdout  mean over generated images of min_j ||gen_i - holdout_j||_2 (coverage: small
              = generated images land near real unseen images)

L2 is the plain Euclidean norm over the 4,096 pixels of the (64, 64) field (NOT per-pixel
RMS), so the numbers are directly comparable across rows but not to a per-pixel scale.

A `HOLDOUT (ref)` row gives both NN columns for the held-out set itself -- NN-train is
held-out-to-training, NN-holdout is held-out-to-other-held-out (self excluded). That row
is the yardstick: an arm whose NN-holdout is far above the reference is generating images
unlike anything real; an arm whose NN-train is far below it is copying.

Ground truth is NOT read from a gt_t*.npy directory (there is no time axis to match on):
the 200 held-out images come straight from data_lens, the ONE held-out source, and every
arm's generated_{i:03d}.npy set is compared to them distributionally.

The sqrt arms write their fields back in INTENSITY space (lens_synth.write_pair /
data_lens.invert_intensity), so all rows -- and the training/held-out images used here --
live in one space and one table.

Python 3.9 compatible (no `X | None`).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np

import data_lens
from metrics import conv_mmd2, pdf_js

CONV_SEED = 0
_UNCOND_RE = re.compile(r"generated_(\d+)\.npy$")


def _load_meta(run_dir):
    path = os.path.join(run_dir, "meta.json")
    if not os.path.isfile(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def _n_train(meta):
    for key in ("n_train", "train_snapshots"):
        v = meta.get(key)
        if isinstance(v, int):
            return v
        if isinstance(v, (list, tuple)):
            return len(v)
    anchors = meta.get("anchors")
    if isinstance(anchors, (list, tuple)):
        return len(anchors)
    return None


def _generated(run_dir):
    """(n, 64, 64) float32 of this run's generated_{i:03d}.npy, ordered by i; None if none."""
    found = []
    for path in glob.glob(os.path.join(run_dir, "generated_*.npy")):
        m = _UNCOND_RE.match(os.path.basename(path))
        if m:
            found.append((int(m.group(1)), path))
    if not found:
        return None
    found.sort()
    return np.stack([np.load(p) for _, p in found], axis=0).astype(np.float32)


def nn_l2(a, b, exclude_self=False):
    """mean over rows of a of min over rows of b of ||a_i - b_j||_2.

    a: (n, 64, 64), b: (m, 64, 64). exclude_self masks the diagonal (a is b)."""
    A = np.asarray(a, dtype=np.float64).reshape(a.shape[0], -1)
    B = np.asarray(b, dtype=np.float64).reshape(b.shape[0], -1)
    # ||x-y||^2 = |x|^2 + |y|^2 - 2 x.y  (clipped: rounding can make it slightly negative)
    d2 = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * (A @ B.T)
    d2 = np.maximum(d2, 0.0)
    if exclude_self:
        n = min(d2.shape)
        d2[np.arange(n), np.arange(n)] = np.inf
    return float(np.sqrt(d2.min(axis=1)).mean())


def evaluate_run(run_dir, holdout, train_full, lens_class):
    meta = _load_meta(run_dir)
    if meta is None:
        return None
    if meta.get("lens_class") not in (None, lens_class):
        return None  # another class's run sharing the same outputs dir
    gen = _generated(run_dir)
    if gen is None:
        return None
    n_tr = _n_train(meta)
    if n_tr is None or n_tr < 1:
        return None
    train = train_full[: min(int(n_tr), train_full.shape[0])]

    return {
        "run_dir": run_dir,
        "arm": meta.get("arm", os.path.basename(os.path.normpath(run_dir))),
        "lens_class": lens_class,
        "param_count": meta.get("param_count"),
        "seed": meta.get("seed"),
        "n_train": int(n_tr),
        "aggregation": meta.get("aggregation"),
        "n_gen": int(gen.shape[0]),
        "n_holdout": int(holdout.shape[0]),
        "pdf_js": float(pdf_js(holdout, gen)),
        "conv_mmd2": float(conv_mmd2(holdout, gen, seed=CONV_SEED)),
        "nn_train": nn_l2(gen, train),
        "nn_holdout": nn_l2(gen, holdout),
        # Range diagnostics -- NOT metrics, and nothing here changes a metric.
        # Both distribution metrics bin/normalise on the JOINT range of (held-out, this
        # arm's generated set): metrics.pdf_js takes lo/hi = min/max over both, and
        # metrics.conv_mmd2 normalises its encoder input the same way. So an arm whose
        # pixels leave the data's range stretches its own bins, and its PDF-JS / MMD2 are
        # not on the same scale as an in-range arm's. That is the metric's pre-existing
        # design (it applied to round 1 too), so it is left alone -- changing it now would
        # be an un-preregistered lever -- but these three numbers make the caveat
        # checkable per row instead of invisible. Compare gen_min/gen_max against
        # holdout_min/holdout_max, and gen_mean_total against holdout_mean_total (every
        # pmf arm's images have total intensity exactly S by construction, so a residual
        # NN-holdout gap can come from that mismatch rather than from the spatial fit).
        "gen_min": float(gen.min()),
        "gen_max": float(gen.max()),
        "gen_mean_total": float(gen.reshape(gen.shape[0], -1).sum(axis=1).mean()),
        "holdout_min": float(holdout.min()),
        "holdout_max": float(holdout.max()),
        "holdout_mean_total": float(holdout.reshape(holdout.shape[0], -1).sum(axis=1).mean()),
    }


def reference_rows(holdout, train_full, n_trains, lens_class):
    """One `HOLDOUT (ref)` row per distinct N: the held-out set's own NN distances."""
    nn_hold_self = nn_l2(holdout, holdout, exclude_self=True)
    rows = []
    for n in sorted(set(int(n) for n in n_trains)):
        rows.append({
            "run_dir": None,
            "arm": "HOLDOUT (ref)",
            "lens_class": lens_class,
            "param_count": None,
            "seed": None,
            "n_train": n,
            "aggregation": None,
            "n_gen": int(holdout.shape[0]),
            "n_holdout": int(holdout.shape[0]),
            "pdf_js": None,
            "conv_mmd2": None,
            "nn_train": nn_l2(holdout, train_full[: n]),
            "nn_holdout": nn_hold_self,
        })
    return rows


def render_table(rows, lens_class):
    def fmt(v):
        return "—" if v is None else "{:.4g}".format(v)

    def sort_key(r):
        return (r["n_train"] if r["n_train"] is not None else -1,
                0 if r["arm"] == "HOLDOUT (ref)" else 1, str(r["arm"]))

    lines = [
        "### {} — 200 generated vs 200 held-out".format(lens_class),
        "",
        "| Arm | N | PDF–JS ↓ | MMD² (Conv2D) ↓ | NN-train | NN-holdout |",
        "|---|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=sort_key):
        params = r.get("param_count")
        label = r["arm"] if params in (None, 0) else "{} ({})".format(r["arm"], params)
        lines.append("| {} | {} | {} | {} | {} | {} |".format(
            label, r["n_train"], fmt(r["pdf_js"]), fmt(r["conv_mmd2"]),
            fmt(r["nn_train"]), fmt(r["nn_holdout"]),
        ))
    return "\n".join(lines)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Evaluation table for the lens pilot.")
    p.add_argument("--lens-class", required=True, help="axion / cdm / no_sub")
    p.add_argument("--outputs-dir", nargs="+", default=["outputs_lens"],
                   help="one or more directories, each scanned for */meta.json + */generated_*.npy. "
                        "Several are merged into ONE table -- that is how round 2 is scored against "
                        "the round-1 arms without rerunning them "
                        "(--outputs-dir outputs_lens_pilot/<cls> outputs_lens_round2/<cls>).")
    p.add_argument("--runs", nargs="*", default=None, help="explicit run directories instead")
    p.add_argument("--n-holdout", type=int, default=data_lens.N_HOLDOUT)
    p.add_argument("--n-train-max", type=int, default=data_lens.N_TRAIN_MAX,
                   help="training images loaded once; each row uses its own n_train prefix")
    p.add_argument("--results", default=None, help="default: results_lens_<class>.jsonl")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cls = args.lens_class
    results_path = args.results if args.results else "results_lens_%s.jsonl" % cls

    # Held-out and training images in INTENSITY space (sqrt arms write back into it).
    holdout = data_lens.holdout_images(cls, n_holdout=args.n_holdout, sqrt=False)
    train_full = data_lens.raw_training_images(cls, args.n_train_max, sqrt=False)
    print("[evaluate_lens] class=%s holdout=%s train=%s" % (cls, holdout.shape, train_full.shape))

    if args.runs:
        run_dirs = list(args.runs)
    else:
        outputs_dirs = ([args.outputs_dir] if isinstance(args.outputs_dir, str)
                        else list(args.outputs_dir))
        for d in outputs_dirs:
            if not os.path.isdir(d):
                raise SystemExit("evaluate_lens.py: --outputs-dir %r is not a directory" % d)
        run_dirs = []
        for d in outputs_dirs:
            run_dirs.extend(sorted(x for x in glob.glob(os.path.join(d, "*")) if os.path.isdir(x)))
        print("[evaluate_lens] scanning %d directories: %s" % (len(outputs_dirs), outputs_dirs))

    rows = []
    for run_dir in run_dirs:
        row = evaluate_run(run_dir, holdout, train_full, cls)
        if row is None:
            print("[evaluate_lens] skipped %s (no meta.json / no generated_*.npy / other class)" % run_dir)
        else:
            rows.append(row)

    if not rows:
        raise SystemExit("evaluate_lens.py: no usable runs found under %r" % (args.outputs_dir,))

    seen = {}
    for r in rows:
        key = (r["arm"], r["n_train"])
        seen.setdefault(key, []).append(r["run_dir"])
    for key, dirs in sorted(seen.items()):
        if len(dirs) > 1:
            print("[evaluate_lens] WARNING: arm %r at N=%s appears %d times: %s"
                  % (key[0], key[1], len(dirs), dirs))

    # Out-of-range arms: their PDF-JS / MMD2 are computed on a stretched range (see the
    # comment in evaluate_run). Say so out loud rather than leaving it to whoever reads
    # the table months later.
    for r in sorted(rows, key=lambda z: (z["n_train"], str(z["arm"]))):
        if r.get("gen_max") is None:
            continue
        if r["gen_max"] > 1.05 * r["holdout_max"] or r["gen_min"] < r["holdout_min"] - 0.05:
            print("[evaluate_lens] NOTE: arm %r N=%s generates outside the held-out range "
                  "([%.4g, %.4g] vs [%.4g, %.4g]) -- its PDF-JS / MMD2 are binned on a "
                  "stretched range and are not on the same scale as an in-range arm's"
                  % (r["arm"], r["n_train"], r["gen_min"], r["gen_max"],
                     r["holdout_min"], r["holdout_max"]))
        # Total-intensity mismatch is the more discriminating signal and is invisible in
        # min/max: round-1 DCGAN-1.2m sits inside [0, 1] per pixel yet totals 899 against
        # the held-out 149 (measured 2026-09-06), which is most of why its PDF-JS is 0.71.
        if r["holdout_mean_total"] > 0:
            ratio = r["gen_mean_total"] / r["holdout_mean_total"]
            if ratio > 1.25 or ratio < 0.8:
                print("[evaluate_lens] NOTE: arm %r N=%s has mean total intensity %.4g vs "
                      "held-out %.4g (x%.2f) -- part of its distance to the held-out set is "
                      "this scale mismatch, not spatial structure"
                      % (r["arm"], r["n_train"], r["gen_mean_total"],
                         r["holdout_mean_total"], ratio))

    rows = rows + reference_rows(holdout, train_full, [r["n_train"] for r in rows], cls)

    with open(results_path, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    print("[evaluate_lens] wrote %d rows to %s" % (len(rows), results_path))
    print()
    print(render_table(rows, cls))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
