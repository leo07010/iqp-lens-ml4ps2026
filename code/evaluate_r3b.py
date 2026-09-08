#!/usr/bin/env python3
"""Round-3b evaluation = round 3's table plus the three columns round 3b was declared for.

docs/IQP_LENS_ROUND3_PREREGISTRATION.md, "Round 3b" and its amendment. NEW file: every metric
is imported from evaluate_r3.py, which is NOT modified (round 3's 28-job sbatch is queued
against it). The scoring context, the held-out set, the Conv2D encoder seed and the
`HOLDOUT split-half` / `RANDOM-PAIRING (ref)` reference rows are therefore identical to round
3's, so round 3 and round 3b rows can sit in ONE table and be compared directly -- pass both
output trees to --outputs-dir.

Added columns (per the amendment)
---------------------------------
projection corr (mean / min)
    read out of each run's meta.json, where train_iqp_affine_r3.py and controls_pca_r3.py
    write it. Blank for every round-3 arm: a fixed-latent-block arm has no "best reconstruction
    reachable inside the model's latent space" to report, and inventing one here would compare
    two different quantities in one column.
NN-within
    the generated set's own nearest-neighbour L2 (self excluded) -- the DIVERSITY column.
    Same function and same units as NN-train / NN-holdout (evaluate_lens.nn_l2), so the three
    are directly comparable: an arm that emits 200 near-copies of one image has NN-within ~ 0
    while its NN-holdout can still look respectable.
e1 sd / e2 sd
    the standard deviation of the generated set's ellipticity components -- the DISPERSION
    columns (the mechanism test found the M = 2 working point under-dispersed: e1 sd 0.048
    against the data's 0.099).

Each of the three has its held-out counterpart printed in the SAME row (`ho NN-within`,
`ho e1 sd`, `ho e2 sd`), computed once on the 200 held-out frames, so a reader never has to
scroll to the reference row to know whether a number is high or low. Those three columns are
identical down every row by construction; that is the point of them.

Outputs: the primary round-3 table with the six extra columns, the unchanged intensity-aligned
secondary table, and one jsonl row per run (round-3 keys plus the new ones).

Python 3.9 compatible (no `X | None`).
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

import data_lens_r3
import evaluate_r3
from evaluate_lens import nn_l2

QC = evaluate_r3.QCOL


def _meta(run_dir):
    path = os.path.join(run_dir, "meta.json")
    if not os.path.isfile(path):
        return {}
    with open(path) as fh:
        return json.load(fh)


def _blank_extra():
    return {"projection_corr_mean": None, "projection_corr_min": None,
            "projection_corr_median": None, "ref_projection_mean_crop_corr": None,
            "ref_projection_nearest_train_corr": None, "M": None,
            "nn_within": None, "e1_sd": None, "e2_sd": None}


def extra_columns(run_dir, ctx):
    """The round-3b additions for one run: meta-carried projection numbers + the diversity /
    dispersion numbers recomputed here from the generated frames."""
    out = _blank_extra()
    m = _meta(run_dir)
    for k in ("projection_corr_mean", "projection_corr_min", "projection_corr_median",
              "ref_projection_mean_crop_corr", "ref_projection_nearest_train_corr", "M"):
        v = m.get(k)
        out[k] = None if v is None else (float(v) if k != "M" else int(v))
    gen = evaluate_r3._stack(run_dir)
    if gen is not None and gen.shape[0] > 1:
        gq = evaluate_r3.quantity_table(gen)
        out["nn_within"] = nn_l2(gen, gen, exclude_self=True)
        out["e1_sd"] = float(gq[:, QC["e1"]].std())
        out["e2_sd"] = float(gq[:, QC["e2"]].std())
    out.update(ctx["holdout_dispersion"])
    return out


def render_table_r3b(rows):
    def f(v, spec="{:.4g}"):
        return "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else spec.format(v)

    def key(r):
        return (int(r["n_train"]), 0 if r["arm"].startswith("HOLDOUT") else
                (2 if r["arm"].startswith("RANDOM") else 1), str(r["arm"]))

    head = ("| Arm | N | physics ↓ | proj corr ↑ | proj min ↑ | NN-within ↑ | ho NN-within | "
            "e1 sd | e2 sd | ho e1 sd | ho e2 sd | W1(offset) | W1(flux) | W1(radius) | "
            "W1(e1) | W1(e2) | profile RMSE | PDF–JS ↓ | MMD² ↓ | NN-train | NN-holdout | "
            "\\|Δe1\\| | \\|Δe2\\| | \\|Δflux\\| | \\|Δradius\\| | paired L2 |")
    lines = [head, "|" + "---|" * 26]
    for r in sorted(rows, key=key):
        pc = r.get("param_count")
        label = r["arm"] if pc in (None, 0) else "%s (%s)" % (r["arm"], pc)
        lines.append("| " + " | ".join([
            label, str(r["n_train"]), f(r.get("physics")),
            f(r.get("projection_corr_mean")), f(r.get("projection_corr_min")),
            f(r.get("nn_within")), f(r.get("holdout_nn_within")),
            f(r.get("e1_sd")), f(r.get("e2_sd")),
            f(r.get("holdout_e1_sd")), f(r.get("holdout_e2_sd")),
            f(r.get("w1_offset")), f(r.get("w1_flux")), f(r.get("w1_radius")),
            f(r.get("w1_e1")), f(r.get("w1_e2")), f(r.get("profile_rmse")),
            f(r.get("pdf_js")), f(r.get("conv_mmd2")), f(r.get("nn_train")),
            f(r.get("nn_holdout")), f(r.get("pair_de1")), f(r.get("pair_de2")),
            f(r.get("pair_dflux")), f(r.get("pair_dradius")), f(r.get("pair_l2"))]) + " |")
    return "\n".join([
        "### Round 3 + 3b — generated vs 200 held-out lens images", "",
        "`proj corr` = held-out projection correlation (mean / min over the 200 held-out crops): "
        "the best reconstruction reachable inside that arm's own latent space, z optimised with "
        "everything else frozen. Blank for round-3 arms, which have no such quantity. "
        "`NN-within` is the generated set's own nearest-neighbour L2 (diversity; ~0 means the "
        "arm emits near-duplicates), `e1 sd` / `e2 sd` its ellipticity dispersion; the `ho *` "
        "columns are the held-out set's own values of the same three and are the same in every "
        "row on purpose.", ""] + lines)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Round-3b evaluation table (round 3 + 3b arms).")
    p.add_argument("--outputs-dir", nargs="+", default=["outputs_lens_r3b"],
                   help="one or more output trees; pass outputs_lens_r3 too to merge round 3")
    p.add_argument("--runs", nargs="*", default=None)
    p.add_argument("--n-holdout", type=int, default=data_lens_r3.N_HOLDOUT)
    p.add_argument("--n-train-max", type=int, default=data_lens_r3.N_TRAIN_MAX)
    p.add_argument("--results", default="results_r3b.jsonl")
    p.add_argument("--random-pair-arm", default="IQP-lowvar-joint-MMD")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-missing", action="store_true",
                   help="tolerate an --outputs-dir that does not exist (round 3 may not have "
                        "run yet); evaluate_r3.py errors out instead")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    d = data_lens_r3.load_r3(args.n_train_max, n_holdout=args.n_holdout)
    ctx = {"holdout": d.holdout_images, "holdout_crops": d.holdout_crops,
           "train_images": d.train_images}
    ctx["hold_q"] = evaluate_r3.quantity_table(ctx["holdout"])
    ctx["hold_profile"] = evaluate_r3.mean_profile(ctx["holdout"])
    ctx["holdout_mean_total"] = float(ctx["holdout"].reshape(ctx["holdout"].shape[0], -1)
                                      .sum(axis=1).mean())
    ctx["holdout_aligned"] = evaluate_r3.align_to_target(ctx["holdout"], ctx["holdout_mean_total"])
    ctx["train_aligned"] = evaluate_r3.align_to_target(ctx["train_images"], ctx["holdout_mean_total"])
    ctx["holdout_dispersion"] = {
        "holdout_nn_within": nn_l2(ctx["holdout"], ctx["holdout"], exclude_self=True),
        "holdout_e1_sd": float(ctx["hold_q"][:, QC["e1"]].std()),
        "holdout_e2_sd": float(ctx["hold_q"][:, QC["e2"]].std()),
    }
    print("[evaluate_r3b] holdout=%s train=%s | held-out NN-within=%.4f e1 sd=%.4f e2 sd=%.4f"
          % (ctx["holdout"].shape, ctx["train_images"].shape,
             ctx["holdout_dispersion"]["holdout_nn_within"],
             ctx["holdout_dispersion"]["holdout_e1_sd"],
             ctx["holdout_dispersion"]["holdout_e2_sd"]))

    if args.runs:
        run_dirs = list(args.runs)
    else:
        run_dirs = []
        for dd in args.outputs_dir:
            if not os.path.isdir(dd):
                if args.skip_missing:
                    print("[evaluate_r3b] skipping absent --outputs-dir %r" % dd)
                    continue
                raise SystemExit("evaluate_r3b.py: --outputs-dir %r is not a directory" % dd)
            run_dirs.extend(sorted(x for x in glob.glob(os.path.join(dd, "*")) if os.path.isdir(x)))

    rows = []
    for rd in run_dirs:
        row = evaluate_r3.score_run(rd, ctx)
        if row is None:
            print("[evaluate_r3b] skipped %s (no meta.json / no generated_*.npy)" % rd)
            continue
        row.update(extra_columns(rd, ctx))
        rows.append(row)
    if not rows:
        raise SystemExit("evaluate_r3b.py: no usable runs under %r" % (args.outputs_dir,))

    seen = {}
    for r in rows:
        seen.setdefault((r["arm"], r["n_train"]), []).append(r["run_dir"])
    for k, v in sorted(seen.items()):
        if len(v) > 1:
            print("[evaluate_r3b] WARNING: arm %r at N=%s appears %d times: %s"
                  % (k[0], k[1], len(v), v))

    hmt = ctx["holdout_mean_total"]
    for r in sorted(rows, key=lambda z: (z["n_train"], str(z["arm"]))):
        if hmt > 0 and r.get("gen_mean_total"):
            ratio = r["gen_mean_total"] / hmt
            if ratio > 1.25 or ratio < 0.8:
                print("[evaluate_r3b] NOTE: arm %r N=%s mean total intensity %.4g vs held-out "
                      "%.4g (x%.2f) -- part of its NN / PDF-JS / MMD2 distance is this scale "
                      "mismatch, not spatial structure"
                      % (r["arm"], r["n_train"], r["gen_mean_total"], hmt, ratio))
        if r.get("nn_within") is not None and r.get("nn_holdout"):
            if r["nn_within"] < 0.25 * ctx["holdout_dispersion"]["holdout_nn_within"]:
                print("[evaluate_r3b] NOTE: arm %r N=%s has NN-within %.4g against the held-out "
                      "set's %.4g -- it is emitting near-duplicates, so read its NN-holdout / "
                      "PDF-JS as the score of ONE image, not of a distribution"
                      % (r["arm"], r["n_train"], r["nn_within"],
                         ctx["holdout_dispersion"]["holdout_nn_within"]))

    ref = evaluate_r3.holdout_reference(ctx, [r["n_train"] for r in rows])
    for r in ref:
        r.update(_blank_extra())
        r.update(ctx["holdout_dispersion"])
        # the split-half row's own diversity/dispersion ARE the held-out values
        r["nn_within"] = ctx["holdout_dispersion"]["holdout_nn_within"]
        r["e1_sd"] = ctx["holdout_dispersion"]["holdout_e1_sd"]
        r["e2_sd"] = ctx["holdout_dispersion"]["holdout_e2_sd"]
    pair = evaluate_r3.random_pairing_rows(rows, ctx, args.random_pair_arm, args.seed)
    for r in pair:
        r.update(_blank_extra())
        r.update(ctx["holdout_dispersion"])
    rows = rows + ref + pair

    with open(args.results, "w") as fh:
        for r in rows:
            r.pop("_crops_dir", None)
            fh.write(json.dumps(r) + "\n")
    print("[evaluate_r3b] wrote %d rows to %s" % (len(rows), args.results))
    print()
    print(render_table_r3b(rows))
    print()
    print(evaluate_r3.render_table_aligned(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
