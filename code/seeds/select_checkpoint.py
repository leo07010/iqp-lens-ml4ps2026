#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validation-based early stopping for the classical baselines.

Why this exists.  The convergence sweep showed that neither baseline has a defensible fixed
step count: the RBM is still improving well past 20,000 steps, and the DCGAN is not monotone
at all -- at N = 100 its scores get worse after roughly 30,000 steps, and on GPU it does not
even reproduce run to run.  Quoting one hard-coded number for both would either undertrain
the RBM or quote the DCGAN at an arbitrary point on a curve that goes up and down.  So the
baselines get what a practitioner would give them: a step count chosen on a validation set.

The rule, fixed before looking at any held-out score:

  * The candidate checkpoints are the sweep snapshots
    (baseline_rbm_sweep.py: 20k/40k/80k/160k;  baseline_dcgan_sweep.py: every 5k to 60k).
  * Each candidate is scored on the SELECTION block -- 200 real images that are disjoint from
    the 200 held-out images and come from the same validation permutation (seeds/val_extra.py).
  * Two scores are used, PDF-JS and raw Conv2D-MMD^2.  They are on different scales, so they
    are combined by RANK: each candidate gets its rank in each score (1 = best) and the
    candidate with the smallest mean rank wins.  Ties break toward FEWER steps.
  * Only then is the winner scored on the held-out set, by the unchanged evaluate_r3
    functions, and only those held-out numbers are reported.

The held-out set is never used to choose anything.  The IQP arms are NOT early-stopped: they
keep their fixed 4,000 steps, so the baselines are if anything favoured by this procedure.

Output: one row per (arm, N, seed) with the full evaluate_r3 column set of the SELECTED
snapshot, plus `selected_steps`, `val_pdf_js`, `val_conv_mmd2`, the per-candidate validation
curve, and -- for the appendix comparison -- the same held-out columns for the fixed-20k
snapshot when the sweep contains one.

    python3 seeds/select_checkpoint.py --prefix outputs_convergence/rbm_n100 \\
        --results results_seeds_baselines_es.jsonl

Python 3.9 compatible.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time

REPO = "<REPO_ROOT>/code"
sys.path.insert(0, REPO)

import numpy as np                                      # noqa: E402
import torch                                            # noqa: E402

import data_lens_class                                  # noqa: E402
import data_lens_r3                                     # noqa: E402
import evaluate_r3                                      # noqa: E402
import evaluate_r3b                                     # noqa: E402
from metrics import conv_mmd2, pdf_js                   # noqa: E402

sys.path.insert(0, os.path.join(REPO, "seeds"))
import val_extra                                        # noqa: E402

FIXED_REFERENCE_STEPS = 20000       # the count round 3 used, kept for the appendix comparison
STEP_RE = re.compile(r"_s(\d+)k$")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prefix", nargs="+", required=True,
                   help="sweep out-prefix; its snapshots are <prefix>_s<k>k")
    p.add_argument("--results", default=os.path.join(REPO, "results_seeds_baselines_es.jsonl"))
    p.add_argument("--n-holdout", type=int, default=200)
    p.add_argument("--append", action="store_true")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--fixed-reference-steps", type=int, default=FIXED_REFERENCE_STEPS,
                   help="the step count the published protocol used, kept alongside the "
                        "selected one for the appendix comparison (20,000 for the classical "
                        "baselines; 4,000 for the IQP arms)")
    p.add_argument("--label", default="baseline",
                   help="prefix for the `kind` field, so an IQP sweep and a baseline sweep "
                        "can share this selector without landing in each other's tables")
    return p.parse_args(argv)


def snapshots_of(prefix):
    """[(steps, run_dir), ...] ascending, from <prefix>_s<k>k."""
    out = []
    for d in sorted(glob.glob(prefix + "_s*k")):
        if not os.path.isdir(d) or not os.path.isfile(os.path.join(d, "meta.json")):
            continue
        m = STEP_RE.search(os.path.basename(d))
        if m:
            out.append((int(m.group(1)) * 1000, d))
    return sorted(out)


def ranks(vals):
    """1-based competition ranks, smallest value = rank 1."""
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    r = [0] * len(vals)
    for pos, i in enumerate(order):
        r[i] = pos + 1
    return r


def select(prefix, sel_images, n_holdout):
    snaps = snapshots_of(prefix)
    if not snaps:
        print("[select] no snapshots under %s_s*k -- skipping" % prefix, flush=True)
        return None

    curve = []
    for steps, rd in snaps:
        gen = evaluate_r3._stack(rd)
        if gen is None:
            print("[select] %s has no generated_*.npy -- excluded" % rd, flush=True)
            continue
        curve.append({
            "steps": steps, "run_dir": rd,
            "val_pdf_js": float(pdf_js(sel_images, gen)),
            "val_conv_mmd2": float(conv_mmd2(sel_images, gen, seed=evaluate_r3.CONV_SEED)),
        })
    if not curve:
        return None

    r_js = ranks([c["val_pdf_js"] for c in curve])
    r_mmd = ranks([c["val_conv_mmd2"] for c in curve])
    for c, a, b in zip(curve, r_js, r_mmd):
        c["rank_pdf_js"] = a
        c["rank_conv_mmd2"] = b
        c["mean_rank"] = 0.5 * (a + b)
    # ties break toward fewer steps: the secondary key is the step count itself
    best = min(curve, key=lambda c: (c["mean_rank"], c["steps"]))
    print("[select] %s -> %d steps (mean rank %.1f of %d candidates; "
          "val PDF-JS %.5f, val MMD2 %.4f)"
          % (os.path.basename(prefix), best["steps"], best["mean_rank"], len(curve),
             best["val_pdf_js"], best["val_conv_mmd2"]), flush=True)
    return {"curve": curve, "best": best}


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(int(args.threads))
    t0 = time.time()

    # group prefixes by class so each class's data is installed once
    by_class = {}
    for pref in args.prefix:
        snaps = snapshots_of(pref)
        cls = None
        if snaps:
            meta = evaluate_r3._load_meta(snaps[0][1])
            cls = (meta or {}).get("class_name")
        by_class.setdefault(cls, []).append(pref)

    rows = []
    for cls in sorted(by_class, key=lambda x: (x is not None, str(x))):
        if cls:
            data_lens_class.install(cls)
        sel = val_extra.selection_images(cls)
        n_sel, n_held = val_extra.check_disjoint(cls)
        print("[select] class=%r selection block %s, disjoint from the %d held-out images"
              % (cls, sel.shape, n_held), flush=True)

        # the held-out scoring context, built exactly as evaluate_r3b builds it
        d = data_lens_r3.load_r3(data_lens_r3.N_TRAIN_MAX, n_holdout=args.n_holdout)
        ctx = {"holdout": d.holdout_images, "holdout_crops": d.holdout_crops,
               "train_images": d.train_images}
        ctx["hold_q"] = evaluate_r3.quantity_table(ctx["holdout"])
        ctx["hold_profile"] = evaluate_r3.mean_profile(ctx["holdout"])
        ctx["holdout_mean_total"] = float(ctx["holdout"].reshape(ctx["holdout"].shape[0], -1)
                                          .sum(axis=1).mean())
        ctx["holdout_aligned"] = evaluate_r3.align_to_target(ctx["holdout"],
                                                             ctx["holdout_mean_total"])
        ctx["train_aligned"] = evaluate_r3.align_to_target(ctx["train_images"],
                                                           ctx["holdout_mean_total"])
        ctx["holdout_dispersion"] = {
            "holdout_nn_within": evaluate_r3b.nn_l2(ctx["holdout"], ctx["holdout"],
                                                    exclude_self=True),
            "holdout_e1_sd": float(ctx["hold_q"][:, evaluate_r3b.QC["e1"]].std()),
            "holdout_e2_sd": float(ctx["hold_q"][:, evaluate_r3b.QC["e2"]].std()),
        }

        for pref in by_class[cls]:
            res = select(pref, sel, args.n_holdout)
            if res is None:
                continue
            best = res["best"]
            row = evaluate_r3.score_run(best["run_dir"], ctx)
            if row is None:
                print("[select] could not score %s" % best["run_dir"], flush=True)
                continue
            row.pop("_crops_dir", None)
            row.update(evaluate_r3b.extra_columns(best["run_dir"], ctx))
            row.update(ctx["holdout_dispersion"])
            meta = evaluate_r3._load_meta(best["run_dir"]) or {}
            row.update({
                "kind": "%s_early_stopped" % args.label,
                "selection": "validation rank-mean of PDF-JS and raw Conv2D-MMD2, "
                             "ties to fewer steps",
                "selected_steps": int(best["steps"]),
                "val_pdf_js": best["val_pdf_js"],
                "val_conv_mmd2": best["val_conv_mmd2"],
                "val_n_images": int(sel.shape[0]),
                "n_candidates": len(res["curve"]),
                "candidate_steps": [c["steps"] for c in res["curve"]],
                "val_curve": [{k: c[k] for k in ("steps", "val_pdf_js", "val_conv_mmd2",
                                                 "mean_rank")} for c in res["curve"]],
                "sweep_prefix": os.path.abspath(pref),
                "class_name": cls, "S": meta.get("S"),
                "arm_base": str(meta.get("arm", "")).split(" s")[0],
            })
            # arm names carry the snapshot tag ("RBM-10bit s080k"); the table wants the arm
            row["arm_snapshot"] = row.get("arm")
            row["arm"] = row["arm_base"]
            rows.append(row)

            # the fixed-20k reference, for the appendix comparison
            fixed = [c for c in res["curve"] if c["steps"] == int(args.fixed_reference_steps)]
            if fixed:
                frow = evaluate_r3.score_run(fixed[0]["run_dir"], ctx)
                if frow is not None:
                    frow.pop("_crops_dir", None)
                    fmeta = evaluate_r3._load_meta(fixed[0]["run_dir"]) or {}
                    frow.update({
                        "kind": "%s_fixed_steps" % args.label,
                        "selected_steps": int(args.fixed_reference_steps),
                        "val_pdf_js": fixed[0]["val_pdf_js"],
                        "val_conv_mmd2": fixed[0]["val_conv_mmd2"],
                        "sweep_prefix": os.path.abspath(pref), "class_name": cls,
                        "S": fmeta.get("S"),
                        "arm_snapshot": frow.get("arm"),
                        "arm": str(fmeta.get("arm", "")).split(" s")[0],
                    })
                    rows.append(frow)

    mode = "a" if args.append else "w"
    with open(args.results, mode) as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print("[select] %s %d rows to %s (%.1f s)"
          % ("appended" if args.append else "wrote", len(rows), args.results,
             time.time() - t0), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
