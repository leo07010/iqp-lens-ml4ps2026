#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Re-score any round-3-shaped run directory under the **v11 metric protocol** (metrics_v11.py),
with the v10 ("legacy") numbers printed and stored beside them.

WHAT IT SCORES
---------------------------------------------------------------------------------------
Any tree whose run dirs look like round 3's -- `<run>/meta.json`, `<run>/generated_*.npy`
(and optionally `<run>/crops/`).  That covers `outputs_lens_r3`, `outputs_lens_r3b`,
`outputs_lens_class`, `outputs_lens_scaling`, `outputs_convergence` and the per-seed trees.
The held-out set, the arm labels, the flux-alignment target and the Conv2D encoder seed are
evaluate_r3/evaluate_r3b's; only the two metrics change (see metrics_v11.py's docstring for
what was wrong with v10 and what v11 fixes).  A run whose meta.json carries
`class_name` is scored against THAT class's 200 held-out images, exactly as
evaluate_class.py does; everything else is scored against the pooled 200.

CLI
---------------------------------------------------------------------------------------
    # one or more output trees -> one jsonl (v11 + legacy columns in every row)
    python3 evaluate_v11.py --outputs-dir outputs_lens_r3 outputs_lens_r3b \
        --results results_v11_seed0.jsonl

    # explicit run dirs instead of whole trees
    python3 evaluate_v11.py --runs outputs_lens_r3/rbm_n100 outputs_lens_r3b/pca4_gauss_n11 \
        --results /tmp/x.jsonl

    # add the 200-vs-200 real-vs-real floor rows (pooled + every class present)
    python3 evaluate_v11.py --outputs-dir outputs_lens_class --floor --n-boot 20 \
        --results results_v11_class.jsonl --floor-results results_v11_class_floor.jsonl

    # append to an existing jsonl instead of overwriting it
    python3 evaluate_v11.py --outputs-dir outputs_convergence --append \
        --results results_v11_seed0.jsonl

Useful flags: `--no-legacy` (skip the v10 columns, ~2x faster), `--threads N` (torch CPU
threads, default 4), `--quiet` (no per-row print), `--n-holdout` (default 200).

WHAT EVERY ROW CARRIES
---------------------------------------------------------------------------------------
    identity   run_dir, tree, arm, arm_base, script, param_count, seed, n_train, n_gen,
               class_name, classes_pooled, selected_steps (meta `steps`), sweep
    v11        pdf_js, pdf_js_aligned, conv_mmd2, conv_mmd2u, conv_mmd2_aligned,
               conv_mmd2u_aligned        (`*u` = unbiased U-statistic; the biased V-statistic
                                          keeps the plain name Table 2 quotes)
    v10        pdf_js_legacy, conv_mmd2_legacy, pdf_js_aligned_legacy,
               conv_mmd2_aligned_legacy
    honesty    gen_min, gen_max, gen_clip_frac(_above/_below), gen_clip_frac_aligned,
               gen_mean_total, holdout_mean_total, nn_holdout, sigma, sigma_aligned,
               ref_lo, ref_hi, conv_seed, metric_protocol="v11"

`gen_clip_frac` is the fraction of generated pixels v11's fixed [0, 1] histogram range and
ref-derived encoder range had to clip: under v10 those pixels stretched the yardstick
instead, which is the bug v11 removes, so this column is how a reader sees which arms were
being flattered.

This file is NEW.  metrics.py, evaluate_r3.py, evaluate_r3b.py, evaluate_class.py and every
existing results_*.jsonl are untouched.

Python 3.9 compatible (no `X | None`).  CPU only -- ~0.5 s per run.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import data_lens_class                                   # noqa: E402
import data_lens_r3 as D                                 # noqa: E402
import evaluate_r3                                       # noqa: E402
import metrics as legacy_metrics                         # noqa: E402
import metrics_v11 as M11                                # noqa: E402
from evaluate_lens import nn_l2                          # noqa: E402

_GEN_RE = re.compile(r"generated_(\d+)\.npy$")
CLASSES = data_lens_class.CLASSES


# --------------------------------------------------------------------------------------
# run discovery (same rule as evaluate_r3._stack / _load_meta)
# --------------------------------------------------------------------------------------
def load_meta(run_dir):
    path = os.path.join(run_dir, "meta.json")
    if not os.path.isfile(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def stack_generated(dir_path):
    found = []
    for path in glob.glob(os.path.join(dir_path, "generated_*.npy")):
        m = _GEN_RE.match(os.path.basename(path))
        if m:
            found.append((int(m.group(1)), path))
    if not found:
        return None
    found.sort()
    return np.stack([np.load(p) for _, p in found], axis=0).astype(np.float32)


def discover(outputs_dirs, runs, skip_missing=True):
    """Run dirs under each tree.  A child that has no meta.json but contains children that do
    is descended into once, so a per-seed tree (`outputs_seeds/s1/<run>`) can be passed as
    `--outputs-dir outputs_seeds` as well as `--outputs-dir outputs_seeds/s1`."""
    if runs:
        return [os.path.normpath(r) for r in runs]
    out = []
    for dd in outputs_dirs:
        dd = dd if os.path.isabs(dd) else os.path.join(REPO, dd)
        if not os.path.isdir(dd):
            if skip_missing:
                print("[evaluate_v11] skipping absent --outputs-dir %r" % dd)
                continue
            raise SystemExit("evaluate_v11.py: --outputs-dir %r is not a directory" % dd)
        for child in sorted(x for x in glob.glob(os.path.join(dd, "*")) if os.path.isdir(x)):
            if os.path.isfile(os.path.join(child, "meta.json")):
                out.append(child)
                continue
            deeper = sorted(x for x in glob.glob(os.path.join(child, "*"))
                            if os.path.isdir(x) and os.path.isfile(os.path.join(x, "meta.json")))
            if deeper:
                out.extend(deeper)
    return out


def class_of(meta, run_dir):
    """The class a run belongs to, or None for the pooled split.  meta['class_name'] is what
    train_iqp_affine_class.py writes; the directory-name fallback matches the sbatch's
    `<class>_*` naming."""
    c = meta.get("class_name")
    if c in CLASSES:
        return c
    base = os.path.basename(os.path.normpath(run_dir))
    for cc in CLASSES:
        if base.startswith(cc + "_"):
            return cc
    return None


# --------------------------------------------------------------------------------------
# one scoring context (pooled, or one class): held-out images + the frozen yardsticks
# --------------------------------------------------------------------------------------
class Context(object):
    __slots__ = ("class_name", "holdout", "holdout_aligned", "target", "prep_raw", "prep_al")

    def __repr__(self):
        return ("Context(class=%s, holdout=%s, target=%.4f, sigma=%.4f/%.4f)"
                % (self.class_name, tuple(self.holdout.shape), self.target,
                   self.prep_raw.sigma, self.prep_al.sigma))


def build_context(class_name, n_holdout=D.N_HOLDOUT):
    """Held-out set + the two RefContexts (raw and flux-aligned).  The pooled context MUST be
    built before any per-class one: data_lens_class.install() monkey-patches data_lens_r3 and
    has no uninstall."""
    if class_name is not None:
        data_lens_class.install(class_name)
    d = D.load_r3(D.N_TRAIN_MAX, n_holdout=int(n_holdout))
    ctx = Context()
    ctx.class_name = class_name
    ctx.holdout = np.ascontiguousarray(d.holdout_images)
    ctx.target = float(ctx.holdout.reshape(ctx.holdout.shape[0], -1).sum(axis=1).mean())
    ctx.holdout_aligned = evaluate_r3.align_to_target(ctx.holdout, ctx.target)
    ctx.prep_raw = M11.prepare_ref(ctx.holdout, seed=M11.CONV_SEED)
    ctx.prep_al = M11.prepare_ref(ctx.holdout_aligned, seed=M11.CONV_SEED)
    return ctx


# --------------------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------------------
def score_run(run_dir, ctx, with_legacy=True):
    meta = load_meta(run_dir)
    if meta is None:
        return None
    gen = stack_generated(run_dir)
    if gen is None:
        return None

    gen_al = evaluate_r3.align_to_target(gen, ctx.target)
    js = M11.pdf_js_v11(ctx.holdout, gen)
    js_al = M11.pdf_js_v11(ctx.holdout_aligned, gen_al,
                           lo=ctx.prep_al.lo, hi=ctx.prep_al.hi)
    mm = M11.conv_mmd2_v11(ctx.holdout, gen, prepared=ctx.prep_raw)
    mm_al = M11.conv_mmd2_v11(ctx.holdout_aligned, gen_al, prepared=ctx.prep_al)

    row = {
        "run_dir": run_dir,
        "tree": os.path.basename(os.path.dirname(os.path.normpath(run_dir))),
        "arm": meta.get("arm", os.path.basename(os.path.normpath(run_dir))),
        "arm_base": meta.get("arm_base", meta.get("arm")),
        "script": meta.get("script"),
        "param_count": meta.get("param_count"),
        "seed": meta.get("seed"),
        "n_train": int(meta.get("n_train", 0)),
        "n_gen": int(gen.shape[0]),
        "n_holdout": int(ctx.holdout.shape[0]),
        "class_name": ctx.class_name,
        "classes_pooled": ctx.class_name is None,
        "selected_steps": meta.get("steps"),
        "sweep": meta.get("sweep"),
        "objective": meta.get("objective"),
        "protocol": meta.get("protocol"),
        "metric_protocol": "v11",
        # ---- v11 ----
        "pdf_js": js["pdf_js"],
        "pdf_js_aligned": js_al["pdf_js"],
        "conv_mmd2": mm["conv_mmd2"],
        "conv_mmd2u": mm["conv_mmd2u"],
        "conv_mmd2_aligned": mm_al["conv_mmd2"],
        "conv_mmd2u_aligned": mm_al["conv_mmd2u"],
        # ---- honesty / provenance ----
        "gen_min": js["gen_min"],
        "gen_max": js["gen_max"],
        "gen_clip_frac": js["gen_clip_frac"],
        "gen_clip_frac_above": js["gen_clip_frac_above"],
        "gen_clip_frac_below": js["gen_clip_frac_below"],
        "gen_clip_frac_aligned": js_al["gen_clip_frac"],
        "gen_max_aligned": js_al["gen_max"],
        "gen_mean_total": float(gen.reshape(gen.shape[0], -1).sum(axis=1).mean()),
        "holdout_mean_total": ctx.target,
        "nn_holdout": nn_l2(gen, ctx.holdout),
        "sigma": ctx.prep_raw.sigma,
        "sigma_aligned": ctx.prep_al.sigma,
        "ref_lo": ctx.prep_raw.lo,
        "ref_hi": ctx.prep_raw.hi,
        "ref_lo_aligned": ctx.prep_al.lo,
        "ref_hi_aligned": ctx.prep_al.hi,
        "conv_seed": M11.CONV_SEED,
    }
    if with_legacy:
        row["pdf_js_legacy"] = float(legacy_metrics.pdf_js(ctx.holdout, gen))
        row["conv_mmd2_legacy"] = float(legacy_metrics.conv_mmd2(ctx.holdout, gen,
                                                                 seed=M11.CONV_SEED))
        row["pdf_js_aligned_legacy"] = float(legacy_metrics.pdf_js(ctx.holdout_aligned, gen_al))
        row["conv_mmd2_aligned_legacy"] = float(
            legacy_metrics.conv_mmd2(ctx.holdout_aligned, gen_al, seed=M11.CONV_SEED))
    return row


def render_table(rows):
    def f(v, spec="{:.4g}"):
        return "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else spec.format(v)

    head = ("| Arm | class | N | steps | PDF–JS v11 | PDF–JS v10 | MMD² v11 | MMD² v10 | "
            "MMD²_u v11 | MMD²_al v11 | MMD²_al v10 | MMD²_u,al v11 | gen_max | clip frac |")
    lines = [head, "|" + "---|" * 14]
    for r in sorted(rows, key=lambda z: (str(z.get("class_name")), int(z.get("n_train") or 0),
                                         str(z.get("selected_steps")), str(z["arm"]))):
        lines.append("| " + " | ".join([
            str(r["arm"]), str(r.get("class_name") or "pooled"), str(r.get("n_train")),
            str(r.get("selected_steps") or "—"),
            f(r.get("pdf_js")), f(r.get("pdf_js_legacy")),
            f(r.get("conv_mmd2")), f(r.get("conv_mmd2_legacy")),
            f(r.get("conv_mmd2u")),
            f(r.get("conv_mmd2_aligned")), f(r.get("conv_mmd2_aligned_legacy")),
            f(r.get("conv_mmd2u_aligned")),
            f(r.get("gen_max")), f(r.get("gen_clip_frac"), "{:.3g}")]) + " |")
    return "\n".join(["### v11 vs v10, same runs, same held-out set", ""] + lines)


# --------------------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="v11 re-scoring of round-3-shaped run trees.",
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outputs-dir", nargs="*", default=["outputs_lens_r3", "outputs_lens_r3b"])
    p.add_argument("--runs", nargs="*", default=None)
    p.add_argument("--results", default=os.path.join(REPO, "results_v11_seed0.jsonl"))
    p.add_argument("--append", action="store_true")
    p.add_argument("--no-legacy", action="store_true")
    p.add_argument("--n-holdout", type=int, default=D.N_HOLDOUT)
    p.add_argument("--floor", action="store_true",
                   help="also compute the 200-vs-200 real-vs-real floor for every kind "
                        "present and append its summary rows to --results")
    p.add_argument("--n-boot", type=int, default=20)
    p.add_argument("--n-side", type=int, default=200)
    p.add_argument("--draw-mode", choices=("resample", "disjoint"), default="resample")
    p.add_argument("--floor-results", default=os.path.join(REPO, "results_v11_floor.jsonl"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(int(args.threads))
    t0 = time.time()
    with_legacy = not args.no_legacy

    run_dirs = discover(args.outputs_dir, args.runs)
    if not run_dirs:
        raise SystemExit("evaluate_v11.py: no run directories found")

    # group by class BEFORE building any context (install() is one-way)
    groups = {}
    for rd in run_dirs:
        meta = load_meta(rd)
        if meta is None:
            print("[evaluate_v11] skipped %s (no meta.json)" % rd)
            continue
        groups.setdefault(class_of(meta, rd), []).append(rd)
    if not groups:
        raise SystemExit("evaluate_v11.py: no run directory had a meta.json")

    order = ([None] if None in groups else []) + [c for c in CLASSES if c in groups]
    rows = []
    contexts = {}
    for c in order:
        ctx = build_context(c, n_holdout=args.n_holdout)
        contexts[c] = ctx
        print("[evaluate_v11] %s | held-out=%s target=%.4f sigma=%.4f (aligned %.4f) "
              "ref-range=[%.3g, %.3g]"
              % ("pooled" if c is None else c, tuple(ctx.holdout.shape), ctx.target,
                 ctx.prep_raw.sigma, ctx.prep_al.sigma, ctx.prep_raw.lo, ctx.prep_raw.hi),
              flush=True)
        for rd in groups[c]:
            row = score_run(rd, ctx, with_legacy=with_legacy)
            if row is None:
                print("[evaluate_v11] skipped %s (no generated_*.npy)" % rd)
                continue
            rows.append(row)
            if not args.quiet:
                print("[evaluate_v11]   %-46s N=%-4s js %.4e -> %.4e | mmd2 %.5f -> %.5f "
                      "| mmd2u %.5f | gen_max %.3f clip %.3g"
                      % (row["arm"], row["n_train"], row.get("pdf_js_legacy", float("nan")),
                         row["pdf_js"], row.get("conv_mmd2_legacy", float("nan")),
                         row["conv_mmd2"], row["conv_mmd2u"], row["gen_max"],
                         row["gen_clip_frac"]), flush=True)
    if not rows:
        raise SystemExit("evaluate_v11.py: nothing scored")

    floor_summaries = []
    if args.floor:
        class_names, val_y, val_x = M11.read_val_arrays()
        floor_rows = []
        for kind_key, c in enumerate(order):
            pool = M11.val_pool(val_x, val_y, class_names, c)
            fr, fs = M11.floor_kind(pool, contexts[c].holdout, n_boot=args.n_boot,
                                    n_side=args.n_side, seed=args.seed, kind_key=kind_key,
                                    draw_mode=args.draw_mode, class_name=c,
                                    with_legacy=with_legacy, verbose=not args.quiet)
            floor_rows.extend(fr)
            floor_summaries.append(fs)
        del val_x
        with open(args.floor_results, "w") as fh:
            for r in floor_rows:
                fh.write(json.dumps(r) + "\n")
            for s in floor_summaries:
                fh.write(json.dumps(s) + "\n")
        print("[evaluate_v11] wrote %d floor draws (+%d summaries) to %s"
              % (len(floor_rows), len(floor_summaries), args.floor_results))

    mode = "a" if args.append else "w"
    with open(args.results, mode) as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
        for s in floor_summaries:
            s = dict(s)
            s["arm"] = "HOLDOUT floor %dv%d (v11)" % (args.n_side, args.n_side)
            s["run_dir"] = None
            fh.write(json.dumps(s) + "\n")
    print("[evaluate_v11] wrote %d rows to %s (%s, %.1f s)"
          % (len(rows) + len(floor_summaries), args.results,
             "appended" if args.append else "overwrote", time.time() - t0))
    print()
    print(render_table(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
