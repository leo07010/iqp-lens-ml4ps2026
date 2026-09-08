#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Merge the per-seed result files and write the mean +- sd versions of the paper's tables.

Reads
  seed 0 (already on disk, never re-run)      results_r3.jsonl        Table 2 rows 1-2, 5-6
                                              results_r3b.jsonl       Table 2 rows 3-4
                                              results_class.jsonl     Table 3
                                              results_scaling.jsonl   Fig. 5
  seeds 1-4 (seeds/submit_seeds_iqp.sbatch)   results_seeds_raw/pooled_s<K>.jsonl
                                              results_seeds_raw/class_s<K>.jsonl
  baselines, ALL seeds, validation-early-stopped
                                              results_seeds_baselines_es.jsonl
  section-3.2 diagnostics                     results_seeds_diag.jsonl

Writes
  results_seeds_table2.jsonl    every row that feeds Table 2, each stamped with `seed`
  results_seeds_class.jsonl     ditto for Table 3
  results_seeds_scaling.jsonl   ditto for Fig. 5
  paper/ml4ps2026/seed_tables.md   mean +- sd (4 significant figures) for all three, plus
                                   the section-3.2 diagnostics and an overlap report

The overlap report is the point of the exercise: for every column it names the best arm and
then lists every arm whose mean +- 1 sd interval OVERLAPS the best arm's, i.e. every "win"
in the current paper that the seed spread does not support.

`sd` is the sample standard deviation (ddof = 1) over the seeds present.  A cell whose
`n_seeds` is below the full count is reported anyway but marked, so a partially finished
sweep can never be mistaken for a complete one.

    python3 aggregate_seeds.py [--seeds 0 1 2 3 4] [--out paper/ml4ps2026/seed_tables.md]

Python 3.9 compatible.  CPU only, seconds.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
PAPER = "<HOME>/gsoc2026-deeplense-qml/paper/ml4ps2026"
RAW = os.path.join(REPO, "results_seeds_raw")

# ---------------------------------------------------------------------------- the tables
# (label, arm string).  Order is the paper's.
TABLE2_ARMS = [
    ("IQP, paper protocol (Alg. 1)",         "IQP-lowvar-paper-MMD"),
    ("IQP, fixed basis, joint",              "IQP-lowvar-joint-MMD-resample"),
    ("IQP, learned dir. M=4 (flat MMD)",     "IQP-affine-M4-MMDflat-resample"),
    ("IQP, learned dir. M=4 (paper kernel)", "IQP-affine-M4-auto3-resample"),
    ("RBM",                                  "RBM-10bit"),
    ("DCGAN",                                "DCGAN-32"),
]
TABLE2_FLOOR = "HOLDOUT split-half"
TABLE2_N = [11, 100]
TABLE2_METRICS = [("pdf_js", "PDF-JS"), ("conv_mmd2", "MMD2"), ("conv_mmd2_aligned", "MMD2_al")]

CLASS_ARMS = [
    ("axion only",  "IQP-affine-M4-MMDflat-resample-axion"),
    ("cdm only",    "IQP-affine-M4-MMDflat-resample-cdm"),
    ("no sub only", "IQP-affine-M4-MMDflat-resample-no_sub"),
]
CLASS_POOLED = ("pooled", "IQP-affine-M4-MMDflat-resample")
CLASS_METRICS = [("pdf_js", "PDF-JS"), ("conv_mmd2", "MMD2"),
                 ("conv_mmd2_aligned", "MMD2_al"), ("projection_corr_mean", "proj. corr.")]

SCALING_ARMS = [
    ("IQP, M=4", "IQP-affine-M4-MMDflat-resample"),
    ("RBM",      "RBM-10bit"),
    ("DCGAN",    "DCGAN-32"),
]
SCALING_N = [10, 30, 50, 70, 90, 100]
SCALING_METRICS = [("conv_mmd2", "MMD2"), ("conv_mmd2_aligned", "MMD2_al")]

# lower is better for every metric except the projection correlation
HIGHER_IS_BETTER = {"projection_corr_mean", "corr", "corr_anchor", "corr_others_mean"}

# arms whose seed-0 rows live in results_r3.jsonl rather than results_r3b.jsonl
R3_ARMS = {"IQP-lowvar-paper-MMD", "IQP-lowvar-joint-MMD-resample", "RBM-10bit", "DCGAN-32",
           TABLE2_FLOOR}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--out", default=os.path.join(PAPER, "seed_tables.md"))
    p.add_argument("--repo", default=REPO)
    return p.parse_args(argv)


# ------------------------------------------------------------------------------- helpers
def read_jsonl(path):
    if not os.path.isfile(path):
        return []
    with open(path) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def sig4(x):
    """Four significant figures, except where that would hide the point.

    A correlation of 0.999966 renders as a bare "1" at 4 s.f., which reads as an exact fit
    and cannot be compared against the paper's quoted 0.99997.  When rounding collapses a
    value onto 1 (or -1) without it actually being 1, widen to six figures.
    """
    if x is None:
        return "--"
    x = float(x)
    if math.isnan(x) or math.isinf(x):
        return "--"
    out = "%.4g" % x
    if out in ("1", "-1") and abs(abs(x) - 1.0) > 0:
        return "%.6g" % x
    return out


def mean_sd(vals):
    vals = [float(v) for v in vals if v is not None]
    n = len(vals)
    if n == 0:
        return None, None, 0
    m = sum(vals) / n
    if n == 1:
        return m, 0.0, 1
    var = sum((v - m) ** 2 for v in vals) / (n - 1)
    return m, math.sqrt(var), n


def cell(vals, n_expected):
    m, s, n = mean_sd(vals)
    if n == 0:
        return "--", None, None, 0
    mark = "" if n == n_expected else ("!" if n > n_expected else "*")
    if n == 1:
        return "%s (1 seed)%s" % (sig4(m), mark), m, 0.0, 1
    return "%s +- %s%s" % (sig4(m), sig4(s), mark), m, s, n


def overlaps(a, b):
    """Do the two (mean, sd) intervals [m - s, m + s] intersect?"""
    (ma, sa), (mb, sb) = a, b
    return not (ma + sa < mb - sb or mb + sb < ma - sa)


# ------------------------------------------------------------------- collect the seed rows
IQP_T2_ARMS = {"IQP-lowvar-paper-MMD", "IQP-lowvar-joint-MMD-resample",
               "IQP-affine-M4-MMDflat-resample", "IQP-affine-M4-auto3-resample"}
BASELINE_ARMS = {"RBM-10bit", "DCGAN-32"}
ES_RESULTS = "results_seeds_baselines_es.jsonl"


def collect(seeds, repo):
    """-> (table2, class, scaling, table2_fixed, scaling_fixed) rows, each stamped with `seed`.

    The IQP arms and the two classical baselines come from different places, because they are
    now trained under different protocols:

      IQP        fixed 4,000 steps, no early stopping.  Seed 0 from the round-3 / round-3b
                 result files, seeds 1-4 from seeds/submit_seeds_iqp.sbatch.
      baselines  validation-based early stopping over the sweep snapshots, ALL five seeds
                 from seeds/submit_seeds_baselines_es.sbatch via seeds/select_checkpoint.py.
                 The convergence sweep showed no fixed step count is defensible for either
                 (the RBM is undertrained at 20k, the DCGAN is non-monotone and degrades),
                 so a fixed count would have been a choice made for them rather than by them.

    The old fixed-20,000-step baseline rows are still collected, into the `_fixed` lists, so
    the appendix can show what changed.  They are NOT in the primary tables.
    """
    t2, cls, scl, t2_fixed, scl_fixed = [], [], [], [], []
    # select_checkpoint.py emits a fixed-20k row for EVERY seed including 0, so the seed-0
    # baseline rows sitting in results_r3.jsonl / results_scaling.jsonl are the same protocol
    # measured twice.  Take them only when the ES file has not been produced.
    have_es = bool(read_jsonl(os.path.join(repo, ES_RESULTS)))

    # ---- seed 0: the existing, untouched result files (IQP arms and the floor)
    if 0 in seeds:
        r3 = read_jsonl(os.path.join(repo, "results_r3.jsonl"))
        r3b = read_jsonl(os.path.join(repo, "results_r3b.jsonl"))
        rcl = read_jsonl(os.path.join(repo, "results_class.jsonl"))
        rsc = read_jsonl(os.path.join(repo, "results_scaling.jsonl"))
        for r in r3:
            arm, n = r.get("arm"), r.get("n_train")
            if n not in TABLE2_N:
                continue
            if arm in (IQP_T2_ARMS & R3_ARMS) or arm == TABLE2_FLOOR:
                t2.append(dict(r, seed=0, _src="results_r3.jsonl"))
            elif arm in BASELINE_ARMS and not have_es:
                t2_fixed.append(dict(r, seed=0, _src="results_r3.jsonl", _steps=20000))
        for r in r3b:
            if r.get("arm") in (IQP_T2_ARMS - R3_ARMS) and r.get("n_train") in TABLE2_N:
                t2.append(dict(r, seed=0, _src="results_r3b.jsonl"))
        wantc = {a for _l, a in CLASS_ARMS}
        for r in rcl:
            if r.get("arm") in wantc:
                cls.append(dict(r, seed=0, _src="results_class.jsonl"))
        for r in rsc:
            arm, n = r.get("arm"), r.get("n_train")
            if n not in SCALING_N:
                continue
            if arm == "IQP-affine-M4-MMDflat-resample" or arm == TABLE2_FLOOR:
                scl.append(dict(r, seed=0, _src="results_scaling.jsonl"))
            elif arm in BASELINE_ARMS and not have_es:
                scl_fixed.append(dict(r, seed=0, _src="results_scaling.jsonl", _steps=20000))

    # ---- seeds 1..K: the IQP sweep's own evaluations
    for s in [x for x in seeds if x != 0]:
        pooled = read_jsonl(os.path.join(RAW, "pooled_s%d.jsonl" % s))
        clsrows = read_jsonl(os.path.join(RAW, "class_s%d.jsonl" % s))
        for r in pooled:
            arm, n = r.get("arm"), r.get("n_train")
            name = "pooled_s%d.jsonl" % s
            if (arm in IQP_T2_ARMS or arm == TABLE2_FLOOR) and n in TABLE2_N:
                t2.append(dict(r, seed=s, _src=name))
            if (arm == "IQP-affine-M4-MMDflat-resample" or arm == TABLE2_FLOOR) and n in SCALING_N:
                scl.append(dict(r, seed=s, _src=name))
        wantc = {a for _l, a in CLASS_ARMS}
        for r in clsrows:
            if r.get("arm") in wantc:
                cls.append(dict(r, seed=s, _src="class_s%d.jsonl" % s))

    # ---- the baselines, every seed, from the early-stopping run
    for r in read_jsonl(os.path.join(repo, ES_RESULTS)):
        arm, n, sd = r.get("arm"), r.get("n_train"), r.get("seed")
        if arm not in BASELINE_ARMS or sd not in seeds:
            continue
        row = dict(r, _src=ES_RESULTS, _steps=r.get("selected_steps"))
        if r.get("kind") == "baseline_early_stopped":
            if n in TABLE2_N:
                t2.append(row)
            if n in SCALING_N:
                scl.append(row)
        elif r.get("kind") == "baseline_fixed_steps":
            if n in TABLE2_N:
                t2_fixed.append(row)
            if n in SCALING_N:
                scl_fixed.append(row)

    # the pooled reference row of Table 3 is Table 2's M = 4 flat arm
    for r in t2:
        if r.get("arm") == CLASS_POOLED[1]:
            cls.append(dict(r, _pooled=True))
    return t2, cls, scl, t2_fixed, scl_fixed


def check_data_invariance(rows, lines):
    """The held-out set must be the same object for every seed.  If it is not, every
    comparison in this file is invalid, so say so loudly rather than printing a table.

    Keyed on (n_holdout, class_name): the `HOLDOUT split-half` reference rows legitimately
    carry n_holdout = 100 (one half of the 200), and each per-class arm legitimately has its
    own class's held-out set.  Within one key the held-out statistics must not move at all.
    """
    seen = {}
    for r in rows:
        hmt = r.get("holdout_mean_total")
        if hmt is None:
            continue
        hs = r.get("holdout_std")
        hs_key = (tuple(sorted((str(a), float(b)) for a, b in hs.items()))
                  if isinstance(hs, dict) else hs)
        k = (r.get("n_holdout"), r.get("class_name"))
        seen.setdefault(k, {}).setdefault((float(hmt), hs_key), set()).add(r.get("seed"))
    bad = [(k, v) for k, v in sorted(seen.items(), key=lambda t: str(t[0])) if len(v) > 1]
    if bad:
        lines.append("> **STOP -- the held-out set moved between seeds.** "
                     "The numbers below are not comparable.")
        for k, vals in bad:
            for v, ss in vals.items():
                lines.append(">  n_holdout=%s class=%s  holdout_mean_total=%s  seeds %s"
                             % (k[0], k[1], sig4(v[0]),
                                sorted(x for x in ss if x is not None)))
        lines.append("")
        return False
    parts = []
    for k, vals in sorted(seen.items(), key=lambda t: str(t[0])):
        v = list(vals.keys())[0]
        ss = sorted(x for x in list(vals.values())[0] if x is not None)
        parts.append("n_holdout=%s%s mean total %s over seeds %s"
                     % (k[0], "" if k[1] is None else (" class=%s" % k[1]), sig4(v[0]),
                        ",".join(str(x) for x in ss) if ss else "-"))
    lines.append("Held-out set identical for every seed, on every held-out block "
                 "(%s) -- the split does not move with `--seed`, as `data_lens_r3.load_r3` "
                 "promises." % "; ".join(parts))
    lines.append("")
    return True


# --------------------------------------------------------------------------- table writers
def dedupe(rows, what="rows"):
    """Collapse rows that describe the SAME (arm, N, seed, class, run dir).

    A duplicate is not harmless here: averaging one seed twice pulls the mean toward it and
    shrinks the sample sd, which is exactly the quantity this file exists to report.  It has
    already happened once -- seed 0's projection-restart rows were written by two different
    jobs -- so the guard is permanent rather than a one-off clean-up.  Duplicates are counted
    and announced; they are never silently averaged.
    """
    seen, out, dups = {}, [], 0
    for r in rows:
        k = (r.get("arm"), r.get("n_train"), r.get("seed"), r.get("class_name"),
             r.get("run_dir"), r.get("kind"), r.get("_src"))
        if k in seen:
            dups += 1
            # keep whichever copy carries more columns: an older job may have written the
            # same run with a script version that recorded fewer of them
            i = seen[k]
            if len(r) > len(out[i]):
                out[i] = r
            continue
        seen[k] = len(out)
        out.append(r)
    if dups:
        print("[aggregate] WARNING: dropped %d duplicate %s (same arm/N/seed/class/run_dir)"
              % (dups, what))
    return out


def check_seed_counts(rows, what, expected):
    """Announce any (arm, N, class) cell that does not have exactly one row per seed."""
    by = {}
    for r in rows:
        by.setdefault((r.get("arm"), r.get("n_train"), r.get("class_name")), []).append(
            r.get("seed"))
    for k, seeds_seen in sorted(by.items(), key=lambda t: str(t[0])):
        got = [x for x in seeds_seen if x is not None]
        if not got:
            continue
        if len(got) != len(set(got)):
            print("[aggregate] WARNING: %s %s has repeated seeds %s" % (what, k, sorted(got)))
        elif len(got) != expected:
            print("[aggregate] note: %s %s has %d/%d seeds (%s)"
                  % (what, k, len(got), expected, sorted(got)))


REPEAT_RUNS = []          # (arm, N, seed, metric, value_a, value_b) for identical configs


def dedupe_cells(rows, what="rows", metrics=("pdf_js", "conv_mmd2")):
    """Collapse rows that are the SAME logical cell even when the run directories differ.

    The M = 4 flat arm at N = 100 was trained once for round 3b and once for the scaling
    sweep, at the same seed and the same configuration.  Both trees get re-scored, so the
    cell arrives twice and seed 0 would be weighted double.  Collapse on
    (arm, N, seed, class, kind) and check the copies actually agree first -- if two runs of
    the "same" configuration disagree, that is a determinism bug and must not be averaged
    away silently.
    """
    seen, out, dups, disagree = {}, [], 0, []
    for r in rows:
        k = (r.get("arm"), r.get("n_train"), r.get("seed"), r.get("class_name"), r.get("kind"))
        if k[2] is None:                       # reference rows carry no seed: keep them all
            out.append(r)
            continue
        if k in seen:
            dups += 1
            a = out[seen[k]]
            for m in metrics:
                va, vb = a.get(m), r.get(m)
                if va is not None and vb is not None and abs(float(va) - float(vb)) > 1e-9:
                    disagree.append("%s N=%s seed=%s %s: %.6g vs %.6g"
                                    % (k[0], k[1], k[2], m, va, vb))
                    REPEAT_RUNS.append((k[0], k[1], k[2], m, float(va), float(vb)))
            continue
        seen[k] = len(out)
        out.append(r)
    if dups:
        print("[aggregate] collapsed %d duplicate %s onto their (arm, N, seed, class) cell"
              % (dups, what))
    for d in disagree:
        # not necessarily a bug: two runs of one configuration can differ where the framework
        # itself is non-deterministic.  Reported, quantified in `determinism_section`, and
        # never averaged into a seed spread.
        print("[aggregate] repeat run of an identical config differs -- %s" % d)
    return out


def grouped(rows, key_fn):
    out = {}
    for r in rows:
        out.setdefault(key_fn(r), []).append(r)
    return out


def build_table(rows, arms, ns, metrics, n_expected, label_head="Model"):
    """-> (markdown lines, stats dict {(arm,n,metric): (mean, sd, n_seeds)})."""
    by = grouped(rows, lambda r: (r.get("arm"), r.get("n_train")))
    stats = {}
    head = ["| %s | N |" % label_head] + []
    hdr = "| %s | N | " % label_head + " | ".join(nm for _k, nm in metrics) + " | seeds |"
    sep = "|" + "---|" * (3 + len(metrics))
    lines = [hdr, sep]
    for lab, arm in arms:
        for n in ns:
            rs = by.get((arm, n), [])
            cells, nseeds = [], 0
            for mk, _nm in metrics:
                txt, m, s, k = cell([r.get(mk) for r in rs], n_expected)
                cells.append(txt)
                nseeds = max(nseeds, k)
                if m is not None:
                    stats[(arm, n, mk)] = (m, s if s is not None else 0.0, k)
            lines.append("| %s | %d | %s | %d |" % (lab, n, " | ".join(cells), nseeds))
    return lines, stats


def overlap_report(stats, arms, ns, metrics, lines, title):
    lines.append("### %s -- wins that the seed spread does not support" % title)
    lines.append("")
    lines.append("| N | metric | best arm (mean) | overlaps it within 1 sd |")
    lines.append("|---|---|---|---|")
    any_row = False
    for n in ns:
        for mk, nm in metrics:
            hib = mk in HIGHER_IS_BETTER
            # A cell with one seed has sd = 0 by construction, and a zero-width interval
            # almost never overlaps anything -- it would be reported as an unambiguous win
            # produced by a MISSING standard deviation rather than a small one.
            have = [(lab, arm, stats[(arm, n, mk)]) for lab, arm in arms
                    if (arm, n, mk) in stats and stats[(arm, n, mk)][2] >= 2]
            thin = [lab for lab, arm in arms
                    if (arm, n, mk) in stats and stats[(arm, n, mk)][2] < 2]
            if len(have) < 2:
                if thin:
                    lines.append("| %d | %s | _not enough seeds_ | %s excluded (1 seed) |"
                                 % (n, nm, ", ".join(thin)))
                    any_row = True
                continue
            best = max(have, key=lambda t: t[2][0]) if hib else min(have, key=lambda t: t[2][0])
            b_lab, _b_arm, (bm, bs, _bk) = best
            tied = [lab for lab, _a, (m, s, _k) in have
                    if lab != b_lab and overlaps((bm, bs), (m, s))]
            note = ", ".join(tied) if tied else "**none -- the win holds**"
            if thin:
                note += " _(%s excluded: 1 seed)_" % ", ".join(thin)
            lines.append("| %d | %s | %s (%s +- %s) | %s |"
                         % (n, nm, b_lab, sig4(bm), sig4(bs), note))
            any_row = True
    if not any_row:
        lines.append("| -- | -- | not enough seeds yet | -- |")
    lines.append("")


RESTART_ARMS = [
    ("IQP, learned dir. M=4 (flat MMD)",     "IQP-affine-M4-MMDflat-resample"),
    ("IQP, learned dir. M=4 (paper kernel)", "IQP-affine-M4-auto3-resample"),
]
RESTART_COLS = [
    ("proj_corr_restart0_zero_init",           "restart 0 only (zero init)"),
    ("proj_corr_restart1_nearest_train_init",  "restart 1 only (nearest-train init)"),
    ("proj_corr_best_excl_restart1",           "best excl. restart 1"),
    ("proj_corr_best_by_loss",                 "best by LOSS"),
    ("proj_corr_best_all_restarts",            "best by CORR (reported)"),
]


def restarts_section(repo, seeds, lines):
    """The held-out projection correlation split by which restart produced it.

    The paper reports the best over 8 restarts, and restart 1 starts at the nearest TRAINING
    image's trained z.  A reviewer will read that as starting next to the answer, so the
    number is reported here four ways instead of one.
    """
    rows = dedupe(read_jsonl(os.path.join(repo, "results_seeds_restarts.jsonl")),
                  "projection-restart rows")
    lines.append("## Held-out projection correlation, by restart")
    lines.append("")
    if not rows:
        lines.append("_results_seeds_restarts.jsonl is empty -- "
                     "`seeds/diag_projection_restarts.py` has not run._")
        lines.append("")
        return
    bad = [r for r in rows if not r.get("replay_matches_recorded")]
    lines.append("Restart 0 starts at z = 0 and uses no information from the training set. "
                 "Restart 1 starts at the nearest training image's trained z. Restarts 2-7 "
                 "start at draws from the Gaussian fitted to the trained z. Higher is better.")
    lines.append("")
    lines.append("Two selection rules are shown. **best by CORR** is what the paper reports: "
                 "of the eight answers, keep the one that correlates best with the held-out "
                 "image -- i.e. select on the metric being reported. **best by LOSS** keeps "
                 "the one with the lowest training objective, which is the choice a model "
                 "without access to the target could actually make. The gap between the two "
                 "columns is the size of that advantage.")
    lines.append("")
    hdr = "| arm | N | " + " | ".join(nm for _k, nm in RESTART_COLS) + " | restart 1 wins | seeds |"
    lines.append(hdr)
    lines.append("|" + "---|" * (4 + len(RESTART_COLS)))
    by = grouped(rows, lambda r: (r.get("arm"), r.get("n_train")))
    for lab, arm in RESTART_ARMS:
        for n in TABLE2_N:
            rs = by.get((arm, n), [])
            if not rs:
                continue
            cells, k = [], 0
            for mk, _nm in RESTART_COLS:
                txt, _m, _s, kk = cell([r.get(mk) for r in rs], len(seeds))
                cells.append(txt)
                k = max(k, kk)
            win, _m, _s, _k = cell([r.get("restart1_win_share") for r in rs], len(seeds))
            lines.append("| %s | %d | %s | %s | %d |" % (lab, n, " | ".join(cells), win, k))
    lines.append("")
    if bad:
        lines.append("> **%d of %d replays did not reproduce the run's recorded "
                     "`projection_corr_mean`.** Those rows are in the table above but must not "
                     "be quoted until the cause is found: %s"
                     % (len(bad), len(rows),
                        ", ".join(os.path.basename(r.get("run_dir", "?")) for r in bad)))
    else:
        lines.append("All %d replays reproduced their run's recorded `projection_corr_mean` "
                     "to within 1e-6, so the decomposition is of the reported number itself, "
                     "not of a re-run." % len(rows))
    lines.append("")


THETA0_ARMS = [
    ("theta0 warm-start = KL (published)",   "IQP-affine-M4-MMDflat-resample"),
    ("theta0 warm-start = flat MMD",         "IQP-affine-M4-MMDflat-resample-th0mmdflat"),
]


def selected_steps_section(repo, seeds, lines):
    """What the validation set actually picked for each baseline cell."""
    rows = [r for r in read_jsonl(os.path.join(repo, ES_RESULTS))
            if r.get("kind") == "baseline_early_stopped"]
    lines.append("## Baseline step counts chosen on the validation set")
    lines.append("")
    if not rows:
        lines.append("_%s is empty -- the early-stopping run has not finished._" % ES_RESULTS)
        lines.append("")
        return
    lines.append("The step count is chosen per (arm, N, seed) on 200 real images that are "
                 "disjoint from the 200 held-out images, by the mean of the PDF-JS and raw "
                 "MMD$^2$ ranks over the sweep snapshots (ties to fewer steps). The held-out "
                 "set chooses nothing. The IQP arms are NOT early-stopped: they keep their "
                 "fixed 4,000 steps, so this procedure can only favour the baselines.")
    lines.append("")
    lines.append("| arm | N | steps chosen, per seed | median | candidates |")
    lines.append("|---|---|---|---|---|")
    by = grouped(rows, lambda r: (r.get("arm"), r.get("n_train")))
    for arm in ("RBM-10bit", "DCGAN-32"):
        for n in sorted({k[1] for k in by if k[0] == arm}):
            rs = sorted(by[(arm, n)], key=lambda r: (r.get("seed") is None, r.get("seed")))
            steps = [r.get("selected_steps") for r in rs]
            got = [x for x in steps if x is not None]
            med = sorted(got)[len(got) // 2] if got else None
            cand = rs[0].get("n_candidates") if rs else None
            lines.append("| %s | %s | %s | %s | %s |"
                         % (arm, n, ", ".join(str(x) for x in steps) if steps else "--",
                            med if med is not None else "--", cand))
    lines.append("")


def fixed_vs_es_section(t2_fixed, t2_stats, seeds, lines):
    """Appendix: what the fixed 20,000-step protocol gave, next to the chosen one."""
    lines.append("### Appendix -- the same baselines at a fixed 20,000 steps")
    lines.append("")
    if not t2_fixed:
        lines.append("_No fixed-step baseline rows collected yet._")
        lines.append("")
        return
    lines.append("The protocol the round-3 tables used. Kept only for comparison; the primary "
                 "table above uses the validation-chosen step count.")
    lines.append("")
    hdr = "| arm | N | " + " | ".join("%s (20k)" % nm for _k, nm in TABLE2_METRICS)
    hdr += " | " + " | ".join("%s (chosen)" % nm for _k, nm in TABLE2_METRICS) + " | seeds |"
    lines.append(hdr)
    lines.append("|" + "---|" * (3 + 2 * len(TABLE2_METRICS)))
    by = grouped(t2_fixed, lambda r: (r.get("arm"), r.get("n_train")))
    for arm in ("RBM-10bit", "DCGAN-32"):
        for n in TABLE2_N:
            rs = by.get((arm, n), [])
            if not rs:
                continue
            fixed_cells, k = [], 0
            for mk, _nm in TABLE2_METRICS:
                txt, _m, _s, kk = cell([r.get(mk) for r in rs], len(seeds))
                fixed_cells.append(txt)
                k = max(k, kk)
            chosen = []
            for mk, _nm in TABLE2_METRICS:
                st = t2_stats.get((arm, n, mk))
                chosen.append("%s +- %s" % (sig4(st[0]), sig4(st[1])) if st else "--")
            lines.append("| %s | %d | %s | %s | %d |"
                         % (arm, n, " | ".join(fixed_cells), " | ".join(chosen), k))
    lines.append("")


def theta0_section(repo, seeds, lines):
    """Does the hardware-compatible arm need its KL warm-start?"""
    rows = read_jsonl(os.path.join(repo, "results_seeds_theta0.jsonl"))
    lines.append("## theta0 warm-start ablation (M = 4, flat MMD)")
    lines.append("")
    if not rows:
        lines.append("_results_seeds_theta0.jsonl is empty -- the ablation has not run._")
        lines.append("")
        return
    lines.append("The published M = 4 flat-MMD arm is called hardware-compatible because its "
                 "loss is estimable from shots, but its theta0 warm-start minimises KL, which "
                 "is not. The second row replaces that warm-start with the flat-MMD loss "
                 "against the same mean crop, making every stage shot-estimable; nothing else "
                 "changes (`seeds/train_iqp_affine_r3_theta0obj.py`, verified to reproduce the "
                 "original bit-for-bit at `--theta0-objective kl`).")
    lines.append("")
    lines.append("| theta0 warm-start | N | " + " | ".join(nm for _k, nm in CLASS_METRICS)
                 + " | seeds |")
    lines.append("|" + "---|" * (3 + len(CLASS_METRICS)))
    pool = read_jsonl(os.path.join(repo, "results_seeds_table2.jsonl")) + rows
    by = grouped(pool, lambda r: (r.get("arm"), r.get("n_train")))
    stats = {}
    for lab, arm in THETA0_ARMS:
        for n in TABLE2_N:
            rs = by.get((arm, n), [])
            if not rs:
                continue
            cells, k = [], 0
            for mk, _nm in CLASS_METRICS:
                txt, m, sd, kk = cell([r.get(mk) for r in rs], len(seeds))
                cells.append(txt)
                k = max(k, kk)
                if m is not None:
                    stats[(arm, n, mk)] = (m, sd or 0.0, kk)
            lines.append("| %s | %d | %s | %d |" % (lab, n, " | ".join(cells), k))
    lines.append("")
    overlap_report(stats, THETA0_ARMS, TABLE2_N, CLASS_METRICS, lines,
                   "theta0 warm-start")


def determinism_section(lines):
    """Same configuration, same seed, run twice -- how far apart do the scores land?

    This is not a designed experiment; it falls out of the fact that a few configurations were
    trained twice under different job names (round 3's baselines and the convergence sweep's
    20k snapshot; round 3b's M = 4 arm and the scaling sweep's N = 100 cell).  Those pairs are
    a direct measurement of framework non-determinism, which matters because it sets a floor
    on how small a difference between two arms can mean anything at all.
    """
    lines.append("## Repeat runs of an identical configuration")
    lines.append("")
    if not REPEAT_RUNS:
        lines.append("Every configuration that was trained twice at the same seed produced "
                     "bit-identical scores.")
        lines.append("")
        return
    lines.append("Each row is one configuration trained twice at the SAME seed under two job "
                 "names. A non-zero gap is framework non-determinism, not seed variance, and "
                 "it bounds how small a between-arm difference can be meaningful.")
    lines.append("")
    lines.append("| arm | N | seed | metric | run A | run B | relative gap |")
    lines.append("|---|---|---|---|---|---|---|")
    arms_hit = set()
    # the same pair is seen once per table list it feeds, so show each measurement once
    uniq, seen_pairs = [], set()
    for rec in REPEAT_RUNS:
        key = (rec[0], rec[1], rec[2], rec[3], round(rec[4], 12), round(rec[5], 12))
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        uniq.append(rec)
    for arm, n, sd, m, a, b in sorted(uniq, key=lambda t: (str(t[0]), t[1] or 0, str(t[3]))):
        denom = max(abs(a), abs(b), 1e-12)
        lines.append("| %s | %s | %s | %s | %s | %s | %.1f%% |"
                     % (arm, n, sd, m, sig4(a), sig4(b), 100.0 * abs(a - b) / denom))
        arms_hit.add(arm)
    lines.append("")
    lines.append("Only %s differ; every other repeated configuration reproduced exactly. "
                 "cuDNN has no deterministic transposed-convolution kernel, so the DCGAN "
                 "cannot be reproduced bit-for-bit on this hardware, while the RBM and every "
                 "IQP arm can." % ", ".join(sorted(arms_hit)))
    lines.append("")


def protocol_notes(lines):
    """Facts the appendix has to state, recorded here so they cannot be lost."""
    lines.append("## Protocol notes")
    lines.append("")
    lines.append("- **Early stopping is asymmetric, in the baselines' favour.** RBM and DCGAN "
                 "get a step count chosen on a validation set disjoint from the held-out set. "
                 "The IQP arms do not: they run a fixed 4,000 steps at every N and every seed.")
    lines.append("- **DCGAN is not reproducible on this hardware, and the size of that is "
                 "measured above.** cuDNN has no deterministic transposed-convolution kernel. "
                 "The \"Repeat runs of an identical configuration\" table is not a designed "
                 "experiment -- a few configurations happened to be trained twice under "
                 "different job names -- but it puts a number on the problem: the same DCGAN "
                 "configuration at the same seed lands 3-57% apart on PDF-JS and MMD$^2$, "
                 "which is the same order as the gap between DCGAN and the IQP arms. A DCGAN "
                 "number is therefore a draw from a distribution even before the seed is "
                 "changed.")
    lines.append("- **Nothing else in the study shows that behaviour.** Every RBM and every "
                 "IQP configuration that was trained twice reproduced exactly, so their seed "
                 "spreads are genuine seed variance rather than hardware noise.")
    lines.append("- **The seed does not move the data.** The training subset, the latent pool "
                 "and the 200 held-out images are fixed at `R3_SPLIT_SEED` regardless of an "
                 "arm's `--seed` (`data_lens_r3.py:264-267`), asserted before every sweep.")
    lines.append("")


V11_RESULTS = "results_v11_all.jsonl"
V11_METRICS = [("pdf_js", "PDF-JS"), ("conv_mmd2", "MMD2"),
               ("conv_mmd2u", "MMD2_u"), ("conv_mmd2_aligned", "MMD2_al")]


def es_run_map(repo):
    """run_dir -> the authoritative (arm, kind, seed, steps) for every baseline run.

    evaluate_v11 copies each run's own `meta["arm"]`, and a sweep snapshot's meta carries the
    snapshot tag ("RBM-10bit s080k").  Matching on the plain arm name therefore drops every
    early-stopped baseline AND lets the old fixed-20k round-3 run dirs -- which are untagged
    and sit in outputs_lens_r3 -- take their place in the table.  select_checkpoint.py already
    resolved the tag when it wrote results_seeds_baselines_es.jsonl, so that file is the
    authority on what each baseline run dir actually is.
    """
    out = {}
    for r in read_jsonl(os.path.join(repo, ES_RESULTS)):
        rd = r.get("run_dir")
        if rd and r.get("kind") in ("baseline_early_stopped", "baseline_fixed_steps"):
            k = os.path.abspath(rd)
            # ONE run directory can serve both roles: when the validation set happens to pick
            # 20,000 steps, the selected checkpoint IS the fixed-20k reference.  Keeping only
            # the last kind silently drops that seed from the primary table (measured: DCGAN
            # at N = 10 fell to 3 of 5 seeds), so both roles are recorded.
            e = out.setdefault(k, {"arm": r.get("arm"), "seed": r.get("seed"),
                                   "steps": r.get("selected_steps"), "kinds": set()})
            e["kinds"].add(r.get("kind"))
            if r.get("kind") == "baseline_early_stopped":
                e["steps"] = r.get("selected_steps")
    return out


def collect_v11(seeds, repo):
    """Rows re-scored under the v11 metric protocol (evaluate_v11.py).

    v11 fixes two things v10 got wrong: PDF-JS binned on the JOINT range of (held-out, arm),
    so every arm was measured on its own axis, and Conv2D-MMD^2 took its normalisation and
    its RBF bandwidth from the POOLED features, so every arm was measured with its own
    kernel.  v11 fixes the bins at [0, 1] and derives the scale and sigma from the held-out
    set alone, identically for every arm and seed.  Each v11 row also carries the v10 value
    as `*_legacy`, so both protocols come out of one pass over one set of generated images.

    A v11 row is keyed the same way a legacy row is, so nothing else in this file changes.
    """
    rows = read_jsonl(os.path.join(repo, V11_RESULTS))
    esmap = es_run_map(repo)
    t2, cls, scl, t2_fixed, scl_fixed = [], [], [], [], []
    want2 = {a for _l, a in TABLE2_ARMS} | {TABLE2_FLOOR}
    wantc = {a for _l, a in CLASS_ARMS}
    wants = {a for _l, a in SCALING_ARMS} | {TABLE2_FLOOR}
    dropped_snapshots = 0
    for r in rows:
        # evaluate_v11 appends its floor SUMMARY rows to the same file; they are handled by
        # the dedicated floor block below, not as arms.
        if str(r.get("kind", "")).startswith("floor"):
            continue
        arm, n, sd = r.get("arm"), r.get("n_train"), r.get("seed")
        rd = os.path.abspath(r["run_dir"]) if r.get("run_dir") else None
        kind = None
        kinds = None
        if rd is not None and rd in esmap:
            e = esmap[rd]
            arm, sd, kinds = e["arm"], e["seed"], e["kinds"]
            kind = ("baseline_early_stopped" if "baseline_early_stopped" in kinds
                    else "baseline_fixed_steps")
            r = dict(r, arm=arm, kind=kind, seed=sd, selected_steps=e["steps"])
        elif arm in BASELINE_ARMS:
            # untagged baseline arm not produced by the ES protocol == an old fixed-20k
            # round-3 / scaling run.  It belongs in the appendix, never in the primary table.
            kind = "baseline_fixed_steps"
            r = dict(r, kind=kind)
        elif " s" in str(arm) and str(arm).rstrip("k").rsplit(" s", 1)[-1].isdigit():
            # a sweep snapshot the validation set did not select: not a row of any table
            dropped_snapshots += 1
            continue
        if sd is not None and sd not in seeds:
            continue
        row = dict(r, _src=V11_RESULTS)
        if r.get("class_name"):
            if arm in wantc:
                cls.append(row)
            continue
        # a run dir that is BOTH the selection and the 20k reference belongs in both tables
        if kinds and "baseline_fixed_steps" in kinds:
            fx = dict(row, kind="baseline_fixed_steps", selected_steps=20000)
            if n in TABLE2_N:
                t2_fixed.append(fx)
            if n in SCALING_N:
                scl_fixed.append(fx)
        if kind == "baseline_fixed_steps":
            if not kinds:
                if n in TABLE2_N:
                    t2_fixed.append(row)
                if n in SCALING_N:
                    scl_fixed.append(row)
            continue
        if arm in want2 and n in TABLE2_N:
            t2.append(row)
        if arm in wants and n in SCALING_N:
            scl.append(row)
    if dropped_snapshots:
        print("[aggregate] v11: ignored %d unselected sweep snapshots" % dropped_snapshots)
    if not esmap:
        print("[aggregate] v11 WARNING: %s is empty, so the v11 baseline rows fall back to "
              "the old fixed-20k runs" % ES_RESULTS)
    for r in t2:
        if r.get("arm") == CLASS_POOLED[1]:
            cls.append(dict(r, _pooled=True))
    return (dedupe_cells(dedupe(t2), "v11 Table 2 rows"),
            dedupe_cells(dedupe(cls), "v11 Table 3 rows"),
            dedupe_cells(dedupe(scl), "v11 scaling rows"),
            dedupe_cells(dedupe(t2_fixed), "v11 fixed-20k rows"),
            dedupe_cells(dedupe(scl_fixed), "v11 fixed-20k scaling rows"))


def v11_section(repo, seeds, lines):
    """The v11 tables -- the primary ones. The legacy tables follow, for comparison."""
    n_expected = len(seeds)
    path = os.path.join(repo, V11_RESULTS)
    lines.append("# Part 1 -- v11 metric protocol (primary)")
    lines.append("")
    if not os.path.isfile(path):
        lines.append("_%s does not exist yet -- evaluate_v11.py has not been run over the "
                     "seed trees. Part 2 below is the v10 protocol and is what the current "
                     "draft quotes._" % V11_RESULTS)
        lines.append("")
        return None
    t2, cls, scl, t2_fixed, scl_fixed = collect_v11(seeds, repo)
    # the seed-band scaling figure reads this: v11 is the primary protocol, so the figure
    # must not keep plotting the v10 numbers
    for name, rws in (("results_seeds_table2_v11.jsonl", t2),
                      ("results_seeds_class_v11.jsonl", cls),
                      ("results_seeds_scaling_v11.jsonl", scl)):
        with open(os.path.join(repo, name), "w") as fh:
            for r in rws:
                fh.write(json.dumps(r) + "\n")
        print("[aggregate] wrote %d rows to %s" % (len(rws), os.path.join(repo, name)))
    if not t2:
        lines.append("_%s exists but has no rows for the requested arms and seeds._"
                     % V11_RESULTS)
        lines.append("")
        return None
    lines.append("PDF-JS on a fixed [0, 1] grid; Conv2D-MMD$^2$ with the normalisation and "
                 "the RBF bandwidth taken from the held-out set alone, so the kernel is the "
                 "same object for every arm and every seed. MMD$^2_u$ is the unbiased "
                 "estimator, which -- unlike the biased one -- does not reward an arm for "
                 "having fewer effective samples.")
    lines.append("")
    for r in t2 + scl:
        cf = r.get("gen_clip_frac")
        if cf is not None and cf > 0.001:
            lines.append("> Note: arm `%s` at N=%s seed=%s has %.2f%% of its pixels outside "
                         "[0, 1] and is clipped by the fixed grid."
                         % (r.get("arm"), r.get("n_train"), r.get("seed"), 100.0 * cf))
    lines.append("")

    lines.append("## Table 2 (v11)")
    lines.append("")
    # No split-half floor row here: v11's floor is the 200-vs-200 bootstrap below, which is
    # the like-for-like comparison (every model row is also 200 vs 200).
    tb, st2 = build_table(t2, TABLE2_ARMS, TABLE2_N, V11_METRICS, n_expected)
    lines.extend(tb)
    lines.append("")
    overlap_report(st2, TABLE2_ARMS, TABLE2_N, V11_METRICS, lines, "Table 2 (v11)")

    lines.append("## Table 3 (v11)")
    lines.append("")
    tb, st3 = build_table(cls, CLASS_ARMS + [CLASS_POOLED], TABLE2_N, V11_METRICS,
                          n_expected, label_head="Training set")
    lines.extend(tb)
    lines.append("")
    lines.append("The projection correlation of Table 3 is a plain image correlation, not one "
                 "of the two metrics v11 changes, so its values are protocol-independent and "
                 "are reported once, in Part 2 and in the restart section.")
    lines.append("")
    overlap_report(st3, CLASS_ARMS + [CLASS_POOLED], TABLE2_N, V11_METRICS, lines,
                   "Table 3 (v11)")

    lines.append("## Figure 5 (v11)")
    lines.append("")
    tb, sts = build_table(scl, SCALING_ARMS, SCALING_N, V11_METRICS, n_expected,
                          label_head="Arm")
    lines.extend(tb)
    lines.append("")
    overlap_report(sts, SCALING_ARMS, SCALING_N, V11_METRICS, lines, "Figure 5 (v11)")

    floor = [r for r in read_jsonl(os.path.join(repo, "results_v11_floor.jsonl"))
             if r.get("kind") == "floor_v11"]
    if floor:
        lines.append("### 200-vs-200 real-vs-real floor (v11)")
        lines.append("")
        lines.append("The published floor is a 100-vs-100 split of the held-out set, but every "
                     "model row is 200 vs 200, and the biased MMD$^2$ estimator is sensitive "
                     "to sample size. These rows are 200 vs 200, bootstrapped.")
        lines.append("")
        lines.append("| kind | " + " | ".join(nm for _k, nm in V11_METRICS) + " | draws |")
        lines.append("|" + "---|" * (2 + len(V11_METRICS)))
        by = grouped(floor, lambda r: r.get("class_name"))
        for k in sorted(by, key=lambda x: (x is not None, str(x))):
            rs = by[k]
            cells, kk = [], 0
            for mk, _nm in V11_METRICS:
                txt, _m, _s, n = cell([r.get(mk) for r in rs], len(rs))
                cells.append(txt)
                kk = max(kk, n)
            lines.append("| %s | %s | %d |" % (k or "pooled", " | ".join(cells), kk))
        lines.append("")
    return st2


def params_section(repo, rows, lines):
    """Parameter counts split into shared vs per-image, and the flux scale S.

    The paper's Params column is one number per arm, but the arms distribute parameters very
    differently: the fixed-basis IQP arms and the RBM both carry a per-training-image block
    that grows with N, while the DCGAN does not.  A single number therefore compares the arms
    at N = 11 and N = 100 on different footings.  Both numbers are arithmetic on each run's own
    meta.json -- nothing is recomputed and no run is re-scored.

    S is the training set's mean total flux, which every decoded image is multiplied by.  It
    was never copied into the results rows, so a table could not show that N = 11 and N = 100
    are scored at different S.
    """
    try:
        _sys_path = os.path.join(repo, "seeds")
        if _sys_path not in sys.path:
            sys.path.insert(0, _sys_path)
        from rescore_extra_metrics import param_block
    except Exception as exc:                                   # pragma: no cover
        lines.append("## Parameter counts")
        lines.append("")
        lines.append("_could not import seeds/rescore_extra_metrics.param_block: %s_" % exc)
        lines.append("")
        return

    lines.append("## Parameter counts and the flux scale S")
    lines.append("")
    lines.append("`shared` is what is reused across every image; `total at N` adds the "
                 "per-image block. The DCGAN has no per-image block, so its two columns are "
                 "equal, and its discriminator is listed separately because it is "
                 "training-only and is never sampled from.")
    lines.append("")
    lines.append("> **The current Params column mixes three conventions.** Table 2 lists "
                 "`1,024` for the two fixed-basis IQP arms (one image's parameter vector: the "
                 "974-phase core plus one image's 50 latent phases), `5,164 / 5,520` for the "
                 "learned-direction arms (shared + N x M, an honest total that grows with N), "
                 "`11,010` for the RBM (again one image's model), and `199,837` for the DCGAN "
                 "(generator only). Read as totals, the fixed-basis arms hold 1,524 parameters "
                 "at N = 11 and 5,974 at N = 100, and the RBM holds 11,510 and 15,960 -- so "
                 "the row that looks like the smallest model at N = 100 is not.")
    lines.append("")
    lines.append("| arm | N | shared | per image | total at N | discriminator | S |")
    lines.append("|---|---|---|---|---|---|---|")
    seen = set()
    by = grouped(rows, lambda r: (r.get("arm"), r.get("n_train")))
    for lab, arm in TABLE2_ARMS:
        for n in TABLE2_N:
            rs = by.get((arm, n), [])
            rd = next((r.get("run_dir") for r in rs if r.get("run_dir")), None)
            if not rd or (arm, n) in seen:
                continue
            seen.add((arm, n))
            meta_path = os.path.join(rd, "meta.json")
            if not os.path.isfile(meta_path):
                continue
            with open(meta_path) as fh:
                meta = json.load(fh)
            shared, total, extra = param_block(meta)
            if extra.get("params_discriminator") is None and "DCGAN" in lab:
                # baseline_dcgan_sweep.py does not record the discriminator size, but it
                # builds the same architecture at the same --ndf as baseline_dcgan_r3.py,
                # which does. Read it from there rather than leaving the column blank.
                ref = os.path.join(repo, "outputs_lens_r3", "dcgan_n100", "meta.json")
                if os.path.isfile(ref):
                    with open(ref) as rfh:
                        extra["params_discriminator"] = json.load(rfh).get(
                            "param_count_discriminator")
            lines.append("| %s | %d | %s | %s | %s | %s | %s |"
                         % (lab, n, shared if shared is not None else "--",
                            extra.get("params_per_image") if extra.get("params_per_image") is not None else "--",
                            total if total is not None else "--",
                            extra.get("params_discriminator") or "--",
                            sig4(meta.get("S"))))
    lines.append("")


IQP_ES_RESULTS = "results_seeds_iqp_es.jsonl"
IQP_ES_ARMS = [
    ("IQP, learned dir. M=4 (flat MMD)",     "IQP-affine-M4-MMDflat-resample"),
    ("IQP, learned dir. M=4 (paper kernel)", "IQP-affine-M4-auto3-resample"),
]
IQP_ES_METRICS = [("pdf_js", "PDF-JS"), ("conv_mmd2", "MMD2"),
                  ("conv_mmd2_aligned", "MMD2_al"), ("projection_corr_mean", "proj. corr.")]


IQP_ES_V11 = "results_v11_iqp_es.jsonl"


def iqp_es_run_map(repo):
    """run_dir -> (arm_base, kind, seed, steps) for the IQP early-stopping checkpoints.

    The v11 rows copy each snapshot's tagged arm ("... s016k") and carry no `kind`, so the
    selector's own output is what says which snapshot was chosen and which is the reference.
    """
    out = {}
    for r in read_jsonl(os.path.join(repo, IQP_ES_RESULTS)):
        rd = r.get("run_dir")
        if rd and r.get("kind") in ("iqp_early_stopped", "iqp_fixed_steps"):
            out.setdefault(os.path.abspath(rd), {"arm": r.get("arm"), "seed": r.get("seed"),
                                                 "steps": r.get("selected_steps"),
                                                 "kinds": set()})["kinds"].add(r.get("kind"))
    return out


def iqp_es_v11_rows(repo):
    """-> {kind: [rows]} for the IQP early-stopping checkpoints, scored under v11."""
    emap = iqp_es_run_map(repo)
    out = {"iqp_early_stopped": [], "iqp_fixed_steps": []}
    for r in read_jsonl(os.path.join(repo, IQP_ES_V11)):
        rd = os.path.abspath(r["run_dir"]) if r.get("run_dir") else None
        if rd is None or rd not in emap:
            continue
        e = emap[rd]
        for k in e["kinds"]:
            out[k].append(dict(r, arm=e["arm"], seed=e["seed"], selected_steps=e["steps"],
                               kind=k))
    for k in out:
        out[k] = dedupe_cells(dedupe(out[k]), "IQP ES v11 %s rows" % k)
    return out


def iqp_es_section(repo, seeds, t2_legacy, lines):
    """The IQP arms under the baselines' own validation early-stopping rule.

    Symmetry: the classical baselines pick their step count on a validation set, so leaving the
    IQP arms at a fixed 4,000 steps hands the baselines an advantage the paper never states.
    This section removes it and reports what changes.

    One subtlety decides how the table is read.  The published arms ran 4,000 steps with the
    cosine schedule set for 4,000 steps (T_max = --steps).  The snapshot taken at step 4,000 of
    a 16,000-step run is NOT the same model: it sat on a schedule that was still decaying
    toward 16,000, so its learning rate at every step differed.  Both are shown, on separate
    rows, because conflating them would silently compare two different training protocols:

      published, fixed 4,000    the paper's runs (results_seeds_table2.jsonl)
      step 4,000 of a 16,000-step schedule   the same-length point on this sweep's schedule
      validation-selected       what the baselines' rule picks

    All three are scored by the same evaluate_r3 functions against the same 200 held-out
    images, so the columns are directly comparable to each other and to Part 2's Table 2.
    """
    rows = read_jsonl(os.path.join(repo, IQP_ES_RESULTS))
    lines.append("## IQP early stopping (symmetric with the baselines)")
    lines.append("")
    if not rows:
        lines.append("_%s is empty -- seeds/submit_seeds_iqp_es.sbatch has not finished._"
                     % IQP_ES_RESULTS)
        lines.append("")
        return
    sel = dedupe_cells(dedupe([r for r in rows if r.get("kind") == "iqp_early_stopped"]),
                       "IQP early-stopped rows")
    fx16 = dedupe_cells(dedupe([r for r in rows if r.get("kind") == "iqp_fixed_steps"]),
                        "IQP step-4000-of-16000 rows")
    pub = [r for r in t2_legacy if r.get("arm") in {a for _l, a in IQP_ES_ARMS}]

    lines.append("Metrics here are the v10 protocol, because all three rows come from the same "
                 "`evaluate_r3` pass and the point of the table is the comparison between them.")
    lines.append("")
    lines.append("### Selected step counts")
    lines.append("")
    lines.append("| arm | N | steps chosen, per seed | median | candidates |")
    lines.append("|---|---|---|---|---|")
    by_sel = grouped(sel, lambda r: (r.get("arm"), r.get("n_train")))
    for lab, arm in IQP_ES_ARMS:
        for n in sorted({k[1] for k in by_sel if k[0] == arm}):
            rs = sorted(by_sel[(arm, n)], key=lambda r: (r.get("seed") is None, r.get("seed")))
            st = [r.get("selected_steps") for r in rs]
            got = [x for x in st if x is not None]
            med = sorted(got)[len(got) // 2] if got else "--"
            lines.append("| %s | %s | %s | %s | %s |"
                         % (lab, n, ", ".join(str(x) for x in st), med,
                            rs[0].get("n_candidates") if rs else "--"))
    lines.append("")

    lines.append("### Selected vs fixed")
    lines.append("")
    hdr = "| arm | N | protocol | " + " | ".join(nm for _k, nm in IQP_ES_METRICS) + " | seeds |"
    lines.append(hdr)
    lines.append("|" + "---|" * (4 + len(IQP_ES_METRICS)))
    verdicts = []
    for lab, arm in IQP_ES_ARMS:
        for n in TABLE2_N:
            variants = [
                ("published, fixed 4,000", [r for r in pub
                                            if r.get("arm") == arm and r.get("n_train") == n]),
                ("step 4,000 of a 16,000-step schedule",
                 [r for r in fx16 if r.get("arm") == arm and r.get("n_train") == n]),
                ("validation-selected", [r for r in sel
                                         if r.get("arm") == arm and r.get("n_train") == n]),
            ]
            stats = {}
            for vname, rs in variants:
                if not rs:
                    continue
                cells, k = [], 0
                for mk, _nm in IQP_ES_METRICS:
                    txt, m, sd, kk = cell([r.get(mk) for r in rs], len(seeds))
                    cells.append(txt)
                    k = max(k, kk)
                    if m is not None:
                        stats[(vname, mk)] = (m, sd or 0.0, kk)
                lines.append("| %s | %d | %s | %s | %d |" % (lab, n, vname, " | ".join(cells), k))
            for mk, nm in IQP_ES_METRICS:
                a = stats.get(("published, fixed 4,000", mk))
                b = stats.get(("validation-selected", mk))
                if not a or not b or a[2] < 2 or b[2] < 2:
                    continue
                if not overlaps((a[0], a[1]), (b[0], b[1])):
                    better = ("selected" if ((b[0] > a[0]) == (mk in HIGHER_IS_BETTER))
                              else "published")
                    verdicts.append("%s N=%d %s: %s is better, and the two do not overlap "
                                    "(%s vs %s)" % (lab, n, nm, better, sig4(a[0]), sig4(b[0])))
    lines.append("")

    # ---- the same comparison under the primary (v11) protocol
    v11 = iqp_es_v11_rows(repo)
    pub11 = [r for r in read_jsonl(os.path.join(repo, "results_seeds_table2_v11.jsonl"))
             if r.get("arm") in {a for _l, a in IQP_ES_ARMS}]
    if v11["iqp_early_stopped"] and pub11:
        lines.append("### Selected vs fixed, v11 metrics (the primary protocol)")
        lines.append("")
        lines.append("The selection itself is unchanged -- it used the same two v10 metrics on "
                     "the validation set that chose the baselines' step counts, because that "
                     "symmetry is the point. Only the reporting moves to v11.")
        lines.append("")
        m11 = [(k, nm) for k, nm in V11_METRICS]
        lines.append("| arm | N | protocol | " + " | ".join(nm for _k, nm in m11) + " | seeds |")
        lines.append("|" + "---|" * (4 + len(m11)))
        for lab, arm in IQP_ES_ARMS:
            for n in TABLE2_N:
                st11 = {}
                for vname, rs in (("published, fixed 4,000",
                                   [r for r in pub11 if r.get("arm") == arm
                                    and r.get("n_train") == n]),
                                  ("validation-selected",
                                   [r for r in v11["iqp_early_stopped"] if r.get("arm") == arm
                                    and r.get("n_train") == n])):
                    if not rs:
                        continue
                    cells, k = [], 0
                    for mk, _nm in m11:
                        txt, m, sd, kk = cell([r.get(mk) for r in rs], len(seeds))
                        cells.append(txt)
                        k = max(k, kk)
                        if m is not None:
                            st11[(vname, mk)] = (m, sd or 0.0, kk)
                    lines.append("| %s | %d | %s | %s | %d |"
                                 % (lab, n, vname, " | ".join(cells), k))
                for mk, nm in m11:
                    a = st11.get(("published, fixed 4,000", mk))
                    b = st11.get(("validation-selected", mk))
                    if not a or not b or a[2] < 2 or b[2] < 2:
                        continue
                    if not overlaps((a[0], a[1]), (b[0], b[1])):
                        better = ("selected" if ((b[0] > a[0]) == (mk in HIGHER_IS_BETTER))
                                  else "published")
                        verdicts.append("[v11] %s N=%d %s: %s is better, and the two do not "
                                        "overlap (%s vs %s)"
                                        % (lab, n, nm, better, sig4(a[0]), sig4(b[0])))
        lines.append("")

    lines.append("### Does early stopping change the IQP numbers by more than 1 sd?")
    lines.append("")
    v10 = [v for v in verdicts if not v.startswith("[v11]")]
    v11v = [v for v in verdicts if v.startswith("[v11]")]
    for proto, vs in (("v10 metrics", v10), ("v11 metrics (primary)", v11v)):
        if vs:
            lines.append("**%s** -- %d of the %d (arm, N, metric) comparisons exceed 1 sd:"
                         % (proto, len(vs), 2 * len(TABLE2_N) * len(IQP_ES_METRICS)))
            for v in vs:
                lines.append("- %s" % v.replace("[v11] ", ""))
        else:
            lines.append("**%s** -- none. Every (arm, N, metric) comparison between the "
                         "validation-selected checkpoint and the published fixed-4,000 run "
                         "overlaps within 1 sd." % proto)
        lines.append("")
    lines.append("**Reading.** Giving the IQP arms the baselines' own early-stopping rule does "
                 "not move their numbers: under the primary v11 protocol nothing separates the "
                 "two, and the selected checkpoints are, if anything, marginally better. The "
                 "asymmetry in the published protocol was therefore real but immaterial, which "
                 "is the useful thing to be able to say -- the baselines' advantage came from "
                 "being undertrained at a fixed 20,000 steps, not from being allowed to stop "
                 "early.")
    lines.append("")
    lines.append("The step counts the validation set picks for the IQP arms scatter across the "
                 "whole 1k-16k range with no stable pattern (unlike the baselines, which agree "
                 "seed-to-seed on 80k-160k for the RBM and 5k for the DCGAN). That is itself "
                 "the finding: the IQP validation curve is flat, so there is no well-defined "
                 "stopping point to find, and 4,000 steps was already on the plateau.")
    lines.append("")
    lines.append("_The `step 4,000 of a 16,000-step schedule` rows carry no projection "
                 "correlation: that column is written only for the selected checkpoint._")
    lines.append("")

    # The question this whole section exists to answer: once BOTH sides are early-stopped,
    # does Table 2's conclusion change?  Computed, not asserted.
    t2v11 = read_jsonl(os.path.join(repo, "results_seeds_table2_v11.jsonl"))
    lines.append("### With both sides early-stopped, who wins Table 2?")
    lines.append("")
    lines.append("| N | metric | best IQP (selected) | best baseline | winner |")
    lines.append("|---|---|---|---|---|")
    for n in TABLE2_N:
        for mk, nm in V11_METRICS:
            iqp = []
            for lab, arm in IQP_ES_ARMS:
                rs = [r for r in v11["iqp_early_stopped"]
                      if r.get("arm") == arm and r.get("n_train") == n]
                m, sd, k = mean_sd([r.get(mk) for r in rs])
                if m is not None and k >= 2:
                    iqp.append((lab, m, sd))
            base = []
            for lab, arm in (("RBM", "RBM-10bit"), ("DCGAN", "DCGAN-32")):
                rs = [r for r in t2v11 if r.get("arm") == arm and r.get("n_train") == n]
                m, sd, k = mean_sd([r.get(mk) for r in rs])
                if m is not None and k >= 2:
                    base.append((lab, m, sd))
            if not iqp or not base:
                continue
            hib = mk in HIGHER_IS_BETTER
            bi = (max if hib else min)(iqp, key=lambda t: t[1])
            bb = (max if hib else min)(base, key=lambda t: t[1])
            if overlaps((bi[1], bi[2]), (bb[1], bb[2])):
                who = "tie within 1 sd"
            elif (bi[1] > bb[1]) == hib:
                who = "**IQP**"
            else:
                who = "**%s**" % bb[0]
            lines.append("| %d | %s | %s (%s +- %s) | %s (%s +- %s) | %s |"
                         % (n, nm, bi[0], sig4(bi[1]), sig4(bi[2]),
                            bb[0], sig4(bb[1]), sig4(bb[2]), who))
    lines.append("")


def diag_section(repo, seeds, lines):
    rows = read_jsonl(os.path.join(repo, "results_seeds_diag.jsonl"))
    lines.append("## Section 3.2 diagnostics")
    lines.append("")
    if not rows:
        lines.append("_results_seeds_diag.jsonl is empty -- the diagnostics have not run._")
        lines.append("")
        return
    a = [r for r in rows if r.get("diag") == "single_crop_fit"]
    if a:
        lines.append("**Single-crop full-parameter fit** (paper: flat MMD 0.99997, "
                     "paper kernel 0.9962). Higher is better.")
        lines.append("")
        lines.append("| objective | corr | seeds |")
        lines.append("|---|---|---|")
        for obj in ("KL", "MMD-flat", "MMD-auto3"):
            vals = [r.get("corr") for r in a if r.get("objective") == obj]
            txt, _m, _s, k = cell(vals, len(seeds))
            lines.append("| %s | %s | %d |" % (obj, txt, k))
        lines.append("")
        s0 = {r.get("objective"): r.get("corr") for r in a if r.get("seed") == 0}
        if s0:
            ok_flat = s0.get("MMD-flat") is not None and abs(s0["MMD-flat"] - 0.999972091339941) < 1e-9
            ok_auto = s0.get("MMD-auto3") is not None and abs(s0["MMD-auto3"] - 0.9962141508256679) < 1e-9
            ok_kl = s0.get("KL") is not None and abs(s0["KL"] - 0.9999058739771814) < 1e-9
            lines.append("Seed-0 reproduction check against `mech_tests/mech_test/results.json`: "
                         "KL %s (%s), MMD-flat %s (%s), MMD-auto3 %s (%s)."
                         % (sig4(s0.get("KL")), "exact" if ok_kl else "**MISMATCH**",
                            sig4(s0.get("MMD-flat")), "exact" if ok_flat else "**MISMATCH**",
                            sig4(s0.get("MMD-auto3")), "exact" if ok_auto else "**MISMATCH**"))
            lines.append("")
    b = [r for r in rows if r.get("diag") == "frozen_core_recon"]
    if b:
        lines.append("**Frozen-core reconstruction of the other training images** "
                     "(paper: 0.47). Higher is better; the round-3 codec's counterpart of the "
                     "round-2 number the paper cites.")
        lines.append("")
        lines.append("| N | anchor corr | other images, mean corr | seeds |")
        lines.append("|---|---|---|---|")
        for n in sorted({r.get("n_train") for r in b}):
            rs = [r for r in b if r.get("n_train") == n]
            ta, _m, _s, _k = cell([r.get("corr_anchor") for r in rs], len(seeds))
            to, _m, _s, k = cell([r.get("corr_others_mean") for r in rs], len(seeds))
            lines.append("| %s | %s | %s | %d |" % (n, ta, to, k))
        lines.append("")


# ------------------------------------------------------------------------------------ main
def main(argv=None):
    args = parse_args(argv)
    seeds = sorted(set(args.seeds))
    n_expected = len(seeds)
    repo = args.repo

    t2, cls, scl, t2_fixed, scl_fixed = collect(seeds, repo)
    t2 = dedupe_cells(dedupe(t2), "Table 2 rows")
    cls = dedupe_cells(dedupe(cls), "Table 3 rows")
    scl = dedupe_cells(dedupe(scl), "scaling rows")
    t2_fixed = dedupe_cells(dedupe(t2_fixed), "fixed-20k rows")
    scl_fixed = dedupe_cells(dedupe(scl_fixed), "fixed-20k scaling rows")
    check_seed_counts(t2_fixed, "fixed-20k Table 2", n_expected)
    check_seed_counts([r for r in t2 if r.get('arm') != TABLE2_FLOOR],
                      "Table 2", n_expected)
    check_seed_counts(cls, "Table 3", n_expected)
    for name, rows in (("results_seeds_table2.jsonl", t2),
                       ("results_seeds_class.jsonl", cls),
                       ("results_seeds_scaling.jsonl", scl),
                       ("results_seeds_table2_fixed20k.jsonl", t2_fixed),
                       ("results_seeds_scaling_fixed20k.jsonl", scl_fixed)):
        path = os.path.join(repo, name)
        with open(path, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        print("[aggregate] wrote %d rows to %s" % (len(rows), path))

    L = []
    L.append("# Seed statistics for the ML4PS-2026 tables")
    L.append("")
    L.append("Every number is the mean +- sample standard deviation (ddof = 1) over seeds "
             "%s, to four significant figures. A `*` marks a cell built from fewer seeds "
             "than the %d requested." % (", ".join(str(s) for s in seeds), n_expected))
    L.append("")
    L.append("The seed changes model initialisation, mini-batch and restart draws, the latent "
             "prior draw, the generation draw, and the auto3 kernel's Monte-Carlo bandwidth "
             "estimate. It does NOT change the data: the training subset, the latent pool and "
             "the 200 held-out images are fixed at `R3_SPLIT_SEED` "
             "(`data_lens_r3.py:264-267`), and evaluation uses the same `CONV_SEED = 0` and "
             "the same evaluator seed for every arm.")
    L.append("")
    L.append("Baselines (RBM, DCGAN) use validation-based early stopping; the IQP arms do "
             "not and run a fixed 4,000 steps. See Protocol notes at the end.")
    L.append("")
    L.append("Generated by `aggregate_seeds.py`. `main.tex` is not modified by it.")
    L.append("")
    ok = check_data_invariance(t2 + cls + scl, L)

    v11_section(repo, seeds, L)

    L.append("# Part 2 -- v10 metric protocol (legacy, what the current draft quotes)")
    L.append("")
    L.append("## Table 2 -- paper-style comparison")
    L.append("")
    lines, st2 = build_table(t2, TABLE2_ARMS + [("Held-out split-half (floor)", TABLE2_FLOOR)],
                             TABLE2_N, TABLE2_METRICS, n_expected)
    L.extend(lines)
    L.append("")
    overlap_report(st2, TABLE2_ARMS, TABLE2_N, TABLE2_METRICS, L, "Table 2")
    fixed_vs_es_section(t2_fixed, st2, seeds, L)

    L.append("## Table 3 -- per-class generation")
    L.append("")
    lines, st3 = build_table(cls, CLASS_ARMS + [CLASS_POOLED], TABLE2_N, CLASS_METRICS,
                             n_expected, label_head="Training set")
    L.extend(lines)
    L.append("")
    overlap_report(st3, CLASS_ARMS + [CLASS_POOLED], TABLE2_N, CLASS_METRICS, L, "Table 3")

    L.append("## Figure 5 -- data scaling")
    L.append("")
    lines, sts = build_table(scl, SCALING_ARMS, SCALING_N, SCALING_METRICS, n_expected,
                             label_head="Arm")
    L.extend(lines)
    L.append("")
    overlap_report(sts, SCALING_ARMS, SCALING_N, SCALING_METRICS, L, "Figure 5")

    params_section(repo, t2, L)
    selected_steps_section(repo, seeds, L)
    iqp_es_section(repo, seeds, t2, L)
    theta0_section(repo, seeds, L)
    diag_section(repo, seeds, L)
    restarts_section(repo, seeds, L)
    determinism_section(L)
    protocol_notes(L)

    L.append("## Provenance")
    L.append("")
    srcs = {r.get("_src") for r in (t2 + cls + scl + t2_fixed + scl_fixed) if r.get("_src")}
    # `_src` only tags the legacy rows, so list every file this script actually reads --
    # otherwise a reader tracing a Part 1 number back to source finds nothing to open.
    for extra_src in (V11_RESULTS, "results_v11_floor.jsonl", ES_RESULTS,
                      "results_seeds_table2.jsonl", "results_seeds_class.jsonl",
                      "results_seeds_scaling.jsonl", "results_seeds_theta0.jsonl",
                      "results_seeds_restarts.jsonl", "results_seeds_diag.jsonl"):
        if os.path.isfile(os.path.join(repo, extra_src)):
            srcs.add(extra_src)
    for s in sorted(srcs):
        L.append("- `%s`" % s)
    L.append("")

    with open(args.out, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("[aggregate] wrote %s (%d lines)" % (args.out, len(L)))
    if not ok:
        print("[aggregate] FATAL: the held-out set is not identical across seeds")
        return 1

    # a short console summary so the sbatch log carries the headline numbers
    print("\n--- Table 2 (mean +- sd) ---")
    for lab, arm in TABLE2_ARMS:
        for n in TABLE2_N:
            got = [sig4(st2[(arm, n, mk)][0]) + " +- " + sig4(st2[(arm, n, mk)][1])
                   if (arm, n, mk) in st2 else "--" for mk, _ in TABLE2_METRICS]
            k = max([st2[(arm, n, mk)][2] for mk, _ in TABLE2_METRICS if (arm, n, mk) in st2]
                    or [0])
            print("  %-40s N=%3d  %s  [%d seeds]" % (lab, n, "  ".join(got), k))
    return 0


if __name__ == "__main__":
    sys.exit(main())
