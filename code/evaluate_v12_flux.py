#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Re-score already-trained runs under a DIFFERENT total-intensity (flux) convention.

WHY THIS NEEDS NO RETRAINING
---------------------------------------------------------------------------------------
Every generative arm in this study emits a normalised 32x32 pmf q(x, y) and then multiplies
it by ONE scalar S -- the mean total intensity of the N training crops -- before pasting it
into a black 64x64 frame:

    crop_i = q_i * S            (saved verbatim as <run>/crops/generated_%03d.npy)
    frame_i = paste(crop_i, dy_i, dx_i)      (saved as <run>/generated_%03d.npy)

The pmf and the paste offset are therefore recoverable EXACTLY from the saved artefacts
(q_i = crop_i / crop_i.sum(), offsets from meta["gen_offsets"]), so swapping S for a
per-image S_i is a post-hoc decode step, not a different model.  Part E measured what that
swap is worth by retraining with `--flux resample`; this file does the same swap on runs that
were trained with the fixed S, so every arm in the paper can be quoted under BOTH conventions
without spending a GPU-hour.  `--flux fixed` reproduces the published frames bit-for-bit,
which is asserted (not assumed) by `--check`.

THE TWO CONVENTIONS
---------------------------------------------------------------------------------------
    fixed      S_i = mean(training crop totals) for every i.  What Table 2 (v11) reports.
    resample   S_i ~ KDE(training crop totals), i.e. pick one of the N training totals
               uniformly with replacement and add Gaussian noise of Silverman bandwidth
                   h = 0.9 * min(sd, IQR / 1.34) * N ** (-1/5).
               Draws are clipped strictly positive.

               Part E's trainer used a fixed jitter of 0.2 x the median nearest-neighbour
               spacing instead.  In 1-D with N = 11 that spacing is small, so the sampled
               flux distribution was 11 near-delta spikes rather than a smooth density;
               Silverman's rule scales the bandwidth with the spread of the whole sample and
               removes that artefact.  Both are reported by --flux-rule so the choice is
               visible rather than buried.

The training totals are NOT read out of meta.json (it stores only their mean, `S`).  They are
recomputed from `data_lens_r3.load_r3(n_train)`, whose split is frozen at R3_SPLIT_SEED = 0
and is independent of any arm's seed, and the recomputed mean is asserted against meta["S"]
to 1e-6 for every run.  Held-out totals are never touched.

The metrics, the held-out set, the flux-alignment target and the Conv2D encoder seed are
evaluate_v11.py's, imported rather than reimplemented.

CLI
---------------------------------------------------------------------------------------
    python3 evaluate_v12_flux.py --results results_v12_fluxres.jsonl
    python3 evaluate_v12_flux.py --groups table2 msweep --check --limit 20

Python 3.9 compatible.  CPU only.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import OrderedDict

import numpy as np
import torch

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import codec10                                            # noqa: E402
import data_lens_r3 as D                                  # noqa: E402
import evaluate_r3                                        # noqa: E402
import evaluate_v11 as EV                                 # noqa: E402
import metrics_v11 as M11                                 # noqa: E402
from evaluate_lens import nn_l2                           # noqa: E402

CLASSES = EV.CLASSES
FLUX_MODES = ("fixed", "resample")

# ---- the jsonl indexes this file reads to learn WHICH runs the paper quotes -------------
IDX_T2 = os.path.join(REPO, "results_seeds_table2_v11.jsonl")
IDX_SCALE = os.path.join(REPO, "results_seeds_scaling_v11.jsonl")
IDX_CLS = os.path.join(REPO, "results_seeds_class_v11.jsonl")
IDX_SUITE = os.path.join(REPO, "results_seeds_mlp_suite.jsonl")
IDX_SUITE_ES = os.path.join(REPO, "results_seeds_mlp_suite_es.jsonl")
IDX_V12 = os.path.join(REPO, "results_seeds_v12.jsonl")

SCALING_NS = (10, 30, 50, 70, 90, 100)
MSWEEP_MS = (1, 2, 3, 4, 6, 8, 16)


# ======================================================================================
# flux samplers
# ======================================================================================
def silverman_bandwidth(x):
    """h = 0.9 * min(sd, IQR / 1.34) * n ** (-1/5), the standard rule-of-thumb.

    Falls back to the sd alone when the IQR is 0 (possible for a very small, tied sample) and
    to 0 when the sample has no spread at all, in which case the KDE degenerates to the
    empirical distribution -- which is the honest answer, not an error.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    n = x.size
    if n < 2:
        return 0.0
    sd = float(x.std(ddof=1))
    iqr = float(np.subtract(*np.percentile(x, [75, 25])))
    scale = min(sd, iqr / 1.34) if iqr > 0 else sd
    if not np.isfinite(scale) or scale <= 0:
        scale = sd
    return 0.9 * float(scale) * (float(n) ** (-0.2))


def sample_flux_kde(totals, n_draw, seed, rule="silverman"):
    """n_draw per-image flux scales from a Gaussian KDE of the N training crop totals.

    The RNG is seeded with the RUN's own seed, so a given (arm, N, seed) always draws the same
    fluxes and the fixed-vs-resampled comparison is paired within a run.
    """
    x = np.asarray(totals, dtype=np.float64).reshape(-1)
    if x.size == 0:
        raise ValueError("sample_flux_kde: no training totals")
    if rule == "silverman":
        h = silverman_bandwidth(x)
    elif rule == "nnspacing":                    # part E's trainer rule, for comparison only
        if x.size < 2:
            h = 0.0
        else:
            xs = np.sort(x)
            gaps = np.diff(xs)
            nn = np.minimum(np.concatenate(([np.inf], gaps)),
                            np.concatenate((gaps, [np.inf])))
            h = 0.2 * float(np.median(nn))
    else:
        raise ValueError("unknown flux rule %r" % rule)
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, x.size, size=int(n_draw))
    draws = x[idx] + rng.standard_normal(size=int(n_draw)) * h
    return np.maximum(draws, 1e-6), float(h)


# ======================================================================================
# rebuilding the frames
# ======================================================================================
_CROP_RE = re.compile(r"generated_(\d+)\.npy$")


def load_crops(run_dir):
    """(n, 32, 32) float64 of <run>/crops/, ordered by index -- the same order the frames and
    meta["gen_offsets"] use."""
    cdir = os.path.join(run_dir, "crops")
    if not os.path.isdir(cdir):
        return None
    found = []
    for name in os.listdir(cdir):
        m = _CROP_RE.match(name)
        if m:
            found.append((int(m.group(1)), os.path.join(cdir, name)))
    if not found:
        return None
    found.sort()
    return np.stack([np.load(p) for _, p in found], axis=0).astype(np.float64)


def frames_from_crops(crops, offsets, S_gen):
    """q_i = crop_i / sum(crop_i); frame_i = paste(q_i * S_i, dy_i, dx_i).

    float32 at the end, because that is what the trainers saved and what the evaluator has
    always consumed; the renormalise-and-rescale happens in float64.
    """
    n = crops.shape[0]
    if offsets.shape[0] != n or S_gen.shape[0] != n:
        raise ValueError("frames_from_crops: %d crops, %d offsets, %d fluxes"
                         % (n, offsets.shape[0], S_gen.shape[0]))
    out = np.empty((n, codec10.FRAME, codec10.FRAME), dtype=np.float32)
    for i in range(n):
        tot = float(crops[i].sum())
        if not np.isfinite(tot) or tot <= 0:
            raise ValueError("frames_from_crops: crop %d has total %r" % (i, tot))
        crop = (crops[i] / tot) * float(S_gen[i])
        out[i] = codec10.paste(crop.astype(np.float32), int(offsets[i, 0]), int(offsets[i, 1]))
    return out


# ======================================================================================
# the training totals of a run (recomputed, then asserted against meta["S"])
# ======================================================================================
_TOTALS = {}


def train_totals_for(n_train, class_name):
    """The N per-crop training totals.  Cached per (N, class).

    data_lens_class.install() monkey-patches data_lens_r3 one-way, so this MUST be called in
    the same class order evaluate_v11.build_context uses (pooled first, then each class).
    """
    key = (int(n_train), class_name)
    if key not in _TOTALS:
        d = D.load_r3(int(n_train), n_holdout=1)
        _TOTALS[key] = np.asarray(d.train_totals, dtype=np.float64).copy()
    return _TOTALS[key]


# ======================================================================================
# scoring one run under one flux convention
# ======================================================================================
def score_frames(gen, run_dir, meta, ctx, spec, flux_mode, extra):
    gen_al = evaluate_r3.align_to_target(gen, ctx.target)
    js = M11.pdf_js_v11(ctx.holdout, gen)
    js_al = M11.pdf_js_v11(ctx.holdout_aligned, gen_al, lo=ctx.prep_al.lo, hi=ctx.prep_al.hi)
    mm = M11.conv_mmd2_v11(ctx.holdout, gen, prepared=ctx.prep_raw)
    mm_al = M11.conv_mmd2_v11(ctx.holdout_aligned, gen_al, prepared=ctx.prep_al)
    row = {
        "run_dir": run_dir,
        "arm": spec["arm"],
        "arm_base": meta.get("arm_base", meta.get("arm")),
        "group": spec["group"],
        "family": spec["family"],
        "n_duplicate_dirs": spec.get("n_duplicate_dirs", 0),
        "script": meta.get("script"),
        "param_count": meta.get("param_count"),
        "seed": spec["seed"],
        "n_train": int(spec["n_train"]),
        "n_gen": int(gen.shape[0]),
        "n_holdout": int(ctx.holdout.shape[0]),
        "class": spec.get("class_name"),
        "class_name": spec.get("class_name"),
        "classes_pooled": spec.get("class_name") is None,
        "selected_steps": meta.get("steps"),
        "objective": meta.get("objective"),
        "protocol": meta.get("protocol"),
        "metric_protocol": "v11",
        "flux_mode": flux_mode,
        # ---- the four metric columns ----
        "pdf_js": js["pdf_js"],
        "pdf_js_aligned": js_al["pdf_js"],
        "conv_mmd2": mm["conv_mmd2"],
        "conv_mmd2u": mm["conv_mmd2u"],
        "conv_mmd2_aligned": mm_al["conv_mmd2"],
        "conv_mmd2u_aligned": mm_al["conv_mmd2u"],
        # ---- provenance / honesty ----
        "proj_corr": meta.get("projection_corr_mean"),
        "gen_min": js["gen_min"],
        "gen_max": js["gen_max"],
        "gen_clip_frac": js["gen_clip_frac"],
        "gen_clip_frac_aligned": js_al["gen_clip_frac"],
        "gen_mean_total": float(gen.reshape(gen.shape[0], -1).sum(axis=1).mean()),
        "gen_total_sd": float(gen.reshape(gen.shape[0], -1).sum(axis=1).std(ddof=1)),
        "holdout_mean_total": ctx.target,
        "nn_holdout": nn_l2(gen, ctx.holdout),
        "sigma": ctx.prep_raw.sigma,
        "sigma_aligned": ctx.prep_al.sigma,
        "conv_seed": M11.CONV_SEED,
    }
    row.update(extra)
    return row


# ======================================================================================
# the run inventory: which runs the v12 tables quote, read off the published jsonls
# ======================================================================================
def read_jsonl(path):
    if not os.path.isfile(path):
        raise SystemExit("evaluate_v12_flux: missing index %s" % path)
    return [json.loads(l) for l in open(path) if l.strip()]


MLP_ARM_RE = re.compile(r"^IQP-mlpH(\d+)-M(\d+)-(MMDflat|auto3)-resample$")


def build_inventory(groups):
    """[{run_dir, arm, family, group, n_train, seed, class_name}], de-duplicated by run_dir.

    Every entry is taken from a jsonl the paper already quotes, so this file cannot invent a
    run that no published table stands on.  The suite's `sweep/` and `phase2/` trees contain
    BIT-IDENTICAL re-runs of the same (arm, N, seed) -- deduplicating on (family, arm, N,
    seed, class) keeps one of each, which is what a per-seed statistic needs.
    """
    want = set(groups)
    items = []

    def add(r, family, group, arm=None):
        items.append({"run_dir": os.path.normpath(r["run_dir"]),
                      "arm": arm or r["arm"],
                      "family": family,
                      "group": group,
                      "n_train": int(r["n_train"]),
                      "seed": int(r["seed"]),
                      "class_name": r.get("class_name")})

    # ---- Table 2 (v11): the four IQP arms + RBM.  DCGAN learns its own per-image flux, so
    # there is no fixed S to swap and it is deliberately absent.
    if "table2" in want:
        keep = {"IQP-lowvar-paper-MMD", "IQP-lowvar-joint-MMD-resample",
                "IQP-affine-M4-MMDflat-resample", "IQP-affine-M4-auto3-resample", "RBM-10bit"}
        for r in read_jsonl(IDX_T2):
            if r["arm"] in keep and int(r["n_train"]) in (11, 100) and r.get("class_name") is None:
                add(r, "v11", "table2")

    suite = read_jsonl(IDX_SUITE) if want & {"table2_mlp", "msweep", "scaling", "perclass"} else []

    # ---- part F's Table 2 targets: the MLP M=2 H=32 arm under both objectives
    if "table2_mlp" in want:
        for r in suite:
            if (r["arm"] in ("IQP-mlpH32-M2-MMDflat-resample", "IQP-mlpH32-M2-auto3-resample")
                    and int(r["n_train"]) in (11, 100) and r.get("class_name") is None):
                add(r, "mlp", "table2_mlp")
        # the validation-selected checkpoint out of {1k, 2k, 4k, 8k, 16k}
        for r in read_jsonl(IDX_SUITE_ES):
            if (r.get("kind") == "mlpsuite_early_stopped"
                    and r["arm"] in ("IQP-mlpH32-M2-MMDflat-resample",
                                     "IQP-mlpH32-M2-auto3-resample")
                    and int(r["n_train"]) in (11, 100)):
                add(r, "mlp", "table2_mlp_selected", arm=r["arm"] + " selected")

    # ---- the M sweep at H = 32
    if "msweep" in want:
        for r in suite:
            m = MLP_ARM_RE.match(str(r["arm"]))
            if (m and int(m.group(1)) == 32 and int(m.group(2)) in MSWEEP_MS
                    and m.group(3) == "MMDflat"
                    and int(r["n_train"]) in (11, 100) and r.get("class_name") is None):
                add(r, "mlp", "msweep")

    # ---- data scaling: MLP M=2 flat, affine M=4 flat, RBM
    if "scaling" in want:
        for r in suite:
            if (r["arm"] == "IQP-mlpH32-M2-MMDflat-resample"
                    and int(r["n_train"]) in SCALING_NS and r.get("class_name") is None):
                add(r, "mlp", "scaling")
        for r in read_jsonl(IDX_SCALE):
            if (r["arm"] in ("IQP-affine-M4-MMDflat-resample", "RBM-10bit")
                    and int(r["n_train"]) in SCALING_NS and r.get("class_name") is None):
                add(r, "v11", "scaling")

    # ---- per class
    if "perclass" in want:
        for r in suite:
            if (r["arm"] == "IQP-mlpH32-M2-MMDflat-resample"
                    and int(r["n_train"]) in (11, 100) and r.get("class_name") in CLASSES):
                add(r, "mlp", "perclass")
        for r in read_jsonl(IDX_CLS):
            if r.get("class_name") in CLASSES and int(r["n_train"]) in (11, 100):
                add(r, "v11", "perclass")

    # ---- validation against part E: the v12 arms that TRAINED with the fixed S and have a
    # twin that trained with `--flux resample`.  Re-fluxing the fixed-S twin here must land on
    # the trained twin's numbers (up to the RNG stream and the bandwidth rule), which is the
    # only end-to-end check that this file's post-hoc swap is the same operation part E paid
    # GPU time for.
    if "validate_e" in want:
        for r in read_jsonl(IDX_V12):
            arm = str(r.get("arm") or "")
            if (arm.startswith("IQP-v12") and "fluxres" not in arm and " s" not in arm
                    and int(r["n_train"]) == 11):
                add(r, "v12", "validate_e")

    # De-duplicate on (family, arm, N, seed, class), NOT on run_dir.  Two things collide here
    # and they need different treatment:
    #   * one directory quoted by two tables (the N = 100 scaling point is also a Table 2
    #     cell) -- keep it once, merge the group tags;
    #   * two DIRECTORIES holding the same (arm, N, seed) because the suite re-ran the same
    #     configuration in phase 1 (`sweep/`) and phase 2 (`phase2/`).  Those runs are
    #     bit-identical (the trainer is deterministic per seed), so counting both would halve
    #     the apparent seed sd of exactly the arm this paper leans on.  Keep one, and report
    #     how many were dropped so the choice is visible.
    out = OrderedDict()
    dropped = []
    for it in sorted(items, key=lambda z: (z["family"], z["arm"], z["n_train"], z["seed"],
                                           str(z["class_name"]), z["run_dir"])):
        key = (it["family"], it["arm"], it["n_train"], it["seed"], it["class_name"])
        if key in out:
            if it["group"] not in out[key]["groups"]:
                out[key]["groups"].append(it["group"])
            if it["run_dir"] != out[key]["run_dir"]:
                dropped.append(it["run_dir"])
                out[key]["dup_run_dirs"].append(it["run_dir"])
            continue
        it = dict(it)
        it["groups"] = [it.pop("group")]
        it["dup_run_dirs"] = []
        out[key] = it
    res = []
    for it in out.values():
        it["group"] = "+".join(sorted(it["groups"]))
        it["n_duplicate_dirs"] = len(it["dup_run_dirs"])
        res.append(it)
    if dropped:
        print("[flux] de-duplicated %d re-run directory/-ies with an (arm, N, seed) already "
              "present (bit-identical re-runs, e.g. %s)" % (len(dropped), dropped[0]))
    return res


# ======================================================================================
def main(argv=None):
    p = argparse.ArgumentParser(description="re-score saved runs under a per-image flux")
    p.add_argument("--results", default=os.path.join(REPO, "results_v12_fluxres.jsonl"))
    p.add_argument("--groups", nargs="*",
                   default=["table2", "table2_mlp", "msweep", "scaling", "perclass",
                            "validate_e"])
    p.add_argument("--flux-modes", nargs="*", default=list(FLUX_MODES))
    p.add_argument("--flux-rule", choices=("silverman", "nnspacing"), default="silverman")
    p.add_argument("--n-holdout", type=int, default=D.N_HOLDOUT)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--check", action="store_true",
                   help="assert that --flux fixed reproduces the saved frames exactly")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)
    torch.set_num_threads(int(args.threads))
    t0 = time.time()

    inv = build_inventory(args.groups)
    if args.limit:
        inv = inv[:int(args.limit)]
    if not inv:
        raise SystemExit("evaluate_v12_flux: empty inventory for groups %r" % (args.groups,))
    print("[flux] %d run dirs over groups %s" % (len(inv), ", ".join(sorted(args.groups))),
          flush=True)

    # pooled context first: data_lens_class.install() has no uninstall
    by_class = OrderedDict()
    for it in inv:
        by_class.setdefault(it["class_name"], []).append(it)
    order = ([None] if None in by_class else []) + [c for c in CLASSES if c in by_class]

    rows, checks, skipped = [], [], []
    for cname in order:
        ctx = EV.build_context(cname, n_holdout=args.n_holdout)
        print("[flux] context %s | held-out=%s target=%.4f"
              % ("pooled" if cname is None else cname, tuple(ctx.holdout.shape), ctx.target),
              flush=True)
        for it in by_class[cname]:
            rd = it["run_dir"]
            meta = EV.load_meta(rd)
            if meta is None:
                skipped.append((rd, "no meta.json"))
                continue
            crops = load_crops(rd)
            if crops is None:
                skipped.append((rd, "no crops/"))
                continue
            if crops.shape[1:] != (codec10.GRID, codec10.GRID):
                skipped.append((rd, "crop shape %s" % (crops.shape[1:],)))
                continue
            offsets = np.asarray(meta.get("gen_offsets"), dtype=np.int64)
            if offsets.ndim != 2 or offsets.shape[0] != crops.shape[0]:
                skipped.append((rd, "gen_offsets %s vs %d crops"
                                % (offsets.shape, crops.shape[0])))
                continue

            totals = train_totals_for(it["n_train"], cname)
            S_mean = float(totals.mean())
            if abs(S_mean - float(meta["S"])) > 1e-6 * max(1.0, abs(S_mean)):
                raise SystemExit("evaluate_v12_flux: recomputed S %.9f != meta S %.9f for %s "
                                 "-- the training split does not match this run"
                                 % (S_mean, float(meta["S"]), rd))

            for mode in args.flux_modes:
                if mode == "fixed":
                    S_gen = np.full(crops.shape[0], S_mean, dtype=np.float64)
                    extra = {"flux_rule": "mean", "flux_bandwidth": 0.0,
                             "flux_train_mean": S_mean,
                             "flux_train_sd": float(totals.std(ddof=1)) if totals.size > 1 else 0.0}
                elif mode == "resample":
                    S_gen, h = sample_flux_kde(totals, crops.shape[0], it["seed"],
                                               rule=args.flux_rule)
                    extra = {"flux_rule": args.flux_rule, "flux_bandwidth": h,
                             "flux_train_mean": S_mean,
                             "flux_train_sd": float(totals.std(ddof=1)) if totals.size > 1 else 0.0}
                else:
                    raise SystemExit("unknown --flux-modes entry %r" % mode)
                gen = frames_from_crops(crops, offsets, S_gen)
                if mode == "fixed" and args.check:
                    saved = EV.stack_generated(rd)
                    d = float(np.abs(saved.astype(np.float64)
                                     - gen.astype(np.float64)).max()) if saved is not None else -1.0
                    checks.append((rd, d))
                rows.append(score_frames(gen, rd, meta, ctx, it, mode, extra))
                if not args.quiet:
                    r = rows[-1]
                    print("[flux]   %-44s N=%-4d s%d %-8s js %.4e mmd2 %.5f mmd2_al %.5f"
                          % (r["arm"][:44], r["n_train"], r["seed"], mode,
                             r["pdf_js"], r["conv_mmd2"], r["conv_mmd2_aligned"]), flush=True)

    with open(args.results, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print("[flux] wrote %d rows to %s (%.1f s)" % (len(rows), args.results, time.time() - t0))
    if checks:
        worst = max(d for _, d in checks)
        print("[flux] --check: max |saved - rebuilt| over %d runs = %.3g%s"
              % (len(checks), worst, "  OK" if worst < 1e-3 else "  *** MISMATCH ***"))
        if worst >= 1e-3:
            for rd, d in sorted(checks, key=lambda t: -t[1])[:5]:
                print("[flux]   %.3g  %s" % (d, rd))
            return 1
    if skipped:
        print("[flux] skipped %d run(s):" % len(skipped))
        for rd, why in skipped[:10]:
            print("[flux]   %s -- %s" % (why, rd))
    return 0


if __name__ == "__main__":
    sys.exit(main())
