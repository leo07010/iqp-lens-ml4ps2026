#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Recompute Table 2's "Held-out split-half (floor)" row at 200-vs-200, bootstrapped.

WHY (the 100-vs-100 floor is not comparable to the model rows)
----------------------------------------------------------------------
evaluate_r3.holdout_reference splits the 200 held-out images 100/100 and scores that half
against the other half as the table's floor row.  Every OTHER row in the table (every
generator) is scored 200 GENERATED vs the FULL 200 held-out (evaluate_r3.score_run:
ctx["holdout"] is all 200).  conv_mmd2 is the BIASED empirical MMD^2 (metrics.py docstring,
Appendix I Eq. 62-63): the bias term of a biased-MMD^2 estimator scales like 1/n, so an
n=100 floor carries roughly TWICE the upward bias of the n=200 rows it is supposed to be a
lower bound for -- comparing them directly overstates how much better the floor is than the
generators. This script recomputes the floor at the SAME n=200-vs-200 the model rows use, and
draws it more than once (bootstrap) so the table can report a mean +- sd instead of the one
number one 100/100 split happens to give.

BOOTSTRAP POOL (disjoint from the scored 200 by construction, not by luck)
----------------------------------------------------------------------
The 200 held-out images used everywhere else (data_lens_r3.load_r3's `holdout_images`, and
per class after data_lens_class.install(c)) are the PREFIX of a fixed permutation:
    pooled : perm  = default_rng([R3_SPLIT_SEED, 3, 1]).permutation(n_val_all)
             held-out = perm[:200]                          (data_lens_r3.py:168-171)
    class c: vperm = default_rng([R3_SPLIT_SEED, 3, 1, class_index]).permutation(idx_va)
             held-out = vperm[:200]                          (data_lens_class.py:128-129)
The on-disk round-3 cache keeps only those 200 -- there is nothing left over to bootstrap
from -- so this script re-derives the SAME permutation straight from the source .npz (only
`val_x` / `val_y`, never `train_x`, which is ~1.3 GB of the file's 1.4 GB) and takes
`perm[200:]` / `vperm[200:]` as the bootstrap pool.  Same rng, same seed, so it is bit-for-bit
the leftover of the split every held-out set is drawn from, and a permutation's prefix and
suffix never overlap -- disjointness is structural, not asserted after the fact.

ALIGNED TARGET (must be the model rows' constant, not each draw's own mean)
----------------------------------------------------------------------
The four "*_aligned" metrics rescale every image to a fixed target total intensity before
scoring (evaluate_r3.align_to_target).  Every model row's target is the held-out set's OWN
mean total intensity, `ctx["holdout_mean_total"]` (evaluate_r3.py:457-458; the identical
pattern in seeds/rescore_extra_metrics.py:206 for per-class rows).  For this floor's aligned
numbers to sit in the same column as the model rows', its target must be that exact constant
-- not the mean of whichever 200 images a bootstrap draw happens to contain -- so it is
computed once per kind from `data_lens_r3.load_r3(100, n_holdout=200).holdout_images` (pooled)
or the same call after `data_lens_class.install(c)` (per class), and reused for every
replicate of that kind.

    python3 seeds/floor_bootstrap.py --n-boot 20 --n-side 200 --seed 0 \\
        --results results_seeds_floor.jsonl

Rows are written fresh (mode "w"): this script always regenerates the full bootstrap in one
run, unlike the seed-replicate scripts that append one row per pre-existing run dir.

Python 3.9 compatible (no `X | None`).  CPU only -- the encoder forward passes and 200x200
distance matrices here are trivial next to a training run.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

REPO = "<HOME>/gsoc2026-deeplense-qml/experiments/13_iqp_turbulence_replication"
sys.path.insert(0, REPO)

import numpy as np                                      # noqa: E402
import torch                                            # noqa: E402

import data_lens_class                                  # noqa: E402
import data_lens_r3 as D                                # noqa: E402
import evaluate_r3                                      # noqa: E402
import metrics                                          # noqa: E402

CLASSES = data_lens_class.CLASSES        # ("axion", "cdm", "no_sub"), dataset's own order


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-boot", type=int, default=20,
                   help="number of non-overlapping real-vs-real draws per kind "
                        "(pooled + 3 classes)")
    p.add_argument("--n-side", type=int, default=200,
                   help="size of each of the two disjoint real sides A, B per draw "
                        "(200 matches every model row's n_gen / n_holdout)")
    p.add_argument("--seed", type=int, default=0,
                   help="seed for the bootstrap SAMPLING rng only; the data split "
                        "(R3_SPLIT_SEED) and conv_mmd2's encoder seed (CONV_SEED) are fixed "
                        "constants this flag does not touch")
    p.add_argument("--npz", default=None,
                   help="override the source .npz path (default: data_lens_r3's own "
                        "default / $LENS_NPZ)")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--results", default=os.path.join(REPO, "results_seeds_floor.jsonl"))
    return p.parse_args(argv)


# --------------------------------------------------------------------------------------
# aligned target: the SAME constant the model rows align to (see module docstring)
# --------------------------------------------------------------------------------------
def aligned_target(class_name):
    """Held-out set's own mean total intensity, computed exactly the way evaluate_r3 / class
    rescoring compute `ctx["holdout_mean_total"]` -- bit-for-bit the model rows' constant."""
    if class_name is not None:
        data_lens_class.install(class_name)
    d = D.load_r3(D.N_TRAIN_MAX, n_holdout=D.N_HOLDOUT)
    ho = d.holdout_images
    return float(ho.reshape(ho.shape[0], -1).sum(axis=1).mean())


# --------------------------------------------------------------------------------------
# bootstrap pool: perm[N_HOLDOUT:] of the SAME permutation the held-out 200 are perm[:N_HOLDOUT]
# of (see module docstring). Reads val_x/val_y only -- train_x (~1.3 GB) is never touched.
# --------------------------------------------------------------------------------------
def _read_val_arrays(npz_path):
    npz = np.load(npz_path)
    class_names = [str(s) for s in np.asarray(npz["class_names"]).tolist()]
    val_y = np.asarray(npz["val_y"]).astype(np.int64)
    val_x = np.asarray(npz["val_x"], dtype=np.float32)
    return class_names, val_y, val_x


def pooled_pool(val_x, n_val_all):
    perm = np.random.default_rng([D.R3_SPLIT_SEED, 3, 1]).permutation(n_val_all)
    pool_idx = perm[D.N_HOLDOUT:]
    return np.ascontiguousarray(val_x[pool_idx])


def class_pool(val_x, val_y, class_names, class_name):
    ci = class_names.index(class_name)
    idx_va = np.flatnonzero(val_y == ci)
    vperm = np.random.default_rng([D.R3_SPLIT_SEED, 3, 1, ci]).permutation(idx_va)
    pool_idx = vperm[D.N_HOLDOUT:]
    return np.ascontiguousarray(val_x[pool_idx])


# --------------------------------------------------------------------------------------
# one kind's bootstrap (pooled, or one class)
# --------------------------------------------------------------------------------------
def bootstrap_kind(kind, class_name, pool, target, n_boot, n_side, seed, results_fh):
    """`n_boot` disjoint-A/B real-vs-real draws from `pool`, one JSON row per draw appended to
    `results_fh`.  A different sampling rng per (kind, draw) -- `kind_key` keeps the four
    kinds' draws from reusing the same permutation -- but conv_mmd2's encoder seed is always
    the fixed CONV_SEED (the encoder must not move, or the feature space would not be shared
    across kinds/draws the way it is across every table row)."""
    need = 2 * int(n_side)
    if pool.shape[0] < need:
        raise SystemExit("floor_bootstrap: %s pool has only %d images, need >= %d "
                         "(2 x n_side=%d)" % (kind, pool.shape[0], need, n_side))
    kind_key = 0 if class_name is None else (1 + CLASSES.index(class_name))

    js_vals = np.empty(n_boot, dtype=np.float64)
    mmd_vals = np.empty(n_boot, dtype=np.float64)
    js_al_vals = np.empty(n_boot, dtype=np.float64)
    mmd_al_vals = np.empty(n_boot, dtype=np.float64)

    for b in range(n_boot):
        rng = np.random.default_rng([seed, 9, b, kind_key])
        draw = rng.permutation(pool.shape[0])[:need]
        a = pool[draw[:n_side]]
        b_img = pool[draw[n_side:need]]

        js = float(metrics.pdf_js(a, b_img))
        mmd = float(metrics.conv_mmd2(a, b_img, seed=evaluate_r3.CONV_SEED))
        a_al = evaluate_r3.align_to_target(a, target)
        b_al = evaluate_r3.align_to_target(b_img, target)
        js_al = float(metrics.pdf_js(a_al, b_al))
        mmd_al = float(metrics.conv_mmd2(a_al, b_al, seed=evaluate_r3.CONV_SEED))

        js_vals[b], mmd_vals[b] = js, mmd
        js_al_vals[b], mmd_al_vals[b] = js_al, mmd_al

        row = {"kind": "floor_bootstrap", "class_name": class_name, "boot": int(b),
               "n_side": int(n_side), "pdf_js": js, "conv_mmd2": mmd,
               "pdf_js_aligned": js_al, "conv_mmd2_aligned": mmd_al,
               "seed": int(seed), "conv_seed": int(evaluate_r3.CONV_SEED),
               "aligned_target_total": float(target), "pool_size": int(pool.shape[0])}
        results_fh.write(json.dumps(row) + "\n")
        print("[floor-boot] %-7s %2d/%2d  pdf_js=%.6e conv_mmd2=%.6f  "
              "aligned pdf_js=%.6e conv_mmd2=%.6f"
              % (kind, b + 1, n_boot, js, mmd, js_al, mmd_al), flush=True)

    return js_vals, mmd_vals, js_al_vals, mmd_al_vals


def _mean_sd(arr):
    m = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1)) if arr.size > 1 else float("nan")
    return m, sd


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(int(args.threads))
    t0 = time.time()

    # Pooled target FIRST: data_lens_class.install() monkey-patches data_lens_r3's module-level
    # _arrays/build_cache to one class until re-installed, so the pooled constant has to be
    # read before any install() call touches that state (see aligned_target docstring).
    print("[floor-boot] aligned targets (held-out mean total, same constant every model row "
          "uses) ...", flush=True)
    target_pooled = aligned_target(None)
    targets = {}
    for c in CLASSES:
        targets[c] = aligned_target(c)
    print("[floor-boot]   pooled=%.4f  %s"
          % (target_pooled, "  ".join("%s=%.4f" % (c, targets[c]) for c in CLASSES)),
          flush=True)

    npz_path = args.npz if args.npz else D._npz_path()
    if not os.path.exists(npz_path):
        raise SystemExit("floor_bootstrap: lens dataset not found: %s" % npz_path)
    print("[floor-boot] reading val_x/val_y from %s (train_x not touched) ..." % npz_path,
          flush=True)
    class_names, val_y, val_x = _read_val_arrays(npz_path)
    n_val_all = int(val_y.shape[0])

    pools = {None: pooled_pool(val_x, n_val_all)}
    for c in CLASSES:
        pools[c] = class_pool(val_x, val_y, class_names, c)
    del val_x, npz_path  # the 1.4 GB source is done being needed; keep only the pools

    print("[floor-boot] bootstrap pool sizes: pooled=%d  %s"
          % (pools[None].shape[0],
             "  ".join("%s=%d" % (c, pools[c].shape[0]) for c in CLASSES)), flush=True)

    summary = []
    with open(args.results, "w") as fh:
        summary.append(("pooled", bootstrap_kind("pooled", None, pools[None], target_pooled,
                                                  args.n_boot, args.n_side, args.seed, fh)))
        for c in CLASSES:
            summary.append((c, bootstrap_kind(c, c, pools[c], targets[c],
                                              args.n_boot, args.n_side, args.seed, fh)))

    print()
    print("[floor-boot] ==== 200-vs-200 real-vs-real floor: mean +/- sd over %d draws "
          "(n_side=%d) ====" % (args.n_boot, args.n_side))
    print("%-8s  %-24s  %-20s  %-24s  %-20s"
          % ("kind", "pdf_js", "conv_mmd2", "pdf_js_aligned", "conv_mmd2_aligned"))
    for name, (js, mmd, js_al, mmd_al) in summary:
        js_m, js_sd = _mean_sd(js)
        mmd_m, mmd_sd = _mean_sd(mmd)
        jsa_m, jsa_sd = _mean_sd(js_al)
        mmda_m, mmda_sd = _mean_sd(mmd_al)
        print("%-8s  %10.4e +/- %8.2e  %8.6f +/- %8.6f  %10.4e +/- %8.2e  %8.6f +/- %8.6f"
              % (name if name else "pooled", js_m, js_sd, mmd_m, mmd_sd,
                 jsa_m, jsa_sd, mmda_m, mmda_sd))

    n_rows = args.n_boot * (1 + len(CLASSES))
    print("\n[floor-boot] wrote %d rows to %s (%.1f s)"
          % (n_rows, args.results, time.time() - t0), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
