#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Metric protocol **v11** -- PDF-JS and Conv2D-MMD^2 with every generator-dependent knob removed.

WHY A NEW PROTOCOL (what the 2026-09-09 audit found in metrics.py)
---------------------------------------------------------------------------------------
Three properties of the round-3 protocol (`metrics.py`, used by `evaluate_r3.py` /
`evaluate_r3b.py` / `evaluate_class.py`) let one arm's own output change the yardstick the
whole table is scored with:

1. PDF-JS bins (`metrics.py:32-36`).  The 50 histogram bins span the JOINT min/max of
   real UNION generated, so an arm whose brightest pixel is 2.7 (several IQP arms reach
   1.5-2.7 while the data live in [0,1]) widens every bin and is scored on a coarser
   histogram than an arm that stays inside [0,1].  Measured: setting ONE pixel of the DCGAN
   output to 4.0 divides its PDF-JS by ~13.  The PDF-JS winner therefore moves with the bin
   rule, which is a property of the ruler, not of the samples.
2. Conv2D-MMD^2 input scaling (`metrics.py:94-95`).  The encoder input is normalised by the
   JOINT min/max as well, so each arm enters the feature extractor on its own scale.
3. Conv2D-MMD^2 bandwidth (`metrics.py:112-116`).  sigma comes from the median heuristic on
   the POOLED (real + generated) features, so every arm is scored with a DIFFERENT Gaussian
   kernel.  Measured: sharing one bandwidth across arms moves RBM from 2nd to 4th.
4. The `HOLDOUT split-half` floor is 100-vs-100 (`evaluate_r3.holdout_reference`) while every
   model row is 200-vs-200.  The bias of the biased MMD^2 estimator scales like 1/n, so the
   floor carries ~2x the upward bias of the rows it is a floor for.

v11 DEFINITIONS (the fixes; every one of them is generator-independent)
---------------------------------------------------------------------------------------
`pdf_js_v11(real, gen)`
    Both sets are CLIPPED to the data range [0, 1] (DeepLense Model II images are per-image
    min-max normalised, so 0 and 1 are the data's exact extremes, not a convention), the
    histogram is 50 EQUAL-WIDTH bins fixed on [0, 1], JS in bits (log base 2), eps-smoothed
    exactly as before.  The bin grid no longer depends on either input.  The amount of
    generated mass the clip moved is REPORTED (`gen_clip_frac`, `gen_max`) rather than
    silently rewarded or punished: an arm that puts 0.3% of its pixels above 1 is now visibly
    doing so instead of quietly coarsening its own histogram.
    For the flux-aligned block the data range is not [0, 1] any more, so pass the
    reference-derived range (`lo=`, `hi=`; see `range_of`), which keeps the same rule --
    the bins come from the HELD-OUT side only.

`conv_mmd2_v11(real, gen, ref)`
    Same fixed random Conv2D encoder as v10 (`metrics.ConvEncoder`, CONV_SEED = 0,
    1->16->32->64 stride-2 blocks + GAP), but
      * the encoder's input normalisation uses ONLY the reference (held-out) set's min/max;
        anything outside is clipped, so an arm cannot rescale the feature extractor's input;
      * the Gaussian bandwidth sigma is the median heuristic computed ONCE on the REFERENCE
        features and shared by every arm, every N and every floor draw.
    Both estimators are returned: `conv_mmd2` (biased V-statistic, the one Table 2 quotes and
    the one v10 computed) and `conv_mmd2u` (unbiased U-statistic, diagonal removed from both
    within-set terms).  N = M = 200.  The flux-aligned variant is the identical rule applied
    to arrays that `evaluate_r3.align_to_target` has already rescaled, with `ref` = the
    aligned held-out set.

floor (`floor_kind`)
    200-vs-200 real-vs-real draws -- the same n as every model row -- from the validation
    images LEFT OVER by the held-out split (perm[200:] of the same permutation whose prefix
    perm[:200] is the held-out set; per class, vperm[200:] of that class's permutation), so
    the floor pool is disjoint from the scored held-out set by construction.  Every draw is
    scored with the SAME ref-derived scale and bandwidth as the model rows, i.e. against the
    held-out set's encoder context, and reported as mean +- sd over the draws, for the biased
    AND the unbiased estimator, raw and flux-aligned, per class as well as pooled.
    `--draw-mode disjoint` additionally makes the draws mutually exclusive (one permutation
    cut into consecutive 400-image blocks); it needs n_boot * 2 * n_side images, which the
    pooled pool (8,710) has and a single class's pool (~2,770) does not, so the default
    `resample` mode (an independent permutation per draw, A and B disjoint within a draw) is
    what the per-class floors use.  The `resample` rng is bit-for-bit
    `seeds/floor_bootstrap.py`'s, so v11 and legacy floors are computed on the SAME draws.

Legacy numbers are kept alongside (`*_legacy`) so nothing has to be re-derived to compare.

This file is NEW.  It does not modify metrics.py, evaluate_r3.py, evaluate_r3b.py or
evaluate_class.py, all of which produced the frozen v10 numbers.

CLI (floor only; per-run scoring lives in evaluate_v11.py)
---------------------------------------------------------------------------------------
    python3 metrics_v11.py --n-boot 20 --n-side 200 --results results_v11_floor.jsonl
    python3 metrics_v11.py --n-boot 20 --draw-mode disjoint --classes ""   # pooled only

Python 3.9 compatible (no `X | None`).  CPU only.
"""
from __future__ import annotations

import argparse
import json
import os
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

__all__ = ["pdf_js_v11", "conv_mmd2_v11", "prepare_ref", "RefContext", "range_of",
           "floor_kind", "val_pool", "read_val_arrays", "CONV_SEED", "N_BINS",
           "DATA_LO", "DATA_HI", "N_EVAL"]

CONV_SEED = evaluate_r3.CONV_SEED        # 0 -- the SAME encoder v10 used
N_BINS = 50                              # the reference paper's bin count, kept
DATA_LO, DATA_HI = 0.0, 1.0              # DeepLense Model II images are per-image min-max normalised
N_EVAL = 200
CLASSES = data_lens_class.CLASSES        # ("axion", "cdm", "no_sub")


# ======================================================================================
# (i) PDF Jensen-Shannon divergence, fixed [0, 1] bins
# ======================================================================================
def range_of(ref):
    """(lo, hi) of a reference array -- the only thing allowed to set a bin grid or an
    encoder input scale in v11.  For the raw held-out images this returns (0.0, 1.0)."""
    r = np.asarray(ref, dtype=np.float64)
    return float(r.min()), float(r.max())


def _clip_stats(arr, lo, hi, prefix):
    a = np.asarray(arr, dtype=np.float64)
    n = float(a.size)
    below = float(np.count_nonzero(a < lo)) / n if n else 0.0
    above = float(np.count_nonzero(a > hi)) / n if n else 0.0
    return {prefix + "_min": float(a.min()) if n else float("nan"),
            prefix + "_max": float(a.max()) if n else float("nan"),
            prefix + "_clip_frac_below": below,
            prefix + "_clip_frac_above": above,
            prefix + "_clip_frac": below + above}


def _hist_pmf(x, edges, eps):
    p, _ = np.histogram(x, bins=edges)
    p = p.astype(np.float64) / max(p.sum(), 1.0) + eps
    return p / p.sum()


def pdf_js_v11(real, gen, bins=N_BINS, lo=DATA_LO, hi=DATA_HI, eps=1e-12):
    """JS divergence (bits) on 50 equal-width bins FIXED on [lo, hi]; both sides clipped there.

    Returns a dict, not a scalar, because the clip is part of the measurement and has to be
    reported with it:
        pdf_js            the divergence
        gen_max / gen_min raw extremes of `gen` BEFORE the clip
        gen_clip_frac     fraction of generated pixels the clip moved (+ _above / _below)
        real_*            the same three for the reference side (0 for real data on [0,1])
    """
    x = np.asarray(real, dtype=np.float64).ravel()
    y = np.asarray(gen, dtype=np.float64).ravel()
    lo = float(lo)
    hi = float(hi)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        raise ValueError("pdf_js_v11: need a finite lo < hi, got (%r, %r)" % (lo, hi))
    edges = np.linspace(lo, hi, int(bins) + 1)
    p = _hist_pmf(np.clip(x, lo, hi), edges, eps)
    q = _hist_pmf(np.clip(y, lo, hi), edges, eps)
    m = 0.5 * (p + q)
    kl_pm = float(np.sum(p * (np.log2(p) - np.log2(m))))
    kl_qm = float(np.sum(q * (np.log2(q) - np.log2(m))))
    out = {"pdf_js": 0.5 * kl_pm + 0.5 * kl_qm,
           "bins": int(bins), "range_lo": lo, "range_hi": hi}
    out.update(_clip_stats(y, lo, hi, "gen"))
    out.update(_clip_stats(x, lo, hi, "real"))
    return out


# ======================================================================================
# (ii) Conv2D feature-space MMD^2, reference-only scale and bandwidth
# ======================================================================================
class RefContext(object):
    """The frozen yardstick: one encoder, one input scale, one bandwidth -- all from `ref`."""

    __slots__ = ("lo", "hi", "sigma", "encoder", "feat", "n_ref", "seed")

    def __repr__(self):
        return ("RefContext(n_ref=%d, lo=%.6g, hi=%.6g, sigma=%.6f, seed=%d)"
                % (self.n_ref, self.lo, self.hi, self.sigma, self.seed))

    def as_dict(self):
        return {"ref_lo": self.lo, "ref_hi": self.hi, "sigma": self.sigma,
                "n_ref": self.n_ref, "conv_seed": self.seed}


def build_encoder(seed=CONV_SEED):
    """metrics.ConvEncoder with metrics.conv_mmd2's exact initialisation -- same weights as v10."""
    gen_t = torch.Generator().manual_seed(int(seed))
    encoder = legacy_metrics.ConvEncoder()
    with torch.no_grad():
        for p in encoder.parameters():
            if p.dim() > 1:
                fan_in = int(np.prod(p.shape[1:]))
                bound = 1.0 / np.sqrt(fan_in)
                p.uniform_(-bound, bound, generator=gen_t)
            else:
                p.zero_()
    encoder.eval()
    return encoder


def _features(arr, encoder, lo, hi):
    """Encoder features of `arr` after CLIPPING to [lo, hi] and mapping that range to [0, 1]."""
    x = np.asarray(arr, dtype=np.float64)
    x = np.clip(x, lo, hi)
    x = (x - lo) / max(hi - lo, 1e-12)
    t = torch.as_tensor(x, dtype=torch.float32).unsqueeze(1)
    with torch.no_grad():
        return encoder(t).double()


def _median_heuristic(feat):
    d = torch.cdist(feat, feat, p=2)
    iu = torch.triu_indices(d.shape[0], d.shape[0], offset=1)
    med = float(torch.median(d[iu[0], iu[1]]))
    return med if med > 0 else 1.0


def prepare_ref(ref, seed=CONV_SEED, lo=None, hi=None):
    """Freeze the yardstick on the reference (held-out) set: input range and bandwidth."""
    r_lo, r_hi = range_of(ref)
    ctx = RefContext()
    ctx.lo = r_lo if lo is None else float(lo)
    ctx.hi = r_hi if hi is None else float(hi)
    ctx.seed = int(seed)
    ctx.encoder = build_encoder(seed)
    ctx.feat = _features(ref, ctx.encoder, ctx.lo, ctx.hi)
    ctx.n_ref = int(ctx.feat.shape[0])
    ctx.sigma = _median_heuristic(ctx.feat)
    return ctx


def _subsample(arr, n, rng):
    arr = np.asarray(arr)
    if n is None or arr.shape[0] <= n:
        return arr
    idx = rng.choice(arr.shape[0], size=n, replace=False)
    return arr[np.sort(idx)]


def _gauss(a, b, sigma):
    dd = torch.cdist(a, b, p=2) ** 2
    return torch.exp(-dd / (2.0 * sigma * sigma))


def conv_mmd2_v11(real, gen, ref=None, prepared=None, seed=CONV_SEED, n_eval=N_EVAL,
                  subsample_seed=0):
    """Biased AND unbiased MMD^2 in the fixed feature space, scaled and banded by `ref` only.

    Pass either `ref` (the held-out array) or a `prepared` RefContext -- the latter avoids
    re-encoding the reference for every row, and guarantees every row shares one sigma.

    Returns a dict:
        conv_mmd2    biased  V-statistic  = mean K(f,f) + mean K(g,g) - 2 mean K(f,g)
        conv_mmd2u   unbiased U-statistic (both diagonals removed)
        sigma, ref_lo, ref_hi, n_ref, conv_seed, n_real, n_gen
        gen_clip_frac / gen_max / ...  what the ref-range clip did to `gen`
    """
    if prepared is None:
        if ref is None:
            raise ValueError("conv_mmd2_v11: need ref= or prepared=")
        prepared = prepare_ref(ref, seed=seed)
    rng = np.random.default_rng(int(subsample_seed))
    real = _subsample(real, n_eval, rng)
    gen = _subsample(gen, n_eval, rng)

    f = _features(real, prepared.encoder, prepared.lo, prepared.hi)
    g = _features(gen, prepared.encoder, prepared.lo, prepared.hi)
    n = int(f.shape[0])
    m = int(g.shape[0])
    kff = _gauss(f, f, prepared.sigma)
    kgg = _gauss(g, g, prepared.sigma)
    kfg = _gauss(f, g, prepared.sigma)

    biased = float(kff.mean() + kgg.mean() - 2.0 * kfg.mean())
    if n > 1 and m > 1:
        sff = float(kff.sum() - torch.diagonal(kff).sum()) / (n * (n - 1))
        sgg = float(kgg.sum() - torch.diagonal(kgg).sum()) / (m * (m - 1))
        unbiased = float(sff + sgg - 2.0 * float(kfg.mean()))
    else:
        unbiased = float("nan")

    out = {"conv_mmd2": biased, "conv_mmd2u": unbiased,
           "n_real": n, "n_gen": m}
    out.update(prepared.as_dict())
    out.update(_clip_stats(gen, prepared.lo, prepared.hi, "gen"))
    return out


# ======================================================================================
# floor: 200-vs-200 real-vs-real, same yardstick as the model rows
# ======================================================================================
def read_val_arrays(npz_path=None):
    """(class_names, val_y, val_x) from the source .npz.  Reads val_* only -- train_x is
    ~1.3 GB of the file's 1.4 GB and is never touched here."""
    npz_path = npz_path if npz_path else D._npz_path()
    if not os.path.exists(npz_path):
        raise FileNotFoundError("lens dataset not found: %s" % npz_path)
    npz = np.load(npz_path)
    class_names = [str(s) for s in np.asarray(npz["class_names"]).tolist()]
    val_y = np.asarray(npz["val_y"]).astype(np.int64)
    val_x = np.asarray(npz["val_x"], dtype=np.float32)
    return class_names, val_y, val_x


def val_pool(val_x, val_y=None, class_names=None, class_name=None):
    """The validation images NOT in the held-out block: perm[N_HOLDOUT:] of the very
    permutation data_lens_r3 (pooled) / data_lens_class (per class) take perm[:N_HOLDOUT]
    from, so disjointness from the scored held-out set is structural."""
    if class_name is None:
        perm = np.random.default_rng([D.R3_SPLIT_SEED, 3, 1]).permutation(int(val_x.shape[0]))
        return np.ascontiguousarray(val_x[perm[D.N_HOLDOUT:]])
    ci = list(class_names).index(class_name)
    idx_va = np.flatnonzero(np.asarray(val_y) == ci)
    vperm = np.random.default_rng([D.R3_SPLIT_SEED, 3, 1, ci]).permutation(idx_va)
    return np.ascontiguousarray(val_x[vperm[D.N_HOLDOUT:]])


def _draws(pool_n, n_boot, n_side, seed, kind_key, draw_mode):
    """Index pairs (A, B) for each replicate.

    resample : one independent permutation per draw -- A and B disjoint within a draw, draws
               may share images.  Bit-for-bit seeds/floor_bootstrap.py's rng, so the v11 and
               legacy floors land on the same images.
    disjoint : ONE permutation cut into consecutive 2*n_side blocks -- the draws are mutually
               exclusive as well.  Needs n_boot * 2 * n_side images.
    """
    need = 2 * int(n_side)
    if draw_mode == "disjoint":
        if pool_n < n_boot * need:
            raise ValueError("draw-mode disjoint needs %d images, pool has %d"
                             % (n_boot * need, pool_n))
        perm = np.random.default_rng([int(seed), 9, 777, int(kind_key)]).permutation(pool_n)
        for b in range(n_boot):
            blk = perm[b * need:(b + 1) * need]
            yield b, blk[:n_side], blk[n_side:]
        return
    if pool_n < need:
        raise ValueError("floor draw needs %d images, pool has %d" % (need, pool_n))
    for b in range(n_boot):
        rng = np.random.default_rng([int(seed), 9, b, int(kind_key)])
        draw = rng.permutation(pool_n)[:need]
        yield b, draw[:n_side], draw[n_side:need]


def holdout_of(class_name=None):
    """The 200 held-out images every model row is scored against (pooled, or one class).
    Call the pooled one BEFORE any data_lens_class.install(): install() monkey-patches
    data_lens_r3's module state and there is no uninstall."""
    if class_name is not None:
        data_lens_class.install(class_name)
    d = D.load_r3(D.N_TRAIN_MAX, n_holdout=D.N_HOLDOUT)
    return np.ascontiguousarray(d.holdout_images)


def floor_kind(pool, holdout, n_boot=20, n_side=200, seed=0, kind_key=0,
               draw_mode="resample", class_name=None, with_legacy=True, verbose=True):
    """One kind's floor (pooled, or one class).  Returns (rows, summary).

    Every draw is scored with the SAME RefContext the model rows use -- built from `holdout`,
    not from the draw -- for the raw and for the flux-aligned block.
    """
    target = float(np.asarray(holdout, dtype=np.float64)
                   .reshape(holdout.shape[0], -1).sum(axis=1).mean())
    ho_al = evaluate_r3.align_to_target(holdout, target)
    prep_raw = prepare_ref(holdout, seed=CONV_SEED)
    prep_al = prepare_ref(ho_al, seed=CONV_SEED)
    lo_al, hi_al = prep_al.lo, prep_al.hi

    rows = []
    for b, ia, ib in _draws(int(pool.shape[0]), int(n_boot), int(n_side), int(seed),
                            int(kind_key), draw_mode):
        a = pool[ia]
        c = pool[ib]
        js = pdf_js_v11(a, c)
        mm = conv_mmd2_v11(a, c, prepared=prep_raw)
        a_al = evaluate_r3.align_to_target(a, target)
        c_al = evaluate_r3.align_to_target(c, target)
        js_al = pdf_js_v11(a_al, c_al, lo=lo_al, hi=hi_al)
        mm_al = conv_mmd2_v11(a_al, c_al, prepared=prep_al)
        row = {"kind": "floor_v11", "class_name": class_name, "boot": int(b),
               "n_side": int(n_side), "draw_mode": draw_mode, "seed": int(seed),
               "pdf_js": js["pdf_js"], "pdf_js_aligned": js_al["pdf_js"],
               "conv_mmd2": mm["conv_mmd2"], "conv_mmd2u": mm["conv_mmd2u"],
               "conv_mmd2_aligned": mm_al["conv_mmd2"],
               "conv_mmd2u_aligned": mm_al["conv_mmd2u"],
               "sigma": prep_raw.sigma, "sigma_aligned": prep_al.sigma,
               "ref_lo": prep_raw.lo, "ref_hi": prep_raw.hi,
               "aligned_target_total": target, "pool_size": int(pool.shape[0]),
               "conv_seed": CONV_SEED, "metric_protocol": "v11"}
        if with_legacy:
            row["pdf_js_legacy"] = float(legacy_metrics.pdf_js(a, c))
            row["conv_mmd2_legacy"] = float(legacy_metrics.conv_mmd2(a, c, seed=CONV_SEED))
            row["pdf_js_aligned_legacy"] = float(legacy_metrics.pdf_js(a_al, c_al))
            row["conv_mmd2_aligned_legacy"] = float(
                legacy_metrics.conv_mmd2(a_al, c_al, seed=CONV_SEED))
        rows.append(row)
        if verbose:
            print("[floor-v11] %-7s %2d/%2d  js=%.4e mmd2=%.6f mmd2u=%.6f | al js=%.4e "
                  "mmd2=%.6f mmd2u=%.6f"
                  % (class_name if class_name else "pooled", b + 1, n_boot,
                     row["pdf_js"], row["conv_mmd2"], row["conv_mmd2u"],
                     row["pdf_js_aligned"], row["conv_mmd2_aligned"],
                     row["conv_mmd2u_aligned"]), flush=True)

    keys = ["pdf_js", "pdf_js_aligned", "conv_mmd2", "conv_mmd2u",
            "conv_mmd2_aligned", "conv_mmd2u_aligned"]
    if with_legacy:
        keys += ["pdf_js_legacy", "conv_mmd2_legacy", "pdf_js_aligned_legacy",
                 "conv_mmd2_aligned_legacy"]
    summary = {"kind": "floor_v11_summary", "class_name": class_name,
               "n_boot": int(n_boot), "n_side": int(n_side), "draw_mode": draw_mode,
               "seed": int(seed), "pool_size": int(pool.shape[0]),
               "sigma": prep_raw.sigma, "sigma_aligned": prep_al.sigma,
               "aligned_target_total": target, "metric_protocol": "v11"}
    for k in keys:
        v = np.asarray([r[k] for r in rows], dtype=np.float64)
        summary[k + "_mean"] = float(v.mean())
        summary[k + "_sd"] = float(v.std(ddof=1)) if v.size > 1 else float("nan")
    return rows, summary


# ======================================================================================
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="v11 floor: 200-vs-200 real-vs-real, pooled + per class.",
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-boot", type=int, default=20)
    p.add_argument("--n-side", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--draw-mode", choices=("resample", "disjoint"), default="resample")
    p.add_argument("--classes", nargs="*", default=list(CLASSES),
                   help="per-class floors to compute as well as the pooled one; pass none to "
                        "do pooled only")
    p.add_argument("--npz", default=None)
    p.add_argument("--no-legacy", action="store_true", help="skip the legacy columns")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--results", default=os.path.join(REPO, "results_v11_floor.jsonl"))
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(int(args.threads))
    t0 = time.time()
    classes = [c for c in (args.classes or []) if c]

    # pooled held-out FIRST: data_lens_class.install() has no uninstall
    holdouts = {None: holdout_of(None)}
    for c in classes:
        holdouts[c] = holdout_of(c)
    print("[floor-v11] held-out sets: " + "  ".join(
        "%s=%s" % (k if k else "pooled", tuple(v.shape)) for k, v in holdouts.items()), flush=True)

    class_names, val_y, val_x = read_val_arrays(args.npz)
    pools = {None: val_pool(val_x, class_name=None)}
    for c in classes:
        pools[c] = val_pool(val_x, val_y, class_names, c)
    del val_x
    print("[floor-v11] pools: " + "  ".join(
        "%s=%d" % (k if k else "pooled", v.shape[0]) for k, v in pools.items()), flush=True)

    all_rows = []
    summaries = []
    for kind_key, c in enumerate([None] + classes):
        rows, summ = floor_kind(pools[c], holdouts[c], n_boot=args.n_boot, n_side=args.n_side,
                                seed=args.seed, kind_key=kind_key, draw_mode=args.draw_mode,
                                class_name=c, with_legacy=not args.no_legacy)
        all_rows.extend(rows)
        summaries.append(summ)

    with open(args.results, "w") as fh:
        for r in all_rows:
            fh.write(json.dumps(r) + "\n")
        for s in summaries:
            fh.write(json.dumps(s) + "\n")

    print()
    print("### v11 floor (%d draws of %d-vs-%d real images, draw-mode=%s)"
          % (args.n_boot, args.n_side, args.n_side, args.draw_mode))
    print("| kind | PDF–JS | MMD² | MMD²_u | PDF–JS (al) | MMD² (al) | MMD²_u (al) | "
          "PDF–JS legacy | MMD² legacy |")
    print("|---|---|---|---|---|---|---|---|---|")
    for s in summaries:
        def mv(k):
            if (k + "_mean") not in s:
                return "—"
            return "%.4g ± %.2g" % (s[k + "_mean"], s[k + "_sd"])
        print("| %s | %s | %s | %s | %s | %s | %s | %s | %s |"
              % (s["class_name"] if s["class_name"] else "pooled",
                 mv("pdf_js"), mv("conv_mmd2"), mv("conv_mmd2u"), mv("pdf_js_aligned"),
                 mv("conv_mmd2_aligned"), mv("conv_mmd2u_aligned"),
                 mv("pdf_js_legacy"), mv("conv_mmd2_legacy")))
    print("\n[floor-v11] wrote %d rows (+%d summaries) to %s (%.1f s)"
          % (len(all_rows), len(summaries), args.results, time.time() - t0), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
