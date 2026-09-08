#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extra evaluation columns for every finished run, added post-hoc without touching evaluate_r3.

The audit raised four things the published columns cannot answer.  All four are recomputed
here from each run's own `generated_*.npy`, against the same held-out set and the same
CONV_SEED evaluate_r3 uses, so a row here joins onto that run's evaluate_r3 row by `run_dir`.
No existing script, run directory or results file is modified: this writes one new file.

1. `pdf_js_fixed01`
   `metrics.pdf_js` bins on the JOINT range of (held-out, that arm).  An arm that emits
   pixels outside the real range therefore gets its histogram computed on a stretched axis,
   and two arms are not binned on the same grid at all -- the comparison moves with the
   arm.  This column fixes the grid: 50 bins on [0, 1], log base 2, both sides clipped into
   [0, 1] first.  The held-out images already live in [0, 1] exactly, so the clip only ever
   bites on generated pixels, which is the asymmetry being measured.
   No aligned counterpart is written: flux-alignment multiplies every image by its own
   scalar, so aligned pixels are not on the [0, 1] scale that makes a fixed grid meaningful.

2. `conv_mmd2_fixedsigma`, `conv_mmd2_fixedsigma_aligned`
   `metrics.conv_mmd2` takes its normalisation range from min/max over BOTH sets and its
   RBF bandwidth from the median pairwise distance over the POOLED features.  Both therefore
   depend on the arm being scored, so each arm is measured with a slightly different kernel.
   These columns take the range and the median-heuristic sigma from the HELD-OUT (real)
   features alone, which makes the kernel identical for every arm and every seed.  Everything
   else -- the encoder weights, CONV_SEED, the subsampling -- is unchanged.

3. `params_shared`, `params_total_at_N`
   The paper's Params column is one number per arm, but the arms distribute parameters very
   differently between "shared across images" and "one set per training image".  Both numbers
   are derived from each run's own meta.json, arithmetic only:
     fixed-basis IQP   d_core                 + N * d_lat
     affine IQP        param_count_shared     + N * M            (= its recorded param_count)
     RBM               W + shared hidden bias + N * (10 + 40)    (per-image b_v and b_h_lat)
     DCGAN             generator | discriminator, neither per-image
   `params_generator` / `params_discriminator` carry the DCGAN split.

4. `S`
   The training set's mean total flux, which sets the scale every decoded image is
   multiplied by.  It is in meta.json but was never copied into the results rows, so a table
   could not show that N = 11 and N = 100 are scored at different S (~112.3 vs ~124.9).

    python3 seeds/rescore_extra_metrics.py --outputs-dir outputs_lens_r3 outputs_lens_r3b \\
        --results results_seeds_extra.jsonl

Per-class runs are detected by `class_name` in meta.json and scored against THEIR class's
held-out set, exactly as evaluate_class.py does.

Python 3.9 compatible.  CPU only.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

REPO = "<HOME>/gsoc2026-deeplense-qml/experiments/13_iqp_turbulence_replication"
sys.path.insert(0, REPO)

import numpy as np                                      # noqa: E402
import torch                                            # noqa: E402

import data_lens_class                                  # noqa: E402
import data_lens_r3                                     # noqa: E402
import evaluate_r3                                      # noqa: E402
import metrics                                          # noqa: E402

FIXED_BINS = 50
FIXED_LO, FIXED_HI = 0.0, 1.0
EPS = 1e-12


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outputs-dir", nargs="+", required=True)
    p.add_argument("--results", default=os.path.join(REPO, "results_seeds_extra.jsonl"))
    p.add_argument("--n-holdout", type=int, default=200)
    p.add_argument("--append", action="store_true")
    p.add_argument("--threads", type=int, default=4)
    return p.parse_args(argv)


# ------------------------------------------------------------------------------- metrics
def pdf_js_fixed01(real, gen, bins=FIXED_BINS):
    """JS divergence on a grid that does not move with the arm: `bins` bins on [0, 1]."""
    x = np.clip(np.asarray(real, dtype=np.float64).ravel(), FIXED_LO, FIXED_HI)
    y = np.clip(np.asarray(gen, dtype=np.float64).ravel(), FIXED_LO, FIXED_HI)
    edges = np.linspace(FIXED_LO, FIXED_HI, int(bins) + 1)
    p, _ = np.histogram(x, bins=edges)
    q, _ = np.histogram(y, bins=edges)
    p = p.astype(np.float64) / max(p.sum(), 1.0) + EPS
    q = q.astype(np.float64) / max(q.sum(), 1.0) + EPS
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * (np.log2(p) - np.log2(m)))
                 + 0.5 * np.sum(q * (np.log2(q) - np.log2(m))))


def _encoder(seed):
    """metrics.conv_mmd2's encoder, built exactly as it builds it."""
    gen_t = torch.Generator().manual_seed(int(seed))
    enc = metrics.ConvEncoder()
    with torch.no_grad():
        for p in enc.parameters():
            if p.dim() > 1:
                fan_in = int(np.prod(p.shape[1:]))
                bound = 1.0 / np.sqrt(fan_in)
                p.uniform_(-bound, bound, generator=gen_t)
            else:
                p.zero_()
    enc.eval()
    return enc


def real_side_kernel(real, seed=evaluate_r3.CONV_SEED, n_eval=metrics.N_EVAL):
    """Everything conv_mmd2 would otherwise read off the arm: the normalisation range, the
    encoder, and the median-heuristic sigma -- all from the held-out features alone, so the
    kernel is one fixed object shared by every arm and every seed."""
    rng = np.random.default_rng(seed)
    real_s = metrics._subsample(real, n_eval, rng)
    lo = float(np.min(real_s))
    hi = float(np.max(real_s))
    enc = _encoder(seed)
    f = metrics._features(real_s, enc, lo, hi)
    d = torch.cdist(f, f, p=2)
    iu = torch.triu_indices(d.shape[0], d.shape[0], offset=1)
    med = float(torch.median(d[iu[0], iu[1]]))
    return {"lo": lo, "hi": hi, "enc": enc, "f": f, "sigma": med if med > 0 else 1.0,
            "seed": seed, "n_eval": n_eval}


def conv_mmd2_fixedsigma(kern, gen):
    """Biased MMD^2 with the real-side kernel of `real_side_kernel`.

    The generated set is subsampled with a FRESH rng at the same seed so that it gets the
    same draw metrics.conv_mmd2 would give it (there the rng is advanced by the real-side
    draw first, but at n_eval = 200 = the set size both are no-ops; the call is kept so the
    two agree if either set ever grows)."""
    rng = np.random.default_rng(kern["seed"])
    _ = metrics._subsample(np.zeros((1, 1, 1)), kern["n_eval"], rng)   # match rng advance
    gen_s = metrics._subsample(gen, kern["n_eval"], rng)
    g = metrics._features(gen_s, kern["enc"], kern["lo"], kern["hi"])
    f = kern["f"]
    s = kern["sigma"]

    def k(a, b):
        return torch.exp(-(torch.cdist(a, b, p=2) ** 2) / (2.0 * s * s))

    return float(k(f, f).mean() + k(g, g).mean() - 2.0 * k(f, g).mean())


# -------------------------------------------------------------------------- parameter book
def param_block(meta):
    """(params_shared, params_total_at_N, extras) from a run's own meta.json."""
    n = int(meta.get("n_train", 0))
    script = meta.get("script")
    extra = {"params_generator": None, "params_discriminator": None,
             "params_per_image": None, "params_note": None}

    if script == "train_iqp_r3.py":
        d_core = int(meta.get("d_core", 0))
        d_lat = int(meta.get("d_lat", 0))
        extra["params_per_image"] = d_lat
        extra["params_note"] = "core shared, %d phases free per training image" % d_lat
        return d_core, d_core + n * d_lat, extra

    if script == "train_iqp_affine_r3.py":
        shared = int(meta.get("param_count_shared", 0))
        M = int(meta.get("M", 0))
        extra["params_per_image"] = M
        extra["params_note"] = "theta0 + Phi shared, %d latent coordinates per image" % M
        return shared, shared + n * M, extra

    # the early-stopping protocol's selected checkpoints were written by the SWEEP
    # scripts, which stamp their own name, so both spellings have to dispatch here
    if script in ("baseline_rbm_r3.py", "baseline_rbm_sweep.py"):
        # JointRBM: W (10 x n_hidden) and b_h[40:] are shared; b_v (10) and b_h[:40] are
        # per image (baseline_rbm_r3.py:63-74).
        nh = int(meta.get("n_hidden", 1000))
        n_vis = 10
        n_bh_lat = int(meta.get("d_lat", 50)) - n_vis
        shared = n_vis * nh + (nh - n_bh_lat)
        per_img = n_vis + n_bh_lat
        extra["params_per_image"] = per_img
        extra["params_note"] = ("W and the %d shared hidden biases; %d + %d per-image visible "
                                "and latent-block hidden biases" % (nh - n_bh_lat, n_vis, n_bh_lat))
        return shared, shared + n * per_img, extra

    if script in ("baseline_dcgan_r3.py", "baseline_dcgan_sweep.py"):
        pg = meta.get("param_count")
        pd = meta.get("param_count_discriminator")
        extra["params_generator"] = pg
        extra["params_discriminator"] = pd
        extra["params_per_image"] = 0
        extra["params_note"] = ("generator; the discriminator is training-only and not "
                                "sampled from"
                                + ("" if pd is not None else
                                   " (this run's meta records no discriminator count)"))
        return (None if pg is None else int(pg)), (None if pg is None else int(pg)), extra

    pc = meta.get("param_count")
    return (None if pc is None else int(pc)), (None if pc is None else int(pc)), extra


# ------------------------------------------------------------------------------- scoring
def build_ctx(class_name, n_holdout):
    if class_name:
        data_lens_class.install(class_name)
    d = data_lens_r3.load_r3(data_lens_r3.N_TRAIN_MAX, n_holdout=n_holdout)
    ho = d.holdout_images
    hmt = float(ho.reshape(ho.shape[0], -1).sum(axis=1).mean())
    ho_al = evaluate_r3.align_to_target(ho, hmt)
    return {"holdout": ho, "holdout_aligned": ho_al, "holdout_mean_total": hmt,
            "kern": real_side_kernel(ho), "kern_al": real_side_kernel(ho_al),
            "class_name": class_name}


def score_run(run_dir, ctx):
    meta = evaluate_r3._load_meta(run_dir)
    if meta is None:
        return None
    gen = evaluate_r3._stack(run_dir)
    if gen is None:
        return None
    gen_al = evaluate_r3.align_to_target(gen, ctx["holdout_mean_total"])
    shared, total, extra = param_block(meta)

    row = {
        "kind": "extra_metrics",
        "script": "seeds/rescore_extra_metrics.py",
        "run_dir": os.path.abspath(run_dir),
        "arm": meta.get("arm"), "seed": meta.get("seed"),
        "n_train": int(meta.get("n_train", 0)), "n_gen": int(gen.shape[0]),
        "n_holdout": int(ctx["holdout"].shape[0]),
        "class_name": ctx["class_name"], "train_script": meta.get("script"),
        "pdf_js_fixed01": pdf_js_fixed01(ctx["holdout"], gen),
        "conv_mmd2_fixedsigma": conv_mmd2_fixedsigma(ctx["kern"], gen),
        "conv_mmd2_fixedsigma_aligned": conv_mmd2_fixedsigma(ctx["kern_al"], gen_al),
        "conv_sigma_fixed": float(ctx["kern"]["sigma"]),
        "conv_sigma_fixed_aligned": float(ctx["kern_al"]["sigma"]),
        "params_shared": shared, "params_total_at_N": total,
        "S": meta.get("S"),
        "gen_frac_above_one": float((np.asarray(gen) > 1.0).mean()),
        "gen_frac_below_zero": float((np.asarray(gen) < 0.0).mean()),
    }
    row.update(extra)
    print("[extra] %-46s N=%-3d seed=%s  js01 %.5f  mmd2_fs %.4f / %.4f  "
          "params %s+N*%s=%s  S=%.2f  >1 %.3f%%"
          % (row["arm"], row["n_train"], row["seed"], row["pdf_js_fixed01"],
             row["conv_mmd2_fixedsigma"], row["conv_mmd2_fixedsigma_aligned"],
             row["params_shared"], row["params_per_image"], row["params_total_at_N"],
             float(row["S"] or 0.0), 100.0 * row["gen_frac_above_one"]), flush=True)
    return row


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(int(args.threads))
    t0 = time.time()

    run_dirs = []
    for dd in args.outputs_dir:
        if not os.path.isdir(dd):
            print("[extra] skipping absent %r" % dd)
            continue
        run_dirs.extend(sorted(x for x in glob.glob(os.path.join(dd, "*")) if os.path.isdir(x)))

    # group by class so each class's held-out set is installed once, not once per run
    by_class = {}
    for rd in run_dirs:
        meta = evaluate_r3._load_meta(rd)
        if meta is None:
            continue
        by_class.setdefault(meta.get("class_name"), []).append(rd)

    rows = []
    for cls in sorted(by_class, key=lambda x: (x is not None, str(x))):
        ctx = build_ctx(cls, args.n_holdout)
        print("[extra] class=%r holdout=%s mean total %.4f | fixed sigma %.5f (raw) "
              "%.5f (aligned)" % (cls, ctx["holdout"].shape, ctx["holdout_mean_total"],
                                  ctx["kern"]["sigma"], ctx["kern_al"]["sigma"]), flush=True)
        for rd in by_class[cls]:
            r = score_run(rd, ctx)
            if r is not None:
                rows.append(r)

    mode = "a" if args.append else "w"
    with open(args.results, mode) as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print("[extra] %s %d rows to %s (%.1f s)"
          % ("appended" if args.append else "wrote", len(rows), args.results,
             time.time() - t0), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
