#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fig. 5 of arXiv 2603.06440 ("Correlation-Complexity Map") recomputed on <user>'s data.

Every point on the map is a (I_QCLI, I_CCI) pair for one dataset in one representation at
one sample count M, paired with an i.i.d.-uniform NULL drawn at the SAME (n, M). The null
is not decoration: both indices are strongly M- and n-dependent finite-sample statistics
(diagnostics_qcli_cci.py module docstring, point 6), so only the displacement of a dataset
from its own null is interpretable.

Points computed
---------------
  lens-18bit    (x, y, z) tuple codes, the round-1 training representation, n = 18
                per-image (M = 4,096 = one image's whole pixel pool) and pooled over the
                100 training images (M = 409,600). Reused verbatim from round 1's
                outputs_lens_pilot/qcli_cci_lens.json (exact 2**18 estimators), averaged
                over the three lens classes -- they agree to <1%.
  lens-10bit    centred 32x32 crop as a pmf over 1024 codes, the round-3 representation,
                n = 10. Both the EXACT pmf (no sampling) and M = 4,096 draws per image.
  d2-18bit      turbulence D2, n = 18. Pooled over the 11 anchors and per-snapshot, at the
                paper's N_t = 5e4 and (new here) at M = 4,096 to match the lens per-image M.
  mnist-16      MNIST binarised after 7x7 mean-pooling to 4x4 (threshold 0.5), n = 16.
  mnist-484     MNIST binarised on the central 22x22 crop, n = 484.
  mnist-784     MNIST binarised at full 28x28 -- this is Ref [38]'s actual definition
                (paper.txt:370, "784-bit binarised MNIST"), n = 784.
  blobs-16      Ref [38]'s binary blobs (paper.txt:608): one of eight fixed 4x4 patterns,
                every pixel independently flipped with probability p. The eight patterns
                are qml_benchmarks.data.spin_blobs.generate_8blobs's config1..config8,
                hard-copied below; p = 0.05 is the value Ref [38]'s text states (the
                package default is 0.01, also computed as a sensitivity row).
  null-curve    I_QCLI of pure i.i.d.-uniform draws as a function of M, for n = 10, 16, 18
                -- the background against which every other number has to be read.

A SECOND, stronger null is attached to the two lens rows (`perm_QCLI` / `perm_CCI`): the
i.i.d.-uniform null does not control for the fact that a lens image's 18-bit code pool is an
EXHAUSTIVE sweep of all 4,096 (kx, ky) cells with exactly one kz each -- the graph of a
function, not a sample from a distribution -- and that alone drives the Walsh spectrum far
from binomial. The structure-matched null keeps the sweep and the exact intensity histogram
and destroys only the spatial arrangement: 18-bit -> permute the kz values across pixels;
10-bit -> permute the pmf values across the 1,024 codes. Whatever survives that permutation
is representation, not lens structure.

n <= 18 uses the exact 2**n estimators (diagnostics_qcli_cci.qcli_exact / cci_exact);
n = 484 / 784 use paper_suite/map/map_estimators.py (Appendix C Monte Carlo with L = 2e4
for QCLI, plug-in from samples for CCI), which self-test against the exact ones.

CPU only, single process, OMP_NUM_THREADS=4. Python 3.9 compatible.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import codec10                       # noqa: E402
import data_lens_r3                  # noqa: E402
import diagnostics_qcli_cci as qc    # noqa: E402
import map_estimators as me          # noqa: E402

OUT_DEFAULT = os.path.join(HERE, "results_map.json")
LENS_R1_JSON = os.path.join(REPO, "outputs_lens_pilot", "qcli_cci_lens.json")
D2_R1_JSON = os.path.join(REPO, "outputs_anchor_fit", "qcli_cci_d2.json")
MNIST_DIR = "<WORK>/data/mnist"

BLOB_CONFIGS = [                                    # qml-benchmarks generate_8blobs, +-1 form
    [[1, 1, -1, -1], [1, 1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1]],
    [[-1, -1, 1, 1], [-1, -1, 1, 1], [-1, -1, -1, -1], [-1, -1, -1, -1]],
    [[-1, -1, -1, -1], [-1, -1, -1, -1], [1, 1, -1, -1], [1, 1, -1, -1]],
    [[-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, 1, 1], [-1, -1, 1, 1]],
    [[-1, -1, -1, -1], [-1, 1, 1, -1], [-1, 1, 1, -1], [-1, -1, -1, -1]],
    [[1, -1, -1, -1], [-1, 1, -1, -1], [-1, -1, 1, -1], [-1, -1, -1, 1]],
    [[-1, -1, -1, 1], [-1, -1, 1, -1], [-1, 1, -1, -1], [1, -1, -1, -1]],
    [[1, -1, -1, 1], [-1, -1, -1, -1], [-1, -1, -1, -1], [1, -1, -1, 1]],
]


# ==========================================================================================
# generic point builders
# ==========================================================================================

def point(dataset, representation, n, M, qcli, cci, qcli_null, cci_null, estimator,
          note="", extra=None):
    row = {"dataset": dataset, "representation": representation, "n": int(n),
           "M": (int(M) if not isinstance(M, str) else M),
           "I_QCLI": float(qcli), "I_CCI": float(cci),
           "null_QCLI": (None if qcli_null is None else float(qcli_null)),
           "null_CCI": (None if cci_null is None else float(cci_null)),
           "estimator": estimator, "note": note}
    if extra:
        row.update(extra)
    return row


def exact_from_codes(codes, n):
    p = np.bincount(np.asarray(codes, dtype=np.int64), minlength=1 << n).astype(np.float64)
    p /= p.sum()
    return qc.qcli_exact(p, n)["I_QCLI"], qc.cci_exact(p, n)["I_CCI"]


def exact_from_pmf(p, n):
    p = np.asarray(p, dtype=np.float64)
    p = p / p.sum()
    return qc.qcli_exact(p, n)["I_QCLI"], qc.cci_exact(p, n)["I_CCI"]


def uniform_null_exact(n, M, rng):
    codes = rng.integers(0, 1 << n, size=int(M), dtype=np.int64)
    return exact_from_codes(codes, n)


def big_from_bits(bits, n, L, seed):
    q = me.qcli_mc_bits(bits, n, L=L, seed=seed)
    c = me.cci_from_samples(bits, n)
    return q["I_QCLI"], c["I_CCI"], c


def uniform_null_big(n, M, L, rng, seed):
    bits = (rng.random((int(M), n)) < 0.5).astype(np.uint8)
    return big_from_bits(bits, n, L, seed)


def mean_std(v):
    a = np.asarray(v, dtype=np.float64)
    return float(a.mean()), float(a.std())


# ==========================================================================================
# datasets
# ==========================================================================================

def load_mnist_images(n_max=70000):
    """(N, 28, 28) float32 in [0, 1] from the raw idx gz files (CVDF mirror)."""
    path = os.path.join(MNIST_DIR, "train-images-idx3-ubyte.gz")
    with gzip.open(path, "rb") as fh:
        blob = fh.read()
    magic, num, rows, cols = np.frombuffer(blob[:16], dtype=">u4")
    if int(magic) != 2051:
        raise ValueError("bad MNIST magic %d in %s" % (magic, path))
    arr = np.frombuffer(blob[16:], dtype=np.uint8).reshape(int(num), int(rows), int(cols))
    return arr[:min(int(n_max), int(num))].astype(np.float32) / 255.0


def mnist_bits(images, mode, threshold=0.5):
    """mode: '16' (7x7 mean-pool to 4x4), '484' (central 22x22), '784' (full 28x28)."""
    if mode == "16":
        pooled = images.reshape(-1, 4, 7, 4, 7).mean(axis=(2, 4))
        return (pooled > threshold).astype(np.uint8).reshape(images.shape[0], 16)
    if mode == "484":
        crop = images[:, 3:25, 3:25]
        return (crop > threshold).astype(np.uint8).reshape(images.shape[0], 484)
    if mode == "784":
        return (images > threshold).astype(np.uint8).reshape(images.shape[0], 784)
    raise ValueError("unknown mnist mode %r" % (mode,))


def binary_blobs_bits(M, p, seed):
    """Ref [38] Experiment 2 (paper.txt:608): pick one of eight 4x4 patterns uniformly, then
    flip each of the 16 pixels independently with probability p."""
    rng = np.random.default_rng(seed)
    base = (np.asarray(BLOB_CONFIGS, dtype=np.int8).reshape(8, 16) + 1) // 2   # -> {0,1}
    which = rng.integers(0, 8, size=int(M))
    X = base[which].astype(np.uint8)
    flips = (rng.random((int(M), 16)) < p).astype(np.uint8)
    return (X ^ flips).astype(np.uint8)


def bits_to_codes(bits):
    n = bits.shape[1]
    shifts = (1 << np.arange(n, dtype=np.int64))
    return (np.asarray(bits, dtype=np.int64) * shifts[None, :]).sum(axis=1)


# ==========================================================================================
def run(args):
    t0 = time.time()
    rows = []
    rng = np.random.default_rng(args.seed)
    log = lambda s: print(s, flush=True)

    # ---------------------------------------------------------------- 1. lens, 18-bit tuples
    log("[1] lens 18-bit tuples (reused from round 1: %s)" % LENS_R1_JSON)
    with open(LENS_R1_JSON) as fh:
        r1 = json.load(fh)
    classes = r1["meta"]["classes"]
    pi_q = np.concatenate([r1["per_image"][c]["I_QCLI"] for c in classes])
    pi_c = np.concatenate([r1["per_image"][c]["I_CCI"] for c in classes])
    pi_qn = np.concatenate([r1["per_image"][c]["I_QCLI_null"] for c in classes])
    pi_cn = np.concatenate([r1["per_image"][c]["I_CCI_null"] for c in classes])
    M_pi = int(r1["per_image"][classes[0]]["M"])
    rows.append(point("lens", "18-bit (x,y,z) tuples, per-image", 18, M_pi,
                      pi_q.mean(), pi_c.mean(), pi_qn.mean(), pi_cn.mean(),
                      "exact 2**18",
                      "mean over %d val images (3 classes pooled); sd %.4f / %.4f"
                      % (pi_q.size, pi_q.std(), pi_c.std()),
                      {"sd_QCLI": float(pi_q.std()), "sd_CCI": float(pi_c.std()),
                       "n_images": int(pi_q.size), "source": "round1 json"}))
    po_q = np.array([r1["pooled"][c]["I_QCLI"] for c in classes])
    po_c = np.array([r1["pooled"][c]["I_CCI"] for c in classes])
    po_qn = np.array([r1["pooled"][c]["I_QCLI_null"] for c in classes])
    po_cn = np.array([r1["pooled"][c]["I_CCI_null"] for c in classes])
    M_po = int(r1["pooled"][classes[0]]["M"])
    rows.append(point("lens", "18-bit (x,y,z) tuples, pooled 100 train", 18, M_po,
                      po_q.mean(), po_c.mean(), po_qn.mean(), po_cn.mean(),
                      "exact 2**18", "mean over the 3 classes (spread %.4f / %.4f)"
                      % (po_q.ptp(), po_c.ptp()), {"source": "round1 json"}))

    # ------------------------------------------- 1b. lens 18-bit structure-matched null
    log("[1b] lens 18-bit permutation null (%d images/class) (%.1fs)"
        % (args.n_perm, time.time() - t0))
    import codec                                                           # noqa: E402
    import data_lens                                                       # noqa: E402
    perm_q, perm_c = [], []
    for cls in classes:
        snaps, _anch, hold = data_lens.load_lens(cls, n_train=2, n_holdout=args.n_perm)
        for k in hold[:args.n_perm]:
            codes = np.asarray(snaps[k].pool_codes, dtype=np.int64)
            kx, ky, kz = codec.decode(codes)
            a, b = exact_from_codes(codec.encode(kx, ky, rng.permutation(kz)), 18)
            perm_q.append(a); perm_c.append(b)
    rows[0]["perm_QCLI"] = float(np.mean(perm_q))
    rows[0]["perm_CCI"] = float(np.mean(perm_c))
    rows[0]["perm_note"] = ("kz permuted across the 4096 pixels: keeps the exhaustive (kx,ky) "
                            "sweep and the exact kz histogram, destroys the spatial layout "
                            "(%d images/class)" % args.n_perm)
    log("     perm null  I_QCLI=%.4f  I_CCI=%.4f" % (rows[0]["perm_QCLI"], rows[0]["perm_CCI"]))

    # ---------------------------------------------------------------- 2. lens, 10-bit crops
    log("[2] lens 10-bit centred-crop pmf (%.1fs)" % (time.time() - t0))
    d = data_lens_r3.load_r3(n_train=100, n_holdout=200)
    ho_pmfs = np.stack([codec10.image_to_pmf(np.clip(d.holdout_crops[i], 0.0, None))
                        for i in range(d.holdout_crops.shape[0])], axis=0)

    q_ex, c_ex, q_sa, c_sa, qn_sa, cn_sa = [], [], [], [], [], []
    qp_ex, cp_ex, qp_sa, cp_sa = [], [], [], []
    M10 = args.m10
    for i in range(ho_pmfs.shape[0]):
        a, b = exact_from_pmf(ho_pmfs[i], 10)
        q_ex.append(a); c_ex.append(b)
        codes = codec10.sample_codes_from_pmf(ho_pmfs[i], M10, rng)
        a, b = exact_from_codes(codes, 10)
        q_sa.append(a); c_sa.append(b)
        a, b = uniform_null_exact(10, M10, rng)
        qn_sa.append(a); cn_sa.append(b)
        pmf_perm = rng.permutation(ho_pmfs[i])          # same histogram, no spatial layout
        a, b = exact_from_pmf(pmf_perm, 10)
        qp_ex.append(a); cp_ex.append(b)
        a, b = exact_from_codes(codec10.sample_codes_from_pmf(pmf_perm, M10, rng), 10)
        qp_sa.append(a); cp_sa.append(b)
    q_unif_pop, c_unif_pop = exact_from_pmf(np.full(1 << 10, 1.0 / (1 << 10)), 10)
    qm, qs = mean_std(q_ex); cm, cs = mean_std(c_ex)
    rows.append(point("lens", "10-bit crop pmf, per-image (exact pmf)", 10, "exact pmf",
                      qm, cm, q_unif_pop, c_unif_pop, "exact 2**10 on the population pmf",
                      "no sampling; null = the exact uniform pmf. sd %.4f / %.4f" % (qs, cs),
                      {"sd_QCLI": qs, "sd_CCI": cs, "n_images": len(q_ex),
                       "perm_QCLI": float(np.mean(qp_ex)), "perm_CCI": float(np.mean(cp_ex)),
                       "perm_note": "pmf values permuted across the 1024 codes"}))
    qm, qs = mean_std(q_sa); cm, cs = mean_std(c_sa)
    qnm, _ = mean_std(qn_sa); cnm, _ = mean_std(cn_sa)
    rows.append(point("lens", "10-bit crop pmf, per-image (M=%d draws)" % M10, 10, M10,
                      qm, cm, qnm, cnm, "exact 2**10 on the empirical pmf",
                      "M/2**n = %.1f: saturated regime, read against the null. sd %.4f / %.4f"
                      % (M10 / 1024.0, qs, cs),
                      {"sd_QCLI": qs, "sd_CCI": cs, "n_images": len(q_sa),
                       "perm_QCLI": float(np.mean(qp_sa)), "perm_CCI": float(np.mean(cp_sa)),
                       "perm_note": "pmf values permuted across the 1024 codes, then resampled"}))

    pooled_pmf = d.train_pmfs.mean(axis=0)
    a, b = exact_from_pmf(pooled_pmf, 10)
    ap, bp = exact_from_pmf(rng.permutation(pooled_pmf), 10)
    rows.append(point("lens", "10-bit crop pmf, pooled 100 train (exact pmf)", 10, "exact pmf",
                      a, b, q_unif_pop, c_unif_pop, "exact 2**10 on the population pmf",
                      "mean of the 100 training crops' pmfs",
                      {"perm_QCLI": ap, "perm_CCI": bp,
                       "perm_note": "pooled pmf values permuted across the 1024 codes"}))
    pooled_codes = np.concatenate([
        codec10.sample_codes_from_pmf(d.train_pmfs[i], M10, rng)
        for i in range(d.train_pmfs.shape[0])])
    a, b = exact_from_codes(pooled_codes, 10)
    an, bn = uniform_null_exact(10, pooled_codes.size, rng)
    rows.append(point("lens", "10-bit crop pmf, pooled 100 train (M=%d draws)"
                      % pooled_codes.size, 10, pooled_codes.size, a, b, an, bn,
                      "exact 2**10 on the empirical pmf", "100 x %d draws" % M10))

    # ---------------------------------------------------------------- 3. turbulence D2
    log("[3] turbulence D2 18-bit (%.1fs)" % (time.time() - t0))
    with open(D2_R1_JSON) as fh:
        r2 = json.load(fh)
    pooled_rows = dict((str(r["M"]), r) for r in r2["pooled"])
    r = pooled_rows["50000"]
    an, bn = uniform_null_exact(18, int(r["n_codes"]), rng)
    rows.append(point("turbulence D2", "18-bit (x,y,|u|) tuples, pooled 11 anchors", 18,
                      int(r["n_codes"]), r["I_QCLI"], r["I_CCI"], an, bn, "exact 2**18",
                      "paper's N_t=5e4 per anchor; QCLI/CCI reused from round 1, null new",
                      {"source": "round1 json + new matched null"}))
    ps = r2["per_snapshot"]
    ps_q = float(np.mean([x["I_QCLI"] for x in ps]))
    ps_c = float(np.mean([x["I_CCI"] for x in ps]))
    an, bn = uniform_null_exact(18, int(ps[0]["n_codes"]), rng)
    row_ps = point("turbulence D2", "18-bit tuples, per-snapshot (N_t=5e4)", 18,
                   int(ps[0]["n_codes"]), ps_q, ps_c, an, bn, "exact 2**18",
                   "mean over t = %s" % [x["t"] for x in ps],
                   {"source": "round1 json + new matched null"})
    rows.append(row_ps)

    if args.with_d2:
        import data_d2                                                    # noqa: E402
        import codec                                                      # noqa: E402
        M50 = int(ps[0]["n_codes"])
        q4, c4, q4p, c4p, q50p, c50p = [], [], [], [], [], []
        for t in (1, 500, 1000):
            snap = data_d2.load_d2_snapshot(t, plane="mid")
            pool = np.asarray(snap.pool_codes, dtype=np.int64)
            kx, ky, kz = codec.decode(pool)
            pool_perm = codec.encode(kx, ky, rng.permutation(kz))   # kz shuffled over the grid
            a, b = exact_from_codes(codec.sample_codes(pool, M_pi, rng), 18)
            q4.append(a); c4.append(b)
            a, b = exact_from_codes(codec.sample_codes(pool_perm, M_pi, rng), 18)
            q4p.append(a); c4p.append(b)
            a, b = exact_from_codes(codec.sample_codes(pool_perm, M50, rng), 18)
            q50p.append(a); c50p.append(b)
        an, bn = uniform_null_exact(18, M_pi, rng)
        rows.append(point("turbulence D2", "18-bit tuples, per-snapshot (M=%d)" % M_pi, 18,
                          M_pi, float(np.mean(q4)), float(np.mean(c4)), an, bn,
                          "exact 2**18",
                          "NEW: M matched to the lens per-image point; mean over t=1,500,1000",
                          {"perm_QCLI": float(np.mean(q4p)), "perm_CCI": float(np.mean(c4p)),
                           "perm_note": "kz permuted over the native grid points"}))
        row_ps["perm_QCLI"] = float(np.mean(q50p))
        row_ps["perm_CCI"] = float(np.mean(c50p))
        row_ps["perm_note"] = ("kz permuted over the native grid points, then subsampled to "
                               "the same M; keeps the |u| histogram, destroys the flow field")
        log("     D2 perm null (M=%d): QCLI=%.4f CCI=%.4f" % (M50, row_ps["perm_QCLI"],
                                                              row_ps["perm_CCI"]))

    # ---------------------------------------------------------------- 4. MNIST
    log("[4] binarized MNIST (%.1fs)" % (time.time() - t0))
    imgs = load_mnist_images(n_max=60000)
    for M in args.mnist_m16:
        bits = mnist_bits(imgs[:M], "16")
        a, b = exact_from_codes(bits_to_codes(bits), 16)
        an, bn = uniform_null_exact(16, M, rng)
        rows.append(point("binarized MNIST", "16-bit (7x7 mean-pool to 4x4, thr 0.5)", 16, M,
                          a, b, an, bn, "exact 2**16",
                          "downsampled stand-in for Ref [38]'s MNIST (which is 784-bit)"))
    for mode, nb in (("484", 484), ("784", 784)):
        M = args.mnist_m_big
        bits = mnist_bits(imgs[:M], mode)
        a, b, cinfo = big_from_bits(bits, nb, args.L, args.seed)
        an, bn, cinfo_n = uniform_null_big(nb, M, args.L, rng, args.seed + 1)
        note = ("Ref [38]'s own definition (paper.txt:370)" if mode == "784"
                else "central 22x22 crop; 484 is Ref [38]'s D-WAVE size, not its MNIST size")
        rows.append(point("binarized MNIST", "%s-bit (thr 0.5)" % mode, nb, M, a, b, an, bn,
                          "QCLI: Appendix C MC L=%d; CCI: plug-in from samples" % args.L,
                          note,
                          {"H_joint": cinfo["H_joint"], "H_marg_sum": cinfo["H_marg_sum"],
                           "null_H_marg_sum": cinfo_n["H_marg_sum"]}))

    # ---------------------------------------------------------------- 5. BinaryBlobs
    log("[5] BinaryBlobs (%.1fs)" % (time.time() - t0))
    for p_flip in args.blob_p:
        M = args.blob_m
        bits = binary_blobs_bits(M, p_flip, args.seed)
        a, b = exact_from_codes(bits_to_codes(bits), 16)
        an, bn = uniform_null_exact(16, M, rng)
        rows.append(point("BinaryBlobs", "16-bit, 8 patterns, flip p=%.2f" % p_flip, 16, M,
                          a, b, an, bn, "exact 2**16",
                          "Ref [38] paper.txt:608 states p=0.05; qml-benchmarks default 0.01"))

    # ---------------------------------------------------------------- 6. null-vs-M curves
    log("[6] i.i.d.-uniform null curves (%.1fs)" % (time.time() - t0))
    curves = {}
    for n in (10, 16, 18):
        pts = []
        for M in args.curve_m:
            if M > 4_000_000:
                continue
            a, b = uniform_null_exact(n, M, rng)
            pts.append({"M": int(M), "I_QCLI": a, "I_CCI": b,
                        "M_over_Cnhalf": M / float(math.comb(n, n // 2))})
        curves[str(n)] = pts
        log("    n=%2d  " % n + "  ".join("M=%d:%.3f" % (p["M"], p["I_QCLI"]) for p in pts))

    out = {"meta": {"generated": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "seed": args.seed, "L_subset_budget": args.L,
                    "wallclock_s": time.time() - t0,
                    "script": os.path.abspath(__file__),
                    "sources": {"lens_18bit": LENS_R1_JSON, "d2_18bit": D2_R1_JSON,
                                "mnist": MNIST_DIR,
                                "blobs": "qml-benchmarks generate_8blobs configs (hard-copied)"}},
           "points": rows, "uniform_null_curves": curves}
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=False, default=float)

    print()
    print("| dataset | representation | n | M | I_QCLI | I_CCI | null QCLI | null CCI "
          "| perm QCLI | perm CCI |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print("| %s | %s | %d | %s | %.4f | %.4f | %s | %s | %s | %s |"
              % (r["dataset"], r["representation"], r["n"], r["M"], r["I_QCLI"], r["I_CCI"],
                 "-" if r["null_QCLI"] is None else "%.4f" % r["null_QCLI"],
                 "-" if r["null_CCI"] is None else "%.4f" % r["null_CCI"],
                 "%.4f" % r["perm_QCLI"] if "perm_QCLI" in r else "-",
                 "%.4f" % r["perm_CCI"] if "perm_CCI" in r else "-"))
    print("\n[done] wrote %s (%.1fs)" % (args.out, time.time() - t0))
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=OUT_DEFAULT)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--L", type=int, default=me.DEFAULT_L)
    p.add_argument("--m10", type=int, default=4096)
    p.add_argument("--mnist-m16", type=int, nargs="*", default=[10000, 60000])
    p.add_argument("--mnist-m-big", type=int, default=10000)
    p.add_argument("--blob-m", type=int, default=10000)
    p.add_argument("--blob-p", type=float, nargs="*", default=[0.05, 0.01])
    p.add_argument("--curve-m", type=int, nargs="*",
                   default=[1000, 4096, 10000, 50000, 100000, 409600, 1000000])
    p.add_argument("--n-perm", type=int, default=50,
                   help="images PER CLASS for the 18-bit structure-matched permutation null")
    p.add_argument("--with-d2", action="store_true",
                   help="also recompute a D2 per-snapshot point at M=4096 from the D2 cache")
    p.add_argument("--quick", action="store_true", help="tiny smoke run")
    a = p.parse_args(argv)
    if a.quick:
        a.n_perm = 3
        a.m10 = 1024
        a.mnist_m16 = [2000]
        a.mnist_m_big = 500
        a.blob_m = 2000
        a.curve_m = [1000, 10000]
        a.L = 2000
    return a


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
