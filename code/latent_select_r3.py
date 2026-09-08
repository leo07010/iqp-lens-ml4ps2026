#!/usr/bin/env python3
"""Round-3 latent-block selection (docs/IQP_LENS_ROUND3_PREREGISTRATION.md, "Latent block").

The IQP model has 1,024 trainable Z-string coefficients at n = 10 (every subset). 50 of them
are declared "the latent block": the only ones a per-image copy exists for; the other 974 are
shared. Which 50 is a real design lever, so round 3 fixes TWO candidate rules in advance and
runs both:

  primary  (lowvar50)  the 50 subsets with |s| <= 2 whose Walsh coefficients vary MOST across
                       a pool of 2,000 training images that is DISJOINT from every N
                       (data_lens_r3's latent pool). "lowvar" is short for low-ORDER,
                       high-VARIANCE -- the name matches analysis_lens/latent_recommended_
                       lowvar50.npy, which applied the same rule at n = 12 in round 2.
  ablation (hamming50) the paper's Hamming-first 50: the empty set, the 10 singletons and the
                       first 39 pairs in itertools.combinations order, i.e. exactly
                       iqp_exact.subsets_upto(10, 2)[:50].

At n = 10 there are only 1 + 10 + 45 = 56 subsets with |s| <= 2, so the primary rule keeps 50
of 56 and the two sets are far from disjoint -- the ablation is a test of ORDERING within the
low-order block, not of low-order-vs-high-order. That is a property of n = 10, stated here so
the P12 prediction is read with it in mind.

Selection uses ONLY the disjoint pool: no training image and no held-out image is touched.

Writes  r3_latent_lowvar50.npy , r3_latent_hamming50.npy  (int64 bitmasks) and
        r3_latent_select.json   (variances, order histograms, membership).

Python 3.9 compatible.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

import data_lens_r3
from iqp_exact import fwht, popcount_table, subset_masks

N_QUBITS = 10
D_LAT = 50
ORDER_CUT = 2
LOWVAR_FILE = "r3_latent_lowvar50.npy"
HAMMING_FILE = "r3_latent_hamming50.npy"
JSON_FILE = "r3_latent_select.json"


def walsh_coeffs(pmfs):
    """(m, 1024) Walsh coefficients <Z_s> of m pmfs -- fwht of each row (iqp_exact.fwht)."""
    p = torch.as_tensor(np.asarray(pmfs, dtype=np.float64))
    return fwht(p).numpy()


def lowvar50(pmfs, d_lat=D_LAT, order_cut=ORDER_CUT, n_qubits=N_QUBITS):
    """The d_lat subsets with |s| <= order_cut of largest cross-image variance, ranked."""
    W = walsh_coeffs(pmfs)
    var = W.var(axis=0)
    pc = popcount_table(n_qubits).astype(np.int64)
    cand = np.flatnonzero(pc <= int(order_cut))
    order = cand[np.argsort(var[cand])[::-1]]
    if order.size < d_lat:
        raise ValueError("only %d subsets with |s| <= %d, need %d" % (order.size, order_cut, d_lat))
    return np.asarray(order[:d_lat], dtype=np.int64), var, pc


def hamming50(d_lat=D_LAT, n_qubits=N_QUBITS):
    """The paper's Hamming-first block: subsets_upto(n, 2)[:d_lat] as bitmasks."""
    masks = subset_masks(n_qubits, 2)
    if masks.size < d_lat:
        raise ValueError("subsets_upto(%d, 2) has only %d entries" % (n_qubits, masks.size))
    return np.asarray(masks[:d_lat], dtype=np.int64)


def describe(mask, n_qubits=N_QUBITS):
    """Human-readable name of a bitmask: bit j < 5 is a ky bit, bit j >= 5 a kx bit
    (codec10: code = (kx << 5) | ky)."""
    bits = [b for b in range(n_qubits) if (int(mask) >> b) & 1]
    if not bits:
        return "empty"
    return "+".join(("ky%d" % b) if b < 5 else ("kx%d" % (b - 5)) for b in bits)


def order_hist(masks, n_qubits=N_QUBITS):
    pc = popcount_table(n_qubits).astype(np.int64)
    return np.bincount(pc[np.asarray(masks, dtype=np.int64)], minlength=n_qubits + 1).tolist()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Round-3 latent-block selection.")
    p.add_argument("--n-pool", type=int, default=data_lens_r3.N_LATENT_POOL)
    p.add_argument("--d-lat", type=int, default=D_LAT)
    p.add_argument("--order-cut", type=int, default=ORDER_CUT)
    p.add_argument("--out-dir", default=os.path.dirname(os.path.abspath(__file__)))
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    pmfs = data_lens_r3.latent_pool_pmfs(args.n_pool)
    print("[latent_select_r3] pool: %d images (disjoint from every N), pmf shape %s"
          % (pmfs.shape[0], (pmfs.shape,)))

    lv, var, pc = lowvar50(pmfs, d_lat=args.d_lat, order_cut=args.order_cut)
    hm = hamming50(d_lat=args.d_lat)

    n_low_order = int((pc <= args.order_cut).sum())
    inter = sorted(set(lv.tolist()) & set(hm.tolist()))

    print("[latent_select_r3] subsets with |s| <= %d at n=%d : %d (so the primary rule keeps "
          "%d of them)" % (args.order_cut, N_QUBITS, n_low_order, args.d_lat))
    print("[latent_select_r3] lowvar50  |s| histogram (|s|=0..%d): %s"
          % (N_QUBITS, order_hist(lv)))
    print("[latent_select_r3] hamming50 |s| histogram (|s|=0..%d): %s"
          % (N_QUBITS, order_hist(hm)))
    print("[latent_select_r3] |lowvar50 ^ hamming50| = %d of %d" % (len(inter), args.d_lat))
    print("[latent_select_r3] lowvar50 masks  : %s" % (lv.tolist(),))
    print("[latent_select_r3] hamming50 masks : %s" % (hm.tolist(),))
    print("[latent_select_r3] lowvar50 top-12 by variance: %s"
          % ([describe(m) for m in lv[:12]],))
    print("[latent_select_r3] variance of the kept 50: min=%.4g max=%.4g ; of the dropped "
          "|s|<=%d ones: %s"
          % (float(var[lv].min()), float(var[lv].max()), args.order_cut,
             ["%.4g" % v for v in sorted(var[np.setdiff1d(np.flatnonzero(pc <= args.order_cut), lv)],
                                         reverse=True)]))

    out_dir = args.out_dir
    np.save(os.path.join(out_dir, LOWVAR_FILE), lv)
    np.save(os.path.join(out_dir, HAMMING_FILE), hm)
    payload = {
        "n_qubits": N_QUBITS,
        "d_lat": int(args.d_lat),
        "order_cut": int(args.order_cut),
        "n_pool": int(pmfs.shape[0]),
        "pool_is_disjoint_from_training": True,
        "n_subsets_with_order_le_cut": n_low_order,
        "lowvar50": {
            "rule": "|s| <= %d, ranked by cross-image variance of <Z_s> over the pool, top %d"
                    % (args.order_cut, args.d_lat),
            "masks": [int(m) for m in lv.tolist()],
            "order_histogram": order_hist(lv),
            "members": [{"mask": int(m), "order": int(pc[m]), "desc": describe(int(m)),
                         "var": float(var[m])} for m in lv],
        },
        "hamming50": {
            "rule": "iqp_exact.subsets_upto(%d, 2)[:%d] (empty + 10 singles + %d pairs)"
                    % (N_QUBITS, args.d_lat, args.d_lat - 11),
            "masks": [int(m) for m in hm.tolist()],
            "order_histogram": order_hist(hm),
            "members": [{"mask": int(m), "order": int(pc[m]), "desc": describe(int(m)),
                         "var": float(var[m])} for m in hm],
        },
        "intersection_size": len(inter),
        "files": {"lowvar50": LOWVAR_FILE, "hamming50": HAMMING_FILE},
    }
    with open(os.path.join(out_dir, JSON_FILE), "w") as fh:
        json.dump(payload, fh, indent=2)
    print("[latent_select_r3] wrote %s, %s, %s to %s"
          % (LOWVAR_FILE, HAMMING_FILE, JSON_FILE, out_dir))
    return 0


def load_latent_masks(name, out_dir=None):
    """'lowvar' / 'hamming' -> (50,) int64 bitmasks, from the files main() wrote."""
    out_dir = out_dir if out_dir else os.path.dirname(os.path.abspath(__file__))
    fname = {"lowvar": LOWVAR_FILE, "hamming": HAMMING_FILE}.get(str(name))
    if fname is None:
        raise ValueError("--latent must be 'lowvar' or 'hamming', got %r" % (name,))
    path = os.path.join(out_dir, fname)
    if not os.path.isfile(path):
        raise SystemExit("latent file %s missing -- run `python3 latent_select_r3.py` first" % path)
    masks = np.asarray(np.load(path), dtype=np.int64)
    if masks.size != D_LAT:
        raise ValueError("%s holds %d masks, expected %d" % (path, masks.size, D_LAT))
    if masks.min() < 0 or masks.max() >= (1 << N_QUBITS):
        raise ValueError("%s holds masks outside [0, 2^%d)" % (path, N_QUBITS))
    if len(set(masks.tolist())) != masks.size:
        raise ValueError("%s holds duplicate masks" % path)
    return masks


if __name__ == "__main__":
    raise SystemExit(main())
