#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""QCLI / CCI diagnostics on the DeepLense Model II images (pre-registration prediction P4).

Uses diagnostics_qcli_cci.py's EXACT (2^18-enumeration) estimators (diagnose_codes ->
qcli_exact + cci_exact) on the 18-bit codes data_lens.py builds from each image's 4,096
pixels. Two scopes per class, each with an i.i.d.-uniform null at the SAME sample count
(the finite-sample floor is not negligible: I_QCLI for genuine uniform noise does not go
to 0 until M is large compared with C(18, 9) = 48,620, and per-image M here is only
4,096 -- see diagnostics_qcli_cci.py's module docstring, point 6):

  per-image : each of the 200 held-out val images on its own (M = 4,096 = the whole pixel
              pool of one image, no subsampling), reported as mean +/- std over images.
              This is the regime P4 is about.
  pooled    : the N = 100 training images' pools concatenated (M = 409,600 codes), one
              number per class -- the analogue of qcli_cci_d2.py's pooled row.

Writes a JSON with every per-image value and prints one markdown table covering all
three classes.

Python 3.9 compatible (no `X | None`, no match statements).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

import codec
import data_lens
import diagnostics_qcli_cci as qc

REPO = os.path.dirname(os.path.abspath(__file__))
OUT_DEFAULT = os.path.join(REPO, "outputs_lens", "qcli_cci_lens.json")
N_QUBITS = 18


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="QCLI/CCI on the lens images (pre-registration P4).")
    p.add_argument("--quick", action="store_true",
                   help="login-node smoke: 3 images per class, pooled over 3 training images")
    p.add_argument("--n-images", type=int, default=data_lens.N_HOLDOUT,
                   help="per-image scope: how many held-out images per class")
    p.add_argument("--n-train", type=int, default=data_lens.N_TRAIN_MAX,
                   help="pooled scope: how many training images per class")
    p.add_argument("--classes", nargs="*", default=None, help="default: all three")
    p.add_argument("--out", type=str, default=OUT_DEFAULT)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def _mean_std(vals):
    a = np.asarray(vals, dtype=np.float64)
    if a.size == 0:
        return float("nan"), float("nan")
    return float(a.mean()), float(a.std())


def per_image_scope(snapshots, ids, rng):
    """(qcli list, cci list, uniform-null qcli list, uniform-null cci list, M)."""
    q_data, c_data, q_null, c_null = [], [], [], []
    m = None
    for k in ids:
        codes = np.asarray(snapshots[k].pool_codes, dtype=np.int64)
        m = int(codes.size)
        d = qc.diagnose_codes(codes, n=N_QUBITS)
        q_data.append(d["I_QCLI"])
        c_data.append(d["I_CCI"])
        u = rng.integers(0, 1 << N_QUBITS, size=m, dtype=np.int64)
        du = qc.diagnose_codes(u, n=N_QUBITS)
        q_null.append(du["I_QCLI"])
        c_null.append(du["I_CCI"])
    return q_data, c_data, q_null, c_null, m


def pooled_scope(snapshots, ids, rng):
    codes = np.concatenate([np.asarray(snapshots[k].pool_codes, dtype=np.int64) for k in ids])
    d = qc.diagnose_codes(codes, n=N_QUBITS)
    u = rng.integers(0, 1 << N_QUBITS, size=codes.size, dtype=np.int64)
    du = qc.diagnose_codes(u, n=N_QUBITS)
    return d, du, int(codes.size)


def main(argv=None):
    args = parse_args(argv)
    t0 = time.time()
    rng = np.random.default_rng(args.seed)

    classes = args.classes if args.classes else data_lens.get_class_names()
    n_images = 3 if args.quick else int(args.n_images)
    n_train = 3 if args.quick else int(args.n_train)

    results = {"meta": {"classes": list(classes), "n_images": n_images, "n_train": n_train,
                        "seed": args.seed, "quick": bool(args.quick),
                        "estimator": "exact (2^18 enumeration)"},
               "per_image": {}, "pooled": {}}
    table_rows = []

    for cls in classes:
        snapshots, anchors, holdout = data_lens.load_lens(
            cls, n_train=n_train, n_holdout=n_images
        )
        print("[qcli_cci_lens] %s: %d train, %d held-out" % (cls, len(anchors), len(holdout)),
              flush=True)

        q_d, c_d, q_n, c_n, m_img = per_image_scope(snapshots, holdout, rng)
        results["per_image"][cls] = {
            "M": m_img, "n_images": len(holdout),
            "I_QCLI": [float(v) for v in q_d], "I_CCI": [float(v) for v in c_d],
            "I_QCLI_null": [float(v) for v in q_n], "I_CCI_null": [float(v) for v in c_n],
        }
        qm, qs = _mean_std(q_d)
        cm, cs = _mean_std(c_d)
        qnm, qns = _mean_std(q_n)
        cnm, cns = _mean_std(c_n)
        table_rows.append((cls, "per-image (%d val)" % len(holdout), len(holdout), m_img,
                           "%.4f ± %.4f" % (qm, qs), "%.4f ± %.4f" % (cm, cs)))
        table_rows.append((cls, "per-image uniform null", len(holdout), m_img,
                           "%.4f ± %.4f" % (qnm, qns), "%.4f ± %.4f" % (cnm, cns)))
        print("  per-image  I_QCLI=%.4f±%.4f  I_CCI=%.4f±%.4f   (null %.4f±%.4f / %.4f±%.4f)"
              % (qm, qs, cm, cs, qnm, qns, cnm, cns), flush=True)

        d, du, m_pool = pooled_scope(snapshots, anchors, rng)
        results["pooled"][cls] = {
            "M": m_pool, "n_train": len(anchors),
            "I_QCLI": float(d["I_QCLI"]), "I_CCI": float(d["I_CCI"]),
            "I_QCLI_null": float(du["I_QCLI"]), "I_CCI_null": float(du["I_CCI"]),
        }
        table_rows.append((cls, "pooled (%d train)" % len(anchors), len(anchors), m_pool,
                           "%.4f" % d["I_QCLI"], "%.4f" % d["I_CCI"]))
        table_rows.append((cls, "pooled uniform null", len(anchors), m_pool,
                           "%.4f" % du["I_QCLI"], "%.4f" % du["I_CCI"]))
        print("  pooled     I_QCLI=%.4f  I_CCI=%.4f   (null %.4f / %.4f)"
              % (d["I_QCLI"], d["I_CCI"], du["I_QCLI"], du["I_CCI"]), flush=True)

    results["meta"]["wallclock_s"] = time.time() - t0
    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2, sort_keys=True, default=float)

    print()
    print("| Class | Scope | # images | M (codes) | I_QCLI | I_CCI |")
    print("|---|---|---|---|---|---|")
    for row in table_rows:
        print("| %s | %s | %d | %d | %s | %s |" % row)
    print()
    print("[qcli_cci_lens] wrote %s (%.1fs)" % (args.out, time.time() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
