"""QCLI/CCI diagnostics on the real D2 turbulence dataset (pre-registration item P5,
missed by submit_d2_main.sbatch's main run).

Uses diagnostics_qcli_cci.py's EXACT (2^18-enumeration) estimators
(diagnose_codes / qcli_exact / cci_exact) on native-grid tuple codes from
data_d2.load_d2_snapshot (one anchor at a time -- never data_d2.load_d2(), which would
also materialize the 200 held-out tau this script never needs):

  (1) pooled   : the 11 anchors' pool_codes, M drawn WITH replacement (codec.sample_codes)
                 from EACH anchor and concatenated, for M in {1e3, 1e4, 5e4, 1e5, "all"
                 (no subsampling, full ~101k-point native pool per anchor)}.
  (2) per-snapshot: anchors t=1, 500, 1000 individually, M=5e4.
  (3) reference: iid-uniform 18-bit codes (seed matches --seed), one draw per M in (1)
                 with the SAME total sample count as that M's pooled draw, to show the
                 finite-sample floor documented in diagnostics_qcli_cci.py's module
                 docstring point 6 (I_QCLI for genuine iid noise does NOT -> 0 once M is
                 comparable to C(18,9) = 48620 -- exactly the M=5e4 regime used here).

Outputs outputs_anchor_fit/qcli_cci_d2.json (all three tables) and prints them to stdout
(the sbatch step redirects this to logs_d2/qcli_cci_d2.log).

Python 3.9 compatible (no `X | None`, no match statements).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

import codec
import data_d2
import diagnostics_qcli_cci as qc

REPO = os.path.dirname(os.path.abspath(__file__))
OUT_DEFAULT = os.path.join(REPO, "outputs_anchor_fit", "qcli_cci_d2.json")
N_QUBITS = 18
M_LIST_FULL = [1000, 10000, 50000, 100000, "all"]
M_LIST_QUICK = [1000]
PER_SNAPSHOT_TS = [1, 500, 1000]
PER_SNAPSHOT_M = 50000


# --------------------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true", help="only M=1000, login-node smoke")
    p.add_argument("--plane", type=str, default="mid")
    p.add_argument("--out", type=str, default=OUT_DEFAULT)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def load_anchor_pools(plane):
    """{t: pool_codes int64 array} for every t in data_d2.ANCHOR_TS, loaded one snapshot at
    a time (load_d2_snapshot) -- avoids materializing the 200 held-out tau."""
    pools = {}
    for t in data_d2.ANCHOR_TS:
        snap = data_d2.load_d2_snapshot(t, plane=plane)
        pools[t] = np.asarray(snap.pool_codes, dtype=np.int64)
    return pools


def draw_m(pool_codes, m, rng):
    if m == "all":
        return pool_codes
    return codec.sample_codes(pool_codes, int(m), rng)


def pooled_codes(pools, m, rng):
    parts = [draw_m(pools[t], m, rng) for t in data_d2.ANCHOR_TS]
    return np.concatenate(parts)


# --------------------------------------------------------------------------------------
def run(args):
    t0 = time.time()
    m_list = M_LIST_QUICK if args.quick else M_LIST_FULL
    rng = np.random.default_rng(args.seed)

    print("[load] anchors=%s plane=%s" % (data_d2.ANCHOR_TS, args.plane), flush=True)
    pools = load_anchor_pools(args.plane)
    sizes = dict((t, int(pools[t].size)) for t in pools)
    print("[load] pool sizes: %s (%.1fs)" % (sizes, time.time() - t0), flush=True)

    results = {"pooled": [], "per_snapshot": [], "uniform_ref": []}

    print("\n=== (1) pooled (11 anchors, M drawn from each, concatenated) ===")
    print("%-10s %12s %10s %10s" % ("M", "n_codes", "I_QCLI", "I_CCI"))
    pooled_n = {}
    for m in m_list:
        codes = pooled_codes(pools, m, rng)
        pooled_n[m] = int(codes.size)
        d = qc.diagnose_codes(codes, n=N_QUBITS)
        row = dict(M=m, n_codes=int(codes.size), I_QCLI=d["I_QCLI"], I_CCI=d["I_CCI"])
        results["pooled"].append(row)
        print("%-10s %12d %10.6f %10.6f" % (str(m), row["n_codes"], d["I_QCLI"], d["I_CCI"]), flush=True)

    print("\n=== (2) per-snapshot (M=%d) ===" % PER_SNAPSHOT_M)
    print("%-10s %10s %10s" % ("t", "I_QCLI", "I_CCI"))
    for t in PER_SNAPSHOT_TS:
        codes = draw_m(pools[t], PER_SNAPSHOT_M, rng)
        d = qc.diagnose_codes(codes, n=N_QUBITS)
        row = dict(t=t, M=PER_SNAPSHOT_M, n_codes=int(codes.size), I_QCLI=d["I_QCLI"], I_CCI=d["I_CCI"])
        results["per_snapshot"].append(row)
        print("%-10s %10.6f %10.6f" % (t, d["I_QCLI"], d["I_CCI"]), flush=True)

    print("\n=== (3) reference: iid-uniform 18-bit codes (seed=%d), matched n_codes ===" % args.seed)
    print("%-10s %12s %10s" % ("M", "n_codes", "I_QCLI(uniform)"))
    for m in m_list:
        n_draw = pooled_n[m]
        codes = rng.integers(0, 1 << N_QUBITS, size=n_draw, dtype=np.int64)
        p = codec.codes_to_pmf(codes)
        q = qc.qcli_exact(p, N_QUBITS)
        row = dict(M=m, n_codes=n_draw, I_QCLI=q["I_QCLI"])
        results["uniform_ref"].append(row)
        print("%-10s %12d %10.6f" % (str(m), n_draw, q["I_QCLI"]), flush=True)

    results["meta"] = dict(anchors=data_d2.ANCHOR_TS, plane=args.plane, seed=args.seed,
                            quick=args.quick, wallclock_s=time.time() - t0)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2, sort_keys=True, default=float)
    print("\n[done] wrote %s (%.1fs)" % (args.out, time.time() - t0), flush=True)


def main(argv=None):
    args = parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
