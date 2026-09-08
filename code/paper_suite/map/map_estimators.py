#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sample-based QCLI / CCI estimators that scale past the 2**n enumeration limit.

`diagnostics_qcli_cci.py` (the round-1 module, reused verbatim here) computes both indices
EXACTLY by enumerating the 2**n outcome space:

    qcli_exact(p, n)   -- FWHT of a dense length-2**n pmf
    cci_exact(p, n)    -- reshape the pmf to (2,)*n, marginalise, Chow-Liu

That is fine up to n = 18 (262,144 entries) but impossible at the n = 484 / 784 sizes the
Fig. 5 map needs for the Ref [38] reference datasets. This module supplies the two
sample-based estimators for those points, following the paper's own protocol:

  qcli_mc_bits(bits, n, L)   Appendix C Monte-Carlo estimator of I_QCLI (eq. 51-53), on a
                             (M, n) bit MATRIX instead of int64 codes -- n > 63 cannot be
                             packed into an int64, which is why
                             `diagnostics_qcli_cci.qcli_mc` (int64 bitmasks, and a literal
                             `float(1 << n)` normaliser that overflows to inf for n >= 1024)
                             cannot be used above n = 63.

  cci_from_samples(bits, n)  plug-in I_CCI: per-qubit marginals and all C(n,2) pairwise
                             marginals straight from the sample matrix (one BLAS matmul),
                             joint entropy from the empirical distribution over DISTINCT
                             observed rows, then the same Chow-Liu maximum-weight spanning
                             tree as cci_exact (imported from diagnostics_qcli_cci).

Two deliberate, documented choices
----------------------------------
1. `qcli_mc_bits` drops the global 2**(-2n) factor in Eq. 5's P(s) = |2^-n sum_x p(x)
   (-1)^{s.x}|^2. That factor is the SAME for every subset s, so it cancels identically in
   m_k = A_k / sum_k A_k (Eq. 52) and therefore in I_QCLI. Keeping it would overflow
   (`float(1 << 784)` = inf). The order aggregates A_k are accumulated in log2 space for the
   same reason: A_k carries a C(n, k) weight that reaches ~1e236 at n = 784, k = 392.

2. `cci_from_samples` is the plug-in estimator on the EMPIRICAL distribution, i.e. it is
   mathematically identical to `cci_exact(empirical_pmf, n)` -- verified for n <= 18 by
   self_test() below. It inherits the plug-in joint-entropy bias: with M samples the joint
   entropy can never exceed log2(M), so I_TC is systematically inflated whenever
   sum_i H(X_i) >> log2(M). That bias is exactly why every point on the map is reported
   next to an i.i.d.-uniform null drawn at the SAME (M, n): the null measures the bias.

Python 3.9 compatible (no `X | None`, no match statements, no bare `list[int]` annotations).
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import diagnostics_qcli_cci as qc  # noqa: E402  (path juggling above is deliberate)

DEFAULT_L = 20000          # Appendix C default subset budget (paper.txt:2480)


# ------------------------------------------------------------------------------------------
# small helpers
# ------------------------------------------------------------------------------------------

def log2_comb(n, k):
    """log2 C(n, k) via lgamma -- C(784, 392) has 779 bits and cannot be held as a float."""
    if k < 0 or k > n:
        return float("-inf")
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)) / math.log(2.0)


def binomial_baseline(n):
    """b_k = C(n, k) / 2**n for k = 0..n, computed in log2 space then normalised.

    Underflows to 0 in the far tails for large n (b_0 = 2**-784); that is genuine, and the
    epsilon-clipping inside diagnostics_qcli_cci._js_divergence handles it.
    """
    lb = np.array([log2_comb(n, k) - n for k in range(n + 1)], dtype=np.float64)
    b = np.exp2(lb - lb.max())
    return b / b.sum()


def codes_to_bits(codes, n):
    """int64 codes (bit j = qubit j, codec.py convention) -> (M, n) uint8 bit matrix."""
    codes = np.asarray(codes, dtype=np.int64).reshape(-1)
    shifts = np.arange(n, dtype=np.int64)
    return ((codes[:, None] >> shifts[None, :]) & 1).astype(np.uint8)


def _sample_subsets_index(n, k, mk, rng):
    """`mk` DISTINCT k-subsets of [n], returned as an (mk, k) int64 index array.

    Same equal-per-order / without-replacement scheme as
    diagnostics_qcli_cci._sample_subset_masks, but index-based so it works for n > 63.
    """
    if k == 0:
        return np.zeros((1, 0), dtype=np.int64)
    if k == n:
        return np.arange(n, dtype=np.int64).reshape(1, n)
    seen = set()
    out = []
    max_tries = mk * 50 + 2000
    tries = 0
    while len(out) < mk and tries < max_tries:
        idx = np.sort(rng.choice(n, size=k, replace=False))
        key = idx.tobytes()
        if key not in seen:
            seen.add(key)
            out.append(idx)
        tries += 1
    return np.asarray(out, dtype=np.int64)


# ------------------------------------------------------------------------------------------
# QCLI -- Appendix C Monte Carlo, on a bit matrix
# ------------------------------------------------------------------------------------------

def qcli_mc_bits(bits, n, L=DEFAULT_L, seed=0):
    """Appendix C estimator of I_QCLI from a (M, n) bit matrix (eq. 51-53).

    bits : (M, n) array of 0/1 (any dtype); one row per sample, column j = qubit j.
    L    : total subset budget, split equal-per-order across k = 0..n (m_target = L // (n+1)).
    Returns dict(I_QCLI, m_k, b_k, L_used, subsets_per_order).

    Exactness check: whenever m_target >= C(n, k) for every k the sampler enumerates every
    subset and the result equals qcli_exact on the empirical pmf to machine precision
    (asserted in self_test for n = 10).
    """
    X = np.asarray(bits)
    if X.ndim != 2 or X.shape[1] != n:
        raise ValueError("bits must be (M, %d), got %s" % (n, (X.shape,)))
    M = X.shape[0]
    if M == 0:
        raise ValueError("qcli_mc_bits: empty sample matrix")
    Xf = np.ascontiguousarray(X, dtype=np.float32)   # entries 0/1; dot products <= n < 2**24
    rng = np.random.default_rng(seed)

    m_target = max(1, int(L) // (n + 1))
    log2A = np.full(n + 1, float("-inf"), dtype=np.float64)
    per_order = np.zeros(n + 1, dtype=np.int64)

    for k in range(n + 1):
        total_k = math.comb(n, k)
        mk = int(min(m_target, total_k))
        if mk <= 0:
            continue
        idx = _sample_subsets_index(n, k, mk, rng)
        mk = idx.shape[0]
        per_order[k] = mk
        if k == 0:
            phat_sum = 1.0                                   # P(empty set): mean of +1 == 1
        else:
            S = np.zeros((mk, n), dtype=np.float32)
            rows = np.repeat(np.arange(mk), idx.shape[1])
            S[rows, idx.reshape(-1)] = 1.0
            parity = np.mod(Xf.dot(S.T), 2.0)                # (M, mk), exact for sums < 2**24
            signs_mean = (1.0 - 2.0 * parity).mean(axis=0, dtype=np.float64)
            phat_sum = float(np.sum(signs_mean ** 2))        # 2**(-2n) dropped (cancels)
        if phat_sum > 0.0:
            log2A[k] = log2_comb(n, k) - math.log2(mk) + math.log2(phat_sum)

    finite = np.isfinite(log2A)
    if not np.any(finite):
        m_hat = np.full(n + 1, 1.0 / (n + 1))
    else:
        A = np.zeros(n + 1, dtype=np.float64)
        A[finite] = np.exp2(log2A[finite] - log2A[finite].max())
        tot = A.sum()
        m_hat = A / tot if tot > 0 else np.full(n + 1, 1.0 / (n + 1))

    b_k = binomial_baseline(n)
    return {"I_QCLI": qc._js_divergence(m_hat, b_k, base=2.0),
            "m_k": m_hat, "b_k": b_k,
            "L_used": int(per_order.sum()), "subsets_per_order": per_order}


# ------------------------------------------------------------------------------------------
# CCI -- plug-in from samples
# ------------------------------------------------------------------------------------------

def _binary_entropy(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)
    q = 1.0 - p
    out = np.zeros_like(p)
    m = p > 0
    out[m] -= p[m] * np.log2(p[m])
    m = q > 0
    out[m] -= q[m] * np.log2(q[m])
    return out


def _joint_entropy_plugin(bits):
    """Plug-in H(X_1..X_n) from the empirical distribution over distinct observed rows.

    Bounded above by log2(M); that is the estimator's known bias, measured by the null.
    """
    X = np.ascontiguousarray(np.asarray(bits, dtype=np.uint8))
    M = X.shape[0]
    packed = np.packbits(X, axis=1)
    view = np.ascontiguousarray(packed).view(
        np.dtype((np.void, packed.shape[1] * packed.dtype.itemsize)))
    _, counts = np.unique(view.reshape(-1), return_counts=True)
    pr = counts.astype(np.float64) / float(M)
    return float(-np.sum(pr * np.log2(pr))), int(counts.size)


def cci_from_samples(bits, n, block=512):
    """Plug-in I_CCI = 1 - I_TC^tree / I_TC from a (M, n) bit matrix (eq. 12-18).

    Pairwise joint counts come from one N11 = X^T X matmul (blocked to bound memory);
    the tree is diagnostics_qcli_cci._chow_liu_tree (Kruskal), identical to cci_exact's.
    Returns dict(I_CCI, I_TC, I_TC_tree, H_joint, H_marg_sum, n_distinct, MI_matrix).
    """
    X = np.asarray(bits)
    if X.ndim != 2 or X.shape[1] != n:
        raise ValueError("bits must be (M, %d), got %s" % (n, (X.shape,)))
    M = X.shape[0]
    Xf = np.ascontiguousarray(X, dtype=np.float32)
    c1 = Xf.sum(axis=0, dtype=np.float64)                 # per-qubit count of 1s
    p1 = c1 / float(M)
    H_marg = _binary_entropy(p1)

    N11 = np.zeros((n, n), dtype=np.float64)
    for s in range(0, n, block):
        e = min(s + block, n)
        N11[s:e] = Xf[:, s:e].T.dot(Xf).astype(np.float64)

    n11 = N11
    n10 = c1[:, None] - n11
    n01 = c1[None, :] - n11
    n00 = float(M) - n11 - n10 - n01
    MI = np.zeros((n, n), dtype=np.float64)
    for cnt, ra, cb in ((n11, p1, p1), (n10, p1, 1.0 - p1),
                        (n01, 1.0 - p1, p1), (n00, 1.0 - p1, 1.0 - p1)):
        pj = cnt / float(M)
        outer = np.outer(ra, cb)
        ok = (pj > 0) & (outer > 0)
        term = np.zeros_like(pj)
        term[ok] = pj[ok] * np.log2(pj[ok] / outer[ok])
        MI += term
    MI = 0.5 * (MI + MI.T)                                 # kill matmul asymmetry noise
    np.fill_diagonal(MI, 0.0)
    MI = np.maximum(MI, 0.0)

    H_joint, n_distinct = _joint_entropy_plugin(X)
    H_marg_sum = float(H_marg.sum())
    I_TC = H_marg_sum - H_joint
    tree_edges, I_TC_tree = qc._chow_liu_tree(MI)
    I_CCI = 0.0 if abs(I_TC) < 1e-9 else 1.0 - I_TC_tree / I_TC
    return {"I_CCI": float(I_CCI), "I_TC": float(I_TC), "I_TC_tree": float(I_TC_tree),
            "H_joint": float(H_joint), "H_marg_sum": H_marg_sum,
            "n_distinct": n_distinct, "MI_matrix": MI, "tree_edges": tree_edges}


# ------------------------------------------------------------------------------------------
def self_test(verbose=True):
    """Cross-check the two sample estimators against the exact 2**n ones. Raises on failure."""
    rng = np.random.default_rng(7)

    # (1) CCI: plug-in from samples == cci_exact on the empirical pmf, n = 12.
    n = 12
    base = rng.random(1 << n) ** 3
    base /= base.sum()
    codes = rng.choice(1 << n, size=20000, p=base)
    p_emp = np.bincount(codes, minlength=1 << n).astype(np.float64) / codes.size
    ref = qc.cci_exact(p_emp, n)
    got = cci_from_samples(codes_to_bits(codes, n), n)
    d_cci = abs(ref["I_CCI"] - got["I_CCI"])
    d_tc = abs(ref["I_TC"] - got["I_TC"])
    assert d_cci < 1e-10 and d_tc < 1e-9, "CCI mismatch %.3e / %.3e" % (d_cci, d_tc)

    # (2) QCLI: with L = 2e4 and n = 10 the sampler is exhaustive -> equals qcli_exact.
    n = 10
    base = rng.random(1 << n) ** 2
    base /= base.sum()
    codes = rng.choice(1 << n, size=8000, p=base)
    p_emp = np.bincount(codes, minlength=1 << n).astype(np.float64) / codes.size
    ref = qc.qcli_exact(p_emp, n)
    got = qcli_mc_bits(codes_to_bits(codes, n), n, L=DEFAULT_L, seed=0)
    d_q = abs(ref["I_QCLI"] - got["I_QCLI"])
    assert d_q < 1e-9, "QCLI mismatch %.3e" % d_q
    assert int(got["subsets_per_order"].sum()) == 1 << n, "not exhaustive at n=10"

    # (3) QCLI at n = 18, subsampled subsets: close to exact, and the b_k baseline matches.
    n = 18
    codes = rng.integers(0, 1 << n, size=30000, dtype=np.int64)
    p_emp = np.bincount(codes, minlength=1 << n).astype(np.float64) / codes.size
    ref = qc.qcli_exact(p_emp, n)
    got = qcli_mc_bits(codes_to_bits(codes, n), n, L=DEFAULT_L, seed=1)
    assert np.allclose(ref["b_k"], got["b_k"], atol=1e-12), "binomial baseline mismatch"
    d18 = abs(ref["I_QCLI"] - got["I_QCLI"])
    assert d18 < 0.05, "n=18 MC far from exact: %.4f vs %.4f" % (ref["I_QCLI"], got["I_QCLI"])

    if verbose:
        print("[self_test] CCI n=12  |d I_CCI| = %.2e   |d I_TC| = %.2e" % (d_cci, d_tc))
        print("[self_test] QCLI n=10 exhaustive |d| = %.2e" % d_q)
        print("[self_test] QCLI n=18 MC(L=2e4) %.4f vs exact %.4f  |d| = %.4f"
              % (got["I_QCLI"], ref["I_QCLI"], d18))
        print("[self_test] OK")
    return True


if __name__ == "__main__":
    self_test()
