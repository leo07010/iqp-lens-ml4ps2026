#!/usr/bin/env python3
"""CPU-seconds unit tests for diagnostics_qcli_cci.py (QCLI / CCI). Login-node safe.

Run: /usr/bin/python3 test_diagnostics.py

See diagnostics_qcli_cci.py's module docstring for the documented design decisions this test
suite relies on (in particular: m_k includes s=empty-set at k=0, and the resulting universal
P(empty set) = 2^-2n floor, which is why the "uniform pmf" and "even-parity" cases below do NOT
give the naively-expected I_QCLI=0 / m_n=1 -- the exact, verified numbers are asserted instead).
"""
import math
import time

import numpy as np

from diagnostics_qcli_cci import (
    fwht_np, qcli_exact, cci_exact, qcli_mc,
)


def brute_force_P(p, n):
    """O(4^n) double loop over Eq. 5, used only at small n as an independent cross-check of
    the FWHT-based qcli_exact implementation."""
    P = np.zeros(1 << n, dtype=np.float64)
    for s in range(1 << n):
        acc = 0.0
        for x in range(1 << n):
            parity = bin(s & x).count("1") & 1
            acc += p[x] * (1.0 if parity == 0 else -1.0)
        P[s] = (acc / (1 << n)) ** 2
    return P


def brute_force_pairwise_marginal(p, n, i, j):
    pij = np.zeros((2, 2), dtype=np.float64)
    for x in range(1 << n):
        bi = (x >> i) & 1
        bj = (x >> j) & 1
        pij[bi, bj] += p[x]
    return pij


def test_fwht_matches_bruteforce_P_n6():
    rng = np.random.default_rng(0)
    n = 6
    raw = rng.random(1 << n)
    p = raw / raw.sum()

    P_bf = brute_force_P(p, n)
    w = fwht_np(p)
    P_fast = (w / (1 << n)) ** 2

    err = float(np.max(np.abs(P_fast - P_bf)))
    print("[fwht vs bruteforce P(s), n=6] max abs err = %.3e" % err)
    assert err < 1e-12, err


def test_cci_mi_matches_bruteforce_n6():
    rng = np.random.default_rng(1)
    n = 6
    raw = rng.random(1 << n)
    p = raw / raw.sum()
    res = cci_exact(p, n)

    max_err = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            pij_bf = brute_force_pairwise_marginal(p, n, i, j)
            pi_bf = pij_bf.sum(axis=1)
            pj_bf = pij_bf.sum(axis=0)
            mi_bf = 0.0
            for a in range(2):
                for b in range(2):
                    if pij_bf[a, b] > 0:
                        mi_bf += pij_bf[a, b] * math.log2(pij_bf[a, b] / (pi_bf[a] * pj_bf[b]))
            max_err = max(max_err, abs(mi_bf - res["MI_matrix"][i, j]))
    print("[cci MI matrix vs bruteforce, n=6] max abs err = %.3e" % max_err)
    assert max_err < 1e-12, max_err


def test_uniform_pmf_exact_population():
    n = 6
    p = np.full(1 << n, 1.0 / (1 << n))
    q = qcli_exact(p, n)
    c = cci_exact(p, n)
    print("[uniform pmf, n=%d] m_k = %s" % (n, np.round(q["m_k"], 6)))
    print("[uniform pmf, n=%d] I_QCLI = %.6f  I_TC = %.3e  I_CCI = %.6f"
          % (n, q["I_QCLI"], c["I_TC"], c["I_CCI"]))

    # I_TC is EXACTLY 0 for the exact population-uniform pmf (each bit marginally fair,
    # H(X_i)=1; joint exactly uniform over 2^n outcomes, H(X)=n; I_TC = n - n = 0). We define
    # I_CCI := 0 by convention in that case (decision 2 in the module docstring) rather than a
    # literal 0/0 = NaN.
    assert abs(c["I_TC"]) < 1e-9, c["I_TC"]
    assert c["I_CCI"] == 0.0, c["I_CCI"]

    # Verified analytically (decision 1 in the module docstring): P(empty set) = 2^-2n for ANY
    # valid pmf, and for the *exact* population-uniform pmf, P(s) = 0 for every s != empty set
    # (standard character-sum orthogonality). So m_0 = 1 and m_k = 0 for k > 0 EXACTLY. This is
    # a real, non-obvious consequence of Eq. 5-6 as literally stated -- NOT an implementation
    # bug, and NOT the same as the (false) naive claim "I_QCLI = 0 for uniform data".
    assert abs(q["m_k"][0] - 1.0) < 1e-12, q["m_k"]
    assert float(np.max(np.abs(q["m_k"][1:]))) < 1e-12, q["m_k"]
    # Consequently I_QCLI is LARGE here (m is a point mass at k=0, far from the smooth binomial
    # b), not small -- see test_near_uniform_empirical_pmf below for the practically-meaningful
    # "null" check the paper actually describes (finite-M empirical pmf, not exact population).
    assert q["I_QCLI"] > 0.5, q["I_QCLI"]


def test_near_uniform_empirical_pmf_small_qcli():
    # The paper's Eq. 4 pmf is itself always a *finite-M empirical* pmf, and II.3 claims
    # "if the data have no structured parity content beyond iid noise, m ~ b in expectation,
    # I_QCLI -> 0". NON-OBVIOUS FINDING (swept numerically, not in the paper): this only holds
    # for M well BELOW roughly C(n, n/2) -- the number of subsets at the middle order. Above
    # that crossover, I_QCLI for genuinely-iid-uniform draws *increases* with M, converging
    # toward the degenerate exact-population-uniform value from test_uniform_pmf_exact_population
    # (m -> [1,0,...,0]) rather than -> 0, because for s != empty set,
    # E[P(s)] ~ 2^-2n / M (empirical-sampling variance) while the order-k AGGREGATE
    # sum_{|s|=k} P(s) ~ C(n,k) * 2^-2n / M is being compared against the M-independent
    # P(empty set) = 2^-2n floor -- so the k>=1 power only dominates the floor when
    # C(n,k)/M >> 1, i.e. M << C(n, n/2). For the real turbulence application (n=18,
    # M=N_t=5e4), C(18,9)=48620 is comparable to M=5e4 -- right at this crossover, so I_QCLI
    # there may be sensitive to the exact M/n_t choice; flagged for the parent orchestrator.
    # Numerically (n=8, C(8,4)=70): M=20 -> I_QCLI~0.04 (small, as II.3 claims); M=20000 ->
    # I_QCLI~0.94 (large, degenerate-uniform regime). We pick M well under the crossover here.
    n = 8
    rng = np.random.default_rng(2)
    M = 20
    x = rng.integers(0, 1 << n, size=M)
    p = np.bincount(x, minlength=1 << n).astype(np.float64) / M
    q = qcli_exact(p, n)
    print("[near-uniform empirical pmf, n=%d, M=%d (<< C(n,n/2)=%d)] I_QCLI = %.6f"
          % (n, M, math.comb(n, n // 2), q["I_QCLI"]))
    assert q["I_QCLI"] < 0.15, q["I_QCLI"]


def test_independent_nonuniform_bits():
    n = 8
    rng = np.random.default_rng(3)
    probs = rng.uniform(0.2, 0.8, size=n)
    p = np.ones(1 << n, dtype=np.float64)
    for x in range(1 << n):
        val = 1.0
        for i in range(n):
            bit = (x >> i) & 1
            val *= probs[i] if bit else (1.0 - probs[i])
        p[x] = val
    assert abs(p.sum() - 1.0) < 1e-9

    q = qcli_exact(p, n)
    c = cci_exact(p, n)
    print("[independent non-uniform bits, n=%d] I_TC = %.3e (expect ~0)" % (n, c["I_TC"]))
    print("[independent non-uniform bits, n=%d] max|MI| = %.3e"
          % (n, float(np.max(np.abs(c["MI_matrix"])))))
    print("[independent non-uniform bits, n=%d] I_QCLI = %.6f (printed only, not asserted -- "
          "see note below)" % (n, q["I_QCLI"]))

    # Independence => every Chow-Liu edge weight (pairwise MI) is exactly 0, and I_TC = 0
    # exactly.
    assert abs(c["I_TC"]) < 1e-9, c["I_TC"]
    assert float(np.max(np.abs(c["MI_matrix"]))) < 1e-9

    # I_QCLI is NOT expected to be ~0 here: for independent-but-biased bits,
    # sum_x p(x)(-1)^(s.x) = prod_{i in s} (1 - 2*probs[i]), generically nonzero for nearly
    # every subset s (exact cancellation only happens per-bit at probs[i] == 0.5 exactly), so
    # the order spectrum m_k need not track the iid-FAIR-bit binomial baseline b_k at all even
    # though the bits are independent.


def test_even_parity_distribution():
    n = 6
    p = np.zeros(1 << n, dtype=np.float64)
    support = 0
    for x in range(1 << n):
        if bin(x).count("1") % 2 == 0:
            p[x] = 1.0
            support += 1
    p /= support

    q = qcli_exact(p, n)
    c = cci_exact(p, n)
    print("[even-parity dist, n=%d] m_k = %s" % (n, np.round(q["m_k"], 6)))
    print("[even-parity dist, n=%d] I_QCLI = %.6f  I_TC = %.6f  I_TC_tree = %.6f  I_CCI = %.6f"
          % (n, q["I_QCLI"], c["I_TC"], c["I_TC_tree"], c["I_CCI"]))

    # Verified analytically: P(s) = 2^-2n for s in {empty-set, full-set (weight n)}, and P(s)=0
    # for every other s (see module docstring decision 1). So m_0 = m_n = 0.5 EXACTLY, all
    # other m_k = 0 -- NOT "m_n = 1 alone" (a naive expectation that ignores the universal
    # empty-set floor). I_QCLI is still large: neither the m_0=m_n=0.5 split nor the
    # all-other-zero shape resembles the smooth binomial b_k.
    assert abs(q["m_k"][0] - 0.5) < 1e-12, q["m_k"]
    assert abs(q["m_k"][n] - 0.5) < 1e-12, q["m_k"]
    assert float(np.max(np.abs(q["m_k"][1:n]))) < 1e-12, q["m_k"]
    assert q["I_QCLI"] > 0.5, q["I_QCLI"]

    # Pure high-order (n-th order only) dependence beyond pairwise structure: for n>=3 the
    # pairwise marginals of the even-parity-uniform distribution are exactly the independent
    # product 1/4 each (verified: fixing any two bits, exactly half of the remaining n-2 bits'
    # assignments satisfy the parity constraint), so I_TC_tree = 0 exactly, while
    # I_TC = sum H(Xi) - H(X) = n - (n-1) = 1 bit exactly (H(X) = log2(2^(n-1)) = n-1). Hence
    # I_CCI = 1 - 0/1 = 1 exactly: all correlation is irreducibly high-order (n-th order),
    # invisible to any pairwise/tree model, matching the "pure high-order dependence" story.
    assert float(np.max(np.abs(c["MI_matrix"]))) < 1e-9, c["MI_matrix"]
    assert abs(c["I_TC"] - 1.0) < 1e-9, c["I_TC"]
    assert abs(c["I_TC_tree"]) < 1e-9, c["I_TC_tree"]
    assert abs(c["I_CCI"] - 1.0) < 1e-9, c["I_CCI"]


def test_markov_chain_tree_structured():
    n = 6
    rng = np.random.default_rng(4)
    p1 = rng.uniform(0.2, 0.8)
    trans = rng.uniform(0.1, 0.9, size=(n - 1, 2))  # trans[i, a] = P(X_{i+1}=1 | X_i=a)

    p = np.zeros(1 << n, dtype=np.float64)
    for x in range(1 << n):
        bits = [(x >> i) & 1 for i in range(n)]
        val = p1 if bits[0] else (1.0 - p1)
        for i in range(n - 1):
            pt1 = trans[i, bits[i]]
            val *= pt1 if bits[i + 1] else (1.0 - pt1)
        p[x] = val
    assert abs(p.sum() - 1.0) < 1e-9

    c = cci_exact(p, n)
    print("[Markov chain, n=%d] I_TC = %.6f  I_TC_tree = %.6f  I_CCI = %.6f"
          % (n, c["I_TC"], c["I_TC_tree"], c["I_CCI"]))
    # p is exactly a tree-structured (chain) distribution, so Chow-Liu's maximum-weight
    # spanning tree recovers the exact decomposition (Eq. 16) and I_TC_tree = I_TC exactly,
    # regardless of tie-breaking in which specific max-weight tree is found.
    assert abs(c["I_CCI"]) < 1e-6, c["I_CCI"]


def test_n18_timing():
    n = 18
    rng = np.random.default_rng(5)

    p_uniform = np.full(1 << n, 1.0 / (1 << n))
    t0 = time.time()
    q_u = qcli_exact(p_uniform, n)
    c_u = cci_exact(p_uniform, n)
    dt_u = time.time() - t0
    print("[n=18 uniform] wall = %.3fs  I_QCLI = %.6f  I_CCI = %.6f"
          % (dt_u, q_u["I_QCLI"], c_u["I_CCI"]))
    assert dt_u < 30.0, dt_u

    raw = rng.random(1 << n)
    p_rand = raw / raw.sum()
    t0 = time.time()
    q_r = qcli_exact(p_rand, n)
    c_r = cci_exact(p_rand, n)
    dt_r = time.time() - t0
    print("[n=18 random]  wall = %.3fs  I_QCLI = %.6f  I_CCI = %.6f"
          % (dt_r, q_r["I_QCLI"], c_r["I_CCI"]))
    assert dt_r < 30.0, dt_r


def test_qcli_mc_matches_exact():
    n = 10
    rng = np.random.default_rng(6)
    M = 20000
    x = rng.integers(0, 1 << n, size=M)
    p = np.bincount(x, minlength=1 << n).astype(np.float64) / M

    exact = qcli_exact(p, n)
    mc = qcli_mc(x, n, L=20000, seed=7)
    err = abs(mc["I_QCLI"] - exact["I_QCLI"])
    print("[qcli_mc vs exact, n=%d, M=%d, L=20000] exact=%.6f mc=%.6f err=%.4f"
          % (n, M, exact["I_QCLI"], mc["I_QCLI"], err))
    assert err < 0.05, err


def main():
    tests = [
        test_fwht_matches_bruteforce_P_n6,
        test_cci_mi_matches_bruteforce_n6,
        test_uniform_pmf_exact_population,
        test_near_uniform_empirical_pmf_small_qcli,
        test_independent_nonuniform_bits,
        test_even_parity_distribution,
        test_markov_chain_tree_structured,
        test_n18_timing,
        test_qcli_mc_matches_exact,
    ]
    for t in tests:
        print("--- %s ---" % t.__name__)
        t()
        print("PASS\n")
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
