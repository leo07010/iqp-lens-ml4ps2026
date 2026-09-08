"""QCLI (Quantum Correlation-Likeness Indicator) and CCI (Classical Correlation-Complexity
Indicator) diagnostics for the arXiv 2603.06440 replication.

See INTERFACES.md (codec/file-ownership contract) and replication_spec.md Sec. 7-8 / Sec. 12
items 10-11 for the derivation and the open questions the original paper leaves unresolved.
Formula citations below are to paper.txt line numbers (arXiv 2603.06440 HTML dump).

Formulas:
    P(s)   = |2^-n * sum_x p(x) * (-1)^(s.x)|^2                          (Eq. 5, paper.txt:521-529)
    m_k    = sum_{|s|=k} P(s) / sum_{s subseteq [n]} P(s), k = 0..n      (Eq. 6, paper.txt:532-538)
    b_k    = C(n,k) / 2^n                                                (paper.txt:539-541)
    I_QCLI = D_JS(m || b), log base 2                                    (Eq. 7/53, paper.txt:549-558, 2461-2469)

    I_TC      = sum_i H(X_i) - H(X_1,...,X_n)                            (Eq. 12/18, paper.txt:626-635)
    I_TC_tree = sum_{(i,j) in T*} I(X_i;X_j)                             (Eq. 13, paper.txt:637-645)
    T*        = Chow-Liu maximum-weight spanning tree over pairwise MI   (Eq. 14-15, paper.txt:647-680)
    I_CCI     = 1 - I_TC_tree / I_TC                                     (Eq. 12, paper.txt:626-630)

Design decisions not fully pinned down by the paper text (documented, not silently assumed):

1. m_k / P(s) sums INCLUDE s = the empty set at k = 0. This part is *not* ambiguous: Eq. 6
   literally says "k = 0, 1, ..., n" with "sum_{k=0}^n m_k = 1", and Appendix C's Eq. 5 repeats
   the same range for the Monte Carlo estimator. What *is* a real, non-obvious consequence,
   verified analytically and by brute force in test_diagnostics.py, is that
       P(empty set) = |2^-n * sum_x p(x)|^2 = 2^(-2n)
   for ANY valid pmf (since sum_x p(x) = 1 identically) -- the k=0 term carries *zero*
   information about p, it is a universal floor. For n=18 this floor is 2^-36 ~ 1.5e-11,
   negligible next to any real dataset's signal, but for synthetic pmfs engineered to have
   *exactly* zero power at every s != empty set (e.g. the exact population-uniform pmf, or the
   exact even-parity-uniform pmf below), this floor becomes the *entire* m vector -- giving
   I_QCLI values that are large, not the naively-expected ~0. See test_diagnostics.py for the
   exact numbers and derivation; this is a literal reading of Eq. 5-6, not an implementation bug.

2. I_CCI when I_TC ~ 0 (fully independent data, or the degenerate uniform-pmf case above): we
   define I_CCI := 0.0 by convention (there is no dependence to explain away, so the tree
   trivially "captures all of nothing"), guarded by a small epsilon on I_TC, rather than
   returning NaN from the literal 0/0. Raw I_TC and I_TC_tree are always returned alongside
   I_CCI so downstream code can distinguish "truly independent" (I_TC ~ 0) from a genuine
   near-1.0 CCI ratio.

3. Entropy H and mutual information I(Xi;Xj) both use log base 2 (matching I_QCLI's explicit
   log2 convention, Eq. 53). This choice is immaterial to I_CCI itself: the log base appears
   identically in I_TC and I_TC_tree's numerator/denominator and cancels in their ratio; it only
   sets the units (bits) of the raw I_TC / I_TC_tree magnitudes returned alongside I_CCI.

4. qcli_mc (Appendix C's Monte Carlo estimator, paper.txt:2399-2483) operates directly on raw
   int64 sample codes (Eq. 4's M-sample empirical pmf), NOT a pre-built dense 2^n pmf array.
   This matches Appendix C's actual point: for large n you can never afford to materialize a
   dense 2^n array, only evaluate (-1)^(s.x) parities over the M raw samples for each of the L
   Monte-Carlo-sampled subsets s (P_hat(s) = (2^-n * mean_i (-1)^popcount(s & x_i))^2, which is
   exactly Eq. 4's empirical pmf substituted into Eq. 5 without ever binning it into an array).

5. qcli_exact / cci_exact for n=18 real data use *exact* enumeration (2^18 = 262,144 is
   tractable), per replication_spec.md Sec. 12 item 10's recommendation, not Appendix C's Monte
   Carlo default (L=2e4) which the paper describes as being for "larger n".

6. IMPORTANT FINDING (numerically swept in test_diagnostics.py, not stated in the paper): the
   paper's II.3 claim "if the data have no structured parity content beyond iid noise, m ~ b in
   expectation, I_QCLI -> 0" only holds for finite-M empirical pmfs with M well BELOW roughly
   C(n, n/2) (subset count at the middle order). Above that crossover, I_QCLI for genuinely iid
   *uniform* draws INCREASES with M, converging toward the degenerate exact-population-uniform
   value (decision 1: m -> [1,0,...,0], I_QCLI large) rather than -> 0. Reason: for s != empty
   set, E[P(s)] ~ 2^-2n/M (sampling variance), so order-k aggregate power ~ C(n,k)*2^-2n/M only
   dominates the M-independent P(empty set)=2^-2n floor when C(n,k)/M >> 1, i.e. M << C(n,n/2).
   For the real turbulence application (n=18, M=N_t=5e4), C(18,9)=48620 is comparable to
   M=5e4 -- right at this crossover, so I_QCLI there may be sensitive to the exact N_t choice.
   Flagged for whoever runs the real n=18 D1/D2 QCLI computation.

Python 3.9 compatible (no `X | None`, no match statements, no bare `list[int]` annotations).
"""
import argparse
import json
import math
import time

import numpy as np

try:
    import codec
except ImportError:
    codec = None


# --------------------------------------------------------------------------------------------
# Walsh-Hadamard transform + popcount
# --------------------------------------------------------------------------------------------

def fwht_np(v):
    """Unnormalized Walsh-Hadamard transform via vectorized butterfly recursion (numpy only).

    Input v: 1D real array of length 2**n. Output w: same length, with
        w[s] = sum_x v[x] * (-1)**popcount(s & x)
    where bit j of both the output index s and the input index x corresponds to qubit j
    (bit 0 = LSB), matching codec.py's `qubit j == bit j of code` convention. Verified against
    a brute-force double loop for n=6 in test_diagnostics.py (max abs err < 1e-12).
    """
    v = np.asarray(v, dtype=np.float64).copy()
    length = v.shape[0]
    if length <= 0 or (length & (length - 1)) != 0:
        raise ValueError("fwht_np requires length a power of 2, got %d" % length)
    h = 1
    while h < length:
        v = v.reshape(-1, 2, h)
        x = v[:, 0, :]
        y = v[:, 1, :]
        v = np.concatenate([x + y, x - y], axis=1).reshape(-1)
        h *= 2
    return v


def _popcount(x):
    """Vectorized popcount for int64 array (or scalar-broadcastable) values known to fit in
    a modest number of bits (loop over bit positions until all values are exhausted)."""
    x = np.asarray(x, dtype=np.int64)
    count = np.zeros(x.shape, dtype=np.int64)
    y = x.copy()
    while np.any(y):
        count += y & 1
        y >>= 1
    return count


def popcount_table(n):
    """Return int64 array of length 2**n with popcount(s) for s = 0 .. 2**n - 1."""
    idx = np.arange(1 << n, dtype=np.int64)
    return _popcount(idx)


# --------------------------------------------------------------------------------------------
# QCLI
# --------------------------------------------------------------------------------------------

def _js_divergence(m, b, base=2.0, eps=1e-12):
    """Jensen-Shannon divergence D_JS(m || b) in the given log base, with epsilon-clipping and
    renormalization for numerical stability when some entries are exactly zero (Appendix C,
    paper.txt: "we apply a small epsilon-clipping and renormalization to avoid numerical issues
    when some m_hat_k are zero under finite sampling"). Does not mutate its inputs."""
    m = np.clip(np.asarray(m, dtype=np.float64), eps, None)
    b = np.clip(np.asarray(b, dtype=np.float64), eps, None)
    m = m / m.sum()
    b = b / b.sum()
    mmix = 0.5 * (m + b)
    log_base = math.log(base)

    def _kl(a, ref):
        return float(np.sum(a * (np.log(a / ref) / log_base)))

    return 0.5 * _kl(m, mmix) + 0.5 * _kl(b, mmix)


def qcli_exact(p, n):
    """Exact QCLI: enumerate all 2**n subsets s via the Walsh-Hadamard transform of p.

    p: length-2**n array, a valid pmf (sums to 1).
    Returns dict(I_QCLI, m_k (shape (n+1,), RAW/unclipped Eq. 6 order shares), b_k (shape
    (n+1,), binomial baseline), P_total (= sum_s P(s), the Eq. 6 denominator)).
    """
    p = np.asarray(p, dtype=np.float64)
    if p.shape != (1 << n,):
        raise ValueError("p must have shape (2**n,) = (%d,), got %s" % (1 << n, p.shape))
    w = fwht_np(p)                       # w[s] = sum_x p(x) (-1)^(s.x)
    P = (w / float(1 << n)) ** 2         # Eq. 5
    pc = popcount_table(n)
    m_num = np.bincount(pc, weights=P, minlength=n + 1)   # sum_{|s|=k} P(s)
    P_total = float(m_num.sum())
    if P_total > 0:
        m_k = m_num / P_total
    else:
        # Cannot happen for a valid pmf (P(empty set) = 2^-2n > 0 always), guarded anyway.
        m_k = np.full(n + 1, 1.0 / (n + 1))
    b_k = np.array([math.comb(n, k) for k in range(n + 1)], dtype=np.float64) / float(1 << n)
    I_QCLI = _js_divergence(m_k, b_k, base=2.0)
    return {"I_QCLI": I_QCLI, "m_k": m_k, "b_k": b_k, "P_total": P_total}


def _sample_subset_masks(n, k, mk, rng):
    """Sample `mk` DISTINCT k-subsets of [n] uniformly without replacement, as int64 bitmasks."""
    total = math.comb(n, k)
    mk = min(mk, total)
    if k == 0:
        return np.array([0], dtype=np.int64)
    if k == n:
        return np.array([(1 << n) - 1], dtype=np.int64)
    seen = set()
    out = []
    max_tries = mk * 50 + 2000
    tries = 0
    while len(out) < mk and tries < max_tries:
        idx = rng.choice(n, size=k, replace=False)
        mask = 0
        for b in idx:
            mask |= 1 << int(b)
        if mask not in seen:
            seen.add(mask)
            out.append(mask)
        tries += 1
    return np.array(out, dtype=np.int64)


def qcli_mc(samples, n, L, seed):
    """Appendix C Monte Carlo estimator of I_QCLI (paper.txt:2399-2483).

    `samples`: 1D array of M raw int64 codes (an n-bit bitstring per sample; bit j = qubit j,
    e.g. from codec.sample_tuples / codec.field_to_codes_full), i.e. Eq. 4's M-sample empirical
    dataset -- NOT a pre-built dense 2**n pmf array (see module docstring, decision 4).
    L: total subset budget, split with equal-per-order allocation across k=0..n (Appendix C).
    seed: RNG seed for subset sampling.

    Returns dict(I_QCLI, m_k (shape (n+1,), Eq. 52 estimate), b_k).
    """
    rng = np.random.default_rng(seed)
    codes = np.asarray(samples, dtype=np.int64)
    M = codes.shape[0]
    m_target = max(1, L // (n + 1))
    A_k = np.zeros(n + 1, dtype=np.float64)
    chunk_size = 256
    for k in range(n + 1):
        total_k = math.comb(n, k)
        mk = min(m_target, total_k)
        if mk <= 0:
            continue
        masks = _sample_subset_masks(n, k, mk, rng)
        mk = masks.shape[0]
        signs_mean = np.zeros(mk, dtype=np.float64)
        for start in range(0, mk, chunk_size):
            chunk = masks[start:start + chunk_size]
            anded = chunk[:, None] & codes[None, :]
            parity = _popcount(anded) & 1
            signs = 1.0 - 2.0 * parity.astype(np.float64)     # (-1)^parity
            signs_mean[start:start + chunk_size] = signs.mean(axis=1)
        Phat = (signs_mean / float(1 << n)) ** 2               # Eq. 5 with p = empirical pmf
        A_k[k] = (total_k / float(mk)) * Phat.sum()            # Eq. 51
    A_total = A_k.sum()
    if A_total > 0:
        m_hat = A_k / A_total
    else:
        m_hat = np.full(n + 1, 1.0 / (n + 1))
    b_k = np.array([math.comb(n, k) for k in range(n + 1)], dtype=np.float64) / float(1 << n)
    I_QCLI = _js_divergence(m_hat, b_k, base=2.0)
    return {"I_QCLI": I_QCLI, "m_k": m_hat, "b_k": b_k}


# --------------------------------------------------------------------------------------------
# CCI
# --------------------------------------------------------------------------------------------

def _chow_liu_tree(MI):
    """Maximum-weight spanning tree over pairwise MI via Kruskal's algorithm with a hand-rolled
    union-find (no scipy/networkx). Returns (tree_edges: list of (i, j), total_weight: float)."""
    n = MI.shape[0]
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((MI[i, j], i, j))
    edges.sort(key=lambda e: -e[0])

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        parent[ra] = rb
        return True

    tree_edges = []
    total = 0.0
    for w, i, j in edges:
        if union(i, j):
            tree_edges.append((i, j))
            total += w
            if len(tree_edges) == n - 1:
                break
    return tree_edges, total


def cci_exact(p, n, eps=1e-9):
    """Exact CCI via plug-in entropy/MI estimates over the full 2**n outcome space and a
    Chow-Liu maximum-weight spanning tree (Eq. 12-18, paper.txt:626-688).

    p: length-2**n array, a valid pmf.
    Returns dict(I_CCI, I_TC, I_TC_tree, tree_edges (list of (i,j) qubit-index pairs),
    H_marg (shape (n,), per-qubit marginal entropies), H_joint (scalar), MI_matrix (shape
    (n,n), symmetric, zero diagonal)).
    """
    p = np.asarray(p, dtype=np.float64)
    if p.shape != (1 << n,):
        raise ValueError("p must have shape (2**n,) = (%d,), got %s" % (1 << n, p.shape))
    # order='F': axis 0 varies fastest in the flat index, i.e. flat_index = sum_i bit_i * 2**i,
    # which is exactly codec.py's "qubit j == bit j of code" convention (bit 0 = LSB = axis 0).
    # (The default C-order reshape would instead make the LAST axis = bit 0, silently swapping
    # qubit labels -- caught by test_cci_mi_matches_bruteforce_n6's brute-force cross check,
    # which indexes bits directly via (x >> i) & 1.)
    P = p.reshape((2,) * n, order="F")

    nz = P > 0
    H_joint = float(-np.sum(P[nz] * np.log2(P[nz])))

    H_marg = np.zeros(n, dtype=np.float64)
    marg1 = np.zeros((n, 2), dtype=np.float64)
    for i in range(n):
        axes = tuple(j for j in range(n) if j != i)
        mi_ = P.sum(axis=axes)                 # shape (2,)
        marg1[i] = mi_
        nzm = mi_ > 0
        H_marg[i] = float(-np.sum(mi_[nzm] * np.log2(mi_[nzm])))

    I_TC = float(H_marg.sum() - H_joint)

    MI = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            axes = tuple(k for k in range(n) if k != i and k != j)
            pij = P.sum(axis=axes)              # shape (2, 2) with axis order (i, j) preserved
            outer = np.outer(marg1[i], marg1[j])
            nzp = pij > 0
            val = float(np.sum(pij[nzp] * np.log2(pij[nzp] / outer[nzp])))
            MI[i, j] = val
            MI[j, i] = val

    tree_edges, I_TC_tree = _chow_liu_tree(MI)

    if abs(I_TC) < eps:
        I_CCI = 0.0
    else:
        I_CCI = 1.0 - I_TC_tree / I_TC

    return {
        "I_CCI": I_CCI,
        "I_TC": I_TC,
        "I_TC_tree": I_TC_tree,
        "tree_edges": tree_edges,
        "H_marg": H_marg,
        "H_joint": H_joint,
        "MI_matrix": MI,
    }


# --------------------------------------------------------------------------------------------
# Combined entry point
# --------------------------------------------------------------------------------------------

def diagnose_codes(codes, n=18):
    """From int64 codes (codec.py convention: bit j = qubit j), compute both QCLI and CCI on
    the pooled empirical pmf. Uses codec.codes_to_pmf for the default n=18 (imports codec.py
    directly per INTERFACES.md); falls back to a generic bincount pmf builder otherwise."""
    codes = np.asarray(codes, dtype=np.int64)
    if n == 18 and codec is not None:
        p = codec.codes_to_pmf(codes)
    else:
        p = np.bincount(codes, minlength=1 << n).astype(np.float64) / len(codes)
    q = qcli_exact(p, n)
    c = cci_exact(p, n)
    return {
        "I_QCLI": q["I_QCLI"],
        "I_CCI": c["I_CCI"],
        "n_samples": int(len(codes)),
        "qcli": q,
        "cci": c,
    }


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------

def _fallback_synthetic_snapshots(n_snapshots, seed):
    """Smooth synthetic (64, 64) fields (sum of drifting Gaussian blobs) used when synth_data.py
    (owned by agent CORE) is not importable yet, so this module can be smoke-tested standalone."""
    rng = np.random.default_rng(seed)
    grid = 64
    yy, xx = np.mgrid[0:grid, 0:grid].astype(np.float64)
    n_blobs = 3
    centers0 = rng.uniform(10, 54, size=(n_blobs, 2))
    vel = rng.uniform(-1.5, 1.5, size=(n_blobs, 2))
    sigma = rng.uniform(6, 14, size=n_blobs)
    amp = rng.uniform(0.5, 2.0, size=n_blobs)
    snaps = np.zeros((n_snapshots, grid, grid), dtype=np.float32)
    for t in range(n_snapshots):
        field = np.zeros((grid, grid), dtype=np.float64)
        for b in range(n_blobs):
            cy = (centers0[b, 0] + vel[b, 0] * t) % grid
            cx = (centers0[b, 1] + vel[b, 1] * t) % grid
            field += amp[b] * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma[b] ** 2)))
        field += 0.05 * rng.standard_normal((grid, grid))
        snaps[t] = field.astype(np.float32)
    return snaps


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return dict((k, _to_jsonable(v)) for k, v in obj.items())
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def run_synthetic(n, n_snapshots, n_tuples, seed):
    if codec is None:
        raise RuntimeError("codec.py could not be imported; cannot build tuple codes")
    try:
        import synth_data
        snaps = synth_data.make_synthetic_snapshots(n_snapshots, seed)
    except Exception:
        snaps = _fallback_synthetic_snapshots(n_snapshots, seed)

    rng = np.random.default_rng(seed)
    a_z = float(snaps.min())
    b_z = float(snaps.max())
    if b_z <= a_z:
        b_z = a_z + 1.0

    all_codes = []
    for i in range(snaps.shape[0]):
        all_codes.append(codec.sample_tuples(snaps[i], n_tuples, rng, a_z, b_z))
    codes = np.concatenate(all_codes)
    return diagnose_codes(codes, n=n)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--codes", type=str, default=None, help="path to a .npy of int64 codes")
    ap.add_argument("--n", type=int, default=18)
    ap.add_argument("--out", type=str, default=None, help="write result dict as JSON here")
    ap.add_argument("--synthetic", action="store_true",
                     help="use synth_data.make_synthetic_snapshots (or a fallback) instead of --codes")
    ap.add_argument("--n-snapshots", type=int, default=8)
    ap.add_argument("--n-tuples", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    if args.synthetic:
        result = run_synthetic(args.n, args.n_snapshots, args.n_tuples, args.seed)
    elif args.codes:
        codes = np.load(args.codes)
        result = diagnose_codes(codes, n=args.n)
    else:
        ap.error("must pass --codes PATH or --synthetic")
        return
    dt = time.time() - t0

    print("n_samples = %d" % result["n_samples"])
    print("I_QCLI    = %.6f" % result["I_QCLI"])
    print("I_CCI     = %.6f" % result["I_CCI"])
    print("wall_s    = %.3f" % dt)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(_to_jsonable(result), f, indent=2)
        print("wrote %s" % args.out)


if __name__ == "__main__":
    main()
