"""Shared machinery for the two controlled synthetic studies of arXiv 2603.06440
(Fig. 3 = QCLI-MMD support mismatch, Fig. 4 = QCLI-maximisation raises CCI).

What this module adds on top of the existing replication code in the parent directory:

  * `random_gate_masks`  -- draw G DISTINCT Pauli-Z strings of locality 1..L (the paper's
    "G randomly selected <=L-local parameterised gates"); the empty string is excluded
    because it is only a global phase.
  * `BatchIQP`           -- `iqp_exact.IQPExact` generalised in two ways that the controlled
    studies need: (a) an ARBITRARY generator subset instead of the "all |s| <= K" truncation,
    (b) a leading BATCH dimension, so B circuits (each with its own mask set and its own
    parameters) are simulated / trained in one FWHT.  All formulas are unchanged; `fwht`
    itself is imported from `iqp_exact` so the two code paths cannot drift apart.
  * `qcli_torch`         -- the exact QCLI of Eq. 5-7 on a batch of pmfs, differentiable, GPU.
  * `cci_batch`          -- the exact CCI of Eq. 12-15 computed from the Walsh expectation
    values (one FWHT) instead of 2^n-sized marginal sums; verified against
    `diagnostics_qcli_cci.cci_exact` in `test_controlled.py`.
  * `spsa_maximize`      -- the paper's optimiser (Table 3 "SPSA-optimized parameters",
    Table 4 "Random initialization; SPSA optimization"), batched.

Convention notes (must match the rest of the repo):
  * bit j of an int64 code == qubit j (codec.py); `fwht(p)[s] = E_p[(-1)^{s.x}]`.
  * QCLI INCLUDES the k = 0 (empty-subset) term, exactly as Eq. 6 is written.  This is a
    deliberate, already-audited decision of `diagnostics_qcli_cci.py` (see the notes in
    `test_diagnostics.py:97-105`): it makes the *exact* uniform distribution score ~1, not 0,
    and it puts a sample-size-dependent floor on the QCLI of any finite dataset.  Both
    controlled scripts therefore also report an i.i.d.-uniform null at the matched sample
    count so the floor is visible rather than silently baked into the x-axis.

Python 3.9 compatible (no `X | None`, no PEP-585 builtin generics).
"""
from __future__ import annotations

import itertools
import math
import os
import sys
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

# --- make the parent replication package importable ------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPL_DIR = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
if _REPL_DIR not in sys.path:
    sys.path.insert(0, _REPL_DIR)

from iqp_exact import fwht, popcount_table, sigma_for_avg_weight  # noqa: E402
import diagnostics_qcli_cci as diag  # noqa: E402

__all__ = [
    "REPL_DIR",
    "random_gate_masks",
    "n_available_masks",
    "BatchIQP",
    "popcount_index",
    "qcli_torch",
    "qcli_numpy_batch",
    "cci_batch",
    "empirical_pmf",
    "codes_to_pmf_batch",
    "kernel_c_by_weight",
    "mmd2_from_expvals",
    "segment_by_weight",
    "spsa_maximize",
    "rademacher_like",
    "uniform_like",
    "uniform_null_qcli",
    "resolve_sigmas_for_codes",
]

REPL_DIR = _REPL_DIR


# =======================================================================================
# Gate (Pauli-Z string) selection
# =======================================================================================
def n_available_masks(n, locality):
    """Number of distinct non-empty Pauli-Z strings with 1 <= |s| <= locality on n qubits."""
    return int(sum(math.comb(int(n), k) for k in range(1, int(locality) + 1)))


def random_gate_masks(n, locality, n_gates, rng):
    """`n_gates` DISTINCT bitmasks with 1 <= popcount <= locality, drawn uniformly.

    Returns (masks int64 array sorted ascending, n_effective).

    DEVIATION, reported by both scripts: the paper asks for 150 2-local gates at
    n in {8, 12, 16}, but there are only C(n,1)+C(n,2) = 36 / 78 / 136 distinct <=2-local
    Pauli-Z strings, so 150 distinct gates do not exist.  In an IQP circuit repeated gates on
    the same qubit set commute and their angles simply add, i.e. a repeated draw is a
    reparameterisation, not extra expressivity.  We therefore draw
    min(n_gates, n_available_masks(n, locality)) distinct strings -- the most expressive
    circuit consistent with the requested budget -- and record `gates_effective` alongside
    `gates_requested` in every output row.
    """
    n = int(n)
    locality = int(locality)
    total = n_available_masks(n, locality)
    n_eff = int(min(int(n_gates), total))

    # Enumerate the candidate pool.  For the sizes used here (n <= 16, locality <= 4) the pool
    # is at most C(16,1..4) = 2516 entries, so explicit enumeration is cheap and exact.
    pool = np.empty(total, dtype=np.int64)
    idx = 0
    for k in range(1, locality + 1):
        for comb in itertools.combinations(range(n), k):
            m = 0
            for j in comb:
                m |= (1 << j)
            pool[idx] = m
            idx += 1
    assert idx == total
    pick = rng.choice(total, size=n_eff, replace=False)
    masks = np.sort(pool[pick]).astype(np.int64)
    return masks, n_eff


# =======================================================================================
# Random-number helpers that tolerate a generator whose device does not match the tensor
# =======================================================================================
_GEN_OK = {}


def _gen_for(generator, device):
    """Return a generator usable for `device`, or None.

    torch requires a CUDA `torch.Generator` for CUDA tensors; some builds additionally refuse
    `torch.randint(..., generator=cuda_gen)`.  Rather than discovering that 5000 steps into a
    GPU job, probe once per (generator, device) pair and fall back to host-side generation
    (the perturbation is only (B, P) floats, so the copy is negligible).
    """
    if generator is None:
        return None
    key = (id(generator), str(device))
    ok = _GEN_OK.get(key)
    if ok is None:
        try:
            torch.randint(0, 2, (1,), generator=generator, device=device)
            ok = True
        except Exception:
            ok = False
        _GEN_OK[key] = ok
    return generator if ok else None


def rademacher_like(theta, generator=None):
    """+-1 tensor with theta's shape/dtype/device, drawn from `generator` when it can serve
    that device and from the host otherwise."""
    g = _gen_for(generator, theta.device)
    if g is not None or generator is None:
        return torch.randint(0, 2, theta.shape, generator=g, device=theta.device,
                             dtype=theta.dtype) * 2.0 - 1.0
    d = torch.randint(0, 2, theta.shape, generator=generator, dtype=torch.int8) * 2 - 1
    return d.to(device=theta.device, dtype=theta.dtype)


def uniform_like(shape, ref, generator=None):
    """U[0,1) tensor of `shape` on ref's device in float64, generator-tolerant."""
    g = _gen_for(generator, ref.device)
    if g is not None or generator is None:
        return torch.rand(*shape, generator=g, device=ref.device, dtype=torch.float64)
    return torch.rand(*shape, generator=generator, dtype=torch.float64).to(ref.device)


# =======================================================================================
# Batched exact IQP simulator on an arbitrary generator subset
# =======================================================================================
class BatchIQP(object):
    """Exact simulation of B IQP circuits sharing the qubit count n and the parameter count P.

        U(theta) = H^{otimes n} exp(i sum_{s in masks} theta_s Z_s) H^{otimes n}
        q_theta(x) = |<x|U|0^n>|^2

    `masks` is either a (P,) array (all B circuits share one architecture -- the Fig. 3
    learner case) or a (B, P) array (one architecture per batch row -- the Fig. 3 target and
    Fig. 4 case).  Everything is computed exactly via FWHTs; no Monte-Carlo estimator.
    """

    def __init__(self, n, masks, device="cpu", dtype=torch.float32):
        self.n = int(n)
        self.size = 1 << self.n
        self.device = torch.device(device)
        self.rdtype = dtype
        m = np.asarray(masks, dtype=np.int64)
        if m.ndim == 1:
            m = m[None, :]
            self.shared = True
        elif m.ndim == 2:
            self.shared = False
        else:
            raise ValueError("masks must be 1-D or 2-D, got shape %s" % (m.shape,))
        if m.max() >= self.size or m.min() < 0:
            raise ValueError("mask out of range for n=%d" % self.n)
        self.masks_np = m
        self.n_params = int(m.shape[1])
        self.masks = torch.as_tensor(m, dtype=torch.long, device=self.device)
        self._pc_np = popcount_table(self.n)
        self.pc = torch.as_tensor(self._pc_np.astype(np.int64), device=self.device)

    # ---------------------------------------------------------------------------------
    def theta_full(self, theta):
        """(B, P) trainable angles -> (B, 2^n) dense Walsh coefficient vector."""
        if theta.dim() == 1:
            theta = theta[None, :]
        B = theta.shape[0]
        if theta.shape[1] != self.n_params:
            raise ValueError("expected %d params, got %d" % (self.n_params, theta.shape[1]))
        idx = self.masks
        if idx.shape[0] == 1 and B > 1:
            idx = idx.expand(B, self.n_params)
        elif idx.shape[0] != B:
            raise ValueError("masks batch %d incompatible with theta batch %d" % (idx.shape[0], B))
        base = torch.zeros(B, self.size, dtype=theta.dtype, device=theta.device)
        return base.scatter_add(1, idx, theta)

    def phase(self, theta):
        """phi(y) = sum_s theta_s (-1)^{popcount(s & y)}, shape (B, 2^n)."""
        return fwht(self.theta_full(theta))

    def probs(self, theta):
        """q_theta(x), shape (B, 2^n).  Real arithmetic so autograd stays cheap."""
        phi = self.phase(theta)
        re = fwht(torch.cos(phi))
        im = fwht(torch.sin(phi))
        scale = 1.0 / float(self.size * self.size)
        return (re * re + im * im) * scale

    def expvals(self, theta):
        """<Z_s>_theta for all 2^n subsets, shape (B, 2^n)."""
        return fwht(self.probs(theta))

    def sample(self, theta, n_samples, generator=None):
        """(B, n_samples) int64 bitstring codes drawn from q_theta (inverse CDF, float64)."""
        with torch.no_grad():
            q = self.probs(theta.detach()).to(torch.float64)
            q = torch.clamp(q, min=0.0)
            q = q / q.sum(dim=1, keepdim=True)
            cdf = torch.cumsum(q, dim=1)
            cdf[:, -1] = 1.0
            B = q.shape[0]
            u = uniform_like((B, int(n_samples)), q, generator=generator)
            idx = torch.searchsorted(cdf, u, right=True)
            idx = torch.clamp(idx, max=self.size - 1)
        return idx.to(torch.int64)

    def init_theta(self, batch, rng, std=0.01):
        """N(0, std) initialisation, the repo default (`train_iqp.py --init-std 0.01`)."""
        t = rng.normal(0.0, float(std), size=(int(batch), self.n_params))
        return torch.as_tensor(t, dtype=self.rdtype, device=self.device)


# =======================================================================================
# QCLI (Eq. 5-7), batched
# =======================================================================================
def popcount_index(n, device):
    return torch.as_tensor(popcount_table(int(n)).astype(np.int64), device=torch.device(device))


def segment_by_weight(v, n, pc_idx):
    """Sum a (B, 2^n) quantity into (B, n+1) bins by Hamming weight of the subset index."""
    B = v.shape[0]
    out = torch.zeros(B, int(n) + 1, dtype=v.dtype, device=v.device)
    return out.index_add(1, pc_idx, v)


def _binomial_baseline(n, device, dtype):
    b = np.array([math.comb(int(n), k) for k in range(int(n) + 1)], dtype=np.float64) / float(1 << int(n))
    return torch.as_tensor(b, dtype=dtype, device=device)


def qcli_torch(p, n, pc_idx, b_k=None, eps=1e-12):
    """Exact I_QCLI of Eq. 7 for a batch of pmfs p (B, 2^n).  Differentiable.

    Mirrors `diagnostics_qcli_cci.qcli_exact` term for term (including the k=0 subset and the
    same epsilon-clip-and-renormalise of the JS divergence, in log base 2).
    """
    n = int(n)
    w = fwht(p)                                # (B, 2^n) = E[chi_s] (the 2^-2n prefactor of
    P = w * w                                  # Eq. 5 is constant in s and cancels in m_k)
    m_num = segment_by_weight(P, n, pc_idx)    # (B, n+1)
    m = m_num / m_num.sum(dim=1, keepdim=True).clamp_min(eps)
    if b_k is None:
        b_k = _binomial_baseline(n, p.device, p.dtype)
    b = b_k[None, :].expand_as(m)
    m = torch.clamp(m, min=eps)
    b = torch.clamp(b, min=eps)
    m = m / m.sum(dim=1, keepdim=True)
    b = b / b.sum(dim=1, keepdim=True)
    mix = 0.5 * (m + b)
    ln2 = math.log(2.0)
    kl_mb = (m * (torch.log(m / mix) / ln2)).sum(dim=1)
    kl_bm = (b * (torch.log(b / mix) / ln2)).sum(dim=1)
    return 0.5 * kl_mb + 0.5 * kl_bm


def qcli_numpy_batch(pmfs, n):
    """Reference path: `diagnostics_qcli_cci.qcli_exact` applied row by row (numpy, float64)."""
    return np.array([diag.qcli_exact(np.asarray(p, dtype=np.float64), int(n))["I_QCLI"] for p in pmfs],
                    dtype=np.float64)


# =======================================================================================
# CCI (Eq. 12-15) from Walsh expectation values, batched
# =======================================================================================
def _pair_index(n):
    ii, jj = [], []
    for i in range(n):
        for j in range(i + 1, n):
            ii.append(i)
            jj.append(j)
    return np.asarray(ii, dtype=np.int64), np.asarray(jj, dtype=np.int64)


def _xlog2x(a):
    out = np.zeros_like(a)
    nz = a > 0
    out[nz] = a[nz] * np.log2(a[nz])
    return out


def cci_batch(p, n, eps=1e-9):
    """Exact I_CCI for a batch of pmfs.  `p` is (B, 2^n) torch tensor or numpy array.

    Identity used (this is where the speed comes from): every 1- and 2-qubit marginal of p is
    an affine function of three Walsh coefficients, which one FWHT gives for all subsets at
    once, so the O(n^2 * 2^n) marginal sums of `diagnostics_qcli_cci.cci_exact` collapse to
    O(n^2) after a single O(n 2^n) transform:

        p_i(a)     = (1 + (-1)^a <Z_i>) / 2
        p_ij(a, b) = (1 + (-1)^a <Z_i> + (-1)^b <Z_j> + (-1)^{a+b} <Z_i Z_j>) / 4

    Everything downstream (H_joint, I_TC, the pairwise MI matrix, the Chow-Liu maximum-weight
    spanning tree via `diagnostics_qcli_cci._chow_liu_tree`, and I_CCI = 1 - I_TC^tree / I_TC)
    is the plug-in estimator of Eq. 12-15, unchanged.  Verified against `cci_exact` in
    `test_controlled.py::test_cci_batch_matches_cci_exact`.

    Returns dict of numpy arrays: I_CCI (B,), I_TC (B,), I_TC_tree (B,), H_joint (B,).
    """
    n = int(n)
    single = np.asarray([1 << i for i in range(n)], dtype=np.int64)
    ii, jj = _pair_index(n)
    pair_masks = (1 << ii) | (1 << jj)

    if isinstance(p, torch.Tensor):
        # Do the transform and the 2^n-sized entropy sum on the device and ship back only the
        # n + n(n-1)/2 Walsh coefficients the tree needs (136 numbers at n = 16), so that the
        # per-step cost of Fig. 4 is not dominated by a 2^n device-to-host copy.
        pt = p.detach().to(torch.float64)
        if pt.dim() == 1:
            pt = pt[None, :]
        ezt = fwht(pt)
        idx = torch.as_tensor(np.concatenate([single, pair_masks]), device=pt.device)
        ez_small = ezt.index_select(1, idx).cpu().numpy()
        logp = torch.where(pt > 0, torch.log2(pt.clamp_min(1e-300)), torch.zeros_like(pt))
        H_joint = (-(pt * logp).sum(dim=1)).cpu().numpy()
        mi_ = ez_small[:, :n]
        zij = ez_small[:, n:]
    else:
        pn = np.asarray(p, dtype=np.float64)
        if pn.ndim == 1:
            pn = pn[None, :]
        ez = fwht(torch.as_tensor(pn)).numpy()
        H_joint = -_xlog2x(pn).sum(axis=1)                               # (B,)
        mi_ = ez[:, single]                                              # (B, n) = <Z_i>
        zij = ez[:, pair_masks]                                          # (B, E) = <Z_i Z_j>
    B = mi_.shape[0]
    p_i1 = 0.5 * (1.0 - mi_)                                             # P(x_i = 1)
    p_i0 = 0.5 * (1.0 + mi_)
    H_marg = -(_xlog2x(p_i0) + _xlog2x(p_i1))                            # (B, n)
    I_TC = H_marg.sum(axis=1) - H_joint                                  # (B,)

    zi = mi_[:, ii]
    zj = mi_[:, jj]
    p00 = 0.25 * (1.0 + zi + zj + zij)
    p01 = 0.25 * (1.0 + zi - zj - zij)
    p10 = 0.25 * (1.0 - zi + zj - zij)
    p11 = 0.25 * (1.0 - zi - zj + zij)
    # I(X_i;X_j) = H(X_i) + H(X_j) - H(X_i,X_j)
    H_ij = -(_xlog2x(p00) + _xlog2x(p01) + _xlog2x(p10) + _xlog2x(p11))
    MI = H_marg[:, ii] + H_marg[:, jj] - H_ij                            # (B, E)
    MI = np.clip(MI, 0.0, None)

    I_CCI = np.zeros(B, dtype=np.float64)
    I_TC_tree = np.zeros(B, dtype=np.float64)
    M = np.zeros((n, n), dtype=np.float64)
    for b in range(B):
        M[:] = 0.0
        M[ii, jj] = MI[b]
        M[jj, ii] = MI[b]
        _edges, tree_w = diag._chow_liu_tree(M)
        I_TC_tree[b] = tree_w
        I_CCI[b] = 0.0 if abs(I_TC[b]) < eps else 1.0 - tree_w / I_TC[b]
    return {"I_CCI": I_CCI, "I_TC": I_TC, "I_TC_tree": I_TC_tree, "H_joint": H_joint}


# =======================================================================================
# Empirical pmfs
# =======================================================================================
def empirical_pmf(codes, n):
    """(M,) int64 codes -> (2^n,) float64 empirical pmf (Eq. 4)."""
    codes = np.asarray(codes, dtype=np.int64)
    return np.bincount(codes, minlength=1 << int(n)).astype(np.float64) / float(codes.size)


def codes_to_pmf_batch(codes, n):
    """(B, M) int64 torch codes -> (B, 2^n) float64 torch empirical pmfs."""
    B, M = codes.shape
    out = torch.zeros(B, 1 << int(n), dtype=torch.float64, device=codes.device)
    src = torch.full((B, M), 1.0 / float(M), dtype=torch.float64, device=codes.device)
    return out.scatter_add(1, codes, src)


def uniform_null_qcli(n, n_samples, n_reps, seed):
    """I_QCLI of i.i.d.-uniform n-bit draws at the matched sample count (the finite-sample
    floor of the indicator; see the module docstring)."""
    rng = np.random.default_rng(int(seed))
    vals = []
    for _ in range(int(n_reps)):
        codes = rng.integers(0, 1 << int(n), size=int(n_samples), dtype=np.int64)
        vals.append(diag.qcli_exact(empirical_pmf(codes, n), int(n))["I_QCLI"])
    return float(np.mean(vals)), float(np.std(vals))


# =======================================================================================
# MMD (Appendix G, exact closed form)
# =======================================================================================
def kernel_c_by_weight(n, sigmas):
    """Walsh coefficients of the multi-bandwidth Gaussian kernel, collapsed to the (n+1,)
    per-Hamming-weight values (c_s depends on s only through |s|).  Averaged over bandwidths,
    matching `iqp_exact.IQPExact.kernel_weights` / Ref [38] Eq. 45."""
    n = int(n)
    acc = np.zeros(n + 1, dtype=np.float64)
    sigmas = list(sigmas)
    for sig in sigmas:
        r = math.exp(-1.0 / (2.0 * float(sig) ** 2))
        alpha = 0.5 * (1.0 + r)
        beta = 0.5 * (1.0 - r)
        acc += np.array([(alpha ** (n - k)) * (beta ** k) for k in range(n + 1)], dtype=np.float64)
    return acc / float(len(sigmas))


def mmd2_from_expvals(ez_model, ez_data, c_by_weight, n, pc_idx):
    """MMD^2 = sum_s c_s (<Z_s>_model - <Z_s>_data)^2, batched.

    ez_model/ez_data: (B, 2^n);  c_by_weight: (B, n+1) or (n+1,).  Returns (B,).
    Grouping by Hamming weight first turns the 2^n-long weighted sum into an (n+1)-long dot
    product without any approximation (c_s is constant within a weight class).
    """
    d = ez_model - ez_data
    seg = segment_by_weight(d * d, int(n), pc_idx)          # (B, n+1)
    if c_by_weight.dim() == 1:
        c_by_weight = c_by_weight[None, :]
    return (seg * c_by_weight).sum(dim=1)


def resolve_sigmas_for_codes(n, codes, seed):
    """The repo's three-bandwidth rule (`train_iqp.resolve_sigmas('auto3', ...)`) applied to one
    target dataset.  Imported lazily so that `common` stays importable without codec/synth_data."""
    import train_iqp
    sigmas, weights = train_iqp.resolve_sigmas("auto3", int(n), np.asarray(codes, dtype=np.int64), int(seed))
    return sigmas, weights


# =======================================================================================
# SPSA (Table 3 / Table 4 optimiser), batched
# =======================================================================================
def spsa_maximize(objective, theta0, n_steps, a=0.2, c=0.1, alpha=0.602, gamma=0.101,
                  A_frac=0.1, generator=None, callback=None, maximize=True):
    """Batched Simultaneous Perturbation Stochastic Approximation.

    `objective(theta)` takes (B, P) and returns (B,).  One shared Rademacher perturbation per
    batch row per step; two objective evaluations per step (Spall's standard 2-measurement
    form).  Gains follow Spall's recommended schedule
        a_k = a / (k + 1 + A)^alpha ,  c_k = c / (k + 1)^gamma ,  A = A_frac * n_steps.
    `callback(step, theta, value)` is called after every accepted step (value = objective at
    the *new* theta only if `callback` asks for it -- we pass the cheap midpoint estimate
    0.5*(f_plus + f_minus) to avoid a third evaluation; scripts that need the exact value at
    theta recompute it themselves).

    Returns the final theta.  No autograd is used or needed.
    """
    theta = theta0.clone()
    A = float(A_frac) * float(n_steps)
    sign = 1.0 if maximize else -1.0
    for k in range(int(n_steps)):
        ak = float(a) / ((k + 1.0 + A) ** float(alpha))
        ck = float(c) / ((k + 1.0) ** float(gamma))
        delta = rademacher_like(theta, generator=generator)
        fp = objective(theta + ck * delta)
        fm = objective(theta - ck * delta)
        ghat = ((fp - fm) / (2.0 * ck))[:, None] * delta      # Rademacher: 1/delta == delta
        theta = theta + sign * ak * ghat
        if callback is not None:
            callback(k, theta, 0.5 * (fp + fm))
    return theta
