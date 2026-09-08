"""Exact IQP simulation (n <= ~20) via fast Walsh-Hadamard transforms.

Implements DESIGN_exact_iqp.md:
    U(theta) = H^{otimes n} exp(i sum_s theta_s Z_s) H^{otimes n},  q_theta(x) = |<x|U|0^n>|^2

with everything computed EXACTLY (no Monte-Carlo estimator):
    phi   = W Theta                       (W = unnormalized Walsh-Hadamard, W[y,s] = (-1)^{popcount(y&s)})
    psi   = 2^{-n} W exp(i phi)
    q     = |psi|^2
    <Z_s> = (W q)(s)   for all s at once
    MMD^2 = sum_s c_s (<Z_s>_theta - <Z_s>_data)^2 ,  c_s = mean_sigma alpha^{n-|s|} beta^{|s|}

Kernel convention (confirmed against Ref [38] = arXiv 2503.02934, Sec. 3.3/4.2 and its code):
    k(x, x') = exp(-||x - x'||^2 / (2 sigma^2)) on RAW 0/1 bitstrings, so ||x-x'||^2 = Hamming distance.
    r = exp(-1/(2 sigma^2)), alpha = (1+r)/2, beta = (1-r)/2 = p_sigma of Ref [38] Eq. 17.
    Several bandwidths are AVERAGED (Ref [38] Eq. 45), not summed.

Python 3.9 compatible (no `X | None`, no `list[int]` annotations).
"""
from __future__ import annotations

import itertools
import math
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

__all__ = [
    "fwht",
    "subsets_upto",
    "subset_masks",
    "popcount_table",
    "sigma_for_avg_weight",
    "IQPExact",
]


# --------------------------------------------------------------------------------------
# Fast Walsh-Hadamard transform
# --------------------------------------------------------------------------------------
def fwht(x: torch.Tensor) -> torch.Tensor:
    """Unnormalized Walsh-Hadamard transform along the LAST dimension (length 2^n).

    Returns y with y[..., s] = sum_v x[..., v] * (-1)^{popcount(v & s)}.
    Autograd-safe, works for real and complex dtypes and any number of leading batch dims.
    Cost: n butterfly passes, O(n 2^n).
    """
    size = x.shape[-1]
    n = int(round(math.log2(size)))
    if (1 << n) != size:
        raise ValueError("fwht: last dimension must be a power of two, got %d" % size)
    batch = tuple(x.shape[:-1])
    y = x
    for j in range(n):
        y = y.reshape(batch + (1 << j, 2, 1 << (n - 1 - j)))
        a = y[..., 0, :]
        b = y[..., 1, :]
        y = torch.stack((a + b, a - b), dim=-2)
        y = y.reshape(batch + (size,))
    return y


# --------------------------------------------------------------------------------------
# Subset (Pauli-Z string) bookkeeping
# --------------------------------------------------------------------------------------
def subsets_upto(n: int, K: int) -> List[Tuple[int, ...]]:
    """Canonical parameter ordering of the paper: increasing Hamming weight, then
    itertools.combinations (lexicographic) order within a weight.
    {} , {0}, {1}, ..., {n-1}, {0,1}, {0,2}, ...
    """
    out = []  # type: List[Tuple[int, ...]]
    for w in range(0, K + 1):
        for c in itertools.combinations(range(n), w):
            out.append(c)
    return out


def subset_masks(n: int, K: int) -> np.ndarray:
    """int64 bitmasks of subsets_upto(n, K) in the same order (mask = sum_j 1 << j)."""
    subs = subsets_upto(n, K)
    masks = np.zeros(len(subs), dtype=np.int64)
    for i, c in enumerate(subs):
        m = 0
        for j in c:
            m |= (1 << j)
        masks[i] = m
    return masks


def popcount_table(n: int) -> np.ndarray:
    """int8 array of length 2^n with the Hamming weight of every index."""
    size = 1 << n
    idx = np.arange(size, dtype=np.int64)
    pc = np.zeros(size, dtype=np.int8)
    for j in range(n):
        pc += ((idx >> j) & 1).astype(np.int8)
    return pc


def sigma_for_avg_weight(n: int, avg_weight: float) -> float:
    """Bandwidth sigma such that the induced Walsh weight distribution has mean `avg_weight`.

    Ref [38] Eq. 17: |a| ~ Binomial(n, p_sigma), p_sigma = (1 - exp(-1/(2 sigma^2)))/2.
    Setting n * p_sigma = avg_weight and solving:  sigma = sqrt(-1 / (2 log(1 - 2 w / n))).
    (Helper for choosing bandwidths; not used by default.)
    """
    p = float(avg_weight) / float(n)
    if not (0.0 < p < 0.5):
        raise ValueError("avg_weight must be in (0, n/2), got %r for n=%d" % (avg_weight, n))
    return math.sqrt(-1.0 / (2.0 * math.log(1.0 - 2.0 * p)))


# --------------------------------------------------------------------------------------
# The simulator
# --------------------------------------------------------------------------------------
_REAL_OF = {
    torch.complex64: torch.float32,
    torch.complex128: torch.float64,
    torch.float32: torch.float32,
    torch.float64: torch.float64,
}


class IQPExact(object):
    """Exact simulator of the IQP generator on n qubits with |s| <= K trainable parameters.

    Parameters
    ----------
    n : number of qubits (18 for the replication).
    K : maximum Pauli-Z weight of the trainable generators (6 -> 31,180 params, 7 -> 63,004).
    device : torch device.
    dtype : torch.complex64 (default) / torch.complex128 for the amplitudes; the real working
            dtype is derived from it (float32 / float64).
    """

    def __init__(self, n, K, device="cpu", dtype=torch.complex64):
        if dtype not in _REAL_OF:
            raise ValueError("unsupported dtype %r" % (dtype,))
        self.n = int(n)
        self.K = int(K)
        self.size = 1 << self.n
        self.device = torch.device(device)
        self.rdtype = _REAL_OF[dtype]
        self.cdtype = torch.complex128 if self.rdtype == torch.float64 else torch.complex64

        masks_np = subset_masks(self.n, self.K)
        self.masks = torch.as_tensor(masks_np, dtype=torch.long, device=self.device)
        self.n_params = int(masks_np.size)
        self._popcount_np = popcount_table(self.n)
        self.popcount = torch.as_tensor(self._popcount_np.astype(np.int64), device=self.device)

    # ---------------------------------------------------------------- parameter plumbing
    def theta_full(self, theta: torch.Tensor) -> torch.Tensor:
        """Scatter the (P,) trainable vector into the dense (2^n,) coefficient vector."""
        if theta.shape[-1] != self.n_params:
            raise ValueError("expected %d params, got %d" % (self.n_params, theta.shape[-1]))
        base = torch.zeros(self.size, dtype=theta.dtype, device=theta.device)
        return base.index_add(0, self.masks.to(theta.device), theta)

    def split(self, theta: torch.Tensor, d_lat: int = 50):
        """(theta_lat, theta_core) = (first d_lat entries, the rest)."""
        return theta[..., :d_lat], theta[..., d_lat:]

    @staticmethod
    def join(theta_lat: torch.Tensor, theta_core: torch.Tensor) -> torch.Tensor:
        return torch.cat((theta_lat, theta_core), dim=-1)

    # ---------------------------------------------------------------- circuit quantities
    def phase(self, theta: torch.Tensor) -> torch.Tensor:
        """phi(y) = sum_s theta_s (-1)^{popcount(s & y)}, shape (2^n,)."""
        return fwht(self.theta_full(theta))

    def amplitudes(self, theta: torch.Tensor) -> torch.Tensor:
        """psi(x) = <x| U(theta) |0^n>, complex, shape (2^n,)."""
        phi = self.phase(theta)
        re = fwht(torch.cos(phi))
        im = fwht(torch.sin(phi))
        scale = 1.0 / float(self.size)
        return torch.complex(re * scale, im * scale).to(self.cdtype)

    def probs(self, theta: torch.Tensor) -> torch.Tensor:
        """q_theta(x) = |<x|U(theta)|0^n>|^2, shape (2^n,). Real arithmetic (autograd-friendly)."""
        phi = self.phase(theta)
        re = fwht(torch.cos(phi))
        im = fwht(torch.sin(phi))
        scale = 1.0 / float(self.size * self.size)
        return (re * re + im * im) * scale

    def expvals(self, theta: torch.Tensor) -> torch.Tensor:
        """<Z_s>_theta for every subset s (all 2^n of them), shape (2^n,)."""
        return fwht(self.probs(theta))

    # ---------------------------------------------------------------- MMD
    def kernel_weights(self, sigmas: Sequence[float]) -> torch.Tensor:
        """Walsh coefficients c_s of the (multi-bandwidth) Gaussian kernel, shape (2^n,).

        k_sigma(x, x') = exp(-d_H(x, x') / (2 sigma^2)) on 0/1 bitstrings
                       = sum_s alpha^{n-|s|} beta^{|s|} chi_s(x) chi_s(x'),
        r = exp(-1/(2 sigma^2)), alpha = (1+r)/2, beta = (1-r)/2.
        Multiple bandwidths are AVERAGED (Ref [38] Eq. 45: L = (1/L) sum_i MMD^2_{sigma_i}).
        """
        sigmas = list(sigmas)
        if len(sigmas) == 0:
            raise ValueError("need at least one sigma")
        pc = self._popcount_np.astype(np.int64)
        acc = np.zeros(self.size, dtype=np.float64)
        for sig in sigmas:
            r = math.exp(-1.0 / (2.0 * float(sig) ** 2))
            alpha = 0.5 * (1.0 + r)
            beta = 0.5 * (1.0 - r)
            # c_s = alpha^(n-|s|) beta^|s| depends only on |s|; computed directly (no logs) so beta -> 0
            # gives the correct limit c = [1, 0, 0, ...] instead of NaN (fixed 2026-09-05).
            c_by_weight = np.array([(alpha ** (self.n - k)) * (beta ** k) for k in range(self.n + 1)], dtype=np.float64)
            acc += c_by_weight[pc]
        acc /= float(len(sigmas))
        return torch.as_tensor(acc, dtype=self.rdtype, device=self.device)

    def data_expvals(self, pmf) -> torch.Tensor:
        """<Z_s>_data for every s from an empirical pmf (numpy or torch, length 2^n)."""
        p = torch.as_tensor(np.asarray(pmf, dtype=np.float64), device=self.device)
        return fwht(p).to(self.rdtype)

    def mmd2(self, theta: torch.Tensor, ez_data: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """MMD^2(q_theta, p) = sum_s c_s (<Z_s>_theta - <Z_s>_data)^2 (exact, all 2^n subsets)."""
        d = self.expvals(theta) - ez_data
        return (weights * d * d).sum()

    # ---------------------------------------------------------------- sampling
    def sample(self, theta: torch.Tensor, n_samples: int, generator=None) -> torch.Tensor:
        """Draw n_samples int64 bitstring codes from q_theta (inverse-CDF, float64 internally)."""
        with torch.no_grad():
            q = self.probs(theta.detach()).to(torch.float64)
            q = torch.clamp(q, min=0.0)
            q = q / q.sum()
            cdf = torch.cumsum(q, dim=0)
            cdf[-1] = 1.0
            u = torch.rand(int(n_samples), generator=generator, device=q.device, dtype=torch.float64)
            idx = torch.searchsorted(cdf, u, right=True)
            idx = torch.clamp(idx, max=self.size - 1)
        return idx.to(torch.int64)
