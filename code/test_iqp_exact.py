"""Verification plan of DESIGN_exact_iqp.md.  Runs on CPU in a few seconds.

    python3 -u test_iqp_exact.py        (prints PASS/FAIL per check)
    pytest test_iqp_exact.py            (same checks as pytest test functions)
"""
from __future__ import annotations

import itertools
import math
import os
import time
from functools import reduce

import numpy as np
import torch

from iqp_exact import IQPExact, fwht, popcount_table, subset_masks, subsets_upto
from metrics import conv_mmd2, pdf_js

# Same CPU thread cap as train_iqp.py / baseline_{rbm,dcgan}.py: the FWHT butterfly is a
# chain of tiny reshape/stack ops on a (2^18,) tensor, and torch's default (= all cores,
# often 100+) thread pool makes sync overhead dominate real compute -- ~1000x slower
# (measured 22 s/step at 112 threads vs 0.02 s/step at 4). Must match what train_iqp.py's
# main() sets, or this timing test does not measure the configuration that gets submitted.
torch.set_num_threads(min(4, os.cpu_count() or 1))

torch.manual_seed(0)


# --------------------------------------------------------------------------- helpers
def dense_reference(n, K, theta):
    """q and <Z_s> from explicit 2^n x 2^n numpy matrices: H^(x)n diag(e^{i phi}) H^(x)n |0>."""
    size = 1 << n
    h = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.float64) / math.sqrt(2.0)
    H = reduce(np.kron, [h] * n)
    masks = subset_masks(n, K)
    idx = np.arange(size, dtype=np.int64)
    chi = np.empty((masks.size, size), dtype=np.float64)
    for i, m in enumerate(masks):
        chi[i] = 1.0 - 2.0 * (popcount_table(n)[(idx & int(m))] & 1)
    phi = theta @ chi
    U = H @ np.diag(np.exp(1j * phi)) @ H
    psi = U[:, 0]
    q = np.abs(psi) ** 2
    W = np.empty((size, size), dtype=np.float64)
    pc = popcount_table(n)
    for s in range(size):
        W[s] = 1.0 - 2.0 * (pc[idx & s] & 1)
    return q, W @ q


def brute_force_mmd2(n, q, p, sigmas):
    size = 1 << n
    pc = popcount_table(n).astype(np.float64)
    idx = np.arange(size, dtype=np.int64)
    dH = pc[(idx[:, None] ^ idx[None, :])]
    Kmat = np.zeros((size, size), dtype=np.float64)
    for sig in sigmas:
        Kmat += np.exp(-dH / (2.0 * sig * sig))
    Kmat /= len(sigmas)
    d = q - p
    return float(d @ Kmat @ d)


_RESULTS = []


def _report(name, ok, detail=""):
    _RESULTS.append((name, ok))
    print("%-46s %s   %s" % (name, "PASS" if ok else "FAIL", detail), flush=True)
    assert ok, name


# --------------------------------------------------------------------------- checks
def test_1_dense_statevector():
    """(1) n = 6..10 vs an explicit dense statevector build, complex128."""
    worst_q, worst_z = 0.0, 0.0
    for n in range(6, 11):
        K = 3
        sim = IQPExact(n=n, K=K, device="cpu", dtype=torch.complex128)
        rng = np.random.default_rng(n)
        theta = rng.normal(0.0, 0.7, size=sim.n_params)
        tt = torch.tensor(theta, dtype=torch.float64)
        q = sim.probs(tt).numpy()
        ez = sim.expvals(tt).numpy()
        q_ref, ez_ref = dense_reference(n, K, theta)
        worst_q = max(worst_q, float(np.max(np.abs(q - q_ref))))
        worst_z = max(worst_z, float(np.max(np.abs(ez - ez_ref))))
        amp = sim.amplitudes(tt).numpy()
        worst_q = max(worst_q, float(np.max(np.abs(np.abs(amp) ** 2 - q_ref))))
    _report("1. n=6..10 vs dense statevector", worst_q < 1e-10 and worst_z < 1e-10,
            "max|dq|=%.2e max|d<Z>|=%.2e" % (worst_q, worst_z))


def test_2_n18_normalization():
    """(2) n = 18: sum(q) = 1 and <Z_empty> = 1 for random theta; theta = 0 -> U = I -> q = delta_0.

    NOTE: DESIGN_exact_iqp.md says "theta = 0 -> q uniform -> <Z_s> = delta_{s,empty}".  That is
    wrong: U(0) = H^(x)n I H^(x)n = I, so U(0)|0^n> = |0^n>, q = delta_0 and <Z_s> = chi_s(0) = 1
    for EVERY s.  Verified against the dense statevector reference in check 1.  A uniform q would
    require a single Hadamard layer, not two.
    """
    sim = IQPExact(n=18, K=6, device="cpu", dtype=torch.complex64)
    theta = torch.tensor(np.random.default_rng(1).normal(0, 0.05, sim.n_params), dtype=torch.float32)
    q = sim.probs(theta)
    s = float(q.sum())
    z0 = float(sim.expvals(theta)[0])
    q0 = sim.probs(torch.zeros(sim.n_params, dtype=torch.float32))
    delta_err = float(torch.max(torch.abs(q0 - torch.eye(1, sim.size, dtype=q0.dtype)[0])))
    ez0 = sim.expvals(torch.zeros(sim.n_params, dtype=torch.float32))
    ones_err = float(torch.max(torch.abs(ez0 - 1.0)))
    ok = abs(s - 1.0) < 1e-5 and abs(z0 - 1.0) < 1e-5 and delta_err < 1e-6 and ones_err < 1e-5
    _report("2. n=18 norm / theta=0 gives U=I", ok,
            "sum(q)-1=%.2e <Z_0>-1=%.2e |q(0)-delta_0|=%.2e max|<Z_s>-1|=%.2e"
            % (s - 1.0, z0 - 1.0, delta_err, ones_err))


def test_3_closed_form_mmd():
    """(3) n = 8: closed-form MMD^2 vs an explicit 2^n x 2^n kernel matrix."""
    n, K = 8, 3
    sigmas = [0.6, 0.3]
    sim = IQPExact(n=n, K=K, device="cpu", dtype=torch.complex128)
    rng = np.random.default_rng(3)
    theta = torch.tensor(rng.normal(0.0, 0.5, sim.n_params), dtype=torch.float64)
    p = rng.random(1 << n) ** 3
    p /= p.sum()
    w = sim.kernel_weights(sigmas)
    ez_data = sim.data_expvals(p)
    closed = float(sim.mmd2(theta, ez_data, w))
    q = sim.probs(theta).numpy()
    brute = brute_force_mmd2(n, q, p, sigmas)
    err = abs(closed - brute)
    _report("3. closed-form MMD^2 vs kernel matrix", err < 1e-10,
            "closed=%.12e brute=%.12e |d|=%.2e" % (closed, brute, err))


def test_4_gradcheck():
    """(4) torch.autograd.gradcheck on the MMD loss, n = 8, float64."""
    n, K = 8, 2
    sim = IQPExact(n=n, K=K, device="cpu", dtype=torch.complex128)
    rng = np.random.default_rng(4)
    p = rng.random(1 << n)
    p /= p.sum()
    w = sim.kernel_weights([0.6, 0.3])
    ez_data = sim.data_expvals(p)
    theta = torch.tensor(rng.normal(0.0, 0.4, sim.n_params), dtype=torch.float64, requires_grad=True)
    ok = torch.autograd.gradcheck(lambda t: sim.mmd2(t, ez_data, w), (theta,), eps=1e-6,
                                  atol=1e-8, rtol=1e-5)
    _report("4. autograd.gradcheck (n=8, float64)", bool(ok), "%d params" % sim.n_params)


def test_5_metrics():
    """(5) pdf_js(x, x) = 0, conv_mmd2(x, x) ~ 0, different distributions > 0."""
    rng = np.random.default_rng(5)
    a = rng.normal(0.5, 0.15, size=(40, 64, 64)).astype(np.float32)
    b = rng.normal(0.9, 0.35, size=(40, 64, 64)).astype(np.float32)
    js_same = pdf_js(a, a)
    js_diff = pdf_js(a, b)
    mmd_same = conv_mmd2(a, a, seed=0)
    mmd_diff = conv_mmd2(a, b, seed=0)
    ok = (abs(js_same) < 1e-12 and js_diff > 1e-3 and abs(mmd_same) < 1e-12 and mmd_diff > 1e-6)
    _report("5. metrics self/cross sanity", ok,
            "js(x,x)=%.2e js(x,y)=%.4f mmd2(x,x)=%.2e mmd2(x,y)=%.4e"
            % (js_same, js_diff, mmd_same, mmd_diff))


def test_6_timing():
    """(6) wall-clock of one exact forward+backward at n = 18, K = 6 on CPU."""
    sim = IQPExact(n=18, K=6, device="cpu", dtype=torch.complex64)
    rng = np.random.default_rng(6)
    p = np.bincount(rng.integers(0, sim.size, size=50000), minlength=sim.size).astype(np.float64) / 50000
    w = sim.kernel_weights([0.6, 0.3])
    ez_data = sim.data_expvals(p)
    theta = torch.tensor(rng.normal(0, 0.01, sim.n_params), dtype=torch.float32, requires_grad=True)
    for _ in range(2):  # warm-up
        sim.mmd2(theta, ez_data, w).backward()
        theta.grad = None
    t0 = time.time()
    reps = 10
    for _ in range(reps):
        loss = sim.mmd2(theta, ez_data, w)
        loss.backward()
        theta.grad = None
    dt = (time.time() - t0) / reps
    _report("6. n=18 K=6 fwd+bwd timing", dt < 5.0,
            "%.4f s/step (%d params, loss=%.3e) -> 3e4 steps ~ %.1f min"
            % (dt, sim.n_params, float(loss.detach()), dt * 3e4 / 60.0))


def test_7_fwht_involution():
    """(extra) W W = 2^n I and the Walsh identity chi_s(y) hold for random vectors."""
    n = 10
    x = torch.randn(3, 1 << n, dtype=torch.float64)
    y = fwht(fwht(x)) / (1 << n)
    err = float(torch.max(torch.abs(y - x)))
    _report("7. FWHT involution W W = 2^n I", err < 1e-10, "max err = %.2e" % err)


def test_8_param_counts():
    """(extra) canonical ordering / parameter counts of the contract."""
    n = 18
    c6 = len(subsets_upto(n, 6))
    c7 = len(subsets_upto(n, 7))
    first = subsets_upto(n, 6)[:21]
    ok = (c6 == 31180 and c7 == 63004 and first[0] == () and first[1] == (0,)
          and first[18] == (17,) and first[19] == (0, 1) and first[20] == (0, 2))
    _report("8. subset ordering / param counts", ok, "K=6 -> %d, K=7 -> %d" % (c6, c7))


if __name__ == "__main__":
    print("=" * 96)
    for fn in (test_7_fwht_involution, test_8_param_counts, test_1_dense_statevector,
               test_2_n18_normalization, test_3_closed_form_mmd, test_4_gradcheck,
               test_5_metrics, test_6_timing):
        try:
            fn()
        except AssertionError:
            pass
        except Exception as exc:  # pragma: no cover
            _report(fn.__name__, False, "EXCEPTION %r" % (exc,))
    print("=" * 96)
    n_ok = sum(1 for _, ok in _RESULTS if ok)
    print("%d/%d checks passed" % (n_ok, len(_RESULTS)))
    raise SystemExit(0 if n_ok == len(_RESULTS) else 1)
