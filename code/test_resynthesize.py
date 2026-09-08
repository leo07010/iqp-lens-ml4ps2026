#!/usr/bin/env python3
"""CPU smoke tests for resynthesize.py (arXiv 2603.06440 replication). Finishes in seconds,
safe for the login node. Python 3.9 compatible.

(a) field_from_probs' (kx, ky, kz) reshape of the flat 2^18 code-indexed array matches an
    independent brute-force implementation built from codec.decode -- for a RANDOM q, not
    just a special-case one, so the test actually exercises the bit-layout assumption, not
    just an easy corner case.
(b) Run exact + sample1e6 against the real trained theta of outputs_d2_main/iqp_K6_seed0 for
    3 held-out tau (CPU): shapes, finiteness, value range, and that sample1e6 is closer to
    exact than the original sample1e5 (generated_t{tau}.npy already on disk) is.
(c) Prints (does not assert) pdf_js(GT, exact) vs pdf_js(GT, original sample1e5) for the same
    3 tau, using outputs_d2_main/ground_truth.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import torch

import codec
import metrics
import resynthesize
from iqp_exact import IQPExact

REPO = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(REPO, "outputs_d2_main", "iqp_K6_seed0")
GT_DIR = os.path.join(REPO, "outputs_d2_main", "ground_truth")


# --------------------------------------------------------------------------------------
# (a) reshape/conditional-expectation vs brute force (codec.decode), random q
# --------------------------------------------------------------------------------------
def _brute_field_from_probs(q, a_z, b_z):
    """Independent path: does NOT assume the reshape(64,64,64)->(kx,ky,kz) layout --
    decodes every one of the 2^18 codes explicitly via codec.decode and bins by (ky, kx),
    exactly mirroring codec.aggregate's own binning convention."""
    q = np.asarray(q, dtype=np.float64)
    codes = np.arange(q.size, dtype=np.int64)
    kx, ky, kz = codec.decode(codes)
    z = codec.dequantize_z(kz, a_z, b_z)
    num = np.zeros((codec.GRID, codec.GRID), dtype=np.float64)
    den = np.zeros((codec.GRID, codec.GRID), dtype=np.float64)
    np.add.at(num, (ky, kx), q * z)
    np.add.at(den, (ky, kx), q)
    gmean = 0.5 * (a_z + b_z)
    den_pos = np.where(den > 0, den, 1.0)
    return np.where(den > 1e-12, num / den_pos, gmean)


def test_reshape_matches_brute_force():
    rng = np.random.default_rng(12345)
    q = rng.random(codec.NCODES).astype(np.float64) + 1e-9  # strictly positive, arbitrary scale
    a_z, b_z = 0.3, 1.7

    fast = resynthesize.field_from_probs(q, a_z, b_z)
    brute = _brute_field_from_probs(q, a_z, b_z)

    assert fast.shape == (64, 64)
    max_abs_diff = float(np.max(np.abs(fast - brute)))
    print("[test a] reshape vs brute-force max|diff| = %.3e (must be < 1e-12)" % max_abs_diff)
    assert max_abs_diff < 1e-12, "reshape-based field_from_probs disagrees with brute force: %.3e" % max_abs_diff


# --------------------------------------------------------------------------------------
# (b) + (c) real theta, 3 held-out tau, CPU
# --------------------------------------------------------------------------------------
def _pick_3_taus():
    taus = resynthesize.tau_list_from_dir(SRC_DIR)
    assert len(taus) >= 3, "expected >=3 generated_t*.npy in %s, found %d" % (SRC_DIR, len(taus))
    return taus[:3]


def test_real_theta_exact_and_sample1e6():
    if not os.path.isdir(SRC_DIR):
        print("[test b/c] SKIPPED: %s not found" % SRC_DIR)
        return
    import json
    with open(os.path.join(SRC_DIR, "meta.json")) as fh:
        meta = json.load(fh)

    theta_core_np = np.load(os.path.join(SRC_DIR, "theta_core.npy"))
    theta_lat_anchors = np.load(os.path.join(SRC_DIR, "theta_lat_anchors.npy"))
    anchors = [int(a) for a in meta["anchors"]]
    a_z, b_z = float(meta["a_z"]), float(meta["b_z"])
    K = int(meta["K"])

    device = torch.device("cpu")
    sim = IQPExact(n=codec.N_QUBITS, K=K, device=device, dtype=torch.complex64)
    theta_core_t = torch.as_tensor(theta_core_np, dtype=sim.rdtype, device=device)

    taus = _pick_3_taus()
    print("[test b/c] using tau =", taus)

    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(meta["seed"]) + 99_000_000)

    exact_fields, sample_fields, orig_fields, gt_fields = [], [], [], []
    for tau in taus:
        import train_iqp
        lat_np = train_iqp.interp_theta_lat(anchors, theta_lat_anchors, tau)
        lat_t = torch.as_tensor(lat_np, dtype=sim.rdtype, device=device)
        theta = sim.join(lat_t, theta_core_t)

        f_exact = resynthesize.iqp_exact_field(sim, theta, a_z, b_z)
        assert f_exact.shape == (64, 64)
        assert np.isfinite(f_exact).all()
        assert f_exact.min() >= a_z - 1e-6 and f_exact.max() <= b_z + 1e-6, \
            "exact field out of [a_z,b_z]: [%.6g,%.6g] vs [%.6g,%.6g]" % (f_exact.min(), f_exact.max(), a_z, b_z)
        exact_fields.append(f_exact)

        f_sample = resynthesize.iqp_sample_field(sim, theta, a_z, b_z, 1_000_000, gen)
        assert f_sample.shape == (64, 64)
        assert np.isfinite(f_sample).all()
        sample_fields.append(f_sample)

        orig_path = os.path.join(SRC_DIR, "generated_t%d.npy" % tau)
        orig_fields.append(np.load(orig_path))

        gt_path = os.path.join(GT_DIR, "gt_t%d.npy" % tau)
        gt_fields.append(np.load(gt_path) if os.path.isfile(gt_path) else None)

    exact_all = np.concatenate([f.ravel() for f in exact_fields])
    sample_all = np.concatenate([f.ravel() for f in sample_fields])
    orig_all = np.concatenate([f.ravel() for f in orig_fields])

    rms_sample1e6_vs_exact = resynthesize.rms(sample_all, exact_all)
    rms_orig1e5_vs_exact = resynthesize.rms(orig_all, exact_all)
    ratio = rms_sample1e6_vs_exact / rms_orig1e5_vs_exact if rms_orig1e5_vs_exact > 0 else float("nan")

    print("[test b] RMS(sample1e6, exact)      = %.6e" % rms_sample1e6_vs_exact)
    print("[test b] RMS(orig sample1e5, exact) = %.6e" % rms_orig1e5_vs_exact)
    print("[test b] ratio sample1e6/sample1e5  = %.6f  (< 1 means noise did shrink)" % ratio)
    assert rms_sample1e6_vs_exact < rms_orig1e5_vs_exact, (
        "sample1e6 should be closer to exact than the original sample1e5 was: %.6e vs %.6e"
        % (rms_sample1e6_vs_exact, rms_orig1e5_vs_exact)
    )

    if all(g is not None for g in gt_fields):
        gt_stack = np.stack(gt_fields, axis=0)
        exact_stack = np.stack(exact_fields, axis=0)
        orig_stack = np.stack(orig_fields, axis=0)
        js_exact = float(metrics.pdf_js(gt_stack, exact_stack))
        js_orig = float(metrics.pdf_js(gt_stack, orig_stack))
        print("[test c] pdf_js(GT, exact)             = %.6f  (informational, not asserted)" % js_exact)
        print("[test c] pdf_js(GT, orig sample1e5)     = %.6f  (informational, not asserted)" % js_orig)
    else:
        print("[test c] SKIPPED: ground truth files missing for one of", taus)


if __name__ == "__main__":
    test_reshape_matches_brute_force()
    test_real_theta_exact_and_sample1e6()
    print("[test_resynthesize] all tests passed")
