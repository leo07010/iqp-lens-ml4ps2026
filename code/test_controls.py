"""Smoke tests for controls.py (CTRL-marginal, CTRL-interp). CPU, seconds, login-node safe.

Run: /usr/bin/python3 test_controls.py
No pytest dependency; plain asserts + a runner so it works with the bare system interpreter
(see INTERFACES.md: python 3.9, no `X | None`).
"""
import json
import os
import shutil
import sys
import tempfile

import numpy as np

import controls
from codec import N_QUBITS


def _tmpdir():
    d = tempfile.mkdtemp(prefix="test_controls_")
    return d


# --------------------------------------------------------------------------------------
def test_marginal_smoke():
    """3 anchors, 4 holdout tau, tiny sample sizes: files exist, right shape, finite."""
    out_dir = os.path.join(_tmpdir(), "marginal")
    try:
        get_field, anchor_ts, _ = controls._load_data("synthetic", seed=0)
        anchor_ts = anchor_ts[:3]
        holdout_taus = controls.make_holdout_taus(anchor_ts, n_holdout=4, seed=0)
        assert len(holdout_taus) == 4, holdout_taus

        meta, p_vecs, a_z, b_z = controls.run_marginal(
            get_field, anchor_ts, holdout_taus, seed=0, n_synth=2000, out_dir=out_dir, n_tuples=2000
        )

        assert meta["arm"] == "CTRL-marginal"
        assert meta["param_count"] == N_QUBITS * len(anchor_ts) == 54
        assert p_vecs.shape == (3, N_QUBITS)
        assert np.all((p_vecs >= 0) & (p_vecs <= 1))
        assert b_z > a_z

        meta_path = os.path.join(out_dir, "meta.json")
        assert os.path.isfile(meta_path), meta_path
        with open(meta_path) as fh:
            meta_on_disk = json.load(fh)
        assert meta_on_disk["arm"] == "CTRL-marginal"

        for tau in holdout_taus:
            fpath = os.path.join(out_dir, "generated_t{}.npy".format(tau))
            assert os.path.isfile(fpath), fpath
            field = np.load(fpath)
            assert field.shape == (64, 64), field.shape
            assert field.dtype == np.float32, field.dtype
            assert np.isfinite(field).all()
        print("test_marginal_smoke: PASS")
    finally:
        shutil.rmtree(os.path.dirname(out_dir), ignore_errors=True)


def test_marginal_interp_exact_at_anchor():
    """Interpolated 18-dim param vector at tau == an anchor must equal that anchor's fitted p_vec exactly."""
    get_field, anchor_ts, _ = controls._load_data("synthetic", seed=1)
    anchor_ts = anchor_ts[:4]  # [1, 100, 200, 300]
    rng = np.random.default_rng(1)
    anchor_fields = [get_field(t) for t in anchor_ts]
    a_z = float(min(f.min() for f in anchor_fields))
    b_z = float(max(f.max() for f in anchor_fields))
    p_vecs = np.stack(
        [controls.fit_marginal_params(f, a_z, b_z, rng, n_tuples=3000) for f in anchor_fields], axis=0
    )

    for k, t in enumerate(anchor_ts):  # first, interior, last anchor
        p_at_anchor = controls.interp_anchors(t, anchor_ts, p_vecs)
        assert np.array_equal(p_at_anchor, p_vecs[k]), (t, k)
    print("test_marginal_interp_exact_at_anchor: PASS")


def test_interp_smoke():
    """3 anchors, 4 holdout tau: files exist, right shape, finite, param_count == 0."""
    out_dir = os.path.join(_tmpdir(), "interp")
    try:
        get_field, anchor_ts, _ = controls._load_data("synthetic", seed=0)
        anchor_ts = anchor_ts[:3]
        holdout_taus = controls.make_holdout_taus(anchor_ts, n_holdout=4, seed=0)

        meta, anchor_fields = controls.run_interp(get_field, anchor_ts, holdout_taus, seed=0, out_dir=out_dir)

        assert meta["arm"] == "CTRL-interp"
        assert meta["param_count"] == 0
        assert anchor_fields.shape == (3, 64, 64)

        for tau in holdout_taus:
            fpath = os.path.join(out_dir, "generated_t{}.npy".format(tau))
            assert os.path.isfile(fpath), fpath
            field = np.load(fpath)
            assert field.shape == (64, 64), field.shape
            assert field.dtype == np.float32, field.dtype
            assert np.isfinite(field).all()
        print("test_interp_smoke: PASS")
    finally:
        shutil.rmtree(os.path.dirname(out_dir), ignore_errors=True)


def test_interp_exact_at_anchor():
    """Generated field at tau == an anchor must equal that anchor's ground-truth field exactly
    (first, interior, and last anchor -- exercises all three _bracket branches)."""
    get_field, anchor_ts, _ = controls._load_data("synthetic", seed=2)
    anchor_ts = anchor_ts[:4]  # [1, 100, 200, 300]
    anchor_fields = np.stack([get_field(t) for t in anchor_ts], axis=0)

    for k, t in enumerate(anchor_ts):
        gen = controls.interp_anchors(t, anchor_ts, anchor_fields)
        gt = get_field(t)
        assert np.array_equal(gen, gt), (t, k)
        assert np.array_equal(gen, anchor_fields[k]), (t, k)
    print("test_interp_exact_at_anchor: PASS")


def test_interp_midpoint():
    """Sanity: interpolation at the exact midpoint between two anchors averages their fields."""
    anchor_ts = [0, 10]
    a = np.zeros((64, 64), dtype=np.float32)
    b = np.ones((64, 64), dtype=np.float32) * 4.0
    anchor_vals = np.stack([a, b], axis=0)
    mid = controls.interp_anchors(5, anchor_ts, anchor_vals)
    assert np.allclose(mid, 2.0), mid.mean()
    print("test_interp_midpoint: PASS")


def test_product_bernoulli_sampling_matches_p():
    """Empirical per-bit means from 1e5 sampled codes must match p_vec within 0.01."""
    rng = np.random.default_rng(42)
    p_vec = rng.uniform(0.05, 0.95, size=N_QUBITS)
    codes = controls.sample_product_bernoulli(p_vec, n=100_000, rng=np.random.default_rng(7))
    bits = controls._code_bits(codes)
    p_hat = bits.mean(axis=0)
    max_err = np.max(np.abs(p_hat - p_vec))
    assert max_err < 0.01, (max_err, p_hat, p_vec)
    print("test_product_bernoulli_sampling_matches_p: PASS (max |p_hat - p| = {:.5f})".format(max_err))


def test_code_bits_roundtrip():
    """bit j of the packed code must equal input bit j (sanity on the bit-packing convention)."""
    rng = np.random.default_rng(0)
    p_vec = np.full(N_QUBITS, 0.5)
    codes = controls.sample_product_bernoulli(p_vec, n=10, rng=rng)
    bits = controls._code_bits(codes)
    for i, code in enumerate(codes):
        for j in range(N_QUBITS):
            assert bits[i, j] == ((int(code) >> j) & 1)
    print("test_code_bits_roundtrip: PASS")


def test_cli_d2_missing_exits_cleanly():
    """--data d2 without data_d2.py must sys.exit with a clear message, not crash with a traceback.

    data_d2.py now exists (with a real d2_cache) -- in that case, this test instead checks
    controls._load_data("d2") returns the (get_field, anchor_ts, holdout_taus) contract with
    the shared 11-anchor / 200-holdout split (INTERFACES.md / data_d2.get_holdout_taus)."""
    try:
        result = controls._load_data("d2", seed=0)
    except SystemExit as e:
        assert "data_d2" in str(e), str(e)
        print("test_cli_d2_missing_exits_cleanly: PASS (data_d2.py not available)")
        return
    except ImportError as e:
        print("test_cli_d2_missing_exits_cleanly: SKIP (data_d2.py raised ImportError: %s)" % e)
        return
    assert isinstance(result, tuple) and len(result) == 3, result
    get_field, anchor_ts, holdout_taus = result
    assert callable(get_field)
    assert len(anchor_ts) == 11, anchor_ts
    assert holdout_taus is not None and len(holdout_taus) == 200, (
        "expected 200 holdout tau (data_d2.get_holdout_taus), got %r"
        % (None if holdout_taus is None else len(holdout_taus))
    )
    print("test_cli_d2_missing_exits_cleanly: PASS (data_d2.py exists; _load_data('d2') -> "
          "%d anchors, %d holdout tau)" % (len(anchor_ts), len(holdout_taus)))


def test_cli_main_marginal_end_to_end():
    """Exercise main() through argv, matching the real CLI entrypoint."""
    out_dir = os.path.join(_tmpdir(), "marginal_cli")
    try:
        controls.main([
            "--arm", "marginal", "--data", "synthetic", "--seed", "0",
            "--out", out_dir, "--anchors-max", "3", "--n-holdout", "3",
            "--n-synth", "1000", "--n-tuples", "1000",
        ])
        assert os.path.isfile(os.path.join(out_dir, "meta.json"))
        npys = [f for f in os.listdir(out_dir) if f.endswith(".npy")]
        assert len(npys) == 3, npys
        print("test_cli_main_marginal_end_to_end: PASS")
    finally:
        shutil.rmtree(os.path.dirname(out_dir), ignore_errors=True)


def test_cli_main_interp_end_to_end():
    out_dir = os.path.join(_tmpdir(), "interp_cli")
    try:
        controls.main([
            "--arm", "interp", "--data", "synthetic", "--seed", "0",
            "--out", out_dir, "--anchors-max", "3", "--n-holdout", "3",
        ])
        assert os.path.isfile(os.path.join(out_dir, "meta.json"))
        npys = [f for f in os.listdir(out_dir) if f.endswith(".npy")]
        assert len(npys) == 3, npys
        print("test_cli_main_interp_end_to_end: PASS")
    finally:
        shutil.rmtree(os.path.dirname(out_dir), ignore_errors=True)


ALL_TESTS = [
    test_marginal_smoke,
    test_marginal_interp_exact_at_anchor,
    test_interp_smoke,
    test_interp_exact_at_anchor,
    test_interp_midpoint,
    test_product_bernoulli_sampling_matches_p,
    test_code_bits_roundtrip,
    test_cli_d2_missing_exits_cleanly,
    test_cli_main_marginal_end_to_end,
    test_cli_main_interp_end_to_end,
]


if __name__ == "__main__":
    failures = []
    for t in ALL_TESTS:
        try:
            t()
        except Exception as e:
            failures.append((t.__name__, e))
            print("{}: FAIL ({})".format(t.__name__, e))
    print("\n{}/{} tests passed".format(len(ALL_TESTS) - len(failures), len(ALL_TESTS)))
    if failures:
        sys.exit(1)
