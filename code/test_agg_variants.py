#!/usr/bin/env python3
"""CPU tests for agg_variants.py (arXiv 2603.06440 replication, D2 aggregation-convention
ablation). Finishes in well under a second, safe for the login node. Python 3.9 compatible.

(a) One sample per cell (all 4096 codes distinct grid cells): aggregate_last, aggregate_first,
    aggregate_random_one, and codec.aggregate must all agree -- with exactly one sample per
    cell there is nothing to pick between, so every convention (mean-of-one, last, first,
    random-of-one) reduces to the same value.
(b) Multiple samples per cell, every cell hit at least once (no empty-cell fill exercised):
    aggregate_last must equal a straightforward per-cell "walk the array, remember the most
    recent z seen for this cell" brute-force reduction. Same check for aggregate_first against
    "remember the FIRST z seen".
(c) aggregate_random_one, same data as (b): the returned z for every cell must be a MEMBER of
    that cell's own observed z set (not some fabricated average), and re-running with a
    different rng seed can disagree with aggregate_last (documenting that it need not pick the
    same sample -- not a hard assert, since a different seed COULD coincidentally agree).
(d) Shape / finite / dtype sanity for all three functions on random large code arrays that DO
    leave some cells empty, exercising _fill_empty's neighbour-mean/global-mean fallback.
"""
from __future__ import annotations

import numpy as np

import agg_variants
import codec

A_Z, B_Z = 0.3, 1.7


def _full_grid_codes_one_per_cell(rng):
    """Exactly one code per (ky, kx) cell -- 4096 codes total, each cell visited once, in
    (ky, kx) row-major order (ky slowest) so cell order matches field.reshape(64,64).ravel()."""
    ky, kx = np.meshgrid(np.arange(codec.GRID), np.arange(codec.GRID), indexing="ij")
    kz = rng.integers(0, codec.GRID, size=(codec.GRID, codec.GRID))
    return codec.encode(kx.ravel(), ky.ravel(), kz.ravel())


def test_one_sample_per_cell_all_conventions_agree():
    rng = np.random.default_rng(0)
    codes = _full_grid_codes_one_per_cell(rng)

    mean_field = codec.aggregate(codes, A_Z, B_Z)
    last_field = agg_variants.aggregate_last(codes, A_Z, B_Z)
    first_field = agg_variants.aggregate_first(codes, A_Z, B_Z)
    random_field = agg_variants.aggregate_random_one(codes, A_Z, B_Z, np.random.default_rng(1))

    for name, field in (("last", last_field), ("first", first_field), ("random", random_field)):
        assert field.shape == (64, 64), (name, field.shape)
        assert np.isfinite(field).all(), name
        max_abs_diff = float(np.max(np.abs(field.astype(np.float64) - mean_field.astype(np.float64))))
        print("[test a] one-sample-per-cell: max|%s - codec.aggregate(mean)| = %.3e" % (name, max_abs_diff))
        assert max_abs_diff < 1e-5, "%s disagrees with codec.aggregate when every cell has 1 sample: %.3e" % (
            name, max_abs_diff)


def _multi_sample_codes(rng, n_extra_per_cell=3):
    """Every one of the 4096 cells gets a base sample (so no cell is ever empty), PLUS
    `n_extra_per_cell` additional random-z draws scattered over a random subset of cells,
    all concatenated in a single fixed, known array order (base pass first, then each extra
    pass) so a python-loop brute force can replay the exact same order."""
    ky0, kx0 = np.meshgrid(np.arange(codec.GRID), np.arange(codec.GRID), indexing="ij")
    ky0 = ky0.ravel(); kx0 = kx0.ravel()
    kz0 = rng.integers(0, codec.GRID, size=ky0.shape[0])
    parts = [codec.encode(kx0, ky0, kz0)]
    for _ in range(n_extra_per_cell):
        n = 2000
        ky = rng.integers(0, codec.GRID, size=n)
        kx = rng.integers(0, codec.GRID, size=n)
        kz = rng.integers(0, codec.GRID, size=n)
        parts.append(codec.encode(kx, ky, kz))
    return np.concatenate(parts)


def _brute_force_last_and_first(codes, a_z, b_z):
    c = np.asarray(codes, dtype=np.int64)
    kx, ky, kz = codec.decode(c)
    z = codec.dequantize_z(kz, a_z, b_z)
    last_seen = {}
    first_seen = {}
    for i in range(c.shape[0]):
        cell = (int(ky[i]), int(kx[i]))
        last_seen[cell] = float(z[i])  # overwritten every time -> ends up the LAST
        if cell not in first_seen:
            first_seen[cell] = float(z[i])  # only set once -> stays the FIRST
    last_field = np.zeros((64, 64), dtype=np.float64)
    first_field = np.zeros((64, 64), dtype=np.float64)
    for (cy, cx), v in last_seen.items():
        last_field[cy, cx] = v
    for (cy, cx), v in first_seen.items():
        first_field[cy, cx] = v
    return last_field, first_field


def test_last_and_first_match_brute_force_with_multiple_samples_per_cell():
    rng = np.random.default_rng(42)
    codes = _multi_sample_codes(rng)
    # sanity: every cell was hit at least once (base pass covers all 4096) -> fill is a no-op
    kx, ky, _ = codec.decode(codes)
    hit = np.zeros((64, 64), dtype=bool)
    hit[ky, kx] = True
    assert hit.all(), "test construction bug: not every cell was hit by the base pass"

    last_field = agg_variants.aggregate_last(codes, A_Z, B_Z)
    first_field = agg_variants.aggregate_first(codes, A_Z, B_Z)
    brute_last, brute_first = _brute_force_last_and_first(codes, A_Z, B_Z)

    d_last = float(np.max(np.abs(last_field.astype(np.float64) - brute_last)))
    d_first = float(np.max(np.abs(first_field.astype(np.float64) - brute_first)))
    print("[test b] max|aggregate_last - brute-force last|   = %.3e" % d_last)
    print("[test b] max|aggregate_first - brute-force first| = %.3e" % d_first)
    assert d_last < 1e-5, "aggregate_last != brute-force last-per-cell: %.3e" % d_last
    assert d_first < 1e-5, "aggregate_first != brute-force first-per-cell: %.3e" % d_first
    # and last should differ from first somewhere -- otherwise this test data is degenerate
    assert not np.allclose(last_field, first_field), "test construction bug: last == first everywhere"


def test_random_one_picks_a_member_of_the_cells_own_observed_set():
    rng = np.random.default_rng(7)
    codes = _multi_sample_codes(rng)
    kx, ky, kz = codec.decode(codes)
    z = codec.dequantize_z(kz, A_Z, B_Z)
    observed = {}
    for i in range(codes.shape[0]):
        observed.setdefault((int(ky[i]), int(kx[i])), []).append(float(z[i]))

    random_field = agg_variants.aggregate_random_one(codes, A_Z, B_Z, np.random.default_rng(123))
    assert random_field.shape == (64, 64)
    assert np.isfinite(random_field).all()

    # membership check with a float32-rounding-sized tolerance (random_field went through
    # _fill_empty's astype(np.float32), so an exact float64 equality would spuriously fail)
    n_checked = 0
    for (cy, cx), zs in observed.items():
        val = float(random_field[cy, cx])
        assert any(abs(val - z0) < 1e-5 for z0 in zs), (
            "aggregate_random_one returned %r for cell (%d,%d) not in observed set %r" % (val, cy, cx, zs))
        n_checked += 1
    print("[test c] aggregate_random_one: %d/%d cells checked, all values are members of their "
          "cell's own observed set" % (n_checked, len(observed)))

    random_field_2 = agg_variants.aggregate_random_one(codes, A_Z, B_Z, np.random.default_rng(999))
    same = np.array_equal(random_field, random_field_2)
    print("[test c] two different rng seeds -> identical field? %s (informational, not asserted; "
          "coincidental agreement on some cells is fine)" % same)


def test_shape_finite_with_empty_cells():
    rng = np.random.default_rng(2024)
    n = 5000  # << 4096 cells possible -> many cells guaranteed empty, exercises _fill_empty
    ky = rng.integers(0, codec.GRID, size=n)
    kx = rng.integers(0, codec.GRID, size=n)
    kz = rng.integers(0, codec.GRID, size=n)
    codes = codec.encode(kx, ky, kz)

    kxu, kyu, _ = codec.decode(codes)
    n_hit = len(set(zip(kyu.tolist(), kxu.tolist())))
    print("[test d] %d samples hit %d/4096 cells (rest must come from neighbour/global fill)"
          % (n, n_hit))
    assert n_hit < codec.GRID * codec.GRID, "test construction bug: expected some empty cells"

    for name, field in (
        ("last", agg_variants.aggregate_last(codes, A_Z, B_Z)),
        ("first", agg_variants.aggregate_first(codes, A_Z, B_Z)),
        ("random_one", agg_variants.aggregate_random_one(codes, A_Z, B_Z, np.random.default_rng(3))),
    ):
        assert field.shape == (64, 64), (name, field.shape)
        assert field.dtype == np.float32, (name, field.dtype)
        assert np.isfinite(field).all(), "%s: non-finite values after fill" % name
        lo, hi = float(field.min()), float(field.max())
        print("[test d] %-10s shape=%s dtype=%s range=[%.4f, %.4f]" % (name, field.shape, field.dtype, lo, hi))
        assert lo >= A_Z - 1e-6 and hi <= B_Z + 1e-6, "%s out of [a_z,b_z]: [%.6g,%.6g]" % (name, lo, hi)


if __name__ == "__main__":
    test_one_sample_per_cell_all_conventions_agree()
    test_last_and_first_match_brute_force_with_multiple_samples_per_cell()
    test_random_one_picks_a_member_of_the_cells_own_observed_set()
    test_shape_finite_with_empty_cells()
    print("[test_agg_variants] all tests passed")
