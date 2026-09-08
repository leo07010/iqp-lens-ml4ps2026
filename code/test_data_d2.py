#!/usr/bin/env python3
"""Login-node CPU smoke tests for data_d2.py (D2 turbulence loader, INTERFACES.md).

Everything here finishes in seconds on CPU: it reads the memmap cache written by
preprocess_d2.py, it never trains anything (repo rule: no training/eval on the
login node). If the cache only has a partial number of t (e.g. a --limit smoke
cache), the anchors/holdout-length checks (which need the full 1000-snapshot
fixed split) are SKIPPED rather than failed, and this is printed clearly.

Set D2_CACHE_DIR to point at a specific cache (default: data_d2.CACHE_DIR_DEFAULT,
the full production cache under <WORK>/data/turbulence_khojasteh2022/d2_cache).

Run: /usr/bin/python3 test_data_d2.py
"""
import os
import sys
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import codec  # noqa: E402
import data_d2  # noqa: E402

CACHE_DIR = os.environ.get("D2_CACHE_DIR", data_d2.CACHE_DIR_DEFAULT)
ZIP_PATH = os.environ.get("D2_ZIP_PATH", "<WORK>/data/turbulence_khojasteh2022/2D_velocity_snapshot_sub_domain_2.zip")

FAILURES = []
SKIPPED = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print("[{}] {}{}".format(status, name, (" -- " + detail) if detail else ""))
    if not cond:
        FAILURES.append(name)


def skip(name, reason):
    print("[SKIP] {} -- {}".format(name, reason))
    SKIPPED.append(name)


def _n_t_cached():
    zval = data_d2._open_zval_memmap(CACHE_DIR)
    return zval.shape[0]


def test_field_shape_and_finite():
    for t in (1, 2, 5):
        if t > _n_t_cached():
            continue
        snap = data_d2.load_d2_snapshot(t, plane="mid", cache_dir=CACHE_DIR)
        check("field_shape_t{}".format(t), snap.field.shape == (64, 64), "shape={}".format(snap.field.shape))
        check("field_dtype_t{}".format(t), snap.field.dtype == np.float32, "dtype={}".format(snap.field.dtype))
        check("field_finite_t{}".format(t), bool(np.isfinite(snap.field).all()))
        check("pool_codes_dtype_t{}".format(t), snap.pool_codes.dtype == np.int64)
        check("pool_codes_nonempty_t{}".format(t), snap.pool_codes.size > 0, "size={}".format(snap.pool_codes.size))


def test_pool_codes_cover_all_bins():
    """kx,ky decoded from a snapshot's native pool must cover all 64x64 = 4096
    (kx, ky) bins with no gaps, and per-bin counts should be roughly the grid's
    native resolution / 64 in each axis (~4.8 x's, ~5.1 y's => ~20-30 points/bin,
    per the pre-registration's "~24 native points per 64x64 cell")."""
    snap = data_d2.load_d2_snapshot(1, plane="mid", cache_dir=CACHE_DIR)
    kx, ky, _kz = codec.decode(snap.pool_codes)
    check("kx_range", int(kx.min()) == 0 and int(kx.max()) == 63, "min={} max={}".format(kx.min(), kx.max()))
    check("ky_range", int(ky.min()) == 0 and int(ky.max()) == 63, "min={} max={}".format(ky.min(), ky.max()))

    counts = np.zeros((64, 64), dtype=np.int64)
    np.add.at(counts, (ky, kx), 1)
    n_empty = int((counts == 0).sum())
    check("all_4096_bins_covered", n_empty == 0, "{} empty bins".format(n_empty))
    if n_empty == 0:
        cmin, cmax = int(counts.min()), int(counts.max())
        print("    per-bin point count: min={} max={} mean={:.2f}".format(cmin, cmax, float(counts.mean())))
        check("per_bin_count_in_expected_range", 15 <= cmin and cmax <= 40,
              "min={} max={} (expected roughly 20-30)".format(cmin, cmax))


def test_field_matches_direct_native_average():
    """Independently recompute, for plane='mid', the mean of the RAW (unquantized)
    z_val per (kx, ky) spatial bin directly from the cached native grid (NOT via
    codec.aggregate/pool_codes), and compare to snap.field. snap.field averages
    each point's DEQUANTIZED (bin-centre) z, so it can differ from the raw-value
    average by up to half a z-quantization bin width; anything beyond that would
    indicate a real bug (wrong axis order, off-by-one indexing, wrong a_z/b_z)."""
    plane = "mid"
    snap = data_d2.load_d2_snapshot(1, plane=plane, cache_dir=CACHE_DIR)
    x_unique, y_unique, _plane_z = data_d2._load_grid_xy(CACHE_DIR)
    a_x, b_x, a_y, b_y = data_d2._grid_bounds(x_unique, y_unique)
    a_z, b_z = data_d2.get_z_range(plane, CACHE_DIR)

    zval_all = data_d2._open_zval_memmap(CACHE_DIR)
    grid_t = np.asarray(zval_all[0])[data_d2.PLANE_INDEX[plane]]  # (ny, nx) raw z_val, t=1

    ny, nx = grid_t.shape
    x_grid = np.broadcast_to(x_unique.reshape(1, nx), (ny, nx))
    y_grid = np.broadcast_to(y_unique.reshape(ny, 1), (ny, nx))
    kx_grid = codec.quantize_z(x_grid, a_x, b_x)
    ky_grid = codec.quantize_z(y_grid, a_y, b_y)

    ssum = np.zeros((64, 64), dtype=np.float64)
    cnt = np.zeros((64, 64), dtype=np.float64)
    np.add.at(ssum, (ky_grid.ravel(), kx_grid.ravel()), grid_t.ravel().astype(np.float64))
    np.add.at(cnt, (ky_grid.ravel(), kx_grid.ravel()), 1.0)
    direct_mean = np.where(cnt > 0, ssum / np.maximum(cnt, 1), np.nan)

    covered = cnt > 0
    check("direct_average_covers_all_bins", bool(covered.all()), "{} empty".format(int((~covered).sum())))

    half_bin = 0.5 * (b_z - a_z) / 64.0
    diff = np.abs(direct_mean[covered] - snap.field[covered])
    max_diff = float(diff.max()) if diff.size else 0.0
    check("field_matches_direct_average_within_half_bin", max_diff <= half_bin + 1e-6,
          "max_diff={:.6g} half_bin={:.6g}".format(max_diff, half_bin))


def test_t_is_1_based():
    """t=1 must map to zval_all index 0 (and to file member t=1 in the zip), not
    index 1. Cross-checks the preprocessing cache directly against the raw zip
    (independent of data_d2.py's own quantization) when the zip is available."""
    zval_all = data_d2._open_zval_memmap(CACHE_DIR)
    cached_t1 = np.asarray(zval_all[0])
    cached_t2 = np.asarray(zval_all[1]) if zval_all.shape[0] > 1 else None
    if cached_t2 is not None:
        check("t1_and_t2_differ", not np.allclose(cached_t1, cached_t2),
              "t=1 and t=2 cached grids should not be identical")

    if not os.path.exists(ZIP_PATH):
        skip("t1_matches_raw_zip", "zip not found at {}".format(ZIP_PATH))
        return
    try:
        with zipfile.ZipFile(ZIP_PATH) as zf:
            with zf.open("2D_DNS_3900/2D_sub_domain_2_z=[-1,0,1]_t=1.txt") as f:
                raw = f.read().decode("ascii")
    except KeyError as exc:
        skip("t1_matches_raw_zip", "member not found: {}".format(exc))
        return

    lines = [ln for ln in raw.split("\n") if ln and not ln.startswith("#")]
    x, y, z, u, v, w = [float(v) for v in lines[0].split()]
    zmag = (u * u + v * v + w * w) ** 0.5
    x_unique, y_unique, plane_z = data_d2._load_grid_xy(CACHE_DIR)
    xi = int(np.searchsorted(x_unique, x))
    yi = int(np.searchsorted(y_unique, y))
    pi = int(np.searchsorted(plane_z, z))
    cached_val = float(cached_t1[pi, yi, xi])
    check("t1_matches_raw_zip", abs(cached_val - zmag) < 1e-4,
          "cached={:.6f} raw_zmag={:.6f}".format(cached_val, zmag))


def test_holdout_and_anchors():
    holdout = data_d2.get_holdout_taus()
    check("holdout_length_200", len(holdout) == data_d2.N_HOLDOUT, "len={}".format(len(holdout)))
    check("holdout_sorted", holdout == sorted(holdout))
    overlap = set(data_d2.ANCHOR_TS) & set(holdout)
    check("holdout_anchors_disjoint", len(overlap) == 0, "overlap={}".format(overlap))
    check("holdout_in_range", all(1 <= t <= 1000 for t in holdout))
    check("holdout_deterministic", data_d2.get_holdout_taus() == holdout, "two calls must agree (fixed seed)")

    n_t = _n_t_cached()
    if n_t < 1000:
        skip("load_d2_full_split", "cache only has {} t (need 1000 for a real load_d2() call)".format(n_t))
        return
    snapshots, anchors, holdout_taus = data_d2.load_d2(plane="mid", cache_dir=CACHE_DIR)
    check("load_d2_anchors_match", anchors == data_d2.ANCHOR_TS)
    check("load_d2_holdout_match", holdout_taus == holdout)
    check("load_d2_snapshot_count", len(snapshots) == len(set(data_d2.ANCHOR_TS) | set(holdout)),
          "got {}".format(len(snapshots)))
    for t in data_d2.ANCHOR_TS[:2] + holdout[:2]:
        check("load_d2_dict_has_t{}".format(t), t in snapshots)


def main():
    test_field_shape_and_finite()
    test_pool_codes_cover_all_bins()
    test_field_matches_direct_native_average()
    test_t_is_1_based()
    test_holdout_and_anchors()

    print()
    if SKIPPED:
        print("SKIPPED ({}): {}".format(len(SKIPPED), ", ".join(SKIPPED)))
    if FAILURES:
        print("FAILED ({}): {}".format(len(FAILURES), ", ".join(FAILURES)))
        sys.exit(1)
    else:
        print("ALL TESTS PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
