#!/usr/bin/env python3
"""D2 turbulence loader for the arXiv 2603.06440 replication (INTERFACES.md).

Reads the cache written by preprocess_d2.py (grid_xy.npz + zval_all.npy, default
<WORK>/data/turbulence_khojasteh2022/d2_cache/) via memmap -- never loads
the full (1000, 3, 328, 308) array into RAM -- and turns native D2 grid points into
the shared (kx, ky, kz) / 18-bit-code representation from codec.py.

Design decisions (matching docs/IQP_TURBULENCE_REPLICATION_PREREGISTRATION.md,
2026-09-05 entry, and the "Quantization interval" row of the design table):
  - z_val (the quantity quantized into kz / used for `.field`) is the 3D velocity
    magnitude sqrt(u^2 + v^2 + w^2), NOT the spatial z-coordinate of the plane.
  - a_z, b_z (get_z_range) are the empirical min/max of the NATIVE z_val over the
    11 anchor snapshots (t in {1, 100, ..., 1000}) only -- not the whole dataset.
    (INTERFACES.md's "Tuples" bullet and "Data-side single-source rules" #3 now match
    this exactly. FIXED 2026-09-05: every d2-consumer file in this directory --
    train_iqp.py, controls.py, baseline_rbm.py, baseline_dcgan.py, make_ground_truth.py
    -- used to independently RE-DERIVE a_z/b_z from `.field` values at the anchors
    instead of calling get_z_range() directly; `.field` is already a 64x64-bin
    aggregate, whose averaging narrows the range (~18.8% observed: (0.0159, 1.6676) vs
    the true native (0.0, 2.034849)) versus the bounds actually used here to encode
    `.pool_codes` -- an encode/decode scale mismatch that broke every d2 result. All
    five files now call get_z_range(plane) directly; do not reintroduce a `.field`-derived
    a_z/b_z for d2.)
  - a_x, b_x / a_y, b_y are the native grid's own min/max (deterministic, dataset-wide
    by construction since every t shares one grid) -- not anchor-restricted.
  - `plane` selects which of the 3 native z-planes (or all 3 pooled) a Snapshot's
    tuples are drawn from: "low"/"mid"/"high" (z ~= 1.969/3.0/3.984, ascending) or
    "pooled" (all three planes' points as one pool, 303,072 points/snapshot). The
    paper's primary arm is plane="mid" (the default); "pooled" is the sensitivity arm.
  - Snapshot.field is ALWAYS reconstructed FROM Snapshot.pool_codes via codec.aggregate
    (never a separately-quantized copy), so the 64x64 "ground truth" field and the
    tuples every generative arm trains on come from exactly the same source.

Python 3.9 compatible (no `X | None`).
"""
import json
import os

import numpy as np

import codec

CACHE_DIR_DEFAULT = "<WORK>/data/turbulence_khojasteh2022/d2_cache"
ANCHOR_TS = [1, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
N_HOLDOUT = 200
N_SNAPSHOTS = 1000
HOLDOUT_SEED = 0

# Native grid axis sizes (preprocess_d2.py hard-asserts these); plane index is by
# ascending z-value (low=z~1.969, mid=z~3.0, high=z~3.984), matching zval_all's axis-1
# order (preprocess_d2.py sorts planes by their own z column, not file position).
PLANE_INDEX = {"low": 0, "mid": 1, "high": 2}
PLANE_CHOICES = ("low", "mid", "high", "pooled")


class Snapshot(object):
    """One D2 snapshot's public representation (INTERFACES.md snapshot contract).

    .t          : 1-based snapshot index.
    .field      : (64, 64) float32, row = y, col = x. codec.aggregate of .pool_codes.
    .pool_codes : int64 array of native-grid 18-bit codes (encode(kx, ky, kz) for every
                  native (x, y[, plane]) point of this snapshot at the given plane
                  spec) -- the fixed pool codec.sample_codes draws training tuples from.
    """
    __slots__ = ("t", "field", "pool_codes")

    def __init__(self, t, field, pool_codes):
        self.t = t
        self.field = field
        self.pool_codes = pool_codes

    def __repr__(self):
        return "Snapshot(t=%d, field.shape=%s, pool_codes.size=%d)" % (
            self.t, tuple(self.field.shape), self.pool_codes.size,
        )


def _cache_dir(cache_dir):
    return cache_dir if cache_dir else CACHE_DIR_DEFAULT


def _check_plane(plane):
    if plane not in PLANE_CHOICES:
        raise ValueError("plane must be one of %s, got %r" % (PLANE_CHOICES, plane))


def _load_grid_xy(cache_dir):
    path = os.path.join(cache_dir, "grid_xy.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "%s not found -- run preprocess_d2.py first (see INTERFACES.md)" % path
        )
    npz = np.load(path)
    return npz["x"].astype(np.float64), npz["y"].astype(np.float64), npz["plane_z"].astype(np.float64)


def _open_zval_memmap(cache_dir):
    path = os.path.join(cache_dir, "zval_all.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "%s not found -- run preprocess_d2.py first (see INTERFACES.md)" % path
        )
    return np.load(path, mmap_mode="r")  # (n_t_cached, 3, 328, 308) float32, read-only


def get_holdout_taus():
    """200 unseen tau, fixed: numpy seed 0, sampled without replacement from
    [1, 1000] minus the 11 anchors, then sorted ascending. Shared by every arm
    (INTERFACES.md / pre-registration); recomputed deterministically each call
    (no I/O, no dependence on --data d2 cache state)."""
    rng = np.random.default_rng(HOLDOUT_SEED)
    pool = np.setdiff1d(
        np.arange(1, N_SNAPSHOTS + 1, dtype=np.int64), np.array(ANCHOR_TS, dtype=np.int64)
    )
    unseen = rng.choice(pool, size=N_HOLDOUT, replace=False)
    return np.sort(unseen).astype(np.int64).tolist()


def get_z_range(plane="mid", cache_dir=None):
    """(a_z, b_z): native z_val (velocity magnitude) min/max over the 11 anchor
    snapshots only, for `plane` ("low"/"mid"/"high"/"pooled"). Computed once and
    cached to <cache_dir>/z_range_<plane>.json so every process sees identical
    fixed bounds without re-scanning the anchors' native grids each time."""
    _check_plane(plane)
    cache_dir = _cache_dir(cache_dir)
    range_path = os.path.join(cache_dir, "z_range_{}.json".format(plane))
    if os.path.exists(range_path):
        with open(range_path) as fh:
            d = json.load(fh)
        return float(d["a_z"]), float(d["b_z"])

    zval_all = _open_zval_memmap(cache_dir)
    chunks = []
    for t in ANCHOR_TS:
        grid_t = np.asarray(zval_all[t - 1])  # (3, ny, nx) float32, copies out of the memmap
        if plane == "pooled":
            chunks.append(grid_t.reshape(-1))
        else:
            chunks.append(grid_t[PLANE_INDEX[plane]].reshape(-1))
    all_vals = np.concatenate(chunks).astype(np.float64)
    a_z, b_z = float(all_vals.min()), float(all_vals.max())
    if b_z <= a_z:
        b_z = a_z + 1.0

    os.makedirs(cache_dir, exist_ok=True)
    with open(range_path, "w") as fh:
        json.dump({"plane": plane, "a_z": a_z, "b_z": b_z, "anchors": ANCHOR_TS}, fh, indent=2)
    return a_z, b_z


def _grid_bounds(x_unique, y_unique):
    return float(x_unique.min()), float(x_unique.max()), float(y_unique.min()), float(y_unique.max())


def _build_snapshot(t, plane, cache_dir, zval_all, x_unique, y_unique, a_x, b_x, a_y, b_y, a_z, b_z):
    """Quantize snapshot `t`'s native grid at `plane` into a Snapshot(field, pool_codes).

    kx/ky come from the SAME grid for every t (deterministic from grid_xy.npz);
    kz comes from this snapshot's own native z_val (velocity magnitude) at `plane`.
    """
    grid_t = np.asarray(zval_all[t - 1])  # (3, ny, nx) float32; copy out of the memmap
    ny, nx = grid_t.shape[1], grid_t.shape[2]

    # kx, ky are identical for every t (fixed grid) -- compute once per call, cheap (ny*nx ~1e5).
    x_grid = np.broadcast_to(x_unique.reshape(1, nx), (ny, nx))
    y_grid = np.broadcast_to(y_unique.reshape(ny, 1), (ny, nx))
    kx_grid = codec.quantize_z(x_grid, a_x, b_x)  # reuse: same floor-clip formula
    ky_grid = codec.quantize_z(y_grid, a_y, b_y)

    if plane == "pooled":
        n_planes = grid_t.shape[0]
        kx = np.tile(kx_grid.ravel(), n_planes)
        ky = np.tile(ky_grid.ravel(), n_planes)
        zflat = grid_t.reshape(n_planes, -1).ravel()
    else:
        p = PLANE_INDEX[plane]
        kx = kx_grid.ravel()
        ky = ky_grid.ravel()
        zflat = grid_t[p].ravel()

    kz = codec.quantize_z(zflat, a_z, b_z)
    pool_codes = codec.encode(kx, ky, kz).astype(np.int64)
    field = codec.aggregate(pool_codes, a_z, b_z)
    return Snapshot(t=int(t), field=field, pool_codes=pool_codes)


def load_d2_snapshot(t, plane="mid", cache_dir=None):
    """One snapshot, built on demand (used by arms that only ever need one t at a
    time, e.g. baseline_rbm.py / baseline_dcgan.py for an arbitrary --n-train)."""
    _check_plane(plane)
    cache_dir = _cache_dir(cache_dir)
    zval_all = _open_zval_memmap(cache_dir)
    x_unique, y_unique, _plane_z = _load_grid_xy(cache_dir)
    a_x, b_x, a_y, b_y = _grid_bounds(x_unique, y_unique)
    a_z, b_z = get_z_range(plane, cache_dir)
    return _build_snapshot(int(t), plane, cache_dir, zval_all, x_unique, y_unique, a_x, b_x, a_y, b_y, a_z, b_z)


def load_d2(plane="mid", cache_dir=None):
    """(snapshots, anchors, holdout_taus).

    snapshots: dict {t: Snapshot} for t in the 11 anchors UNION the 200 fixed
    holdout tau (211 keys) -- the exact set every current consumer (controls.py,
    train_iqp.py, make_ground_truth.py) indexes via `snapshots[t]`. Deliberately NOT
    all 1000 t (would hold ~700 MB+ of pool_codes in RAM for no consumer that needs
    it); arms that need an arbitrary/large t-set (baseline_rbm.py, baseline_dcgan.py
    with --n-train != 11) go through load_d2_snapshot(t) instead, which is backed by
    the same on-disk memmap and computes one snapshot at a time.
    anchors: [1, 100, ..., 1000] (list[int], 1-based, fixed).
    holdout_taus: 200 unseen tau (list[int], 1-based, fixed seed 0), see get_holdout_taus().
    """
    _check_plane(plane)
    cache_dir = _cache_dir(cache_dir)
    zval_all = _open_zval_memmap(cache_dir)
    x_unique, y_unique, _plane_z = _load_grid_xy(cache_dir)
    a_x, b_x, a_y, b_y = _grid_bounds(x_unique, y_unique)
    a_z, b_z = get_z_range(plane, cache_dir)

    holdout_taus = get_holdout_taus()
    ts_needed = sorted(set(ANCHOR_TS) | set(holdout_taus))

    snapshots = {}
    for t in ts_needed:
        snapshots[t] = _build_snapshot(
            t, plane, cache_dir, zval_all, x_unique, y_unique, a_x, b_x, a_y, b_y, a_z, b_z
        )
    return snapshots, list(ANCHOR_TS), holdout_taus


if __name__ == "__main__":
    import sys

    cache_dir = sys.argv[1] if len(sys.argv) > 1 else CACHE_DIR_DEFAULT
    plane = sys.argv[2] if len(sys.argv) > 2 else "mid"

    print("[data_d2] cache_dir =", cache_dir, " plane =", plane)

    holdout = get_holdout_taus()
    print("[data_d2] anchors           :", ANCHOR_TS)
    print("[data_d2] holdout[:10]      :", holdout[:10], "... (n=%d)" % len(holdout))
    assert len(set(ANCHOR_TS) & set(holdout)) == 0, "holdout must not overlap anchors"
    assert len(holdout) == N_HOLDOUT

    a_z, b_z = get_z_range(plane, cache_dir)
    print("[data_d2] a_z, b_z (native, anchors, plane=%s) = %.6f, %.6f" % (plane, a_z, b_z))

    snap1 = load_d2_snapshot(1, plane, cache_dir)
    print("[data_d2] t=1: pool_codes.size =", snap1.pool_codes.size,
          " field shape =", snap1.field.shape, " field min/mean/max = %.4f/%.4f/%.4f"
          % (float(snap1.field.min()), float(snap1.field.mean()), float(snap1.field.max())))

    kx, ky, _kz = codec.decode(snap1.pool_codes)
    counts = np.zeros((64, 64), dtype=np.int64)
    np.add.at(counts, (ky, kx), 1)
    n_bins_covered = int((counts > 0).sum())
    print("[data_d2] (kx,ky) bins covered: %d/4096; per-bin count min/max (over covered bins) = %d/%d"
          % (n_bins_covered, int(counts[counts > 0].min()) if n_bins_covered else -1,
             int(counts.max())))

    snapshots, anchors, holdout_full = load_d2(plane, cache_dir)
    print("[data_d2] load_d2(): %d snapshots built (anchors ∪ holdout), anchors=%d holdout=%d"
          % (len(snapshots), len(anchors), len(holdout_full)))
    print("[data_d2] self-check OK")
