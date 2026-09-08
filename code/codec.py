"""Shared tuple <-> 18-bit code codec for the 2603.06440 replication (see INTERFACES.md).

Snapshot: float32 (64, 64) field, row = y, col = x.  Tuple (kx, ky, kz) with 6-bit bins each.
code = (kx << 12) | (ky << 6) | kz ; qubit j == bit j (LSB = qubit 0).
Python 3.9 compatible (no `X | None`).
"""
import numpy as np

N_QUBITS = 18
GRID = 64
NCODES = 1 << N_QUBITS


def quantize_z(z, a_z, b_z):
    """Paper eq. 55 (0-based): floor(64 (z - a)/(b - a)), clipped to [0, 63]."""
    k = np.floor(GRID * (np.asarray(z, dtype=np.float64) - a_z) / (b_z - a_z))
    return np.clip(k, 0, GRID - 1).astype(np.int64)


def dequantize_z(kz, a_z, b_z):
    """Bin centre."""
    return a_z + (np.asarray(kz, dtype=np.float64) + 0.5) * (b_z - a_z) / GRID


def encode(kx, ky, kz):
    kx = np.asarray(kx, dtype=np.int64); ky = np.asarray(ky, dtype=np.int64); kz = np.asarray(kz, dtype=np.int64)
    return (kx << 12) | (ky << 6) | kz


def decode(codes):
    codes = np.asarray(codes, dtype=np.int64)
    return (codes >> 12) & 63, (codes >> 6) & 63, codes & 63


def sample_tuples(field, n, rng, a_z, b_z):
    """Draw n grid cells uniformly with replacement; return int64 codes of shape (n,)."""
    field = np.asarray(field)
    assert field.shape == (GRID, GRID), field.shape
    ky = rng.integers(0, GRID, size=n)
    kx = rng.integers(0, GRID, size=n)
    kz = quantize_z(field[ky, kx], a_z, b_z)
    return encode(kx, ky, kz)


def sample_codes(pool_codes, n, rng):
    """Draw n int64 codes WITH replacement from a fixed pool of codes.

    The pool is a snapshot's full set of observed (x, y, z) codes: for synthetic data,
    `field_to_codes_full` (every one of the 4096 grid cells, exactly once); for D2, the
    native ~308x328-grid pool that data_d2.py provides (INTERFACES.md). This is the ONE
    tuple-sampling primitive every arm (IQP, RBM, CTRL-marginal) uses once a pool is
    available, in place of re-quantizing a 64x64 field per draw (`sample_tuples`) -- on
    the native D2 grid a 64x64-bin pool holds a genuine spread of z per bin that
    `sample_tuples` (uniform-cell-then-quantize on the already-binned field) cannot see.
    """
    pool_codes = np.asarray(pool_codes, dtype=np.int64)
    if pool_codes.size == 0:
        raise ValueError("sample_codes: empty code pool")
    idx = rng.integers(0, pool_codes.shape[0], size=int(n))
    return pool_codes[idx]


def field_to_codes_full(field, a_z, b_z):
    """All 4096 cells exactly once (deterministic empirical pmf of a snapshot)."""
    ky, kx = np.meshgrid(np.arange(GRID), np.arange(GRID), indexing="ij")
    kz = quantize_z(np.asarray(field)[ky, kx], a_z, b_z)
    return encode(kx.ravel(), ky.ravel(), kz.ravel())


def codes_to_pmf(codes):
    return np.bincount(np.asarray(codes, dtype=np.int64), minlength=NCODES).astype(np.float64) / len(codes)


def aggregate(codes, a_z, b_z, max_iter=100):
    """Mean decoded z per (ky, kx) cell; empty cells filled from non-empty 8-neighbours, fallback global mean."""
    kx, ky, kz = decode(codes)
    z = dequantize_z(kz, a_z, b_z)
    ssum = np.zeros((GRID, GRID), dtype=np.float64)
    cnt = np.zeros((GRID, GRID), dtype=np.float64)
    np.add.at(ssum, (ky, kx), z)
    np.add.at(cnt, (ky, kx), 1.0)
    field = np.where(cnt > 0, ssum / np.maximum(cnt, 1), np.nan)
    gmean = np.nanmean(field) if np.isfinite(field).any() else 0.5 * (a_z + b_z)
    for _ in range(max_iter):
        empty = ~np.isfinite(field)
        if not empty.any():
            break
        padded = np.pad(field, 1, mode="constant", constant_values=np.nan)
        stack = np.stack([padded[1 + dy:GRID + 1 + dy, 1 + dx:GRID + 1 + dx]
                          for dy in (-1, 0, 1) for dx in (-1, 0, 1) if not (dy == 0 and dx == 0)])
        with np.errstate(all="ignore"):
            neigh = np.nanmean(stack, axis=0)
        fill = np.isfinite(neigh) & empty
        field = np.where(fill, neigh, field)
    field = np.where(np.isfinite(field), field, gmean)
    return field.astype(np.float32)
