#!/usr/bin/env python3
"""Alternative per-cell aggregation conventions for tuple-based arms (arXiv 2603.06440
replication). The paper never specifies how multiple sampled tuples landing in the same
(kx, ky) grid cell are combined into that cell's single scalar z (spec doc unknown #10).
codec.aggregate -- the convention every trained arm's outputs_d2_main/ actually used --
takes the MEAN of all z's landing in a cell. diag_scripts/q1c_agg_convention.py showed that
switching to LAST-WRITE-WINS (keep only the most-recently-drawn sample's z per cell, discard
the rest) narrows the sampled histogram back toward the true field's spread: on the same IQP
samples, PDF-JS moved from ~0.24 (mean) to ~0.028/0.045 (last), much closer to the paper's
reported ~0.024/0.022 on D2.

This module generalizes that one-off diagnostic into three named "keep exactly ONE tuple per
cell" conventions -- aggregate_last (the one implicated above), plus aggregate_first and
aggregate_random_one as controls/sanity checks -- without touching codec.py (in active use by
running jobs; codec.aggregate itself is untouched and still the MEAN baseline).

All three share codec.aggregate's own empty-cell fill (iterative 8-neighbour mean, falling
back to the global mean of the filled field), so cells with zero draws are handled identically
regardless of which single-sample convention picked the survivor for the non-empty cells.

Python 3.9 compatible (no `X | None`).
"""
from __future__ import annotations

import numpy as np

import codec


def _fill_empty(field, a_z, b_z, max_iter=100):
    """codec.aggregate's empty-cell fill, factored out so it can be applied to a field
    that was already built by some OTHER per-cell reduction than the mean."""
    field = np.asarray(field, dtype=np.float64).copy()
    gmean = np.nanmean(field) if np.isfinite(field).any() else 0.5 * (a_z + b_z)
    grid = codec.GRID
    for _ in range(max_iter):
        empty = ~np.isfinite(field)
        if not empty.any():
            break
        padded = np.pad(field, 1, mode="constant", constant_values=np.nan)
        stack = np.stack([padded[1 + dy:grid + 1 + dy, 1 + dx:grid + 1 + dx]
                          for dy in (-1, 0, 1) for dx in (-1, 0, 1) if not (dy == 0 and dx == 0)])
        with np.errstate(all="ignore"):
            neigh = np.nanmean(stack, axis=0)
        fill = np.isfinite(neigh) & empty
        field = np.where(fill, neigh, field)
    field = np.where(np.isfinite(field), field, gmean)
    return field.astype(np.float32)


def aggregate_last(codes, a_z, b_z, max_iter=100):
    """Per (ky, kx) cell, keep the z of the LAST sample -- in `codes`' own array order --
    that lands in that cell; every earlier sample in the same cell is discarded (not
    averaged). Matches diag_scripts/q1c_agg_convention.py's agg_last(): numpy's fancy-index
    assignment `flat[lin] = z` keeps, per repeated `lin` value, the entry at the HIGHEST
    position in the `lin`/`z` arrays as passed in -- i.e. the one that appears last in
    `codes`."""
    c = np.asarray(codes, dtype=np.int64)
    kx, ky, kz = codec.decode(c)
    z = codec.dequantize_z(kz, a_z, b_z)
    lin = ky * codec.GRID + kx
    flat = np.full(codec.GRID * codec.GRID, np.nan, dtype=np.float64)
    flat[lin] = z  # last-write-wins per repeated cell index, see docstring
    field = flat.reshape(codec.GRID, codec.GRID)
    return _fill_empty(field, a_z, b_z, max_iter=max_iter)


def aggregate_first(codes, a_z, b_z, max_iter=100):
    """Per cell, keep the z of the FIRST sample (in `codes`' own array order) instead of
    the last. Implemented as aggregate_last on the reversed array: reversing puts the true
    first occurrence at the highest position within the reversed array, so last-write-wins
    on the reversed array recovers first-write-wins on the original."""
    c = np.asarray(codes, dtype=np.int64)
    return aggregate_last(c[::-1], a_z, b_z, max_iter=max_iter)


def aggregate_random_one(codes, a_z, b_z, rng, max_iter=100):
    """Per cell, keep the z of ONE uniformly-random sample among those landing in that
    cell. Implemented by shuffling the samples with `rng` and then applying
    aggregate_last: a uniform random permutation restricted to any fixed subset (the
    samples sharing one cell) is itself uniform over that subset's orderings, so "last in
    the shuffled order" is a uniformly random pick among that cell's samples."""
    c = np.asarray(codes, dtype=np.int64)
    perm = rng.permutation(c.shape[0])
    return aggregate_last(c[perm], a_z, b_z, max_iter=max_iter)
