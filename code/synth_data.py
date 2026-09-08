"""Synthetic stand-in snapshots so every module can be smoke-tested before the real D2 data lands.

Contract (INTERFACES.md): a snapshot is float32 (64, 64), row = y, col = x, values ~ [0, 1].
The field is a sum of a few Gaussian blobs whose centres / widths / amplitudes drift smoothly with
time, plus a slow global drift, so that consecutive snapshots are strongly correlated (which is what
the IQP latent-adaptation / interpolation scheme assumes).

Python 3.9 compatible.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

import codec

GRID = 64
N_BLOBS = 5
#: nominal anchor times of the paper (11 snapshots: t = 1, 100, 200, ..., 1000)
NOMINAL_ANCHORS = (1,) + tuple(range(100, 1001, 100))
N_HOLDOUT = 200

#: Shared defaults so every arm (controls.py, baseline_rbm.py, baseline_dcgan.py,
#: train_iqp.py) builds the SAME synthetic ground truth and the SAME anchor/held-out
#: split for --data synthetic, regardless of each arm's own --seed (which only
#: controls that arm's model-training stochasticity, not the data -- exactly like the
#: real D2 dataset does not change with a model's training seed).
N_SNAPSHOTS_SYNTH = 1000
SYNTH_DATA_SEED = 0


def make_synthetic_snapshots(n_snapshots: int, seed: int = 0) -> np.ndarray:
    """(n_snapshots, 64, 64) float32 field in ~[0, 1], smoothly varying in time."""
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(np.linspace(0.0, 1.0, GRID), np.linspace(0.0, 1.0, GRID), indexing="ij")

    cx0 = rng.uniform(0.2, 0.8, size=N_BLOBS)
    cy0 = rng.uniform(0.2, 0.8, size=N_BLOBS)
    vx = rng.uniform(-0.25, 0.25, size=N_BLOBS)
    vy = rng.uniform(-0.25, 0.25, size=N_BLOBS)
    w0 = rng.uniform(0.08, 0.22, size=N_BLOBS)
    dw = rng.uniform(-0.04, 0.04, size=N_BLOBS)
    amp = rng.uniform(0.4, 1.0, size=N_BLOBS)
    phase = rng.uniform(0.0, 2.0 * np.pi, size=N_BLOBS)
    freq = rng.uniform(0.5, 2.0, size=N_BLOBS)

    out = np.zeros((n_snapshots, GRID, GRID), dtype=np.float64)
    denom = max(n_snapshots - 1, 1)
    for i in range(n_snapshots):
        s = i / denom  # normalized time in [0, 1]
        f = np.zeros((GRID, GRID), dtype=np.float64)
        for b in range(N_BLOBS):
            cx = 0.5 + (cx0[b] - 0.5 + vx[b] * s)
            cy = 0.5 + (cy0[b] - 0.5 + vy[b] * s)
            w = max(w0[b] + dw[b] * s, 0.04)
            a = amp[b] * (0.75 + 0.25 * np.sin(2.0 * np.pi * freq[b] * s + phase[b]))
            f += a * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * w * w))
        f += 0.10 * np.sin(2.0 * np.pi * (xx + 0.3 * s)) * np.cos(2.0 * np.pi * (yy - 0.2 * s))
        out[i] = f

    lo, hi = out.min(), out.max()
    out = (out - lo) / max(hi - lo, 1e-12)
    return out.astype(np.float32)


def make_split(n_snapshots: int, seed: int = 0, n_holdout: int = N_HOLDOUT) -> Tuple[np.ndarray, np.ndarray]:
    """Anchor indices and held-out (unseen) indices into a snapshot array of length n_snapshots.

    The paper trains on 11 anchors t in {1, 100, 200, ..., 1000} out of 1000 snapshots and evaluates
    on 200 unseen times.  Here the nominal anchor times are mapped linearly onto the available index
    range and de-duplicated, and `n_holdout` unseen indices are drawn (fixed seed) from the rest.
    Returned values are 0-based indices into the snapshot array; they double as the time label tau
    used in the `generated_t{tau}.npy` file names.
    """
    if n_snapshots < 2:
        raise ValueError("need at least 2 snapshots")
    span = float(max(NOMINAL_ANCHORS) - min(NOMINAL_ANCHORS))
    anchors = []
    for t in NOMINAL_ANCHORS:
        idx = int(round((t - min(NOMINAL_ANCHORS)) / span * (n_snapshots - 1)))
        idx = int(np.clip(idx, 0, n_snapshots - 1))
        if idx not in anchors:
            anchors.append(idx)
    anchors_arr = np.asarray(sorted(anchors), dtype=np.int64)

    rest = np.setdiff1d(np.arange(n_snapshots, dtype=np.int64), anchors_arr)
    rng = np.random.default_rng(seed + 12345)
    k = min(int(n_holdout), int(rest.size))
    unseen = np.sort(rng.choice(rest, size=k, replace=False)) if k > 0 else np.zeros(0, dtype=np.int64)
    return anchors_arr, unseen.astype(np.int64)


def make_split_truncated(n_snapshots: int, seed: int = SYNTH_DATA_SEED, n_holdout: int = N_HOLDOUT,
                          anchors_max=None) -> Tuple[np.ndarray, np.ndarray]:
    """`make_split`, plus the smoke-test truncation every arm needs identically.

    When `anchors_max` is a positive int, only the first `anchors_max` (sorted) anchors
    are kept and the held-out indices are re-drawn (same seed + 12345 offset) from the
    now-shorter [anchors[0], anchors[-1]] span, so a smoke run still exercises
    interpolation/synthesis with a tiny anchor set. `anchors_max` None/<=0 is a no-op
    (full split, matching plain `make_split`). This is the ONE place that logic lives --
    every caller (controls.py, baseline_rbm.py, baseline_dcgan.py, train_iqp.py) must
    call this instead of re-implementing the truncation, or their smoke-mode anchors/
    held-out tau will silently diverge from each other.
    """
    anchors, unseen = make_split(n_snapshots, seed=seed, n_holdout=n_holdout)
    if anchors_max is not None and anchors_max > 0 and anchors_max < anchors.size:
        anchors = anchors[:anchors_max]
        pool = np.setdiff1d(
            np.arange(int(anchors[0]), int(anchors[-1]) + 1, dtype=np.int64), anchors
        )
        rng = np.random.default_rng(int(seed) + 12345)
        k = min(int(n_holdout), int(pool.size))
        unseen = np.sort(rng.choice(pool, size=k, replace=False)) if k > 0 else np.zeros(0, dtype=np.int64)
    return anchors, unseen.astype(np.int64)


def make_split_labels(n_snapshots: int = N_SNAPSHOTS_SYNTH, seed: int = SYNTH_DATA_SEED,
                       n_holdout: int = N_HOLDOUT, anchors_max=None) -> Tuple[np.ndarray, np.ndarray]:
    """`make_split_truncated`, relabelled to 1-based time t in [1, n_snapshots] (array index = t - 1),
    matching the paper's own t in {1, ..., 1000} convention and every arm's file-naming
    (`generated_t{tau}.npy`, tau = this function's t label).

    This is the ONE split every arm (controls.py, baseline_rbm.py, baseline_dcgan.py,
    train_iqp.py) must call for --data synthetic -- no file draws its own held-out set
    (no independent `seed + 12345` / `+ 999` / `+ 10_000` RNG anywhere else), so every
    arm/seed evaluates against the identical 200 held-out tau.
    """
    anchors_idx, unseen_idx = make_split_truncated(n_snapshots, seed=seed, n_holdout=n_holdout,
                                                     anchors_max=anchors_max)
    return (anchors_idx + 1).astype(np.int64), (unseen_idx + 1).astype(np.int64)


def anchor_a_z_b_z(all_snaps: np.ndarray, anchors) -> Tuple[float, float]:
    """a_z, b_z = min/max of the RAW field values at the given 1-based anchor t's (pre-registration:
    computed once from the training anchors and shared by every arm, incl. DCGAN even when it also
    trains on non-anchor t's -- see `pool_and_quantized_field`)."""
    idx = np.asarray(anchors, dtype=np.int64) - 1
    vals = all_snaps[idx]
    a_z, b_z = float(vals.min()), float(vals.max())
    if b_z <= a_z:
        b_z = a_z + 1.0
    return a_z, b_z


def pool_and_quantized_field(raw_field: np.ndarray, a_z: float, b_z: float):
    """(pool_codes int64 (4096,), quantized field float32 (64, 64)) for one synthetic snapshot.

    The "code pool" every arm must draw its N_t training tuples from (`codec.sample_codes`)
    is every one of the 4096 grid cells' code exactly once (`codec.field_to_codes_full`);
    the "ground truth" field every arm trains on / is evaluated against is that pool's own
    aggregate -- i.e. `raw_field` with z quantized to 6 bits, the resolution every
    generative arm is capped at. (data_d2.py's native ~308x328 grid pool is the
    d2-equivalent of this function and additionally captures real intra-bin z spread;
    see INTERFACES.md.) Audit fix: previously DCGAN trained on the raw, un-quantized
    field -- an unrecorded and unfair precision advantage over IQP/RBM/CTRL-marginal,
    which were always capped at 6-bit z through the codec even before this change.
    """
    pool = codec.field_to_codes_full(raw_field, a_z, b_z)
    field_q = codec.aggregate(pool, a_z, b_z)
    return pool, field_q
