"""Full-frame (64x64) view of the round-3 lens split, for the 12-qubit fine-grid arm.

WHY THIS FILE EXISTS
--------------------
Every round-3 arm models a 32x32 centroid-aligned CROP (data_lens_r3.align_crop) and encodes
it with codec10 on 10 qubits (1,024 codes).  The v11 paper's stated limitation is that the
crop throws away the outer frame and that 1,024 codes is a coarse grid.  The 12-qubit arm
tests the second half of that claim: 4,096 codes = the WHOLE 64x64 frame at full resolution,
codec12's code = (kx << 6) | ky, no cropping at all.

WHAT IS AND IS NOT SHARED WITH data_lens_r3
-------------------------------------------
SHARED, by construction rather than by copy: this module calls data_lens_r3.load_r3() and
uses the images it returns, so the split (R3_SPLIT_SEED = 0), the 100-image training pool,
the 2,000-image latent pool it is disjoint from, and the SAME 200 held-out frames are
literally the same arrays the 10-qubit arms train and are scored on.  Nothing here re-draws
an index.

DIFFERENT: no 32x32 window.  Each 64x64 frame is centroid-aligned by an INTEGER whole-frame
translation (the same "no interpolation, no invented light" rule align_crop follows) so that
the intensity centroid lands on the frame centre, and the pmf is codec12's pmf of that whole
aligned frame.  The translation is recorded per image, exactly as align_crop records its
(dy, dx), and generation draws a translation from the training ones with replacement and
applies its inverse -- the whole-frame analogue of codec10.paste, so the POSITION
distribution of generated frames matches the data's the same way round 3's does.  Without it
every generated frame would be centroid-centred while the held-out frames are not, and the
frame-level Conv2D-MMD^2 would be measuring that artefact instead of the model.

S is the mean total intensity of the ALIGNED TRAINING frames, the 64x64 counterpart of
data_lens_r3's mean crop total (which is 0.53% smaller, that being the light the crop drops).

Python 3.9 compatible (no `X | None`).
"""
from __future__ import annotations

import numpy as np

import codec12
import data_lens_r3
from data_lens_r3 import R3_SPLIT_SEED, N_TRAIN_MAX, N_LATENT_POOL, N_HOLDOUT, centroid

__all__ = ["FRAME", "NCODES", "R3_SPLIT_SEED", "N_TRAIN_MAX", "N_LATENT_POOL", "N_HOLDOUT",
           "MAX_SHIFT", "align_shift", "shift_image", "unshift_image", "align_frame",
           "N12Data", "load_n12", "sample_shifts"]

FRAME = codec12.GRID            # 64
NCODES = codec12.NCODES         # 4096
CENTRE = 0.5 * (FRAME - 1)      # 31.5
MAX_SHIFT = FRAME // 4          # 16; the observed |shift| is <= 6, the clamp never bites


# --------------------------------------------------------------------------------------
# whole-frame integer alignment
# --------------------------------------------------------------------------------------
def align_shift(img):
    """(sy, sx): the INTEGER translation that puts the intensity centroid on the centre.

    sy = round(CENTRE - cy), clamped to +-MAX_SHIFT.  Integer, so no interpolation and no
    invented light -- the same rule data_lens_r3.align_crop uses for its window corner.
    """
    cy, cx = centroid(img)
    sy = int(np.clip(int(np.round(CENTRE - cy)), -MAX_SHIFT, MAX_SHIFT))
    sx = int(np.clip(int(np.round(CENTRE - cx)), -MAX_SHIFT, MAX_SHIFT))
    return sy, sx


def shift_image(img, sy, sx):
    """Translate by (+sy, +sx) into a black frame of the same size (no wrap-around).

    out[y + sy, x + sx] = img[y, x].  Light pushed past an edge is dropped, exactly as
    align_crop drops what falls outside its window; the loss is reported by load_n12 as
    meta["align_light_fraction"].
    """
    img = np.asarray(img, dtype=np.float32)
    ny, nx = img.shape
    out = np.zeros((ny, nx), dtype=np.float32)
    ys0, ys1 = max(0, sy), min(ny, ny + sy)
    xs0, xs1 = max(0, sx), min(nx, nx + sx)
    if ys1 > ys0 and xs1 > xs0:
        out[ys0:ys1, xs0:xs1] = img[ys0 - sy:ys1 - sy, xs0 - sx:xs1 - sx]
    return out


def unshift_image(img, sy, sx):
    """Inverse of shift_image (exact wherever shift_image did not push light off the edge)."""
    return shift_image(img, -int(sy), -int(sx))


def align_frame(img):
    """(aligned64 float32, sy, sx) -- the whole-frame analogue of align_crop."""
    sy, sx = align_shift(img)
    return shift_image(img, sy, sx), sy, sx


def _aligned_of(images):
    """(aligned, shifts, totals) for a stack of frames; n = 0 is legal."""
    n = int(images.shape[0])
    if n == 0:
        return (np.empty((0, FRAME, FRAME), dtype=np.float32),
                np.empty((0, 2), dtype=np.int64),
                np.empty((0,), dtype=np.float64))
    out = np.empty((n, FRAME, FRAME), dtype=np.float32)
    shifts = np.empty((n, 2), dtype=np.int64)
    for i in range(n):
        a, sy, sx = align_frame(images[i])
        out[i] = a
        shifts[i] = (sy, sx)
    totals = np.asarray(out, dtype=np.float64).reshape(n, -1).sum(axis=1)
    return out, shifts, totals


# --------------------------------------------------------------------------------------
# dataset
# --------------------------------------------------------------------------------------
class N12Data(object):
    """The 12-qubit view of the round-3 split.

    train_images   (N, 64, 64) float32   raw frames, byte-identical to R3Data.train_images
    train_aligned  (N, 64, 64) float32   centroid-aligned by an integer whole-frame shift
    train_pmfs     (N, 4096)   float64   codec12 pmf of each aligned frame (rows sum to 1)
    train_shifts   (N, 2)      int64     (sy, sx) applied by the alignment; generation draws
                                         from these with replacement and applies the inverse
    train_totals   (N,)        float64   per-frame total intensity AFTER alignment
    S              float                 mean of train_totals -- the one scale the arm uses
    holdout_* : the same five for the 200 held-out frames (holdout_images is byte-identical
                to R3Data.holdout_images, so the arm is scored against the same ground truth)
    """
    __slots__ = ("n_train", "train_images", "train_aligned", "train_pmfs", "train_shifts",
                 "train_totals", "S", "holdout_images", "holdout_aligned", "holdout_pmfs",
                 "holdout_shifts", "holdout_totals", "meta")

    def __repr__(self):
        return "N12Data(n_train=%d, S=%.4f, holdout=%d)" % (
            self.n_train, self.S, self.holdout_images.shape[0])


def load_n12(n_train, seed=0, n_holdout=N_HOLDOUT, cache_dir=None):
    """The round-3 dataset seen at 64x64 / 4,096 codes.  `seed` is accepted for signature
    symmetry with every other loader and, as in data_lens_r3.load_r3, does NOT move the
    split: the images come straight from load_r3, whose split seed is fixed at 0."""
    d3 = data_lens_r3.load_r3(n_train, seed=seed, n_holdout=n_holdout, cache_dir=cache_dir)
    tr_img = d3.train_images
    ho_img = d3.holdout_images

    tr_al, tr_sh, tr_tot = _aligned_of(tr_img)
    ho_al, ho_sh, ho_tot = _aligned_of(ho_img)

    n = int(tr_img.shape[0])
    n_ho = int(ho_img.shape[0])
    tr_pmf = (np.stack([codec12.image_to_pmf(tr_al[i]) for i in range(n)], axis=0)
              if n else np.zeros((0, NCODES), dtype=np.float64))
    ho_pmf = (np.stack([codec12.image_to_pmf(ho_al[i]) for i in range(n_ho)], axis=0)
              if n_ho else np.zeros((0, NCODES), dtype=np.float64))

    raw_tot = np.asarray(tr_img, dtype=np.float64).reshape(n, -1).sum(axis=1)

    d = N12Data()
    d.n_train = n
    d.train_images = tr_img
    d.train_aligned = tr_al
    d.train_pmfs = tr_pmf
    d.train_shifts = tr_sh
    d.train_totals = tr_tot
    d.S = float(tr_tot.mean())
    d.holdout_images = ho_img
    d.holdout_aligned = ho_al
    d.holdout_pmfs = ho_pmf
    d.holdout_shifts = ho_sh
    d.holdout_totals = ho_tot
    d.meta = {"n_train": n, "n_holdout": n_ho, "split_seed": R3_SPLIT_SEED,
              "seed_arg": int(seed), "S": d.S, "frame": FRAME, "n_codes": NCODES,
              "crop": None, "encoding": "codec12 full frame, code = (kx << 6) | ky",
              "align": "integer whole-frame shift of the centroid to the frame centre",
              "align_light_fraction": float(tr_tot.sum() / max(raw_tot.sum(), 1e-30)),
              "max_abs_shift": int(np.abs(tr_sh).max()) if n else 0,
              "S_crop_r3": float(d3.S)}
    return d


def sample_shifts(train_shifts, n, rng):
    """n alignment shifts drawn WITH REPLACEMENT from the training ones -- the whole-frame
    counterpart of data_lens_r3.sample_offsets, and the only source of generated positions."""
    idx = rng.integers(0, int(train_shifts.shape[0]), size=int(n))
    return np.asarray(train_shifts, dtype=np.int64)[idx]
