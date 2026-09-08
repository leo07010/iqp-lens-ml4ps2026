#!/usr/bin/env python3
"""Round-3 data layer: centroid-aligned 32x32 crops of pooled DeepLense Model II images.

docs/IQP_LENS_ROUND3_PREREGISTRATION.md, "Representation":

  * classes are POOLED (round 3 asks whether the model family can generate plausible lens
    images at all, not whether it can condition on a class), so there is no --lens-class;
  * each 64x64 image is aligned on its intensity centroid (INTEGER shift) and cropped to the
    central 32x32 window; the corner (dy, dx) of that window is recorded per image;
  * the crop is turned into a 1,024-cell pmf by codec10; the total intensity S of the crop is
    carried outside the circuit (one scalar per arm, the mean over the TRAINING crops).

Measured on the 200 held-out images by test_r3.py (the pre-registration quotes 99.4% / 98.4%
from a 3,000-image measurement on val; this file's test re-measures on its own 200):
the aligned crop keeps ~99.4% of the light.

Splits, all fixed at R3_SPLIT_SEED and decoupled from any arm's --seed (data must not move
with a training seed):

    train  : a prefix of ONE permutation of train_x  -> N = 11 is a SUBSET of N = 100
    latent : the NEXT 2,000 entries of the same permutation -> disjoint from every N by
             construction (latent_select_r3.py may not peek at the training images)
    holdout: 200 images from a fixed permutation of val_x

Cache: the source .npz is 1.4 GB and numpy inflates a whole array at a time, so the first call
materialises <WORK>/data/lens_cache/r3_images.npz (2,300 images, ~38 MB) and every
later call/job reads only that.

This file is NEW (round 3). It does not import or modify data_lens.py, which round 2's running
job 347344 is using.

Python 3.9 compatible (no `X | None`).
"""
import json
import os

import numpy as np

import codec10

NPZ_DEFAULT = "<HOME>/mae-lensing/model_II.npz"
CACHE_DIR_DEFAULT = "<WORK>/data/lens_cache"
CACHE_NAME = "r3_images.npz"
CACHE_META = "r3_meta.json"
CACHE_VERSION = 2        # v2: the pooled draws are NOT re-sorted (see build_cache)

R3_SPLIT_SEED = 0        # the ONE split seed
N_TRAIN_MAX = 100        # largest N any arm asks for
N_LATENT_POOL = 2000     # unsupervised latent-selection pool, disjoint from the training set
N_HOLDOUT = 200
CROP = codec10.GRID      # 32
FRAME = codec10.FRAME    # 64


# --------------------------------------------------------------------------------------
# alignment
# --------------------------------------------------------------------------------------
def centroid(img):
    """(cy, cx) intensity-weighted centroid of a 2-D non-negative array, in pixel units."""
    w = np.clip(np.asarray(img, dtype=np.float64), 0.0, None)
    total = float(w.sum())
    if not (total > 0):
        raise ValueError("centroid: total intensity must be > 0, got %r" % total)
    ny, nx = w.shape
    ys = np.arange(ny, dtype=np.float64)
    xs = np.arange(nx, dtype=np.float64)
    cy = float((w.sum(axis=1) * ys).sum() / total)
    cx = float((w.sum(axis=0) * xs).sum() / total)
    return cy, cx


def align_crop(img, crop=CROP):
    """(crop32, dy, dx): the `crop`x`crop` window centred on the intensity centroid.

    The shift is INTEGER (no interpolation, so no light is invented or smoothed):
        dy = round(cy - (crop-1)/2), clamped to [0, FRAME - crop]
    and the same for dx, so the centroid lands within half a pixel of the crop centre
    whenever the clamp does not bite. `paste(crop32, dy, dx)` (codec10.paste) is the exact
    inverse of the windowing.

    Returns (crop32 float32, dy int, dx int).
    """
    img = np.asarray(img, dtype=np.float32)
    if img.ndim != 2 or img.shape[0] != img.shape[1]:
        raise ValueError("align_crop: expected a square 2-D image, got %s" % (img.shape,))
    frame = int(img.shape[0])
    crop = int(crop)
    if crop > frame:
        raise ValueError("align_crop: crop %d larger than frame %d" % (crop, frame))
    cy, cx = centroid(img)
    half = 0.5 * (crop - 1)
    dy = int(np.clip(int(np.round(cy - half)), 0, frame - crop))
    dx = int(np.clip(int(np.round(cx - half)), 0, frame - crop))
    out = np.ascontiguousarray(img[dy:dy + crop, dx:dx + crop]).astype(np.float32)
    return out, dy, dx


def crop_light_fraction(img, crop=CROP):
    """Fraction of the image's total intensity kept by align_crop -- the alignment's only
    lossy step, reported rather than assumed."""
    c, _dy, _dx = align_crop(img, crop=crop)
    total = float(np.asarray(img, dtype=np.float64).sum())
    return float(np.asarray(c, dtype=np.float64).sum() / total) if total > 0 else 0.0


# --------------------------------------------------------------------------------------
# cache
# --------------------------------------------------------------------------------------
def _cache_dir(cache_dir=None):
    if cache_dir:
        return cache_dir
    return os.environ.get("LENS_CACHE_DIR", CACHE_DIR_DEFAULT)


def _npz_path():
    return os.environ.get("LENS_NPZ", NPZ_DEFAULT)


def build_cache(cache_dir=None, npz_path=None, force=False):
    """Materialise the round-3 cache (idempotent). Reads the 1.4 GB source .npz ONCE.

    Writes  <cache>/r3_images.npz  with
        train   (100,  64, 64) float32   the N = 100 training images (prefix -> N = 11)
        latent  (2000, 64, 64) float32   the latent-selection pool, DISJOINT from train
        val     (200,  64, 64) float32   the held-out images
    and <cache>/r3_meta.json with the source indices and the split seed.
    """
    cache_dir = _cache_dir(cache_dir)
    meta_path = os.path.join(cache_dir, CACHE_META)
    data_path = os.path.join(cache_dir, CACHE_NAME)
    if os.path.exists(meta_path) and os.path.exists(data_path) and not force:
        with open(meta_path) as fh:
            meta = json.load(fh)
        if meta.get("cache_version") == CACHE_VERSION:
            return meta

    npz_path = npz_path if npz_path else _npz_path()
    if not os.path.exists(npz_path):
        raise FileNotFoundError("lens dataset not found: %s" % npz_path)
    os.makedirs(cache_dir, exist_ok=True)

    npz = np.load(npz_path)
    class_names = [str(s) for s in np.asarray(npz["class_names"]).tolist()]
    train_y = np.asarray(npz["train_y"]).astype(np.int64)
    val_y = np.asarray(npz["val_y"]).astype(np.int64)
    n_train_all = int(train_y.shape[0])
    n_val_all = int(val_y.shape[0])

    # ONE permutation of the POOLED training set. train prefix, then the latent pool, so the
    # two are disjoint whatever N is asked for later.
    #
    # The indices are kept in PERMUTATION order and deliberately NOT re-sorted. train_y and
    # val_y are blocked by class (all of class 0, then all of class 1, ...), so sorting a
    # pooled draw would make the N = 11 prefix 11 images of ONE class -- the opposite of the
    # pre-registration's "classes pooled" -- and would make holdout[:100] vs holdout[100:]
    # a class-composition contrast rather than a split-half reference. (Measured before the
    # fix: N=11 was 11/11 axion, and the split-half W1(rms radius) floor was 1.36 held-out
    # standard deviations instead of ~0.1.) Round 2's data_lens.py sorts safely because it
    # draws WITHIN a class; this loader draws across all three.
    rng_tr = np.random.default_rng([R3_SPLIT_SEED, 3, 0])
    perm = rng_tr.permutation(n_train_all)
    need = N_TRAIN_MAX + N_LATENT_POOL
    if n_train_all < need:
        raise ValueError("train_x has %d images, need %d" % (n_train_all, need))
    train_idx = perm[:N_TRAIN_MAX]
    latent_idx = perm[N_TRAIN_MAX:need]

    rng_va = np.random.default_rng([R3_SPLIT_SEED, 3, 1])
    if n_val_all < N_HOLDOUT:
        raise ValueError("val_x has %d images, need %d" % (n_val_all, N_HOLDOUT))
    val_idx = rng_va.permutation(n_val_all)[:N_HOLDOUT]

    train_x = np.asarray(npz["train_x"])
    train_arr = train_x[train_idx].astype(np.float32)
    latent_arr = train_x[latent_idx].astype(np.float32)
    del train_x
    val_x = np.asarray(npz["val_x"])
    val_arr = val_x[val_idx].astype(np.float32)
    del val_x

    np.savez_compressed(data_path, train=train_arr, latent=latent_arr, val=val_arr)
    meta = {
        "cache_version": CACHE_VERSION,
        "npz_path": npz_path,
        "class_names": class_names,
        "classes_pooled": True,
        "split_seed": R3_SPLIT_SEED,
        "n_train_max": N_TRAIN_MAX,
        "n_latent_pool": N_LATENT_POOL,
        "n_holdout": N_HOLDOUT,
        "crop": CROP,
        "train_indices": [int(v) for v in train_idx.tolist()],
        "latent_indices": [int(v) for v in latent_idx.tolist()],
        "val_indices": [int(v) for v in val_idx.tolist()],
        "train_class_counts": np.bincount(train_y[train_idx], minlength=len(class_names)).tolist(),
        "train11_class_counts": np.bincount(train_y[train_idx[:11]], minlength=len(class_names)).tolist(),
        "holdout_firsthalf_class_counts": np.bincount(val_y[val_idx[:N_HOLDOUT // 2]],
                                                      minlength=len(class_names)).tolist(),
        "holdout_secondhalf_class_counts": np.bincount(val_y[val_idx[N_HOLDOUT // 2:]],
                                                       minlength=len(class_names)).tolist(),
        "latent_class_counts": np.bincount(train_y[latent_idx], minlength=len(class_names)).tolist(),
        "val_class_counts": np.bincount(val_y[val_idx], minlength=len(class_names)).tolist(),
        "disjoint_train_latent": bool(len(set(train_idx.tolist()) & set(latent_idx.tolist())) == 0),
    }
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)
    return meta


def get_cache_meta(cache_dir=None):
    return build_cache(cache_dir=cache_dir)


def _arrays(cache_dir=None):
    cache_dir = _cache_dir(cache_dir)
    build_cache(cache_dir=cache_dir)
    with np.load(os.path.join(cache_dir, CACHE_NAME)) as z:
        return (np.asarray(z["train"], dtype=np.float32),
                np.asarray(z["latent"], dtype=np.float32),
                np.asarray(z["val"], dtype=np.float32))


# --------------------------------------------------------------------------------------
# public contract
# --------------------------------------------------------------------------------------
class R3Data(object):
    """Everything an arm needs, computed once.

    train_images  (N, 64, 64) float32   the raw training images (CTRL-copy writes these)
    train_crops   (N, 32, 32) float32   centroid-aligned crops
    train_pmfs    (N, 1024)   float64   codec10 pmf of each crop (rows sum to 1)
    train_offsets (N, 2)      int64     (dy, dx) crop corners -- the paste offsets every
                                        generative arm draws from with replacement
    train_totals  (N,)        float64   per-crop total intensity
    S             float                 mean of train_totals: the ONE scale each arm uses
    holdout_images (200, 64, 64) float32  the held-out ground truth (full frames)
    holdout_crops / holdout_offsets / holdout_totals: the same three for the held-out set
                                        (used only by the evaluation's reference rows)
    """
    __slots__ = ("n_train", "train_images", "train_crops", "train_pmfs", "train_offsets",
                 "train_totals", "S", "holdout_images", "holdout_crops", "holdout_offsets",
                 "holdout_totals", "meta")

    def __repr__(self):
        return "R3Data(n_train=%d, S=%.4f, holdout=%d)" % (
            self.n_train, self.S, self.holdout_images.shape[0])


def _crops_of(images):
    if images.shape[0] == 0:            # n_holdout=0 is legal (training-only callers)
        return (np.empty((0, CROP, CROP), dtype=np.float32),
                np.empty((0, 2), dtype=np.int64),
                np.empty((0,), dtype=np.float64))
    crops = np.empty((images.shape[0], CROP, CROP), dtype=np.float32)
    offs = np.empty((images.shape[0], 2), dtype=np.int64)
    for i in range(images.shape[0]):
        c, dy, dx = align_crop(images[i])
        crops[i] = c
        offs[i] = (dy, dx)
    totals = np.asarray(crops, dtype=np.float64).reshape(crops.shape[0], -1).sum(axis=1)
    return crops, offs, totals


def load_r3(n_train, seed=0, n_holdout=N_HOLDOUT, cache_dir=None):
    """The round-3 dataset. `seed` is accepted for signature symmetry with every other loader
    but the SPLIT is fixed at R3_SPLIT_SEED -- the data must not change with an arm's seed."""
    n_train = int(n_train)
    if n_train < 2:
        raise ValueError("need n_train >= 2, got %d" % n_train)
    if n_train > N_TRAIN_MAX:
        raise ValueError("n_train=%d exceeds the cached maximum %d" % (n_train, N_TRAIN_MAX))
    n_holdout = int(min(int(n_holdout), N_HOLDOUT))

    train_all, _latent_all, val_all = _arrays(cache_dir)
    train_images = np.ascontiguousarray(train_all[:n_train])
    holdout_images = np.ascontiguousarray(val_all[:n_holdout])

    tr_crops, tr_off, tr_tot = _crops_of(train_images)
    ho_crops, ho_off, ho_tot = _crops_of(holdout_images)

    pmfs = np.stack([codec10.image_to_pmf(tr_crops[i]) for i in range(n_train)], axis=0)

    d = R3Data()
    d.n_train = n_train
    d.train_images = train_images
    d.train_crops = tr_crops
    d.train_pmfs = pmfs
    d.train_offsets = tr_off
    d.train_totals = tr_tot
    d.S = float(tr_tot.mean())
    d.holdout_images = holdout_images
    d.holdout_crops = ho_crops
    d.holdout_offsets = ho_off
    d.holdout_totals = ho_tot
    d.meta = {"n_train": n_train, "n_holdout": n_holdout, "split_seed": R3_SPLIT_SEED,
              "seed_arg": int(seed), "S": d.S, "crop": CROP}
    return d


def latent_pool(n=N_LATENT_POOL, cache_dir=None):
    """(n, 64, 64) float32 images from the latent-selection pool -- images that are NOT in
    any N (guaranteed by the split, checked by test_r3.py)."""
    _train_all, latent_all, _val_all = _arrays(cache_dir)
    n = int(min(int(n), latent_all.shape[0]))
    return np.ascontiguousarray(latent_all[:n])


def latent_pool_pmfs(n=N_LATENT_POOL, cache_dir=None):
    """(n, 1024) float64 codec10 pmfs of the latent pool's aligned crops."""
    imgs = latent_pool(n, cache_dir=cache_dir)
    crops, _off, _tot = _crops_of(imgs)
    return np.stack([codec10.image_to_pmf(crops[i]) for i in range(crops.shape[0])], axis=0)


def sample_offsets(train_offsets, n, rng):
    """(n, 2) int64 paste corners drawn WITH REPLACEMENT from the training images' (dy, dx).
    The one place every generative arm gets its offsets, so they all use the same rule."""
    off = np.asarray(train_offsets, dtype=np.int64)
    idx = rng.integers(0, off.shape[0], size=int(n))
    return off[idx]


# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    meta = build_cache()
    print("[data_lens_r3] cache      =", os.path.join(_cache_dir(), CACHE_NAME))
    print("[data_lens_r3] classes    =", meta["class_names"], "pooled =", meta["classes_pooled"])
    print("[data_lens_r3] class cnts : train", meta["train_class_counts"],
          "(N=11 prefix", meta["train11_class_counts"], ")",
          "latent", meta["latent_class_counts"], "val", meta["val_class_counts"])
    print("[data_lens_r3] holdout halves:", meta["holdout_firsthalf_class_counts"],
          "vs", meta["holdout_secondhalf_class_counts"], "(both must be mixed)")
    print("[data_lens_r3] disjoint train/latent:", meta["disjoint_train_latent"])
    for n in (11, 100):
        d = load_r3(n)
        frac = np.asarray([crop_light_fraction(d.train_images[i]) for i in range(n)])
        print("[data_lens_r3] N=%-3d S=%.4f  crop light kept mean=%.4f min=%.4f  "
              "offsets dy in [%d, %d] dx in [%d, %d]"
              % (n, d.S, float(frac.mean()), float(frac.min()),
                 int(d.train_offsets[:, 0].min()), int(d.train_offsets[:, 0].max()),
                 int(d.train_offsets[:, 1].min()), int(d.train_offsets[:, 1].max())))
    hf = np.asarray([crop_light_fraction(d.holdout_images[i]) for i in range(d.holdout_images.shape[0])])
    print("[data_lens_r3] holdout crop light kept mean=%.4f min=%.4f" % (float(hf.mean()), float(hf.min())))
    print("[data_lens_r3] self-check OK")
