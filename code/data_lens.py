#!/usr/bin/env python3
"""DeepLense Model II loader for the arXiv 2603.06440 pilot on <user>'s lensing images.

Mirrors data_d2.py's contract exactly (INTERFACES.md, "data_lens.py contract" section)
so every arm's `--data lens` branch is a near-copy of its `--data d2` branch:

    load_lens(cls, n_train, seed=0)      -> (snapshots, anchors, holdout_ids)
    load_lens_snapshot(i, cls, n_train)  -> Snapshot   (same object as snapshots[i])
    get_z_range()                        -> (0.0, 1.0)
    get_holdout_ids(n_holdout=200)       -> [1001, ..., 1200]

Encoding (docs/IQP_LENS_PILOT_PREREGISTRATION.md, "Data"), applied literally:
  - one 64x64 image = one "snapshot"; the paper's tuple (x, y, z) becomes
    (pixel column, pixel row, intensity);
  - kx = column, ky = row, kz = floor(64 * intensity) clipped to [0, 63], i.e.
    (a_z, b_z) = (0, 1) FIXED for every class (so the three classes are comparable and
    no arm may re-derive its own bounds -- the lens analogue of data_d2.get_z_range);
  - the code pool of an image is its 4,096 pixels, exactly once each
    (codec.field_to_codes_full), so `.field` == codec.aggregate(`.pool_codes`) with one
    sample per cell = the bin-centre-dequantized image;
  - `.field` is therefore the QUANTIZED image (6-bit z), the same "ground truth"
    convention every other dataset in this directory uses.

Index convention (the lens analogue of d2's 1-based t):
  - snapshot key 1 .. n_train        = the TRAINING images (drawn from train_x of `cls`);
  - snapshot key 1001 .. 1000+n_hold = the HELD-OUT images (drawn from val_x of `cls`).
    HOLDOUT_ID_OFFSET = 1000 keeps the two ranges disjoint so a single dict can carry
    both, exactly like load_d2()'s (anchors U holdout) dict.
  - The training draw is a PREFIX of a single fixed permutation, so the N = 11 training
    set is a subset of the N = 100 one (evaluate_lens.py relies on this to score both
    regimes against one loaded training array).

Cache: the source .npz is 1.4 GB and numpy has to inflate a whole array at a time, so
the first call materializes a tiny per-class cache (100 train + 200 val images per class
= ~15 MB total) under LENS_CACHE_DIR / CACHE_DIR_DEFAULT and every later call/job reads
only that. Same role as preprocess_d2.py's d2_cache.

Python 3.9 compatible (no `X | None`, no match statements).
"""
import json
import os

import numpy as np

import codec
import codec12

NPZ_DEFAULT = "<HOME>/mae-lensing/model_II.npz"
CACHE_DIR_DEFAULT = "<WORK>/data/lens_cache"

A_Z = 0.0                # fixed for every class (pre-registration "Data")
B_Z = 1.0
N_TRAIN_MAX = 100        # the largest N any arm asks for (the paper's "data-rich" regime)
N_HOLDOUT = 200
HOLDOUT_ID_OFFSET = 1000
SPLIT_SEED = 0           # the ONE split seed; decoupled from any arm's own --seed
CACHE_VERSION = 1

ENCODINGS = ("tuple", "pmf")
PMF_POOL_SIZE = 50000    # N_t of the paper protocol: the `pmf` pool IS a 5e4 draw from p
POOL_SEED = 20260906     # fixes the pmf pool draw; decoupled from any arm's own --seed


class Snapshot(object):
    """One lens image's public representation -- identical attribute contract to
    data_d2.Snapshot (INTERFACES.md snapshot contract), plus two pmf-encoding extras.

    .t          : snapshot key (1..n_train training, 1001.. held-out). Named `t` (not `i`)
                  so d2-shaped code paths need no renaming.
    .field      : (64, 64) float32, row = y, col = x. The QUANTIZED image -- the ground
                  truth every arm is scored against. IDENTICAL under both encodings.
    .pool_codes : int64. encoding="tuple" (round 1, the paper's literal encoding):
                  (4096,), every pixel quantized to an 18-bit (kx, ky, kz) code, each once.
                  encoding="pmf" (round 2): (PMF_POOL_SIZE,), 12-bit (kx, ky) codes drawn
                  with replacement from .pmf at a FIXED seed -- the paper-protocol arm's
                  N_t = 5e4 sample of the data distribution.
    .encoding   : "tuple" or "pmf"; which of the two the pool speaks.
    .pmf        : encoding="pmf" only -- (4096,) float64 summing to 1, .field as a spatial
                  distribution in codec12 flat-index order (index = kx*64 + ky). None for
                  encoding="tuple".
    .total_intensity : encoding="pmf" only -- float, .field.sum(); the scale the circuit
                  does NOT model. None for encoding="tuple".

    Why .pmf is built from .field and not from the raw intensities: .field is the ground
    truth of this whole directory, so pmf_to_image(.pmf, .total_intensity) == .field
    exactly, and a pmf arm's reconstruction error is entirely the model's (test_lens_round2
    checks the identity).
    """
    __slots__ = ("t", "field", "pool_codes", "encoding", "pmf", "total_intensity")

    def __init__(self, t, field, pool_codes, encoding="tuple", pmf=None, total_intensity=None):
        self.t = t
        self.field = field
        self.pool_codes = pool_codes
        self.encoding = encoding
        self.pmf = pmf
        self.total_intensity = total_intensity

    def __repr__(self):
        return "Snapshot(t=%d, field.shape=%s, pool_codes.size=%d, encoding=%s)" % (
            self.t, tuple(self.field.shape), self.pool_codes.size, self.encoding,
        )


# --------------------------------------------------------------------------------------
# cache
# --------------------------------------------------------------------------------------
def _cache_dir(cache_dir=None):
    if cache_dir:
        return cache_dir
    return os.environ.get("LENS_CACHE_DIR", CACHE_DIR_DEFAULT)


def _npz_path():
    return os.environ.get("LENS_NPZ", NPZ_DEFAULT)


def _cache_meta_path(cache_dir):
    return os.path.join(cache_dir, "lens_cache_meta.json")


def build_cache(cache_dir=None, npz_path=None, force=False):
    """Materialize the per-class cache (idempotent). Reads the 1.4 GB .npz ONCE.

    Writes, per class name `c`:
        <cache>/train_<c>.npy  (N_TRAIN_MAX, 64, 64) float32 raw intensities
        <cache>/val_<c>.npy    (N_HOLDOUT,   64, 64) float32 raw intensities
    plus lens_cache_meta.json recording class_names (in the file's own order), the
    selected source indices, and the split seed.
    """
    cache_dir = _cache_dir(cache_dir)
    npz_path = npz_path if npz_path else _npz_path()
    meta_path = _cache_meta_path(cache_dir)
    if os.path.exists(meta_path) and not force:
        with open(meta_path) as fh:
            meta = json.load(fh)
        if meta.get("cache_version") == CACHE_VERSION:
            return meta

    if not os.path.exists(npz_path):
        raise FileNotFoundError("lens dataset not found: %s" % npz_path)
    os.makedirs(cache_dir, exist_ok=True)

    npz = np.load(npz_path)
    class_names = [str(s) for s in np.asarray(npz["class_names"]).tolist()]
    train_y = np.asarray(npz["train_y"]).astype(np.int64)
    val_y = np.asarray(npz["val_y"]).astype(np.int64)

    # Pick indices first (cheap, labels only), then inflate the big arrays once each.
    train_idx = {}
    val_idx = {}
    for ci, cname in enumerate(class_names):
        rng_tr = np.random.default_rng([SPLIT_SEED, ci, 0])
        rng_va = np.random.default_rng([SPLIT_SEED, ci, 1])
        pool_tr = np.flatnonzero(train_y == ci)
        pool_va = np.flatnonzero(val_y == ci)
        if pool_tr.size < N_TRAIN_MAX:
            raise ValueError("class %s has only %d train images (< %d)" % (cname, pool_tr.size, N_TRAIN_MAX))
        if pool_va.size < N_HOLDOUT:
            raise ValueError("class %s has only %d val images (< %d)" % (cname, pool_va.size, N_HOLDOUT))
        # permutation prefix: the N=11 training set is a subset of the N=100 one
        train_idx[cname] = np.sort(rng_tr.choice(pool_tr, size=N_TRAIN_MAX, replace=False)).tolist()
        val_idx[cname] = np.sort(rng_va.choice(pool_va, size=N_HOLDOUT, replace=False)).tolist()

    train_x = np.asarray(npz["train_x"])
    for cname in class_names:
        arr = train_x[np.asarray(train_idx[cname], dtype=np.int64)].astype(np.float32)
        np.save(os.path.join(cache_dir, "train_%s.npy" % cname), arr)
    del train_x

    val_x = np.asarray(npz["val_x"])
    for cname in class_names:
        arr = val_x[np.asarray(val_idx[cname], dtype=np.int64)].astype(np.float32)
        np.save(os.path.join(cache_dir, "val_%s.npy" % cname), arr)
    del val_x

    meta = {
        "cache_version": CACHE_VERSION,
        "npz_path": npz_path,
        "class_names": class_names,
        "split_seed": SPLIT_SEED,
        "n_train_max": N_TRAIN_MAX,
        "n_holdout": N_HOLDOUT,
        "a_z": A_Z,
        "b_z": B_Z,
        "train_indices": train_idx,
        "val_indices": val_idx,
    }
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)
    return meta


def get_cache_meta(cache_dir=None):
    return build_cache(cache_dir=cache_dir)


def get_class_names(cache_dir=None):
    """The three class names in the .npz's own order (index i == label i)."""
    return list(get_cache_meta(cache_dir)["class_names"])


def _check_class(cls, cache_dir=None):
    names = get_class_names(cache_dir)
    if cls not in names:
        raise ValueError("--lens-class must be one of %s, got %r" % (names, cls))
    return cls


# --------------------------------------------------------------------------------------
# public contract
# --------------------------------------------------------------------------------------
def get_z_range(plane=None, cache_dir=None):
    """(a_z, b_z) = (0.0, 1.0), FIXED for every class -- the lens analogue of
    data_d2.get_z_range(). `plane` is accepted and ignored so call sites that pass d2's
    positional plane argument keep working."""
    return float(A_Z), float(B_Z)


def get_holdout_ids(n_holdout=N_HOLDOUT):
    """The held-out snapshot keys: 1001 .. 1000 + n_holdout. The ONE held-out source for
    `--data lens` (INTERFACES.md rule 1); no arm may draw its own."""
    n = int(n_holdout) if n_holdout and int(n_holdout) > 0 else 0
    n = min(n, N_HOLDOUT)
    return [HOLDOUT_ID_OFFSET + 1 + j for j in range(n)]


def _raw_arrays(cls, cache_dir=None):
    cache_dir = _cache_dir(cache_dir)
    build_cache(cache_dir=cache_dir)
    _check_class(cls, cache_dir)
    train = np.load(os.path.join(cache_dir, "train_%s.npy" % cls))
    val = np.load(os.path.join(cache_dir, "val_%s.npy" % cls))
    return np.asarray(train, dtype=np.float32), np.asarray(val, dtype=np.float32)


def transform_intensity(img, sqrt=False):
    """The optional intensity transform of the secondary arm: identity, or sqrt.
    Both map [0, 1] -> [0, 1], so (a_z, b_z) = (0, 1) stays valid either way."""
    x = np.asarray(img, dtype=np.float32)
    if sqrt:
        x = np.sqrt(np.clip(x, 0.0, None))
    return x.astype(np.float32)


def invert_intensity(field, sqrt=False):
    """Inverse of transform_intensity -- used by the sqrt arm to write its generated
    fields back into INTENSITY space, so they are scored against the SAME ground truth
    as every other arm (see docs/IQP_LENS_PILOT_PREREGISTRATION.md prediction P5)."""
    x = np.asarray(field, dtype=np.float32)
    if sqrt:
        x = np.square(np.clip(x, 0.0, None))
    return x.astype(np.float32)


def _check_encoding(encoding):
    if encoding not in ENCODINGS:
        raise ValueError("encoding must be one of %s, got %r" % (list(ENCODINGS), encoding))
    return encoding


def _make_snapshot(key, raw_img, sqrt=False, encoding="tuple", pool_size=PMF_POOL_SIZE):
    """Build one Snapshot. `.field` (the ground truth) is identical for both encodings; only
    the code pool differs (see the Snapshot docstring)."""
    _check_encoding(encoding)
    z = transform_intensity(raw_img, sqrt=sqrt)
    codes18 = codec.field_to_codes_full(z, A_Z, B_Z).astype(np.int64)
    field = codec.aggregate(codes18, A_Z, B_Z)
    if encoding == "tuple":
        return Snapshot(t=int(key), field=field, pool_codes=codes18, encoding="tuple")

    # pmf: 12 qubits, the image itself is the distribution over the 4,096 pixels.
    pmf = codec12.image_to_pmf(field)
    total = codec12.total_intensity(field)
    # The pool draw is fixed by (POOL_SEED, key) -- it must not move with an arm's --seed,
    # exactly like the split (rule 2), so every arm sees the same 5e4 codes for this image.
    rng = np.random.default_rng([POOL_SEED, int(key)])
    pool_codes = codec12.sample_codes_from_pmf(pmf, int(pool_size), rng)
    return Snapshot(t=int(key), field=field, pool_codes=pool_codes, encoding="pmf",
                    pmf=pmf, total_intensity=total)


def load_lens(cls, n_train=11, seed=0, n_holdout=N_HOLDOUT, sqrt=False, cache_dir=None,
              encoding="tuple", pool_size=PMF_POOL_SIZE):
    """(snapshots, anchors, holdout_ids) -- the lens analogue of data_d2.load_d2().

    snapshots   : dict {key: Snapshot} over anchors U holdout_ids.
    anchors     : [1, ..., n_train]  (the TRAINING images, in draw order = index order;
                  Algorithm 1 trains theta_core on anchor 1 and warm-starts theta_lat
                  through 2, 3, ..., n_train in exactly this order).
    holdout_ids : [1001, ..., 1000 + n_holdout] (the held-out val images).

    `seed` is accepted for signature symmetry with every other loader but the SPLIT is
    fixed at SPLIT_SEED (like d2's, and like synth_data's SYNTH_DATA_SEED): the data must
    not change with an arm's training seed.

    `encoding` selects the code pool only ("tuple" = round 1's 18-bit (kx, ky, kz) pool,
    "pmf" = round 2's 12-bit (kx, ky) pool drawn from the image-as-pmf); `.field` -- the
    ground truth -- is byte-identical either way.
    """
    cls = str(cls)
    n_train = int(n_train)
    _check_encoding(encoding)
    if n_train < 2:
        raise ValueError("need n_train >= 2 (Algorithm 1 needs a core anchor plus >= 1 latent anchor)")
    if n_train > N_TRAIN_MAX:
        raise ValueError("n_train=%d exceeds the cached maximum %d" % (n_train, N_TRAIN_MAX))
    train_raw, val_raw = _raw_arrays(cls, cache_dir)

    anchors = list(range(1, n_train + 1))
    holdout_ids = get_holdout_ids(n_holdout)

    snapshots = {}
    for k in anchors:
        snapshots[k] = _make_snapshot(k, train_raw[k - 1], sqrt=sqrt,
                                      encoding=encoding, pool_size=pool_size)
    for k in holdout_ids:
        snapshots[k] = _make_snapshot(k, val_raw[k - HOLDOUT_ID_OFFSET - 1], sqrt=sqrt,
                                      encoding=encoding, pool_size=pool_size)
    return snapshots, anchors, holdout_ids


def load_lens_snapshot(i, cls, n_train=N_TRAIN_MAX, sqrt=False, cache_dir=None,
                       encoding="tuple", pool_size=PMF_POOL_SIZE):
    """One snapshot, built on demand -- must return the SAME values as load_lens()'s
    snapshots[i] (data_d2.load_d2_snapshot's contract)."""
    i = int(i)
    _check_encoding(encoding)
    train_raw, val_raw = _raw_arrays(cls, cache_dir)
    kw = {"sqrt": sqrt, "encoding": encoding, "pool_size": pool_size}
    if i > HOLDOUT_ID_OFFSET:
        j = i - HOLDOUT_ID_OFFSET - 1
        if not (0 <= j < val_raw.shape[0]):
            raise KeyError("no held-out snapshot %d" % i)
        return _make_snapshot(i, val_raw[j], **kw)
    if not (1 <= i <= min(int(n_train), train_raw.shape[0])):
        raise KeyError("no training snapshot %d (n_train=%s)" % (i, n_train))
    return _make_snapshot(i, train_raw[i - 1], **kw)


def mean_total_intensity(cls, n_train, sqrt=False, cache_dir=None):
    """S = mean total intensity of the N QUANTIZED training images (Addendum table: the
    scale a pmf arm multiplies its generated q(kx, ky) by). Derived from `.field`, so it
    is the same number whatever the encoding, and it uses ONLY training images."""
    train = raw_training_images(cls, n_train, sqrt=sqrt, cache_dir=cache_dir)
    return float(np.asarray(train, dtype=np.float64).reshape(train.shape[0], -1).sum(axis=1).mean())


def raw_training_images(cls, n_train, sqrt=False, cache_dir=None):
    """(n_train, 64, 64) float32 QUANTIZED training fields, in anchor order -- the array
    evaluate_lens.py measures nearest-neighbour (memorisation) distances against, and the
    array controls_lens.py copies/averages."""
    snaps, anchors, _ = load_lens(cls, n_train=n_train, n_holdout=0, sqrt=sqrt, cache_dir=cache_dir)
    return np.stack([snaps[a].field for a in anchors], axis=0).astype(np.float32)


def holdout_images(cls, n_holdout=N_HOLDOUT, sqrt=False, cache_dir=None):
    """(n_holdout, 64, 64) float32 QUANTIZED held-out fields, in holdout-id order."""
    _train, val_raw = _raw_arrays(cls, cache_dir)
    ids = get_holdout_ids(n_holdout)
    return np.stack(
        [_make_snapshot(k, val_raw[k - HOLDOUT_ID_OFFSET - 1], sqrt=sqrt).field for k in ids], axis=0
    ).astype(np.float32)


# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    cache = sys.argv[1] if len(sys.argv) > 1 else None
    meta = build_cache(cache_dir=cache)
    print("[data_lens] cache_dir  =", _cache_dir(cache))
    print("[data_lens] class_names=", meta["class_names"])
    print("[data_lens] a_z, b_z   =", get_z_range())
    print("[data_lens] holdout ids=", get_holdout_ids()[:5], "...", get_holdout_ids()[-3:])
    for cname in meta["class_names"]:
        snaps, anchors, hold = load_lens(cname, n_train=11, cache_dir=cache)
        s1 = snaps[anchors[0]]
        agg = codec.aggregate(s1.pool_codes, A_Z, B_Z)
        print("[data_lens] %-8s anchors=%s holdout=%d pool=%d field(min/mean/max)=%.4f/%.4f/%.4f "
              "field==aggregate(pool): %s"
              % (cname, anchors[:3], len(hold), s1.pool_codes.size,
                 float(s1.field.min()), float(s1.field.mean()), float(s1.field.max()),
                 bool(np.allclose(agg, s1.field))))
        psnaps, panchors, _ph = load_lens(cname, n_train=11, n_holdout=2,
                                          encoding="pmf", cache_dir=cache)
        p1 = psnaps[panchors[0]]
        back = codec12.pmf_to_image(p1.pmf, p1.total_intensity)
        print("[data_lens] %-8s pmf: pool=%d sum=%.6f S=%.3f field identical: %s "
              "pmf_to_image round-trip max|err|=%.3g"
              % (cname, p1.pool_codes.size, float(p1.pmf.sum()), p1.total_intensity,
                 bool(np.array_equal(p1.field, s1.field)),
                 float(np.abs(back - p1.field).max())))
        print("[data_lens] %-8s S(mean total intensity, N=11) = %.4f"
              % (cname, mean_total_intensity(cname, 11, cache_dir=cache)))
    print("[data_lens] self-check OK")
