#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per-class (axion / cdm / no_sub) version of the round-3 lens split.

WHY THIS FILE EXISTS
--------------------
Every result in the ML4PS-2026 paper is trained on the three dark-matter substructure
classes POOLED (data_lens_r3.build_cache draws ONE permutation over all 80,194 training
images).  The appendix asks a narrower question -- "does the same arm train and generate
for each class on its own?" -- so each class needs its own N = 100 training set and its
own 200 held-out images.

WHAT IT DOES NOT DO
-------------------
It does not modify data_lens_r3.py, train_iqp_affine_r3.py, evaluate_r3.py or
evaluate_r3b.py.  Those four are what the pooled results were produced with and are still
referenced by queued/finished jobs.  Instead `install(class_name)` swaps THREE functions
inside the already-imported data_lens_r3 module (`_arrays`, `build_cache`,
`get_cache_meta`) so that `data_lens_r3.load_r3(...)` -- which every training and
evaluation script calls -- transparently returns that one class's images.  Nothing else in
the pipeline changes, which is the point: the arm, the codec, the metrics and the held-out
protocol are bit-for-bit the pooled ones.

THE SPLIT RULE (and how it differs from pooled)
-----------------------------------------------
Pooled:      perm = default_rng([0, 3, 0]).permutation(80194)
             train = perm[:100],  latent pool = perm[100:2100]
             val   = default_rng([0, 3, 1]).permutation(8910)[:200]

Per class c: idx_c  = the training indices whose label == c   (~26,731 of them)
             perm_c = default_rng([0, 3, 0, c]).permutation(idx_c)
             train  = perm_c[:100],  latent pool = perm_c[100:2100]
             val_c  = default_rng([0, 3, 1, c]).permutation(the class's val indices)[:200]

The rule is the same "one permutation, take the prefix" rule, restricted to one class.  It
is NOT a subset of the pooled prefix: the pooled N = 100 contains only ~33 images of each
class, so a per-class N = 100 cannot be carved out of it.  The class index is appended to
the seed so the three classes get independent permutations.

Python 3.9 compatible (no `X | None` annotations).  No network.
"""
from __future__ import annotations

import json
import os

import numpy as np

import data_lens_r3 as D

CLASS_CACHE_NAME = "r3c_%s_images.npz"
CLASS_CACHE_META = "r3c_%s_meta.json"
CLASS_CACHE_VERSION = 1

# what the sbatch / figure scripts iterate over, in the dataset's own class_names order
CLASSES = ("axion", "cdm", "no_sub")

_INSTALLED = {"class_name": None}


# --------------------------------------------------------------------------------------
# cache
# --------------------------------------------------------------------------------------
def _paths(class_name, cache_dir=None):
    cache_dir = D._cache_dir(cache_dir)
    return (os.path.join(cache_dir, CLASS_CACHE_NAME % class_name),
            os.path.join(cache_dir, CLASS_CACHE_META % class_name))


def _cached_meta(class_name, cache_dir=None):
    data_path, meta_path = _paths(class_name, cache_dir)
    if not (os.path.exists(data_path) and os.path.exists(meta_path)):
        return None
    with open(meta_path) as fh:
        meta = json.load(fh)
    if meta.get("cache_version") != CLASS_CACHE_VERSION:
        return None
    return meta


def build_class_cache(cache_dir=None, npz_path=None, force=False, classes=CLASSES):
    """Materialise one cache per class.  Reads the 1.4 GB source .npz ONCE for all three.

    Writes <cache>/r3c_<class>_images.npz with the same three arrays as the pooled cache
        train  (100,  64, 64) float32
        latent (2000, 64, 64) float32
        val    (200,  64, 64) float32
    all drawn from that class only.  Idempotent: returns immediately if every requested
    class is already cached at this CLASS_CACHE_VERSION.
    """
    cache_dir = D._cache_dir(cache_dir)
    metas = {}
    todo = []
    for c in classes:
        m = None if force else _cached_meta(c, cache_dir)
        if m is None:
            todo.append(c)
        else:
            metas[c] = m
    if not todo:
        return metas

    npz_path = npz_path if npz_path else D._npz_path()
    if not os.path.exists(npz_path):
        raise FileNotFoundError("lens dataset not found: %s" % npz_path)
    os.makedirs(cache_dir, exist_ok=True)

    npz = np.load(npz_path)
    class_names = [str(s) for s in np.asarray(npz["class_names"]).tolist()]
    train_y = np.asarray(npz["train_y"]).astype(np.int64)
    val_y = np.asarray(npz["val_y"]).astype(np.int64)

    need = D.N_TRAIN_MAX + D.N_LATENT_POOL
    plan = {}
    for c in todo:
        if c not in class_names:
            raise ValueError("unknown class %r, dataset has %r" % (c, class_names))
        ci = class_names.index(c)
        idx_tr = np.flatnonzero(train_y == ci)
        idx_va = np.flatnonzero(val_y == ci)
        if idx_tr.shape[0] < need:
            raise ValueError("class %s has %d training images, need %d"
                             % (c, idx_tr.shape[0], need))
        if idx_va.shape[0] < D.N_HOLDOUT:
            raise ValueError("class %s has %d val images, need %d"
                             % (c, idx_va.shape[0], D.N_HOLDOUT))
        # same permutation-prefix rule as the pooled split, restricted to this class
        perm = np.random.default_rng([D.R3_SPLIT_SEED, 3, 0, ci]).permutation(idx_tr)
        vperm = np.random.default_rng([D.R3_SPLIT_SEED, 3, 1, ci]).permutation(idx_va)
        plan[c] = {"ci": ci,
                   "train_idx": perm[:D.N_TRAIN_MAX],
                   "latent_idx": perm[D.N_TRAIN_MAX:need],
                   "val_idx": vperm[:D.N_HOLDOUT],
                   "n_train_all_class": int(idx_tr.shape[0]),
                   "n_val_all_class": int(idx_va.shape[0])}

    train_x = np.asarray(npz["train_x"])
    parts = {}
    for c in todo:
        p = plan[c]
        parts[c] = (train_x[p["train_idx"]].astype(np.float32),
                    train_x[p["latent_idx"]].astype(np.float32))
    del train_x
    val_x = np.asarray(npz["val_x"])
    for c in todo:
        plan[c]["val_arr"] = val_x[plan[c]["val_idx"]].astype(np.float32)
    del val_x

    for c in todo:
        p = plan[c]
        data_path, meta_path = _paths(c, cache_dir)
        tr, lat = parts[c]
        np.savez_compressed(data_path, train=tr, latent=lat, val=p["val_arr"])
        meta = {
            "cache_version": CLASS_CACHE_VERSION,
            "npz_path": npz_path,
            "class_name": c,
            "class_index": p["ci"],
            "class_names": class_names,
            "classes_pooled": False,
            "split_seed": D.R3_SPLIT_SEED,
            "split_rule": "per-class permutation prefix, rng seed [split_seed, 3, 0|1, class_index]",
            "n_train_max": D.N_TRAIN_MAX,
            "n_latent_pool": D.N_LATENT_POOL,
            "n_holdout": D.N_HOLDOUT,
            "crop": D.CROP,
            "n_train_all_class": p["n_train_all_class"],
            "n_val_all_class": p["n_val_all_class"],
            "train_indices": [int(v) for v in p["train_idx"].tolist()],
            "latent_indices": [int(v) for v in p["latent_idx"].tolist()],
            "val_indices": [int(v) for v in p["val_idx"].tolist()],
            "disjoint_train_latent": bool(
                len(set(p["train_idx"].tolist()) & set(p["latent_idx"].tolist())) == 0),
            "train_all_same_class": bool(np.all(train_y[p["train_idx"]] == p["ci"])),
            "val_all_same_class": bool(np.all(val_y[p["val_idx"]] == p["ci"])),
        }
        with open(meta_path, "w") as fh:
            json.dump(meta, fh, indent=2)
        metas[c] = meta
    return metas


def class_arrays(class_name, cache_dir=None):
    """(train, latent, val) float32 arrays for one class -- the per-class analogue of
    data_lens_r3._arrays."""
    build_class_cache(cache_dir=cache_dir, classes=(class_name,))
    data_path, _meta_path = _paths(class_name, cache_dir)
    with np.load(data_path) as z:
        return (np.asarray(z["train"], dtype=np.float32),
                np.asarray(z["latent"], dtype=np.float32),
                np.asarray(z["val"], dtype=np.float32))


def class_meta(class_name, cache_dir=None):
    return build_class_cache(cache_dir=cache_dir, classes=(class_name,))[class_name]


# --------------------------------------------------------------------------------------
# the switch
# --------------------------------------------------------------------------------------
def install(class_name, cache_dir=None):
    """Point data_lens_r3 at ONE class.  Call before load_r3 / latent_pool is used.

    Returns the class's cache meta.  Idempotent for the same class; calling it with a
    different class re-points the module (the arrays are re-read from the class cache).
    """
    if class_name not in CLASSES:
        raise ValueError("class_name must be one of %r, got %r" % (list(CLASSES), class_name))
    meta = class_meta(class_name, cache_dir=cache_dir)
    arrays = class_arrays(class_name, cache_dir=cache_dir)

    def _arrays_patched(cd=None):
        return (arrays[0].copy(), arrays[1].copy(), arrays[2].copy())

    def _build_cache_patched(cache_dir=None, npz_path=None, force=False):
        return meta

    D._arrays = _arrays_patched
    D.build_cache = _build_cache_patched
    D.get_cache_meta = _build_cache_patched
    _INSTALLED["class_name"] = class_name
    return meta


def installed_class():
    return _INSTALLED["class_name"]


# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    metas = build_class_cache()
    for c in CLASSES:
        m = metas[c]
        print("[data_lens_class] %-7s train=%d latent=%d val=%d | pure(train,val)=(%s,%s) "
              "disjoint=%s | class pool: train %d val %d"
              % (c, m["n_train_max"], m["n_latent_pool"], m["n_holdout"],
                 m["train_all_same_class"], m["val_all_same_class"],
                 m["disjoint_train_latent"], m["n_train_all_class"], m["n_val_all_class"]))
    install("axion")
    d = D.load_r3(100, n_holdout=200)
    print("[data_lens_class] load_r3 after install('axion'):", d)
