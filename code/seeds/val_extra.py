#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A second, disjoint block of validation images -- for choosing hyper-parameters.

The round-3 cache holds exactly the 200 held-out images and nothing else
(`<WORK>/data/lens_cache/r3_images.npz`), so any procedure that needs real images it
is allowed to *select on* has nowhere to get them.  Selecting a baseline's step count on the
held-out set would be selecting on the reported test set, which is the thing the whole
comparison is supposed to avoid.

This module materialises the NEXT block of the same validation permutation:

    perm  = np.random.default_rng([R3_SPLIT_SEED, 3, 1]).permutation(n_val_all)
    held-out (already cached, never selected on)   perm[:200]
    selection block (this module)                  perm[200:400]

so the two blocks come from one split rule, one permutation, and are disjoint by
construction.  The per-class blocks follow data_lens_class's rule the same way:

    vperm = np.random.default_rng([R3_SPLIT_SEED, 3, 1, class_index]).permutation(idx_va)
    held-out  vperm[:200]        selection block  vperm[200:400]

Both are cached to `<cache>/r3_valextra_images.npz` after one read of the 1.4 GB source, in
the same style as data_lens_r3.build_cache.  Nothing in data_lens_r3 or data_lens_class is
modified; this only reads them.

    python3 seeds/val_extra.py            # build the cache and print a summary

Python 3.9 compatible.
"""
from __future__ import annotations

import json
import os
import sys

REPO = "<REPO_ROOT>/code"
sys.path.insert(0, REPO)

import numpy as np                                      # noqa: E402

import data_lens_class                                  # noqa: E402
import data_lens_r3 as D                                # noqa: E402

CACHE_NAME = "r3_valextra_images.npz"
CACHE_META = "r3_valextra_meta.json"
CACHE_VERSION = 1
N_SELECT = 200          # size of the selection block, same as the held-out block


def _paths(cache_dir=None):
    cache_dir = D._cache_dir(cache_dir)
    return os.path.join(cache_dir, CACHE_NAME), os.path.join(cache_dir, CACHE_META)


def build(cache_dir=None, npz_path=None, force=False):
    """Materialise the selection block for the pooled set and for each class. Idempotent."""
    data_path, meta_path = _paths(cache_dir)
    if not force and os.path.exists(data_path) and os.path.exists(meta_path):
        with open(meta_path) as fh:
            m = json.load(fh)
        if m.get("cache_version") == CACHE_VERSION:
            return m

    npz_path = npz_path if npz_path else D._npz_path()
    if not os.path.exists(npz_path):
        raise FileNotFoundError("lens dataset not found: %s" % npz_path)
    os.makedirs(D._cache_dir(cache_dir), exist_ok=True)

    npz = np.load(npz_path)
    class_names = [str(s) for s in np.asarray(npz["class_names"]).tolist()]
    val_y = np.asarray(npz["val_y"]).astype(np.int64)
    n_val_all = int(val_y.shape[0])

    perm = np.random.default_rng([D.R3_SPLIT_SEED, 3, 1]).permutation(n_val_all)
    if perm.size < D.N_HOLDOUT + N_SELECT:
        raise ValueError("val_x has %d images, need %d for a disjoint selection block"
                         % (perm.size, D.N_HOLDOUT + N_SELECT))
    pooled_idx = perm[D.N_HOLDOUT:D.N_HOLDOUT + N_SELECT]

    per_class_idx = {}
    for c in data_lens_class.CLASSES:
        ci = class_names.index(c)
        idx_va = np.flatnonzero(val_y == ci)
        vperm = np.random.default_rng([D.R3_SPLIT_SEED, 3, 1, ci]).permutation(idx_va)
        if vperm.size < D.N_HOLDOUT + N_SELECT:
            raise ValueError("class %r has %d val images, need %d"
                             % (c, vperm.size, D.N_HOLDOUT + N_SELECT))
        per_class_idx[c] = vperm[D.N_HOLDOUT:D.N_HOLDOUT + N_SELECT]

    val_x = np.asarray(npz["val_x"])
    arrays = {"pooled": val_x[pooled_idx].astype(np.float32)}
    for c, idx in per_class_idx.items():
        arrays[c] = val_x[idx].astype(np.float32)
    del val_x

    np.savez_compressed(data_path, **arrays)
    meta = {"cache_version": CACHE_VERSION, "npz_path": npz_path,
            "split_seed": D.R3_SPLIT_SEED, "n_holdout": D.N_HOLDOUT, "n_select": N_SELECT,
            "rule": "perm[N_HOLDOUT:N_HOLDOUT+N_SELECT] of the same validation permutation "
                    "the held-out block takes perm[:N_HOLDOUT] from",
            "class_names": class_names,
            "pooled_indices": [int(v) for v in pooled_idx.tolist()],
            "class_indices": {c: [int(v) for v in i.tolist()] for c, i in per_class_idx.items()}}
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
    return meta


def selection_images(class_name=None, cache_dir=None):
    """(N_SELECT, 64, 64) float32 real images that are NOT in the held-out set."""
    build(cache_dir=cache_dir)
    data_path, _ = _paths(cache_dir)
    key = class_name if class_name else "pooled"
    with np.load(data_path) as z:
        if key not in z:
            raise KeyError("no selection block for %r in %s" % (key, data_path))
        return np.ascontiguousarray(np.asarray(z[key], dtype=np.float32))


def check_disjoint(class_name=None, cache_dir=None):
    """Assert the selection block shares no source index with the held-out block."""
    _dp, meta_path = _paths(cache_dir)
    build(cache_dir=cache_dir)
    with open(meta_path) as fh:
        m = json.load(fh)
    if class_name:
        sel = set(m["class_indices"][class_name])
        hm = data_lens_class.class_meta(class_name)
        held = set(hm["val_indices"])
    else:
        sel = set(m["pooled_indices"])
        held = set(D.get_cache_meta(cache_dir)["val_indices"])
    inter = sel & held
    if inter:
        raise AssertionError("selection block for %r overlaps the held-out block in %d indices"
                             % (class_name, len(inter)))
    return len(sel), len(held)


def main(argv=None):
    m = build()
    print("[val_extra] cache_version=%s rule=%r" % (m["cache_version"], m["rule"]))
    for c in [None] + list(data_lens_class.CLASSES):
        imgs = selection_images(c)
        ns, nh = check_disjoint(c)
        print("[val_extra] %-8s selection %s  disjoint from held-out (%d vs %d indices, 0 shared)"
              % (c if c else "pooled", imgs.shape, ns, nh))
    return 0


if __name__ == "__main__":
    sys.exit(main())
