#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run train_iqp_affine_r3.py on ONE dark-matter substructure class.

This is a WRAPPER, not a fork: it imports train_iqp_affine_r3 and calls its `main`
unchanged.  The only thing it does first is `data_lens_class.install(<class>)`, which
re-points data_lens_r3.load_r3 at that class's 100 training images and 200 held-out
images.  Every hyper-parameter, the codec, the objective, the prior, the projection and
the output contract are therefore bit-for-bit the pooled arm's.

    python3 train_iqp_affine_class.py --class-name axion <all the usual r3 flags>

Afterwards it stamps `class_name` / `classes_pooled: false` / the split provenance into
the run's meta.json, so a run directory says on its own which class it belongs to.

Python 3.9 compatible.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import data_lens_class
import train_iqp_affine_r3


def parse_wrapper_args(argv):
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--class-name", required=True, choices=list(data_lens_class.CLASSES))
    known, rest = p.parse_known_args(argv)
    return known.class_name, rest


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    class_name, rest = parse_wrapper_args(argv)

    cache_meta = data_lens_class.install(class_name)
    print("[class] installed %r: train=%d latent=%d holdout=%d (pure train=%s val=%s)"
          % (class_name, cache_meta["n_train_max"], cache_meta["n_latent_pool"],
             cache_meta["n_holdout"], cache_meta["train_all_same_class"],
             cache_meta["val_all_same_class"]), flush=True)

    # give the arm a class-tagged name unless the caller set one, so the merged results
    # table can tell the three per-class runs apart from each other and from pooled
    if "--arm-name" not in rest:
        base = train_iqp_affine_r3.default_arm_name(train_iqp_affine_r3.parse_args(rest))
        rest = rest + ["--arm-name", "%s-%s" % (base, class_name)]

    rc = train_iqp_affine_r3.main(rest)

    args = train_iqp_affine_r3.parse_args(rest)
    meta_path = os.path.join(args.out, "meta.json")
    if os.path.isfile(meta_path):
        with open(meta_path) as fh:
            meta = json.load(fh)
        meta["class_name"] = class_name
        meta["class_index"] = cache_meta["class_index"]
        meta["classes_pooled"] = False
        meta["split_rule"] = cache_meta["split_rule"]
        meta["wrapper_script"] = "train_iqp_affine_class.py"
        meta["train_indices"] = cache_meta["train_indices"]
        meta["holdout_indices"] = cache_meta["val_indices"]
        with open(meta_path, "w") as fh:
            json.dump(meta, fh, indent=2, sort_keys=True)
        print("[class] stamped class_name=%r into %s" % (class_name, meta_path), flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
