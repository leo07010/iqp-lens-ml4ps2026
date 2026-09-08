#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pre-flight for the per-class appendix run.  CPU only, seconds, no real dataset.

Builds a SYNTHETIC model_II-shaped .npz in a temp dir, points data_lens_class at it, and
checks the four things that could silently ruin the appendix:

  1. each class cache is 100 / 2000 / 200 images and every one of them really is that class
  2. train and latent pool are disjoint, and the three classes' training sets are disjoint
  3. install() actually re-points data_lens_r3.load_r3 (different classes -> different data)
     and un-installing one class and installing another does not leak the first one's images
  4. the two wrappers run end to end (5-step train, then evaluate) and the run's meta.json
     carries class_name

Run it before sbatch; the sbatch runs it again as a fail-fast pre-check.
Python 3.9 compatible.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

FAILS = []


def check(name, cond, detail=""):
    if cond:
        print("  ok   %s" % name)
    else:
        print("  FAIL %s %s" % (name, detail))
        FAILS.append(name)


def make_synth_npz(path, per_class_train=2100, per_class_val=200, seed=0):
    """model_II-shaped: train_x/train_y/val_x/val_y/class_names, labels BLOCKED by class
    exactly like the real file, images in [0, 1] with a class-dependent ring radius so the
    three classes are distinguishable by content."""
    rng = np.random.default_rng(seed)
    names = np.array(["axion", "cdm", "no_sub"])
    yy, xx = np.mgrid[0:64, 0:64]
    tr_x, tr_y, va_x, va_y = [], [], [], []
    for ci in range(3):
        radius = 8.0 + 3.0 * ci
        for split, n, X, Y in (("t", per_class_train, tr_x, tr_y),
                               ("v", per_class_val, va_x, va_y)):
            cy = 32 + rng.normal(0, 1.5, size=n)
            cx = 32 + rng.normal(0, 1.5, size=n)
            for i in range(n):
                r = np.hypot(yy - cy[i], xx - cx[i])
                img = np.exp(-0.5 * ((r - radius) / 2.0) ** 2)
                img = img / max(img.max(), 1e-12)
                X.append(img.astype(np.float32))
                Y.append(ci)
    np.savez(path, train_x=np.stack(tr_x), train_y=np.asarray(tr_y, dtype=np.int64),
             val_x=np.stack(va_x), val_y=np.asarray(va_y, dtype=np.int64),
             class_names=names)


def main():
    tmp = tempfile.mkdtemp(prefix="test_class_")
    npz = os.path.join(tmp, "synth_model_II.npz")
    cache = os.path.join(tmp, "cache")
    os.makedirs(cache, exist_ok=True)
    os.environ["LENS_NPZ"] = npz
    os.environ["LENS_CACHE_DIR"] = cache
    try:
        print("[test_class] building synthetic dataset in %s" % tmp, flush=True)
        make_synth_npz(npz)

        import data_lens_r3 as D
        import data_lens_class as C

        print("[test_class] 1-2: per-class caches")
        metas = C.build_class_cache()
        src = np.load(npz)
        train_y, val_y = src["train_y"], src["val_y"]
        train_sets = {}
        for c in C.CLASSES:
            m = metas[c]
            tr = np.asarray(m["train_indices"])
            la = np.asarray(m["latent_indices"])
            va = np.asarray(m["val_indices"])
            train_sets[c] = set(tr.tolist())
            check("%s sizes 100/2000/200" % c,
                  (tr.shape[0], la.shape[0], va.shape[0]) == (100, 2000, 200),
                  str((tr.shape[0], la.shape[0], va.shape[0])))
            check("%s train is pure" % c, bool(np.all(train_y[tr] == m["class_index"])))
            check("%s latent is pure" % c, bool(np.all(train_y[la] == m["class_index"])))
            check("%s holdout is pure" % c, bool(np.all(val_y[va] == m["class_index"])))
            check("%s train/latent disjoint" % c, len(set(tr.tolist()) & set(la.tolist())) == 0)
            check("%s meta says not pooled" % c, m["classes_pooled"] is False)
        for a, b in (("axion", "cdm"), ("axion", "no_sub"), ("cdm", "no_sub")):
            check("%s/%s training sets disjoint" % (a, b),
                  len(train_sets[a] & train_sets[b]) == 0)

        print("[test_class] 3: install() re-points load_r3")
        C.install("axion")
        da = D.load_r3(100, n_holdout=200)
        C.install("no_sub")
        dn = D.load_r3(100, n_holdout=200)
        check("install changes the training images",
              not np.allclose(da.train_images, dn.train_images))
        check("install changes the held-out images",
              not np.allclose(da.holdout_images, dn.holdout_images))
        check("shapes still (100,64,64)/(200,64,64)",
              da.train_images.shape == (100, 64, 64) and dn.holdout_images.shape == (200, 64, 64),
              "%s %s" % (da.train_images.shape, dn.holdout_images.shape))
        C.install("axion")
        da2 = D.load_r3(100, n_holdout=200)
        check("re-install is deterministic (no leak from no_sub)",
              np.allclose(da.train_images, da2.train_images))
        check("installed_class() reports axion", C.installed_class() == "axion")

        print("[test_class] 4: wrappers end to end (5 steps, cpu)")
        out = os.path.join(tmp, "run", "axion_smoke")
        import train_iqp_affine_class as T
        rc = T.main(["--class-name", "axion", "--M", "4", "--objective", "mmdflat",
                     "--prior", "resample", "--n-train", "4", "--n-holdout", "6",
                     "--n-gen", "4", "--steps", "5", "--theta0-steps", "5",
                     "--infer-restarts", "2", "--infer-steps", "5", "--infer-chunk", "3",
                     "--log-every", "5", "--device", "cpu", "--seed", "0", "--out", out])
        check("train wrapper returns 0", rc in (0, None), str(rc))
        mp = os.path.join(out, "meta.json")
        check("meta.json exists", os.path.isfile(mp))
        if os.path.isfile(mp):
            with open(mp) as fh:
                m = json.load(fh)
            check("meta carries class_name=axion", m.get("class_name") == "axion", str(m.get("class_name")))
            check("meta arm is class-tagged",
                  str(m.get("arm", "")).endswith("-axion"), str(m.get("arm")))
            check("meta classes_pooled is False", m.get("classes_pooled") is False)
        check("generated frames written",
              len([f for f in os.listdir(out) if f.startswith("generated_")]) == 4)

        res = os.path.join(tmp, "results_class_test.jsonl")
        import evaluate_class as E
        rc = E.main(["--outputs-dir", os.path.join(tmp, "run"), "--classes", "axion",
                     "--results", res, "--n-holdout", "6", "--n-train-max", "4"])
        check("evaluate wrapper returns 0", rc == 0, str(rc))
        rows = [json.loads(l) for l in open(res) if l.strip()]
        check("results rows written", len(rows) >= 1, str(len(rows)))
        iqp = [r for r in rows if str(r.get("arm", "")).startswith("IQP-affine")]
        check("an IQP row is present", len(iqp) == 1, str(len(iqp)))
        if iqp:
            r = iqp[0]
            check("row tagged class_name=axion", r.get("class_name") == "axion")
            for k in ("pdf_js", "conv_mmd2", "conv_mmd2_aligned", "nn_holdout"):
                check("row has finite %s" % k,
                      r.get(k) is not None and np.isfinite(float(r[k])), str(r.get(k)))
            check("row has projection_corr_mean",
                  r.get("projection_corr_mean") is not None)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    if FAILS:
        print("[test_class] %d FAILURES: %s" % (len(FAILS), FAILS))
        return 1
    print("[test_class] all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
