#!/usr/bin/env python3
"""Login-node CPU smoke tests for baseline_rbm.py and baseline_dcgan.py.

Everything here must finish in seconds on CPU (repo rule: no real training/eval on
the login node -- see INTERFACES.md). These tests only check: parameter counts match
the paper's stated ~12k (RBM) / 51k,0.2M,1.2M (DCGAN generator) targets, and that a
tiny end-to-end run produces contract-shaped output files (INTERFACES.md's
"Generated-sample contract").

Run: /usr/bin/python3 test_baselines.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable if sys.executable else "/usr/bin/python3"

sys.path.insert(0, HERE)

import baseline_rbm  # noqa: E402
import baseline_dcgan  # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print("[{}] {}{}".format(status, name, (" -- " + detail) if detail else ""))
    if not cond:
        FAILURES.append(name)


def test_rbm_param_count():
    rbm = baseline_rbm.RBM(n_visible=18, n_hidden=630, device="cpu", seed=0)
    n = rbm.param_count()
    check("rbm_param_count == 11988", n == 11988, "got {}".format(n))


def test_dcgan_param_counts():
    for tier, spec in baseline_dcgan.TIERS.items():
        ngf = spec["ngf"]
        target = spec["target"]
        G = baseline_dcgan.make_generator(ngf)
        n = baseline_dcgan.count_params(G)
        within = abs(n - target) <= 0.10 * target
        check(
            "dcgan_tier_{}_param_count_within_10pct".format(tier),
            within,
            "ngf={} params={} target={} ratio={:.3f}".format(ngf, n, target, n / float(target)),
        )


def _run_script(script, args, cwd):
    cmd = [PY, script] + args
    t0 = time.time()
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=300)
    dt = time.time() - t0
    return result, dt


def test_rbm_smoke():
    tmpdir = tempfile.mkdtemp(prefix="rbm_smoke_")
    out_dir = os.path.join(tmpdir, "rbm_out")
    args = [
        "--data", "synthetic",
        "--out", out_dir,
        "--device", "cpu",
        "--seed", "0",
        "--n-tuples", "2000",
        "--steps-core", "30",
        "--steps-lat", "5",
        "--batch", "64",
        "--n-chains-pcd", "20",
        "--gen-chains", "50",
        "--gen-sweeps", "10",
        "--gen-keep", "3",
        "--anchors-max", "3",
        "--n-holdout", "4",
    ]
    result, dt = _run_script("baseline_rbm.py", args, HERE)
    ok = result.returncode == 0
    check("rbm_smoke_exit_0", ok, result.stderr[-800:] if not ok else "{:.1f}s".format(dt))
    if not ok:
        shutil.rmtree(tmpdir, ignore_errors=True)
        return

    meta_path = os.path.join(out_dir, "meta.json")
    check("rbm_smoke_meta_exists", os.path.isfile(meta_path))
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        for key in ("param_count", "seed", "train_snapshots", "wallclock_s"):
            check("rbm_smoke_meta_has_{}".format(key), key in meta)
        check("rbm_smoke_meta_param_count_11988", meta.get("param_count") == 11988)

        gen_files = [fn for fn in os.listdir(out_dir) if fn.startswith("generated_t")]
        check("rbm_smoke_generated_count", len(gen_files) == meta.get("n_holdout", -1),
              "found {} files, meta says n_holdout={}".format(len(gen_files), meta.get("n_holdout")))
        check("rbm_smoke_generated_nonempty", len(gen_files) > 0)

        for fn in gen_files:
            arr = np.load(os.path.join(out_dir, fn))
            check("rbm_smoke_field_shape_{}".format(fn), arr.shape == (64, 64), "shape={}".format(arr.shape))
            check("rbm_smoke_field_finite_{}".format(fn), np.isfinite(arr).all())
            lo, hi = meta["a_z"], meta["b_z"]
            margin = 0.05 * (hi - lo) + 1e-6
            in_range = bool(arr.min() >= lo - margin and arr.max() <= hi + margin)
            check("rbm_smoke_field_range_{}".format(fn), in_range,
                  "min={:.4f} max={:.4f} expected~[{:.4f},{:.4f}]".format(arr.min(), arr.max(), lo, hi))

    shutil.rmtree(tmpdir, ignore_errors=True)


def test_dcgan_smoke():
    tmpdir = tempfile.mkdtemp(prefix="dcgan_smoke_")
    out_dir = os.path.join(tmpdir, "dcgan_out")
    args = [
        "--data", "synthetic",
        "--tier", "51k",
        "--n-train", "3",
        "--steps", "20",
        "--seed", "0",
        "--out", out_dir,
        "--device", "cpu",
        "--n-gen", "8",
    ]
    result, dt = _run_script("baseline_dcgan.py", args, HERE)
    ok = result.returncode == 0
    check("dcgan_smoke_exit_0", ok, result.stderr[-800:] if not ok else "{:.1f}s".format(dt))
    if not ok:
        shutil.rmtree(tmpdir, ignore_errors=True)
        return

    meta_path = os.path.join(out_dir, "meta.json")
    check("dcgan_smoke_meta_exists", os.path.isfile(meta_path))
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        for key in ("param_count", "seed", "train_snapshots", "wallclock_s"):
            check("dcgan_smoke_meta_has_{}".format(key), key in meta)

        gen_files = sorted(fn for fn in os.listdir(out_dir) if fn.startswith("generated_"))
        check("dcgan_smoke_generated_count_8", len(gen_files) == 8, "found {}".format(len(gen_files)))

        for fn in gen_files:
            arr = np.load(os.path.join(out_dir, fn))
            check("dcgan_smoke_field_shape_{}".format(fn), arr.shape == (64, 64), "shape={}".format(arr.shape))
            check("dcgan_smoke_field_finite_{}".format(fn), np.isfinite(arr).all())

    shutil.rmtree(tmpdir, ignore_errors=True)


def main():
    test_rbm_param_count()
    test_dcgan_param_counts()
    test_rbm_smoke()
    test_dcgan_smoke()

    print()
    if FAILURES:
        print("FAILED ({}): {}".format(len(FAILURES), ", ".join(FAILURES)))
        sys.exit(1)
    else:
        print("ALL TESTS PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
