#!/usr/bin/env python3
"""Real-data (--data d2) smoke tests for every arm, guarding against the 2026-09-05
a_z/b_z bug (INTERFACES.md "Data-side single-source rules" #3, data_d2.py module
docstring).

The bug: train_iqp.py / controls.py / baseline_rbm.py / baseline_dcgan.py /
make_ground_truth.py each independently RE-DERIVED a_z/b_z for --data d2 from
`snapshots[t].field`'s own min/max, instead of calling data_d2.get_z_range() directly.
`.field` is already codec.aggregate(pool_codes, a_z, b_z) -- a 64x64-bin average that
narrows the range (~18.8% observed: (0.0159, 1.6676) vs the true native (0.0, 2.034849))
relative to the bounds load_d2() actually used to encode `.pool_codes`. Re-deriving
a_z/b_z from `.field` therefore decodes/aggregates/quantizes downstream on the WRONG
scale -- silently invalid for every d2 result. Also guards baseline_rbm.py's d2 holdout
tau, which used to draw its own ad-hoc set (independent RNG) instead of the one shared
split every arm must use (data_d2.get_holdout_taus(), INTERFACES.md rule 1).

For every arm's actual `--data d2` CLI path (real d2_cache, not synthetic), this checks:
  - meta.json's a_z/b_z EXACTLY match data_d2.get_z_range("mid") (tol 1e-9)
  - held-out tau (from meta.json or generated_t{tau}.npy file names) are IDENTICAL across
    arms and match data_d2.get_holdout_taus()[:n] (train_iqp.py / controls.py /
    baseline_rbm.py must all resolve the same source -- see INTERFACES.md rule 1)
  - generated fields: shape (64, 64), finite, values within [a_z, b_z]
  - make_ground_truth.py's gt_t{tau}.npy ground-truth files exist for those same tau

Needs the real d2_cache (data_d2.CACHE_DIR_DEFAULT, built by preprocess_d2.py). SKIPs
(does not FAIL) if that cache is not present -- a login node without the ~1.2 GB cache
staged is a legitimate state, not a bug in these arms.

Repo rule (INTERFACES.md): login node = seconds-to-minutes CPU only. Every subprocess
call below uses tiny --anchors-max/--n-holdout/--steps* budgets (finishes in seconds;
the real d2_cache is already on disk, so no dataset I/O beyond a few small memmap reads).
Do NOT raise these budgets to anything resembling a real run.

Run: /usr/bin/python3 test_d2_arms_smoke.py
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable if sys.executable else "/usr/bin/python3"
sys.path.insert(0, HERE)

FAILURES = []
ANCHORS_MAX = 3
N_HOLDOUT = 3
A_Z_B_Z_TOL = 1e-9


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print("[{}] {}{}".format(status, name, (" -- " + detail) if detail else ""))
    if not cond:
        FAILURES.append(name)


def _run_script(script, args, cwd, timeout=300):
    cmd = [PY, script] + args
    t0 = time.time()
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    dt = time.time() - t0
    return result, dt


def _d2_cache_available():
    try:
        import data_d2
        cache = data_d2.CACHE_DIR_DEFAULT
        return os.path.exists(os.path.join(cache, "zval_all.npy")) and \
            os.path.exists(os.path.join(cache, "grid_xy.npz"))
    except Exception:
        return False


def _holdout_from_filenames(out_dir, prefix="generated_t", suffix=".npy"):
    taus = []
    if not os.path.isdir(out_dir):
        return taus
    for fn in os.listdir(out_dir):
        m = re.match(r"^{}(\d+){}$".format(re.escape(prefix), re.escape(suffix)), fn)
        if m:
            taus.append(int(m.group(1)))
    return sorted(taus)


def _check_a_z_b_z(name, meta, exp_a_z, exp_b_z):
    a_z = meta.get("a_z")
    b_z = meta.get("b_z")
    check(
        "{}_a_z_matches_get_z_range".format(name),
        a_z is not None and abs(float(a_z) - exp_a_z) < A_Z_B_Z_TOL,
        "a_z={} expected={}".format(a_z, exp_a_z),
    )
    check(
        "{}_b_z_matches_get_z_range".format(name),
        b_z is not None and abs(float(b_z) - exp_b_z) < A_Z_B_Z_TOL,
        "b_z={} expected={}".format(b_z, exp_b_z),
    )
    return a_z, b_z


def _check_generated_fields(name, out_dir, taus, a_z, b_z, prefix="generated_t"):
    if a_z is None or b_z is None:
        return
    margin = 1e-4 * max(1.0, float(b_z) - float(a_z))
    for tau in taus:
        path = os.path.join(out_dir, "{}{}.npy".format(prefix, tau))
        if not os.path.isfile(path):
            check("{}_t{}_file_exists".format(name, tau), False, path)
            continue
        arr = np.load(path)
        check("{}_t{}_shape_64x64".format(name, tau), arr.shape == (64, 64), "shape={}".format(arr.shape))
        check("{}_t{}_finite".format(name, tau), bool(np.isfinite(arr).all()))
        in_range = bool(arr.min() >= float(a_z) - margin and arr.max() <= float(b_z) + margin)
        check(
            "{}_t{}_in_a_z_b_z_range".format(name, tau), in_range,
            "min={:.6f} max={:.6f} expected~[{:.6f},{:.6f}]".format(arr.min(), arr.max(), a_z, b_z),
        )


def test_iqp_d2_smoke(tmpdir, exp_a_z, exp_b_z, exp_holdout):
    out_dir = os.path.join(tmpdir, "iqp")
    args = [
        "--data", "d2", "--K", "6", "--steps-core", "30", "--steps-lat", "5",
        "--n-synth", "2000", "--anchors-max", str(ANCHORS_MAX), "--n-holdout", str(N_HOLDOUT),
        "--device", "cpu", "--out", out_dir,
    ]
    result, dt = _run_script("train_iqp.py", args, HERE)
    ok = result.returncode == 0
    check("iqp_d2_exit_0", ok, result.stderr[-800:] if not ok else "{:.1f}s".format(dt))
    if not ok:
        return
    with open(os.path.join(out_dir, "meta.json")) as f:
        meta = json.load(f)
    a_z, b_z = _check_a_z_b_z("iqp_d2", meta, exp_a_z, exp_b_z)
    unseen = meta.get("unseen")
    check("iqp_d2_holdout_matches_load_d2", unseen == exp_holdout,
          "unseen={} expected={}".format(unseen, exp_holdout))
    _check_generated_fields("iqp_d2", out_dir, exp_holdout, a_z, b_z)


def test_rbm_d2_smoke(tmpdir, exp_a_z, exp_b_z, exp_holdout):
    out_dir = os.path.join(tmpdir, "rbm")
    args = [
        "--data", "d2", "--anchors-max", str(ANCHORS_MAX), "--n-holdout", str(N_HOLDOUT),
        "--steps-core", "5", "--steps-lat", "3", "--n-tuples", "200", "--batch", "64",
        "--n-chains-pcd", "8", "--gen-chains", "50", "--gen-sweeps", "5", "--gen-keep", "3",
        "--device", "cpu", "--out", out_dir,
    ]
    result, dt = _run_script("baseline_rbm.py", args, HERE)
    ok = result.returncode == 0
    check("rbm_d2_exit_0", ok, result.stderr[-800:] if not ok else "{:.1f}s".format(dt))
    if not ok:
        return
    with open(os.path.join(out_dir, "meta.json")) as f:
        meta = json.load(f)
    a_z, b_z = _check_a_z_b_z("rbm_d2", meta, exp_a_z, exp_b_z)
    holdout_ts = meta.get("holdout_ts")
    check("rbm_d2_holdout_matches_load_d2", holdout_ts == exp_holdout,
          "holdout_ts={} expected={}".format(holdout_ts, exp_holdout))
    _check_generated_fields("rbm_d2", out_dir, exp_holdout, a_z, b_z)


def test_ctrl_marginal_d2_smoke(tmpdir, exp_a_z, exp_b_z, exp_holdout):
    out_dir = os.path.join(tmpdir, "ctrl_marginal")
    args = [
        "--arm", "marginal", "--data", "d2", "--anchors-max", str(ANCHORS_MAX),
        "--n-holdout", str(N_HOLDOUT), "--n-synth", "2000", "--n-tuples", "500", "--out", out_dir,
    ]
    result, dt = _run_script("controls.py", args, HERE)
    ok = result.returncode == 0
    check("ctrl_marginal_d2_exit_0", ok, result.stderr[-800:] if not ok else "{:.1f}s".format(dt))
    if not ok:
        return
    with open(os.path.join(out_dir, "meta.json")) as f:
        meta = json.load(f)
    a_z, b_z = _check_a_z_b_z("ctrl_marginal_d2", meta, exp_a_z, exp_b_z)
    holdout = _holdout_from_filenames(out_dir)
    check("ctrl_marginal_d2_holdout_matches_load_d2", holdout == exp_holdout,
          "holdout(from filenames)={} expected={}".format(holdout, exp_holdout))
    _check_generated_fields("ctrl_marginal_d2", out_dir, exp_holdout, a_z, b_z)


def test_ctrl_interp_d2_smoke(tmpdir, exp_a_z, exp_b_z, exp_holdout):
    out_dir = os.path.join(tmpdir, "ctrl_interp")
    args = [
        "--arm", "interp", "--data", "d2", "--anchors-max", str(ANCHORS_MAX),
        "--n-holdout", str(N_HOLDOUT), "--out", out_dir,
    ]
    result, dt = _run_script("controls.py", args, HERE)
    ok = result.returncode == 0
    check("ctrl_interp_d2_exit_0", ok, result.stderr[-800:] if not ok else "{:.1f}s".format(dt))
    if not ok:
        return
    with open(os.path.join(out_dir, "meta.json")) as f:
        meta = json.load(f)
    a_z, b_z = _check_a_z_b_z("ctrl_interp_d2", meta, exp_a_z, exp_b_z)
    holdout = _holdout_from_filenames(out_dir)
    check("ctrl_interp_d2_holdout_matches_load_d2", holdout == exp_holdout,
          "holdout(from filenames)={} expected={}".format(holdout, exp_holdout))
    _check_generated_fields("ctrl_interp_d2", out_dir, exp_holdout, a_z, b_z)


def test_dcgan_d2_smoke(tmpdir, exp_a_z, exp_b_z):
    out_dir = os.path.join(tmpdir, "dcgan")
    args = [
        "--data", "d2", "--tier", "51k", "--n-train", "3", "--steps", "10",
        "--n-gen", "4", "--device", "cpu", "--out", out_dir,
    ]
    result, dt = _run_script("baseline_dcgan.py", args, HERE)
    ok = result.returncode == 0
    check("dcgan_d2_exit_0", ok, result.stderr[-800:] if not ok else "{:.1f}s".format(dt))
    if not ok:
        return
    with open(os.path.join(out_dir, "meta.json")) as f:
        meta = json.load(f)
    a_z, b_z = _check_a_z_b_z("dcgan_d2", meta, exp_a_z, exp_b_z)
    # DCGAN has no time-conditioning (INTERFACES.md): unconditional generated_{i:03d}.npy,
    # not tied to held-out tau -- just check shape/finite/range on whatever it wrote.
    gen_files = sorted(fn for fn in os.listdir(out_dir) if fn.startswith("generated_"))
    check("dcgan_d2_generated_count_4", len(gen_files) == 4, "found {}".format(len(gen_files)))
    if a_z is not None and b_z is not None:
        margin = 1e-4 * max(1.0, float(b_z) - float(a_z))
        for fn in gen_files:
            arr = np.load(os.path.join(out_dir, fn))
            check("dcgan_d2_{}_shape_64x64".format(fn), arr.shape == (64, 64), "shape={}".format(arr.shape))
            check("dcgan_d2_{}_finite".format(fn), bool(np.isfinite(arr).all()))
            in_range = bool(arr.min() >= float(a_z) - margin and arr.max() <= float(b_z) + margin)
            check("dcgan_d2_{}_in_a_z_b_z_range".format(fn), in_range,
                  "min={:.6f} max={:.6f} expected~[{:.6f},{:.6f}]".format(arr.min(), arr.max(), a_z, b_z))


def test_make_ground_truth_d2_smoke(tmpdir, exp_a_z, exp_b_z, exp_holdout):
    out_dir = os.path.join(tmpdir, "ground_truth")
    args = [
        "--data", "d2", "--anchors-max", str(ANCHORS_MAX), "--n-holdout", str(N_HOLDOUT), "--out", out_dir,
    ]
    result, dt = _run_script("make_ground_truth.py", args, HERE)
    ok = result.returncode == 0
    check("gt_d2_exit_0", ok, result.stderr[-800:] if not ok else "{:.1f}s".format(dt))
    if not ok:
        return
    with open(os.path.join(out_dir, "meta.json")) as f:
        meta = json.load(f)
    a_z, b_z = _check_a_z_b_z("gt_d2", meta, exp_a_z, exp_b_z)
    unseen = meta.get("unseen")
    check("gt_d2_holdout_matches_load_d2", unseen == exp_holdout,
          "unseen={} expected={}".format(unseen, exp_holdout))
    for tau in exp_holdout:
        path = os.path.join(out_dir, "gt_t{}.npy".format(tau))
        check("gt_d2_t{}_file_exists".format(tau), os.path.isfile(path), path)
        if os.path.isfile(path):
            arr = np.load(path)
            check("gt_d2_t{}_shape_64x64".format(tau), arr.shape == (64, 64), "shape={}".format(arr.shape))
            check("gt_d2_t{}_finite".format(tau), bool(np.isfinite(arr).all()))


def main():
    if not _d2_cache_available():
        print("SKIP: real d2_cache not found (data_d2.CACHE_DIR_DEFAULT) -- "
              "run preprocess_d2.py first (see INTERFACES.md). Not a failure.")
        sys.exit(0)

    import data_d2
    exp_a_z, exp_b_z = data_d2.get_z_range("mid")
    exp_holdout = data_d2.get_holdout_taus()[:N_HOLDOUT]
    print("[setup] expected a_z={} b_z={} holdout[:{}]={}".format(
        exp_a_z, exp_b_z, N_HOLDOUT, exp_holdout))

    tmpdir = tempfile.mkdtemp(prefix="test_d2_arms_smoke_")
    try:
        test_iqp_d2_smoke(tmpdir, exp_a_z, exp_b_z, exp_holdout)
        test_rbm_d2_smoke(tmpdir, exp_a_z, exp_b_z, exp_holdout)
        test_ctrl_marginal_d2_smoke(tmpdir, exp_a_z, exp_b_z, exp_holdout)
        test_ctrl_interp_d2_smoke(tmpdir, exp_a_z, exp_b_z, exp_holdout)
        test_dcgan_d2_smoke(tmpdir, exp_a_z, exp_b_z)
        test_make_ground_truth_d2_smoke(tmpdir, exp_a_z, exp_b_z, exp_holdout)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    print()
    if FAILURES:
        print("FAILED ({}): {}".format(len(FAILURES), ", ".join(FAILURES)))
        sys.exit(1)
    else:
        print("ALL TESTS PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
