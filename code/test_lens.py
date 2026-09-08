#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Login-node smoke tests for the DeepLense Model II pilot (`--data lens`).

Part A -- data_lens.py's contract (INTERFACES.md "data_lens.py contract"):
  - Snapshot.field == codec.aggregate(Snapshot.pool_codes, a_z, b_z) exactly
  - (a_z, b_z) == (0.0, 1.0) for every class, from get_z_range() only
  - code layout: kx = column, ky = row, kz = quantize_z(intensity), and
    code == (kx << 12) | (ky << 6) | kz
  - the 200 held-out images (keys 1001..1200) do not overlap the training images, by key
    and by pixel content
  - load_lens_snapshot(i) reproduces load_lens()'s snapshots[i]
  - the N = 11 training set is a PREFIX of the N = 100 one (evaluate_lens.py relies on it)
  - the sqrt transform round-trips (invert_intensity . transform_intensity == identity)

Part B -- every arm's actual `--data lens` CLI path with tiny budgets
(--n-train 3 --n-holdout 4, a handful of steps), checking meta.json's a_z/b_z, the
generated-file contract, the `_last` sibling directory, and finite in-range fields.

Part C -- evaluate_lens.py on Part B's outputs (must print a table with the reference row)
and qcli_cci_lens.py --quick.

Repo rule (INTERFACES.md): login node = seconds-to-minutes CPU only. Do NOT raise these
budgets to anything resembling a real run.

Run: /usr/bin/python3 test_lens.py
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

import codec          # noqa: E402
import data_lens      # noqa: E402

FAILURES = []
N_TRAIN = 3
N_HOLDOUT = 4
N_GEN = 3
TOL = 1e-9


def check(name, cond, detail=""):
    print("[{}] {}{}".format("PASS" if cond else "FAIL", name, (" -- " + detail) if detail else ""))
    if not cond:
        FAILURES.append(name)


def _run(script, args, timeout=600):
    t0 = time.time()
    res = subprocess.run([PY, script] + args, cwd=HERE, capture_output=True, text=True,
                         timeout=timeout)
    return res, time.time() - t0


# ======================================================================================
# Part A -- loader contract
# ======================================================================================
def test_loader():
    print("\n=== A. data_lens.py contract ===")
    names = data_lens.get_class_names()
    check("class_names_are_3", len(names) == 3, str(names))

    a_z, b_z = data_lens.get_z_range()
    check("a_z_b_z_are_0_1", a_z == 0.0 and b_z == 1.0, "(%s, %s)" % (a_z, b_z))

    ids = data_lens.get_holdout_ids()
    check("holdout_ids_200", len(ids) == data_lens.N_HOLDOUT, "n=%d" % len(ids))
    check("holdout_ids_start_1001", ids[0] == 1001 and ids[-1] == 1200,
          "%d..%d" % (ids[0], ids[-1]))

    for cls in names:
        snaps, anchors, hold = data_lens.load_lens(cls, n_train=11)
        check("%s_anchors_1..11" % cls, anchors == list(range(1, 12)), str(anchors))
        check("%s_no_key_overlap" % cls, not (set(anchors) & set(hold)))
        check("%s_dict_covers_both" % cls,
              set(snaps.keys()) == set(anchors) | set(hold), "n_keys=%d" % len(snaps))

        s = snaps[1]
        check("%s_pool_size_4096" % cls, s.pool_codes.size == 4096, "n=%d" % s.pool_codes.size)
        agg = codec.aggregate(s.pool_codes, a_z, b_z)
        check("%s_field_eq_aggregate_pool" % cls, np.array_equal(agg, s.field),
              "max|diff|=%.3g" % float(np.abs(agg.astype(np.float64) - s.field).max()))
        check("%s_field_shape_64x64" % cls, s.field.shape == (64, 64), str(s.field.shape))
        check("%s_field_in_range" % cls,
              bool(np.isfinite(s.field).all() and s.field.min() >= a_z and s.field.max() <= b_z),
              "min/max=%.4f/%.4f" % (float(s.field.min()), float(s.field.max())))

        # code layout: kx = column, ky = row, kz = quantize_z(intensity)
        kx, ky, kz = codec.decode(s.pool_codes)
        recoded = (kx.astype(np.int64) << 12) | (ky.astype(np.int64) << 6) | kz.astype(np.int64)
        check("%s_code_roundtrip" % cls, np.array_equal(recoded, s.pool_codes))
        exp_ky, exp_kx = np.meshgrid(np.arange(64), np.arange(64), indexing="ij")
        check("%s_kx_is_column" % cls, np.array_equal(kx, exp_kx.ravel()))
        check("%s_ky_is_row" % cls, np.array_equal(ky, exp_ky.ravel()))
        check("%s_kz_matches_field_bins" % cls,
              np.array_equal(kz, codec.quantize_z(s.field[ky, kx], a_z, b_z)))

        # load_lens_snapshot must reproduce the dict entry
        one = data_lens.load_lens_snapshot(1, cls)
        check("%s_load_lens_snapshot_matches" % cls,
              bool(np.array_equal(one.pool_codes, s.pool_codes) and
                   np.array_equal(one.field, s.field)))

    cls = names[0]
    # held-out content must not appear in the training set (train_x vs val_x, disjoint)
    tr = data_lens.raw_training_images(cls, 100)
    ho = data_lens.holdout_images(cls, 200)
    d2 = ((tr.reshape(tr.shape[0], 1, -1) - ho.reshape(1, ho.shape[0], -1)) ** 2).sum(axis=2)
    check("%s_no_identical_train_holdout_image" % cls, bool(d2.min() > 0.0),
          "min squared L2 = %.6g" % float(d2.min()))

    tr11 = data_lens.raw_training_images(cls, 11)
    check("%s_n11_is_prefix_of_n100" % cls, bool(np.array_equal(tr11, tr[:11])))

    x = np.linspace(0.0, 1.0, 101, dtype=np.float32)
    rt = data_lens.invert_intensity(data_lens.transform_intensity(x, sqrt=True), sqrt=True)
    check("sqrt_transform_roundtrip", bool(np.allclose(rt, x, atol=1e-6)),
          "max|diff|=%.3g" % float(np.abs(rt - x).max()))

    ssq, _a, _h = data_lens.load_lens(cls, n_train=3, n_holdout=2, sqrt=True)
    s1 = ssq[1]
    check("sqrt_field_eq_aggregate_pool",
          bool(np.array_equal(codec.aggregate(s1.pool_codes, a_z, b_z), s1.field)))


# ======================================================================================
# Part B -- arm CLI smokes
# ======================================================================================
def _check_arm_outputs(tag, out_dir, expect_last, n_gen=N_GEN):
    meta_path = os.path.join(out_dir, "meta.json")
    if not os.path.isfile(meta_path):
        check("%s_meta_json_exists" % tag, False, meta_path)
        return
    check("%s_meta_json_exists" % tag, True)
    with open(meta_path) as fh:
        meta = json.load(fh)
    a_z, b_z = data_lens.get_z_range()
    check("%s_a_z_matches_get_z_range" % tag, abs(float(meta.get("a_z", -1)) - a_z) < TOL,
          "a_z=%s" % meta.get("a_z"))
    check("%s_b_z_matches_get_z_range" % tag, abs(float(meta.get("b_z", -1)) - b_z) < TOL,
          "b_z=%s" % meta.get("b_z"))
    check("%s_meta_lens_class" % tag, meta.get("lens_class") == "cdm", str(meta.get("lens_class")))

    files = sorted(f for f in os.listdir(out_dir) if f.startswith("generated_") and f.endswith(".npy"))
    check("%s_generated_count_%d" % (tag, n_gen), len(files) == n_gen, "found %d" % len(files))
    check("%s_no_time_axis_names" % tag, all(not f.startswith("generated_t") for f in files),
          str(files[:3]))
    for f in files[:2]:
        arr = np.load(os.path.join(out_dir, f))
        check("%s_%s_shape" % (tag, f), arr.shape == (64, 64), str(arr.shape))
        check("%s_%s_finite_in_range" % (tag, f),
              bool(np.isfinite(arr).all() and arr.min() >= a_z - 1e-6 and arr.max() <= b_z + 1e-6),
              "min/max=%.4f/%.4f" % (float(arr.min()), float(arr.max())))

    last = out_dir + "_last"
    if expect_last:
        ok = os.path.isdir(last) and os.path.isfile(os.path.join(last, "meta.json"))
        nlast = len([f for f in os.listdir(last)
                     if f.startswith("generated_") and f.endswith(".npy")]) if os.path.isdir(last) else 0
        check("%s_last_dir_written" % tag, ok and nlast == n_gen, "%s (n=%d)" % (last, nlast))
        if ok:
            with open(os.path.join(last, "meta.json")) as fh:
                lm = json.load(fh)
            check("%s_last_meta_tagged" % tag,
                  lm.get("aggregation") == "last" and str(lm.get("arm", "")).endswith("-last"),
                  "arm=%s aggregation=%s" % (lm.get("arm"), lm.get("aggregation")))
    else:
        check("%s_no_last_dir" % tag, not os.path.isdir(last), last)


def test_arms(tmp):
    print("\n=== B. arm CLI smokes (--data lens --n-train %d --n-holdout %d) ===" % (N_TRAIN, N_HOLDOUT))
    common = ["--data", "lens", "--lens-class", "cdm", "--n-train", str(N_TRAIN),
              "--n-holdout", str(N_HOLDOUT)]

    jobs = [
        ("iqp", "train_iqp.py", common + [
            "--synth-mode", "gaussian-latent", "--n-gen", str(N_GEN),
            "--steps-core", "3", "--steps-lat", "2", "--n-tuples", "2000",
            "--n-synth", "2000", "--device", "cpu", "--out", os.path.join(tmp, "iqp")], True),
        ("iqp_sqrt", "train_iqp.py", common + [
            "--lens-sqrt", "--synth-mode", "gaussian-latent", "--n-gen", str(N_GEN),
            "--steps-core", "3", "--steps-lat", "2", "--n-tuples", "2000",
            "--n-synth", "2000", "--device", "cpu", "--out", os.path.join(tmp, "iqp_sqrt")], True),
        ("rbm", "baseline_rbm.py", common + [
            "--synth-mode", "gaussian-latent", "--n-gen", str(N_GEN),
            "--steps-core", "5", "--steps-lat", "3", "--n-tuples", "2000",
            "--gen-chains", "200", "--gen-sweeps", "5", "--gen-keep", "2",
            "--device", "cpu", "--out", os.path.join(tmp, "rbm")], True),
        ("ctrl_marginal", "controls.py", ["--arm", "marginal"] + common + [
            "--synth-mode", "gaussian-latent", "--n-gen", str(N_GEN),
            "--n-tuples", "2000", "--n-synth", "2000",
            "--out", os.path.join(tmp, "ctrl_marginal")], True),
        ("dcgan", "baseline_dcgan.py", ["--data", "lens", "--lens-class", "cdm",
                                        "--n-train", str(N_TRAIN),
                                        "--tier", "51k", "--steps", "3", "--n-gen", str(N_GEN),
                                        "--device", "cpu", "--out", os.path.join(tmp, "dcgan")], False),
        ("ctrl_copy", "controls_lens.py", ["--arm", "copy", "--lens-class", "cdm",
                                           "--n-train", str(N_TRAIN), "--n-gen", str(N_GEN),
                                           "--out", os.path.join(tmp, "ctrl_copy")], False),
        ("ctrl_mean", "controls_lens.py", ["--arm", "mean", "--lens-class", "cdm",
                                           "--n-train", str(N_TRAIN), "--n-gen", str(N_GEN),
                                           "--out", os.path.join(tmp, "ctrl_mean")], False),
    ]

    for tag, script, args, expect_last in jobs:
        res, dt = _run(script, args)
        check("%s_exit_0" % tag, res.returncode == 0,
              "%.1fs%s" % (dt, "" if res.returncode == 0 else " -- " + res.stderr.strip()[-400:]))
        if res.returncode == 0:
            _check_arm_outputs(tag, os.path.join(tmp, tag), expect_last)

    # CTRL-interp must be refused for lens (no time axis to interpolate along)
    res, _ = _run("controls.py", ["--arm", "interp", "--data", "lens", "--lens-class", "cdm",
                                  "--n-train", str(N_TRAIN), "--out", os.path.join(tmp, "ctrl_interp")])
    check("ctrl_interp_refused_for_lens", res.returncode != 0,
          "exit=%d" % res.returncode)


# ======================================================================================
# Part C -- evaluate_lens.py and qcli_cci_lens.py
# ======================================================================================
def test_evaluate(tmp):
    print("\n=== C. evaluate_lens.py / qcli_cci_lens.py ===")
    res, dt = _run("evaluate_lens.py", [
        "--lens-class", "cdm", "--outputs-dir", tmp, "--n-holdout", "6",
        "--n-train-max", str(N_TRAIN), "--results", os.path.join(tmp, "results_lens_cdm.jsonl"),
    ], timeout=900)
    check("evaluate_lens_exit_0", res.returncode == 0,
          "%.1fs%s" % (dt, "" if res.returncode == 0 else " -- " + res.stderr.strip()[-400:]))
    if res.returncode == 0:
        out = res.stdout
        check("evaluate_lens_printed_table", "| Arm | N | PDF–JS" in out)
        check("evaluate_lens_has_reference_row", "HOLDOUT (ref)" in out)
        check("evaluate_lens_has_nn_columns", "NN-train" in out and "NN-holdout" in out)
        rows = [json.loads(l) for l in
                open(os.path.join(tmp, "results_lens_cdm.jsonl")).read().splitlines() if l.strip()]
        check("evaluate_lens_rows_have_all_metrics",
              all(set(("pdf_js", "conv_mmd2", "nn_train", "nn_holdout")) <= set(r) for r in rows),
              "n_rows=%d" % len(rows))
        copy_rows = [r for r in rows if r["arm"] == "CTRL-copy"]
        check("ctrl_copy_nn_train_is_zero",
              bool(copy_rows) and abs(copy_rows[0]["nn_train"]) < 1e-6,
              "nn_train=%s" % (copy_rows[0]["nn_train"] if copy_rows else "missing"))
        print("\n".join(out.splitlines()[-14:]))

    res, dt = _run("qcli_cci_lens.py", ["--quick", "--out", os.path.join(tmp, "qcli_cci_lens.json")],
                   timeout=900)
    check("qcli_cci_lens_quick_exit_0", res.returncode == 0,
          "%.1fs%s" % (dt, "" if res.returncode == 0 else " -- " + res.stderr.strip()[-400:]))
    if res.returncode == 0:
        with open(os.path.join(tmp, "qcli_cci_lens.json")) as fh:
            d = json.load(fh)
        names = data_lens.get_class_names()
        check("qcli_cci_lens_all_classes", set(d["per_image"].keys()) == set(names),
              str(sorted(d["per_image"].keys())))
        check("qcli_cci_lens_has_uniform_null",
              all("I_QCLI_null" in d["per_image"][c] and "I_QCLI_null" in d["pooled"][c]
                  for c in names))
        check("qcli_cci_lens_per_image_M_4096",
              all(d["per_image"][c]["M"] == 4096 for c in names))


def main():
    tmp = tempfile.mkdtemp(prefix="test_lens_")
    try:
        test_loader()
        test_arms(tmp)
        test_evaluate(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    if FAILURES:
        print("FAILED (%d): %s" % (len(FAILURES), ", ".join(FAILURES)))
        return 1
    print("ALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
