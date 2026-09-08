#!/usr/bin/env python3
"""Login-node CPU smoke tests for baseline_rbm_variants.py, plus a DCGAN --steps 300 smoke
(the smallest step count used by submit_baseline_align.sbatch's scan, to confirm the sweep's
cheapest jobs are viable before spending 8gpus budget on all 48 of them).

Everything here must finish in seconds on CPU (repo rule: no real training/eval on the login
node -- see INTERFACES.md).

Run: /usr/bin/python3 test_baseline_variants.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable if sys.executable else "/usr/bin/python3"

sys.path.insert(0, HERE)

import baseline_rbm  # noqa: E402
import baseline_rbm_variants as brv  # noqa: E402

FAILURES = []


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


# ---------------------------------------------------------------------------------------
# (a) literal latent get/set correctness: get_latent() == W.reshape(-1)[:50] == W[0, :50]
# for an (18, 630) W, and set_latent() touches ONLY those 50 entries (nothing else in W,
# b_v, or b_h moves).
# ---------------------------------------------------------------------------------------
def test_literal_latent_is_w_row0():
    rbm = brv.RBMLiteral(n_visible=18, n_hidden=630, device="cpu", seed=0)

    expected = rbm.W.detach().reshape(-1)[:50].clone()
    got = rbm.get_latent()
    check("literal_get_latent_shape_50", tuple(got.shape) == (50,), "got shape {}".format(tuple(got.shape)))
    check("literal_get_latent_matches_flatten", torch.allclose(got, expected))

    # W[0, :50] is exactly the first 50 flattened (row-major) entries of an (18, 630) W.
    row0_50 = rbm.W.detach()[0, :50].clone()
    check("literal_get_latent_matches_w_row0", torch.allclose(got, row0_50))

    W_before = rbm.W.detach().clone()
    bv_before = rbm.b_v.detach().clone()
    bh_before = rbm.b_h.detach().clone()

    new_latent = torch.arange(50, dtype=torch.float32) * 0.001 - 0.01  # distinctive, non-zero values
    rbm.set_latent(new_latent)

    W_after = rbm.W.detach().clone()
    flat_before = W_before.reshape(-1)
    flat_after = W_after.reshape(-1)
    check("literal_set_latent_first50_updated", torch.allclose(flat_after[:50], new_latent))
    check("literal_set_latent_rest_of_w_unchanged", torch.allclose(flat_after[50:], flat_before[50:]))
    check("literal_set_latent_bv_unchanged", torch.allclose(rbm.b_v.detach(), bv_before))
    check("literal_set_latent_bh_unchanged", torch.allclose(rbm.b_h.detach(), bh_before))

    # And get_latent() round-trips the value we just set.
    check("literal_get_latent_roundtrip", torch.allclose(rbm.get_latent(), new_latent))


def test_biases_latent_unaffected_by_variants_module():
    """Sanity: importing baseline_rbm_variants must not change baseline_rbm.RBM's own
    (unmodified) [b_v, b_h[:32]] latent behaviour."""
    rbm = baseline_rbm.RBM(n_visible=18, n_hidden=630, device="cpu", seed=0)
    latent = rbm.get_latent()
    check("biases_latent_shape_50", tuple(latent.shape) == (50,))
    check("biases_latent_is_rbm_class", type(rbm) is baseline_rbm.RBM)


# ---------------------------------------------------------------------------------------
# (b) cd1_sgd recipe smoke: tiny end-to-end run (20 steps, 3 anchors, 4 holdout) produces
# contract-shaped output.
# ---------------------------------------------------------------------------------------
def test_cd1_sgd_smoke():
    tmpdir = tempfile.mkdtemp(prefix="rbm_variant_cd1_smoke_")
    out_dir = os.path.join(tmpdir, "rbm_out")
    args = [
        "--data", "synthetic",
        "--out", out_dir,
        "--device", "cpu",
        "--seed", "0",
        "--latent", "biases",
        "--recipe", "cd1_sgd",
        "--n-tuples", "2000",
        "--steps-core", "20",
        "--steps-lat", "5",
        "--batch", "64",
        "--gen-chains", "50",
        "--gen-sweeps", "10",
        "--gen-keep", "3",
        "--anchors-max", "3",
        "--n-holdout", "4",
    ]
    result, dt = _run_script("baseline_rbm_variants.py", args, HERE)
    ok = result.returncode == 0
    check("cd1_sgd_smoke_exit_0", ok, result.stderr[-800:] if not ok else "{:.1f}s".format(dt))
    if not ok:
        shutil.rmtree(tmpdir, ignore_errors=True)
        return

    meta_path = os.path.join(out_dir, "meta.json")
    check("cd1_sgd_smoke_meta_exists", os.path.isfile(meta_path))
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        for key in ("param_count", "seed", "train_snapshots", "wallclock_s", "arm", "recipe", "latent_mode"):
            check("cd1_sgd_smoke_meta_has_{}".format(key), key in meta)
        check("cd1_sgd_smoke_meta_recipe_cd1_sgd", meta.get("recipe") == "cd1_sgd")
        check("cd1_sgd_smoke_meta_latent_biases", meta.get("latent_mode") == "biases")
        check("cd1_sgd_smoke_meta_arm_name", meta.get("arm") == "RBM-12k-biases-cd1_sgd",
              "got {}".format(meta.get("arm")))
        check("cd1_sgd_smoke_meta_param_count_11988", meta.get("param_count") == 11988)

        gen_files = [fn for fn in os.listdir(out_dir) if fn.startswith("generated_t")]
        check("cd1_sgd_smoke_generated_count", len(gen_files) == meta.get("n_holdout", -1),
              "found {} files, meta says n_holdout={}".format(len(gen_files), meta.get("n_holdout")))
        check("cd1_sgd_smoke_generated_nonempty", len(gen_files) > 0)
        for fn in gen_files:
            arr = np.load(os.path.join(out_dir, fn))
            check("cd1_sgd_smoke_field_shape_{}".format(fn), arr.shape == (64, 64), "shape={}".format(arr.shape))
            check("cd1_sgd_smoke_field_finite_{}".format(fn), np.isfinite(arr).all())

    shutil.rmtree(tmpdir, ignore_errors=True)


def test_literal_recipe_pcd_adam_smoke():
    """literal latent x pcd_adam recipe combo (the other cell of the 2x2 not covered above)."""
    tmpdir = tempfile.mkdtemp(prefix="rbm_variant_literal_smoke_")
    out_dir = os.path.join(tmpdir, "rbm_out")
    args = [
        "--data", "synthetic",
        "--out", out_dir,
        "--device", "cpu",
        "--seed", "0",
        "--latent", "literal",
        "--recipe", "pcd_adam",
        "--n-tuples", "2000",
        "--steps-core", "20",
        "--steps-lat", "5",
        "--batch", "64",
        "--n-chains-pcd", "20",
        "--gen-chains", "50",
        "--gen-sweeps", "10",
        "--gen-keep", "3",
        "--anchors-max", "3",
        "--n-holdout", "4",
    ]
    result, dt = _run_script("baseline_rbm_variants.py", args, HERE)
    ok = result.returncode == 0
    check("literal_pcd_smoke_exit_0", ok, result.stderr[-800:] if not ok else "{:.1f}s".format(dt))
    if not ok:
        shutil.rmtree(tmpdir, ignore_errors=True)
        return
    with open(os.path.join(out_dir, "meta.json")) as f:
        meta = json.load(f)
    check("literal_pcd_smoke_meta_arm_name", meta.get("arm") == "RBM-12k-literal-pcd_adam",
          "got {}".format(meta.get("arm")))
    shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------------------
# (c) DCGAN --steps 300 smoke (n-train 3, n-gen 4, CPU) -- the smallest step count in the
# baseline-alignment sweep's DCGAN scan; baseline_dcgan.py itself is unmodified.
# ---------------------------------------------------------------------------------------
def test_dcgan_steps300_smoke():
    tmpdir = tempfile.mkdtemp(prefix="dcgan_steps300_smoke_")
    out_dir = os.path.join(tmpdir, "dcgan_out")
    args = [
        "--data", "synthetic",
        "--tier", "51k",
        "--n-train", "3",
        "--steps", "300",
        "--seed", "0",
        "--out", out_dir,
        "--device", "cpu",
        "--n-gen", "4",
    ]
    result, dt = _run_script("baseline_dcgan.py", args, HERE)
    ok = result.returncode == 0
    check("dcgan_steps300_smoke_exit_0", ok, result.stderr[-800:] if not ok else "{:.1f}s".format(dt))
    if not ok:
        shutil.rmtree(tmpdir, ignore_errors=True)
        return

    meta_path = os.path.join(out_dir, "meta.json")
    check("dcgan_steps300_smoke_meta_exists", os.path.isfile(meta_path))
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        check("dcgan_steps300_smoke_meta_steps_300", meta.get("steps") == 300)
        gen_files = sorted(fn for fn in os.listdir(out_dir) if fn.startswith("generated_"))
        check("dcgan_steps300_smoke_generated_count_4", len(gen_files) == 4, "found {}".format(len(gen_files)))
        for fn in gen_files:
            arr = np.load(os.path.join(out_dir, fn))
            check("dcgan_steps300_smoke_field_shape_{}".format(fn), arr.shape == (64, 64), "shape={}".format(arr.shape))
            check("dcgan_steps300_smoke_field_finite_{}".format(fn), np.isfinite(arr).all())

    shutil.rmtree(tmpdir, ignore_errors=True)


def main():
    test_literal_latent_is_w_row0()
    test_biases_latent_unaffected_by_variants_module()
    test_cd1_sgd_smoke()
    test_literal_recipe_pcd_adam_smoke()
    test_dcgan_steps300_smoke()

    print()
    if FAILURES:
        print("FAILED ({}): {}".format(len(FAILURES), ", ".join(FAILURES)))
        sys.exit(1)
    else:
        print("ALL TESTS PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
