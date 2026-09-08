#!/usr/bin/env python3
"""Data-scaling suite smoke tests. CPU only, seconds -- submit_scaling.sbatch runs this as a
fail-fast pre-check so a broken arm cannot burn a dev/8gpu allocation.

Covers, in order:
  1  each of the three scaling arms (baseline_dcgan_r3.py, baseline_rbm_r3.py,
     train_iqp_affine_r3.py --M 4 --objective mmdflat --prior resample) at N = 3 -- a value
     that is neither of the two N data_lens_r3.py has previously been exercised at in this repo
     (11, 100) -- confirming the loader (data_lens_r3.load_r3) and each arm's output contract
     (generated_%03d.npy + crops/ + meta.json, arm name, n_train) hold at an arbitrary N.
  2  evaluate_r3b.py over the three smoke outputs: the three arms all reach the table.
  3  paper_suite/scaling/plot_scaling.py renders a figure from that (single-N-point) results
     file without crashing.
  4  paper_suite/scaling/plot_pseudotime.py renders a figure from the REAL, already-finished
     outputs_lens_r3/iqp_pseudotime_n100 (round 3) -- not smoke data, since this script draws
     no new samples, it only re-renders an existing pairing -- into a throwaway path, and its
     recomputed e1/latent numbers are cross-checked against the meta.json it reads (independent
     of results_r3.jsonl, so this test would catch a wiring bug even if the results file were
     stale).

Python 3.9 compatible.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))          # .../paper_suite/scaling -> .../13_iqp_turbulence_replication
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch                                            # noqa: E402
torch.set_num_threads(min(4, os.cpu_count() or 1))

import data_lens_r3                                      # noqa: E402  (fail fast if the cache is missing)
import plot_pseudotime                                   # noqa: E402
import plot_scaling                                       # noqa: E402

FAILED = []
T0 = time.time()


def check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    if not ok:
        FAILED.append(name)
    print("[%s] %-64s %s  (%.1fs)" % (tag, name, detail, time.time() - T0))
    return bool(ok)


def section(title):
    print("\n--- %s ---" % title)


def _run(mod_name, argv):
    """Call an arm's main() in-process (no subprocess: the tests must stay seconds)."""
    import importlib
    mod = importlib.import_module(mod_name)
    old = sys.argv
    sys.argv = [mod_name] + list(argv)
    try:
        rc = mod.main(argv)
    finally:
        sys.argv = old
    return rc


def _n_files(d):
    if not os.path.isdir(d):
        return 0
    return len([f for f in os.listdir(d) if f.startswith("generated_") and f.endswith(".npy")])


def _meta(d):
    with open(os.path.join(d, "meta.json")) as fh:
        return json.load(fh)


N_SMOKE = 3
COMMON_SMOKE = ["--n-train", str(N_SMOKE), "--n-holdout", "4", "--n-gen", "2", "--device", "cpu"]


def test_dcgan_smoke(root):
    section("1a. baseline_dcgan_r3.py smoke at N=%d" % N_SMOKE)
    out = os.path.join(root, "dcgan_smoke")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = _run("baseline_dcgan_r3", COMMON_SMOKE + ["--steps", "10", "--log-every", "10",
                                                        "--out", out])
    m = _meta(out) if os.path.isfile(os.path.join(out, "meta.json")) else {}
    ok = (rc == 0 and m.get("arm") == "DCGAN-32" and m.get("n_train") == N_SMOKE
          and _n_files(out) == 2 and _n_files(os.path.join(out, "crops")) == 2)
    check("DCGAN smoke runs at N=%d and writes the round-3 output contract" % N_SMOKE, ok,
          "arm=%s n_train=%s files=%d crops=%d" % (m.get("arm"), m.get("n_train"), _n_files(out),
                                                    _n_files(os.path.join(out, "crops"))))
    if ok:
        frame = np.load(os.path.join(out, "generated_000.npy"))
        crop = np.load(os.path.join(out, "crops", "generated_000.npy"))
        check("  DCGAN output contract: 64x64 frame, 32x32 crop, non-negative",
              frame.shape == (64, 64) and crop.shape == (32, 32)
              and float(frame.min()) >= 0.0 and float(crop.min()) >= 0.0)
    return out


def test_rbm_smoke(root):
    section("1b. baseline_rbm_r3.py smoke at N=%d" % N_SMOKE)
    out = os.path.join(root, "rbm_smoke")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = _run("baseline_rbm_r3", COMMON_SMOKE + ["--steps", "10", "--n-pool", "500",
                                                      "--gen-chains", "8", "--gen-sweeps", "5",
                                                      "--gen-keep", "2", "--gen-chunk", "2",
                                                      "--log-every", "10", "--out", out])
    m = _meta(out) if os.path.isfile(os.path.join(out, "meta.json")) else {}
    ok = (rc == 0 and m.get("arm") == "RBM-10bit" and m.get("n_train") == N_SMOKE
          and _n_files(out) == 2 and _n_files(os.path.join(out, "crops")) == 2)
    check("RBM smoke runs at N=%d and writes the round-3 output contract" % N_SMOKE, ok,
          "arm=%s n_train=%s files=%d crops=%d" % (m.get("arm"), m.get("n_train"), _n_files(out),
                                                    _n_files(os.path.join(out, "crops"))))
    if ok:
        frame = np.load(os.path.join(out, "generated_000.npy"))
        crop = np.load(os.path.join(out, "crops", "generated_000.npy"))
        check("  RBM output contract: 64x64 frame, 32x32 crop, non-negative",
              frame.shape == (64, 64) and crop.shape == (32, 32)
              and float(frame.min()) >= 0.0 and float(crop.min()) >= 0.0)
    return out


def test_iqp_affine_smoke(root):
    section("1c. train_iqp_affine_r3.py --M 4 --objective mmdflat --prior resample smoke at N=%d"
           % N_SMOKE)
    out = os.path.join(root, "iqp_smoke")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = _run("train_iqp_affine_r3", COMMON_SMOKE + [
            "--M", "4", "--objective", "mmdflat", "--prior", "resample",
            "--steps", "10", "--theta0-steps", "10", "--infer-restarts", "3",
            "--infer-steps", "10", "--infer-chunk", "2", "--log-every", "10", "--out", out])
    m = _meta(out) if os.path.isfile(os.path.join(out, "meta.json")) else {}
    ok = (rc == 0 and m.get("arm") == "IQP-affine-M4-MMDflat-resample" and m.get("n_train") == N_SMOKE
          and _n_files(out) == 2 and _n_files(os.path.join(out, "crops")) == 2)
    check("IQP-affine smoke runs at N=%d and writes the round-3 output contract" % N_SMOKE, ok,
          "arm=%s n_train=%s files=%d crops=%d" % (m.get("arm"), m.get("n_train"), _n_files(out),
                                                    _n_files(os.path.join(out, "crops"))))
    if ok:
        frame = np.load(os.path.join(out, "generated_000.npy"))
        crop = np.load(os.path.join(out, "crops", "generated_000.npy"))
        check("  IQP-affine output contract: 64x64 frame, 32x32 crop, non-negative",
              frame.shape == (64, 64) and crop.shape == (32, 32)
              and float(frame.min()) >= 0.0 and float(crop.min()) >= 0.0)
    return out


def test_evaluate(root):
    section("2. evaluate_r3b.py over the three smoke outputs")
    res = os.path.join(root, "results_smoke_scaling.jsonl")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = _run("evaluate_r3b", ["--outputs-dir", root, "--n-holdout", "4",
                                   "--n-train-max", str(N_SMOKE), "--results", res,
                                   "--random-pair-arm", "IQP-affine-M4-MMDflat-resample",
                                   "--skip-missing"])
    printed = buf.getvalue()
    if not os.path.isfile(res):
        check("evaluate_r3b produced a results file", False, "no such file %r" % res)
        return None
    rows = [json.loads(l) for l in open(res)]
    arms = sorted(set(str(r.get("arm")) for r in rows))
    check("evaluate_r3b ran and produced rows", rc == 0 and len(rows) >= 3, "%d rows" % len(rows))
    check("all three scaling arms reached the table",
          set(("DCGAN-32", "RBM-10bit", "IQP-affine-M4-MMDflat-resample")).issubset(set(arms)),
          "arms = %s" % arms)
    scored = [r for r in rows if not str(r.get("arm")).startswith(("HOLDOUT", "RANDOM"))]
    check("every scored row carries finite conv_mmd2 / pdf_js",
          all(r.get("conv_mmd2") is not None and np.isfinite(r["conv_mmd2"])
              and r.get("pdf_js") is not None and np.isfinite(r["pdf_js"]) for r in scored),
          "%d scored rows" % len(scored))
    for tok in ("proj corr", "NN-within"):
        check("the printed table has the %r column" % tok, tok in printed)
    return res


def test_plot_scaling(root, results_path):
    section("3. plot_scaling.py on the (single-N-point) smoke results")
    if results_path is None:
        check("plot_scaling.py runs on the smoke results", False, "no results file from step 2")
        return
    out_png = os.path.join(root, "scaling_smoke.png")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = plot_scaling.main(["--results", results_path, "--out", out_png])
    ok = (rc == 0 and os.path.isfile(out_png) and os.path.getsize(out_png) > 1000)
    check("plot_scaling.py renders a PNG from a single-N results file without crashing", ok,
          "rc=%s size=%s" % (rc, os.path.getsize(out_png) if os.path.isfile(out_png) else None))
    check("plot_scaling.py logged all three arms' values",
          all(a in buf.getvalue() for a in ("DCGAN-32", "RBM-10bit",
                                            "IQP-affine-M4-MMDflat-resample")))


def test_pseudotime_plot(root):
    section("4. plot_pseudotime.py on the REAL round-3 pseudotime output (not smoke data)")
    pt_dir = os.path.join(ROOT, "outputs_lens_r3", "iqp_pseudotime_n100")
    if not os.path.isfile(os.path.join(pt_dir, "meta.json")):
        check("plot_pseudotime.py's source data is present", False,
              "no meta.json under %r -- round 3's pseudotime run is not finished/available" % pt_dir)
        return
    with open(os.path.join(pt_dir, "meta.json")) as fh:
        meta = json.load(fh)
    out_png = os.path.join(root, "pseudotime_smoke.png")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = plot_pseudotime.main(["--pseudotime-dir", pt_dir, "--out", out_png])
    printed = buf.getvalue()
    ok = (rc == 0 and os.path.isfile(out_png) and os.path.getsize(out_png) > 1000)
    check("plot_pseudotime.py renders a PNG from the real round-3 output", ok,
          "rc=%s size=%s" % (rc, os.path.getsize(out_png) if os.path.isfile(out_png) else None))
    expect_clamped = "%d/%d" % (int(meta["prior"]["n_targets_clamped"]), int(meta["n_gen"]))
    check("  its recomputed clamped-target count matches meta.json's own prior.n_targets_clamped "
          "(independent cross-check of the e1/latent recompute, not a copy of a stored number)",
          expect_clamped in printed, "expected %r in stdout" % expect_clamped)


def main():
    root = tempfile.mkdtemp(prefix="scaling_smoke_")
    try:
        test_dcgan_smoke(root)
        test_rbm_smoke(root)
        test_iqp_affine_smoke(root)
        res = test_evaluate(root)
        test_plot_scaling(root, res)
        test_pseudotime_plot(root)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print("\n=========================================================")
    if FAILED:
        print("FAILED (%d): %s" % (len(FAILED), FAILED))
        return 1
    print("ALL SCALING-SUITE SMOKE TESTS PASSED (%.1f s)" % (time.time() - T0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
