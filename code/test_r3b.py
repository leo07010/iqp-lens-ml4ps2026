#!/usr/bin/env python3
"""Round-3b unit + smoke tests. CPU only, seconds -- submit_lens_r3b.sbatch runs it as a
fail-fast pre-check so a broken arm cannot burn an 8gpus allocation.

Covers, in order:
  1  affine model: Phi row normalisation, theta shapes, gradients reach theta0 / Phi / z,
     probabilities normalised
  2  the projection metric's correlation is the same in pmf space and in decoded crop space
  3  restart machinery: nearest_train_index, build_restart_inits' declared order (zeros,
     nearest-training z, random block), and the arm-name -zpen suffix
  4  --infer-chunk invariance: the same held-out projection corrs at chunk 2 and chunk 4
     (the batched inference must be exactly N_ho x R independent problems)
  5  resample prior: the jitter's typical DISTANCE is 0.2 x the median NN spacing
  6  PCA control: rank cut, projection corr == 1 on the training crops when M >= N-1,
     rendered draws are non-negative and normalised
  7  arm smokes at N = 3, 40 steps: M4 x {kl, mmdflat, auto3}, M2-kl, M8-kl-zpen and the
     three PCA controls -- output contract and meta keys
  8  evaluate_r3b over everything the smokes wrote: the projection / diversity / dispersion
     columns reach the table and the jsonl
  9  zpen is RELATIVE to L0 (train_iqp_affine_r3.zpen_penalty): same --zpen and the same z give
     the same penalty/L0 ratio at the KL and MMDflat objectives, even though their L0 differ by
     orders of magnitude

Python 3.9 compatible.
"""
from __future__ import annotations

import contextlib
import io
import json
import math
import os
import shutil
import sys
import tempfile
import time

import numpy as np
import torch

torch.set_num_threads(min(4, os.cpu_count() or 1))

import codec10
import controls_pca_r3
import data_lens_r3
import evaluate_r3b
import train_iqp_affine_r3 as A
from iqp_exact import IQPExact
from train_iqp_r3 import probs_batch, sample_latents_resample

FAILED = []
T0 = time.time()


def check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    if not ok:
        FAILED.append(name)
    print("[%s] %-58s %s  (%.1fs)" % (tag, name, detail, time.time() - T0))
    return bool(ok)


def section(title):
    print("\n--- %s ---" % title)


class _Args(object):
    def __init__(self, **kw):
        self.__dict__.update(kw)


# ======================================================================================
def test_model():
    section("1. affine model: normalisation, shapes, gradients")
    torch.manual_seed(0)
    sim = IQPExact(10, 10, dtype=torch.complex128)
    M, B = 3, 5
    theta0 = torch.randn(1024, dtype=torch.float64) * 0.01
    phi_raw = torch.randn(M, 1024, dtype=torch.float64) * 7.0      # deliberately mis-scaled
    z = torch.randn(B, M, dtype=torch.float64) * 0.05
    phi = A.normalise_phi(phi_raw)
    norms = phi.norm(dim=1).detach().numpy()
    check("Phi rows are L2-normalised to sqrt(1024) = 32",
          np.allclose(norms, A.PHI_SCALE, atol=1e-9),
          "norms in [%.6f, %.6f]" % (norms.min(), norms.max()))
    check("Phi normalisation is scale-invariant in the raw parameter",
          torch.allclose(A.normalise_phi(phi_raw), A.normalise_phi(3.7 * phi_raw), atol=1e-12))
    th = A.theta_of(theta0, phi_raw, z)
    check("theta_of shape is (B, 1024)", tuple(th.shape) == (B, 1024), str(tuple(th.shape)))
    check("z = 0 gives exactly theta0 in every row",
          torch.allclose(A.theta_of(theta0, phi_raw, torch.zeros(B, M, dtype=torch.float64)),
                         theta0.unsqueeze(0).expand(B, 1024)))

    for p in (theta0, phi_raw, z):
        p.requires_grad_(True)
    pmf = torch.rand(B, 1024, dtype=torch.float64) + 1e-3
    pmf = pmf / pmf.sum(dim=1, keepdim=True)
    q = probs_batch(sim, A.theta_of(theta0, phi_raw, z))
    check("probabilities are normalised", np.allclose(q.detach().numpy().sum(axis=1), 1.0, atol=1e-10),
          "row sums in [%.12f, %.12f]" % (float(q.sum(1).min()), float(q.sum(1).max())))
    check("probabilities are non-negative", bool((q.detach() >= -1e-15).all()))
    loss = A.kl_rows(pmf, q).mean()
    loss.backward()
    g = [theta0.grad, phi_raw.grad, z.grad]
    check("KL gradient reaches theta0, Phi and z (all non-zero, all finite)",
          all(x is not None and torch.isfinite(x).all() and float(x.abs().max()) > 0 for x in g),
          "|g| max = %s" % ["%.3g" % float(x.abs().max()) for x in g])
    check("kl_rows / mmd_rows return one number per image",
          A.kl_rows(pmf, q).shape == (B,) and
          A.mmd_rows(q, torch.zeros_like(q), torch.ones(1024, dtype=torch.float64)).shape == (B,))
    kl_self = float(A.kl_rows(pmf, pmf).abs().max())
    check("KL(p || p) == 0", kl_self < 1e-12, "max |KL| = %.3g" % kl_self)


def test_corr_space():
    section("2. the projection correlation is codec-invariant")
    rng = np.random.default_rng(0)
    a = rng.random((4, 1024)) ** 3
    b = rng.random((4, 1024)) ** 3
    a = a / a.sum(axis=1, keepdims=True)
    b = b / b.sum(axis=1, keepdims=True)
    c_pmf = A.corr_rows(a, b)
    c_img = np.asarray([np.corrcoef(codec10.pmf_to_image(a[i], 137.0).ravel(),
                                    codec10.pmf_to_image(b[i], 4.2).ravel())[0, 1]
                        for i in range(4)])
    check("corr in pmf space == corr of the decoded 32x32 crops (any positive S)",
          np.allclose(c_pmf, c_img, atol=1e-9), "max |diff| = %.3g" % np.abs(c_pmf - c_img).max())
    check("corr_rows of identical rows is 1", np.allclose(A.corr_rows(a, a), 1.0, atol=1e-12))
    cm = A.corr_matrix(a, a)
    check("corr_matrix diagonal is 1 and it agrees with corr_rows",
          np.allclose(np.diag(cm), 1.0, atol=1e-12) and
          np.allclose(cm[0], A.corr_rows(np.repeat(a[:1], 4, axis=0), a), atol=1e-12))


def test_restarts_and_names():
    section("3. restart initialisation and arm names")
    tr = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    ho = np.array([[0.9, 0.1, 0.0], [0.0, 0.05, 0.95]])
    near = A.nearest_train_index(ho, tr)
    check("nearest_train_index picks the nearest training pmf in L2",
          near.tolist() == [0, 2], "got %s" % near.tolist())

    z_tr = np.array([[0.1, -0.2], [0.3, 0.4], [-0.5, 0.6]])
    rb = np.arange(2 * 6 * 2, dtype=np.float64).reshape(2, 6, 2)
    z0 = A.build_restart_inits(z_tr, near, 8, rb)
    check("restart block shape is (m, R, M)", z0.shape == (2, 8, 2), str(z0.shape))
    check("restart 0 is exactly zeros", np.allclose(z0[:, 0, :], 0.0))
    check("restart 1 IS the nearest training image's trained z (the fix the M=8 collapse needed)",
          np.allclose(z0[0, 1], z_tr[0]) and np.allclose(z0[1, 1], z_tr[2]),
          "z0[.,1] = %s" % z0[:, 1, :].tolist())
    check("restarts 2..R-1 are the supplied random block, in order",
          np.allclose(z0[:, 2:, :], rb))
    check("R = 1 degenerates to the zeros start only",
          A.build_restart_inits(z_tr, near, 1).shape == (2, 1, 2))

    check("arm name at the default z penalty carries no tag",
          A.default_arm_name(_Args(arm_name=None, M=4, objective="kl", zpen=1e-3, prior="gauss"))
          == "IQP-affine-M4-KL-gauss")
    check("arm name at a NON-default z penalty carries -zpen",
          A.default_arm_name(_Args(arm_name=None, M=8, objective="kl", zpen=0.3, prior="resample"))
          == "IQP-affine-M8-KL-zpen-resample")
    check("objective labels match the pre-registration's arm table",
          [A.default_arm_name(_Args(arm_name=None, M=2, objective=o, zpen=1e-3, prior="gauss"))
           for o in ("kl", "mmdflat", "auto3")]
          == ["IQP-affine-M2-KL-gauss", "IQP-affine-M2-MMDflat-gauss",
              "IQP-affine-M2-auto3-gauss"])
    check("PCA control arm names match the pre-registration's arm table",
          [controls_pca_r3.default_arm_name(_Args(arm_name=None, M=m, prior=p))
           for m, p in ((2, "gauss"), (4, "gauss"), (4, "resample"))]
          == ["CTRL-pca2-gauss", "CTRL-pca4-gauss", "CTRL-pca4-resample"])


def test_resample_scale():
    section("5. resample prior jitter scale")
    rng = np.random.default_rng(0)
    z = rng.normal(size=(20, 4)) * 0.03
    draws, spacing, jstd = sample_latents_resample(z, 4000, np.random.default_rng(1),
                                                   jitter_scale=0.2)
    check("jitter_std is 0.2 x the median nearest-neighbour spacing",
          abs(jstd - 0.2 * spacing) < 1e-12, "spacing=%.6g jitter=%.6g" % (spacing, jstd))
    # each draw's distance from its own base is not recoverable without the index, so measure
    # the distance to the NEAREST training latent: an upper bound on it, and within 20% here.
    d = np.sqrt(((draws[:, None, :] - z[None, :, :]) ** 2).sum(-1)).min(axis=1)
    check("the typical perturbation DISTANCE is ~jitter_std, not jitter_std x sqrt(M)",
          0.4 * jstd < float(d.mean()) < 1.3 * jstd,
          "mean NN distance %.6g vs jitter_std %.6g (sqrt(M) x = %.6g)"
          % (float(d.mean()), jstd, jstd * math.sqrt(4)))


def test_pca_control():
    section("6. PCA control units")
    rng = np.random.default_rng(0)
    X = rng.random((6, 1024)) ** 2
    X = X / X.sum(axis=1, keepdims=True)
    mu, V = controls_pca_r3.fit_pca(X, 20)
    check("fit_pca drops the null directions (N centred rows span N-1 dims)",
          V.shape[0] == 5, "kept %d components for N=6" % V.shape[0])
    check("components are orthonormal",
          np.allclose(V @ V.T, np.eye(V.shape[0]), atol=1e-9))
    rec = controls_pca_r3.reconstruct(controls_pca_r3.project(X, mu, V), mu, V)
    check("projection is exact on the training rows when M >= N-1 (corr == 1)",
          np.allclose(A.corr_rows(rec, X), 1.0, atol=1e-9),
          "min corr = %.12f" % float(A.corr_rows(rec, X).min()))
    mu2, V2 = controls_pca_r3.fit_pca(X, 2)
    rec2 = controls_pca_r3.reconstruct(controls_pca_r3.project(X, mu2, V2), mu2, V2)
    check("a TRUNCATED projection is not exact (so the metric can discriminate)",
          float(A.corr_rows(rec2, X).mean()) < 0.999,
          "M=2 train corr = %.4f" % float(A.corr_rows(rec2, X).mean()))
    neg = np.array([[-1.0, 2.0, 3.0], [0.5, 0.5, -4.0]])
    p, nbad = controls_pca_r3.to_pmf(neg)
    check("to_pmf clips to non-negative and renormalises to 1",
          bool((p >= 0).all()) and np.allclose(p.sum(axis=1), 1.0) and nbad == 0,
          "%s" % p.tolist())
    p2, nbad2 = controls_pca_r3.to_pmf(np.array([[0.0, 0.0, -4.0]]))
    check("an all-non-positive reconstruction falls back to uniform AND is counted",
          nbad2 == 1 and np.allclose(p2, 1.0 / 3.0), "n_degenerate = %d" % nbad2)


# ======================================================================================
def _run(mod_name, argv):
    """Call an arm's main() in-process (no subprocess: the tests must stay seconds)."""
    import importlib
    mod = importlib.import_module(mod_name)
    old = sys.argv
    sys.argv = [mod_name] + list(argv)
    try:
        rc = mod.main()
    finally:
        sys.argv = old
    return rc


def _n_files(d):
    return len([f for f in os.listdir(d) if f.startswith("generated_") and f.endswith(".npy")])


def _meta(d):
    with open(os.path.join(d, "meta.json")) as fh:
        return json.load(fh)


SMOKE = ["--n-train", "3", "--n-holdout", "6", "--n-gen", "4", "--device", "cpu",
         "--steps", "40", "--theta0-steps", "30", "--log-every", "20",
         "--infer-restarts", "4", "--infer-steps", "20"]


def test_chunk_invariance(root):
    section("4. --infer-chunk invariance of the held-out projection")
    outs = []
    for chunk in (2, 6):
        o = os.path.join(root, "chunk%d" % chunk)
        with contextlib.redirect_stdout(io.StringIO()):
            _run("train_iqp_affine_r3", SMOKE + ["--M", "4", "--objective", "kl",
                                                 "--infer-chunk", str(chunk), "--out", o])
        outs.append(np.asarray(_meta(o)["projection_corrs"], dtype=np.float64))
    d = float(np.abs(outs[0] - outs[1]).max())
    check("chunk 2 and chunk 6 give the SAME per-image projection corrs",
          d < 1e-12, "max |diff| = %.3g" % d)
    check("both runs projected all 6 held-out images", outs[0].size == 6 and outs[1].size == 6)


def test_arm_smokes(root):
    section("7. arm smokes (N = 3, 40 steps)")
    cases = [
        ("m4_kl", ["--M", "4", "--objective", "kl", "--prior", "gauss"],
         "IQP-affine-M4-KL-gauss"),
        ("m4_mmdflat", ["--M", "4", "--objective", "mmdflat", "--prior", "resample"],
         "IQP-affine-M4-MMDflat-resample"),
        ("m4_auto3", ["--M", "4", "--objective", "auto3", "--prior", "resample"],
         "IQP-affine-M4-auto3-resample"),
        ("m2_kl", ["--M", "2", "--objective", "kl", "--prior", "resample"],
         "IQP-affine-M2-KL-resample"),
        ("m8_zpen", ["--M", "8", "--objective", "kl", "--prior", "resample", "--zpen", "0.3"],
         "IQP-affine-M8-KL-zpen-resample"),
    ]
    for tag, extra, arm in cases:
        out = os.path.join(root, tag)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = _run("train_iqp_affine_r3", SMOKE + extra + ["--out", out])
        m = _meta(out)
        pc = np.asarray(m["projection_corrs"], dtype=np.float64)
        ok = (rc == 0 and m["arm"] == arm and _n_files(out) == 4
              and _n_files(os.path.join(out, "crops")) == 4
              and np.all(np.isfinite(pc)) and pc.size == 6
              and abs(m["projection_corr_mean"] - float(pc.mean())) < 1e-12
              and m["param_count"] == 1024 + m["M"] * 1024 + 3 * m["M"])
        check("smoke %s -> %s" % (tag, arm), ok,
              "train corr %.4f, projection corr %.4f (min %.4f), restarts used %s"
              % (m["train_corr_mean"], m["projection_corr_mean"], m["projection_corr_min"],
                 m["projection_restart_hist"]))
        frame = np.load(os.path.join(out, "generated_000.npy"))
        crop = np.load(os.path.join(out, "crops", "generated_000.npy"))
        check("  %s output contract: 64x64 frame, 32x32 crop, non-negative" % tag,
              frame.shape == (64, 64) and crop.shape == (32, 32)
              and float(frame.min()) >= 0.0 and float(crop.min()) >= 0.0)
        if tag == "m8_zpen":
            check("  the z-penalty arm records the non-default zpen it ran with",
                  abs(m["zpen"] - 0.3) < 1e-12 and m["projection_lr"] == 0.005,
                  "zpen=%s infer lr=%s" % (m["zpen"], m["projection_lr"]))

    for tag, mm, prior, arm in (("pca2_gauss", 2, "gauss", "CTRL-pca2-gauss"),
                                ("pca4_gauss", 4, "gauss", "CTRL-pca4-gauss"),
                                ("pca4_resample", 4, "resample", "CTRL-pca4-resample")):
        out = os.path.join(root, tag)
        with contextlib.redirect_stdout(io.StringIO()):
            rc = _run("controls_pca_r3", ["--M", str(mm), "--prior", prior, "--n-train", "3",
                                          "--n-holdout", "6", "--n-gen", "4", "--out", out])
        m = _meta(out)
        ok = (rc == 0 and m["arm"] == arm and _n_files(out) == 4
              and _n_files(os.path.join(out, "crops")) == 4
              and m["projection_corr_mean"] is not None)
        check("smoke %s -> %s" % (tag, arm), ok,
              "M kept %d, train corr %.6f, projection corr %.4f"
              % (m["M"], m["train_corr_mean"], m["projection_corr_mean"]))
        if mm >= 2:      # M >= N-1 = 2 at this N: the training projection must be exact
            check("  %s reconstructs its own training crops exactly (M >= N-1)" % tag,
                  abs(m["train_corr_min"] - 1.0) < 1e-9, "min train corr = %.12f" % m["train_corr_min"])
        crop = np.load(os.path.join(out, "crops", "generated_000.npy"))
        check("  %s draws are non-negative and carry the training scale S" % tag,
              float(crop.min()) >= 0.0 and abs(float(crop.sum()) - m["S"]) < 1e-2,
              "crop sum = %.4f vs S = %.4f" % (float(crop.sum()), m["S"]))


def test_evaluate(root):
    section("8. evaluate_r3b over the smoke outputs")
    res = os.path.join(root, "results_smoke_r3b.jsonl")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = _run("evaluate_r3b", ["--outputs-dir", root, "--n-holdout", "6",
                                   "--n-train-max", "3", "--results", res,
                                   "--random-pair-arm", "IQP-affine-M4-KL-gauss"])
    printed = buf.getvalue()
    sys.stdout.write(printed)
    rows = [json.loads(l) for l in open(res)]
    arms = sorted(set(r["arm"] for r in rows))
    check("evaluate_r3b ran and produced rows", rc == 0 and len(rows) >= 9, "%d rows" % len(rows))
    check("all eight round-3b arm kinds reached the table",
          set(("IQP-affine-M4-KL-gauss", "IQP-affine-M4-MMDflat-resample",
               "IQP-affine-M4-auto3-resample", "IQP-affine-M2-KL-resample",
               "IQP-affine-M8-KL-zpen-resample", "CTRL-pca2-gauss", "CTRL-pca4-gauss",
               "CTRL-pca4-resample")).issubset(set(arms)), "arms = %s" % arms)
    check("HOLDOUT split-half reference present", "HOLDOUT split-half" in arms)
    scored = [r for r in rows if not r["arm"].startswith(("HOLDOUT", "RANDOM"))]
    check("every scored row carries a finite projection corr (mean and min)",
          all(r["projection_corr_mean"] is not None and np.isfinite(r["projection_corr_mean"])
              and r["projection_corr_min"] is not None for r in scored))
    check("every scored row carries the diversity / dispersion columns",
          all(r["nn_within"] is not None and np.isfinite(r["nn_within"])
              and r["e1_sd"] is not None and r["e2_sd"] is not None for r in scored))
    check("the held-out reference values are attached to every row and agree",
          len(set(round(r["holdout_nn_within"], 9) for r in rows)) == 1
          and len(set(round(r["holdout_e1_sd"], 9) for r in rows)) == 1)
    hs = [r for r in rows if r["arm"] == "HOLDOUT split-half"][0]
    check("the split-half row's own diversity IS the held-out value",
          abs(hs["nn_within"] - hs["holdout_nn_within"]) < 1e-12)
    check("physics score is finite for every scored row",
          all(np.isfinite(r["physics"]) for r in scored))
    for tok in ("proj corr", "NN-within", "e1 sd", "ho e1 sd", "intensity-aligned"):
        check("the printed table has the %r column" % tok, tok in printed)
    check("the aligned secondary table still renders",
          all(k in rows[0] for k in ("pdf_js_aligned", "conv_mmd2_aligned",
                                     "nn_train_aligned", "nn_holdout_aligned")))


def test_zpen_relative_scale():
    section("9. zpen is relative to L0: same penalty/L0 ratio at every objective")
    torch.manual_seed(0)
    sim = IQPExact(10, 10, dtype=torch.complex128)
    n = 5
    theta0 = torch.randn(1024, dtype=torch.float64) * 0.01
    tr = torch.rand(n, 1024, dtype=torch.float64) + 1e-3
    tr = tr / tr.sum(dim=1, keepdim=True)
    ez = A.fwht(tr)
    flat_w = A.flat_kernel_weights(sim.size).to(dtype=torch.float64)
    theta0_bcast = theta0.unsqueeze(0).repeat(n, 1)
    with torch.no_grad():
        L0_kl = float(A.loss_rows(sim, theta0_bcast, tr, ez, flat_w, "kl").mean().item())
        L0_mmd = float(A.loss_rows(sim, theta0_bcast, tr, ez, flat_w, "mmdflat").mean().item())
    check("KL and MMDflat L0 are genuinely different numbers here (sanity check that this "
          "test isn't a degenerate case where the ratio check below is trivially true)",
          L0_kl > 0 and L0_mmd > 0 and L0_kl / L0_mmd > 2.0,
          "L0_kl=%.6g L0_mmd=%.6g ratio=%.3g" % (L0_kl, L0_mmd, L0_kl / max(L0_mmd, 1e-300)))

    zpen = 1e-3
    z = torch.tensor([[0.02, -0.01, 0.03], [0.10, 0.00, -0.05]], dtype=torch.float64)
    pen_kl = float(A.zpen_penalty(z, zpen, L0_kl, reduce="mean").item())
    pen_mmd = float(A.zpen_penalty(z, zpen, L0_mmd, reduce="mean").item())
    ratio_kl = pen_kl / L0_kl
    ratio_mmd = pen_mmd / L0_mmd
    d = abs(ratio_kl - ratio_mmd)
    check("same --zpen, same z -> same penalty/L0 ratio at KL and at MMDflat "
          "(the whole point of scaling the penalty by L0)",
          d < 1e-9,
          "ratio_kl=%.12g ratio_mmd=%.12g diff=%.3g" % (ratio_kl, ratio_mmd, d))

    # infer_z_batched uses reduce="sum" (for --infer-chunk invariance, see its docstring); it
    # must carry the exact same per-image penalty as reduce="mean", just combined differently.
    pen_kl_sum = float(A.zpen_penalty(z, zpen, L0_kl, reduce="sum").item())
    expected_sum = pen_kl * z.shape[0]
    check("  reduce='sum' (infer_z_batched's path) is reduce='mean' x B: same per-image "
          "penalty, only the reduction differs",
          abs(pen_kl_sum - expected_sum) < 1e-9 * max(abs(expected_sum), 1.0),
          "sum=%.9g mean*B=%.9g" % (pen_kl_sum, expected_sum))


def main():
    root = tempfile.mkdtemp(prefix="r3b_smoke_")
    try:
        test_model()
        test_corr_space()
        test_restarts_and_names()
        test_resample_scale()
        test_pca_control()
        test_chunk_invariance(root)
        test_arm_smokes(root)
        test_evaluate(root)
        test_zpen_relative_scale()
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print("\n=========================================================")
    if FAILED:
        print("FAILED (%d): %s" % (len(FAILED), FAILED))
        return 1
    print("ALL ROUND-3B TESTS PASSED (%.1f s)" % (time.time() - T0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
