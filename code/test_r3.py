#!/usr/bin/env python3
"""Round-3 unit + smoke tests. CPU only, tens of seconds -- the sbatch runs it as a fail-fast
pre-check so a broken arm cannot burn an 8gpus allocation.

Covers, in order:
  1  codec10 round trip and code layout
  2  align_crop / paste are mutual inverses on 200 real images (light budget re-measured)
  3  the two latent blocks' |s| distributions and their overlap
  4  the physical-quantity functions against a synthetic elliptical Gaussian of known (e1, e2)
  5  CTRL-ring: the (e1, e2, radius) measured back from a render vs the values asked for
  6  CTRL-interp-e1 is exact at an anchor, and the paired metrics are 0 when gen == target
  6b addendum: mmdflat's flat Walsh weights, gauss8's rank-8 affine subspace, resample's
     jitter-distance calibration (unit tests, no training)
  6c addendum: intensity-aligned rescaling hits its target exactly (unit test, no training)
  7  joint-MMD smoke (N = 3, 20 steps, 4 samples)
  8  KL smoke: the loss is finite and decreases
  9  paper-protocol smoke
 10  pseudotime smoke, including the conditional output contract
 11  RBM and DCGAN smokes
 12  the three unconditional controls + interp_e1 as a subprocess-free arm smoke
 12b addendum arm smokes: MMDflat, prior8, resample (N = 3, 20 steps)
 13  evaluate_r3 over everything the smokes wrote (including the addendum arms and the
     intensity-aligned secondary table)

Python 3.9 compatible.
"""
from __future__ import annotations

import contextlib
import io
import os
import shutil
import sys
import tempfile
import time

import numpy as np
import torch

torch.set_num_threads(min(4, os.cpu_count() or 1))

import codec10
import controls_r3
import data_lens_r3
import evaluate_r3
import latent_select_r3
import train_iqp_r3
from iqp_exact import popcount_table

FAILED = []
T0 = time.time()


def check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    if not ok:
        FAILED.append(name)
    print("[%s] %-52s %s  (%.1fs)" % (tag, name, detail, time.time() - T0))
    return bool(ok)


def section(title):
    print("\n--- %s ---" % title)


# ======================================================================================
def test_codec10():
    section("1. codec10")
    rng = np.random.default_rng(0)
    crop = rng.random((32, 32)) ** 3
    q = codec10.image_to_pmf(crop)
    back = codec10.pmf_to_image(q, codec10.total_intensity(crop))
    check("codec10 pmf round trip", float(np.abs(back - crop.astype(np.float32)).max()) < 1e-6,
          "max|err|=%.3g" % float(np.abs(back - crop.astype(np.float32)).max()))
    kx, ky = codec10.decode(codec10.all_codes())
    check("codec10 encode(decode(c)) == c",
          bool(np.array_equal(codec10.encode(kx, ky), codec10.all_codes())))
    check("codec10 reshape order matches (kx<<5)|ky",
          bool(np.allclose(codec10.image_to_pmf(crop),
                           np.ascontiguousarray(crop.T).ravel() / crop.sum())))
    check("codec10 pmf sums to 1", abs(float(q.sum()) - 1.0) < 1e-12)


def test_align_paste():
    section("2. align_crop / paste")
    d = data_lens_r3.load_r3(11, n_holdout=200)
    imgs = d.holdout_images
    keep = np.empty(imgs.shape[0])
    err = np.empty(imgs.shape[0])
    for i in range(imgs.shape[0]):
        c, dy, dx = data_lens_r3.align_crop(imgs[i])
        f = codec10.paste(c, dy, dx)
        tot = float(imgs[i].sum())
        keep[i] = float(c.sum()) / tot
        err[i] = float(np.abs(f - imgs[i]).sum()) / tot
        if i == 0:
            win_exact = bool(np.array_equal(f[dy:dy + 32, dx:dx + 32], imgs[i][dy:dy + 32, dx:dx + 32]))
    check("paste reproduces the window exactly", win_exact)
    check("paste light loss <= 2% on 200 real images", float(err.max()) <= 0.02,
          "mean=%.4f%% max=%.4f%% (kept mean=%.4f%% min=%.4f%%)"
          % (100 * err.mean(), 100 * err.max(), 100 * keep.mean(), 100 * keep.min()))
    check("codec10.paste rejects an out-of-frame corner",
          _raises(lambda: codec10.paste(np.zeros((32, 32), np.float32), 40, 0)))


def _raises(fn):
    try:
        fn()
    except Exception:
        return True
    return False


def test_latent_blocks():
    section("3. latent blocks")
    pc = popcount_table(10).astype(np.int64)
    lv = latent_select_r3.load_latent_masks("lowvar")
    hm = latent_select_r3.load_latent_masks("hamming")
    hl = np.bincount(pc[lv], minlength=11).tolist()
    hh = np.bincount(pc[hm], minlength=11).tolist()
    print("      lowvar50  |s| histogram = %s" % hl[:4])
    print("      hamming50 |s| histogram = %s" % hh[:4])
    print("      |lowvar ^ hamming| = %d" % len(set(lv.tolist()) & set(hm.tolist())))
    check("lowvar50 has 50 unique masks, all |s| <= 2",
          lv.size == 50 and len(set(lv.tolist())) == 50 and int(pc[lv].max()) <= 2)
    check("hamming50 = empty + 10 singles + 39 pairs", hh[:3] == [1, 10, 39] and sum(hh) == 50)
    check("lowvar50 drops the empty set (its variance is exactly 0)", 0 not in set(lv.tolist()))


def test_moments():
    section("4. physical quantities")
    ny = nx = 64
    ys = np.arange(ny)[:, None] - 31.5
    xs = np.arange(nx)[None, :] - 31.5
    worst = 0.0
    for sx, sy, rho in ((5.0, 5.0, 0.0), (7.0, 4.0, 0.0), (5.0, 5.0, 6.0), (6.5, 4.5, -4.0)):
        C = np.array([[sx * sx, rho], [rho, sy * sy]], dtype=np.float64)
        Ci = np.linalg.inv(C)
        img = np.exp(-0.5 * (Ci[0, 0] * xs * xs + 2 * Ci[0, 1] * xs * ys + Ci[1, 1] * ys * ys))
        tr = C[0, 0] + C[1, 1]
        want = ((C[0, 0] - C[1, 1]) / tr, 2 * C[0, 1] / tr, float(np.sqrt(tr)))
        got = evaluate_r3.image_quantities(img)
        e = max(abs(got["e1"] - want[0]), abs(got["e2"] - want[1]))
        worst = max(worst, e)
        print("      sx=%.1f sy=%.1f rho=%+.1f  e1 %+.4f/%+.4f  e2 %+.4f/%+.4f  R %.3f/%.3f"
              % (sx, sy, rho, got["e1"], want[0], got["e2"], want[1], got["radius"], want[2]))
    check("e1 / e2 recovered from a known elliptical Gaussian to < 0.02", worst < 0.02,
          "worst |err| = %.5f" % worst)
    crop = np.zeros((32, 32), np.float32)
    crop[14:18, 12:20] = 1.0
    frame = codec10.paste(crop, 7, 9)
    a, b = evaluate_r3.image_quantities(crop), evaluate_r3.image_quantities(frame)
    same = max(abs(a["flux"] - b["flux"]), abs(a["radius"] - b["radius"]),
               abs(a["e1"] - b["e1"]), abs(a["e2"] - b["e2"]))
    check("flux / radius / e1 / e2 are unchanged by pasting", same < 1e-9, "max diff %.3g" % same)


def test_ring():
    section("5. CTRL-ring geometry")
    d = data_lens_r3.load_r3(11, n_holdout=0)
    radii, prof = controls_r3.mean_radial_profile(d.train_crops)
    R0 = controls_r3.template_R0(radii, prof)
    rng = np.random.default_rng(0)
    q = controls_r3.crop_moments(d.train_crops)
    pick = rng.integers(0, q.shape[0], size=60)
    drawn = q[pick]
    back = np.asarray([controls_r3.moments(controls_r3.render_ring(radii, prof, *drawn[i][:1].tolist()
                                                                  + drawn[i][1:].tolist(), R0=R0))
                       for i in range(drawn.shape[0])])
    de1 = float(np.abs(back[:, 2] - drawn[:, 2]).mean())
    de2 = float(np.abs(back[:, 3] - drawn[:, 3]).mean())
    dR = float(np.abs(back[:, 1] - drawn[:, 1]).mean())
    dF = float(np.abs(back[:, 0] - drawn[:, 0]).max())
    check("ring render: measured-back e1 error < 0.02", de1 < 0.02, "mean |de1| = %.4f" % de1)
    check("ring render: measured-back e2 error < 0.02", de2 < 0.02, "mean |de2| = %.4f" % de2)
    check("ring render: rms radius error < 0.2 px", dR < 0.2,
          "mean |dR| = %.4f px (template R0 = %.4f)" % (dR, R0))
    check("ring render: total flux hit exactly", dF < 1e-3, "max |dF| = %.3g" % dF)


def test_interp_e1_and_paired():
    section("6. CTRL-interp-e1 and the paired block")
    d = data_lens_r3.load_r3(11, n_holdout=20)
    q = controls_r3.crop_moments(d.train_crops)
    out = controls_r3.interp_crops_by_e1(d.train_crops, q[:, 2], q[:, 2])
    err = float(np.abs(out.astype(np.float64) - d.train_crops.astype(np.float64)).max())
    check("interp_e1 at an anchor e1 returns that training crop", err < 1e-5, "max|err| = %.3g" % err)
    lo = float(q[:, 2].min()) - 1.0
    hi = float(q[:, 2].max()) + 1.0
    clamped = controls_r3.interp_crops_by_e1(d.train_crops, q[:, 2], [lo, hi])
    order = np.argsort(q[:, 2])
    check("interp_e1 clamps outside the training e1 range",
          float(np.abs(clamped[0] - d.train_crops[order[0]]).max()) < 1e-5 and
          float(np.abs(clamped[1] - d.train_crops[order[-1]]).max()) < 1e-5)
    p0 = evaluate_r3.paired_block(d.holdout_crops, d.holdout_crops)
    worst = max(abs(p0[k]) for k in ("pair_de1", "pair_de2", "pair_dflux", "pair_dradius", "pair_l2"))
    check("paired metrics are 0 when generated == target", worst < 1e-9, "worst = %.3g" % worst)


def test_addendum_units():
    section("6b. addendum unit tests: mmdflat weights, gauss8 subspace, resample jitter")

    # --objective mmdflat: c_s = 2^-10 for every one of the 1,024 subsets, sums to 1.
    w = train_iqp_r3.flat_kernel_weights(1024).numpy()
    check("mmdflat weights are a constant vector", bool(np.all(w == w[0])), "w0=%.10g" % w[0])
    check("mmdflat weights equal 2^-10", abs(float(w[0]) - 2.0 ** -10) < 1e-15, "w0=%.10g" % w[0])
    check("mmdflat weights sum to 1", abs(float(w.sum()) - 1.0) < 1e-9, "sum=%.12f" % float(w.sum()))

    # --prior gauss8: draws must lie in the affine subspace spanned by the top-8 principal
    # components of the training latents (residual after projecting out that subspace ~ 0).
    rng = np.random.default_rng(0)
    X = rng.standard_normal((11, 50))
    prior = train_iqp_r3.fit_latent_gaussian(X, "diag", max_rank=8)
    check("gauss8 prior retains rank 8 (min(8, N-1) at N=11)", prior["rank"] == 8,
          "rank=%d" % prior["rank"])
    draws = train_iqp_r3.sample_latents(prior, 200, np.random.default_rng(1))
    mu = X.mean(axis=0)
    Xc = X - mu[None, :]
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    comps8 = Vt[:8]
    resid = (draws - mu[None, :]) - ((draws - mu[None, :]) @ comps8.T) @ comps8
    worst = float(np.abs(resid).max())
    check("gauss8 draws lie in the top-8 PCA affine subspace (residual < 1e-8)",
          worst < 1e-8, "max residual = %.3g" % worst)

    # --prior resample: jitter_std must equal exactly 0.2x the median NN spacing, and the
    # ACTUAL median Euclidean distance from a resample draw to its nearest training latent
    # must land close to that same 0.2x spacing (50% tolerance) -- not 0.2x spacing times
    # sqrt(d), which is what a naive per-coordinate std of 0.2x spacing would give at d=50.
    lat_gen, spacing, jitter_std = train_iqp_r3.sample_latents_resample(
        X, 3000, np.random.default_rng(2), jitter_scale=0.2)
    check("resample jitter_std == 0.2 x median NN spacing",
          abs(jitter_std - 0.2 * spacing) < 1e-9,
          "jitter_std=%.6g  0.2*spacing=%.6g" % (jitter_std, 0.2 * spacing))
    dist = np.sqrt(((lat_gen[:, None, :] - X[None, :, :]) ** 2).sum(axis=-1)).min(axis=1)
    med = float(np.median(dist))
    check("resample: median distance to nearest training latent ~= 0.2x spacing (50% tol)",
          abs(med - jitter_std) < 0.5 * jitter_std,
          "median dist=%.4f  jitter_std=%.4f  spacing=%.4f" % (med, jitter_std, spacing))


def test_intensity_alignment():
    section("6c. addendum unit test: intensity-aligned rescaling (evaluate_r3.align_to_target)")

    # Every image's OWN total must hit the target exactly (not just the set mean) -- each
    # image gets its own scalar, drawn from a different random brightness so a bug that only
    # rescales by the SET's mean (rather than per-image) would still pass a mean-only check.
    rng = np.random.default_rng(3)
    base = rng.random((7, 64, 64)).astype(np.float32)
    brightness = rng.uniform(0.2, 5.0, size=(7, 1, 1)).astype(np.float32)
    imgs = base * brightness
    target = 123.456
    aligned = evaluate_r3.align_to_target(imgs, target)
    totals = aligned.reshape(aligned.shape[0], -1).sum(axis=1).astype(np.float64)
    rel = np.abs(totals - target) / target
    check("align_to_target: every image's own total hits the target (rel err < 1e-9)",
          float(rel.max()) < 1e-9, "worst rel err = %.3g" % float(rel.max()))
    check("align_to_target: the set's mean total therefore also equals the target",
          abs(float(totals.mean()) - target) / target < 1e-9,
          "mean=%.6f target=%.6f" % (float(totals.mean()), target))

    # An all-zero image cannot be scaled multiplicatively into a positive target -- must be
    # left unchanged (finite), not turned into inf/nan by a division by 0.
    zero = np.zeros((1, 64, 64), dtype=np.float32)
    safe = evaluate_r3.align_to_target(zero, target)
    check("align_to_target leaves an all-zero (unscalable) image unchanged, not inf/nan",
          bool(np.all(safe == 0.0)) and bool(np.all(np.isfinite(safe))))

    # CTRL-copy's invariant this feeds into evaluate_r3 (section 13): the SAME per-image rule
    # applied to two bit-identical arrays yields bit-identical outputs, at any target scale.
    same = evaluate_r3.align_to_target(imgs, target)
    check("align_to_target is deterministic (identical input -> identical output)",
          bool(np.array_equal(aligned, same)))


# ======================================================================================
def _run(mod_name, argv):
    """Call a round-3 arm's main() in-process (no subprocess: the tests must stay seconds)."""
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


def test_arm_smokes(root):
    import json

    section("7-12. arm smokes (N = 3, tiny step counts)")
    common = ["--n-train", "3", "--n-holdout", "20", "--n-gen", "4", "--device", "cpu"]

    out = os.path.join(root, "iqp_joint_mmd")
    _run("train_iqp_r3", common + ["--protocol", "joint", "--objective", "mmd",
                                   "--steps", "20", "--log-every", "10", "--out", out])
    m = json.load(open(os.path.join(out, "meta.json")))
    check("joint-MMD smoke wrote 4 frames + 4 crops",
          _n_files(out) == 4 and _n_files(os.path.join(out, "crops")) == 4)
    check("joint-MMD smoke: 64x64 frames, 32x32 crops",
          np.load(os.path.join(out, "generated_000.npy")).shape == (64, 64) and
          np.load(os.path.join(out, "crops", "generated_000.npy")).shape == (32, 32))
    check("joint-MMD smoke: loss finite and decreasing",
          np.isfinite(m["loss_final_total"]) and m["loss_final_total"] < m["loss_first_logged"],
          "%.4g -> %.4g" % (m["loss_first_logged"], m["loss_final_total"]))
    crop0 = np.load(os.path.join(out, "crops", "generated_000.npy")).astype(np.float64)
    check("joint-MMD smoke: crop total == S (pmf arms carry the scale exactly)",
          abs(crop0.sum() - m["S"]) < 1e-3 * m["S"], "%.4f vs S=%.4f" % (crop0.sum(), m["S"]))
    check("joint-MMD smoke: param split is 974 core + 50 latent",
          m["d_core"] == 974 and m["d_lat"] == 50 and m["param_count"] == 1024)

    out = os.path.join(root, "iqp_joint_kl")
    _run("train_iqp_r3", common + ["--protocol", "joint", "--objective", "kl",
                                   "--steps", "20", "--log-every", "10", "--out", out])
    m = json.load(open(os.path.join(out, "meta.json")))
    check("KL smoke: loss finite and decreasing",
          np.isfinite(m["loss_final_total"]) and m["loss_final_total"] < m["loss_first_logged"],
          "%.4g -> %.4g" % (m["loss_first_logged"], m["loss_final_total"]))
    check("KL smoke: KL >= 0", m["loss_final_total"] >= 0.0)

    out = os.path.join(root, "iqp_hamming")
    _run("train_iqp_r3", common + ["--latent", "hamming", "--protocol", "joint",
                                   "--steps", "20", "--log-every", "10", "--out", out])
    m = json.load(open(os.path.join(out, "meta.json")))
    check("hamming latent smoke runs and flags the inert empty coefficient",
          m["latent_empty_included"] and m["latent_effective_dim"] == 49)

    # -- addendum arms (2026-09-07): MMDflat objective, prior8 / resample latent priors -----
    out = os.path.join(root, "iqp_mmdflat")
    _run("train_iqp_r3", common + ["--protocol", "joint", "--objective", "mmdflat",
                                   "--steps", "20", "--log-every", "10", "--out", out])
    m = json.load(open(os.path.join(out, "meta.json")))
    check("MMDflat smoke: arm name and finite decreasing loss",
          m["arm"] == "IQP-lowvar-joint-MMDflat" and _n_files(out) == 4 and
          np.isfinite(m["loss_final_total"]) and m["loss_final_total"] < m["loss_first_logged"],
          "arm=%s  %.4g -> %.4g" % (m["arm"], m["loss_first_logged"], m["loss_final_total"]))
    check("MMDflat smoke: sigmas empty (bandwidth rule is bypassed)", m["sigmas"] == [])

    out = os.path.join(root, "iqp_prior8")
    _run("train_iqp_r3", common + ["--protocol", "joint", "--objective", "mmd",
                                   "--prior", "gauss8", "--steps", "20", "--log-every", "10",
                                   "--out", out])
    m = json.load(open(os.path.join(out, "meta.json")))
    # rank is capped by BOTH max_rank=8 AND the rank actually available at this N (<= N-1
    # after mean-centring); the smoke trains at N=3 (`common`), so min(8, N-1) = 2, not 8 --
    # the "retains rank 8" case needs N-1 >= 8, i.e. N >= 9 (see test_addendum_units, N=11).
    check("prior8 smoke: arm name and rank capped at min(8, N-1) = 2 at this smoke's N=3",
          m["arm"] == "IQP-lowvar-joint-MMD-prior8" and _n_files(out) == 4 and
          m["prior"]["prior_mode"] == "gauss8" and m["prior"]["rank"] == min(8, 3 - 1),
          "arm=%s  rank=%s" % (m["arm"], m["prior"]["rank"]))

    out = os.path.join(root, "iqp_resample")
    _run("train_iqp_r3", common + ["--protocol", "joint", "--objective", "mmd",
                                   "--prior", "resample", "--steps", "20", "--log-every", "10",
                                   "--out", out])
    m = json.load(open(os.path.join(out, "meta.json")))
    check("resample smoke: arm name and jitter/spacing recorded",
          m["arm"] == "IQP-lowvar-joint-MMD-resample" and _n_files(out) == 4 and
          m["prior"]["kind"] == "resample-with-gaussian-jitter" and
          m["prior"]["jitter_std"] > 0.0 and m["prior"]["median_nn_spacing"] > 0.0,
          "arm=%s  jitter_std=%.4g  spacing=%.4g"
          % (m["arm"], m["prior"]["jitter_std"], m["prior"]["median_nn_spacing"]))

    out = os.path.join(root, "iqp_paper")
    _run("train_iqp_r3", common + ["--protocol", "paper", "--steps-core", "20",
                                   "--steps-lat", "10", "--log-every", "10", "--out", out])
    m = json.load(open(os.path.join(out, "meta.json")))
    check("paper-protocol smoke", _n_files(out) == 4 and m["protocol"] == "paper"
          and not m["conditional"] and np.isfinite(m["loss_final_total"]))

    out = os.path.join(root, "iqp_pseudotime")
    _run("train_iqp_r3", common + ["--protocol", "pseudotime", "--n-gen", "20",
                                   "--steps-core", "20", "--steps-lat", "10",
                                   "--log-every", "10", "--out", out])
    m = json.load(open(os.path.join(out, "meta.json")))
    check("pseudotime smoke is conditional with one sample per held-out image",
          m["conditional"] and _n_files(out) == 20 and m["prior"]["n_targets"] == 20,
          "e1 train [%.3f, %.3f], %d targets clamped"
          % (m["prior"]["e1_train_min"], m["prior"]["e1_train_max"], m["prior"]["n_targets_clamped"]))

    out = os.path.join(root, "rbm")
    _run("baseline_rbm_r3", common + ["--steps", "20", "--log-every", "10", "--n-pool", "2000",
                                      "--gen-sweeps", "20", "--gen-keep", "10", "--gen-chunk", "4",
                                      "--out", out])
    m = json.load(open(os.path.join(out, "meta.json")))
    check("RBM smoke (11,010 params, 10 x 1000)", _n_files(out) == 4 and m["param_count"] == 11010)

    out = os.path.join(root, "dcgan")
    _run("baseline_dcgan_r3", common + ["--steps", "20", "--log-every", "10", "--out", out])
    m = json.load(open(os.path.join(out, "meta.json")))
    check("DCGAN smoke (~0.2M generator params)",
          _n_files(out) == 4 and 1.8e5 < m["param_count"] < 2.2e5,
          "generator = %d params" % m["param_count"])

    for arm, label in (("copy", "CTRL-copy"), ("mean", "CTRL-mean"), ("ring", "CTRL-ring"),
                       ("interp_e1", "CTRL-interp-e1")):
        out = os.path.join(root, "ctrl_" + arm)
        _run("controls_r3", ["--arm", arm, "--n-train", "3", "--n-holdout", "20",
                             "--n-gen", "4" if arm != "interp_e1" else "20", "--out", out])
        m = json.load(open(os.path.join(out, "meta.json")))
        want = 3 if arm == "copy" else (20 if arm == "interp_e1" else 4)
        check("control %s smoke" % label,
              _n_files(out) == want and m["arm"] == label and
              _n_files(os.path.join(out, "crops")) == want)
    m = __import__("json").load(open(os.path.join(root, "ctrl_interp_e1", "meta.json")))
    check("CTRL-interp-e1 is tagged conditional", bool(m["conditional"]))


def test_evaluate(root):
    section("13. evaluate_r3 over the smoke outputs")
    res = os.path.join(root, "results_smoke_r3.jsonl")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = _run("evaluate_r3", ["--outputs-dir", root, "--n-holdout", "20",
                                  "--n-train-max", "3", "--results", res])
    printed = buf.getvalue()
    sys.stdout.write(printed)  # still visible in this test's own log, just captured too
    import json
    rows = [json.loads(l) for l in open(res)]
    arms = sorted(set(r["arm"] for r in rows))
    check("evaluate_r3 ran and produced rows", rc == 0 and len(rows) >= 12, "%d rows" % len(rows))
    check("HOLDOUT split-half reference present", "HOLDOUT split-half" in arms)
    check("RANDOM-PAIRING reference present", "RANDOM-PAIRING (ref)" in arms)
    check("addendum arms (MMDflat / prior8 / resample) reached the evaluated table",
          set(("IQP-lowvar-joint-MMDflat", "IQP-lowvar-joint-MMD-prior8",
               "IQP-lowvar-joint-MMD-resample")).issubset(set(arms)), "arms=%s" % arms)
    cond = [r for r in rows if r["arm"] in ("IQP-lowvar-pseudotime-MMD", "CTRL-interp-e1")]
    check("conditional arms carry paired columns",
          len(cond) == 2 and all(r["pair_l2"] is not None for r in cond))
    uncond = [r for r in rows if r["arm"] in ("CTRL-mean", "CTRL-ring", "DCGAN-32")]
    check("unconditional arms leave the paired columns blank",
          all(r["pair_l2"] is None for r in uncond))
    floor = [r for r in rows if r["arm"] == "HOLDOUT split-half"][0]
    check("split-half physics floor is small (a class-blocked split would blow it up)",
          floor["physics"] < 0.6, "physics floor = %.4f" % floor["physics"])
    for r in rows:
        if r.get("physics") is not None and not np.isfinite(r["physics"]):
            check("physics score finite for %s" % r["arm"], False)

    # -- addendum (2026-09-07): intensity-aligned secondary block ------------------------
    ALIGNED_KEYS = ("pdf_js_aligned", "conv_mmd2_aligned", "nn_train_aligned", "nn_holdout_aligned")
    check("every row carries the four aligned columns",
          all(k in r for r in rows for k in ALIGNED_KEYS))
    scored = [r for r in rows if r["arm"] != "RANDOM-PAIRING (ref)"]  # has no distributional block
    check("aligned PDF-JS/MMD2/NN are finite wherever the unaligned ones are",
          all(np.isfinite(r[k]) for r in scored for k in ALIGNED_KEYS))
    copy_rows = [r for r in rows if r["arm"] == "CTRL-copy" and r["n_train"] == 3]
    check("CTRL-copy NN-train is ~0 both before and after intensity alignment "
          "(memorisation ceiling survives the rescaling)",
          len(copy_rows) == 1 and abs(copy_rows[0]["nn_train"]) < 1e-4 and
          abs(copy_rows[0]["nn_train_aligned"]) < 1e-4,
          "nn_train=%s nn_train_aligned=%s"
          % ((copy_rows[0]["nn_train"], copy_rows[0]["nn_train_aligned"]) if copy_rows
             else (None, None)))
    check("evaluate_r3 printed the intensity-aligned table",
          "intensity-aligned" in printed and "NN-train (aligned)" in printed)


def main():
    root = tempfile.mkdtemp(prefix="r3_smoke_")
    try:
        test_codec10()
        test_align_paste()
        test_latent_blocks()
        test_moments()
        test_ring()
        test_interp_e1_and_paired()
        test_addendum_units()
        test_intensity_alignment()
        test_arm_smokes(root)
        test_evaluate(root)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print("\n=========================================================")
    if FAILED:
        print("FAILED (%d): %s" % (len(FAILED), FAILED))
        return 1
    print("ALL ROUND-3 TESTS PASSED (%.1f s)" % (time.time() - T0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
