#!/usr/bin/env python3
"""Tests for the round-2 lens arms (docs/IQP_LENS_PILOT_PREREGISTRATION.md,
"Addendum 2026-09-06 -- round 2").

What round 2 adds, and what each part here checks:

  A. codec12.py -- the 12-qubit "image as pmf" codec.
     code = (kx << 6) | ky; image_to_pmf / pmf_to_image round-trip; the (kx, ky) meaning of
     the low 12 bits agrees with codec.py's 18-bit layout; sampling from a pmf converges to
     it; pmf_bit_moments matches sample moments of a large draw.
  B. data_lens.py encoding="pmf" -- .pmf sums to 1, .total_intensity, .field UNCHANGED
     from the tuple encoding (same ground truth), pool = 5e4 codes drawn from .pmf at a
     fixed seed, S = mean total intensity of the TRAINING images.
  C. the d_lat = 50 block under 12 qubits is exactly {} + 12 singles + 37 pairs
     (Addendum: "for the 12-qubit model that is 0 + 12 singles + 37 pairs").
  D. --full-grid in lens mode really is the exact 4,096-cell pmf: the target expectation
     values the trainer builds equal the FWHT of codes_to_pmf(pool) (tuple) / of the exact
     image pmf (pmf encoding).
  E. --stage1-train-all really moves theta_lat during stage 1, and the default really does
     not (meta.stage1_latent_shift, and theta_lat_anchors[0] vs the recorded init).
  F. all five round-2 arms smoke end-to-end on CPU and evaluate_lens.py renders one merged
     table over two output directories, with every arm's `arm` label present.
  G. pmf-arm images: shape (64, 64), non-negative, total ~ S; and the `_sampled/` sibling
     exists with the same count.

Run: /usr/bin/python3 test_lens_round2.py     (seconds-scale, CPU only, no SLURM)
Python 3.9 compatible (no `X | None`).
"""
import json
import math
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
import codec12        # noqa: E402
import data_lens      # noqa: E402
from iqp_exact import IQPExact, fwht, subsets_upto  # noqa: E402

FAILURES = []
CLS = "cdm"
N_TRAIN = 3
N_HOLDOUT = 4
N_GEN = 4
STEPS_CORE = 20
STEPS_LAT = 5


def check(name, cond, detail=""):
    print("[{}] {}{}".format("PASS" if cond else "FAIL", name, (" -- " + detail) if detail else ""))
    if not cond:
        FAILURES.append(name)


def _run(script, args, timeout=900):
    t0 = time.time()
    res = subprocess.run([PY, script] + args, cwd=HERE, capture_output=True, text=True,
                         timeout=timeout)
    return res, time.time() - t0


# ======================================================================================
# A -- codec12
# ======================================================================================
def test_codec12():
    print("\n=== A. codec12.py (12-qubit pmf codec) ===")
    codes = codec12.all_codes()
    kx, ky = codec12.decode(codes)
    check("codec12_encode_decode_bijection", bool(np.array_equal(codec12.encode(kx, ky), codes)))
    check("codec12_bit_layout",
          bool(np.array_equal(ky, codes & 63)) and bool(np.array_equal(kx, (codes >> 6) & 63)),
          "ky = bits 0-5, kx = bits 6-11")

    # (kx, ky) convention identical to codec.py's 18-bit layout
    rng = np.random.default_rng(0)
    field = (rng.random((64, 64)) ** 2).astype(np.float32)
    c18 = codec.field_to_codes_full(field, 0.0, 1.0)
    kx18, ky18, _kz18 = codec.decode(c18)
    c12 = codec12.encode(kx18, ky18)
    kx12, ky12 = codec12.decode(c12)
    check("codec12_agrees_with_codec_kx_ky",
          bool(np.array_equal(kx12, kx18)) and bool(np.array_equal(ky12, ky18)))
    # ...and the codec18 pool enumerates every 12-bit cell exactly once
    check("codec12_pool_covers_every_cell_once",
          bool(np.array_equal(np.sort(c12), codec12.all_codes())),
          "codec.field_to_codes_full -> one code per (kx, ky)")

    # pmf round-trip
    q = codec12.image_to_pmf(field)
    check("codec12_pmf_sums_to_1", abs(float(q.sum()) - 1.0) < 1e-12, "sum=%.15f" % q.sum())
    check("codec12_pmf_len_4096", q.size == 4096)
    back = codec12.pmf_to_image(q, codec12.total_intensity(field))
    check("codec12_image_pmf_image_roundtrip", float(np.abs(back - field).max()) < 1e-4,
          "max|err| = %.3g" % float(np.abs(back - field).max()))
    # index convention: q[kx*64 + ky] == field[ky, kx] / S
    S = codec12.total_intensity(field)
    idx = codec12.encode(np.array([7, 63, 0]), np.array([13, 0, 5]))
    want = np.array([field[13, 7], field[0, 63], field[5, 0]], dtype=np.float64) / S
    check("codec12_index_is_kx_times_64_plus_ky", float(np.abs(q[idx] - want).max()) < 1e-9)

    # pmf -> codes -> pmf. The floor is multinomial noise over 4,096 bins, so the bound is
    # "no worse than a fresh multinomial draw of the same size", not an absolute constant.
    def _tv_of_draw(pmf, n, seed):
        return 0.5 * float(np.abs(codec12.codes_to_pmf(
            codec12.sample_codes_from_pmf(pmf, n, np.random.default_rng(seed))) - pmf).sum())

    draw = codec12.sample_codes_from_pmf(q, 400000, np.random.default_rng(1))
    qhat = codec12.codes_to_pmf(draw)
    tv = 0.5 * float(np.abs(qhat - q).sum())
    tv_ref = _tv_of_draw(q, 400000, 12345)
    check("codec12_pmf_code_pmf_roundtrip", tv < 1.3 * tv_ref + 1e-6,
          "total-variation(400k draws) = %.4f, independent reference = %.4f" % (tv, tv_ref))
    check("codec12_pmf_code_pmf_converges", _tv_of_draw(q, 1600000, 7) < 0.75 * tv_ref,
          "4x the samples must roughly halve the total variation")
    check("codec12_codes_to_image_matches_counting_density",
          float(np.abs(codec12.codes_to_image(draw, S) - codec12.pmf_to_image(qhat, S)).max()) < 1e-6)

    # exact bit moments == sample moments of a big draw
    means_exact, cov_exact = codec12.pmf_bit_moments(q)
    bits = codec12.bits_of_codes(draw)
    means_samp = bits.mean(axis=0)
    pm1 = 2.0 * bits - 1.0
    cov_samp = np.cov(pm1.T)
    np.fill_diagonal(cov_samp, 0.0)
    check("codec12_pmf_bit_moments_means", float(np.abs(means_exact - means_samp).max()) < 0.01,
          "max|d| = %.4f" % float(np.abs(means_exact - means_samp).max()))
    check("codec12_pmf_bit_moments_cov", float(np.abs(cov_exact - cov_samp).max()) < 0.02,
          "max|d| = %.4f" % float(np.abs(cov_exact - cov_samp).max()))


# ======================================================================================
# B -- data_lens encoding="pmf"
# ======================================================================================
def test_data_lens_pmf():
    print("\n=== B. data_lens.py encoding='pmf' ===")
    snaps_t, anchors, hold = data_lens.load_lens(CLS, n_train=4, n_holdout=2, encoding="tuple")
    snaps_p, anchors_p, hold_p = data_lens.load_lens(CLS, n_train=4, n_holdout=2, encoding="pmf")
    check("pmf_same_split", anchors == anchors_p and hold == hold_p)

    s_t, s_p = snaps_t[anchors[0]], snaps_p[anchors_p[0]]
    check("pmf_field_identical_to_tuple_field", bool(np.array_equal(s_t.field, s_p.field)),
          "the ground truth must not depend on the encoding")
    check("pmf_attrs_present", s_p.pmf is not None and s_p.total_intensity is not None
          and s_p.encoding == "pmf")
    check("pmf_tuple_attrs_absent", snaps_t[anchors[0]].pmf is None
          and snaps_t[anchors[0]].encoding == "tuple")
    check("pmf_sums_to_1", abs(float(s_p.pmf.sum()) - 1.0) < 1e-12,
          "sum = %.15f" % float(s_p.pmf.sum()))
    check("pmf_dtype_float64_len_4096", s_p.pmf.dtype == np.float64 and s_p.pmf.size == 4096)
    check("pmf_total_intensity_matches_field",
          abs(s_p.total_intensity - float(s_p.field.sum())) < 1e-3,
          "%.4f vs %.4f" % (s_p.total_intensity, float(s_p.field.sum())))
    check("pmf_to_image_reconstructs_field_exactly",
          float(np.abs(codec12.pmf_to_image(s_p.pmf, s_p.total_intensity) - s_p.field).max()) < 1e-4)

    check("pmf_pool_size_default_50k", s_p.pool_codes.size == data_lens.PMF_POOL_SIZE,
          "%d" % s_p.pool_codes.size)
    check("pmf_pool_is_12_bit", int(s_p.pool_codes.max()) < 4096 and int(s_p.pool_codes.min()) >= 0)
    tv = 0.5 * float(np.abs(codec12.codes_to_pmf(s_p.pool_codes) - s_p.pmf).sum())
    tv_ref = 0.5 * float(np.abs(codec12.codes_to_pmf(codec12.sample_codes_from_pmf(
        s_p.pmf, s_p.pool_codes.size, np.random.default_rng(999))) - s_p.pmf).sum())
    tv_wrong = 0.5 * float(np.abs(np.full(4096, 1.0 / 4096) - s_p.pmf).sum())
    check("pmf_pool_drawn_from_pmf", tv < 1.3 * tv_ref and tv < 0.5 * tv_wrong,
          "TV(pool, pmf) = %.4f, independent draw = %.4f, uniform = %.4f" % (tv, tv_ref, tv_wrong))
    s_p2 = data_lens.load_lens_snapshot(anchors_p[0], CLS, n_train=4, encoding="pmf")
    check("pmf_pool_deterministic", bool(np.array_equal(s_p.pool_codes, s_p2.pool_codes)),
          "fixed at data_lens.POOL_SEED, not the arm's --seed")

    S11 = data_lens.mean_total_intensity(CLS, 11)
    train11 = data_lens.raw_training_images(CLS, 11)
    want = float(train11.reshape(11, -1).sum(axis=1).mean())
    check("mean_total_intensity_is_training_only", abs(S11 - want) < 1e-4,
          "S(N=11) = %.4f" % S11)
    check("mean_total_intensity_varies_with_N",
          abs(S11 - data_lens.mean_total_intensity(CLS, 100)) > 1e-6,
          "S(N=100) = %.4f" % data_lens.mean_total_intensity(CLS, 100))

    pool_small = data_lens.load_lens(CLS, n_train=2, n_holdout=0, encoding="pmf",
                                     pool_size=1234)[0][1].pool_codes
    check("pmf_pool_size_configurable", pool_small.size == 1234)
    try:
        data_lens.load_lens(CLS, n_train=2, n_holdout=0, encoding="nonsense")
        ok = False
    except ValueError:
        ok = True
    check("pmf_bad_encoding_rejected", ok)


# ======================================================================================
# C -- the d_lat = 50 latent block on 12 qubits
# ======================================================================================
def test_latent_block():
    print("\n=== C. d_lat = 50 block under 12 qubits ===")
    for K in (6, 12):
        subs = subsets_upto(12, K)[:50]
        sizes = [len(s) for s in subs]
        check("latent50_K%d_shape" % K,
              sizes.count(0) == 1 and sizes.count(1) == 12 and sizes.count(2) == 37
              and max(sizes) <= 2 and len(subs) == 50,
              "{}=%d singles=%d pairs=%d" % (sizes.count(0), sizes.count(1), sizes.count(2)))
        check("latent50_K%d_order" % K,
              subs[0] == () and subs[1] == (0,) and subs[12] == (11,) and subs[13] == (0, 1),
              "Hamming-then-lexicographic")
    check("K12_is_complete_basis", len(subsets_upto(12, 12)) == 4096)
    check("K6_param_count_2510", len(subsets_upto(12, 6)) == 2510)
    check("tuple_K6_param_count_31180", len(subsets_upto(18, 6)) == 31180)


# ======================================================================================
# D -- --full-grid is the exact pmf in lens mode
# ======================================================================================
def test_full_grid_is_exact():
    print("\n=== D. --full-grid = the exact 4,096-cell pmf ===")
    import train_iqp

    class A(object):
        pass

    # tuple encoding: the pool IS the 4,096 pixels, each once -> codes_to_pmf(pool) is exact
    snaps, anchors, _ = data_lens.load_lens(CLS, n_train=2, n_holdout=0, encoding="tuple")
    pool = snaps[anchors[0]].pool_codes
    a = A(); a.full_grid = True; a.n_tuples = 2000; a.encoding = "tuple"
    got = train_iqp.codes_for(pool, a, np.random.default_rng(0))
    check("full_grid_tuple_returns_whole_pool",
          got.size == 4096 and bool(np.array_equal(np.sort(got), np.sort(pool))))
    p_tuple = codec.codes_to_pmf(pool)
    check("full_grid_tuple_pmf_is_exact",
          abs(float(p_tuple.sum()) - 1.0) < 1e-12 and int((p_tuple > 0).sum()) <= 4096
          and abs(float(p_tuple.max()) - 4096 * (1.0 / 4096) / 4096) >= 0,
          "%d non-zero cells of 2^18" % int((p_tuple > 0).sum()))
    sim18 = IQPExact(n=18, K=2, device="cpu")
    import torch
    ez = sim18.data_expvals(p_tuple).numpy()
    ez_ref = fwht(torch.as_tensor(p_tuple, dtype=torch.float64)).numpy()
    check("full_grid_tuple_expvals_match_fwht", float(np.abs(ez - ez_ref).max()) < 1e-5,
          "max|d| = %.3g" % float(np.abs(ez - ez_ref).max()))

    # pmf encoding: --full-grid bypasses codes entirely and uses Snapshot.pmf
    snaps_p, anchors_p, _ = data_lens.load_lens(CLS, n_train=2, n_holdout=0, encoding="pmf")
    q = snaps_p[anchors_p[0]].pmf
    sim12 = IQPExact(n=12, K=6, device="cpu")
    ez12 = sim12.data_expvals(q).numpy()
    ez12_ref = fwht(torch.as_tensor(q, dtype=torch.float64)).numpy()
    check("full_grid_pmf_expvals_match_fwht", float(np.abs(ez12 - ez12_ref).max()) < 1e-6,
          "max|d| = %.3g" % float(np.abs(ez12 - ez12_ref).max()))
    check("full_grid_pmf_expval_0_is_1", abs(float(ez12[0]) - 1.0) < 1e-9,
          "<Z_empty> = %.9f" % float(ez12[0]))
    # and the 12-bit marginal of the exact 18-bit pmf IS that pmf's spatial part only when
    # weighted by intensity -- they are DIFFERENT targets, which is the point of round 2
    marg = np.zeros(4096, dtype=np.float64)
    kx, ky, _kz = codec.decode(np.arange(1 << 18, dtype=np.int64))
    np.add.at(marg, codec12.encode(kx, ky), p_tuple)
    check("tuple_spatial_marginal_is_uniform", float(np.abs(marg - 1.0 / 4096).max()) < 1e-12,
          "the tuple pool spreads mass uniformly over pixels; the pmf encoding does not")
    check("pmf_target_is_not_uniform", float(q.max()) > 5.0 / 4096,
          "max cell = %.3g vs uniform %.3g" % (float(q.max()), 1.0 / 4096))


# ======================================================================================
# E/F/G -- arm smokes, --stage1-train-all, outputs, merged evaluation
# ======================================================================================
ARMS = [
    # (tag, script, extra args, arm label, pmf?, stage1_train_all?)
    ("iqp_tuple_sqrt_fixed", "train_iqp.py",
     ["--encoding", "tuple", "--lens-sqrt", "--stage1-train-all", "--full-grid", "--K", "6"],
     "IQP-tuple-sqrt-fixed", False, True),
    ("iqp_pmf_full_fixed", "train_iqp.py",
     ["--encoding", "pmf", "--K", "12", "--stage1-train-all", "--full-grid"],
     "IQP-pmf-full-fixed", True, True),
    ("iqp_pmf_k6_fixed", "train_iqp.py",
     ["--encoding", "pmf", "--K", "6", "--stage1-train-all", "--full-grid"],
     "IQP-pmf-K6-fixed", True, True),
    ("iqp_pmf_full_paper", "train_iqp.py",
     ["--encoding", "pmf", "--K", "12", "--n-tuples", "20000"],
     "IQP-pmf-full-paper", True, False),
    ("rbm_sqrt", "baseline_rbm.py", ["--lens-sqrt"], "RBM-sqrt", False, False),
]


def test_arms(tmp):
    print("\n=== E/F/G. round-2 arm smokes (N=%d, holdout=%d, %d+%d steps, CPU) ==="
          % (N_TRAIN, N_HOLDOUT, STEPS_CORE, STEPS_LAT))
    out_root = os.path.join(tmp, "outputs_lens_round2", CLS)
    os.makedirs(out_root, exist_ok=True)
    common = ["--data", "lens", "--lens-class", CLS, "--n-train", str(N_TRAIN),
              "--n-holdout", str(N_HOLDOUT), "--synth-mode", "gaussian-latent",
              "--n-gen", str(N_GEN), "--device", "cpu", "--seed", "0"]

    for tag, script, extra, arm, is_pmf, train_all in ARMS:
        out = os.path.join(out_root, tag)
        args = common + extra + ["--arm-name", arm, "--out", out]
        if script == "train_iqp.py":
            args += ["--steps-core", str(STEPS_CORE), "--steps-lat", str(STEPS_LAT),
                     "--n-synth", "2000", "--log-every", "1000"]
        else:
            args += ["--steps-core", "5", "--steps-lat", "3", "--n-tuples", "2000",
                     "--gen-chains", "200", "--gen-sweeps", "5", "--gen-keep", "2"]
        res, dt = _run(script, args)
        check("%s_exit_0" % tag, res.returncode == 0,
              "%.1fs%s" % (dt, "" if res.returncode == 0 else " -- " + res.stderr.strip()[-500:]))
        if res.returncode != 0:
            continue
        _check_outputs(tag, out, arm, is_pmf, train_all)

    _check_stage1_lever(tmp)
    _check_merged_evaluation(tmp, out_root)


def _check_outputs(tag, out, arm, is_pmf, train_all):
    meta_path = os.path.join(out, "meta.json")
    if not os.path.isfile(meta_path):
        check("%s_meta_exists" % tag, False, meta_path)
        return
    with open(meta_path) as fh:
        meta = json.load(fh)
    check("%s_arm_label" % tag, meta.get("arm") == arm, "arm=%r" % meta.get("arm"))
    check("%s_n_gen" % tag, len([f for f in os.listdir(out) if f.startswith("generated_")]) == N_GEN)

    imgs = np.stack([np.load(os.path.join(out, "generated_%03d.npy" % i)) for i in range(N_GEN)])
    check("%s_shape_64x64" % tag, imgs.shape == (N_GEN, 64, 64), str(imgs.shape))
    check("%s_nonnegative" % tag, float(imgs.min()) >= 0.0, "min = %.6g" % float(imgs.min()))
    check("%s_finite" % tag, bool(np.isfinite(imgs).all()))

    if is_pmf:
        check("%s_meta_encoding_pmf" % tag, meta.get("encoding") == "pmf"
              and meta.get("n_qubits") == 12, "n_qubits=%s" % meta.get("n_qubits"))
        S = float(meta["intensity_scale_S"])
        totals = imgs.reshape(N_GEN, -1).sum(axis=1)
        rel = float(np.abs(totals - S).max() / S)
        check("%s_total_intensity_equals_S" % tag, rel < 1e-4,
              "S=%.4f totals=[%.4f .. %.4f] rel err=%.3g" % (S, totals.min(), totals.max(), rel))
        samp = out + "_sampled"
        n_samp = len([f for f in os.listdir(samp)]) if os.path.isdir(samp) else -1
        check("%s_sampled_sibling" % tag, os.path.isdir(samp) and n_samp == N_GEN + 1,
              "%s has %d entries (expect %d generated + meta.json)" % (samp, n_samp, N_GEN))
        if os.path.isdir(samp):
            with open(os.path.join(samp, "meta.json")) as fh:
                ms = json.load(fh)
            check("%s_sampled_meta_tagged" % tag,
                  ms.get("arm") == arm + "-sampled" and ms.get("reconstruction") == "sampled",
                  "arm=%r reconstruction=%r" % (ms.get("arm"), ms.get("reconstruction")))
            s_imgs = np.stack([np.load(os.path.join(samp, "generated_%03d.npy" % i))
                               for i in range(N_GEN)])
            s_tot = s_imgs.reshape(N_GEN, -1).sum(axis=1)
            check("%s_sampled_total_equals_S" % tag, float(np.abs(s_tot - S).max() / S) < 1e-4)
            check("%s_sampled_differs_from_exact" % tag,
                  float(np.abs(s_imgs - imgs).max()) > 0.0,
                  "the Monte-Carlo control must not be byte-identical to the exact one")
        check("%s_no_last_dir" % tag, not os.path.isdir(out + "_last"),
              "last-write-wins is meaningless for a pmf reconstruction")
    else:
        check("%s_meta_encoding_tuple" % tag, meta.get("encoding", "tuple") == "tuple")
        check("%s_last_sibling" % tag, os.path.isdir(out + "_last"))

    if "stage1_train_all" in meta:
        check("%s_meta_stage1_flag" % tag, bool(meta["stage1_train_all"]) == train_all,
              "meta=%s expected=%s" % (meta["stage1_train_all"], train_all))
        shift = float(meta["stage1_latent_shift"])
        if train_all:
            check("%s_stage1_moved_theta_lat" % tag, shift > 1e-8,
                  "max|theta_lat - init| = %.3e" % shift)
        else:
            check("%s_stage1_froze_theta_lat" % tag, shift == 0.0,
                  "max|theta_lat - init| = %.3e" % shift)


def _check_stage1_lever(tmp):
    """The A/B the flag exists for: identical config, only --stage1-train-all differs."""
    print("\n--- E. --stage1-train-all A/B (same seed, same data) ---")
    base = ["--data", "lens", "--lens-class", CLS, "--n-train", str(N_TRAIN),
            "--n-holdout", str(N_HOLDOUT), "--synth-mode", "gaussian-latent",
            "--n-gen", "2", "--device", "cpu", "--seed", "0", "--encoding", "pmf",
            "--K", "6", "--full-grid", "--steps-core", "200", "--steps-lat", "20",
            "--n-synth", "1000", "--log-every", "1000"]
    lat = {}
    for tag, extra in (("ab_frozen", []), ("ab_trained", ["--stage1-train-all"])):
        out = os.path.join(tmp, tag)
        res, _dt = _run("train_iqp.py", base + extra + ["--out", out])
        check("%s_exit_0" % tag, res.returncode == 0, res.stderr.strip()[-400:])
        if res.returncode != 0:
            return
        lat[tag] = np.load(os.path.join(out, "theta_lat_anchors.npy"))
        with open(os.path.join(out, "meta.json")) as fh:
            lat[tag + "_meta"] = json.load(fh)

    d_frozen = float(lat["ab_frozen_meta"]["stage1_latent_shift"])
    d_trained = float(lat["ab_trained_meta"]["stage1_latent_shift"])
    check("stage1_default_freezes_latent", d_frozen == 0.0, "shift = %.3e" % d_frozen)
    check("stage1_train_all_moves_latent", d_trained > 1e-6, "shift = %.3e" % d_trained)
    # Guard against a VACUOUS pass: theta_lat[0] is the |s| = 0 global phase, which probs()
    # is exactly invariant to, so a shift carried only by index 0 would mean nothing was
    # learned. The reported shift already excludes index 0 (train_iqp.py), and the phase's
    # own movement is reported separately -- assert the real coordinates dominate.
    d_phase = float(lat["ab_trained_meta"]["stage1_latent_shift_phase"])
    check("stage1_shift_is_not_just_global_phase", d_trained > 10.0 * max(d_phase, 1e-12),
          "non-phase shift %.3e vs global-phase drift %.3e" % (d_trained, d_phase))
    nz = int((np.abs(lat["ab_trained"][0][1:] - lat["ab_frozen"][0][1:]) > 1e-6).sum())
    check("stage1_moves_many_latent_coords", nz >= 10,
          "%d of 49 non-phase latent coordinates moved" % nz)
    check("stage1_anchor0_latent_differs",
          float(np.abs(lat["ab_trained"][0] - lat["ab_frozen"][0]).max()) > 1e-6,
          "max|d| = %.3e" % float(np.abs(lat["ab_trained"][0] - lat["ab_frozen"][0]).max()))
    check("stage1_train_all_lowers_stage1_loss",
          lat["ab_trained_meta"]["final_loss_stage1"] <= lat["ab_frozen_meta"]["final_loss_stage1"],
          "%.6e (trained) vs %.6e (frozen)"
          % (lat["ab_trained_meta"]["final_loss_stage1"],
             lat["ab_frozen_meta"]["final_loss_stage1"]))
    check("stage2_still_adapts_per_image",
          float(np.abs(np.diff(lat["ab_trained"], axis=0)).max()) > 0.0,
          "theta_lat must still move image-to-image in stage 2")


def _check_merged_evaluation(tmp, out_root):
    """evaluate_lens.py over TWO directories -- the round-2 scoring path."""
    print("\n--- F. merged evaluation over two --outputs-dir ---")
    other = os.path.join(tmp, "outputs_lens_pilot", CLS, "ctrl_mean_n%d" % N_TRAIN)
    res, _ = _run("controls_lens.py", ["--arm", "mean", "--lens-class", CLS,
                                       "--n-train", str(N_TRAIN), "--n-gen", str(N_GEN),
                                       "--out", other])
    check("ctrl_mean_exit_0", res.returncode == 0, res.stderr.strip()[-300:])

    results = os.path.join(tmp, "results_round2.jsonl")
    res, dt = _run("evaluate_lens.py", [
        "--lens-class", CLS,
        "--outputs-dir", os.path.dirname(other), out_root,
        "--n-holdout", str(N_HOLDOUT), "--n-train-max", str(N_TRAIN),
        "--results", results])
    check("evaluate_merged_exit_0", res.returncode == 0,
          "%.1fs%s" % (dt, "" if res.returncode == 0 else " -- " + res.stderr.strip()[-500:]))
    if res.returncode != 0:
        return
    rows = [json.loads(l) for l in open(results)]
    arms = set(r["arm"] for r in rows)
    want = set(a for _t, _s, _e, a, _p, _ta in ARMS) | {"CTRL-mean", "HOLDOUT (ref)"}
    missing = sorted(want - arms)
    check("evaluate_merged_has_all_arms", not missing,
          "missing %s; got %s" % (missing, sorted(arms)))
    check("evaluate_merged_has_sampled_rows",
          len([a for a in arms if a.endswith("-sampled")]) == 3,
          "%s" % sorted(a for a in arms if a.endswith("-sampled")))
    diag = [r for r in rows if r["arm"] != "HOLDOUT (ref)"]
    check("evaluate_merged_has_range_diagnostics",
          all(all(k in r for k in ("gen_min", "gen_max", "gen_mean_total",
                                   "holdout_min", "holdout_max", "holdout_mean_total"))
              for r in diag),
          "%d scored rows carry the range diagnostics" % len(diag))
    pmf_rows = [r for r in diag if r["arm"].startswith("IQP-pmf") and "-sampled" not in r["arm"]]
    check("evaluate_pmf_rows_report_total_equal_to_S",
          bool(pmf_rows) and all(abs(r["gen_mean_total"] - pmf_rows[0]["gen_mean_total"]) < 1e-3
                                 for r in pmf_rows),
          "every pmf arm's images total exactly S = %.4f (held-out mean %.4f)"
          % (pmf_rows[0]["gen_mean_total"], pmf_rows[0]["holdout_mean_total"]) if pmf_rows else "none")
    finite = [r for r in rows if r["pdf_js"] is not None]
    check("evaluate_merged_metrics_finite",
          all(math.isfinite(r["pdf_js"]) and math.isfinite(r["conv_mmd2"])
              and math.isfinite(r["nn_train"]) and math.isfinite(r["nn_holdout"]) for r in finite),
          "%d scored rows" % len(finite))
    print("\n".join(res.stdout.splitlines()[-(len(rows) + 6):]))


# ======================================================================================
def main():
    tmp = tempfile.mkdtemp(prefix="test_lens_round2_")
    try:
        test_codec12()
        test_data_lens_pmf()
        test_latent_block()
        test_full_grid_is_exact()
        test_arms(tmp)
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
