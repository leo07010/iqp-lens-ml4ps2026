#!/usr/bin/env python3
"""Round-3b pixel-space PCA control: the classical LINEAR analogue of the learned IQP latent.

docs/IQP_LENS_ROUND3_PREREGISTRATION.md, "Round 3b". NEW file; it imports data_lens_r3 /
codec10 / train_iqp_affine_r3 and modifies nothing round 3's queued sbatch uses.

Why this arm exists
-------------------
Round 3b's IQP learns an M-dimensional AFFINE subspace of PHASE space (theta = theta0 + Phi^T z)
and reads images off it through the circuit. This control learns an M-dimensional affine
subspace of PIXEL space (PCA of the same centred crops, same pmf normalisation, same M, same
priors, same held-out projection metric) and reads images off it by linear reconstruction. The
pre-registration's decision rule is stated against it: beating CTRL-pca4-resample by more than
the split-half floor is "the IQP is a genuinely useful generator", matching it is "the IQP
behaves as a linear model in phase space", losing to it is failure.

Representation (deliberately identical to the IQP arm)
------------------------------------------------------
Each training crop is turned into its codec10 pmf (crop / total intensity), so both arms model
the same 1,024-vector on the same simplex and the scale S is carried outside the model in both.
PCA is taken of those pmfs: mu = mean pmf, V = the top-M right singular vectors of the centred
matrix, and coefficients a_i = (x_i - mu) V^T.

Held-out projection corr
------------------------
The ORTHOGONAL projection onto the affine subspace mu + span(V):

    x_hat = mu + ((x - mu) V^T) V

correlated with the held-out pmf, per image, mean and min. This is the exact analogue of the
IQP arm's "best reconstruction reachable inside the model's latent space", and it is closed
form here -- no optimiser, so no restart machinery. It is reported WITHOUT the non-negativity
clip: the clip is a rendering step for generation, and clipping first would report a different
(strictly non-linear) map than the projection the decision rule names. Both numbers are in
meta.json anyway (projection_corr_* and projection_corr_clipped_*), so the choice is visible.

With M >= N - 1 the training crops span the whole subspace and their own projection corr is
exactly 1; test_r3b.py checks that.

Generation (--prior, the same two rules as the IQP arm)
------------------------------------------------------
gauss     Gaussian fitted to the N training coefficients (full covariance at N >= 100, diagonal
          below -- round 3's N rule).
resample  a training coefficient vector drawn WITH REPLACEMENT plus Gaussian jitter of typical
          DISTANCE 0.2 x the median nearest-neighbour spacing (train_iqp_r3's own function).

A draw is reconstructed, CLIPPED to non-negative (a pmf cannot be negative and the circuit's
output never is, so the control must not be handed an advantage the IQP arm does not have),
renormalised to sum 1, multiplied by S and pasted at an offset drawn with replacement from the
training (dy, dx) -- round 3's output contract, byte-compatible with evaluate_r3.py:

  <out>/generated_{i:03d}.npy , <out>/crops/generated_{i:03d}.npy , <out>/meta.json ,
  pca_mean.npy , pca_components.npy , pca_coeffs_train.npy , pca_coeffs_holdout.npy

Arm name: CTRL-pca{M}-{gauss|resample}.

Python 3.9 compatible (no `X | None`).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

import codec10
import data_lens_r3
from train_iqp_affine_r3 import (corr_matrix, corr_rows, fit_z_gaussian, sample_z_gaussian)
from train_iqp_r3 import sample_latents_resample

NCODES = codec10.NCODES


def fit_pca(X, m, rank_tol=1e-10):
    """(mu, V) with V the top-`m` right singular vectors (rows) of the centred data.

    Directions whose singular value is below `rank_tol` x the largest are DROPPED before the
    top-m cut: N centred crops span at most N - 1 dimensions, so asking for m >= N would
    otherwise hand the control an arbitrary null direction that the held-out data could still
    project onto. The returned V therefore has min(m, rank) rows; meta.json records both the
    requested and the realised M.
    """
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(axis=0)
    Xc = X - mu[None, :]
    _U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    keep = s > (rank_tol * max(float(s.max()), 1e-300))
    Vt = Vt[keep]
    m = int(min(int(m), Vt.shape[0]))
    if m < 1:
        raise ValueError("fit_pca: the centred training set has rank 0")
    return mu, np.ascontiguousarray(Vt[:m])


def project(X, mu, V):
    """Coefficients a = (X - mu) V^T of the orthogonal projection onto mu + span(V)."""
    return (np.asarray(X, dtype=np.float64) - mu[None, :]) @ V.T


def reconstruct(a, mu, V):
    return mu[None, :] + np.asarray(a, dtype=np.float64) @ V


def to_pmf(rows):
    """Clip to non-negative and renormalise each row to sum 1 (the rendering step).

    A row whose clipped mass is 0 (a linear reconstruction that came out non-positive
    everywhere) cannot be normalised; it falls back to the UNIFORM pmf and is counted, because
    silently emitting a zero image would be scored as a legitimate sample. main() puts the
    count in meta.json as gen_degenerate_rows and warns when it is non-zero. On real data it
    is 0: the PCA mean is itself a pmf and the coefficient draws are small.
    """
    r = np.clip(np.asarray(rows, dtype=np.float64), 0.0, None)
    tot = r.sum(axis=1, keepdims=True)
    bad = (tot[:, 0] <= 0.0)
    out = np.where(tot > 0.0, r / np.maximum(tot, 1e-300), 1.0 / float(r.shape[1]))
    return out, int(bad.sum())


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Round-3b pixel-space PCA control.")
    p.add_argument("--M", type=int, default=4)
    p.add_argument("--prior", choices=("gauss", "resample"), default="gauss")
    p.add_argument("--n-train", type=int, default=11)
    p.add_argument("--n-holdout", type=int, default=data_lens_r3.N_HOLDOUT)
    p.add_argument("--n-gen", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--arm-name", default=None)
    return p.parse_args(argv)


def default_arm_name(args):
    if args.arm_name:
        return args.arm_name
    return "CTRL-pca%d-%s" % (int(args.M), args.prior)


def main(argv=None):
    args = parse_args(argv)
    t0 = time.time()
    if int(args.M) < 1:
        raise SystemExit("controls_pca_r3.py: --M must be >= 1, got %d" % args.M)
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "crops"), exist_ok=True)
    rng = np.random.default_rng(args.seed + 4242)

    d = data_lens_r3.load_r3(args.n_train, seed=args.seed, n_holdout=args.n_holdout)
    n = d.n_train
    S = d.S
    tr_pmf = np.asarray(d.train_pmfs, dtype=np.float64)
    n_ho = int(d.holdout_crops.shape[0])
    ho_pmf = (np.stack([codec10.image_to_pmf(d.holdout_crops[i]) for i in range(n_ho)], axis=0)
              if n_ho > 0 else np.zeros((0, NCODES), dtype=np.float64))

    mu, V = fit_pca(tr_pmf, args.M)
    M = int(V.shape[0])
    if M < int(args.M):
        print("[pca_r3b] NOTE: only %d components exist for N=%d (asked for %d)"
              % (M, n, args.M), flush=True)
    a_tr = project(tr_pmf, mu, V)
    corr_tr = corr_rows(reconstruct(a_tr, mu, V), tr_pmf)
    print("[pca_r3b] N=%d M=%d S=%.4f  train projection corr mean=%.4f min=%.4f "
          "(exact when M >= N-1 = %d)" % (n, M, S, float(corr_tr.mean()), float(corr_tr.min()),
                                          n - 1), flush=True)

    # ---------------------------------------------------------------- held-out projection
    if n_ho > 0:
        a_ho = project(ho_pmf, mu, V)
        rec_ho = reconstruct(a_ho, mu, V)
        pc = corr_rows(rec_ho, ho_pmf)
        pc_clip = corr_rows(to_pmf(rec_ho)[0], ho_pmf)
        ref_mean = corr_rows(np.repeat(mu[None, :], n_ho, axis=0), ho_pmf)
        ref_near = corr_matrix(ho_pmf, tr_pmf).max(axis=1)
        print("[pca_r3b] held-out projection corr mean=%.4f min=%.4f (clipped %.4f) | "
              "ref: mean-crop %.4f, nearest-train %.4f"
              % (float(pc.mean()), float(pc.min()), float(pc_clip.mean()),
                 float(ref_mean.mean()), float(ref_near.mean())), flush=True)
    else:
        a_ho = np.zeros((0, M), dtype=np.float64)
        pc = pc_clip = ref_mean = ref_near = np.zeros(0, dtype=np.float64)

    # ---------------------------------------------------------------- generation
    gen_rng = np.random.default_rng(args.seed + 999)
    if args.prior == "resample":
        a_gen, spacing, jitter_std = sample_latents_resample(a_tr, int(args.n_gen), gen_rng,
                                                             jitter_scale=0.2)
        prior_info = {"kind": "resample-with-gaussian-jitter", "prior_mode": "resample",
                      "jitter_scale": 0.2, "median_nn_spacing": float(spacing),
                      "jitter_std": float(jitter_std)}
        print("[pca_r3b] prior: resample + jitter, median NN spacing=%.6g -> jitter std=%.6g"
              % (spacing, jitter_std), flush=True)
    else:
        cov_mode = "full" if n >= 100 else "diag"
        g_prior = fit_z_gaussian(a_tr, cov_mode)
        a_gen = sample_z_gaussian(g_prior, int(args.n_gen), gen_rng)
        prior_info = {"kind": "Gaussian on the training PCA coefficients", "prior_mode": "gauss",
                      "cov_mode": cov_mode, "M": M,
                      "coeff_std_mean": float(g_prior["std"].mean()),
                      "coeff_std_min": float(g_prior["std"].min()),
                      "coeff_std_max": float(g_prior["std"].max())}
        print("[pca_r3b] prior: %s-covariance Gaussian in R^%d, per-coord std %s"
              % (cov_mode, M, ["%.4g" % v for v in g_prior["std"]]), flush=True)

    raw = reconstruct(a_gen, mu, V)
    neg_frac = float((raw < 0).sum()) / float(max(raw.size, 1))
    neg_mass = float(np.clip(-raw, 0.0, None).sum(axis=1).mean())
    q_gen, n_degenerate = to_pmf(raw)
    if n_degenerate:
        print("[pca_r3b] WARNING: %d of %d draws reconstructed to a non-positive image and were "
              "replaced by the uniform pmf" % (n_degenerate, q_gen.shape[0]), flush=True)
    offsets = data_lens_r3.sample_offsets(d.train_offsets, q_gen.shape[0], gen_rng)
    for i in range(q_gen.shape[0]):
        crop = codec10.pmf_to_image(q_gen[i], S)
        np.save(os.path.join(args.out, "crops", "generated_%03d.npy" % i), crop.astype(np.float32))
        np.save(os.path.join(args.out, "generated_%03d.npy" % i),
                codec10.paste(crop, int(offsets[i, 0]), int(offsets[i, 1])).astype(np.float32))
    print("[pca_r3b] linear reconstruction was negative on %.2f%% of cells (mean clipped mass "
          "%.4g of 1.0) -- the clip is the control's only non-linearity"
          % (100.0 * neg_frac, neg_mass), flush=True)

    meta = {
        "arm": default_arm_name(args),
        "script": "controls_pca_r3.py", "round": "3b", "control": "pca",
        # shared mean + M components; the per-image coefficients are the latent
        "param_count": int(NCODES + M * NCODES + n * M),
        "param_count_shared": int(NCODES + M * NCODES),
        "M": M, "M_requested": int(args.M), "latent": "pixel-PCA", "protocol": "pca",
        "objective": None, "prior_arg": args.prior,
        "n_train": int(n), "n_gen": int(q_gen.shape[0]), "n_holdout": int(n_ho),
        "S": float(S), "seed": int(args.seed), "conditional": False,
        "train_corr_mean": float(corr_tr.mean()), "train_corr_min": float(corr_tr.min()),
        "train_corrs": corr_tr.tolist(),
        "projection_corr_mean": float(pc.mean()) if pc.size else None,
        "projection_corr_min": float(pc.min()) if pc.size else None,
        "projection_corr_median": float(np.median(pc)) if pc.size else None,
        "projection_corrs": pc.tolist(),
        "projection_corr_clipped_mean": float(pc_clip.mean()) if pc_clip.size else None,
        "projection_corr_clipped_min": float(pc_clip.min()) if pc_clip.size else None,
        "projection_restarts": 0, "projection_steps": 0,
        "projection_note": "closed-form orthogonal projection, no optimiser",
        "ref_projection_mean_crop_corr": float(ref_mean.mean()) if ref_mean.size else None,
        "ref_projection_nearest_train_corr": float(ref_near.mean()) if ref_near.size else None,
        "z_norm_mean": float(np.linalg.norm(a_tr, axis=1).mean()),
        "z_norm_max": float(np.linalg.norm(a_tr, axis=1).max()),
        "gen_negative_cell_fraction": neg_frac,
        "gen_clipped_mass_mean": neg_mass,
        "gen_degenerate_rows": int(n_degenerate),
        "prior": prior_info,
        "wallclock_seconds": float(time.time() - t0),
        "gen_offsets": [[int(a), int(b)] for a, b in offsets.tolist()],
    }
    with open(os.path.join(args.out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True, default=str)
    np.save(os.path.join(args.out, "pca_mean.npy"), mu)
    np.save(os.path.join(args.out, "pca_components.npy"), V)
    np.save(os.path.join(args.out, "pca_coeffs_train.npy"), a_tr)
    np.save(os.path.join(args.out, "pca_coeffs_holdout.npy"), a_ho)
    print("[pca_r3b] arm=%s N=%d wrote %d images to %s (%.1f s)"
          % (meta["arm"], n, q_gen.shape[0], args.out, time.time() - t0), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
