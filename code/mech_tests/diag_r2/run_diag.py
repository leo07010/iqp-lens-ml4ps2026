#!/usr/bin/env python3
"""Diagnose round-2 IQP-pmf-full-fixed arm haze (rms=26px vs true 9px, corr(mean)=0.00).

Distinguishes:
  (A) per-image fit failure  -- even the TRAINED reconstruction of each training image
      (theta_lat_i + theta_core, no resampling) is bad.
  (B) prior/sampling failure -- per-image fits are fine, but drawing latents from the
      fitted N(mu, diag sigma^2) lands off the training-latent manifold.

Reuses the project's own assembly/codec functions (IQPExact.probs/join, codec12.pmf_to_image,
lens_synth.fit_gaussian_latent/sample_gaussian_latents) -- does not
reimplement any of that logic. Read-only: does not touch any file under the repo.

CPU only, torch.set_num_threads(4). Run: python3 run_diag.py
"""
import json
import os
import sys

import numpy as np
import torch

torch.set_num_threads(4)

REPO = "<HOME>/gsoc2026-deeplense-qml/experiments/13_iqp_turbulence_replication"
sys.path.insert(0, REPO)
os.chdir(REPO)  # data_lens's on-disk cache is resolved relative-safe, but be tidy anyway

import codec12
import data_lens
import lens_synth
from iqp_exact import IQPExact

OUT_DIR = "/tmp/claude-34336/-home-<user>-gsoc2026-deeplense-qml/62d672af-a68e-4727-8a77-150b890871c0/scratchpad/diag_r2"

# Fixed-center rms radius, EXACTLY analysis_lens/analyze_lens.py:299-300,392-402's definition
# (rr = distance from the frame center (31.5, 31.5), not each image's own centroid) -- this is
# the metric the 26px-generated / 9px-true numbers in the task brief were computed with, so we
# match it here rather than controls_r3.moments' centroid-relative second moment (which reports
# a different, larger number for the same images because it also counts any centroid offset).
_YY, _XX = np.mgrid[0:64, 0:64]
_RR = np.sqrt((_XX - 31.5) ** 2 + (_YY - 31.5) ** 2)


def rms_radius(img):
    w = np.clip(np.asarray(img, dtype=np.float64), 0.0, None)
    tot = w.sum()
    if tot <= 0:
        return 0.0
    return float(np.sqrt((w * _RR * _RR).sum() / tot))

ARMS = {
    "n100": "outputs_lens_round2/no_sub/iqp_pmf_full_fixed_n100",
    "n11": "outputs_lens_round2/no_sub/iqp_pmf_full_fixed_n11",
}


def corr(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float((a @ b) / (na * nb))


def nn_dist_median(points):
    """Median nearest-neighbour Euclidean distance within `points` (m, d)."""
    from scipy.spatial import cKDTree  # available (numpy/scipy standard env)
    tree = cKDTree(points)
    d, _ = tree.query(points, k=2)  # k=1 is self (dist 0)
    return float(np.median(d[:, 1]))


def analyze_arm(tag, arm_rel, n_train):
    print("\n" + "=" * 90)
    print("ARM %s  (%s, n_train=%d)" % (tag, arm_rel, n_train))
    print("=" * 90)

    with open(os.path.join(arm_rel, "meta.json")) as f:
        meta = json.load(f)
    S = float(meta["intensity_scale_S"])
    sigmas = meta["sigmas"]
    n_qubits = int(meta["n_qubits"])
    K = int(meta["K"])
    d_lat = int(meta["d_lat"])

    theta_core = np.load(os.path.join(arm_rel, "theta_core.npy"))       # (4046,) float32
    lat_anchors = np.load(os.path.join(arm_rel, "theta_lat_anchors.npy"))  # (n_train,50) float64

    sim = IQPExact(n=n_qubits, K=K, device="cpu", dtype=torch.complex64)
    assert sim.n_params == theta_core.shape[0] + d_lat, (sim.n_params, theta_core.shape, d_lat)

    theta_core_t = torch.tensor(theta_core, dtype=sim.rdtype)

    snapshots, anchors, _ = data_lens.load_lens("no_sub", n_train=n_train, seed=0, n_holdout=0,
                                                encoding="pmf")
    fields = np.stack([snapshots[a].field for a in anchors], axis=0).astype(np.float64)  # (N,64,64)
    pmfs = [snapshots[a].pmf for a in anchors]

    # ---------------------------------------------------------- (1)/(3) per-image reconstruction
    recon_images = np.zeros((n_train, 64, 64), dtype=np.float64)
    rec_corr = np.zeros(n_train)
    rec_rms = np.zeros(n_train)
    rec_l2 = np.zeros(n_train)
    field_rms = np.zeros(n_train)
    for i in range(n_train):
        lat_t = torch.tensor(lat_anchors[i], dtype=sim.rdtype)
        theta_i = IQPExact.join(lat_t, theta_core_t)
        with torch.no_grad():
            q_i = sim.probs(theta_i).to(torch.float64).numpy()
        img_i = codec12.pmf_to_image(q_i, S).astype(np.float64)
        recon_images[i] = img_i
        rec_corr[i] = corr(img_i, fields[i])
        rec_rms[i] = rms_radius(img_i)
        field_rms[i] = rms_radius(fields[i])
        rec_l2[i] = float(np.linalg.norm(img_i - fields[i]))

    print("\n[1]/[3] per-image reconstruction (theta_lat_i TRAINED + theta_core, no resampling):")
    print("  corr(recon_i, field_i):  mean=%.4f  std=%.4f  min=%.4f  max=%.4f"
          % (rec_corr.mean(), rec_corr.std(), rec_corr.min(), rec_corr.max()))
    print("  rms radius(recon_i):     mean=%.3f  std=%.3f   | rms radius(field_i): mean=%.3f std=%.3f"
          % (rec_rms.mean(), rec_rms.std(), field_rms.mean(), field_rms.std()))
    print("  L2(recon_i, field_i):    mean=%.4f  std=%.4f" % (rec_l2.mean(), rec_l2.std()))
    print("  first 5: corr=%s" % np.array2string(rec_corr[:5], precision=4))
    print("           rms_recon=%s" % np.array2string(rec_rms[:5], precision=3))
    print("           rms_field=%s" % np.array2string(field_rms[:5], precision=3))

    # ---------------------------------------------------------- (2) Gaussian-latent sampling
    mu, sigma = lens_synth.fit_gaussian_latent(lat_anchors)
    # sanity: should match meta's latent_mu/sigma (same fit, same inputs)
    mu_meta = np.asarray(meta["latent_mu"])
    sig_meta = np.asarray(meta["latent_sigma"])
    print("\n  [gaussian fit check] max|mu-meta|=%.2e  max|sigma-meta|=%.2e"
          % (np.abs(mu - mu_meta).max(), np.abs(sigma - sig_meta).max()))

    rng = np.random.default_rng(0)
    n_draw = 200
    draws = lens_synth.sample_gaussian_latents(mu, sigma, n_draw, rng)  # (200,50)

    nn_train = nn_dist_median(lat_anchors)  # median NN distance among training latents
    dists_to_nearest_train = np.zeros(n_draw)
    for k in range(n_draw):
        d = np.linalg.norm(lat_anchors - draws[k][None, :], axis=1)
        dists_to_nearest_train[k] = d.min()
    dists_norm = dists_to_nearest_train / nn_train

    mean_field = fields.mean(axis=0)
    gauss_rms = np.zeros(n_draw)
    gauss_corr_mean = np.zeros(n_draw)
    for k in range(n_draw):
        lat_t = torch.tensor(draws[k], dtype=sim.rdtype)
        theta_k = IQPExact.join(lat_t, theta_core_t)
        with torch.no_grad():
            q_k = sim.probs(theta_k).to(torch.float64).numpy()
        img_k = codec12.pmf_to_image(q_k, S).astype(np.float64)
        gauss_rms[k] = rms_radius(img_k)
        gauss_corr_mean[k] = corr(img_k, mean_field)

    print("\n[2] Gaussian-latent sampling (mu,sigma fitted to the %d trained latents; %d fresh draws, seed 0):"
          % (n_train, n_draw))
    print("  median NN distance among TRAINING latents:      %.4f" % nn_train)
    print("  dist(draw, nearest train latent):  mean=%.4f std=%.4f  (x%.2f the training NN scale)"
          % (dists_to_nearest_train.mean(), dists_to_nearest_train.std(), dists_norm.mean()))
    print("  fraction of draws farther than the training-set NN scale: %.1f%%"
          % (100.0 * np.mean(dists_norm > 1.0)))
    print("  rms radius of gaussian-sampled images: mean=%.3f std=%.3f  (cf. mean training rms=%.3f)"
          % (gauss_rms.mean(), gauss_rms.std(), field_rms.mean()))
    print("  corr(gaussian-sampled image, mean training field): mean=%.4f std=%.4f"
          % (gauss_corr_mean.mean(), gauss_corr_mean.std()))

    # cross-check against the arm's own written generated_*.npy (same synth recipe, different seed)
    gen_files = sorted(f for f in os.listdir(arm_rel) if f.startswith("generated_") and f.endswith(".npy"))
    gen_rms = []
    gen_corr_mean = []
    for gf in gen_files[:50]:
        g = np.load(os.path.join(arm_rel, gf)).astype(np.float64)
        r = rms_radius(g)
        gen_rms.append(r)
        gen_corr_mean.append(corr(g, mean_field))
    gen_rms = np.asarray(gen_rms)
    gen_corr_mean = np.asarray(gen_corr_mean)
    print("  [cross-check vs on-disk generated_*.npy, first %d]: rms mean=%.3f std=%.3f | corr(mean field) mean=%.4f"
          % (len(gen_rms), gen_rms.mean(), gen_rms.std(), gen_corr_mean.mean()))

    # ---------------------------------------------------------- (4) latent interpolation
    rng2 = np.random.default_rng(1)
    n_pairs = min(20, n_train * (n_train - 1) // 2)
    idx_pairs = set()
    while len(idx_pairs) < n_pairs:
        a, b = rng2.integers(0, n_train, size=2)
        if a != b:
            idx_pairs.add((min(a, b), max(a, b)))
    idx_pairs = list(idx_pairs)

    mid_rms = np.zeros(len(idx_pairs))
    mid_corr_avg = np.zeros(len(idx_pairs))
    for m, (a, b) in enumerate(idx_pairs):
        lat_mid = 0.5 * (lat_anchors[a] + lat_anchors[b])
        lat_t = torch.tensor(lat_mid, dtype=sim.rdtype)
        theta_m = IQPExact.join(lat_t, theta_core_t)
        with torch.no_grad():
            q_m = sim.probs(theta_m).to(torch.float64).numpy()
        img_m = codec12.pmf_to_image(q_m, S).astype(np.float64)
        mid_rms[m] = rms_radius(img_m)
        avg_endpoint = 0.5 * (recon_images[a] + recon_images[b])
        mid_corr_avg[m] = corr(img_m, avg_endpoint)

    print("\n[4] latent midpoint interpolation (%d random training-latent pairs):" % len(idx_pairs))
    print("  rms radius(midpoint image): mean=%.3f std=%.3f  (cf. mean training rms=%.3f)"
          % (mid_rms.mean(), mid_rms.std(), field_rms.mean()))
    print("  corr(midpoint image, avg of the two endpoint reconstructions): mean=%.4f std=%.4f"
          % (mid_corr_avg.mean(), mid_corr_avg.std()))

    # ---------------------------------------------------------- (5) loss floors
    lc = np.load(os.path.join(arm_rel, "loss_curves.npz"))
    stage1_final = float(lc["stage1"][-1])
    stage2_final = lc["stage2"][:, -1]
    print("\n[5] loss:")
    print("  stage1 final MMD^2 (image 1, 4096 free params): %.3e" % stage1_final)
    print("  stage2 final MMD^2 (per-image, 50 free params): mean=%.3e median=%.3e min=%.3e max=%.3e"
          % (stage2_final.mean(), np.median(stage2_final), stage2_final.min(), stage2_final.max()))

    weights = sim.kernel_weights(sigmas)
    uniform_pmf = np.full(1 << n_qubits, 1.0 / (1 << n_qubits), dtype=np.float64)
    ez_uniform = sim.data_expvals(uniform_pmf)
    mmd_uniform = []
    mmd_sampling_floor = []
    for i, a in enumerate(anchors):
        ez_data_i = sim.data_expvals(pmfs[i])
        d_u = ez_uniform - ez_data_i
        mmd_uniform.append(float((weights * d_u * d_u).sum()))
        # sampling-noise floor: the arm's own 5e4-code Monte-Carlo draw from this SAME exact
        # pmf, scored against the exact pmf -- the MMD^2 a perfect model would still show
        # if it were compared through finite sampling instead of exactly.
        pool_pmf_i = codec12.codes_to_pmf(snapshots[a].pool_codes)
        ez_pool_i = sim.data_expvals(pool_pmf_i)
        d_s = ez_pool_i - ez_data_i
        mmd_sampling_floor.append(float((weights * d_s * d_s).sum()))
    mmd_uniform = np.asarray(mmd_uniform)
    mmd_sampling_floor = np.asarray(mmd_sampling_floor)
    print("  MMD^2(uniform q, data pmf_i):           mean=%.3e  (upper-bound scale, %d anchors)"
          % (mmd_uniform.mean(), n_train))
    print("  MMD^2(5e4-sample draw of pmf_i, pmf_i): mean=%.3e  (finite-sampling noise floor)"
          % (mmd_sampling_floor.mean(),))

    return {
        "tag": tag, "arm_rel": arm_rel, "n_train": n_train,
        "fields": fields, "recon_images": recon_images, "field_rms": field_rms, "rec_rms": rec_rms,
        "rec_corr": rec_corr, "gen_files": gen_files, "arm_dir": arm_rel,
    }


def make_figure(result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_show = 6
    idx = np.linspace(0, result["n_train"] - 1, n_show).astype(int)
    fields = result["fields"]
    recons = result["recon_images"]
    gen_imgs = []
    for gf in result["gen_files"][:n_show]:
        gen_imgs.append(np.load(os.path.join(result["arm_dir"], gf)).astype(np.float64))

    panels = [fields[i] for i in idx] + [recons[i] for i in idx] + gen_imgs
    vmax = max(float(p.max()) for p in panels)
    panels_norm = [p / vmax for p in panels]

    fig, axes = plt.subplots(3, n_show, figsize=(2.0 * n_show, 6.2))
    row_labels = ["training field_i", "reconstruction\n(trained theta_lat_i)", "gaussian-sampled\ngenerated_*"]
    for r in range(3):
        for c in range(n_show):
            ax = axes[r, c]
            ax.imshow(panels_norm[r * n_show + c], vmin=0, vmax=1, cmap="viridis")
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(row_labels[r], fontsize=9)
            if r == 0:
                ax.set_title("anchor %d" % (idx[c] + 1), fontsize=9)
    fig.suptitle("%s: training vs. trained-latent reconstruction vs. gaussian-sampled generation "
                 "(common scale, vmax=%.1f)" % (result["tag"], vmax), fontsize=10)
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "trainfit_%s.png" % result["tag"])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print("\n[6] figure saved: %s" % out_path)


if __name__ == "__main__":
    results = {}
    for tag, arm_rel in ARMS.items():
        n_train = int(tag[1:])
        results[tag] = analyze_arm(tag, arm_rel, n_train)

    for tag, res in results.items():
        make_figure(res)

    print("\nDONE.")
