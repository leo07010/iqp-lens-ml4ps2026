#!/usr/bin/env python3
"""Shared `--synth-mode gaussian-latent` machinery for the lens pilot.

The lens data (docs/IQP_LENS_PILOT_PREREGISTRATION.md) has NO time axis, so Algorithm 1's
Eq. 31 latent interpolation has nothing to interpolate along. Instead, after training the
core on image 1 and warm-starting a per-image latent through images 2..N (unchanged
Algorithm 1 training), the N trained latents are treated as N samples of a latent
distribution: fit N(mu, diag sigma^2) coordinate-wise, draw --n-gen latents from it, and
generate one image per latent.

Three arms use this identically (train_iqp.py, baseline_rbm.py, controls.py's marginal
arm), so the fitting, the dual mean/last-write aggregation, and the output naming live
here rather than being triplicated.

Aggregation: `codec.aggregate` (MEAN of every tuple landing in a cell) is the primary
output; `agg_variants.aggregate_last` (LAST-WRITE-WINS) is written in parallel to
`<out>_last/` because on D2 that convention moved PDF-JS by an order of magnitude
(agg_variants.py docstring) -- both are produced so the choice is not a hidden lever.

Output naming: `generated_{i:03d}.npy` (no time axis), i.e. evaluate.py's
"unconditional" contract, compared distributionally against the held-out set.

Python 3.9 compatible (no `X | None`).
"""
import json
import os

import numpy as np

import agg_variants
import codec


def last_dir(out_dir):
    """The sibling directory holding the LAST-WRITE-WINS aggregation of the same samples."""
    return os.path.normpath(out_dir) + "_last"


def fit_gaussian_latent(latents):
    """(mu, sigma) for a diagonal Gaussian fitted to `latents` (n_anchors, d_lat).

    sigma is the per-coordinate population std (ddof=0) of the N trained latents. With
    N as small as 11 this is a deliberately crude fit -- it is the pre-registered
    synthesis rule, not a tuned choice.
    """
    lat = np.asarray(latents, dtype=np.float64)
    if lat.ndim != 2:
        raise ValueError("latents must be (n_anchors, d_lat), got %s" % (lat.shape,))
    return lat.mean(axis=0), lat.std(axis=0)


def sample_gaussian_latents(mu, sigma, n, rng):
    """(n, d_lat) draws from N(mu, diag sigma^2)."""
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    return rng.normal(loc=mu[None, :], scale=sigma[None, :], size=(int(n), mu.shape[0]))


def latent_stats(mu, sigma):
    """JSON-safe summary of the fitted latent Gaussian for meta.json."""
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    return {
        "d_lat": int(mu.shape[0]),
        "latent_mu": [float(v) for v in mu.tolist()],
        "latent_sigma": [float(v) for v in sigma.tolist()],
        "latent_mu_absmean": float(np.abs(mu).mean()),
        "latent_sigma_mean": float(sigma.mean()),
        "latent_sigma_min": float(sigma.min()),
        "latent_sigma_max": float(sigma.max()),
    }


def write_pair(out_dir, index, codes, a_z, b_z, sqrt=False):
    """Aggregate one image's `codes` BOTH ways and write them.

    mean aggregation -> <out_dir>/generated_{index:03d}.npy
    last-write-wins  -> <out_dir>_last/generated_{index:03d}.npy

    `sqrt`: the secondary sqrt arm generates in sqrt-intensity space; squaring here puts
    the written field back into INTENSITY space so it is comparable to the same ground
    truth every other arm is scored against (data_lens.invert_intensity).
    """
    codes = np.asarray(codes, dtype=np.int64)
    f_mean = codec.aggregate(codes, a_z, b_z)
    f_last = agg_variants.aggregate_last(codes, a_z, b_z)
    if sqrt:
        import data_lens
        f_mean = data_lens.invert_intensity(f_mean, sqrt=True)
        f_last = data_lens.invert_intensity(f_last, sqrt=True)
    d_last = last_dir(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(d_last, exist_ok=True)
    np.save(os.path.join(out_dir, "generated_{:03d}.npy".format(int(index))), f_mean.astype(np.float32))
    np.save(os.path.join(d_last, "generated_{:03d}.npy".format(int(index))), f_last.astype(np.float32))


def write_metas(out_dir, meta):
    """Write meta.json into BOTH the mean and the _last directory, tagging each with its
    aggregation convention and giving the _last arm its own distinguishable `arm` label
    (evaluate_lens.py groups rows by that label)."""
    d_last = last_dir(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(d_last, exist_ok=True)

    m_mean = dict(meta)
    m_mean["aggregation"] = "mean"
    with open(os.path.join(out_dir, "meta.json"), "w") as fh:
        json.dump(m_mean, fh, indent=2, sort_keys=True, default=str)

    m_last = dict(meta)
    m_last["aggregation"] = "last"
    m_last["arm"] = "%s-last" % meta.get("arm", os.path.basename(os.path.normpath(out_dir)))
    with open(os.path.join(d_last, "meta.json"), "w") as fh:
        json.dump(m_last, fh, indent=2, sort_keys=True, default=str)
    return m_mean, m_last


# ======================================================================================
# round-2 pmf encoding (docs/IQP_LENS_PILOT_PREREGISTRATION.md Addendum 2026-09-06)
# ======================================================================================
def sampled_dir(out_dir):
    """Sibling directory holding the SAMPLED counting-density reconstruction of the same
    circuit -- the control for the exact one (`<out>` holds the exact q * S)."""
    return os.path.normpath(out_dir) + "_sampled"


def write_pmf_pair(out_dir, index, q, codes, scale, sqrt=False):
    """Write one pmf-arm image BOTH ways.

    exact   -> <out_dir>/generated_{index:03d}.npy        = codec12.pmf_to_image(q, S)
    sampled -> <out_dir>_sampled/generated_{index:03d}.npy = counting density of `codes`

    The exact one is the PRIMARY output (the Addendum's "exact q(kx, ky) x S"); the sampled
    one is written next to it so "does the 1e5-sample Monte-Carlo step cost anything?" is a
    row in the table instead of an unexamined choice -- the same reason the tuple arms write
    `<out>_last/`. `codes` may be None to skip the sampled directory.
    """
    import codec12

    f_exact = codec12.pmf_to_image(np.asarray(q, dtype=np.float64), float(scale))
    if sqrt:
        import data_lens
        f_exact = data_lens.invert_intensity(f_exact, sqrt=True)
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "generated_{:03d}.npy".format(int(index))), f_exact.astype(np.float32))

    if codes is None:
        return
    f_samp = codec12.codes_to_image(np.asarray(codes, dtype=np.int64), float(scale))
    if sqrt:
        import data_lens
        f_samp = data_lens.invert_intensity(f_samp, sqrt=True)
    d_samp = sampled_dir(out_dir)
    os.makedirs(d_samp, exist_ok=True)
    np.save(os.path.join(d_samp, "generated_{:03d}.npy".format(int(index))), f_samp.astype(np.float32))


def write_metas_pmf(out_dir, meta):
    """meta.json into BOTH the exact and the _sampled directory of a pmf arm, tagged with
    `reconstruction` and giving the sampled one its own `arm` label (evaluate_lens.py groups
    rows by that label). The pmf analogue of write_metas()."""
    d_samp = sampled_dir(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(d_samp, exist_ok=True)

    m_exact = dict(meta)
    m_exact["reconstruction"] = "exact"
    with open(os.path.join(out_dir, "meta.json"), "w") as fh:
        json.dump(m_exact, fh, indent=2, sort_keys=True, default=str)

    m_samp = dict(meta)
    m_samp["reconstruction"] = "sampled"
    m_samp["arm"] = "%s-sampled" % meta.get("arm", os.path.basename(os.path.normpath(out_dir)))
    with open(os.path.join(d_samp, "meta.json"), "w") as fh:
        json.dump(m_samp, fh, indent=2, sort_keys=True, default=str)
    return m_exact, m_samp
