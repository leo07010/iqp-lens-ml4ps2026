#!/usr/bin/env python3
"""Round-3 zero-learning controls (docs/IQP_LENS_ROUND3_PREREGISTRATION.md).

    --arm copy       the N training images, written unchanged. The memorisation ceiling:
                     any arm whose NN-train is at this level is copying its training set.
                     NOTE it is the ONLY arm that does not pass through the crop/paste
                     pipeline, so it keeps the ~0.6% of light that falls outside the 32x32
                     window. Left literal because the pre-registration says "the N training
                     images"; evaluate_r3.py prints the resulting flux offset.
    --arm mean       the mean centred training crop, pasted at sampled offsets. The "no
                     variation at all" floor the decision rule requires the IQP arms to beat.
    --arm ring       the parametric five-number baseline that learns nothing: the mean centred
                     RADIAL profile of the training crops, stretched to a (flux, rms radius,
                     e1, e2) quadruple bootstrapped JOINTLY from the training images, pasted
                     at sampled offsets. If this beats the IQP arms on the physics score then
                     "IQP makes plausible lens images" means only "better than the mean image"
                     (prediction P13).
    --arm interp_e1  (addendum) the conditional control for the pseudo-time arm: for each
                     held-out image's e1, a linear interpolation of the two training crops
                     that bracket it in e1 (nearest single crop outside the range). Written in
                     held-out order and tagged conditional=true, so evaluate_r3.py scores it
                     with the paired metrics.

Ring geometry (the docstring is the specification; test_r3.py measures the error it leaves)
------------------------------------------------------------------------------------------
A circular profile f(r) rendered through a linear map T, I(v) = f(||T^-1 v||), has second
moments Q = q0 * T T^T (q0 = the circular profile's own second moment). Choosing

    E = [[e1, e2], [e2, -e1]]      (in the (x, y) basis)
    T = sqrt(S) * (I + E)^(1/2)    (symmetric square root; needs e1^2 + e2^2 < 1)

gives T T^T = S (I + E), hence

    (Qxx - Qyy) / (Qxx + Qyy) = e1        2 Qxy / (Qxx + Qyy) = e2
    rms radius sqrt(Qxx + Qyy)            = R0 * sqrt(S)

so S = (R / R0)^2 hits a target rms radius R, where R0 is the rms radius of the UNSTRETCHED
template measured on the same 32x32 grid (measured, not derived, so the grid's own
discretisation and window truncation are already inside R0). The rendered field is finally
rescaled to the target total flux.

The 32x32 window truncates the profile, and the truncation is anisotropic once the field is
stretched, so the (e1, e2) measured back from the render is not exactly the (e1, e2) asked
for. test_r3.py measures that residual and prints it instead of assuming it away.

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

CROP = codec10.GRID


# --------------------------------------------------------------------------------------
# moments (evaluate_r3.py owns the canonical copy; duplicated here only for the ring's own
# calibration so controls_r3.py can be run without importing the evaluator)
# --------------------------------------------------------------------------------------
def moments(field):
    """(flux, rms_radius, e1, e2) from the intensity second moments of a 2-D field."""
    w = np.clip(np.asarray(field, dtype=np.float64), 0.0, None)
    tot = float(w.sum())
    if not (tot > 0):
        return 0.0, 0.0, 0.0, 0.0
    ny, nx = w.shape
    ys = np.arange(ny, dtype=np.float64)[:, None]
    xs = np.arange(nx, dtype=np.float64)[None, :]
    cy = float((w * ys).sum() / tot)
    cx = float((w * xs).sum() / tot)
    dy = ys - cy
    dx = xs - cx
    qxx = float((w * dx * dx).sum() / tot)
    qyy = float((w * dy * dy).sum() / tot)
    qxy = float((w * dx * dy).sum() / tot)
    den = qxx + qyy
    if den <= 0:
        return tot, 0.0, 0.0, 0.0
    return tot, float(np.sqrt(den)), float((qxx - qyy) / den), float(2.0 * qxy / den)


def crop_moments(crops):
    """(m, 4) array of (flux, rms_radius, e1, e2) for a stack of crops."""
    return np.asarray([moments(crops[i]) for i in range(crops.shape[0])], dtype=np.float64)


# --------------------------------------------------------------------------------------
# ring template
# --------------------------------------------------------------------------------------
def mean_radial_profile(crops, n_bins=24):
    """(radii, profile): the training crops' mean radial profile, each crop measured from its
    OWN centroid and normalised to unit flux first (so a bright crop does not dominate)."""
    crops = np.asarray(crops, dtype=np.float64)
    ny, nx = crops.shape[1], crops.shape[2]
    ys = np.arange(ny, dtype=np.float64)[:, None]
    xs = np.arange(nx, dtype=np.float64)[None, :]
    edges = np.arange(n_bins + 1, dtype=np.float64)          # bins [0,1), [1,2), ...
    acc = np.zeros(n_bins, dtype=np.float64)
    for i in range(crops.shape[0]):
        w = np.clip(crops[i], 0.0, None)
        tot = float(w.sum())
        if not (tot > 0):
            continue
        cy = float((w * ys).sum() / tot)
        cx = float((w * xs).sum() / tot)
        r = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2).ravel()
        v = (w / tot).ravel()
        num, _ = np.histogram(r, bins=edges, weights=v)
        cnt, _ = np.histogram(r, bins=edges)
        with np.errstate(invalid="ignore", divide="ignore"):
            prof = np.where(cnt > 0, num / np.maximum(cnt, 1), 0.0)
        acc += prof
    acc /= max(crops.shape[0], 1)
    radii = 0.5 * (edges[:-1] + edges[1:])
    return radii, acc


def render_ring(radii, profile, flux, radius, e1, e2, R0, size=CROP):
    """Render the stretched template into a `size`x`size` crop with total flux `flux`."""
    e_mag = float(np.hypot(e1, e2))
    if e_mag >= 0.999:
        scale_e = 0.999 / e_mag
        e1, e2 = float(e1) * scale_e, float(e2) * scale_e
    M = np.array([[1.0 + float(e1), float(e2)], [float(e2), 1.0 - float(e1)]], dtype=np.float64)
    evals, evecs = np.linalg.eigh(M)
    evals = np.maximum(evals, 1e-9)
    S = (float(radius) / float(R0)) ** 2 if R0 > 0 else 1.0
    S = max(S, 1e-9)
    T = np.sqrt(S) * (evecs @ np.diag(np.sqrt(evals)) @ evecs.T)
    Tinv = np.linalg.inv(T)

    c = 0.5 * (size - 1)
    ys = np.arange(size, dtype=np.float64)[:, None] - c
    xs = np.arange(size, dtype=np.float64)[None, :] - c
    vx = Tinv[0, 0] * xs + Tinv[0, 1] * ys
    vy = Tinv[1, 0] * xs + Tinv[1, 1] * ys
    r_ell = np.sqrt(vx * vx + vy * vy)
    img = np.interp(r_ell, radii, profile, left=float(profile[0]), right=0.0)
    img = np.clip(img, 0.0, None)
    tot = float(img.sum())
    if tot > 0:
        img = img * (float(flux) / tot)
    return img.astype(np.float32)


def template_R0(radii, profile, size=CROP):
    """rms radius of the UNSTRETCHED template on the same grid -- measured, not derived."""
    tmpl = render_ring(radii, profile, 1.0, 1.0, 0.0, 0.0, R0=1.0, size=size)
    _f, r0, _e1, _e2 = moments(tmpl)
    return float(r0)


# --------------------------------------------------------------------------------------
# e1 interpolation control
# --------------------------------------------------------------------------------------
def interp_crops_by_e1(train_crops, train_e1, targets):
    """(m, 32, 32) crops: for each target e1, the linear blend of the two training crops that
    bracket it in e1 (weights by e1 distance), clamped to the nearest crop outside the range.

    Exactness at an anchor: if a target equals a training e1 the blend weight is 0 or 1, so the
    output IS that training crop (test_r3.py asserts it byte-for-byte).
    """
    e = np.asarray(train_e1, dtype=np.float64)
    order = np.argsort(e)
    e_s = e[order]
    c_s = np.asarray(train_crops, dtype=np.float64)[order]
    t = np.asarray(targets, dtype=np.float64)
    out = np.empty((t.size, c_s.shape[1], c_s.shape[2]), dtype=np.float32)
    for i, tv in enumerate(t):
        j = int(np.searchsorted(e_s, tv, side="left"))
        if j <= 0:
            out[i] = c_s[0]
            continue
        if j >= e_s.size:
            out[i] = c_s[-1]
            continue
        lo, hi = j - 1, j
        span = e_s[hi] - e_s[lo]
        w = 0.0 if span <= 0 else float((tv - e_s[lo]) / span)
        w = float(np.clip(w, 0.0, 1.0))
        out[i] = ((1.0 - w) * c_s[lo] + w * c_s[hi]).astype(np.float32)
    return out


# --------------------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Round-3 zero-learning controls.")
    p.add_argument("--arm", choices=("copy", "mean", "ring", "interp_e1"), required=True)
    p.add_argument("--n-train", type=int, default=11)
    p.add_argument("--n-holdout", type=int, default=data_lens_r3.N_HOLDOUT)
    p.add_argument("--n-gen", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--arm-name", default=None)
    p.add_argument("--n-bins", type=int, default=24)
    return p.parse_args(argv)


ARM_LABEL = {"copy": "CTRL-copy", "mean": "CTRL-mean", "ring": "CTRL-ring",
             "interp_e1": "CTRL-interp-e1"}


def main(argv=None):
    args = parse_args(argv)
    t0 = time.time()
    rng = np.random.default_rng(args.seed + 4242)
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "crops"), exist_ok=True)

    d = data_lens_r3.load_r3(args.n_train, seed=args.seed, n_holdout=args.n_holdout)
    n = d.n_train
    extra = {}
    conditional = False

    if args.arm == "copy":
        frames = np.asarray(d.train_images, dtype=np.float32)
        crops = np.asarray(d.train_crops, dtype=np.float32)
        offsets = np.asarray(d.train_offsets, dtype=np.int64)
        extra["note"] = ("raw training images, NOT re-pasted -- the only arm that keeps the "
                         "light outside the 32x32 window")
    elif args.arm == "mean":
        m = np.asarray(d.train_crops, dtype=np.float64).mean(axis=0).astype(np.float32)
        crops = np.repeat(m[None, :, :], int(args.n_gen), axis=0)
        offsets = data_lens_r3.sample_offsets(d.train_offsets, int(args.n_gen), rng)
        frames = np.stack([codec10.paste(crops[i], offsets[i, 0], offsets[i, 1])
                           for i in range(crops.shape[0])], axis=0)
        extra["mean_crop_total"] = float(m.sum())
    elif args.arm == "ring":
        radii, profile = mean_radial_profile(d.train_crops, n_bins=args.n_bins)
        R0 = template_R0(radii, profile)
        quads = crop_moments(d.train_crops)                     # (N, 4)
        pick = rng.integers(0, quads.shape[0], size=int(args.n_gen))
        drawn = quads[pick]                                     # joint bootstrap
        crops = np.stack([render_ring(radii, profile, drawn[i, 0], drawn[i, 1],
                                      drawn[i, 2], drawn[i, 3], R0)
                          for i in range(drawn.shape[0])], axis=0)
        offsets = data_lens_r3.sample_offsets(d.train_offsets, int(args.n_gen), rng)
        frames = np.stack([codec10.paste(crops[i], offsets[i, 0], offsets[i, 1])
                           for i in range(crops.shape[0])], axis=0)
        back = crop_moments(crops)
        extra.update({
            "template_R0": float(R0), "n_bins": int(args.n_bins),
            "bootstrap_source": "the N training crops' (flux, rms radius, e1, e2), drawn jointly",
            "render_err_e1_mean_abs": float(np.abs(back[:, 2] - drawn[:, 2]).mean()),
            "render_err_e2_mean_abs": float(np.abs(back[:, 3] - drawn[:, 3]).mean()),
            "render_err_radius_mean_abs": float(np.abs(back[:, 1] - drawn[:, 1]).mean()),
            "render_err_flux_mean_abs": float(np.abs(back[:, 0] - drawn[:, 0]).mean()),
        })
        print("[ctrl_r3] ring: R0=%.4f  |de1|=%.4f |de2|=%.4f |dR|=%.4f px |dF|=%.3g"
              % (R0, extra["render_err_e1_mean_abs"], extra["render_err_e2_mean_abs"],
                 extra["render_err_radius_mean_abs"], extra["render_err_flux_mean_abs"]))
    else:  # interp_e1
        conditional = True
        tr_q = crop_moments(d.train_crops)
        ho_q = crop_moments(d.holdout_crops)
        targets = ho_q[: int(args.n_gen), 2]
        crops = interp_crops_by_e1(d.train_crops, tr_q[:, 2], targets)
        offsets = data_lens_r3.sample_offsets(d.train_offsets, crops.shape[0], rng)
        frames = np.stack([codec10.paste(crops[i], offsets[i, 0], offsets[i, 1])
                           for i in range(crops.shape[0])], axis=0)
        extra.update({
            "e1_train_min": float(tr_q[:, 2].min()), "e1_train_max": float(tr_q[:, 2].max()),
            "e1_target_min": float(targets.min()), "e1_target_max": float(targets.max()),
            "n_targets_clamped": int(np.sum((targets < tr_q[:, 2].min()) |
                                            (targets > tr_q[:, 2].max()))),
        })
        print("[ctrl_r3] interp_e1: %d targets, %d outside the training e1 range (clamped)"
              % (targets.size, extra["n_targets_clamped"]))

    for i in range(frames.shape[0]):
        np.save(os.path.join(args.out, "generated_%03d.npy" % i), frames[i].astype(np.float32))
        np.save(os.path.join(args.out, "crops", "generated_%03d.npy" % i), crops[i].astype(np.float32))

    meta = {
        "arm": args.arm_name if args.arm_name else ARM_LABEL[args.arm],
        "script": "controls_r3.py", "round": 3, "control": args.arm,
        "param_count": 0,
        "n_train": int(n), "n_gen": int(frames.shape[0]), "n_holdout": int(args.n_holdout),
        "S": float(d.S), "seed": int(args.seed), "conditional": bool(conditional),
        "gen_offsets": [[int(a), int(b)] for a, b in np.asarray(offsets).tolist()],
        "wallclock_seconds": float(time.time() - t0),
    }
    meta.update(extra)
    with open(os.path.join(args.out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True, default=str)
    print("[ctrl_r3] arm=%s N=%d wrote %d images to %s (%.1f s)"
          % (meta["arm"], n, frames.shape[0], args.out, time.time() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
