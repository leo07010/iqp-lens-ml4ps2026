#!/usr/bin/env python3
"""Round-3 evaluation: physical plausibility of generated lens images.

docs/IQP_LENS_ROUND3_PREREGISTRATION.md, "Metrics". Everything here is scored against the SAME
200 held-out images (data_lens_r3), and every arm writes the same two things -- a 64x64 pasted
frame and the 32x32 crop it was pasted from -- so no arm is scored on its own terms.

Distributional block (generated set vs held-out set, on the 64x64 frames)
------------------------------------------------------------------------
Five physical quantities per image, from the intensity moments (weights w = I / sum I):

    centroid offset  (dx, dy) = (x_bar, y_bar) - 31.5
    total flux       sum I
    rms radius       sqrt(Qxx + Qyy)
    ellipticity      e1 = (Qxx - Qyy)/(Qxx + Qyy),  e2 = 2 Qxy / (Qxx + Qyy)

For each the 1-D Wasserstein-1 distance to the held-out distribution DIVIDED BY the held-out
standard deviation of that quantity, so the five are commensurable; physics score = the mean
of the five. `W1(offset)` is one column because the pre-registration counts the offset as ONE
of the five quantities: it is the mean of the normalised W1 of dx and of dy.

W1 itself is the mean absolute difference of the two SORTED samples when they have equal
length (the pre-registered definition). CTRL-copy has only N samples, so for unequal lengths
the same quantity is computed from interpolated quantiles on a common grid -- the continuous
definition W1 = int_0^1 |F_a^-1 - F_b^-1| that the sorted form is the equal-n special case of.

Also: centred radial-profile RMSE (each image's profile taken about its OWN centroid, averaged
over the set, compared with the held-out mean profile), and -- for continuity with rounds 1-2
-- PDF-JS, Conv2D-MMD^2 (metrics.py, seed-0 encoder, one feature space for every row) and the
two nearest-neighbour L2 diagnostics (evaluate_lens.nn_l2: NN-train = memorisation, NN-holdout
= coverage).

Intensity-aligned secondary block (addendum, 2026-09-07, scoring note)
------------------------------------------------------------------------
The round-2 diagnostic found PDF-JS / Conv2D-MMD^2 / NN-* dominated by each arm's own mean
total intensity rather than by spatial structure (docs/IQP_LENS_ROUND3_PREREGISTRATION.md,
"Scoring note"). `align_to_target` rescales EVERY image it is given, by its own per-image
scalar, so that image's own total intensity equals a fixed target -- here the held-out set's
mean total. It is applied to every array on every side of these four numbers (generated,
training and held-out alike, all rescaled to the SAME target), so the residual distance is
scale-free on both sides, not just generator-side: `pdf_js_aligned`, `conv_mmd2_aligned`,
`nn_train_aligned`, `nn_holdout_aligned`, printed as a second table and added to every jsonl
row. This is why CTRL-copy's NN-train stays exactly 0 after alignment too -- its generated
images ARE (verbatim) the training images, and applying the identical per-image rescaling to
identical arrays leaves them identical. The physics score is untouched: flux is already a
separate quantity from radius/e1/e2 there, and the latter three are scale-invariant by
construction (moments computed after dividing by each image's own total).

Paired block (addendum; conditional arms only, on the 32x32 crops)
-----------------------------------------------------------------
An arm whose meta.json says conditional=true wrote its i-th sample FOR held-out image i. For
those, |e1_gen - e1_target|, |Delta e2|, |Delta flux|, |Delta radius| (means over the pairs)
and the paired L2 between the generated crop and the target's centred crop. A
`RANDOM-PAIRING (ref)` row applies the same five numbers to an unconditional arm's samples
paired with held-out images at random -- the floor any real conditioning has to beat.

Reference rows
--------------
`HOLDOUT split-half`: the held-out set's first 100 against its last 100, scored by every
distributional metric. That is the floor of the whole table: no generator can do better than
two halves of the real data do against each other. NN-train / NN-holdout for the held-out set
itself are reported the same way round 2 did.

Python 3.9 compatible (no `X | None`).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np

import data_lens_r3
from evaluate_lens import nn_l2
from metrics import conv_mmd2, pdf_js

CONV_SEED = 0
N_PROFILE_BINS = 24
_GEN_RE = re.compile(r"generated_(\d+)\.npy$")
PHYS_KEYS = ("offset", "flux", "radius", "e1", "e2")


# --------------------------------------------------------------------------------------
# physical quantities
# --------------------------------------------------------------------------------------
def image_quantities(field, centre=None):
    """dict(dx, dy, flux, radius, e1, e2) from the intensity moments of a 2-D field.

    `centre` is the reference point the centroid offset is measured from; default = the
    geometric centre (n-1)/2 of the array. Padding a crop with zeros (codec10.paste) changes
    only dx/dy -- flux, radius, e1 and e2 are centroid-relative and identical on the crop and
    on the pasted frame, which is why the same function serves both blocks.
    """
    w = np.clip(np.asarray(field, dtype=np.float64), 0.0, None)
    tot = float(w.sum())
    ny, nx = w.shape
    cy0, cx0 = ((0.5 * (ny - 1), 0.5 * (nx - 1)) if centre is None else centre)
    if not (tot > 0):
        return {"dx": 0.0, "dy": 0.0, "flux": 0.0, "radius": 0.0, "e1": 0.0, "e2": 0.0}
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
    return {"dx": cx - cx0, "dy": cy - cy0, "flux": tot,
            "radius": float(np.sqrt(max(den, 0.0))),
            "e1": float((qxx - qyy) / den) if den > 0 else 0.0,
            "e2": float(2.0 * qxy / den) if den > 0 else 0.0}


def quantity_table(images):
    """(m, 6) array of dx, dy, flux, radius, e1, e2 for a stack of fields."""
    rows = [image_quantities(images[i]) for i in range(images.shape[0])]
    return np.asarray([[r["dx"], r["dy"], r["flux"], r["radius"], r["e1"], r["e2"]]
                       for r in rows], dtype=np.float64)


QCOL = {"dx": 0, "dy": 1, "flux": 2, "radius": 3, "e1": 4, "e2": 5}


def wasserstein1(a, b, n_q=2000):
    """1-D Wasserstein-1. Equal lengths -> mean |sorted(a) - sorted(b)| (the pre-registered
    form); unequal -> the same integral from interpolated quantiles."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size == 0 or b.size == 0:
        return float("nan")
    if a.size == b.size:
        return float(np.abs(np.sort(a) - np.sort(b)).mean())
    qs = (np.arange(int(n_q), dtype=np.float64) + 0.5) / float(n_q)
    return float(np.abs(np.quantile(a, qs) - np.quantile(b, qs)).mean())


def physics_block(gen_q, hold_q):
    """The five normalised W1 numbers plus the physics score."""
    std = {k: float(hold_q[:, QCOL[k]].std(ddof=0)) for k in ("dx", "dy", "flux", "radius", "e1", "e2")}

    def norm_w1(key):
        s = std[key]
        w = wasserstein1(gen_q[:, QCOL[key]], hold_q[:, QCOL[key]])
        return float(w / s) if s > 0 else float("nan")

    out = {"w1_dx": norm_w1("dx"), "w1_dy": norm_w1("dy")}
    out["w1_offset"] = float(0.5 * (out["w1_dx"] + out["w1_dy"]))
    for k in ("flux", "radius", "e1", "e2"):
        out["w1_" + k] = norm_w1(k)
    out["physics"] = float(np.mean([out["w1_" + k] for k in PHYS_KEYS]))
    out["holdout_std"] = std
    return out


# --------------------------------------------------------------------------------------
# intensity alignment (addendum, 2026-09-07)
# --------------------------------------------------------------------------------------
def align_to_target(images, target_total):
    """Rescale EVERY image in `images` by its own scalar so its own total intensity equals
    `target_total` exactly (module docstring, "Intensity-aligned secondary block"). Applied
    to whichever arrays play either side of the four aligned metrics, so the result is
    scale-free on both sides, not just the generated one.

    An image whose own total is <= 0 cannot be scaled multiplicatively into a positive target
    and is returned unchanged -- this never happens for any real arm output here (every
    generator produces a non-negative field with positive total), it just keeps the function
    from dividing by zero.

    Returned in float64 (NOT downcast to float32): every downstream consumer -- pdf_js,
    conv_mmd2, nn_l2 -- casts its inputs internally anyway, and keeping float64 here is what
    lets an aligned image's own total match the target to < 1e-9 relative error rather than
    float32's ~1e-7 machine epsilon."""
    arr = np.asarray(images, dtype=np.float64)
    totals = arr.reshape(arr.shape[0], -1).sum(axis=1)
    safe = np.where(totals > 0, totals, 1.0)
    scale = np.where(totals > 0, float(target_total) / safe, 1.0)
    return arr * scale[:, None, None]


# --------------------------------------------------------------------------------------
# radial profile
# --------------------------------------------------------------------------------------
def mean_profile(images, n_bins=N_PROFILE_BINS):
    """Mean centred radial profile of a stack: each image about its OWN centroid, normalised
    to unit flux first so a bright image does not dominate the mean."""
    imgs = np.asarray(images, dtype=np.float64)
    ny, nx = imgs.shape[1], imgs.shape[2]
    ys = np.arange(ny, dtype=np.float64)[:, None]
    xs = np.arange(nx, dtype=np.float64)[None, :]
    edges = np.arange(n_bins + 1, dtype=np.float64)
    acc = np.zeros(n_bins, dtype=np.float64)
    used = 0
    for i in range(imgs.shape[0]):
        w = np.clip(imgs[i], 0.0, None)
        tot = float(w.sum())
        if not (tot > 0):
            continue
        cy = float((w * ys).sum() / tot)
        cx = float((w * xs).sum() / tot)
        r = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2).ravel()
        v = (w / tot).ravel()
        num, _ = np.histogram(r, bins=edges, weights=v)
        cnt, _ = np.histogram(r, bins=edges)
        acc += np.where(cnt > 0, num / np.maximum(cnt, 1), 0.0)
        used += 1
    return acc / max(used, 1)


def profile_rmse(gen, hold_profile, n_bins=N_PROFILE_BINS):
    return float(np.sqrt(np.mean((mean_profile(gen, n_bins) - hold_profile) ** 2)))


# --------------------------------------------------------------------------------------
# paired block
# --------------------------------------------------------------------------------------
def paired_block(gen_crops, target_crops):
    """Mean |Delta| of the four shape quantities plus the paired crop-to-crop L2."""
    m = min(gen_crops.shape[0], target_crops.shape[0])
    g = quantity_table(gen_crops[:m])
    t = quantity_table(target_crops[:m])
    l2 = np.sqrt(((gen_crops[:m].reshape(m, -1).astype(np.float64)
                   - target_crops[:m].reshape(m, -1).astype(np.float64)) ** 2).sum(axis=1))
    return {"n_pairs": int(m),
            "pair_de1": float(np.abs(g[:, QCOL["e1"]] - t[:, QCOL["e1"]]).mean()),
            "pair_de2": float(np.abs(g[:, QCOL["e2"]] - t[:, QCOL["e2"]]).mean()),
            "pair_dflux": float(np.abs(g[:, QCOL["flux"]] - t[:, QCOL["flux"]]).mean()),
            "pair_dradius": float(np.abs(g[:, QCOL["radius"]] - t[:, QCOL["radius"]]).mean()),
            "pair_l2": float(l2.mean())}


EMPTY_PAIR = {"n_pairs": None, "pair_de1": None, "pair_de2": None, "pair_dflux": None,
              "pair_dradius": None, "pair_l2": None}


# --------------------------------------------------------------------------------------
# run discovery
# --------------------------------------------------------------------------------------
def _load_meta(run_dir):
    path = os.path.join(run_dir, "meta.json")
    if not os.path.isfile(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def _stack(dir_path):
    found = []
    for path in glob.glob(os.path.join(dir_path, "generated_*.npy")):
        m = _GEN_RE.match(os.path.basename(path))
        if m:
            found.append((int(m.group(1)), path))
    if not found:
        return None
    found.sort()
    return np.stack([np.load(p) for _, p in found], axis=0).astype(np.float32)


def score_run(run_dir, ctx):
    meta = _load_meta(run_dir)
    if meta is None:
        return None
    gen = _stack(run_dir)
    if gen is None:
        return None
    crops = _stack(os.path.join(run_dir, "crops"))
    n_tr = int(meta.get("n_train", 0))
    if n_tr < 1:
        return None
    train = ctx["train_images"][:min(n_tr, ctx["train_images"].shape[0])]

    gq = quantity_table(gen)
    row = {"run_dir": run_dir, "arm": meta.get("arm", os.path.basename(os.path.normpath(run_dir))),
           "script": meta.get("script"), "param_count": meta.get("param_count"),
           "seed": meta.get("seed"), "n_train": n_tr, "n_gen": int(gen.shape[0]),
           "n_holdout": int(ctx["holdout"].shape[0]),
           "conditional": bool(meta.get("conditional", False)),
           "protocol": meta.get("protocol"), "objective": meta.get("objective"),
           "latent": meta.get("latent")}
    row.update(physics_block(gq, ctx["hold_q"]))
    row["profile_rmse"] = profile_rmse(gen, ctx["hold_profile"])
    row["pdf_js"] = float(pdf_js(ctx["holdout"], gen))
    row["conv_mmd2"] = float(conv_mmd2(ctx["holdout"], gen, seed=CONV_SEED))
    row["nn_train"] = nn_l2(gen, train)
    row["nn_holdout"] = nn_l2(gen, ctx["holdout"])
    gen_al = align_to_target(gen, ctx["holdout_mean_total"])
    train_al = ctx["train_aligned"][:min(n_tr, ctx["train_aligned"].shape[0])]
    row["pdf_js_aligned"] = float(pdf_js(ctx["holdout_aligned"], gen_al))
    row["conv_mmd2_aligned"] = float(conv_mmd2(ctx["holdout_aligned"], gen_al, seed=CONV_SEED))
    row["nn_train_aligned"] = nn_l2(gen_al, train_al)
    row["nn_holdout_aligned"] = nn_l2(gen_al, ctx["holdout_aligned"])
    row["gen_min"] = float(gen.min())
    row["gen_max"] = float(gen.max())
    row["gen_mean_total"] = float(gen.reshape(gen.shape[0], -1).sum(axis=1).mean())
    row["holdout_mean_total"] = float(ctx["holdout"].reshape(ctx["holdout"].shape[0], -1).sum(axis=1).mean())
    row["has_crops"] = bool(crops is not None)
    if row["conditional"] and crops is not None:
        row.update(paired_block(crops, ctx["holdout_crops"]))
    else:
        row.update(EMPTY_PAIR)
    row["_crops_dir"] = os.path.join(run_dir, "crops") if crops is not None else None
    return row


def holdout_reference(ctx, n_trains):
    """`HOLDOUT split-half`: the held-out set's first half against its second half, scored by
    every distributional metric -- the floor of the table."""
    hold = ctx["holdout"]
    half = hold.shape[0] // 2
    a, b = hold[:half], hold[half:2 * half]
    hold_al = ctx["holdout_aligned"]
    a_al, b_al = hold_al[:half], hold_al[half:2 * half]
    aq = quantity_table(a)
    bq = quantity_table(b)
    rows = []
    for n in sorted(set(int(v) for v in n_trains)):
        r = {"run_dir": None, "arm": "HOLDOUT split-half", "script": None, "param_count": None,
             "seed": None, "n_train": n, "n_gen": int(a.shape[0]),
             "n_holdout": int(b.shape[0]), "conditional": False, "protocol": None,
             "objective": None, "latent": None}
        r.update(physics_block(aq, bq))
        r["profile_rmse"] = profile_rmse(a, mean_profile(b))
        r["pdf_js"] = float(pdf_js(b, a))
        r["conv_mmd2"] = float(conv_mmd2(b, a, seed=CONV_SEED))
        r["nn_train"] = nn_l2(hold, ctx["train_images"][:n])
        r["nn_holdout"] = nn_l2(hold, hold, exclude_self=True)
        # aligned counterparts: same role assignment as the four lines above (a/b for the
        # pdf_js/conv_mmd2 pairing, the FULL held-out set for the nn_train/nn_holdout pairing),
        # every array individually rescaled to ctx["holdout_mean_total"] first.
        r["pdf_js_aligned"] = float(pdf_js(b_al, a_al))
        r["conv_mmd2_aligned"] = float(conv_mmd2(b_al, a_al, seed=CONV_SEED))
        r["nn_train_aligned"] = nn_l2(hold_al, ctx["train_aligned"][:n])
        r["nn_holdout_aligned"] = nn_l2(hold_al, hold_al, exclude_self=True)
        r["gen_min"] = float(a.min())
        r["gen_max"] = float(a.max())
        r["gen_mean_total"] = float(a.reshape(a.shape[0], -1).sum(axis=1).mean())
        r["holdout_mean_total"] = float(b.reshape(b.shape[0], -1).sum(axis=1).mean())
        r["has_crops"] = True
        r.update(EMPTY_PAIR)
        r["_crops_dir"] = None
        rows.append(r)
    return rows


def random_pairing_rows(rows, ctx, arm_label, seed):
    """`RANDOM-PAIRING (ref)`: an UNCONDITIONAL arm's crops paired with held-out images at
    random. The floor of the paired block -- conditioning only means something above it."""
    out = []
    for n in sorted(set(int(r["n_train"]) for r in rows)):
        src = [r for r in rows if r["n_train"] == n and r["arm"] == arm_label
               and r["_crops_dir"] is not None]
        if not src:
            continue
        crops = _stack(src[0]["_crops_dir"])
        if crops is None:
            continue
        rng = np.random.default_rng(int(seed) + 1234 + n)
        m = min(crops.shape[0], ctx["holdout_crops"].shape[0])
        perm = rng.permutation(ctx["holdout_crops"].shape[0])[:m]
        r = {"run_dir": src[0]["run_dir"], "arm": "RANDOM-PAIRING (ref)", "script": None,
             "param_count": None, "seed": int(seed), "n_train": n, "n_gen": m,
             "n_holdout": int(ctx["holdout"].shape[0]), "conditional": True,
             "protocol": "random pairing of %s" % arm_label, "objective": None, "latent": None,
             "profile_rmse": None, "pdf_js": None, "conv_mmd2": None,
             "nn_train": None, "nn_holdout": None, "gen_min": None, "gen_max": None,
             "gen_mean_total": None, "holdout_mean_total": None, "has_crops": True,
             "pdf_js_aligned": None, "conv_mmd2_aligned": None,
             "nn_train_aligned": None, "nn_holdout_aligned": None,
             "_crops_dir": None}
        for k in ("w1_dx", "w1_dy", "w1_offset", "w1_flux", "w1_radius", "w1_e1", "w1_e2",
                  "physics", "holdout_std"):
            r[k] = None
        r.update(paired_block(crops[:m], ctx["holdout_crops"][perm]))
        out.append(r)
    return out


# --------------------------------------------------------------------------------------
def render_table(rows):
    def f(v, spec="{:.4g}"):
        return "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else spec.format(v)

    def key(r):
        return (int(r["n_train"]), 0 if r["arm"].startswith("HOLDOUT") else
                (2 if r["arm"].startswith("RANDOM") else 1), str(r["arm"]))

    head = ("| Arm | N | physics ↓ | W1(offset) | W1(flux) | W1(radius) | W1(e1) | W1(e2) | "
            "profile RMSE | PDF–JS ↓ | MMD² ↓ | NN-train | NN-holdout | "
            "\\|Δe1\\| | \\|Δe2\\| | \\|Δflux\\| | \\|Δradius\\| | paired L2 |")
    lines = [head, "|" + "---|" * 18]
    for r in sorted(rows, key=key):
        pc = r.get("param_count")
        label = r["arm"] if pc in (None, 0) else "%s (%s)" % (r["arm"], pc)
        lines.append("| " + " | ".join([
            label, str(r["n_train"]), f(r.get("physics")), f(r.get("w1_offset")),
            f(r.get("w1_flux")), f(r.get("w1_radius")), f(r.get("w1_e1")), f(r.get("w1_e2")),
            f(r.get("profile_rmse")), f(r.get("pdf_js")), f(r.get("conv_mmd2")),
            f(r.get("nn_train")), f(r.get("nn_holdout")), f(r.get("pair_de1")),
            f(r.get("pair_de2")), f(r.get("pair_dflux")), f(r.get("pair_dradius")),
            f(r.get("pair_l2"))]) + " |")
    return "\n".join(["### Round 3 — generated vs 200 held-out lens images", "", ] + lines)


def render_table_aligned(rows):
    """Second table (addendum, 2026-09-07): the four metrics recomputed with every image --
    generated, training, held-out alike -- individually rescaled to the held-out mean total.
    Physics / W1 / paired columns are unaffected by alignment and are not repeated here."""
    def f(v, spec="{:.4g}"):
        return "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else spec.format(v)

    def key(r):
        return (int(r["n_train"]), 0 if r["arm"].startswith("HOLDOUT") else
                (2 if r["arm"].startswith("RANDOM") else 1), str(r["arm"]))

    head = "| Arm | N | PDF–JS (aligned) ↓ | MMD² (aligned) ↓ | NN-train (aligned) | NN-holdout (aligned) |"
    lines = [head, "|" + "---|" * 6]
    for r in sorted(rows, key=key):
        pc = r.get("param_count")
        label = r["arm"] if pc in (None, 0) else "%s (%s)" % (r["arm"], pc)
        lines.append("| " + " | ".join([
            label, str(r["n_train"]), f(r.get("pdf_js_aligned")), f(r.get("conv_mmd2_aligned")),
            f(r.get("nn_train_aligned")), f(r.get("nn_holdout_aligned"))]) + " |")
    return "\n".join(["### Round 3 (intensity-aligned) — generated vs 200 held-out lens images", "",
                       "Every image (generated / training / held-out) individually rescaled so "
                       "its own total equals the held-out mean total, THEN PDF-JS / Conv2D-MMD^2 "
                       "/ NN-train / NN-holdout are recomputed. Physics score is unaffected "
                       "(flux is a separate quantity; radius/e1/e2 are already scale-invariant) "
                       "and is not repeated here -- see the primary table above.", ""] + lines)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Round-3 evaluation table.")
    p.add_argument("--outputs-dir", nargs="+", default=["outputs_lens_r3"])
    p.add_argument("--runs", nargs="*", default=None)
    p.add_argument("--n-holdout", type=int, default=data_lens_r3.N_HOLDOUT)
    p.add_argument("--n-train-max", type=int, default=data_lens_r3.N_TRAIN_MAX)
    p.add_argument("--results", default="results_r3.jsonl")
    p.add_argument("--random-pair-arm", default="IQP-lowvar-joint-MMD")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    d = data_lens_r3.load_r3(args.n_train_max, n_holdout=args.n_holdout)
    ctx = {"holdout": d.holdout_images, "holdout_crops": d.holdout_crops,
           "train_images": d.train_images}
    ctx["hold_q"] = quantity_table(ctx["holdout"])
    ctx["hold_profile"] = mean_profile(ctx["holdout"])
    # intensity-aligned block (addendum): one fixed target for the whole run -- the held-out
    # set's own mean total -- and the held-out / training arrays pre-rescaled to it ONCE so
    # every row's aligned metrics share exactly the same reference arrays.
    ctx["holdout_mean_total"] = float(ctx["holdout"].reshape(ctx["holdout"].shape[0], -1)
                                       .sum(axis=1).mean())
    ctx["holdout_aligned"] = align_to_target(ctx["holdout"], ctx["holdout_mean_total"])
    ctx["train_aligned"] = align_to_target(ctx["train_images"], ctx["holdout_mean_total"])
    print("[evaluate_r3] holdout=%s train=%s" % (ctx["holdout"].shape, ctx["train_images"].shape))

    if args.runs:
        run_dirs = list(args.runs)
    else:
        run_dirs = []
        for dd in args.outputs_dir:
            if not os.path.isdir(dd):
                raise SystemExit("evaluate_r3.py: --outputs-dir %r is not a directory" % dd)
            run_dirs.extend(sorted(x for x in glob.glob(os.path.join(dd, "*")) if os.path.isdir(x)))

    rows = []
    for rd in run_dirs:
        row = score_run(rd, ctx)
        if row is None:
            print("[evaluate_r3] skipped %s (no meta.json / no generated_*.npy)" % rd)
        else:
            rows.append(row)
    if not rows:
        raise SystemExit("evaluate_r3.py: no usable runs under %r" % (args.outputs_dir,))

    seen = {}
    for r in rows:
        seen.setdefault((r["arm"], r["n_train"]), []).append(r["run_dir"])
    for k, v in sorted(seen.items()):
        if len(v) > 1:
            print("[evaluate_r3] WARNING: arm %r at N=%s appears %d times: %s" % (k[0], k[1], len(v), v))

    # Honest notes, printed rather than folded into a metric.
    hmt = ctx["holdout_mean_total"]
    for r in sorted(rows, key=lambda z: (z["n_train"], str(z["arm"]))):
        if hmt > 0:
            ratio = r["gen_mean_total"] / hmt
            if ratio > 1.25 or ratio < 0.8:
                print("[evaluate_r3] NOTE: arm %r N=%s mean total intensity %.4g vs held-out "
                      "%.4g (x%.2f) -- part of its NN / PDF-JS / MMD2 distance is this scale "
                      "mismatch, not spatial structure" % (r["arm"], r["n_train"],
                                                           r["gen_mean_total"], hmt, ratio))
        if r["conditional"] and r["n_pairs"] is None:
            print("[evaluate_r3] NOTE: arm %r N=%s is tagged conditional but has no crops/ -- "
                  "paired columns left blank" % (r["arm"], r["n_train"]))
    print("[evaluate_r3] NOTE: CTRL-copy writes the RAW training images, the only arm that "
          "keeps the ~0.6%% of light outside the 32x32 window; every other arm's flux is "
          "bounded by its crop.")

    rows = rows + holdout_reference(ctx, [r["n_train"] for r in rows])
    rows = rows + random_pairing_rows(rows, ctx, args.random_pair_arm, args.seed)

    with open(args.results, "w") as fh:
        for r in rows:
            r.pop("_crops_dir", None)
            fh.write(json.dumps(r) + "\n")
    print("[evaluate_r3] wrote %d rows to %s" % (len(rows), args.results))
    print()
    print(render_table(rows))
    print()
    print(render_table_aligned(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
