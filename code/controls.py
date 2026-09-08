"""CTRL-marginal and CTRL-interp control arms for the arXiv 2603.06440 replication.

See docs/IQP_TURBULENCE_REPLICATION_PREREGISTRATION.md, Arms table, and INTERFACES.md
(snapshot / code / output contracts). These two arms exist to test whether "IQP beats
RBM/DCGAN" survives once trivial baselines are added:

  CTRL-marginal: product-Bernoulli over the 18 code bits, fit per anchor snapshot from
                 N_t = 5e4 tuples drawn from that anchor's fixed code pool
                 (codec.sample_codes), piecewise-linearly interpolated across anchors
                 in parameter space, then sampled (n_synth codes) and aggregated to a
                 field. No bit-bit correlations are modelled at all.
  CTRL-interp:   no model, no params -- generated field at unseen tau is the piecewise-linear
                 interpolation, in field space, of the two neighbouring anchor
                 QUANTIZED ground-truth fields (audit fix: interpolating the raw,
                 un-quantized field would give CTRL-interp sub-6-bit precision no
                 generative arm has access to).

Time index t is 1-based throughout (t in {1, ..., 1000}; array index = t - 1), matching
the paper and every other arm. For --data synthetic, the anchor/held-out split comes from
`synth_data.make_split_labels` -- the ONE split source shared by every arm (no independent
seed-offset redraw here); a_z/b_z come from the raw anchor fields
(`synth_data.anchor_a_z_b_z`); each anchor's training-tuple pool is its own full 4096-cell
enumeration (`synth_data.pool_and_quantized_field`).

Both arms write outputs/<arm>/generated_t{tau}.npy (64,64) float32 and outputs/<arm>/meta.json,
per INTERFACES.md's Generated-sample contract. Python 3.9 compatible (no `X | None`).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

from codec import N_QUBITS, aggregate, field_to_codes_full, sample_codes

# The 11 anchor snapshots per the pre-registration: t in {1, 100, 200, ..., 1000}.
ANCHOR_TS = [1, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

N_T_TUPLES = 50000       # paper's N_t: tuples sampled per anchor snapshot to fit marginals
N_SYNTH_DEFAULT = 100000  # paper's "e.g. 10^5 samples" synthesis size


def quantized_field(field, a_z, b_z):
    """GT field convention (audit fix): `field` with z quantized to 6 bits via its own full
    4096-cell code pool and that pool's aggregate -- see synth_data.pool_and_quantized_field
    (kept local here to avoid a hard synth_data dependency for the d2 branch)."""
    pool = field_to_codes_full(field, a_z, b_z)
    return aggregate(pool, a_z, b_z)


def _load_data(data_source, seed, anchors_max=None, n_holdout=200):
    """Return (get_field(t) -> (64,64) float32 RAW field, anchor_ts (list[int], 1-based),
    holdout_taus (list[int], 1-based, or None)).

    synthetic: field data AND the anchor/held-out split both come from synth_data.py
    (make_synthetic_snapshots / make_split_labels) at a FIXED data seed
    (synth_data.SYNTH_DATA_SEED), decoupled from `seed` -- the same call every other arm
    (baseline_rbm.py, baseline_dcgan.py, train_iqp.py) makes, so all arms share one
    synthetic ground truth and one set of held-out tau regardless of each arm's own
    training seed. `anchors_max`/`n_holdout` apply the split's smoke-mode truncation
    here already, so the caller does not need to re-slice for this branch (kept for the
    d2 branch below). holdout_taus is None for d2 unless load_d2() supplies it.
    """
    if data_source == "synthetic":
        from synth_data import N_SNAPSHOTS_SYNTH, SYNTH_DATA_SEED, make_split_labels, make_synthetic_snapshots

        all_snaps = make_synthetic_snapshots(N_SNAPSHOTS_SYNTH, seed=SYNTH_DATA_SEED)
        anchors_t, holdout_t = make_split_labels(
            N_SNAPSHOTS_SYNTH, seed=SYNTH_DATA_SEED, n_holdout=n_holdout, anchors_max=anchors_max
        )

        def get_field(t):
            return np.asarray(all_snaps[int(t) - 1], dtype=np.float32)

        return get_field, [int(a) for a in anchors_t.tolist()], [int(t) for t in holdout_t.tolist()]
    elif data_source == "d2":
        try:
            from data_d2 import load_d2
        except ImportError as exc:
            sys.exit(
                "controls.py: --data d2 requires data_d2.py, which is not available yet "
                "({}). Use --data synthetic for now.".format(exc)
            )
        snapshots, anchors, holdout_taus = load_d2()

        def get_field(t):
            return np.asarray(snapshots[int(t)].field, dtype=np.float32)

        return get_field, [int(a) for a in anchors], [int(h) for h in holdout_taus]
    else:
        raise ValueError("unknown --data {!r}".format(data_source))


def make_holdout_taus(anchor_ts, n_holdout, seed):
    """d2-only fallback: 200 unseen tau drawn uniformly from [anchor_ts[0], anchor_ts[-1]]
    excluding anchors, seeded. NOT used for --data synthetic (that always resolves its
    held-out set from synth_data.make_split_labels inside `_load_data`); only exercised if
    load_d2() does not itself supply a held-out set."""
    rng = np.random.default_rng(int(seed) + 10_000)
    lo, hi = anchor_ts[0], anchor_ts[-1]
    anchor_set = set(anchor_ts)
    pool = [t for t in range(lo, hi + 1) if t not in anchor_set]
    n = min(n_holdout, len(pool))
    chosen = rng.choice(pool, size=n, replace=False)
    return sorted(int(t) for t in chosen)


# --------------------------------------------------------------------------------------
# Piecewise-linear interpolation across anchors (shared by both arms: CTRL-marginal
# interpolates the 18-dim parameter vector, CTRL-interp interpolates the (64,64) field).
# --------------------------------------------------------------------------------------
def _bracket(tau, anchor_ts):
    """Return (i, j, w) such that tau ~ (1-w)*anchor_ts[i] + w*anchor_ts[j], i,j adjacent.

    Clamps to the boundary anchor (w=0) outside [anchor_ts[0], anchor_ts[-1]].
    """
    ts = anchor_ts
    if tau <= ts[0]:
        return 0, 0, 0.0
    if tau >= ts[-1]:
        n = len(ts) - 1
        return n, n, 0.0
    i = int(np.searchsorted(ts, tau, side="right")) - 1
    j = i + 1
    span = ts[j] - ts[i]
    w = 0.0 if span == 0 else (tau - ts[i]) / float(span)
    return i, j, w


def interp_anchors(tau, anchor_ts, anchor_vals):
    """anchor_vals: array (n_anchors, ...) in the same order as anchor_ts."""
    i, j, w = _bracket(tau, anchor_ts)
    return (1.0 - w) * anchor_vals[i] + w * anchor_vals[j]


# --------------------------------------------------------------------------------------
# CTRL-marginal
# --------------------------------------------------------------------------------------
def _code_bits(codes):
    """codes: int64 (n,) -> bits (n, 18) float64, bit j = qubit j (LSB = qubit 0)."""
    codes = np.asarray(codes, dtype=np.int64)
    return ((codes[:, None] >> np.arange(N_QUBITS)) & 1).astype(np.float64)


def fit_marginal_params(field, a_z, b_z, rng, n_tuples=N_T_TUPLES):
    """Per-anchor product-Bernoulli parameters: p_j = mean(bit_j) over n_tuples codes drawn
    (with replacement, codec.sample_codes) from this snapshot's fixed code pool -- its full
    4096-cell enumeration (codec.field_to_codes_full), per the audit's code-pool fix."""
    pool = field_to_codes_full(field, a_z, b_z)
    codes = sample_codes(pool, n_tuples, rng)
    return _code_bits(codes).mean(axis=0)  # (18,)


def sample_product_bernoulli(p_vec, n, rng):
    """n i.i.d. codes with independent bits, bit j ~ Bernoulli(p_vec[j])."""
    p_vec = np.clip(np.asarray(p_vec, dtype=np.float64), 0.0, 1.0)
    bits = (rng.random((n, N_QUBITS)) < p_vec[None, :]).astype(np.int64)
    codes = np.zeros(n, dtype=np.int64)
    for j in range(N_QUBITS):
        codes |= (bits[:, j] << j)
    return codes


def run_marginal_gaussian_latent(get_field, anchor_ts, seed, n_synth, out_dir, n_gen,
                                 a_z_b_z, n_tuples=N_T_TUPLES, sqrt=False, extra_meta=None):
    """CTRL-marginal for data with NO time axis (the lens pilot).

    The product-Bernoulli p-vector plays exactly the role theta_lat plays for IQP/RBM: it
    is fitted per training image (`anchor_ts` = the N training images, index order), and
    then -- instead of Eq. 31 interpolation, which has no time to interpolate along -- a
    diagonal Gaussian N(mu, diag sigma^2) is fitted to the N fitted p-vectors, `n_gen`
    p-vectors are drawn from it (clipped back into [0, 1], since a Bernoulli parameter
    must be a probability), and one image is generated per draw.

    Writes generated_{i:03d}.npy in BOTH aggregation conventions (lens_synth.write_pair).
    """
    import lens_synth

    t0 = time.time()
    rng = np.random.default_rng(seed)
    a_z, b_z = float(a_z_b_z[0]), float(a_z_b_z[1])
    anchor_fields = [get_field(t) for t in anchor_ts]
    p_vecs = np.stack(
        [fit_marginal_params(f, a_z, b_z, rng, n_tuples=n_tuples) for f in anchor_fields], axis=0
    )  # (n_train, 18)

    mu, sigma = lens_synth.fit_gaussian_latent(p_vecs)
    rng_synth = np.random.default_rng(seed + 4242)
    draws = np.clip(lens_synth.sample_gaussian_latents(mu, sigma, n_gen, rng_synth), 0.0, 1.0)
    for i in range(int(n_gen)):
        codes = sample_product_bernoulli(draws[i], n_synth, rng)
        lens_synth.write_pair(out_dir, i, codes, a_z, b_z, sqrt=sqrt)

    meta = {
        "arm": "CTRL-marginal",
        "param_count": int(N_QUBITS * len(anchor_ts)),
        "seed": int(seed),
        "anchors": [int(t) for t in anchor_ts],
        "train_snapshots": len(anchor_ts),
        "n_train": len(anchor_ts),
        "a_z": a_z,
        "b_z": b_z,
        "n_synth": int(n_synth),
        "n_gen": int(n_gen),
        "n_t_tuples": int(n_tuples),
        "synth_mode": "gaussian-latent",
        "wallclock_s": time.time() - t0,
    }
    meta.update(lens_synth.latent_stats(mu, sigma))
    if extra_meta:
        meta.update(extra_meta)
    lens_synth.write_metas(out_dir, meta)
    return meta, p_vecs, a_z, b_z


def run_marginal(get_field, anchor_ts, holdout_taus, seed, n_synth, out_dir, n_tuples=N_T_TUPLES,
                  a_z_b_z=None):
    """a_z_b_z: optional (a_z, b_z) override. synthetic (default None) keeps deriving
    a_z/b_z from the raw anchor fields' own min/max (matches synth_data.anchor_a_z_b_z).
    d2 MUST pass data_d2.get_z_range()'s bounds here instead of re-deriving from
    `.field` -- .field is already codec.aggregate'd onto a 64x64 grid, which averages
    away the extremes and yields a narrower range than the one load_d2() used to
    encode pool_codes (~18.8% narrower, observed), causing an encode/decode scale
    mismatch downstream."""
    t0 = time.time()
    rng = np.random.default_rng(seed)
    anchor_fields = [get_field(t) for t in anchor_ts]
    if a_z_b_z is not None:
        a_z, b_z = float(a_z_b_z[0]), float(a_z_b_z[1])
    else:
        a_z = float(min(f.min() for f in anchor_fields))
        b_z = float(max(f.max() for f in anchor_fields))
    if b_z <= a_z:
        b_z = a_z + 1.0
    p_vecs = np.stack(
        [fit_marginal_params(f, a_z, b_z, rng, n_tuples=n_tuples) for f in anchor_fields], axis=0
    )  # (n_anchors, 18)

    os.makedirs(out_dir, exist_ok=True)
    for tau in holdout_taus:
        p_tau = interp_anchors(tau, anchor_ts, p_vecs)
        codes = sample_product_bernoulli(p_tau, n_synth, rng)
        field = aggregate(codes, a_z, b_z)
        np.save(os.path.join(out_dir, "generated_t{}.npy".format(tau)), field.astype(np.float32))

    meta = {
        "arm": "CTRL-marginal",
        "param_count": int(N_QUBITS * len(anchor_ts)),
        "seed": int(seed),
        "anchors": [int(t) for t in anchor_ts],
        "train_snapshots": len(anchor_ts),
        "a_z": a_z,
        "b_z": b_z,
        "n_synth": int(n_synth),
        "n_t_tuples": int(n_tuples),
        "wallclock_s": time.time() - t0,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    return meta, p_vecs, a_z, b_z


# --------------------------------------------------------------------------------------
# CTRL-interp
# --------------------------------------------------------------------------------------
def run_interp(get_field, anchor_ts, holdout_taus, seed, out_dir, a_z_b_z=None):
    """a_z_b_z: see run_marginal's docstring -- same override contract (None: derive
    from raw anchor fields for synthetic; d2 MUST pass data_d2.get_z_range())."""
    t0 = time.time()
    anchor_fields_raw = np.stack([get_field(t) for t in anchor_ts], axis=0)  # (n_anchors, 64, 64)
    if a_z_b_z is not None:
        a_z, b_z = float(a_z_b_z[0]), float(a_z_b_z[1])
    else:
        a_z = float(anchor_fields_raw.min())
        b_z = float(anchor_fields_raw.max())
    if b_z <= a_z:
        b_z = a_z + 1.0
    # Interpolate the QUANTIZED anchor fields (audit fix), not the raw ones: every
    # generative arm is capped at 6-bit z, so the "no model" control should be too.
    anchor_fields = np.stack(
        [quantized_field(f, a_z, b_z) for f in anchor_fields_raw], axis=0
    )

    os.makedirs(out_dir, exist_ok=True)
    for tau in holdout_taus:
        field = interp_anchors(tau, anchor_ts, anchor_fields)
        np.save(os.path.join(out_dir, "generated_t{}.npy".format(tau)), field.astype(np.float32))

    meta = {
        "arm": "CTRL-interp",
        "param_count": 0,
        "seed": int(seed),
        "anchors": [int(t) for t in anchor_ts],
        "train_snapshots": len(anchor_ts),
        "a_z": a_z,
        "b_z": b_z,
        "wallclock_s": time.time() - t0,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    return meta, anchor_fields


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="CTRL-marginal / CTRL-interp control arms for the IQP turbulence replication."
    )
    p.add_argument("--arm", choices=["marginal", "interp"], required=True)
    p.add_argument("--data", choices=["synthetic", "d2", "lens"], default="synthetic")
    # --- lens pilot (docs/IQP_LENS_PILOT_PREREGISTRATION.md) --------------------------
    p.add_argument("--lens-class", type=str, default=None,
                   help="--data lens: which DeepLense Model II class (axion/cdm/no_sub)")
    p.add_argument("--n-train", type=int, default=11,
                   help="--data lens: N training images (11 or 100)")
    p.add_argument("--lens-sqrt", action="store_true",
                   help="--data lens: quantize sqrt(intensity); generated fields are squared back")
    p.add_argument("--synth-mode", choices=["interp", "gaussian-latent"], default="interp",
                   help="'interp' (default, time-indexed data); 'gaussian-latent' (lens pilot): fit "
                        "N(mu, diag sigma^2) to the N fitted per-image p-vectors and draw --n-gen")
    p.add_argument("--n-gen", type=int, default=200,
                   help="--synth-mode gaussian-latent: number of images to synthesize")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default=None, help="default: outputs/<arm>")
    p.add_argument("--n-synth", type=int, default=N_SYNTH_DEFAULT)
    p.add_argument("--n-tuples", type=int, default=N_T_TUPLES, help="tuples sampled per anchor to fit marginals")
    p.add_argument("--anchors-max", type=int, default=None, help="smoke: use only the first K anchors")
    p.add_argument("--n-holdout", type=int, default=200, help="number of unseen tau to generate")
    return p.parse_args(argv)


def _main_lens(args, out_dir):
    """--data lens: CTRL-marginal only (CTRL-interp needs a time axis to interpolate along,
    which lens images do not have -- see docs/IQP_LENS_PILOT_PREREGISTRATION.md's Arms
    table, where the lens analogues of "no-model" controls are controls_lens.py's
    CTRL-copy / CTRL-mean instead)."""
    import data_lens

    if args.arm != "marginal":
        sys.exit("controls.py: --data lens supports only --arm marginal (CTRL-interp has no "
                 "time axis to interpolate along; use controls_lens.py --arm copy/mean)")
    if not args.lens_class:
        sys.exit("--data lens requires --lens-class (one of %s)" % data_lens.get_class_names())

    snapshots, anchors, _holdout = data_lens.load_lens(
        args.lens_class, n_train=args.n_train, n_holdout=0, sqrt=args.lens_sqrt
    )
    if args.anchors_max is not None:
        anchors = anchors[: args.anchors_max]

    def get_field(t):
        return np.asarray(snapshots[int(t)].field, dtype=np.float32)

    extra = {
        "data": "lens",
        "class_names": data_lens.get_class_names(),
        "lens_class": args.lens_class,
        "lens_sqrt": bool(args.lens_sqrt),
    }
    meta, _p, _a, _b = run_marginal_gaussian_latent(
        get_field, anchors, args.seed, args.n_synth, out_dir, args.n_gen,
        a_z_b_z=data_lens.get_z_range(), n_tuples=args.n_tuples, sqrt=args.lens_sqrt,
        extra_meta=extra,
    )
    print(json.dumps(meta, indent=2, default=str))


def main(argv=None):
    args = parse_args(argv)
    out_dir = args.out if args.out is not None else os.path.join("outputs", args.arm)

    if args.data == "lens":
        return _main_lens(args, out_dir)

    get_field, anchor_ts, holdout_taus = _load_data(
        args.data, args.seed, anchors_max=args.anchors_max, n_holdout=args.n_holdout
    )
    a_z_b_z = None
    if args.data == "d2":
        # d2's split truncation is not resolved inside _load_data (load_d2()'s split is
        # taken as-is); synthetic's is already fully resolved above via make_split_labels.
        if args.anchors_max is not None:
            anchor_ts = anchor_ts[: args.anchors_max]
        if holdout_taus is None:
            holdout_taus = make_holdout_taus(anchor_ts, args.n_holdout, args.seed)
        else:
            holdout_taus = holdout_taus[: args.n_holdout]
        # MUST match the bounds load_d2() used to encode pool_codes (data_d2.get_z_range),
        # not a re-derivation from `.field` (see run_marginal/run_interp docstrings).
        from data_d2 import get_z_range

        a_z_b_z = get_z_range("mid")

    if args.arm == "marginal":
        meta, _, _, _ = run_marginal(
            get_field, anchor_ts, holdout_taus, args.seed, args.n_synth, out_dir, n_tuples=args.n_tuples,
            a_z_b_z=a_z_b_z,
        )
    else:
        meta, _ = run_interp(get_field, anchor_ts, holdout_taus, args.seed, out_dir, a_z_b_z=a_z_b_z)

    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
