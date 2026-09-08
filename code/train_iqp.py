"""Algorithm 1 of arXiv 2603.06440: time-conditioned IQP generator trained with an exact closed-form MMD.

Stage 1 (first anchor):  theta_lat^(1) random and FROZEN, train theta_core for --steps-core steps.
Stage 2 (later anchors): theta_core frozen, train only theta_lat (50 params) for --steps-lat steps,
                         warm-started from the previous anchor.
Synthesis:               piecewise-linear interpolation of theta_lat to each unseen time tau, then
                         --n-synth tuples sampled from q_theta and aggregated onto the 64x64 grid.

All expectation values are EXACT (three FWHTs per step, see DESIGN_exact_iqp.md); the paper's
500-sample-per-step estimator is replaced by its exact limit.

Outputs (INTERFACES.md "Generated-sample contract"):
    outputs/<arm>/generated_t{tau}.npy   (64, 64) float32, one per held-out tau
    outputs/<arm>/meta.json
    outputs/<arm>/theta_core.npy , theta_lat_anchors.npy , loss_curves.npz

Python 3.9 compatible.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import List

import numpy as np
import torch

import codec
import codec12
import synth_data
from iqp_exact import IQPExact, sigma_for_avg_weight


# --------------------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train the time-conditioned IQP generator (Algorithm 1).")
    p.add_argument("--data", choices=("synthetic", "d2", "lens"), default="synthetic")
    p.add_argument("--n-snapshots", type=int, default=1000, help="synthetic data only")
    # --- lens pilot (docs/IQP_LENS_PILOT_PREREGISTRATION.md) --------------------------
    p.add_argument("--lens-class", type=str, default=None,
                   help="--data lens: which DeepLense Model II class (axion/cdm/no_sub)")
    p.add_argument("--n-train", type=int, default=11,
                   help="--data lens: N training images = N anchors (11 or 100)")
    p.add_argument("--lens-sqrt", action="store_true",
                   help="--data lens secondary arm: quantize sqrt(intensity) instead of intensity "
                        "(generated fields are squared back to intensity space on write, so they are "
                        "scored against the SAME ground truth as every other arm)")
    p.add_argument("--encoding", choices=("tuple", "pmf"), default="tuple",
                   help="--data lens only. 'tuple' (default, round 1 = the paper's literal "
                        "encoding): 18 qubits, one code = (kx, ky, kz). 'pmf' (round 2, "
                        "docs/IQP_LENS_PILOT_PREREGISTRATION.md Addendum): 12 qubits, one code = "
                        "(kx, ky), the target distribution IS the normalised image and the total "
                        "intensity S is carried outside the circuit (codec12.py).")
    p.add_argument("--stage1-train-all", action="store_true",
                   help="round-2 fix: train ALL parameters in stage 1 (theta_lat is NOT frozen). "
                        "Stage 2 is unchanged -- it still adapts only theta_lat per image, "
                        "warm-started from the stage-1 latent. Default (off) = the paper protocol.")
    p.add_argument("--arm-name", type=str, default=None,
                   help="override meta.json's `arm` label (evaluate_lens.py groups rows by it). "
                        "Needed when several arms share a parameter count.")
    p.add_argument("--K", type=int, default=6, help="max Pauli weight (6 -> 31,180 params, 7 -> 63,004)")
    p.add_argument("--d-lat", type=int, default=50)
    p.add_argument("--steps-core", type=int, default=30000)
    p.add_argument("--steps-lat", type=int, default=3000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--sigmas", type=str, default="auto3",
                   help="'auto3' (default, pre-registration main setting): Ref [38] three-bandwidth rule "
                        "computed from the t=1 (first anchor) data -- sigma1 gives mean Pauli weight 2, "
                        "sigma3 = sqrt(median pairwise Hamming distance of the t=1 codes), "
                        "sigma2 = sqrt((sigma1^2+sigma3^2)/2); or a comma-separated list of explicit sigmas")
    p.add_argument("--n-tuples", type=int, default=50000, help="N_t data-side tuples per snapshot")
    p.add_argument("--full-grid", action="store_true",
                   help="use all 4096 grid cells once instead of --n-tuples random draws")
    p.add_argument("--init", choices=("data", "random"), default="data")
    p.add_argument("--init-noise", type=float, default=0.0,
                   help="std of the N(0, .) init for |s|>2 params when --init data (Ref[38] default 0)")
    p.add_argument("--init-std", type=float, default=0.01, help="std of the N(0, .) init when --init random")
    p.add_argument("--lat-init", choices=("random", "data"), default="random",
                   help="theta_lat^(1): the paper draws it randomly (default); 'data' keeps the data-init values")
    p.add_argument("--lat-std", type=float, default=0.01, help="std of the random theta_lat^(1)")
    p.add_argument("--n-synth", type=int, default=100000, help="tuples drawn per synthesized snapshot")
    p.add_argument("--synth-mode", choices=("interp", "gaussian-latent"), default="interp",
                   help="'interp' (default, paper Eq. 31): piecewise-linear theta_lat interpolation to "
                        "each held-out time tau -- the only sensible mode for time-indexed data "
                        "(synthetic/d2). 'gaussian-latent': there is no time axis (lens pilot), so fit "
                        "N(mu, diag sigma^2) to the N trained theta_lat vectors and draw --n-gen latents "
                        "from it, one image each.")
    p.add_argument("--n-gen", type=int, default=200,
                   help="--synth-mode gaussian-latent: number of images to synthesize")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="outputs/iqp")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--dtype", choices=("complex64", "complex128"), default="complex64")
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--anchors-max", type=int, default=0, help=">0: use only the first N anchors (smoke tests)")
    p.add_argument("--n-holdout", type=int, default=200, help="number of unseen tau to synthesize")
    return p.parse_args(argv)


# --------------------------------------------------------------------------------------
def load_snapshots_and_split(args):
    """Return (anchors, unseen, a_z, b_z, pool_fn, lens_extra); anchors/unseen are 1-based time labels t
    (array index = t - 1, matching the paper's own t in {1, ..., 1000} and every other arm's
    file naming); pool_fn(t) -> int64 code pool for that (1-based) snapshot.

    synthetic: data AND the anchor/held-out split both come from synth_data.py at a FIXED
    seed (synth_data.SYNTH_DATA_SEED), decoupled from --seed (which only seeds this run's
    model/training stochasticity) -- so every seed of every arm trains/evaluates against the
    identical synthetic ground truth and the identical 200 held-out tau. `make_split_labels`
    is the ONE split source (INTERFACES.md, pre-registration Arms table) -- no independent
    redraw here. a_z/b_z come from the RAW anchor fields (synth_data.anchor_a_z_b_z); the
    per-anchor code pool is that field's full 4096-cell enumeration
    (synth_data.pool_and_quantized_field), so training tuples are always drawn from a fixed
    pool via codec.sample_codes, never re-quantized per draw.
    d2: anchors/unseen/pools come from data_d2.load_d2()'s own pre-fixed split (same
    convention: 1-based t, dict of Snapshot(field, pool_codes) -- INTERFACES.md), NOT
    re-derived here, for the same reason (and so they match controls.py's d2 branch).
    """
    if args.data == "synthetic":
        ds = synth_data.SYNTH_DATA_SEED
        all_snaps = synth_data.make_synthetic_snapshots(args.n_snapshots, seed=ds)
        anchors, unseen = synth_data.make_split_labels(
            all_snaps.shape[0], seed=ds, n_holdout=args.n_holdout,
            anchors_max=args.anchors_max if args.anchors_max > 0 else None,
        )
        a_z, b_z = synth_data.anchor_a_z_b_z(all_snaps, anchors)

        def pool_fn(t):
            return synth_data.pool_and_quantized_field(all_snaps[int(t) - 1], a_z, b_z)[0]

        return anchors, unseen, a_z, b_z, pool_fn, None

    if args.data == "lens":
        # DeepLense Model II pilot: "snapshot" = image, anchors = the N training images in
        # index order, held-out = 200 val images (keys 1001..1200). a_z/b_z are FIXED at
        # (0, 1) by data_lens.get_z_range() for every class -- never re-derived here.
        import data_lens
        if not args.lens_class:
            raise SystemExit("--data lens requires --lens-class (one of %s)"
                             % data_lens.get_class_names())
        snapshots, anchors, holdout = data_lens.load_lens(
            args.lens_class, n_train=args.n_train, seed=args.seed,
            n_holdout=args.n_holdout, sqrt=args.lens_sqrt,
            encoding=args.encoding, pool_size=args.n_tuples,
        )
        anchors = np.asarray(anchors, dtype=np.int64)
        unseen = np.asarray(holdout, dtype=np.int64)
        if args.anchors_max > 0:
            anchors = anchors[: args.anchors_max]
        a_z, b_z = data_lens.get_z_range()

        def pool_fn_lens(t):
            return np.asarray(snapshots[int(t)].pool_codes, dtype=np.int64)

        lens_extra = None
        if args.encoding == "pmf":
            # The pmf arms need the exact per-image distribution (--full-grid) and the ONE
            # intensity scale S = mean total intensity of the TRAINING images only.
            lens_extra = {
                "pmf_fn": (lambda t: np.asarray(snapshots[int(t)].pmf, dtype=np.float64)),
                "S": data_lens.mean_total_intensity(
                    args.lens_class, int(anchors.size), sqrt=args.lens_sqrt),
            }
        return anchors, unseen, a_z, b_z, pool_fn_lens, lens_extra

    try:
        from data_d2 import load_d2, get_z_range  # noqa: F401  (written by the DATA agent)
    except Exception as exc:  # pragma: no cover - depends on a file we do not own
        raise SystemExit("--data d2 needs data_d2.py with load_d2(); not available yet (%s)" % exc)
    snapshots, anchors, unseen = load_d2()
    anchors = np.asarray(anchors, dtype=np.int64)
    unseen = np.asarray(unseen, dtype=np.int64)
    if args.anchors_max > 0:
        anchors = anchors[: args.anchors_max]
    if args.n_holdout > 0:
        unseen = unseen[: args.n_holdout]
    # a_z/b_z MUST be the exact bounds load_d2() used to quantize/encode pool_codes
    # (data_d2.get_z_range, native z_val over the 11 anchors) -- NOT re-derived from
    # `.field`, whose 64x64-bin averaging narrows the range (~18.8% here) and would
    # decode/aggregate/quantize downstream on a different scale than the one used
    # to encode.
    a_z, b_z = get_z_range("mid")

    def pool_fn(t):
        return np.asarray(snapshots[int(t)].pool_codes, dtype=np.int64)

    return anchors, unseen, a_z, b_z, pool_fn, None


def codes_for(pool_codes, args, rng):
    """Draw this snapshot's training tuples from its fixed code pool (INTERFACES.md /
    codec.sample_codes). --full-grid uses the whole pool once.

    What the whole pool IS, per dataset (this is what makes --full-grid "exact"):
      synthetic : the 4096-cell full grid  -> the exact empirical pmf of the snapshot;
      d2        : the whole native-grid pool;
      lens, encoding=tuple : the image's 4,096 pixels, each exactly once
                  (data_lens._make_snapshot -> codec.field_to_codes_full), so
                  codec.codes_to_pmf(pool) IS the image's exact 18-bit pmf -- round 2's
                  "full 4,096-pixel pmf" column is exactly this flag;
      lens, encoding=pmf   : the pool is already a 5e4 draw from the image pmf (the paper
                  protocol's N_t sample), so it is used AS IS, and --full-grid does not go
                  through here at all -- main() takes Snapshot.pmf directly instead.
    """
    if args.full_grid:
        return np.asarray(pool_codes, dtype=np.int64)
    if getattr(args, "encoding", "tuple") == "pmf":
        # already exactly `--n-tuples` codes drawn from p at data_lens.POOL_SEED; resampling
        # them would add a second layer of sampling noise for no reason.
        return np.asarray(pool_codes, dtype=np.int64)
    return codec.sample_codes(pool_codes, args.n_tuples, rng)


def data_init_theta(sim, bits, cov_scale, noise, rng):
    """Port of Ref [38] utils.py:189-225 `initialize_from_data` to our subset ordering.

    |s| = 1 : arcsin(sqrt(<x_j>))          |s| = 2 : Cov_pm1(j,k) * sqrt(2 d / sum Cov^2)
    |s| = 0 : 0 (global phase, no effect)  else    : N(0, noise)
    """
    from iqp_exact import subsets_upto

    d = bits.shape[1]
    means = bits.mean(axis=0)
    pm1 = 2.0 * bits - 1.0
    cov = np.cov(pm1.T)
    np.fill_diagonal(cov, 0.0)
    ssq = float(np.sum(cov ** 2))
    scale = cov_scale if cov_scale is not None else (math.sqrt(2.0 * d / ssq) if ssq > 0 else 0.0)

    subs = subsets_upto(sim.n, sim.K)
    theta = np.zeros(len(subs), dtype=np.float64)
    for i, s in enumerate(subs):
        if len(s) == 1:
            theta[i] = math.asin(math.sqrt(min(max(float(means[s[0]]), 0.0), 1.0)))
        elif len(s) == 2:
            theta[i] = cov[s[0], s[1]] * scale
        elif len(s) > 2 and noise > 0:
            theta[i] = rng.normal(0.0, noise)
    return theta


def data_init_theta_pmf(sim, pmf, cov_scale, noise, rng):
    """`data_init_theta` with EXACT moments under a pmf instead of sample moments of a code
    draw (the --encoding pmf / --full-grid path: there is no code sample to take moments of).

    Same formulas as data_init_theta / Ref [38] utils.py:189-225, with
    <x_j> and Cov_pm1(j, k) computed by codec12.pmf_bit_moments (population, exact).
    """
    from iqp_exact import subsets_upto

    means, cov = codec12.pmf_bit_moments(pmf)
    d = int(sim.n)
    ssq = float(np.sum(cov ** 2))
    scale = cov_scale if cov_scale is not None else (math.sqrt(2.0 * d / ssq) if ssq > 0 else 0.0)

    subs = subsets_upto(sim.n, sim.K)
    theta = np.zeros(len(subs), dtype=np.float64)
    for i, sub in enumerate(subs):
        if len(sub) == 1:
            theta[i] = math.asin(math.sqrt(min(max(float(means[sub[0]]), 0.0), 1.0)))
        elif len(sub) == 2:
            theta[i] = cov[sub[0], sub[1]] * scale
        elif len(sub) > 2 and noise > 0:
            theta[i] = rng.normal(0.0, noise)
    return theta


def sigma3_from_codes(codes, n_qubits, n_pairs=2000, rng=None):
    """sigma3 = sqrt(median pairwise Hamming distance) over `n_pairs` random pairs of codes
    (Ref [38] three-bandwidth rule, pre-registration Design-choices table)."""
    rng = rng if rng is not None else np.random.default_rng(0)
    codes = np.asarray(codes, dtype=np.int64)
    n = codes.shape[0]
    if n < 2:
        raise ValueError("need >= 2 codes to estimate a pairwise Hamming distance")
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    keep = i != j
    i, j = i[keep], j[keep]
    xor = codes[i] ^ codes[j]
    hamming = np.zeros(xor.shape[0], dtype=np.int64)
    for b in range(n_qubits):
        hamming += (xor >> b) & 1
    med = float(np.median(hamming)) if hamming.size > 0 else 0.0
    return math.sqrt(max(med, 1e-12))


def mean_pauli_weight_for_sigma(n_qubits, sigma):
    """Mean Walsh/Pauli weight induced by bandwidth sigma (inverse of iqp_exact.sigma_for_avg_weight):
    n(1 - exp(-1/(2 sigma^2))) / 2."""
    r = math.exp(-1.0 / (2.0 * float(sigma) ** 2))
    return n_qubits * (1.0 - r) / 2.0


def resolve_sigmas(sigmas_arg, n_qubits, first_anchor_codes, seed):
    """--sigmas 'auto3' -> the Ref [38] three-bandwidth rule from the t=1 data; else parse a
    comma-separated float list. Returns (sigmas, mean_pauli_weights)."""
    if str(sigmas_arg).strip().lower() == "auto3":
        sigma1 = sigma_for_avg_weight(n_qubits, 2.0)
        sigma3 = sigma3_from_codes(first_anchor_codes, n_qubits, rng=np.random.default_rng(int(seed) + 777))
        sigma2 = math.sqrt(0.5 * (sigma1 ** 2 + sigma3 ** 2))
        sigmas = [sigma1, sigma2, sigma3]
    else:
        sigmas = [float(s) for s in str(sigmas_arg).split(",") if s.strip()]
        if not sigmas:
            raise ValueError("--sigmas must be 'auto3' or a non-empty comma-separated float list")
    weights = [mean_pauli_weight_for_sigma(n_qubits, s) for s in sigmas]
    return sigmas, weights


def interp_theta_lat(anchor_times, theta_lat_anchors, tau):
    """Piecewise-linear interpolation (paper Eq. 31) of each latent coordinate; constant outside."""
    at = np.asarray(anchor_times, dtype=np.float64)
    out = np.empty(theta_lat_anchors.shape[1], dtype=np.float64)
    for i in range(theta_lat_anchors.shape[1]):
        out[i] = np.interp(float(tau), at, theta_lat_anchors[:, i].astype(np.float64))
    return out


# --------------------------------------------------------------------------------------
def main(argv=None):
    args = parse_args(argv)
    if args.device == "cpu":
        # The FWHT butterfly (fwht()) is a chain of tiny reshape/stack ops on a (2^n,)
        # tensor; torch's default (= all cores, often 100+ on an HPC node) intra-op thread
        # pool makes sync overhead dominate real compute -- measured 22.4 s/step at n=18
        # with the default thread count vs 0.02 s/step at 4 threads (~1000x). Same cap as
        # baseline_rbm.py / baseline_dcgan.py; without it no run fits any SLURM partition.
        torch.set_num_threads(min(4, os.cpu_count() or 1))
    t0 = time.time()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    dtype = torch.complex64 if args.dtype == "complex64" else torch.complex128
    os.makedirs(args.out, exist_ok=True)

    if args.encoding == "pmf" and args.data != "lens":
        raise SystemExit("--encoding pmf is defined for --data lens only (codec12.py)")

    anchors, unseen, a_z, b_z, pool_fn, lens_extra = load_snapshots_and_split(args)
    anchors = np.asarray(anchors, dtype=np.int64)
    unseen = np.asarray(unseen, dtype=np.int64)
    if anchors.size < 2:
        raise SystemExit("need >= 2 anchors")
    print("[data] %d anchors (1-based t) %s, %d held-out tau"
          % (anchors.size, anchors.tolist(), unseen.size), flush=True)
    print("[data] a_z=%.6g b_z=%.6g (from the training anchors only)" % (a_z, b_z), flush=True)

    is_pmf = (args.encoding == "pmf")
    n_qubits = codec12.N_QUBITS if is_pmf else codec.N_QUBITS
    if args.K > n_qubits:
        raise SystemExit("--K %d exceeds the qubit count %d" % (args.K, n_qubits))
    sim = IQPExact(n=n_qubits, K=args.K, device=device, dtype=dtype)
    if args.d_lat > sim.n_params:
        raise SystemExit("--d-lat %d exceeds the parameter count %d" % (args.d_lat, sim.n_params))
    print("[model] encoding=%s n=%d K=%d params=%d d_lat=%d stage1_train_all=%s device=%s dtype=%s"
          % (args.encoding, sim.n, sim.K, sim.n_params, args.d_lat,
             bool(args.stage1_train_all), device, args.dtype), flush=True)

    exact_pmf = bool(is_pmf and args.full_grid)
    pmf_by_anchor = None
    codes_by_anchor = []
    first_bits = None
    first_codes = None
    first_pmf = None
    if exact_pmf:
        # --encoding pmf --full-grid: the target IS the exact 4,096-cell image pmf; no code
        # sample is drawn for training at all.  The fixed 5e4 pool is still used for the
        # sigma3 bandwidth estimate, so that rule is identical across the pmf arms.
        pmf_by_anchor = [lens_extra["pmf_fn"](a) for a in anchors.tolist()]
        first_pmf = pmf_by_anchor[0]
        first_codes = np.asarray(pool_fn(int(anchors[0])), dtype=np.int64)
        print("[data] built %d EXACT image pmfs (%d cells each; --full-grid, no code sampling)"
              % (len(pmf_by_anchor), pmf_by_anchor[0].size), flush=True)
    else:
        for a_i, a in enumerate(anchors.tolist()):
            codes = codes_for(pool_fn(a), args, rng)
            codes_by_anchor.append(codes)
            if a_i == 0:
                bits = ((codes[:, None] >> np.arange(n_qubits)[None, :]) & 1).astype(np.float64)
                first_bits = bits
                first_codes = codes
        print("[data] built %d code sets (%d tuples each)"
              % (len(codes_by_anchor), codes_by_anchor[0].size), flush=True)

    sigmas, mean_pauli_weights = resolve_sigmas(args.sigmas, n_qubits, first_codes, args.seed)
    print("[sigma] --sigmas=%r n=%d -> sigmas=%s mean_pauli_weights=%s"
          % (args.sigmas, n_qubits, ["%.6f" % s for s in sigmas],
             ["%.4f" % w for w in mean_pauli_weights]), flush=True)
    weights = sim.kernel_weights(sigmas)

    if exact_pmf:
        ez_data = [sim.data_expvals(q) for q in pmf_by_anchor]
    else:
        pmf_fn = codec12.codes_to_pmf if is_pmf else codec.codes_to_pmf
        ez_data = [sim.data_expvals(pmf_fn(c)) for c in codes_by_anchor]
    print("[data] built %d target expectation-value vectors" % len(ez_data), flush=True)

    # -------------------------------------------------------------- initialization
    if args.init == "data" and exact_pmf:
        theta0 = data_init_theta_pmf(sim, first_pmf, None, args.init_noise, rng)
    elif args.init == "data":
        theta0 = data_init_theta(sim, first_bits, None, args.init_noise, rng)
    else:
        theta0 = rng.normal(0.0, args.init_std, size=sim.n_params)
    if args.lat_init == "random":
        theta0[: args.d_lat] = rng.normal(0.0, args.lat_std, size=args.d_lat)
    theta0[0] = 0.0  # |s| = 0 is a global phase: exactly zero gradient, pin it to 0

    # -------------------------------------------------------------- stage 1
    # Paper protocol (default): theta_lat^(1) is FROZEN, only theta_core trains.
    # --stage1-train-all (round-2 fix): theta_lat trains too, so image 1's latent is fitted
    # rather than random -- stage 2 then warm-starts from a fitted latent instead of noise.
    train_all = bool(args.stage1_train_all)
    theta_lat = torch.tensor(theta0[: args.d_lat], dtype=sim.rdtype, device=device,
                             requires_grad=train_all)
    theta_core = torch.tensor(theta0[args.d_lat:], dtype=sim.rdtype, device=device, requires_grad=True)
    # snapshot AFTER the cast to sim.rdtype (float32 by default): comparing against the
    # float64 theta0 would report a ~1e-9 "shift" for a latent that never moved at all.
    theta_lat_init = theta_lat.detach().cpu().numpy().copy()

    opt = torch.optim.Adam(([theta_lat, theta_core] if train_all else [theta_core]), lr=args.lr)
    curve_core = np.zeros(args.steps_core, dtype=np.float64)
    for step in range(args.steps_core):
        opt.zero_grad(set_to_none=True)
        loss = sim.mmd2(torch.cat((theta_lat, theta_core)), ez_data[0], weights)
        loss.backward()
        opt.step()
        curve_core[step] = float(loss.detach())
        if step % args.log_every == 0 or step == args.steps_core - 1:
            print("[stage1] step %6d/%d  MMD^2 = %.6e  (%.1f s)%s"
                  % (step, args.steps_core, curve_core[step], time.time() - t0,
                     "  [theta_lat trained]" if train_all else ""), flush=True)
    theta_core = theta_core.detach()
    theta_lat = theta_lat.detach()
    loss1_final = float(curve_core[-1]) if args.steps_core > 0 else float("nan")
    # Report the shift over the NON-phase coordinates only. theta_lat[0] is the |s| = 0
    # subset: probs() is exactly invariant to it, so any movement there is float round-off
    # and says nothing about learning. Measured 2026-09-06 on a 2000-step --stage1-train-all
    # run: index 0 moved 3.1e-4 while max|theta_lat[1:]| reached 0.123, so the headline
    # number is NOT phase-dominated -- but it is reported separately so that stays checkable.
    _lat_now = theta_lat.cpu().numpy()
    lat_shift_stage1 = float(np.abs(_lat_now[1:] - theta_lat_init[1:]).max())
    lat_shift_phase = float(abs(_lat_now[0] - theta_lat_init[0]))
    print("[stage1] max|theta_lat[1:] - init| = %.6e  (global phase theta_lat[0] moved "
          "%.3e, ignored) (train_all=%s)" % (lat_shift_stage1, lat_shift_phase, train_all),
          flush=True)

    # -------------------------------------------------------------- stage 2: theta_lat per anchor
    lat_anchors = np.zeros((anchors.size, args.d_lat), dtype=np.float64)
    lat_anchors[0] = theta_lat.detach().cpu().numpy()
    curve_lat = np.zeros((max(anchors.size - 1, 0), args.steps_lat), dtype=np.float64)
    lat_final = []
    for j in range(1, anchors.size):
        lat = torch.tensor(lat_anchors[j - 1], dtype=sim.rdtype, device=device, requires_grad=True)
        opt = torch.optim.Adam([lat], lr=args.lr)
        for step in range(args.steps_lat):
            opt.zero_grad(set_to_none=True)
            loss = sim.mmd2(torch.cat((lat, theta_core)), ez_data[j], weights)
            loss.backward()
            opt.step()
            curve_lat[j - 1, step] = float(loss.detach())
            if step % args.log_every == 0 or step == args.steps_lat - 1:
                print("[stage2] anchor %2d (t=%d) step %5d/%d  MMD^2 = %.6e  (%.1f s)"
                      % (j, anchors[j], step, args.steps_lat, curve_lat[j - 1, step], time.time() - t0),
                      flush=True)
        lat_anchors[j] = lat.detach().cpu().numpy()
        lat_final.append(float(curve_lat[j - 1, -1]) if args.steps_lat > 0 else float("nan"))

    # -------------------------------------------------------------- synthesis
    intensity_scale = float(lens_extra["S"]) if (is_pmf and lens_extra) else None
    if is_pmf:
        print("[synth] intensity scale S = %.6f (mean total intensity of the %d training images)"
              % (intensity_scale, anchors.size), flush=True)
    gen = torch.Generator(device=device.type)
    gen.manual_seed(args.seed + 999)
    n_written = 0
    lat_gauss = None
    if args.synth_mode == "gaussian-latent":
        # No time axis (lens pilot): fit N(mu, diag sigma^2) to the N trained latents and
        # draw --n-gen latents from it, one image each (lens_synth.py docstring).
        import lens_synth

        mu, sigma = lens_synth.fit_gaussian_latent(lat_anchors)
        lat_gauss = lens_synth.latent_stats(mu, sigma)
        rng_synth = np.random.default_rng(args.seed + 4242)
        lat_draws = lens_synth.sample_gaussian_latents(mu, sigma, args.n_gen, rng_synth)
        for i in range(args.n_gen):
            lat_t = torch.tensor(lat_draws[i], dtype=sim.rdtype, device=device)
            theta_i = torch.cat((lat_t, theta_core))
            if is_pmf:
                # PRIMARY = the exact circuit pmf scaled by S (no Monte-Carlo step at all);
                # the 1e5-sample counting density goes to <out>_sampled/ as the control.
                with torch.no_grad():
                    q = sim.probs(theta_i).to(torch.float64).cpu().numpy()
                codes = sim.sample(theta_i, args.n_synth, generator=gen).cpu().numpy()
                lens_synth.write_pmf_pair(args.out, i, q, codes, intensity_scale,
                                          sqrt=bool(args.lens_sqrt))
            else:
                codes = sim.sample(theta_i, args.n_synth, generator=gen)
                lens_synth.write_pair(args.out, i, codes.cpu().numpy(), a_z, b_z,
                                      sqrt=bool(args.lens_sqrt))
            n_written += 1
        print("[synth] gaussian-latent (%s): wrote %d images to %s (and %s)"
              % (args.encoding, n_written, args.out,
                 lens_synth.sampled_dir(args.out) if is_pmf else lens_synth.last_dir(args.out)),
              flush=True)
    else:
        for tau in unseen.tolist():
            lat_np = interp_theta_lat(anchors, lat_anchors, tau)
            lat_t = torch.tensor(lat_np, dtype=sim.rdtype, device=device)
            codes = sim.sample(torch.cat((lat_t, theta_core)), args.n_synth, generator=gen)
            field = codec.aggregate(codes.cpu().numpy(), a_z, b_z)
            np.save(os.path.join(args.out, "generated_t%d.npy" % tau), field.astype(np.float32))
            n_written += 1
        print("[synth] wrote %d snapshots to %s" % (n_written, args.out), flush=True)

    # -------------------------------------------------------------- artefacts
    np.save(os.path.join(args.out, "theta_core.npy"), theta_core.cpu().numpy())
    np.save(os.path.join(args.out, "theta_lat_anchors.npy"), lat_anchors)
    np.savez_compressed(os.path.join(args.out, "loss_curves.npz"),
                        stage1=curve_core, stage2=curve_lat, anchors=anchors, unseen=unseen)
    meta = {
        "arm": "IQP-%dk" % int(round(sim.n_params / 1000.0)),
        "param_count": int(sim.n_params),
        "seed": int(args.seed),
        "train_snapshots": int(anchors.size),
        "anchors": [int(a) for a in anchors.tolist()],
        "unseen": [int(t) for t in unseen.tolist()],
        "wallclock_s": float(time.time() - t0),
        "final_loss_stage1": loss1_final,
        "final_loss_stage2": lat_final,
        "a_z": a_z,
        "b_z": b_z,
        "n_qubits": int(sim.n),
        "K": int(args.K),
        "d_lat": int(args.d_lat),
        "encoding": args.encoding,
        "stage1_train_all": bool(args.stage1_train_all),
        "full_grid": bool(args.full_grid),
        "exact_data_pmf": bool(exact_pmf),
        "stage1_latent_shift": lat_shift_stage1,
        "stage1_latent_shift_phase": lat_shift_phase,
        "sigmas": sigmas,
        "sigmas_mode": args.sigmas,
        "mean_pauli_weights": mean_pauli_weights,
        "data_seed": (synth_data.SYNTH_DATA_SEED if args.data == "synthetic" else None),
        "synth_mode": args.synth_mode,
        "config": vars(args),
    }
    if args.data == "lens":
        import data_lens

        meta["arm"] = "IQP-%dk%s" % (int(round(sim.n_params / 1000.0)),
                                     "-sqrt" if args.lens_sqrt else "")
        if is_pmf:
            meta["intensity_scale_S"] = intensity_scale
            meta["n_tuples_effective"] = (0 if exact_pmf else int(args.n_tuples))
        meta["class_names"] = data_lens.get_class_names()
        meta["lens_class"] = args.lens_class
        meta["lens_sqrt"] = bool(args.lens_sqrt)
        meta["n_train"] = int(anchors.size)
        meta["train_snapshots"] = int(anchors.size)
        meta["n_gen"] = int(args.n_gen) if args.synth_mode == "gaussian-latent" else int(unseen.size)
    if lat_gauss is not None:
        meta.update(lat_gauss)
    if args.arm_name:
        # explicit label: several round-2 arms share a parameter count, so the auto name
        # (IQP-<params>k) would collide in evaluate_lens.py's table
        meta["arm"] = str(args.arm_name)

    if args.synth_mode == "gaussian-latent":
        import lens_synth

        if is_pmf:
            lens_synth.write_metas_pmf(args.out, meta)
        else:
            lens_synth.write_metas(args.out, meta)
    else:
        with open(os.path.join(args.out, "meta.json"), "w") as fh:
            json.dump(meta, fh, indent=2, sort_keys=True)
    print("[done] wallclock %.1f s -> %s" % (meta["wallclock_s"], args.out), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
