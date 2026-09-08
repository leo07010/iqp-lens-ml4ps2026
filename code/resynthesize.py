#!/usr/bin/env python3
"""Resynthesize generated fields from ALREADY-TRAINED parameters at two aggregation modes
that avoid the sampling noise of the original --n-synth 1e5 draw-and-bin pipeline (paper
2603.06440 only says "aggregate according to the same probability-aggregation/decoding
pipeline" and gives a sample-count range of 1e5-1e6; job 342220 used the low end of that
range with only ~24 samples/cell, which is a plausible source of the observed PDF-JS gap
vs the paper's reported 0.02):

  exact      IQP:           E[z | kx, ky] from the EXACT circuit probabilities
                             q_theta(x) = |<x|U(theta)|0^n>|^2 (iqp_exact.IQPExact.probs),
                             zero sampling noise.
             CTRL-marginal: bits are independent, so E[z | kx, ky] does not depend on
                             (kx, ky) at all -- the exact field is a CONSTANT, which is
                             the correct/deserved result for a model with no spatial
                             structure.
             RBM:           not supported (no tractable closed-form marginal for a
                             trained RBM -- would need summing over 2^612 hidden states).
  sample1e6  same draw-and-bin pipeline as training (codec.aggregate), but with 1e6
             samples -- the paper's upper range -- instead of the original 1e5.

This script does NOT retrain any model:
  - IQP: reuses the stored theta_core.npy / theta_lat_anchors.npy verbatim, via
    train_iqp.interp_theta_lat (imported, not reimplemented) for the latent interpolation
    and iqp_exact.IQPExact (imported) for probs()/sample().
  - CTRL-marginal: controls.py never wrote its fitted product-Bernoulli p_vecs to disk, but
    the fit (controls.fit_marginal_params, imported) is a closed-form, numpy-only, seeded
    statistic (one np.random.default_rng(seed) call per anchor, in anchor order) -- NOT a
    gradient-descent training loop -- so replaying it with the same (seed, anchors, a_z, b_z,
    n_t_tuples) recorded in the source run's meta.json reproduces the identical p_vecs the
    original run used, bit-for-bit (pure numpy, no GPU nondeterminism). This is a recompute
    of a cached statistic, not a new experiment.
  - RBM: baseline_rbm.py's main() never checkpoints W/b_v/b_h or the per-anchor latents_arr,
    so there is genuinely nothing to "resynthesize from" -- getting RBM sample1e6 requires a
    full PCD retrain. That IS retraining (and, since baseline_rbm.py trains on CUDA without
    torch.use_deterministic_algorithms, not even guaranteed to reproduce the original run's
    exact weights), so it is gated behind --rbm-retrain (default OFF); without the flag, RBM
    arms are skipped with a printed warning.

Output layout: outputs_d2_resynth/<arm_dir>__<mode>/{generated_t{tau}.npy, meta.json}, where
meta.json is the source arm's meta.json with resynth_mode / n_synth / source_run_dir added
and "arm" set to "<original arm>+<mode>" (so evaluate.py's Table-2 grouping lists each
resynthesis mode as its own row).

Python 3.9 compatible (no `X | None`).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import time

import numpy as np
import torch

import codec
import controls
import train_iqp
from iqp_exact import IQPExact

_TAU_RE = re.compile(r"generated_t(-?\d+)\.npy$")


# --------------------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------------------
def tau_list_from_dir(src_dir):
    """The exact set of held-out tau this arm actually wrote, read off its own
    generated_t*.npy filenames (robust to meta.json schema differences across arms --
    IQP/RBM store an explicit tau list, CTRL-marginal does not)."""
    taus = []
    for path in glob.glob(os.path.join(src_dir, "generated_t*.npy")):
        m = _TAU_RE.match(os.path.basename(path))
        if m:
            taus.append(int(m.group(1)))
    return sorted(taus)


def rms(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def write_output(out_dir, mode, orig_meta, src_dir, tau_field_map, n_synth):
    os.makedirs(out_dir, exist_ok=True)
    for tau, field in tau_field_map.items():
        np.save(os.path.join(out_dir, "generated_t%d.npy" % tau), np.asarray(field, dtype=np.float32))
    meta = dict(orig_meta)
    meta["resynth_mode"] = mode
    meta["n_synth"] = int(n_synth) if n_synth else 0
    meta["source_run_dir"] = os.path.abspath(src_dir)
    meta["arm"] = "%s+%s" % (orig_meta.get("arm", os.path.basename(os.path.normpath(src_dir))), mode)
    with open(os.path.join(out_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True, default=str)


# --------------------------------------------------------------------------------------
# IQP: exact conditional expectation from q_theta, via the code layout in codec.py
# (code = (kx << 12) | (ky << 6) | kz, i.e. flat index = kx*4096 + ky*64 + kz, which is
# EXACTLY C-order reshape(64, 64, 64) -> axes (kx, ky, kz): no permutation needed there).
# codec.py's field convention is field[y, x] i.e. field[ky, kx], so the (kx, ky) result
# is transposed before returning.
# --------------------------------------------------------------------------------------
def field_from_probs(q, a_z, b_z):
    """q: length-2^18 array of (possibly unnormalized) nonnegative weights over 18-bit
    codes. Returns the float64 (64, 64) field of E[z | kx, ky] = sum_kz z(kz) q(kx,ky,kz) /
    sum_kz q(kx,ky,kz), field[y, x] convention (matches codec.aggregate)."""
    q = np.asarray(q, dtype=np.float64)
    if q.size != codec.NCODES:
        raise ValueError("field_from_probs: expected %d entries, got %d" % (codec.NCODES, q.size))
    q3 = q.reshape(codec.GRID, codec.GRID, codec.GRID)  # (kx, ky, kz), see module docstring
    z_vals = codec.dequantize_z(np.arange(codec.GRID), a_z, b_z)  # (64,)
    num = (q3 * z_vals[None, None, :]).sum(axis=2)  # (kx, ky)
    den = q3.sum(axis=2)  # (kx, ky)
    gmean = 0.5 * (a_z + b_z)
    den_pos = np.where(den > 0, den, 1.0)
    field_kxky = np.where(den > 1e-12, num / den_pos, gmean)
    return field_kxky.T  # (ky, kx) = (y, x)


def iqp_exact_field(sim, theta, a_z, b_z):
    with torch.no_grad():
        q = sim.probs(theta).detach().cpu().numpy().astype(np.float64)
    return field_from_probs(q, a_z, b_z)


def iqp_sample_field(sim, theta, a_z, b_z, n_samples, generator):
    codes = sim.sample(theta, int(n_samples), generator=generator)
    return codec.aggregate(codes.cpu().numpy(), a_z, b_z)


def process_iqp(src_dir, base, meta, taus, modes, device, args):
    theta_core_np = np.load(os.path.join(src_dir, "theta_core.npy"))
    theta_lat_anchors = np.load(os.path.join(src_dir, "theta_lat_anchors.npy"))
    anchors = [int(a) for a in meta["anchors"]]
    a_z, b_z = float(meta["a_z"]), float(meta["b_z"])
    K = int(meta["K"])
    dtype = torch.complex128 if meta.get("config", {}).get("dtype") == "complex128" else torch.complex64

    sim = IQPExact(n=codec.N_QUBITS, K=K, device=device, dtype=dtype)
    theta_core_t = torch.as_tensor(theta_core_np, dtype=sim.rdtype, device=device)

    n_written = 0
    for mode in modes:
        if mode not in ("exact", "sample1e6"):
            continue
        out_dir = os.path.join(args.out, "%s__%s" % (base, mode))
        fields = {}
        gen = torch.Generator(device=device.type)
        gen.manual_seed(int(meta["seed"]) + args.seed_offset)
        for tau in taus:
            lat_np = train_iqp.interp_theta_lat(anchors, theta_lat_anchors, tau)
            lat_t = torch.as_tensor(lat_np, dtype=sim.rdtype, device=device)
            theta = sim.join(lat_t, theta_core_t)
            if mode == "exact":
                fields[tau] = iqp_exact_field(sim, theta, a_z, b_z)
            else:
                fields[tau] = iqp_sample_field(sim, theta, a_z, b_z, args.sample_n, gen)
        n_synth = 0 if mode == "exact" else args.sample_n
        write_output(out_dir, mode, meta, src_dir, fields, n_synth)
        print("[iqp] %s mode=%s -> %s (%d tau)" % (base, mode, out_dir, len(fields)), flush=True)
        n_written += 1
    return n_written


# --------------------------------------------------------------------------------------
# CTRL-marginal: independent-bit product-Bernoulli. Exact field is a CONSTANT (bits are
# independent, so the kz-bit distribution does not depend on kx/ky at all).
# --------------------------------------------------------------------------------------
def ctrl_marginal_exact_field(p_vec, a_z, b_z):
    """p_vec: (18,) per-bit Bernoulli params, bit j = qubit j = bit j of the code (codec.py),
    so bits 0-5 ARE the kz bits directly. Returns a (64, 64) array filled with the single
    constant E[z] (no dependence on kx, ky -- the correct result for an independent-bit
    model)."""
    p_kz = np.clip(np.asarray(p_vec, dtype=np.float64)[:6], 0.0, 1.0)
    kz = np.arange(64)
    bit_table = ((kz[:, None] >> np.arange(6)[None, :]) & 1).astype(np.float64)  # (64, 6)
    p_bit_given = bit_table * p_kz[None, :] + (1 - bit_table) * (1 - p_kz[None, :])
    p_kz_val = p_bit_given.prod(axis=1)  # (64,)
    z_vals = codec.dequantize_z(kz, a_z, b_z)
    const_val = float(np.sum(p_kz_val * z_vals))
    return np.full((codec.GRID, codec.GRID), const_val, dtype=np.float64)


def refit_ctrl_marginal_pvecs(meta):
    """Replay controls.py's exact fit sequence (same seed, same anchor order, same
    n_t_tuples, one rng.integers call per anchor via controls.fit_marginal_params) against
    data_d2.load_d2()'s snapshots -- reproduces the ORIGINAL run's p_vecs bit-for-bit
    (pure numpy, no GPU nondeterminism), not a new fit."""
    from data_d2 import load_d2  # local: only needed for this arm type

    seed = int(meta["seed"])
    anchors = [int(a) for a in meta["anchors"]]
    a_z, b_z = float(meta["a_z"]), float(meta["b_z"])
    n_tuples = int(meta.get("n_t_tuples", 50000))

    snapshots, _all_anchors, _holdout = load_d2()
    rng = np.random.default_rng(seed)
    p_vecs = []
    for t in anchors:
        field = np.asarray(snapshots[t].field, dtype=np.float32)
        p_vecs.append(controls.fit_marginal_params(field, a_z, b_z, rng, n_tuples=n_tuples))
    return anchors, np.stack(p_vecs, axis=0)


def process_ctrl_marginal(src_dir, base, meta, taus, modes, args):
    a_z, b_z = float(meta["a_z"]), float(meta["b_z"])
    anchors, p_vecs = refit_ctrl_marginal_pvecs(meta)

    n_written = 0
    for mode in modes:
        if mode not in ("exact", "sample1e6"):
            continue
        out_dir = os.path.join(args.out, "%s__%s" % (base, mode))
        fields = {}
        rng = np.random.default_rng(int(meta["seed"]) + args.seed_offset)
        for tau in taus:
            p_tau = controls.interp_anchors(tau, anchors, p_vecs)
            if mode == "exact":
                fields[tau] = ctrl_marginal_exact_field(p_tau, a_z, b_z)
            else:
                codes = controls.sample_product_bernoulli(p_tau, args.sample_n, rng)
                fields[tau] = codec.aggregate(codes, a_z, b_z)
        n_synth = 0 if mode == "exact" else args.sample_n
        write_output(out_dir, mode, meta, src_dir, fields, n_synth)
        print("[ctrl_marginal] %s mode=%s -> %s (%d tau)" % (base, mode, out_dir, len(fields)), flush=True)
        n_written += 1
    return n_written


# --------------------------------------------------------------------------------------
# RBM: no checkpoint was ever saved -- sample1e6 requires --rbm-retrain (opt-in, OFF by
# default). "exact" is not supported at all (no tractable closed-form RBM marginal).
# --------------------------------------------------------------------------------------
def process_rbm(src_dir, base, meta, taus, modes, device, args):
    if "exact" in modes:
        print("[skip] %s: RBM has no tractable exact marginal (would need summing over "
              "2^n_hidden hidden states) -- exact mode not supported for RBM" % src_dir, flush=True)
    if "sample1e6" not in modes:
        return 0
    if not args.rbm_retrain:
        print("[skip] %s: RBM sample1e6 needs --rbm-retrain -- baseline_rbm.py's main() "
              "never checkpoints W/b_v/b_h/latents_arr, so there is no stored model to "
              "resynthesize from; --rbm-retrain does a full PCD retrain (genuine retrain, "
              "left OFF by default)" % src_dir, flush=True)
        return 0

    import baseline_rbm

    seed = int(meta["seed"])
    data_mode = meta.get("data", "d2")
    hidden = int(meta.get("hidden", 630))
    n_tuples = int(meta.get("n_tuples", 50000))
    steps_core = int(meta.get("steps_core", 30000))
    steps_lat = int(meta.get("steps_lat", 3000))
    lr = float(meta.get("lr", 1e-3))
    n_chains_pcd = int(meta.get("n_chains_pcd", 100))
    batch = 512
    a_z, b_z = float(meta["a_z"]), float(meta["b_z"])
    anchors = [int(t) for t in meta["train_snapshots"]]
    dev = device.type

    rng_np = np.random.default_rng(seed)
    torch_gen = torch.Generator(device=dev).manual_seed(seed)
    get_field, get_pool = baseline_rbm.load_data_accessors(data_mode, seed)
    rbm = baseline_rbm.RBM(n_visible=baseline_rbm.N_VISIBLE, n_hidden=hidden, device=dev, seed=seed)

    t0 = anchors[0]
    codes0 = codec.sample_codes(get_pool(t0, a_z, b_z), n_tuples, np.random.default_rng(seed))
    baseline_rbm.pcd_train(rbm, codes0, n_steps=steps_core, lr=lr, batch=min(batch, n_tuples),
                            n_chains=n_chains_pcd, latent_only=False, rng_np=rng_np,
                            torch_gen=torch_gen, device=dev)
    latent_by_t = {t0: rbm.get_latent().cpu().numpy()}
    prev_latent = rbm.get_latent()
    for t in anchors[1:]:
        codes_t = codec.sample_codes(get_pool(t, a_z, b_z), n_tuples, np.random.default_rng(seed + t))
        rbm.set_latent(prev_latent)
        baseline_rbm.pcd_train(rbm, codes_t, n_steps=steps_lat, lr=lr, batch=min(batch, n_tuples),
                                n_chains=n_chains_pcd, latent_only=True, rng_np=rng_np,
                                torch_gen=torch_gen, device=dev)
        prev_latent = rbm.get_latent()
        latent_by_t[t] = prev_latent.cpu().numpy()
    anchors_sorted = sorted(latent_by_t.keys())
    latents_arr = np.stack([latent_by_t[a] for a in anchors_sorted])

    gen_chains = int(meta.get("gen_chains", 10000)) * 10  # x10 chains -> x10 samples (task spec)
    gen_sweeps = int(meta.get("gen_sweeps", 200))
    gen_keep = int(meta.get("gen_keep", 10))

    out_dir = os.path.join(args.out, "%s__sample1e6" % base)
    fields = {}
    for tau in taus:
        theta_lat_tau = baseline_rbm.interp_latent(tau, anchors_sorted, latents_arr)
        rbm.set_latent(torch.from_numpy(theta_lat_tau.astype(np.float32)).to(dev))
        codes = baseline_rbm.generate_codes(rbm, gen_chains, gen_sweeps, gen_keep, torch_gen, dev)
        fields[tau] = codec.aggregate(codes, a_z, b_z)
    write_output(out_dir, "sample1e6", meta, src_dir, fields, gen_chains * gen_keep)
    print("[rbm] %s mode=sample1e6 (retrained) -> %s (%d tau, %d samples/tau)"
          % (base, out_dir, len(fields), gen_chains * gen_keep), flush=True)
    return 1


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arms-glob", type=str,
                   default="outputs_d2_main/iqp_*,outputs_d2_main/rbm_*,outputs_d2_main/ctrl_marginal_*",
                   help="comma-separated glob patterns; matched dirs are classified by basename "
                        "prefix (iqp_/rbm_/ctrl_marginal_), anything else is skipped")
    p.add_argument("--modes", type=str, default="exact,sample1e6")
    p.add_argument("--out", type=str, default="outputs_d2_resynth")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--n-holdout", type=int, default=0,
                   help="0 = every tau found in the source dir; >0 = only the first N "
                        "(sorted) -- smoke tests")
    p.add_argument("--sample-n", type=int, default=1_000_000)
    p.add_argument("--rbm-retrain", action="store_true",
                   help="opt-in: full PCD retrain to get RBM sample1e6 (no checkpoint exists "
                        "to resynthesize from otherwise); OFF by default")
    p.add_argument("--seed-offset", type=int, default=99_000_000,
                   help="added to the source arm's seed for this run's own sampling RNG "
                        "(kept well clear of any seed offsets used elsewhere)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(min(4, os.cpu_count() or 1))
    t0 = time.time()

    patterns = [s.strip() for s in args.arms_glob.split(",") if s.strip()]
    modes = [s.strip() for s in args.modes.split(",") if s.strip()]
    src_dirs = sorted(set(d for pat in patterns for d in glob.glob(pat) if os.path.isdir(d)))
    os.makedirs(args.out, exist_ok=True)

    n_arm_modes = 0
    for src_dir in src_dirs:
        base = os.path.basename(os.path.normpath(src_dir))
        meta_path = os.path.join(src_dir, "meta.json")
        if not os.path.isfile(meta_path):
            print("[skip] %s: no meta.json" % src_dir, flush=True)
            continue
        with open(meta_path) as fh:
            meta = json.load(fh)
        taus = tau_list_from_dir(src_dir)
        if args.n_holdout > 0:
            taus = taus[: args.n_holdout]
        if not taus:
            print("[skip] %s: no generated_t*.npy found" % src_dir, flush=True)
            continue

        if base.startswith("iqp_"):
            n_arm_modes += process_iqp(src_dir, base, meta, taus, modes, device, args)
        elif base.startswith("ctrl_marginal_"):
            n_arm_modes += process_ctrl_marginal(src_dir, base, meta, taus, modes, args)
        elif base.startswith("rbm_"):
            n_arm_modes += process_rbm(src_dir, base, meta, taus, modes, device, args)
        else:
            print("[skip] %s: not an IQP/RBM/CTRL-marginal arm (out of scope for this "
                  "resynthesis)" % src_dir, flush=True)

    print("[done] wrote %d arm x mode output dirs to %s (%.1f s)" % (n_arm_modes, args.out, time.time() - t0),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
