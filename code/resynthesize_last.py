#!/usr/bin/env python3
"""Resynthesize the 12 IQP + 3 CTRL-marginal + 3 RBM D2 arms with the SAME sample count the
original outputs_d2_main/ run used (n_synth = 1e5 per tau, the same "config.n_synth"/
"gen_chains*gen_keep" every arm's own meta.json already records) but with per-cell
LAST-WRITE-WINS aggregation (agg_variants.aggregate_last) instead of codec.aggregate's MEAN.

Diagnostic finding this replays at scale (diag_scripts/q1c_agg_convention.py, ad hoc, IQP
only): the paper (2603.06440) never specifies how multiple sampled tuples landing in one
(kx, ky) grid cell should be combined. codec.aggregate takes the MEAN, which -- purely as a
side effect of averaging N_t/4096 ~ 24 i.i.d. draws per cell -- shrinks the sampled field's
per-cell variance and widens the PDF-JS gap to the ground truth (0.24 on that one diagnostic).
LAST-WRITE-WINS keeps exactly one raw sample per cell (no averaging), closing most of that gap
(0.028/0.045 there, vs the paper's reported ~0.024/0.022). This script re-evaluates that
convention swap on EVERY tuple-based arm (IQP x12, CTRL-marginal x3, RBM x3) to see whether
the paper's reported IQP-vs-baselines gap in Table 2 survives once every arm uses the same
aggregation convention -- not just IQP.

Deliberately reuses resynthesize.py's already-reviewed building blocks rather than
reimplementing them (tau_list_from_dir, write_output, refit_ctrl_marginal_pvecs) -- it does
NOT modify resynthesize.py (in active/completed use by job 342454) or codec.py (codec.aggregate
itself, the MEAN baseline, is untouched; codec.decode/dequantize_z/GRID/NCODES are reused
read-only via agg_variants).

Per-arm handling, same taxonomy as resynthesize.py:
  IQP:           reuses the stored theta_core.npy / theta_lat_anchors.npy verbatim (no
                 retraining); draws n_synth=1e5 codes per tau from IQPExact.sample() (same
                 count the original outputs_d2_main run used, per meta["config"]["n_synth"]).
  CTRL-marginal: replays controls.py's closed-form product-Bernoulli fit (deterministic,
                 pure numpy, NOT a new fit -- resynthesize.refit_ctrl_marginal_pvecs) then
                 draws n_synth=1e5 codes per tau from that fitted p_vec.
  RBM:           baseline_rbm.py never checkpoints weights, so there is nothing to
                 resynthesize from -- getting RBM codes at all requires a full PCD retrain,
                 gated behind --rbm-retrain (default OFF, same convention as resynthesize.py).
                 Uses the ORIGINAL run's own gen_chains/gen_sweeps/gen_keep (NOT the x10 bump
                 resynthesize.py's --modes sample1e6 applies) so RBM also lands on ~1e5
                 samples/tau, matching IQP/CTRL-marginal for a fair like-for-like aggregation
                 comparison.

Every arm ALSO writes its raw codes_t{tau}.npy (int32; codes fit in 18 bits, well within
int32's range) alongside generated_t{tau}.npy, so any FUTURE aggregation convention can be
re-evaluated from disk without resampling (RBM retraining) or resampling (IQP/CTRL) again.

Output layout: outputs_d2_last/<arm_dir>__last/{generated_t{tau}.npy, codes_t{tau}.npy,
meta.json}, meta.json = source arm's meta.json + resynth_mode="last", agg="last", n_synth,
source_run_dir, and "arm" set to "<original arm>+last" (so evaluate.py's Table-2 grouping
lists this resynthesis as its own row per arm).

Python 3.9 compatible (no `X | None`).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

import agg_variants
import codec
import controls
import resynthesize
import train_iqp
from iqp_exact import IQPExact

DEFAULT_N_SYNTH = 100_000  # matches every trained arm's own meta.json n_synth / n_t_tuples
SEED_OFFSET = 199_000_000  # distinct from resynthesize.py's 99_000_000 -- independent RNG stream


def write_output_with_codes(out_dir, mode, orig_meta, src_dir, tau_field_map, tau_codes_map, n_synth):
    """resynthesize.write_output, plus codes_t{tau}.npy (int32) per tau."""
    resynthesize.write_output(out_dir, mode, orig_meta, src_dir, tau_field_map, n_synth)
    for tau, codes in tau_codes_map.items():
        np.save(os.path.join(out_dir, "codes_t%d.npy" % tau), np.asarray(codes, dtype=np.int32))
    # write_output() already wrote meta.json with resynth_mode=mode/arm=...+mode; patch in
    # agg="last" so it's easy to tell apart from a future non-"last" resynth using this same
    # "mode" string, without re-deriving everything write_output already computed.
    meta_path = os.path.join(out_dir, "meta.json")
    with open(meta_path) as fh:
        meta = json.load(fh)
    meta["agg"] = "last"
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True, default=str)


# --------------------------------------------------------------------------------------
# IQP
# --------------------------------------------------------------------------------------
def process_iqp_last(src_dir, base, meta, taus, device, args):
    theta_core_np = np.load(os.path.join(src_dir, "theta_core.npy"))
    theta_lat_anchors = np.load(os.path.join(src_dir, "theta_lat_anchors.npy"))
    anchors = [int(a) for a in meta["anchors"]]
    a_z, b_z = float(meta["a_z"]), float(meta["b_z"])
    K = int(meta["K"])
    dtype = torch.complex128 if meta.get("config", {}).get("dtype") == "complex128" else torch.complex64

    sim = IQPExact(n=codec.N_QUBITS, K=K, device=device, dtype=dtype)
    theta_core_t = torch.as_tensor(theta_core_np, dtype=sim.rdtype, device=device)

    out_dir = os.path.join(args.out, "%s__last" % base)
    fields, codes_map = {}, {}
    gen = torch.Generator(device=device.type)
    gen.manual_seed(int(meta["seed"]) + SEED_OFFSET)
    for tau in taus:
        lat_np = train_iqp.interp_theta_lat(anchors, theta_lat_anchors, tau)
        lat_t = torch.as_tensor(lat_np, dtype=sim.rdtype, device=device)
        theta = sim.join(lat_t, theta_core_t)
        codes = sim.sample(theta, int(args.n_synth), generator=gen).cpu().numpy()
        fields[tau] = agg_variants.aggregate_last(codes, a_z, b_z)
        codes_map[tau] = codes

    write_output_with_codes(out_dir, "last", meta, src_dir, fields, codes_map, args.n_synth)
    print("[iqp] %s -> %s (%d tau, n_synth=%d)" % (base, out_dir, len(fields), args.n_synth), flush=True)
    return 1


# --------------------------------------------------------------------------------------
# CTRL-marginal (independent-bit product-Bernoulli; replay the closed-form fit, not a new one)
# --------------------------------------------------------------------------------------
def process_ctrl_marginal_last(src_dir, base, meta, taus, args):
    a_z, b_z = float(meta["a_z"]), float(meta["b_z"])
    anchors, p_vecs = resynthesize.refit_ctrl_marginal_pvecs(meta)

    out_dir = os.path.join(args.out, "%s__last" % base)
    fields, codes_map = {}, {}
    rng = np.random.default_rng(int(meta["seed"]) + SEED_OFFSET)
    for tau in taus:
        p_tau = controls.interp_anchors(tau, anchors, p_vecs)
        codes = controls.sample_product_bernoulli(p_tau, args.n_synth, rng)
        fields[tau] = agg_variants.aggregate_last(codes, a_z, b_z)
        codes_map[tau] = codes

    write_output_with_codes(out_dir, "last", meta, src_dir, fields, codes_map, args.n_synth)
    print("[ctrl_marginal] %s -> %s (%d tau, n_synth=%d)" % (base, out_dir, len(fields), args.n_synth), flush=True)
    return 1


# --------------------------------------------------------------------------------------
# RBM: no checkpoint was ever saved -- needs --rbm-retrain (opt-in, OFF by default). Uses the
# ORIGINAL run's own gen_chains/gen_keep (no x10 bump) so total samples/tau matches IQP/CTRL's
# n_synth=1e5, for a fair like-for-like aggregation-convention comparison.
# --------------------------------------------------------------------------------------
def process_rbm_last(src_dir, base, meta, taus, device, args):
    if not args.rbm_retrain:
        print("[skip] %s: RBM 'last' needs --rbm-retrain -- baseline_rbm.py's main() never "
              "checkpoints W/b_v/b_h/latents_arr, so there is no stored model to resynthesize "
              "from; --rbm-retrain does a full PCD retrain (genuine retrain, left OFF by "
              "default)" % src_dir, flush=True)
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

    t0_train = time.time()
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
    train_s = time.time() - t0_train

    # NOTE: no x10 bump here (unlike resynthesize.py's process_rbm sample1e6 path) -- keep the
    # ORIGINAL run's own gen_chains/gen_keep so total samples/tau == n_tuples-scale (~1e5),
    # matching IQP/CTRL-marginal's args.n_synth for this ablation.
    gen_chains = int(meta.get("gen_chains", 10000))
    gen_sweeps = int(meta.get("gen_sweeps", 200))
    gen_keep = int(meta.get("gen_keep", 10))
    n_synth_rbm = gen_chains * gen_keep

    out_dir = os.path.join(args.out, "%s__last" % base)
    fields, codes_map = {}, {}
    t0_gen = time.time()
    for tau in taus:
        theta_lat_tau = baseline_rbm.interp_latent(tau, anchors_sorted, latents_arr)
        rbm.set_latent(torch.from_numpy(theta_lat_tau.astype(np.float32)).to(dev))
        codes = baseline_rbm.generate_codes(rbm, gen_chains, gen_sweeps, gen_keep, torch_gen, dev)
        fields[tau] = agg_variants.aggregate_last(codes, a_z, b_z)
        codes_map[tau] = codes
    gen_s = time.time() - t0_gen

    write_output_with_codes(out_dir, "last", meta, src_dir, fields, codes_map, n_synth_rbm)
    print("[rbm] %s -> %s (%d tau, n_synth=%d, train=%.1fs, generate=%.1fs, "
          "%.4f ms/pcd-step)" % (base, out_dir, len(fields), n_synth_rbm, train_s, gen_s,
                                  1000.0 * train_s / (steps_core + len(anchors[1:]) * steps_lat)),
          flush=True)
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
    p.add_argument("--out", type=str, default="outputs_d2_last")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--n-holdout", type=int, default=0,
                   help="0 = every tau found in the source dir; >0 = only the first N "
                        "(sorted) -- smoke tests")
    p.add_argument("--n-synth", type=int, default=DEFAULT_N_SYNTH,
                   help="samples/tau for IQP and CTRL-marginal (RBM uses its own recorded "
                        "gen_chains*gen_keep instead, see process_rbm_last)")
    p.add_argument("--rbm-retrain", action="store_true",
                   help="opt-in: full PCD retrain to get RBM codes (no checkpoint exists to "
                        "resynthesize from otherwise); OFF by default")
    return p.parse_args(argv)


def main(argv=None):
    import glob
    args = parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(min(4, os.cpu_count() or 1))
    t0 = time.time()

    patterns = [s.strip() for s in args.arms_glob.split(",") if s.strip()]
    src_dirs = sorted(set(d for pat in patterns for d in glob.glob(pat) if os.path.isdir(d)))
    os.makedirs(args.out, exist_ok=True)

    n_written = 0
    for src_dir in src_dirs:
        base = os.path.basename(os.path.normpath(src_dir))
        meta_path = os.path.join(src_dir, "meta.json")
        if not os.path.isfile(meta_path):
            print("[skip] %s: no meta.json" % src_dir, flush=True)
            continue
        with open(meta_path) as fh:
            meta = json.load(fh)
        taus = resynthesize.tau_list_from_dir(src_dir)
        if args.n_holdout > 0:
            taus = taus[: args.n_holdout]
        if not taus:
            print("[skip] %s: no generated_t*.npy found" % src_dir, flush=True)
            continue

        if base.startswith("iqp_"):
            n_written += process_iqp_last(src_dir, base, meta, taus, device, args)
        elif base.startswith("ctrl_marginal_"):
            n_written += process_ctrl_marginal_last(src_dir, base, meta, taus, args)
        elif base.startswith("rbm_"):
            n_written += process_rbm_last(src_dir, base, meta, taus, device, args)
        else:
            print("[skip] %s: not an IQP/RBM/CTRL-marginal arm (out of scope for this "
                  "resynthesis)" % src_dir, flush=True)

    print("[done] wrote %d arm output dirs to %s (%.1f s)" % (n_written, args.out, time.time() - t0),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
