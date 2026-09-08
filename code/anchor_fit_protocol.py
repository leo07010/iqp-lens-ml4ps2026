"""Attribute the paper-protocol anchor-fit gap (corr~0.69, main run outputs_d2_main/
iqp_K6_seed0) to one of two candidate causes, by isolating each on top of the
anchor_fit_experiment.py "auto3" reference (all params trainable, full pool pmf,
corr~0.875-0.876 at 15k-85k steps -- see outputs_anchor_fit/auto3/curve.jsonl):

  (A) theta_lat^(1) FROZEN at random N(0, 0.01) init (Algorithm 1 stage 1 only trains
      theta_core; theta_lat^(1) never moves for the first anchor).
  (B) the training pmf is a 5e4-tuple SUBSAMPLE of the anchor's ~1e5-point native code
      pool (train_iqp.py --n-tuples 50000, the main run's default), not the full pool's
      exact empirical pmf that anchor_fit_experiment.py trains against.

Four configs (all sigma_mode=auto3, K=6, Adam lr=1e-4, 30000 steps by default),
built from the SAME two ingredients (freeze_lat, subsample) so each config isolates
one or both:

  paper_stage1                 : freeze_lat=True,  subsample=False  -- (A) alone
  paper_stage1_plus_stage2     : paper_stage1, THEN unfreeze theta_lat / freeze
                                  theta_core and fit theta_lat to this SAME anchor for
                                  --steps-lat steps (paper stage 2's per-anchor latent
                                  fit, applied to the anchor itself, which the paper
                                  protocol never actually does for anchor 1) -- the
                                  ceiling of "let stage 2 also touch this anchor"
  allparams_subsample5e4       : freeze_lat=False, subsample=True   -- (B) alone
  paper_stage1_subsample5e4    : freeze_lat=True,  subsample=True   -- (A)+(B), the
                                  EXACT stage-1 config the main run uses for anchor 1
                                  (same seed => same 5e4 subsample, same random lat
                                  init) -- should reproduce the main run's anchor-t=1
                                  corr~0.69 (--summarize also recomputes that number
                                  directly from outputs_d2_main/iqp_K6_seed0's saved
                                  theta_core.npy / theta_lat_anchors.npy for reference).

Reuses anchor_fit_experiment.py's / train_iqp.py's own functions (data loading, kernel
weights, data-dependent init, exact evaluation) -- does NOT modify either file (a job
is running against anchor_fit_experiment.py while this is written).

Outputs per config: outputs_anchor_fit_protocol/<config>/{curve.jsonl, meta.json,
theta_final.npy, field_exact.npy, field_meanagg.npy}. `--summarize` prints the
comparison table plus two reference rows.

Python 3.9 compatible (no `X | None`, no match statements).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

import codec
from iqp_exact import IQPExact, popcount_table
from metrics import pdf_js
from train_iqp import data_init_theta

from anchor_fit_experiment import (
    ANCHOR_T,
    build_weights,
    exact_conditional_field,
    load_anchor,
    relerr_by_order,
    sampled_field,
)

REPO = os.path.dirname(os.path.abspath(__file__))
OUT_ROOT_DEFAULT = os.path.join(REPO, "outputs_anchor_fit_protocol")
MAIN_RUN_DIR_DEFAULT = os.path.join(REPO, "outputs_d2_main", "iqp_K6_seed0")
AUTO3_REF_DIR_DEFAULT = os.path.join(REPO, "outputs_anchor_fit", "auto3")

D_LAT = 50
SIGMA_MODE = "auto3"
K = 6
LR = 1e-4
STEPS_DEFAULT = 30000
STEPS_LAT_DEFAULT = 3000
LAT_STD = 0.01
N_SUBSAMPLE = 50000
STAGE2_EVAL_EVERY = 500

# config -> (freeze_lat, subsample, stage2)
CONFIGS = {
    "paper_stage1":              dict(freeze_lat=True,  subsample=False, stage2=False),
    "paper_stage1_plus_stage2":  dict(freeze_lat=True,  subsample=False, stage2=True),
    "allparams_subsample5e4":    dict(freeze_lat=False, subsample=True,  stage2=False),
    "paper_stage1_subsample5e4": dict(freeze_lat=True,  subsample=True,  stage2=False),
}


# --------------------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", choices=sorted(CONFIGS), default=None)
    p.add_argument("--steps", type=int, default=None, help="override steps (stage 1 / only stage)")
    p.add_argument("--steps-lat", type=int, default=None, help="override stage-2 steps (paper_stage1_plus_stage2 only)")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=5000)
    p.add_argument("--out-root", type=str, default=OUT_ROOT_DEFAULT)
    p.add_argument("--plane", type=str, default="mid")
    p.add_argument("--n-synth-eval", type=int, default=100000)
    p.add_argument("--main-run-dir", type=str, default=MAIN_RUN_DIR_DEFAULT)
    p.add_argument("--auto3-ref-dir", type=str, default=AUTO3_REF_DIR_DEFAULT)
    p.add_argument("--summarize", action="store_true")
    args = p.parse_args(argv)
    if not args.summarize and args.config is None:
        p.error("--config is required unless --summarize is given")
    return args


# --------------------------------------------------------------------------------------
def build_pmf_and_bits(pool_codes, subsample, seed):
    """subsample=False -> exact empirical pmf of the full native pool (anchor_fit_experiment's
    reference recipe). subsample=True -> draw N_SUBSAMPLE codes WITH replacement via
    codec.sample_codes from a rng freshly seeded with `seed` -- bit-for-bit the same draw
    train_iqp.py makes for the FIRST anchor (codes_for(pool_fn(anchors[0]), args, rng) is
    the first thing pulled from a rng seeded with --seed in the main run).
    Returns (p_hat, bits, codes_used) -- codes_used feeds build_weights' sigma3 estimate,
    matching whichever data (full pool or subsample) is actually being fit."""
    if subsample:
        rng = np.random.default_rng(seed)
        codes = codec.sample_codes(pool_codes, N_SUBSAMPLE, rng)
    else:
        codes = np.asarray(pool_codes, dtype=np.int64)
    p_hat = codec.codes_to_pmf(codes)
    bits = ((codes[:, None] >> np.arange(codec.N_QUBITS)[None, :]) & 1).astype(np.float64)
    return p_hat, bits, codes


def init_theta0(sim, bits, rng, randomize_lat):
    """Data-dependent init over ALL subsets (train_iqp.data_init_theta, noise=0 default),
    optionally overriding the first D_LAT entries with the paper's random theta_lat^(1)
    ~ N(0, LAT_STD) (train_iqp.py --lat-init random --lat-std 0.01, its own default)."""
    theta0 = data_init_theta(sim, bits, None, 0.0, rng)
    if randomize_lat:
        theta0[:D_LAT] = rng.normal(0.0, LAT_STD, size=D_LAT)
    theta0[0] = 0.0  # |s| = 0 is a global phase: exactly zero gradient, pin it to 0
    return theta0


def eval_row(sim, theta, ez_data_np, zlut, pc, gt_field, a_z, b_z, n_synth_eval, gen, step, loss_val, t0):
    ez_model = sim.expvals(theta).double().cpu().numpy()
    ez_field = exact_conditional_field(sim, theta, zlut)
    corr = float(np.corrcoef(ez_field.ravel(), gt_field.ravel())[0, 1])
    mfield = sampled_field(sim, theta, a_z, b_z, n_synth_eval, gen)
    js = float(pdf_js(gt_field, mfield))
    relerr = relerr_by_order(ez_model, ez_data_np, pc)
    row = dict(step=step, mmd2=loss_val, corr=corr, pdf_js=js, relerr=relerr,
               wallclock_s=time.time() - t0)
    return row, ez_field, mfield


def train_stage(sim, theta_lat, theta_core, weights, ez_data, opt, steps, eval_every,
                 ez_data_np, zlut, pc, gt_field, a_z, b_z, n_synth_eval, gen, curve_fh, t0,
                 step_offset, tag):
    """One optimization stage. `theta_lat`/`theta_core` are two tensors concatenated into
    the full parameter vector every step; whichever one(s) are in `opt`'s param list (i.e.
    have requires_grad=True) get updated -- this one loop covers "freeze lat, train core"
    (paper_stage1*), "freeze core, train lat" (stage 2), and "train both" (allparams_*)."""
    last = None
    for local_step in range(steps):
        opt.zero_grad(set_to_none=True)
        theta = torch.cat((theta_lat, theta_core))
        loss = sim.mmd2(theta, ez_data, weights)
        loss.backward()
        opt.step()
        step = step_offset + local_step
        if local_step % eval_every == 0 or local_step == steps - 1:
            theta_d = torch.cat((theta_lat, theta_core)).detach()
            row, ez_field, mfield = eval_row(sim, theta_d, ez_data_np, zlut, pc, gt_field,
                                              a_z, b_z, n_synth_eval, gen, step,
                                              float(loss.detach()), t0)
            curve_fh.write(json.dumps(row) + "\n")
            curve_fh.flush()
            last = (ez_field, mfield, row)
            print("[%s step %6d/%d] MMD2=%.6e corr=%.4f pdf_js=%.4f (%.1fs)"
                  % (tag, local_step, steps, row["mmd2"], row["corr"], row["pdf_js"],
                     time.time() - t0), flush=True)
    return last


# --------------------------------------------------------------------------------------
def run_config(args):
    cfg = CONFIGS[args.config]
    steps = args.steps if args.steps is not None else STEPS_DEFAULT
    steps_lat = args.steps_lat if args.steps_lat is not None else STEPS_LAT_DEFAULT
    device = torch.device(args.device)
    if device.type == "cpu":
        # see train_iqp.py / anchor_fit_experiment.py's identical guard: the FWHT at
        # n=18 is dominated by sync overhead under torch's default (often 100+) intra-op
        # thread pool -- ~1000x slower without this cap.
        torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    out_dir = os.path.join(args.out_root, args.config)
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    snap, a_z, b_z = load_anchor(args.plane)
    gt_field = snap.field.astype(np.float32)
    pool_codes = np.asarray(snap.pool_codes, dtype=np.int64)
    print("[data] anchor t=%d pool_codes=%d a_z=%.6g b_z=%.6g (%.1fs)"
          % (ANCHOR_T, pool_codes.size, a_z, b_z, time.time() - t0), flush=True)

    p_hat, bits, codes_used = build_pmf_and_bits(pool_codes, cfg["subsample"], args.seed)
    print("[data] config=%s freeze_lat=%s subsample=%s train_codes=%d"
          % (args.config, cfg["freeze_lat"], cfg["subsample"], codes_used.size), flush=True)

    sim = IQPExact(n=codec.N_QUBITS, K=K, device=device, dtype=torch.complex64)
    weights, sigmas = build_weights(sim, SIGMA_MODE, codes_used, args.seed)
    print("[model] config=%s K=%d params=%d sigma_mode=%s sigmas=%s lr=%g steps=%d device=%s"
          % (args.config, K, sim.n_params, SIGMA_MODE,
             (["%.6f" % s for s in sigmas] if sigmas else None), LR, steps, device), flush=True)

    ez_data = sim.data_expvals(p_hat)
    ez_data_np = ez_data.double().cpu().numpy()
    zlut = codec.dequantize_z(np.arange(64), a_z, b_z)
    pc = popcount_table(codec.N_QUBITS)

    theta0 = init_theta0(sim, bits, rng, randomize_lat=cfg["freeze_lat"])
    theta_lat = torch.tensor(theta0[:D_LAT], dtype=sim.rdtype, device=device,
                              requires_grad=(not cfg["freeze_lat"]))
    theta_core = torch.tensor(theta0[D_LAT:], dtype=sim.rdtype, device=device, requires_grad=True)

    gen = torch.Generator(device=device.type)
    gen.manual_seed(args.seed + 999)

    curve_path = os.path.join(out_dir, "curve.jsonl")
    curve_fh = open(curve_path, "w")

    params_stage1 = [theta_core] if cfg["freeze_lat"] else [theta_lat, theta_core]
    opt1 = torch.optim.Adam(params_stage1, lr=LR)
    last1 = train_stage(sim, theta_lat, theta_core, weights, ez_data, opt1, steps,
                         args.eval_every, ez_data_np, zlut, pc, gt_field, a_z, b_z,
                         args.n_synth_eval, gen, curve_fh, t0, step_offset=0, tag="stage1")

    stage2_row = None
    if cfg["stage2"]:
        theta_core_fixed = theta_core.detach().clone()
        theta_lat_free = theta_lat.detach().clone().requires_grad_(True)
        opt2 = torch.optim.Adam([theta_lat_free], lr=LR)
        last2 = train_stage(sim, theta_lat_free, theta_core_fixed, weights, ez_data, opt2,
                             steps_lat, min(STAGE2_EVAL_EVERY, max(steps_lat, 1)),
                             ez_data_np, zlut, pc, gt_field, a_z, b_z, args.n_synth_eval,
                             gen, curve_fh, t0, step_offset=steps, tag="stage2")
        theta_lat, theta_core = theta_lat_free, theta_core_fixed
        stage2_row = last2[2]

    curve_fh.close()
    ez_field, mfield, stage1_row = last1
    if stage2_row is not None:
        ez_field, mfield, final_row = last2[0], last2[1], stage2_row
    else:
        final_row = stage1_row

    theta_final = torch.cat((theta_lat, theta_core)).detach().cpu().numpy()
    np.save(os.path.join(out_dir, "theta_final.npy"), theta_final)
    np.save(os.path.join(out_dir, "field_exact.npy"), ez_field)
    np.save(os.path.join(out_dir, "field_meanagg.npy"), mfield)

    meta = dict(config=args.config, K=K, lr=LR, sigma_mode=SIGMA_MODE, sigmas=sigmas,
                steps=steps, seed=args.seed, anchor_t=ANCHOR_T, plane=args.plane,
                freeze_lat=cfg["freeze_lat"], subsample=cfg["subsample"],
                n_subsample=(N_SUBSAMPLE if cfg["subsample"] else None),
                wallclock_s=time.time() - t0, stage1_final=stage1_row, final=final_row)
    if cfg["stage2"]:
        meta["steps_lat"] = steps_lat
        meta["stage2_final"] = stage2_row
        meta["corr_before_stage2"] = stage1_row["corr"]
        meta["corr_after_stage2"] = stage2_row["corr"]
    with open(os.path.join(out_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True, default=float)
    print("[done] %s -> %s (%.1fs)" % (args.config, out_dir, time.time() - t0), flush=True)


# --------------------------------------------------------------------------------------
def read_final_row(out_dir):
    """meta.json's 'final' row if present, else the last line of curve.jsonl (so summarize
    still works against a config that's mid-flight / not yet written meta.json)."""
    meta_path = os.path.join(out_dir, "meta.json")
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path))
        return meta.get("final"), meta
    curve_path = os.path.join(out_dir, "curve.jsonl")
    if os.path.exists(curve_path):
        last_line = None
        with open(curve_path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    last_line = line
        if last_line is not None:
            return json.loads(last_line), None
    return None, None


def main_run_reference(args, gt_field, a_z, b_z):
    """Recompute the main run's (outputs_d2_main/iqp_K6_seed0) anchor t=1 corr/pdf_js
    directly from its saved theta_core.npy / theta_lat_anchors.npy[0] (the paper's exact
    stage-1-only theta for anchor 1), using the SAME exact-eval machinery as the rest of
    this script -- not a hardcoded literal."""
    d = args.main_run_dir
    theta_core_path = os.path.join(d, "theta_core.npy")
    theta_lat_path = os.path.join(d, "theta_lat_anchors.npy")
    if not (os.path.exists(theta_core_path) and os.path.exists(theta_lat_path)):
        return None
    theta_core = np.load(theta_core_path)
    theta_lat_anchors = np.load(theta_lat_path)
    sim = IQPExact(n=codec.N_QUBITS, K=K, device="cpu", dtype=torch.complex64)
    theta = torch.tensor(np.concatenate([theta_lat_anchors[0], theta_core]), dtype=sim.rdtype)
    zlut = codec.dequantize_z(np.arange(64), a_z, b_z)
    ez_field = exact_conditional_field(sim, theta, zlut)
    corr = float(np.corrcoef(ez_field.ravel(), gt_field.ravel())[0, 1])
    gen = torch.Generator(device="cpu")
    gen.manual_seed(0 + 999)
    mfield = sampled_field(sim, theta, a_z, b_z, 100000, gen)
    js = float(pdf_js(gt_field, mfield))
    return dict(corr=corr, pdf_js=js)


def summarize(args):
    snap, a_z, b_z = load_anchor(args.plane)
    gt_field = snap.field.astype(np.float32)

    header = "%-28s %10s %10s %14s  " % ("config", "corr", "pdf_js", "mmd2")
    header += "corr_after_stage2"
    print(header)
    for name in CONFIGS:
        out_dir = os.path.join(args.out_root, name)
        row, meta = read_final_row(out_dir)
        if row is None:
            print("%-28s MISSING (%s)" % (name, out_dir))
            continue
        after = "%.4f" % meta["corr_after_stage2"] if (meta and "corr_after_stage2" in meta) else "--"
        print("%-28s %10.4f %10.4f %14.4e  %s" % (name, row["corr"], row["pdf_js"], row["mmd2"], after))

    print()
    print("=== references (anchor t=%d, NOT part of this protocol table) ===" % ANCHOR_T)
    ref1, meta1 = read_final_row(args.auto3_ref_dir)
    if ref1 is not None:
        print("%-28s %10.4f %10.4f %14.4e  (job 342542, all-params, full-pool pmf, step=%s)"
              % ("anchor-fit auto3 (ref)", ref1["corr"], ref1["pdf_js"], ref1["mmd2"], ref1.get("step")))
    else:
        print("%-28s MISSING (%s)" % ("anchor-fit auto3 (ref)", args.auto3_ref_dir))

    ref2 = main_run_reference(args, gt_field, a_z, b_z)
    if ref2 is not None:
        print("%-28s %10.4f %10.4f %14s  (main run outputs_d2_main/iqp_K6_seed0, paper protocol, recomputed here)"
              % ("main run iqp_K6_seed0 (ref)", ref2["corr"], ref2["pdf_js"], "n/a"))
    else:
        print("%-28s MISSING (%s)" % ("main run iqp_K6_seed0 (ref)", args.main_run_dir))


# --------------------------------------------------------------------------------------
def main(argv=None):
    args = parse_args(argv)
    if args.summarize:
        summarize(args)
        return 0
    run_config(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
