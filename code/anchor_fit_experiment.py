"""Decisive anchor-fit experiment: does the IQP model family + exact-MMD objective
actually fit its own training anchor (t=1), if the kernel bandwidth/weighting is
changed? Motivation: the main replication's --sigmas auto3 kernel puts ~99.3% of its
weight on |s|<=3 while the D2 data's <Z_s>^2 energy sits at higher order; the trained
model reconstructs its OWN anchor at only corr~0.69 with <Z_s> error rising from ~5%
(|s|=1) to ~86% (|s|=8). The paper never specifies a bandwidth, so this asks: is the
gap fixable by choosing sigma/weights differently and/or training longer (a paper
under-specification), or is it a ceiling of the K<=6/7 exact-IQP family regardless of
kernel choice?

Trains theta as ONE vector (no core/latent split -- interpolation/synthesis is not the
question here, only "can this family+objective fit one anchor's own data") on anchor
t=1's full native pool_codes (~1e5 points, no subsampling -> exact empirical pmf),
data-dependent init (train_iqp.data_init_theta), plain Adam, for 8 kernel/K/lr configs
(CONFIGS below). Every --eval-every steps (default 5000) it logs, with NO sampling
noise where avoidable:
  (a) corr(E_model[z|x,y], GT field)   -- exact conditional mean from the model pmf
  (b) data vs model <Z_s> relative RMS error, grouped by Pauli weight |s| = 1..8
  (c) pdf_js(mean-agg field of 1e5 samples, GT)  -- the one metric that needs sampling
to outputs_anchor_fit/<config>/curve.jsonl, plus final artefacts (theta, both kinds of
reconstructed field). `--summarize` prints the comparison table (and two references:
GT-vs-itself, and the anchor's own pmf truncated to Walsh order <= 3 -- the "kernel
weights alone would explain everything" null model, same reconstruction algorithm as
diag_scripts/q2c_order_cutoff.py but scored against the anchor instead of held-out tau).

Python 3.9 compatible (no `X | None`, no match statements).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch

import codec
import data_d2
from iqp_exact import IQPExact, fwht, popcount_table
from metrics import pdf_js
from train_iqp import data_init_theta, resolve_sigmas

REPO = os.path.dirname(os.path.abspath(__file__))
OUT_ROOT_DEFAULT = os.path.join(REPO, "outputs_anchor_fit")

ANCHOR_T = 1
TRUNC_ORDER = 3
TRUNC_SEED = 7
TRUNC_N_SAMPLES = 100000

# 8 configs: (sigma_mode, K, lr). sigma_mode is "auto3" (Ref[38] three-bandwidth rule,
# the main run's default), "flat" (c_s = 2^-18 for every s -- equivalent to an
# unweighted L2 distance between pmfs, i.e. what "the kernel puts no extra weight
# anywhere" looks like), or "sigmaX.Y" (single bandwidth X.Y).
CONFIGS = {
    "auto3":              dict(sigma_mode="auto3",     K=6, lr=1e-4, steps=100000),
    "sigma0.3":           dict(sigma_mode="sigma0.3",  K=6, lr=1e-4, steps=100000),
    "sigma0.15":          dict(sigma_mode="sigma0.15", K=6, lr=1e-4, steps=100000),
    "flat":               dict(sigma_mode="flat",      K=6, lr=1e-4, steps=100000),
    "flat_lr1e-3":        dict(sigma_mode="flat",      K=6, lr=1e-3, steps=100000),
    "flat_K7":            dict(sigma_mode="flat",      K=7, lr=1e-4, steps=100000),
    "sigma0.3_K7_lr1e-3": dict(sigma_mode="sigma0.3",  K=7, lr=1e-3, steps=100000),
    "auto3_K7":           dict(sigma_mode="auto3",     K=7, lr=1e-4, steps=100000),
}

RELERR_ORDERS = (1, 2, 3, 4, 5, 6, 7, 8)
TABLE_ORDERS = (1, 2, 4, 6, 8)


# --------------------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", choices=sorted(CONFIGS), default=None)
    p.add_argument("--steps", type=int, default=None, help="override the config's step count")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=5000)
    p.add_argument("--out-root", type=str, default=OUT_ROOT_DEFAULT)
    p.add_argument("--plane", type=str, default="mid")
    p.add_argument("--n-synth-eval", type=int, default=100000)
    p.add_argument("--summarize", action="store_true")
    args = p.parse_args(argv)
    if not args.summarize and args.config is None:
        p.error("--config is required unless --summarize is given")
    return args


# --------------------------------------------------------------------------------------
def build_weights(sim, sigma_mode, pool_codes, seed):
    """Returns (weights: (2^n,) tensor, sigmas: list[float] or None for 'flat')."""
    if sigma_mode == "flat":
        w = torch.full((sim.size,), 1.0 / float(sim.size), dtype=sim.rdtype, device=sim.device)
        return w, None
    if sigma_mode == "auto3":
        sigmas, _ = resolve_sigmas("auto3", sim.n, pool_codes, seed)
    elif sigma_mode.startswith("sigma"):
        sigmas = [float(sigma_mode[len("sigma"):])]
    else:
        raise ValueError("unknown sigma_mode %r" % (sigma_mode,))
    return sim.kernel_weights(sigmas), sigmas


def exact_conditional_field(sim, theta, zlut):
    """E_model[z | x, y] on the full 64x64 grid, from the EXACT model pmf (no sampling noise).
    Same derivation as diag_scripts/q2_anchor_fit.py: code = kx<<12 | ky<<6 | kz, so a
    C-order reshape to (64,64,64) already gives axes (kx,ky,kz)."""
    with torch.no_grad():
        q = sim.probs(theta.detach()).double()
        q = torch.clamp(q, min=0.0)
        q = q / q.sum()
    Q = q.cpu().numpy().reshape(64, 64, 64)          # [kx, ky, kz]
    pxy = Q.sum(axis=2)                              # [kx, ky]
    ez = (Q * zlut[None, None, :]).sum(axis=2) / np.maximum(pxy, 1e-300)
    return ez.T.astype(np.float32)                   # -> [ky, kx] = (row=y, col=x)


def relerr_by_order(ez_model, ez_data, pc, orders=RELERR_ORDERS):
    out = {}
    for k in orders:
        mask = (pc == k)
        num = float(np.sqrt(np.mean((ez_model[mask] - ez_data[mask]) ** 2)))
        den = float(np.sqrt(np.mean(ez_data[mask] ** 2)))
        out[str(k)] = (num / den) if den > 0 else float("nan")
    return out


def sampled_field(sim, theta, a_z, b_z, n_samples, generator):
    codes = sim.sample(theta.detach(), n_samples, generator=generator)
    return codec.aggregate(codes.cpu().numpy(), a_z, b_z)


def order_cutoff_reference(p_hat, order, gt_field, a_z, b_z,
                            n_samples=TRUNC_N_SAMPLES, seed=TRUNC_SEED):
    """Project the anchor's OWN data pmf onto Walsh order <= `order`, resample, aggregate:
    same algorithm as diag_scripts/q2c_order_cutoff.py (there scored against held-out tau;
    here scored against the anchor itself, since that is what the trained models are
    compared to in this experiment)."""
    n = int(round(math.log2(p_hat.size)))
    pc = popcount_table(n)
    keep = (pc <= order)
    e = fwht(torch.as_tensor(p_hat)).numpy() * keep
    q = fwht(torch.as_tensor(e)).numpy() / float(p_hat.size)
    q = np.clip(q, 0, None)
    q = q / q.sum()
    rng = np.random.default_rng(seed)
    codes = rng.choice(p_hat.size, size=n_samples, p=q)
    field = codec.aggregate(codes.astype(np.int64), a_z, b_z)
    corr = float(np.corrcoef(field.ravel(), gt_field.ravel())[0, 1])
    js = float(pdf_js(gt_field, field))
    return corr, js


def load_anchor(plane):
    snap = data_d2.load_d2_snapshot(ANCHOR_T, plane=plane)
    a_z, b_z = data_d2.get_z_range(plane)
    return snap, a_z, b_z


# --------------------------------------------------------------------------------------
def run_config(args):
    cfg = CONFIGS[args.config]
    steps = args.steps if args.steps is not None else cfg["steps"]
    K, lr, sigma_mode = cfg["K"], cfg["lr"], cfg["sigma_mode"]
    device = torch.device(args.device)
    if device.type == "cpu":
        # see train_iqp.py's identical guard: FWHT at n=18 is dominated by sync overhead
        # under torch's default (often 100+) intra-op thread pool -- ~1000x slower.
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

    p_hat = codec.codes_to_pmf(pool_codes)

    sim = IQPExact(n=codec.N_QUBITS, K=K, device=device, dtype=torch.complex64)
    weights, sigmas = build_weights(sim, sigma_mode, pool_codes, args.seed)
    print("[model] config=%s K=%d params=%d sigma_mode=%s sigmas=%s lr=%g steps=%d device=%s"
          % (args.config, K, sim.n_params, sigma_mode,
             (["%.6f" % s for s in sigmas] if sigmas else None), lr, steps, device), flush=True)

    ez_data = sim.data_expvals(p_hat)
    ez_data_np = ez_data.double().cpu().numpy()
    zlut = codec.dequantize_z(np.arange(64), a_z, b_z)
    pc = popcount_table(codec.N_QUBITS)

    bits = ((pool_codes[:, None] >> np.arange(codec.N_QUBITS)[None, :]) & 1).astype(np.float64)
    theta0 = data_init_theta(sim, bits, None, 0.0, rng)
    theta0[0] = 0.0  # |s| = 0 is a global phase: exactly zero gradient, pin it to 0
    theta = torch.tensor(theta0, dtype=sim.rdtype, device=device, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=lr)

    gen = torch.Generator(device=device.type)
    gen.manual_seed(args.seed + 999)

    curve_path = os.path.join(out_dir, "curve.jsonl")
    curve_fh = open(curve_path, "w")
    last_eval = None
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = sim.mmd2(theta, ez_data, weights)
        loss.backward()
        opt.step()
        if step % args.eval_every == 0 or step == steps - 1:
            mmd2_val = float(loss.detach())
            ez_model = sim.expvals(theta.detach()).double().cpu().numpy()
            ez_field = exact_conditional_field(sim, theta, zlut)
            corr = float(np.corrcoef(ez_field.ravel(), gt_field.ravel())[0, 1])
            mfield = sampled_field(sim, theta, a_z, b_z, args.n_synth_eval, gen)
            js = float(pdf_js(gt_field, mfield))
            relerr = relerr_by_order(ez_model, ez_data_np, pc)
            row = dict(step=step, mmd2=mmd2_val, corr=corr, pdf_js=js, relerr=relerr,
                       wallclock_s=time.time() - t0)
            curve_fh.write(json.dumps(row) + "\n")
            curve_fh.flush()
            last_eval = (ez_field, mfield, row)
            print("[step %6d/%d] MMD2=%.6e corr=%.4f pdf_js=%.4f (%.1fs)"
                  % (step, steps, mmd2_val, corr, js, time.time() - t0), flush=True)

    curve_fh.close()
    ez_field, mfield, row = last_eval
    np.save(os.path.join(out_dir, "theta_final.npy"), theta.detach().cpu().numpy())
    np.save(os.path.join(out_dir, "field_exact.npy"), ez_field)
    np.save(os.path.join(out_dir, "field_meanagg.npy"), mfield)
    meta = dict(config=args.config, K=K, lr=lr, sigma_mode=sigma_mode, sigmas=sigmas, steps=steps,
                seed=args.seed, anchor_t=ANCHOR_T, plane=args.plane,
                wallclock_s=time.time() - t0, final=row)
    with open(os.path.join(out_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True, default=float)
    print("[done] %s -> %s (%.1fs)" % (args.config, out_dir, time.time() - t0), flush=True)


# --------------------------------------------------------------------------------------
def summarize(args):
    snap, a_z, b_z = load_anchor(args.plane)
    gt_field = snap.field.astype(np.float32)
    pool_codes = np.asarray(snap.pool_codes, dtype=np.int64)
    p_hat = codec.codes_to_pmf(pool_codes)

    print("=== reference values (anchor t=%d) ===" % ANCHOR_T)
    print("%-24s corr=%.4f pdf_js=%.4f" % ("GT vs itself", 1.0, 0.0))
    corr_trunc, js_trunc = order_cutoff_reference(p_hat, TRUNC_ORDER, gt_field, a_z, b_z)
    print("%-24s corr=%.4f pdf_js=%.4f" % ("order<=%d truncation" % TRUNC_ORDER, corr_trunc, js_trunc))
    print()

    header = "%-22s %12s %8s %8s  " % ("config", "final MMD2", "corr", "pdf_js")
    header += " ".join("relerr|s|=%d" % k for k in TABLE_ORDERS)
    print(header)
    for name in CONFIGS:
        meta_path = os.path.join(args.out_root, name, "meta.json")
        if not os.path.exists(meta_path):
            print("%-22s MISSING (%s)" % (name, meta_path))
            continue
        meta = json.load(open(meta_path))
        row = meta["final"]
        relerr = row["relerr"]
        line = "%-22s %12.4e %8.4f %8.4f  " % (name, row["mmd2"], row["corr"], row["pdf_js"])
        line += " ".join("%10.4f" % relerr.get(str(k), float("nan")) for k in TABLE_ORDERS)
        print(line)


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
