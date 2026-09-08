#!/usr/bin/env python3
"""Convergence sweep for the round-3 RBM baseline: ONE training run, several snapshots.

Motivation (paper addendum, 2026-09-09): the round-3 table trains both classical baselines for
a fixed 20,000 steps, but the RBM's recon-MSE log is still falling ~20-25 % over the last
quarter of that budget at N = 100.  A reviewer is entitled to ask whether the baseline was
simply stopped too early, so this script re-runs the SAME arm for longer and scores it at
several step counts, giving the baseline a best-of-steps advantage that no IQP arm gets.

Why one run with snapshots rather than one run per step count
-------------------------------------------------------------
`baseline_rbm_r3.pcd_train` is a single sequential loop with one RNG stream and no
step-count-dependent schedule, so the model state after step index `k - 1` of a 160k run is
BIT-IDENTICAL to the final state of a standalone `--steps k` run.  Generation is independent
of the training stream (`gibbs_crops` opens a fresh `seed + 31` generator, the latent prior is
refitted from scratch, and the generation / offset RNGs are re-seeded at `seed + 999` for every
snapshot), so each snapshot reproduces exactly what `baseline_rbm_r3.py --steps k` would have
written.  Verified by the byte-compare pre-check in submit_baseline_convergence.sbatch.

Every snapshot directory is a normal round-3 run directory (meta.json + generated_*.npy +
crops/) and is scored by the UNMODIFIED evaluate_r3.py / evaluate_r3b.py.

Nothing here modifies baseline_rbm_r3.py; the training step, the Gibbs reconstruction and the
latent prior are imported from it.

Python 3.9 compatible (no `X | None`).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

import codec10
import data_lens_r3
import baseline_rbm_r3 as B
from train_iqp_r3 import fit_latent_gaussian, sample_latents

DEFAULT_SNAPS = (20000, 40000, 80000, 160000)


def step_tag(steps):
    """20000 -> 's020k' -- sorts lexicographically in the evaluate_r3 table."""
    return "s%03dk" % int(round(int(steps) / 1000.0))


def pcd_train_snapshots(rbm, data_bits, steps, lr, batch, n_chains, seed, log_every,
                        snaps, on_snapshot):
    """`baseline_rbm_r3.pcd_train` verbatim, plus a callback after step index `s - 1`.

    The arithmetic, the RNG stream and the Adam state are untouched; `on_snapshot(s, recon)`
    is called under torch.no_grad() and must not consume `gen`.
    """
    device = rbm.device
    gen = torch.Generator(device=device).manual_seed(int(seed) + 7)
    I, n_pool, _ = data_bits.shape
    v_chain = (torch.rand(I, n_chains, B.N_VISIBLE, generator=gen, device=device) < 0.5).float()

    for t in (rbm.W, rbm.b_v, rbm.b_h_lat, rbm.b_h_core):
        t.requires_grad_(True)
    opt = torch.optim.Adam([rbm.W, rbm.b_v, rbm.b_h_lat, rbm.b_h_core], lr=lr)
    want = sorted(set(int(s) for s in snaps))
    curve = []
    for step in range(int(steps)):
        with torch.no_grad():
            idx = torch.randint(0, n_pool, (I, batch), generator=gen, device=device)
            v_data = torch.gather(data_bits, 1, idx.unsqueeze(-1).expand(-1, -1, B.N_VISIBLE))
            b_h = rbm.b_h()

            h_data_p = rbm.p_h_given_v(v_data, b_h)
            pos_W = torch.einsum("imv,imh->vh", v_data, h_data_p) / float(I * batch)
            pos_bv = v_data.mean(dim=1)
            pos_bh = h_data_p.mean(dim=1)

            h0, _ = rbm.sample_h(v_chain, b_h, gen)
            v1, _ = rbm.sample_v(h0, rbm.b_v, gen)
            h1_p = rbm.p_h_given_v(v1, b_h)
            neg_W = torch.einsum("imv,imh->vh", v1, h1_p) / float(I * n_chains)
            neg_bv = v1.mean(dim=1)
            neg_bh = h1_p.mean(dim=1)
            v_chain = v1

            dW = pos_W - neg_W
            dbv = pos_bv - neg_bv
            dbh = pos_bh - neg_bh
            recon = float((v_data - rbm.p_v_given_h(h_data_p, rbm.b_v)).pow(2).mean().item())

        opt.zero_grad()
        rbm.W.grad = -dW
        rbm.b_v.grad = -dbv
        rbm.b_h_lat.grad = -dbh[:, :B.N_BH_LAT]
        rbm.b_h_core.grad = -dbh[:, B.N_BH_LAT:].mean(dim=0)
        opt.step()
        if step % max(int(log_every), 1) == 0 or step == int(steps) - 1:
            curve.append([int(step), recon])
            print("[rbm_sweep] step %6d recon_mse=%.6f" % (step, recon), flush=True)
        if (step + 1) in want:
            with torch.no_grad():
                on_snapshot(step + 1, recon, list(curve))
    for t in (rbm.W, rbm.b_v, rbm.b_h_lat, rbm.b_h_core):
        t.requires_grad_(False)
    return curve


def emit_snapshot(rbm, d, args, n_steps, recon, curve, elapsed, out_dir):
    """Everything `baseline_rbm_r3.main` does after training, for one snapshot."""
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "crops"), exist_ok=True)
    n = d.n_train
    latents = rbm.get_latents()
    cov_mode = "full" if n >= 100 else "diag"
    prior = fit_latent_gaussian(latents, cov_mode)
    gen_rng = np.random.default_rng(args.seed + 999)
    lat_gen = sample_latents(prior, int(args.n_gen), gen_rng)
    crops = B.gibbs_crops(rbm, lat_gen, d.S, args.gen_chains, args.gen_sweeps,
                          args.gen_keep, args.gen_chunk, args.seed)
    offsets = data_lens_r3.sample_offsets(d.train_offsets, crops.shape[0], gen_rng)
    for i in range(crops.shape[0]):
        np.save(os.path.join(out_dir, "crops", "generated_%03d.npy" % i), crops[i].astype(np.float32))
        frame = codec10.paste(crops[i], int(offsets[i, 0]), int(offsets[i, 1]))
        np.save(os.path.join(out_dir, "generated_%03d.npy" % i), frame.astype(np.float32))

    meta = {
        "arm": "%s %s" % (args.arm_name, step_tag(n_steps)), "arm_base": args.arm_name,
        "script": "baseline_rbm_sweep.py", "round": 3, "sweep": "convergence",
        "param_count": int(rbm.param_count()),
        "n_visible": B.N_VISIBLE, "n_hidden": int(args.n_hidden), "d_lat": B.D_LAT,
        "latent_block": "b_v (10) + b_h[:40]", "core_block": "W (10x%d) + b_h[40:]" % args.n_hidden,
        "protocol": "joint PCD-1", "objective": "PCD-1 (contrastive divergence)",
        "n_train": int(n), "n_gen": int(crops.shape[0]), "n_holdout": int(args.n_holdout),
        "S": float(d.S), "seed": int(args.seed), "device": str(args.device),
        "steps": int(n_steps), "lr": float(args.lr), "batch": int(args.batch),
        "chains": int(args.chains), "n_pool": int(args.n_pool),
        "gen_chains": int(args.gen_chains), "gen_sweeps": int(args.gen_sweeps),
        "gen_keep": int(args.gen_keep),
        "gen_samples_per_image": int(args.gen_chains * args.gen_keep),
        "reconstruction": "sampled (counting density) -- an RBM has no closed-form marginal here",
        "conditional": False,
        "prior": {"kind": "Gaussian in PCA-whitened coordinates", "cov_mode": cov_mode,
                  "rank": int(prior["rank"]),
                  "whitened_cov_max_dev_from_I": float(prior["whitened_cov_dev"])},
        "recon_mse_first": float(curve[0][1]) if curve else None,
        "recon_mse_last": float(recon),
        "train_seconds": float(elapsed),
        "wallclock_seconds": float(elapsed),
        "gen_offsets": [[int(a), int(b)] for a, b in offsets.tolist()],
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True, default=str)
    np.save(os.path.join(out_dir, "latents.npy"), latents)
    np.save(os.path.join(out_dir, "recon_curve.npy"), np.asarray(curve, dtype=np.float64))
    print("[rbm_sweep] snapshot %d steps -> %s (recon_mse %.6f, %.1f s)"
          % (n_steps, out_dir, recon, elapsed), flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="RBM baseline convergence sweep (one run, N snapshots).")
    p.add_argument("--n-train", type=int, default=11)
    p.add_argument("--n-holdout", type=int, default=data_lens_r3.N_HOLDOUT)
    p.add_argument("--n-hidden", type=int, default=B.N_HIDDEN_DEFAULT)
    p.add_argument("--snapshots", type=int, nargs="+", default=list(DEFAULT_SNAPS))
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--chains", type=int, default=64)
    p.add_argument("--n-pool", type=int, default=50000)
    p.add_argument("--n-gen", type=int, default=200)
    p.add_argument("--gen-chains", type=int, default=512)
    p.add_argument("--gen-sweeps", type=int, default=200)
    p.add_argument("--gen-keep", type=int, default=100)
    p.add_argument("--gen-chunk", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-prefix", required=True,
                   help="snapshot k is written to <out-prefix>_s<k/1000>k")
    p.add_argument("--device", default="cpu")
    p.add_argument("--arm-name", default="RBM-10bit")
    p.add_argument("--log-every", type=int, default=2000)
    p.add_argument("--threads", type=int, default=4)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    t0 = time.time()
    if str(args.device) == "cpu":
        torch.set_num_threads(min(int(args.threads), os.cpu_count() or 1))
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    snaps = sorted(set(int(s) for s in args.snapshots))
    total = max(snaps)

    d = data_lens_r3.load_r3(args.n_train, seed=args.seed, n_holdout=args.n_holdout)
    n = d.n_train
    pool = np.stack([codec10.sample_codes_from_pmf(d.train_pmfs[i], args.n_pool,
                                                   np.random.default_rng([20260906, i]))
                     for i in range(n)], axis=0)
    data_bits = torch.stack([B.codes_to_bits_t(pool[i], device) for i in range(n)], dim=0)
    print("[rbm_sweep] N=%d  S=%.4f  data pool %s  snapshots %s"
          % (n, d.S, (tuple(data_bits.shape),), snaps), flush=True)

    rbm = B.JointRBM(n, n_hidden=args.n_hidden, device=device, seed=args.seed)
    print("[rbm_sweep] %d visible x %d hidden -> %d params per image"
          % (B.N_VISIBLE, args.n_hidden, rbm.param_count()), flush=True)

    def on_snapshot(k, recon, curve):
        emit_snapshot(rbm, d, args, k, recon, curve, time.time() - t0,
                      "%s_%s" % (args.out_prefix, step_tag(k)))

    pcd_train_snapshots(rbm, data_bits, total, args.lr, args.batch, args.chains,
                        args.seed, args.log_every, snaps, on_snapshot)
    print("[rbm_sweep] done: %d snapshots, %.1f s total" % (len(snaps), time.time() - t0), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
