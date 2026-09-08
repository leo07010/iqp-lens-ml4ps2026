#!/usr/bin/env python3
"""Convergence sweep for the round-3 DCGAN baseline: ONE training run, snapshots every 5k steps.

Motivation (paper addendum, 2026-09-09): the round-3 table trains both classical baselines for
a fixed 20,000 steps, and the DCGAN's lossD / lossG never settle -- they oscillate for the whole
budget, which is normal for a GAN but means "20k steps" is an arbitrary stopping point.  This
script trains to 60k and scores the generator every 5k steps, so the baseline is reported at its
BEST step count rather than at an arbitrary one.  No IQP arm is given that advantage.

Exactness of the snapshots
--------------------------
`baseline_dcgan_r3.main` is a single sequential loop over one CPU generator (`seed + 5`) that
draws the minibatch index and the noise, with no step-count-dependent schedule.  A snapshot at
step k therefore holds exactly the weights a standalone `--steps k` run would end with.  To make
the SAMPLING identical too, the generator's RNG state is saved before drawing the 200 z vectors
and restored afterwards, so (a) the z draws are the ones a standalone `--steps k` run would have
made, and (b) training continues from an unperturbed stream.  `netG` is put back in train mode
after each snapshot, so BatchNorm running statistics keep updating exactly as before.

Verified by the byte-compare pre-check in submit_baseline_convergence.sbatch.

Every snapshot directory is a normal round-3 run directory (meta.json + generated_*.npy +
crops/) and is scored by the UNMODIFIED evaluate_r3.py / evaluate_r3b.py.  Nothing here modifies
baseline_dcgan_r3.py; the networks and the initialisation are imported from it.

Python 3.9 compatible (no `X | None`).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

import codec10
import data_lens_r3
import baseline_dcgan_r3 as G

DEFAULT_TOTAL = 60000
DEFAULT_EVERY = 5000


def step_tag(steps):
    """5000 -> 's005k' -- sorts lexicographically in the evaluate_r3 table."""
    return "s%03dk" % int(round(int(steps) / 1000.0))


def emit_snapshot(netG, d, args, n_steps, lossD, lossG, curve, elapsed, gen_t, out_dir):
    """Everything `baseline_dcgan_r3.main` does after training, for one snapshot.

    `gen_t`'s state is saved and restored around the z draws so training is unperturbed and the
    draws match a standalone `--steps n_steps` run.
    """
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "crops"), exist_ok=True)
    rng_state = gen_t.get_state().clone()
    was_training = netG.training
    netG.eval()
    crops = np.empty((int(args.n_gen), codec10.GRID, codec10.GRID), dtype=np.float32)
    with torch.no_grad():
        done = 0
        while done < int(args.n_gen):
            m = min(64, int(args.n_gen) - done)
            z = torch.randn(m, args.nz, 1, 1, generator=gen_t).to(next(netG.parameters()).device)
            y = netG(z).squeeze(1).cpu().numpy().astype(np.float32)
            crops[done:done + m] = np.clip((y + 1.0) * 0.5, 0.0, None)
            done += m
    gen_t.set_state(rng_state)
    if was_training:
        netG.train()

    offsets = data_lens_r3.sample_offsets(d.train_offsets, crops.shape[0],
                                          np.random.default_rng(args.seed + 999))
    for i in range(crops.shape[0]):
        np.save(os.path.join(out_dir, "crops", "generated_%03d.npy" % i), crops[i])
        np.save(os.path.join(out_dir, "generated_%03d.npy" % i),
                codec10.paste(crops[i], int(offsets[i, 0]), int(offsets[i, 1])))

    meta = {
        "arm": "%s %s" % (args.arm_name, step_tag(n_steps)), "arm_base": args.arm_name,
        "script": "baseline_dcgan_sweep.py", "round": 3, "sweep": "convergence",
        "param_count": G.count_params(netG),
        "nz": int(args.nz), "ngf": int(args.ngf), "ndf": int(args.ndf),
        "protocol": "DCGAN (BCE, Adam 2e-4, beta1 0.5)", "objective": "non-saturating GAN",
        "n_train": int(d.n_train), "n_gen": int(crops.shape[0]), "n_holdout": int(args.n_holdout),
        "S": float(d.S), "seed": int(args.seed), "device": str(args.device),
        "steps": int(n_steps), "batch": int(min(args.batch, d.n_train)), "conditional": False,
        "scaling": "trained on 2*crop-1, written back as clip((x+1)/2, 0, None); NO rescale to S",
        "gen_mean_total": float(np.asarray(crops, dtype=np.float64)
                                .reshape(crops.shape[0], -1).sum(1).mean()),
        "train_mean_total": float(d.S),
        "lossD_last": float(lossD), "lossG_last": float(lossG),
        "train_seconds": float(elapsed), "wallclock_seconds": float(elapsed),
        "gen_offsets": [[int(a), int(b)] for a, b in offsets.tolist()],
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True, default=str)
    np.save(os.path.join(out_dir, "loss_curve.npy"), np.asarray(curve, dtype=np.float64))
    print("[dcgan_sweep] snapshot %d steps -> %s (lossD %.4f lossG %.4f, mean total %.2f, %.1f s)"
          % (n_steps, out_dir, lossD, lossG, meta["gen_mean_total"], elapsed), flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="DCGAN baseline convergence sweep (one run, N snapshots).")
    p.add_argument("--n-train", type=int, default=11)
    p.add_argument("--n-holdout", type=int, default=data_lens_r3.N_HOLDOUT)
    p.add_argument("--steps", type=int, default=DEFAULT_TOTAL)
    p.add_argument("--snap-every", type=int, default=DEFAULT_EVERY)
    p.add_argument("--ngf", type=int, default=18)
    p.add_argument("--ndf", type=int, default=18)
    p.add_argument("--nz", type=int, default=G.NZ)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--beta1", type=float, default=0.5)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--n-gen", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-prefix", required=True,
                   help="snapshot k is written to <out-prefix>_s<k/1000>k")
    p.add_argument("--device", default="cpu")
    p.add_argument("--arm-name", default="DCGAN-32")
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
    total = int(args.steps)
    every = max(int(args.snap_every), 1)
    snaps = sorted(set(list(range(every, total + 1, every)) + [total]))

    d = data_lens_r3.load_r3(args.n_train, seed=args.seed, n_holdout=args.n_holdout)
    n = d.n_train
    real = torch.as_tensor(np.asarray(d.train_crops, dtype=np.float32), device=device).unsqueeze(1)
    real = real * 2.0 - 1.0
    batch = int(min(args.batch, n))
    print("[dcgan_sweep] N=%d crops %s batch=%d snapshots %s"
          % (n, (tuple(real.shape),), batch, snaps), flush=True)

    netG = G.make_generator(args.ngf, args.nz).to(device)
    netD = G.make_discriminator(args.ndf).to(device)
    netG.apply(G.weights_init)
    netD.apply(G.weights_init)
    pg, pd = G.count_params(netG), G.count_params(netD)
    print("[dcgan_sweep] generator params=%d discriminator params=%d nz=%d ngf=%d ndf=%d"
          % (pg, pd, args.nz, args.ngf, args.ndf), flush=True)

    optG = torch.optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optD = torch.optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    bce = nn.BCELoss()
    gen_t = torch.Generator(device="cpu").manual_seed(int(args.seed) + 5)
    want = set(snaps)
    curve = []
    for step in range(total):
        idx = torch.randint(0, n, (batch,), generator=gen_t).to(device)
        x = real[idx]
        z = torch.randn(batch, args.nz, 1, 1, generator=gen_t).to(device)
        fake = netG(z)

        netD.zero_grad()
        out_r = netD(x).view(-1)
        out_f = netD(fake.detach()).view(-1)
        lossD = bce(out_r, torch.ones_like(out_r)) + bce(out_f, torch.zeros_like(out_f))
        lossD.backward()
        optD.step()

        netG.zero_grad()
        out_f2 = netD(fake).view(-1)
        lossG = bce(out_f2, torch.ones_like(out_f2))
        lossG.backward()
        optG.step()

        ld, lg = float(lossD.item()), float(lossG.item())
        if step % max(int(args.log_every), 1) == 0 or step == total - 1:
            curve.append([int(step), ld, lg])
            print("[dcgan_sweep] step %6d lossD=%.4f lossG=%.4f (%.1f s)"
                  % (step, ld, lg, time.time() - t0), flush=True)
        if (step + 1) in want:
            emit_snapshot(netG, d, args, step + 1, ld, lg, curve, time.time() - t0, gen_t,
                          "%s_%s" % (args.out_prefix, step_tag(step + 1)))
    print("[dcgan_sweep] done: %d snapshots, %.1f s total" % (len(snaps), time.time() - t0), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
