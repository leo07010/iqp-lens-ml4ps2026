#!/usr/bin/env python3
"""Round-3 DCGAN baseline on the centroid-aligned 32x32 crops.

docs/IQP_LENS_ROUND3_PREREGISTRATION.md, arm `DCGAN-32`: a Radford-style generator for 32x32
crops (~0.2M generator parameters, latent 128), 20k steps, samples pasted at offsets drawn
from the training images. It is the "a real generative model of this size, on this data, with
this little data" reference: prediction P14 says no IQP arm beats it on NN-holdout at N = 100.

Not shared with round 2's baseline_dcgan.py: that file builds a 64x64 five-block Radford stack
and is in use by a running job. Round 3 needs a four-block 32x32 stack, so the two networks are
written out here rather than parameterised across files.

Scaling: crops live in [0, 1] (the source images do), so the network sees 2*crop - 1 and its
tanh output is mapped back by (x + 1)/2 and clipped at 0. NOTHING rescales a sample's total
intensity to S -- unlike the pmf arms, a GAN is free to get the flux wrong, and the physics
score's W1(flux) column is where that shows up. (Round 1's 64x64 DCGAN totalled 899 against a
held-out 149; keeping the arm honest here means letting that be visible, not correcting it.)

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

NZ = 128


def make_generator(ngf, nz=NZ):
    """4 x 4 -> 8 -> 16 -> 32, one channel out."""
    return nn.Sequential(
        nn.ConvTranspose2d(nz, ngf * 4, 4, 1, 0, bias=False), nn.BatchNorm2d(ngf * 4), nn.ReLU(True),
        nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 2), nn.ReLU(True),
        nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf), nn.ReLU(True),
        nn.ConvTranspose2d(ngf, 1, 4, 2, 1, bias=True), nn.Tanh(),
    )


def make_discriminator(ndf):
    return nn.Sequential(
        nn.Conv2d(1, ndf, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, True),
        nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 2), nn.LeakyReLU(0.2, True),
        nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 4), nn.LeakyReLU(0.2, True),
        nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=True), nn.Sigmoid(),
    )


def weights_init(m):
    name = m.__class__.__name__
    if name.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif name.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def count_params(m):
    return int(sum(p.numel() for p in m.parameters()))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Round-3 DCGAN baseline on 32x32 lens crops.")
    p.add_argument("--n-train", type=int, default=11)
    p.add_argument("--n-holdout", type=int, default=data_lens_r3.N_HOLDOUT)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--ngf", type=int, default=18, help="18 -> 199,620 generator params (~0.2M)")
    p.add_argument("--ndf", type=int, default=18)
    p.add_argument("--nz", type=int, default=NZ)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--beta1", type=float, default=0.5)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--n-gen", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
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
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "crops"), exist_ok=True)

    d = data_lens_r3.load_r3(args.n_train, seed=args.seed, n_holdout=args.n_holdout)
    n = d.n_train
    real = torch.as_tensor(np.asarray(d.train_crops, dtype=np.float32), device=device).unsqueeze(1)
    real = real * 2.0 - 1.0                                   # [0, 1] -> [-1, 1]
    batch = int(min(args.batch, n))
    print("[dcgan_r3] N=%d crops %s batch=%d" % (n, (tuple(real.shape),), batch))

    netG = make_generator(args.ngf, args.nz).to(device)
    netD = make_discriminator(args.ndf).to(device)
    netG.apply(weights_init)
    netD.apply(weights_init)
    pg, pd = count_params(netG), count_params(netD)
    print("[dcgan_r3] generator params=%d  discriminator params=%d  nz=%d ngf=%d ndf=%d"
          % (pg, pd, args.nz, args.ngf, args.ndf))

    optG = torch.optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optD = torch.optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    bce = nn.BCELoss()
    gen_t = torch.Generator(device="cpu").manual_seed(int(args.seed) + 5)
    curve = []
    for step in range(int(args.steps)):
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

        if step % max(int(args.log_every), 1) == 0 or step == int(args.steps) - 1:
            curve.append([int(step), float(lossD.item()), float(lossG.item())])
            print("[dcgan_r3] step %6d lossD=%.4f lossG=%.4f (%.1f s)"
                  % (step, float(lossD.item()), float(lossG.item()), time.time() - t0))
    train_time = time.time() - t0

    netG.eval()
    crops = np.empty((int(args.n_gen), codec10.GRID, codec10.GRID), dtype=np.float32)
    with torch.no_grad():
        done = 0
        while done < int(args.n_gen):
            m = min(64, int(args.n_gen) - done)
            z = torch.randn(m, args.nz, 1, 1, generator=gen_t).to(device)
            y = netG(z).squeeze(1).cpu().numpy().astype(np.float32)
            crops[done:done + m] = np.clip((y + 1.0) * 0.5, 0.0, None)
            done += m
    offsets = data_lens_r3.sample_offsets(d.train_offsets, crops.shape[0],
                                          np.random.default_rng(args.seed + 999))
    for i in range(crops.shape[0]):
        np.save(os.path.join(args.out, "crops", "generated_%03d.npy" % i), crops[i])
        np.save(os.path.join(args.out, "generated_%03d.npy" % i),
                codec10.paste(crops[i], int(offsets[i, 0]), int(offsets[i, 1])))

    meta = {
        "arm": args.arm_name, "script": "baseline_dcgan_r3.py", "round": 3,
        "param_count": pg, "param_count_discriminator": pd,
        "nz": int(args.nz), "ngf": int(args.ngf), "ndf": int(args.ndf),
        "protocol": "DCGAN (BCE, Adam 2e-4, beta1 0.5)", "objective": "non-saturating GAN",
        "n_train": int(n), "n_gen": int(crops.shape[0]), "n_holdout": int(args.n_holdout),
        "S": float(d.S), "seed": int(args.seed), "device": str(args.device),
        "steps": int(args.steps), "batch": batch, "conditional": False,
        "scaling": "trained on 2*crop-1, written back as clip((x+1)/2, 0, None); NO rescale to S",
        "gen_mean_total": float(np.asarray(crops, dtype=np.float64).reshape(crops.shape[0], -1).sum(1).mean()),
        "train_mean_total": float(d.S),
        "lossD_last": float(curve[-1][1]) if curve else None,
        "lossG_last": float(curve[-1][2]) if curve else None,
        "train_seconds": float(train_time),
        "wallclock_seconds": float(time.time() - t0),
        "gen_offsets": [[int(a), int(b)] for a, b in offsets.tolist()],
    }
    with open(os.path.join(args.out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True, default=str)
    np.save(os.path.join(args.out, "loss_curve.npy"), np.asarray(curve, dtype=np.float64))
    print("[dcgan_r3] wrote %d images to %s; mean total intensity %.2f vs training S %.2f (%.1f s)"
          % (crops.shape[0], args.out, meta["gen_mean_total"], d.S, time.time() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
