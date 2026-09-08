#!/usr/bin/env python3
"""Round-3 RBM baseline: a Bernoulli RBM over the same 10-bit codes as the IQP arms.

docs/IQP_LENS_ROUND3_PREREGISTRATION.md, arm `RBM-10bit`. The point of this arm is that it is
the SAME modelling problem -- a distribution over the 1,024 codes of a centroid-aligned 32x32
crop -- attacked by a classical energy-based model of comparable size, run through the SAME
latent / prior / paste pipeline, so a difference in the table is about the model and not about
the representation.

    10 visible x 1,000 hidden -> 10*1000 + 10 + 1000 = 11,010 parameters ("~12k").

Shared vs per-image split (the RBM's analogue of theta_core / theta_lat):

    latent (50 per image) : b_v (all 10) and b_h[:40]
    core   (shared)       : W (10 x 1000) and b_h[40:]

Training is JOINT, matching the primary IQP protocol: every training image contributes a
positive phase (a minibatch from its own 5e4 codes drawn from its pmf) and keeps its own
persistent negative chains; the shared parameters get the average gradient over images, the
latent ones their own. PCD-1, Adam lr 1e-3.

Generation: the N trained latents are fitted with the SAME PCA-whitened Gaussian rule as the
IQP arms (train_iqp_r3.fit_latent_gaussian, imported rather than re-implemented), --n-gen
latents are drawn, and each is turned into a crop by Gibbs sampling the RBM and taking the
counting density of the kept samples x S. An RBM has no closed-form marginal here, so unlike
the IQP arms this arm's reconstruction is necessarily Monte-Carlo; the sample budget is
recorded in meta.json.

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
from train_iqp_r3 import fit_latent_gaussian, sample_latents

N_VISIBLE = codec10.N_QUBITS      # 10
N_HIDDEN_DEFAULT = 1000
D_LAT = 50
N_BH_LAT = D_LAT - N_VISIBLE      # 40 hidden biases inside the latent block


def codes_to_bits_t(codes, device):
    """int64 codes -> (m, 10) float32 bit matrix on `device` (bit j == qubit j)."""
    c = torch.as_tensor(np.asarray(codes, dtype=np.int64), device=device)
    shifts = torch.arange(N_VISIBLE, device=device, dtype=torch.long)
    return ((c[:, None] >> shifts[None, :]) & 1).to(torch.float32)


def bits_to_codes_np(bits):
    b = np.asarray(bits, dtype=np.int64)
    return (b * (1 << np.arange(N_VISIBLE, dtype=np.int64))[None, :]).sum(axis=1)


class JointRBM(object):
    """W and b_h[40:] shared across images; b_v and b_h[:40] per image."""

    def __init__(self, n_images, n_hidden=N_HIDDEN_DEFAULT, device="cpu", seed=0):
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        self.n_images = int(n_images)
        self.n_hidden = int(n_hidden)
        self.device = torch.device(device)
        self.W = (0.01 * torch.randn(N_VISIBLE, self.n_hidden, generator=g)).to(self.device)
        self.b_v = torch.zeros(self.n_images, N_VISIBLE, device=self.device)
        self.b_h_lat = torch.zeros(self.n_images, N_BH_LAT, device=self.device)
        self.b_h_core = torch.zeros(self.n_hidden - N_BH_LAT, device=self.device)

    def param_count(self):
        """The size of ONE image's RBM (the number quoted in the arm table)."""
        return N_VISIBLE * self.n_hidden + N_VISIBLE + self.n_hidden

    def b_h(self):
        """(n_images, n_hidden) full hidden bias."""
        return torch.cat([self.b_h_lat, self.b_h_core.unsqueeze(0).expand(self.n_images, -1)], dim=1)

    def get_latents(self):
        return torch.cat([self.b_v, self.b_h_lat], dim=1).detach().cpu().numpy().astype(np.float64)

    # --- conditional distributions; v is (I, m, 10), biases broadcast over the image axis ---
    def p_h_given_v(self, v, b_h):
        return torch.sigmoid(torch.einsum("imv,vh->imh", v, self.W) + b_h.unsqueeze(1))

    def p_v_given_h(self, h, b_v):
        return torch.sigmoid(torch.einsum("imh,vh->imv", h, self.W) + b_v.unsqueeze(1))

    def sample_h(self, v, b_h, gen):
        p = self.p_h_given_v(v, b_h)
        return (torch.rand(p.shape, generator=gen, device=p.device) < p).float(), p

    def sample_v(self, h, b_v, gen):
        p = self.p_v_given_h(h, b_v)
        return (torch.rand(p.shape, generator=gen, device=p.device) < p).float(), p


def pcd_train(rbm, data_bits, steps, lr, batch, n_chains, seed, log_every):
    """PCD-1 with manual gradients fed to Adam (same scheme as round 2's baseline_rbm.py).

    data_bits : (I, n_pool, 10) float32 on device -- each image's 5e4 codes as bits.
    """
    device = rbm.device
    gen = torch.Generator(device=device).manual_seed(int(seed) + 7)
    I, n_pool, _ = data_bits.shape
    v_chain = (torch.rand(I, n_chains, N_VISIBLE, generator=gen, device=device) < 0.5).float()

    for t in (rbm.W, rbm.b_v, rbm.b_h_lat, rbm.b_h_core):
        t.requires_grad_(True)
    opt = torch.optim.Adam([rbm.W, rbm.b_v, rbm.b_h_lat, rbm.b_h_core], lr=lr)
    curve = []
    for step in range(int(steps)):
        with torch.no_grad():
            idx = torch.randint(0, n_pool, (I, batch), generator=gen, device=device)
            v_data = torch.gather(data_bits, 1, idx.unsqueeze(-1).expand(-1, -1, N_VISIBLE))
            b_h = rbm.b_h()

            h_data_p = rbm.p_h_given_v(v_data, b_h)
            pos_W = torch.einsum("imv,imh->vh", v_data, h_data_p) / float(I * batch)
            pos_bv = v_data.mean(dim=1)                        # (I, 10)
            pos_bh = h_data_p.mean(dim=1)                      # (I, H)

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
            # reconstruction error, a progress signal only (PCD has no tractable loss)
            recon = float((v_data - rbm.p_v_given_h(h_data_p, rbm.b_v)).pow(2).mean().item())

        opt.zero_grad()
        rbm.W.grad = -dW
        rbm.b_v.grad = -dbv
        rbm.b_h_lat.grad = -dbh[:, :N_BH_LAT]
        rbm.b_h_core.grad = -dbh[:, N_BH_LAT:].mean(dim=0)     # shared: average over images
        opt.step()
        if step % max(int(log_every), 1) == 0 or step == int(steps) - 1:
            curve.append([int(step), recon])
            print("[rbm_r3] step %6d recon_mse=%.6f" % (step, recon))
    for t in (rbm.W, rbm.b_v, rbm.b_h_lat, rbm.b_h_core):
        t.requires_grad_(False)
    return curve


def gibbs_crops(rbm, latents, S, n_chains, n_sweeps, keep, chunk, seed):
    """(m, 32, 32) crops: one Gibbs run per latent, counting density of the kept samples x S."""
    device = rbm.device
    gen = torch.Generator(device=device).manual_seed(int(seed) + 31)
    lat = torch.as_tensor(np.asarray(latents, dtype=np.float32), device=device)
    m = lat.shape[0]
    out = np.empty((m, codec10.GRID, codec10.GRID), dtype=np.float32)
    keep_from = max(int(n_sweeps) - int(keep), 0)
    for start in range(0, m, int(chunk)):
        stop = min(start + int(chunk), m)
        b_v = lat[start:stop, :N_VISIBLE]
        b_h = torch.cat([lat[start:stop, N_VISIBLE:],
                         rbm.b_h_core.unsqueeze(0).expand(stop - start, -1)], dim=1)
        v = (torch.rand(stop - start, n_chains, N_VISIBLE, generator=gen, device=device) < 0.5).float()
        counts = torch.zeros(stop - start, codec10.NCODES, device=device)
        pw = torch.as_tensor(1 << np.arange(N_VISIBLE, dtype=np.int64), device=device)
        for sweep in range(int(n_sweeps)):
            h, _ = rbm.sample_h(v, b_h, gen)
            v, _ = rbm.sample_v(h, b_v, gen)
            if sweep >= keep_from:
                codes = (v.to(torch.long) * pw[None, None, :]).sum(dim=2)      # (chunk, chains)
                counts.scatter_add_(1, codes, torch.ones_like(codes, dtype=counts.dtype))
        dens = (counts / counts.sum(dim=1, keepdim=True).clamp(min=1.0)).cpu().numpy().astype(np.float64)
        for j in range(stop - start):
            out[start + j] = codec10.pmf_to_image(dens[j], S)
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Round-3 RBM baseline on 10-bit lens crop codes.")
    p.add_argument("--n-train", type=int, default=11)
    p.add_argument("--n-holdout", type=int, default=data_lens_r3.N_HOLDOUT)
    p.add_argument("--n-hidden", type=int, default=N_HIDDEN_DEFAULT)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--chains", type=int, default=64)
    p.add_argument("--n-pool", type=int, default=50000, help="codes drawn per training image")
    p.add_argument("--n-gen", type=int, default=200)
    p.add_argument("--gen-chains", type=int, default=512)
    p.add_argument("--gen-sweeps", type=int, default=200)
    p.add_argument("--gen-keep", type=int, default=100)
    p.add_argument("--gen-chunk", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
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
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "crops"), exist_ok=True)

    d = data_lens_r3.load_r3(args.n_train, seed=args.seed, n_holdout=args.n_holdout)
    n = d.n_train
    pool = np.stack([codec10.sample_codes_from_pmf(d.train_pmfs[i], args.n_pool,
                                                   np.random.default_rng([20260906, i]))
                     for i in range(n)], axis=0)
    data_bits = torch.stack([codes_to_bits_t(pool[i], device) for i in range(n)], dim=0)
    print("[rbm_r3] N=%d  S=%.4f  data pool %s" % (n, d.S, (tuple(data_bits.shape),)))

    rbm = JointRBM(n, n_hidden=args.n_hidden, device=device, seed=args.seed)
    print("[rbm_r3] %d visible x %d hidden -> %d params per image (%d shared + %d latent x N)"
          % (N_VISIBLE, args.n_hidden, rbm.param_count(),
             N_VISIBLE * args.n_hidden + args.n_hidden - N_BH_LAT, D_LAT))
    curve = pcd_train(rbm, data_bits, args.steps, args.lr, args.batch, args.chains,
                      args.seed, args.log_every)
    train_time = time.time() - t0

    latents = rbm.get_latents()
    cov_mode = "full" if n >= 100 else "diag"
    prior = fit_latent_gaussian(latents, cov_mode)
    gen_rng = np.random.default_rng(args.seed + 999)
    lat_gen = sample_latents(prior, int(args.n_gen), gen_rng)
    print("[rbm_r3] latent prior: %s cov, rank %d of %d (whitened cov dev from I = %.3g)"
          % (cov_mode, prior["rank"], D_LAT, prior["whitened_cov_dev"]))

    crops = gibbs_crops(rbm, lat_gen, d.S, args.gen_chains, args.gen_sweeps,
                        args.gen_keep, args.gen_chunk, args.seed)
    offsets = data_lens_r3.sample_offsets(d.train_offsets, crops.shape[0], gen_rng)
    for i in range(crops.shape[0]):
        np.save(os.path.join(args.out, "crops", "generated_%03d.npy" % i), crops[i].astype(np.float32))
        frame = codec10.paste(crops[i], int(offsets[i, 0]), int(offsets[i, 1]))
        np.save(os.path.join(args.out, "generated_%03d.npy" % i), frame.astype(np.float32))

    meta = {
        "arm": args.arm_name, "script": "baseline_rbm_r3.py", "round": 3,
        "param_count": int(rbm.param_count()),
        "n_visible": N_VISIBLE, "n_hidden": int(args.n_hidden), "d_lat": D_LAT,
        "latent_block": "b_v (10) + b_h[:40]", "core_block": "W (10x%d) + b_h[40:]" % args.n_hidden,
        "protocol": "joint PCD-1", "objective": "PCD-1 (contrastive divergence)",
        "n_train": int(n), "n_gen": int(crops.shape[0]), "n_holdout": int(args.n_holdout),
        "S": float(d.S), "seed": int(args.seed), "device": str(args.device),
        "steps": int(args.steps), "lr": float(args.lr), "batch": int(args.batch),
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
        "recon_mse_last": float(curve[-1][1]) if curve else None,
        "train_seconds": float(train_time),
        "wallclock_seconds": float(time.time() - t0),
        "gen_offsets": [[int(a), int(b)] for a, b in offsets.tolist()],
    }
    with open(os.path.join(args.out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True, default=str)
    np.save(os.path.join(args.out, "latents.npy"), latents)
    np.save(os.path.join(args.out, "recon_curve.npy"), np.asarray(curve, dtype=np.float64))
    print("[rbm_r3] wrote %d images to %s (%.1f s)" % (crops.shape[0], args.out, time.time() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
