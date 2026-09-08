#!/usr/bin/env python3
"""Bernoulli RBM baseline for the arXiv 2603.06440 replication.

See INTERFACES.md (repo root) for the shared contracts this module must honour, and
replication_spec.md Sec. 4 for what the paper does and does not specify about the RBM
baseline (Table 7 gives only: same 11 anchor snapshots as IQP, same 18-bit bitstring
representation, ~12k parameters, and a "first-50-flattened-params" latent block that is
adapted per-snapshot and interpolated exactly like the IQP latent trajectory). Everything
else below (visible/hidden split, training algorithm, learning rate, step counts, and the
generation procedure) is a design choice documented inline and in the task report --
NOT a value taken from the paper.

Design choices (paper does not specify these):
  - visible = 18 (one per qubit/bit of the codec.py bitstring), hidden = 630
    -> 18*630 + 18 + 630 = 11,988 params, matching the paper's "RBM (12k)" label.
  - Bernoulli-Bernoulli RBM, trained with Persistent Contrastive Divergence, k=1,
    100 persistent chains, Adam (lr=1e-3 by default, matches --lr).
  - Stage 1 (paper Algorithm 1 step 1): train theta_core with theta_lat held at its
    random initial value, on N_t=5e4 tuples drawn (codec.sample_codes) from the t=1
    anchor's fixed code pool (its full 4096-cell enumeration), 3e4 steps, batch 512.
  - Stage 2 (paper Algorithm 1 step 2): for each SUBSEQUENT anchor (t=100,...,1000),
    warm-start theta_lat from the previous anchor's adapted value and run 3000 steps
    updating ONLY the latent block. Latent block ordering (audit fix, pre-reg Design
    choices table): [b_v (all 18 visible biases), b_h[:32] (first 32 of 630 hidden
    biases)] -- NOT `W.reshape(-1)[:50]` (which would confine all 50 latent params to
    couplings from visible unit 0 alone, i.e. only ever move the LSB of z). This
    ordering makes the RBM's block touch every visible bit, mirroring the IQP latent
    block ({-}, all 18 singles, 31 pairs), per the paper's stated purpose (line 1370):
    "so that differences in performance reflect model inductive bias rather than
    differences in adaptation degrees of freedom." The paper's Algorithm 1 text
    (replication_spec.md:49) says latent adaptation runs "for each subsequent anchor
    time t_j" (i.e. t_2..t_11); by this reading, anchor 1's latent is simply the frozen
    random init used during stage 1, not further adapted. This is a documented
    interpretation, not an explicit statement in the paper.
  - Latent-only updates are implemented by zeroing the gradient outside the latent
    block before the (fresh, per-anchor) Adam step, so momentum never leaks into frozen
    core parameters.
  - Interpolation of theta_lat across unseen tau: piecewise-linear over the sorted
    anchor times, same convention as IQP (Eq. 31).
  - Generation: block Gibbs sampling from random init, n_chains x n_sweeps sweeps,
    keep the last `keep` sweeps of all chains as the sample pool (default
    10^4 chains x 200 sweeps x last 10 sweeps = 10^5 codes, matching Sec. III's
    "e.g. 10^5 samples" scale used for IQP synthesis); codes are aggregated into a
    (64,64) field via codec.aggregate.

Time index t is 1-based throughout (t in {1, ..., 1000}; array index = t - 1), matching
the paper and every other arm. For --data synthetic, the anchor/held-out split comes from
`synth_data.make_split_labels` -- the ONE split source shared by every arm (no independent
seed-offset redraw here).

Python 3.9 compatible (no `X | None`).
"""
import argparse
import json
import os
import time

import numpy as np
import torch

import codec

ANCHOR_TS = [1, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
D_LAT = 50
N_VISIBLE = 18


def load_data_accessors(data_mode, seed):
    """Return (get_field(t) -> (64,64) float32 RAW field, get_pool(t, a_z, b_z) -> int64
    code pool for that (1-based) snapshot). synthetic: from synth_data.py (data is drawn
    at a FIXED data seed, synth_data.SYNTH_DATA_SEED, decoupled from `seed` -- this run's
    model-training stochasticity -- so every arm/seed shares one synthetic ground truth;
    `seed` is accepted for signature symmetry but unused here). d2: from
    data_d2.load_d2_snapshot(t), which returns an object with .field (64,64) and
    .pool_codes (int64, native-grid quantized) -- INTERFACES.md."""
    if data_mode == "synthetic":
        from synth_data import N_SNAPSHOTS_SYNTH, SYNTH_DATA_SEED
        from synth_data import make_synthetic_snapshots as fn
        all_snaps = fn(N_SNAPSHOTS_SYNTH, SYNTH_DATA_SEED)

        def get_field(t):
            return np.asarray(all_snaps[int(t) - 1], dtype=np.float32)

        def get_pool(t, a_z, b_z):
            return codec.field_to_codes_full(get_field(t), a_z, b_z)

        return get_field, get_pool
    elif data_mode == "d2":
        # data_d2.py is owned by agent CORE and is written after the dataset arrives
        # (INTERFACES.md); this branch is a placeholder, not yet exercised by any test.
        try:
            from data_d2 import load_d2_snapshot
        except ImportError:
            raise SystemExit(
                "data_d2.py not available yet (owned by agent CORE); use --data synthetic"
            )

        def get_field(t):
            return np.asarray(load_d2_snapshot(int(t)).field, dtype=np.float32)

        def get_pool(t, a_z, b_z):
            return np.asarray(load_d2_snapshot(int(t)).pool_codes, dtype=np.int64)

        return get_field, get_pool
    elif data_mode == "lens":
        raise ValueError("lens accessors are built by main() via lens_accessors(); "
                         "load_data_accessors() is only for synthetic/d2")
    else:
        raise ValueError("unknown --data " + str(data_mode))


def lens_accessors(lens_class, n_train, n_holdout, sqrt=False):
    """(get_field, get_pool, anchors, holdout_ids) for the DeepLense Model II pilot.

    "snapshot" = image; anchors = the N training images in index order (Algorithm 1 trains
    the core on image 1 then warm-starts the latent through 2..N); held-out = the 200 val
    images (keys 1001..1200). a_z/b_z are FIXED at (0, 1) by data_lens.get_z_range() for
    every class -- never re-derived from a field's own min/max (the d2 a_z/b_z bug).
    """
    import data_lens

    snapshots, anchors, holdout_ids = data_lens.load_lens(
        lens_class, n_train=n_train, n_holdout=n_holdout, sqrt=sqrt
    )

    def get_field(t):
        return np.asarray(snapshots[int(t)].field, dtype=np.float32)

    def get_pool(t, a_z, b_z):
        return np.asarray(snapshots[int(t)].pool_codes, dtype=np.int64)

    return get_field, get_pool, list(anchors), list(holdout_ids)


def codes_to_bits(codes, n_bits=N_VISIBLE):
    codes = np.asarray(codes, dtype=np.int64)
    shifts = np.arange(n_bits)
    bits = ((codes[:, None] >> shifts[None, :]) & 1).astype(np.float32)
    return bits


def bits_to_codes(bits):
    bits = np.asarray(bits, dtype=np.int64)
    n_bits = bits.shape[1]
    weights = 1 << np.arange(n_bits)
    return (bits * weights[None, :]).sum(axis=1)


class RBM(object):
    """Bernoulli-Bernoulli RBM with manual (non-autograd) CD/PCD gradients fed into
    torch.optim.Adam via explicit .grad assignment."""

    def __init__(self, n_visible=N_VISIBLE, n_hidden=630, device="cpu", seed=0):
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.device = device
        W0 = 0.01 * torch.randn(n_visible, n_hidden, generator=g)
        self.W = W0.to(device).clone().requires_grad_(True)
        self.b_v = torch.zeros(n_visible, device=device, requires_grad=True)
        self.b_h = torch.zeros(n_hidden, device=device, requires_grad=True)

    def param_count(self):
        return self.W.numel() + self.b_v.numel() + self.b_h.numel()

    def get_latent(self, d_lat=D_LAT):
        """[b_v (all n_visible), b_h[:d_lat - n_visible]] -- see module docstring."""
        with torch.no_grad():
            n_bh = d_lat - self.b_v.numel()
            return torch.cat([self.b_v, self.b_h[:n_bh]]).clone()

    def set_latent(self, theta_lat):
        with torch.no_grad():
            theta_lat = theta_lat.to(self.b_v.device)
            n_bv = self.b_v.numel()
            self.b_v.copy_(theta_lat[:n_bv])
            n_bh = theta_lat.numel() - n_bv
            self.b_h[:n_bh] = theta_lat[n_bv:n_bv + n_bh].to(self.b_h.device)

    def p_h_given_v(self, v):
        return torch.sigmoid(v @ self.W + self.b_h)

    def p_v_given_h(self, h):
        return torch.sigmoid(h @ self.W.t() + self.b_v)

    def sample_h(self, v, gen):
        p = self.p_h_given_v(v)
        u = torch.rand(p.shape, generator=gen, device=p.device)
        return (u < p).float(), p

    def sample_v(self, h, gen):
        p = self.p_v_given_h(h)
        u = torch.rand(p.shape, generator=gen, device=p.device)
        return (u < p).float(), p


def pcd_train(rbm, data_codes, n_steps, lr, batch, n_chains, latent_only, rng_np, torch_gen, device):
    """PCD-1 training. If latent_only, gradients outside the latent block ([b_v, b_h[:D_LAT
    - n_visible]], see RBM.get_latent) are zeroed before the Adam step (W is fully frozen),
    and a fresh Adam optimizer is used (no cross-anchor / cross-stage momentum
    contamination)."""
    data_bits = codes_to_bits(data_codes)
    data_bits_t = torch.from_numpy(data_bits).to(device)
    n_data = data_bits_t.shape[0]

    v_chain = (torch.rand(n_chains, rbm.n_visible, generator=torch_gen, device=device) < 0.5).float()

    params = [rbm.W, rbm.b_v, rbm.b_h]
    opt = torch.optim.Adam(params, lr=lr)
    n_bh_lat = D_LAT - rbm.b_v.numel()  # 32: number of hidden biases inside the latent block

    for _ in range(n_steps):
        idx = rng_np.integers(0, n_data, size=batch)
        v_data = data_bits_t[idx]

        h_data_p = rbm.p_h_given_v(v_data)
        pos_W = v_data.t() @ h_data_p / batch
        pos_bv = v_data.mean(0)
        pos_bh = h_data_p.mean(0)

        h0, _ = rbm.sample_h(v_chain, torch_gen)
        v1, _ = rbm.sample_v(h0, torch_gen)
        h1_p = rbm.p_h_given_v(v1)
        neg_W = v1.t() @ h1_p / n_chains
        neg_bv = v1.mean(0)
        neg_bh = h1_p.mean(0)
        v_chain = v1.detach()

        dW = (pos_W - neg_W).detach()
        dbv = (pos_bv - neg_bv).detach()
        dbh = (pos_bh - neg_bh).detach()

        if latent_only:
            dW = torch.zeros_like(dW)
            dbh = dbh.clone()
            dbh[n_bh_lat:] = 0.0
            # dbv (all n_visible) stays fully unfrozen: it is entirely inside the latent block.

        opt.zero_grad()
        rbm.W.grad = -dW
        rbm.b_v.grad = -dbv
        rbm.b_h.grad = -dbh
        opt.step()

    return v_chain


def interp_latent(tau, anchors_sorted, latents_arr):
    if tau <= anchors_sorted[0]:
        return latents_arr[0]
    if tau >= anchors_sorted[-1]:
        return latents_arr[-1]
    idx = int(np.searchsorted(anchors_sorted, tau)) - 1
    idx = int(np.clip(idx, 0, len(anchors_sorted) - 2))
    t0, t1 = anchors_sorted[idx], anchors_sorted[idx + 1]
    w = (tau - t0) / float(t1 - t0)
    return (1 - w) * latents_arr[idx] + w * latents_arr[idx + 1]


def generate_codes(rbm, n_chains, n_sweeps, keep, torch_gen, device):
    v = (torch.rand(n_chains, rbm.n_visible, generator=torch_gen, device=device) < 0.5).float()
    kept = []
    keep_from = max(n_sweeps - keep, 0)
    for sweep in range(n_sweeps):
        h, _ = rbm.sample_h(v, torch_gen)
        v, _ = rbm.sample_v(h, torch_gen)
        if sweep >= keep_from:
            kept.append(v.detach().cpu().numpy())
    if not kept:
        kept.append(v.detach().cpu().numpy())
    bits = np.concatenate(kept, axis=0)
    return bits_to_codes(bits)


def parse_args():
    p = argparse.ArgumentParser(description="RBM baseline (arXiv 2603.06440 replication)")
    p.add_argument("--data", choices=["synthetic", "d2", "lens"], default="synthetic")
    p.add_argument("--out", default="outputs/rbm")
    # --- lens pilot (docs/IQP_LENS_PILOT_PREREGISTRATION.md) --------------------------
    p.add_argument("--lens-class", type=str, default=None,
                   help="--data lens: which DeepLense Model II class (axion/cdm/no_sub)")
    p.add_argument("--n-train", type=int, default=11,
                   help="--data lens: N training images = N anchors (11 or 100)")
    p.add_argument("--lens-sqrt", action="store_true",
                   help="--data lens: quantize sqrt(intensity); generated fields are squared "
                        "back to intensity space on write")
    p.add_argument("--arm-name", type=str, default=None,
                   help="override meta.json's `arm` label (evaluate_lens.py groups rows by it)")
    p.add_argument("--synth-mode", choices=["interp", "gaussian-latent"], default="interp",
                   help="'interp' (default, paper Eq. 31, time-indexed data); 'gaussian-latent' "
                        "(lens pilot, no time axis): fit N(mu, diag sigma^2) to the N trained "
                        "latents and draw --n-gen latents from it")
    p.add_argument("--n-gen", type=int, default=200,
                   help="--synth-mode gaussian-latent: number of images to synthesize")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-tuples", type=int, default=50000, help="N_t tuples per anchor")
    p.add_argument("--steps-core", type=int, default=30000)
    p.add_argument("--steps-lat", type=int, default=3000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=630)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--n-chains-pcd", type=int, default=100)
    p.add_argument("--gen-chains", type=int, default=10000)
    p.add_argument("--gen-sweeps", type=int, default=200)
    p.add_argument("--gen-keep", type=int, default=10)
    p.add_argument("--anchors-max", type=int, default=0, help="0 = use all 11 anchors")
    p.add_argument("--n-holdout", type=int, default=200)
    return p.parse_args()


def main():
    args = parse_args()
    if args.device == "cpu":
        # Many-core login/HPC nodes make torch's default (= all cores) intra-op thread
        # pool a net loss for these tiny (18- or 630-wide) tensors: thread-sync overhead
        # dominates actual compute. Cap it so CPU smoke tests stay in the seconds range.
        torch.set_num_threads(min(4, os.cpu_count() or 1))
    t_start = time.time()
    os.makedirs(args.out, exist_ok=True)

    rng_np = np.random.default_rng(args.seed)
    torch_gen = torch.Generator(device=args.device).manual_seed(args.seed)
    device = args.device

    if args.data == "lens":
        if not args.lens_class:
            import data_lens as _dl
            raise SystemExit("--data lens requires --lens-class (one of %s)" % _dl.get_class_names())
        get_field, get_pool, lens_anchors, lens_holdout = lens_accessors(
            args.lens_class, args.n_train, args.n_holdout, sqrt=args.lens_sqrt
        )
    else:
        get_field, get_pool = load_data_accessors(args.data, args.seed)
        lens_anchors, lens_holdout = None, None

    holdout_ts = None  # resolved below; synthetic resolves it here already (shared split)
    if args.data == "lens":
        anchors = list(lens_anchors)
        if args.anchors_max > 0:
            anchors = anchors[: args.anchors_max]
        holdout_ts = list(lens_holdout)
    elif args.data == "synthetic":
        from synth_data import N_SNAPSHOTS_SYNTH, SYNTH_DATA_SEED, make_split_labels
        anchors_t, holdout_t = make_split_labels(
            N_SNAPSHOTS_SYNTH, seed=SYNTH_DATA_SEED, n_holdout=args.n_holdout,
            anchors_max=args.anchors_max if args.anchors_max > 0 else None,
        )
        anchors = [int(a) for a in anchors_t.tolist()]
        holdout_ts = [int(t) for t in holdout_t.tolist()]
    else:
        anchors = list(ANCHOR_TS) if args.anchors_max <= 0 else ANCHOR_TS[: args.anchors_max]
    if len(anchors) < 2:
        raise SystemExit("need at least 2 anchors for latent interpolation")
    anchor_fields = {t: get_field(t) for t in anchors}

    if args.data == "lens":
        # FIXED (0, 1) for every class (data_lens.get_z_range) -- never re-derived.
        from data_lens import get_z_range as _lens_z_range
        a_z, b_z = _lens_z_range()
    elif args.data == "d2":
        # a_z/b_z MUST be the exact bounds load_d2()/load_d2_snapshot() used to
        # quantize/encode pool_codes (data_d2.get_z_range, native z_val over the 11
        # anchors) -- NOT re-derived from `.field`, whose 64x64-bin averaging narrows
        # the range (~18.8% here) and would decode/aggregate/quantize downstream on a
        # different scale than the one used to encode.
        from data_d2 import get_z_range
        a_z, b_z = get_z_range("mid")
    else:
        all_vals = np.concatenate([f.ravel() for f in anchor_fields.values()])
        a_z, b_z = float(all_vals.min()), float(all_vals.max())
        if b_z <= a_z:
            b_z = a_z + 1.0

    rbm = RBM(n_visible=N_VISIBLE, n_hidden=args.hidden, device=device, seed=args.seed)

    # Stage 1: train theta_core on anchors[0] (t=1 by default), theta_lat left at its
    # random initial value. Training tuples are drawn (codec.sample_codes) from the
    # anchor's fixed code pool (its full 4096-cell enumeration), not re-quantized per draw.
    t0 = anchors[0]
    codes0 = codec.sample_codes(get_pool(t0, a_z, b_z), args.n_tuples, np.random.default_rng(args.seed))
    pcd_train(
        rbm, codes0, n_steps=args.steps_core, lr=args.lr, batch=min(args.batch, args.n_tuples),
        n_chains=args.n_chains_pcd, latent_only=False, rng_np=rng_np, torch_gen=torch_gen, device=device,
    )

    latent_by_t = {}
    latent_by_t[t0] = rbm.get_latent().cpu().numpy()
    prev_latent = rbm.get_latent()

    # Stage 2: adapt theta_lat only, per subsequent anchor, warm-started.
    for t in anchors[1:]:
        codes_t = codec.sample_codes(get_pool(t, a_z, b_z), args.n_tuples, np.random.default_rng(args.seed + t))
        rbm.set_latent(prev_latent)
        pcd_train(
            rbm, codes_t, n_steps=args.steps_lat, lr=args.lr, batch=min(args.batch, args.n_tuples),
            n_chains=args.n_chains_pcd, latent_only=True, rng_np=rng_np, torch_gen=torch_gen, device=device,
        )
        prev_latent = rbm.get_latent()
        latent_by_t[t] = prev_latent.cpu().numpy()

    anchors_sorted = sorted(latent_by_t.keys())
    latents_arr = np.stack([latent_by_t[t] for t in anchors_sorted])

    if holdout_ts is None:
        # d2: use data_d2's own fixed 200-tau holdout split (INTERFACES.md / pre-registration
        # "Shared by every arm"), truncated to --n-holdout -- the same source and truncation
        # convention train_iqp.py / controls.py use, so held-out tau match across arms.
        from data_d2 import get_holdout_taus
        holdout_ts = get_holdout_taus()[: args.n_holdout] if args.n_holdout > 0 else []

    lat_gauss = None
    if args.synth_mode == "gaussian-latent":
        # No time axis (lens pilot): fit N(mu, diag sigma^2) to the N trained latents and
        # draw --n-gen latents from it, one image each (lens_synth.py docstring).
        import lens_synth

        mu, sigma = lens_synth.fit_gaussian_latent(latents_arr)
        lat_gauss = lens_synth.latent_stats(mu, sigma)
        rng_synth = np.random.default_rng(args.seed + 4242)
        lat_draws = lens_synth.sample_gaussian_latents(mu, sigma, args.n_gen, rng_synth)
        for i in range(args.n_gen):
            rbm.set_latent(torch.from_numpy(lat_draws[i].astype(np.float32)).to(device))
            codes = generate_codes(rbm, args.gen_chains, args.gen_sweeps, args.gen_keep, torch_gen, device)
            lens_synth.write_pair(args.out, i, codes, a_z, b_z, sqrt=bool(args.lens_sqrt))
    else:
        for tau in holdout_ts:
            theta_lat_tau = interp_latent(tau, anchors_sorted, latents_arr)
            rbm.set_latent(torch.from_numpy(theta_lat_tau.astype(np.float32)).to(device))
            codes = generate_codes(rbm, args.gen_chains, args.gen_sweeps, args.gen_keep, torch_gen, device)
            field = codec.aggregate(codes, a_z, b_z)
            np.save(os.path.join(args.out, "generated_t{}.npy".format(tau)), field)

    meta = {
        "param_count": rbm.param_count(),
        "seed": args.seed,
        "train_snapshots": anchors_sorted,
        "wallclock_s": time.time() - t_start,
        "arm": "RBM-%dk" % int(round(rbm.param_count() / 1000.0)),
        "data": args.data,
        "hidden": args.hidden,
        "n_tuples": args.n_tuples,
        "steps_core": args.steps_core,
        "steps_lat": args.steps_lat,
        "lr": args.lr,
        "n_chains_pcd": args.n_chains_pcd,
        "gen_chains": args.gen_chains,
        "gen_sweeps": args.gen_sweeps,
        "gen_keep": args.gen_keep,
        "a_z": a_z,
        "b_z": b_z,
        "n_holdout": len(holdout_ts),
        "holdout_ts": holdout_ts,
        "d_lat": D_LAT,
        "latent_order": "[b_v(18), b_h[:32]]",
        "data_seed": None,
    }
    meta["synth_mode"] = args.synth_mode
    if args.data == "synthetic":
        from synth_data import SYNTH_DATA_SEED as _sds
        meta["data_seed"] = int(_sds)
    if args.data == "lens":
        import data_lens

        meta["arm"] = "RBM-%dk%s" % (int(round(rbm.param_count() / 1000.0)),
                                     "-sqrt" if args.lens_sqrt else "")
        meta["class_names"] = data_lens.get_class_names()
        meta["lens_class"] = args.lens_class
        meta["lens_sqrt"] = bool(args.lens_sqrt)
        meta["n_train"] = len(anchors_sorted)
        meta["train_snapshots"] = len(anchors_sorted)
        meta["anchors"] = list(anchors_sorted)
        meta["n_gen"] = int(args.n_gen) if args.synth_mode == "gaussian-latent" else len(holdout_ts)
    if lat_gauss is not None:
        meta.update(lat_gauss)
    if getattr(args, "arm_name", None):
        meta["arm"] = str(args.arm_name)

    if args.synth_mode == "gaussian-latent":
        import lens_synth

        lens_synth.write_metas(args.out, meta)
    else:
        with open(os.path.join(args.out, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    print("RBM done: param_count={} n_anchors={} holdout={} synth_mode={} wallclock={:.1f}s".format(
        meta["param_count"], len(anchors_sorted), len(holdout_ts), args.synth_mode, meta["wallclock_s"]
    ))


if __name__ == "__main__":
    main()
