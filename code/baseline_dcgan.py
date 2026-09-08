#!/usr/bin/env python3
"""DCGAN baseline for the arXiv 2603.06440 replication.

See INTERFACES.md (repo root) for shared contracts, and replication_spec.md Sec. 5 for
what the paper does and does not specify (Table 8 gives only: trained directly on
quantized 2D turbulence fields (not bitstrings), three capacities ~51k/0.2M/1.2M
params, latent dim 128, two data regimes (11 data-matched / 100 data-rich snapshots)
plus a 10/30/50/70/90/100 scaling study, architecture family = Radford et al. 2015
DCGAN). Layer widths, loss, optimizer, step count, batch size, and init are all design
choices documented inline and in the task report -- NOT values taken from the paper.

Design choices (paper does not specify these):
  - Standard Radford et al. (2015) DCGAN topology for 64x64, single-channel (Z-only)
    output: latent z (128,1,1) -> ConvTranspose2d stack 4x4 -> 8 -> 16 -> 32 -> 64,
    BatchNorm+ReLU on hidden layers, Tanh output. Discriminator mirrors it with
    Conv2d stride-2 + LeakyReLU(0.2) (+BatchNorm except first layer), Sigmoid output.
  - Channel width ngf tuned (by exhaustive count-off, not analytically) to hit the
    three target generator parameter counts within +/-10%:
      tier "51k"  -> ngf=3,  generator params = 55,339  (+8.5%)
      tier "0.2m" -> ngf=9,  generator params = 202,303 (+1.2%)
      tier "1.2m" -> ngf=32, generator params = 1,213,889 (+1.2%)
    Discriminator uses ndf = ngf (mirrored scaling per the task instructions); its
    param count is NOT part of the target and is only recorded in meta.json.
  - Training fields are QUANTIZED first (paper line 1377 / Table 8: "trained directly on
    the quantized 2D turbulence fields") -- each field passes through its snapshot's own
    full-grid code pool and that pool's aggregate (codec.field_to_codes_full +
    codec.aggregate, i.e. z rounded to the 6-bit bin centre used by every other arm's
    codec-based reconstruction) BEFORE the [-1, 1] affine normalization. Audit fix:
    previously this module trained on the raw, continuous field, an unrecorded and
    unfair precision advantage over IQP/RBM/CTRL-marginal (which were always capped at
    6-bit z through the codec).
  - a_z/b_z (the quantization/normalization bounds) come from the 11 CANONICAL anchor
    snapshots ONLY, exactly like every other arm -- not from whichever `--n-train`
    snapshots this run actually trains on (100-snapshot / scaling regimes still
    quantize and normalize against the same 11-anchor a_z/b_z), per the
    pre-registration's "one shared value" rule (item 7 of the code-vs-paper audit).
  - Loss: standard non-saturating BCE GAN loss. Adam(lr=2e-4, betas=(0.5, 0.999)) for
    both G and D. batch = min(16, N_train). No data augmentation.
  - Weight init: N(0, 0.02) on conv/conv-transpose weights and on BatchNorm weight
    (bias 0), the standard DCGAN convention from Radford et al. 2015.
  - Model selection: no early stopping, last checkpoint after --steps generator
    updates (recorded in meta.json), per the task instructions.
  - Data regime: --n-train 11 uses the same 11 anchor times as IQP/RBM/CTRL (from
    synth_data.make_split_labels for synthetic; ANCHOR_TS for d2); any other --n-train
    (100, or the 10/30/50/70/90 scaling values) picks N evenly-spaced 1-based integer t
    in [1, n_snapshots] via rounded linspace. Which specific 100 snapshots make up
    "data-rich" is not given in the paper (replication_spec.md Sec. 12 item 12);
    evenly-spaced was the paper's own recommendation for consistency with the
    anchor-time spirit used for IQP/RBM.
  - Evaluation samples are drawn unconditionally (DCGAN has no time-conditioning
    mechanism in this paper) and compared distributionally to the held-out ground
    truth set, per replication_spec.md Sec. 12 item 13 ("most literal reading of
    'same held-out protocol'").

Time index t is 1-based throughout (t in {1, ..., 1000}; array index = t - 1), matching
the paper and every other arm.

Python 3.9 compatible (no `X | None`).
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

import codec

ANCHOR_TS = [1, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
NZ = 128

TIERS = {
    "51k": {"ngf": 3, "target": 51000},
    "0.2m": {"ngf": 9, "target": 200000},
    "1.2m": {"ngf": 32, "target": 1200000},
}


def load_snapshot_getter(data_mode, seed):
    """Return get(t) -> (64,64) float32 RAW field, 1-based t. synthetic: from
    synth_data.py (data is drawn at the FIXED data seed synth_data.SYNTH_DATA_SEED,
    decoupled from `seed` -- this run's model-training stochasticity -- so every arm/seed
    shares one synthetic ground truth; `seed` is accepted for signature symmetry but
    unused here). d2: from data_d2.load_d2_snapshot(t).field -- INTERFACES.md."""
    if data_mode == "synthetic":
        from synth_data import N_SNAPSHOTS_SYNTH, SYNTH_DATA_SEED
        from synth_data import make_synthetic_snapshots as fn
        all_snaps = fn(N_SNAPSHOTS_SYNTH, SYNTH_DATA_SEED)

        def get(t):
            return np.asarray(all_snaps[int(t) - 1], dtype=np.float32)

        return get
    elif data_mode == "d2":
        # data_d2.py is owned by agent CORE and is written after the dataset arrives
        # (INTERFACES.md); this branch is a placeholder, not yet exercised by any test.
        try:
            from data_d2 import load_d2_snapshot
        except ImportError:
            raise SystemExit(
                "data_d2.py not available yet (owned by agent CORE); use --data synthetic"
            )

        def get(t):
            return np.asarray(load_d2_snapshot(int(t)).field, dtype=np.float32)

        return get
    else:
        raise ValueError("unknown --data " + str(data_mode))


def pick_times(n_train, anchors=None, n_snapshots=1000):
    """1-based t. n_train == 11 uses the SAME 11 anchors as every other arm (`anchors`,
    from synth_data.make_split_labels for synthetic / ANCHOR_TS for d2); any other
    n_train (100, or the 10/30/50/70/90 scaling values) picks n_train evenly-spaced t in
    [1, n_snapshots] via rounded linspace -- which specific 100 is not given by the
    paper (pre-registration Design choices table); evenly-spaced matches the
    anchor-time spirit used for the 11-snapshot regime."""
    if n_train == 11 and anchors is not None:
        return [int(a) for a in anchors]
    ts = np.round(np.linspace(1, n_snapshots, n_train)).astype(int)
    return sorted(set(int(t) for t in ts.tolist()))


def make_generator(ngf, nz=NZ):
    return nn.Sequential(
        nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False), nn.BatchNorm2d(ngf * 8), nn.ReLU(True),
        nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 4), nn.ReLU(True),
        nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 2), nn.ReLU(True),
        nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf), nn.ReLU(True),
        nn.ConvTranspose2d(ngf, 1, 4, 2, 1, bias=True), nn.Tanh(),
    )


def make_discriminator(ndf):
    return nn.Sequential(
        nn.Conv2d(1, ndf, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, True),
        nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 2), nn.LeakyReLU(0.2, True),
        nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 4), nn.LeakyReLU(0.2, True),
        nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 8), nn.LeakyReLU(0.2, True),
        nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=True), nn.Sigmoid(),
    )


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def parse_args():
    p = argparse.ArgumentParser(description="DCGAN baseline (arXiv 2603.06440 replication)")
    p.add_argument("--data", choices=["synthetic", "d2", "lens"], default="synthetic")
    p.add_argument("--tier", choices=list(TIERS.keys()), default="51k")
    p.add_argument("--n-train", type=int, default=11)
    # --- lens pilot (docs/IQP_LENS_PILOT_PREREGISTRATION.md) --------------------------
    p.add_argument("--lens-class", type=str, default=None,
                   help="--data lens: which DeepLense Model II class (axion/cdm/no_sub)")
    p.add_argument("--lens-sqrt", action="store_true",
                   help="--data lens: train on sqrt(intensity)-quantized fields; generated "
                        "fields are squared back to intensity space on write")
    p.add_argument("--steps", type=int, default=20000, help="generator update steps")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="outputs/dcgan")
    p.add_argument("--device", default="cpu")
    p.add_argument("--n-gen", type=int, default=200)
    return p.parse_args()


def main():
    args = parse_args()
    if args.device == "cpu":
        # Many-core login/HPC nodes make torch's default (= all cores) intra-op thread
        # pool a net loss for these tiny conv layers: thread-sync overhead dominates
        # actual compute. Cap it so CPU smoke tests stay in the seconds range.
        torch.set_num_threads(min(4, os.cpu_count() or 1))
    t_start = time.time()
    os.makedirs(args.out, exist_ok=True)
    device = args.device

    torch.manual_seed(args.seed)
    torch_gen = torch.Generator(device=device).manual_seed(args.seed)
    rng_np = np.random.default_rng(args.seed)

    lens_fields = None
    if args.data == "lens":
        # "snapshot" = image; the N training images ARE the training set (no time axis, so
        # no pick_times / linspace selection). a_z/b_z are FIXED at (0, 1) for every class
        # (data_lens.get_z_range) -- the same bounds every other lens arm quantizes with.
        import data_lens

        if not args.lens_class:
            raise SystemExit("--data lens requires --lens-class (one of %s)"
                             % data_lens.get_class_names())
        a_z, b_z = data_lens.get_z_range()
        # already the QUANTIZED fields (data_lens builds .field = aggregate(pool_codes))
        lens_fields = data_lens.raw_training_images(args.lens_class, args.n_train,
                                                    sqrt=args.lens_sqrt)
        train_ts = list(range(1, lens_fields.shape[0] + 1))

        def get_field(t):
            return np.asarray(lens_fields[int(t) - 1], dtype=np.float32)
    else:
        get_field = load_snapshot_getter(args.data, args.seed)

    if args.data == "lens":
        pass
    elif args.data == "synthetic":
        from synth_data import (
            N_SNAPSHOTS_SYNTH, SYNTH_DATA_SEED, anchor_a_z_b_z, make_split_labels,
            make_synthetic_snapshots,
        )
        anchors_t, _ = make_split_labels(N_SNAPSHOTS_SYNTH, seed=SYNTH_DATA_SEED, n_holdout=0)
        all_snaps = make_synthetic_snapshots(N_SNAPSHOTS_SYNTH, seed=SYNTH_DATA_SEED)
        a_z, b_z = anchor_a_z_b_z(all_snaps, anchors_t)
        train_ts = pick_times(args.n_train, anchors=anchors_t.tolist(), n_snapshots=N_SNAPSHOTS_SYNTH)
    else:
        # a_z/b_z MUST be the exact bounds load_d2()/load_d2_snapshot() used to
        # quantize/encode pool_codes (data_d2.get_z_range, native z_val over the 11
        # anchors) -- NOT re-derived from `.field`, whose 64x64-bin averaging narrows
        # the range (~18.8% here) and would decode/aggregate/quantize downstream on a
        # different scale than the one used to encode.
        from data_d2 import get_z_range

        a_z, b_z = get_z_range("mid")
        train_ts = pick_times(args.n_train, anchors=ANCHOR_TS, n_snapshots=1000)

    def quantized_field(t):
        pool = codec.field_to_codes_full(get_field(t), a_z, b_z)
        return codec.aggregate(pool, a_z, b_z)

    if lens_fields is not None:
        # data_lens already returns the pool-aggregated (quantized) field; re-quantizing it
        # is idempotent but pointless, so use it directly.
        fields = np.asarray(lens_fields, dtype=np.float32)
    else:
        fields = np.stack([quantized_field(t) for t in train_ts]).astype(np.float32)  # (N,64,64), quantized

    lo, hi = a_z, b_z  # the SAME bounds every other arm uses, not this run's own fields.min()/max()
    norm = 2.0 * (fields - lo) / (hi - lo) - 1.0
    data_t = torch.from_numpy(norm).unsqueeze(1).to(device)  # (N,1,64,64)
    n_train_actual = data_t.shape[0]

    ngf = TIERS[args.tier]["ngf"]
    G = make_generator(ngf).to(device)
    D = make_discriminator(ngf).to(device)
    G.apply(weights_init)
    D.apply(weights_init)

    g_params = count_params(G)
    d_params = count_params(D)
    target = TIERS[args.tier]["target"]
    tol = 0.10
    within_tol = abs(g_params - target) <= tol * target
    print("DCGAN tier={} ngf={} generator_params={} target={} within_10pct={}".format(
        args.tier, ngf, g_params, target, within_tol
    ))

    opt_g = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))
    bce = nn.BCELoss()

    batch = min(16, n_train_actual)

    for step in range(args.steps):
        idx = rng_np.integers(0, n_train_actual, size=batch)
        real = data_t[idx]

        z = torch.randn(batch, NZ, 1, 1, generator=torch_gen, device=device)
        fake = G(z)

        ones = torch.ones(batch, device=device)
        zeros = torch.zeros(batch, device=device)

        d_real = D(real).view(-1)
        d_fake = D(fake.detach()).view(-1)
        loss_d = bce(d_real, ones) + bce(d_fake, zeros)
        opt_d.zero_grad()
        loss_d.backward()
        opt_d.step()

        d_fake2 = D(fake).view(-1)
        loss_g = bce(d_fake2, ones)
        opt_g.zero_grad()
        loss_g.backward()
        opt_g.step()

    G.eval()
    with torch.no_grad():
        z = torch.randn(args.n_gen, NZ, 1, 1, generator=torch_gen, device=device)
        samples = G(z).cpu().numpy()[:, 0]  # (n_gen, 64, 64) in [-1,1]
    fields_gen = (samples + 1.0) / 2.0 * (hi - lo) + lo
    if args.data == "lens" and args.lens_sqrt:
        import data_lens

        fields_gen = np.stack([data_lens.invert_intensity(f, sqrt=True) for f in fields_gen])
    for i in range(args.n_gen):
        np.save(os.path.join(args.out, "generated_{:03d}.npy".format(i)), fields_gen[i].astype(np.float32))

    meta = {
        "param_count": g_params,
        "disc_param_count": d_params,
        "seed": args.seed,
        "train_snapshots": train_ts,
        "wallclock_s": time.time() - t_start,
        "arm": "DCGAN-%s" % args.tier,
        "data": args.data,
        "tier": args.tier,
        "ngf": ngf,
        "nz": NZ,
        "n_train": n_train_actual,
        "steps": args.steps,
        "lr": args.lr,
        "batch": batch,
        "a_z": a_z,
        "b_z": b_z,
        "lo": lo,
        "hi": hi,
        "n_gen": args.n_gen,
        "within_10pct_of_target": bool(within_tol),
        "quantized_training_data": True,
        "data_seed": None,
    }
    if args.data == "synthetic":
        from synth_data import SYNTH_DATA_SEED as _sds
        meta["data_seed"] = int(_sds)
    if args.data == "lens":
        import data_lens

        meta["arm"] = "DCGAN-%s%s" % (args.tier, "-sqrt" if args.lens_sqrt else "")
        meta["class_names"] = data_lens.get_class_names()
        meta["lens_class"] = args.lens_class
        meta["lens_sqrt"] = bool(args.lens_sqrt)
        meta["train_snapshots"] = n_train_actual
        meta["synth_mode"] = "unconditional"
        meta["aggregation"] = "n/a"
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("DCGAN done: n_train={} n_gen={} wallclock={:.1f}s".format(
        n_train_actual, args.n_gen, meta["wallclock_s"]
    ))


if __name__ == "__main__":
    main()
