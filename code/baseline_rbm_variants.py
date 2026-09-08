#!/usr/bin/env python3
"""RBM baseline-alignment variants (2026-09-06 addendum to the arXiv 2603.06440 replication).

Our RBM/DCGAN baselines (baseline_rbm.py / baseline_dcgan.py) beat the paper's reported
Table 7/8 numbers by a wide margin. The paper gives only an architecture-level description
of the RBM baseline (Table 7: same 11 anchors, same 18-bit bitstring, ~12k params, a
"first-50-flattened-params" latent block adapted per-snapshot) with NO training recipe. This
module does NOT modify baseline_rbm.py -- it imports its data loading / codec / generation
machinery unchanged and adds two independently-selectable axes to sweep, to see which
combination (if any) reproduces the paper's baseline numbers:

  --latent {biases,literal}
      biases  = baseline_rbm.RBM.get_latent/set_latent, UNCHANGED: [b_v(18), b_h[:32]].
                (baseline_rbm.py's own docstring's chosen interpretation -- touches every
                visible bit, per the paper's stated purpose that the two arms' latent DOF
                be comparable to IQP's.)
      literal = the most literal reading of the paper's "first 50 flattened parameters"
                sentence: `W.reshape(-1)[:50]` in row-major order over a (18, 630) W, i.e.
                exactly `W[0, :50]` -- all 50 latent params are couplings from visible unit 0
                (qubit 0 / kz bit 0) alone. This is the interpretation baseline_rbm.py's own
                docstring explicitly REJECTED (it confines every latent DOF to a single
                visible bit); reintroduced here, unmodified in the original file, purely to
                test whether the paper meant this literal (if narrower) reading.

  --recipe {pcd_adam,cd1_sgd}
      pcd_adam = baseline_rbm.py's existing recipe, UNCHANGED: PCD-1 (negative phase from a
                 persistent chain that survives across training steps within one call),
                 100 chains by default, Adam lr=1e-3, batch 512, 30000 stage-1 steps.
      cd1_sgd  = classic Hinton (2002) CD-1: negative phase = ONE Gibbs step reconstructed
                 FROM the current data minibatch (no persistent chain), SGD lr=0.05,
                 momentum 0, batch 64. Stage-1 (theta_core) step count defaults to
                 ~--epochs-core (30) epochs over --n-tuples (N_t=5e4 default -> 50000//64*30
                 = 23400 steps); stage-2 (per-anchor theta_lat adaptation, --steps-lat)
                 keeps the SAME default (3000) as pcd_adam -- only the algorithm changes for
                 stage 2, not that step budget.

The two axes are fully independent (any (latent, recipe) pair is legal); when
--latent biases --recipe pcd_adam (both defaults) this script reproduces baseline_rbm.py's
own arm exactly, seed for seed -- a built-in sanity check against results_d2_main.jsonl's
"RBM-12k" row.

meta.json's "arm" is "RBM-<params/1000>k-<latent>-<recipe>" so evaluate.py's
(arm, train_snapshots) grouping never merges variants together.

Everything else (11 anchors, N_t=5e4 tuples/anchor drawn via codec.sample_codes, a_z/b_z
from the 11 canonical anchors, 200-tau holdout, generation via block Gibbs sampling
10^4 chains x 200 sweeps x last 10 kept -> 10^5 codes -> codec.aggregate) is identical to
baseline_rbm.py's main() and is reused from there unchanged.

Python 3.9 compatible (no `X | None`).
"""
import argparse
import json
import os
import time

import numpy as np
import torch

import codec
from baseline_rbm import (
    ANCHOR_TS, D_LAT, N_VISIBLE, RBM, codes_to_bits, generate_codes, interp_latent,
    load_data_accessors,
)

RECIPE_DEFAULTS = {
    "pcd_adam": {"lr": 1e-3, "batch": 512},
    "cd1_sgd": {"lr": 0.05, "batch": 64},
}


class RBMLiteral(RBM):
    """Latent block = W.reshape(-1)[:D_LAT] (row-major) == W[0, :D_LAT] for an
    (n_visible, n_hidden) W -- the literal reading of the paper's "first 50 flattened
    parameters" sentence, rejected by baseline_rbm.py's own docstring because it confines
    every latent DOF to visible unit 0. See module docstring. Everything else (p_h_given_v,
    p_v_given_h, sample_h, sample_v, param_count) is inherited unchanged from RBM."""

    def get_latent(self, d_lat=D_LAT):
        with torch.no_grad():
            return self.W.reshape(-1)[:d_lat].clone()

    def set_latent(self, theta_lat):
        with torch.no_grad():
            theta_lat = theta_lat.to(self.W.device)
            # W is contiguous by construction (RBM.__init__: 0.01*randn(...).to(device).clone())
            # and only ever updated in-place by an optimizer step, so .view(-1) is a genuine
            # view (no copy) -- this write goes straight into W's storage.
            flat = self.W.view(-1)
            flat[: theta_lat.numel()] = theta_lat


def make_rbm(latent, n_visible, n_hidden, device, seed):
    cls = RBMLiteral if latent == "literal" else RBM
    return cls(n_visible=n_visible, n_hidden=n_hidden, device=device, seed=seed)


def zero_non_latent_grad(rbm, dW, dbv, dbh, latent):
    """Zero the gradient outside the latent block before a latent-only Adam/SGD step, so
    core parameters never move during stage 2 -- generalizes baseline_rbm.pcd_train's
    masking (which hardcodes the "biases" layout) to both latent block layouts."""
    if latent == "biases":
        n_bh_lat = D_LAT - rbm.b_v.numel()  # 32: hidden biases inside [b_v, b_h[:32]]
        dW = torch.zeros_like(dW)
        dbh = dbh.clone()
        dbh[n_bh_lat:] = 0.0
        # dbv (all n_visible) is entirely inside this latent block -- stays unfrozen.
    elif latent == "literal":
        mask = torch.zeros_like(dW)
        mask.view(-1)[:D_LAT] = 1.0
        dW = dW * mask
        dbv = torch.zeros_like(dbv)
        dbh = torch.zeros_like(dbh)
    else:
        raise ValueError("unknown --latent " + str(latent))
    return dW, dbv, dbh


def train_rbm_variant(rbm, data_codes, n_steps, recipe, latent, batch, n_chains, lr, momentum,
                       latent_only, rng_np, torch_gen, device):
    """Generalizes baseline_rbm.pcd_train over two independent axes:
      recipe: pcd_adam (persistent-chain negative phase, Adam) or cd1_sgd (negative phase
              reconstructed from the CURRENT data minibatch -- classic CD-1 -- SGD).
      latent: which parameters a latent_only step is allowed to move (zero_non_latent_grad).
    When recipe=="pcd_adam" and latent=="biases" this reproduces baseline_rbm.pcd_train's
    math and RNG draw order exactly."""
    data_bits = codes_to_bits(data_codes)
    data_bits_t = torch.from_numpy(data_bits).to(device)
    n_data = data_bits_t.shape[0]

    params = [rbm.W, rbm.b_v, rbm.b_h]
    if recipe == "pcd_adam":
        opt = torch.optim.Adam(params, lr=lr)
        v_chain = (torch.rand(n_chains, rbm.n_visible, generator=torch_gen, device=device) < 0.5).float()
    elif recipe == "cd1_sgd":
        opt = torch.optim.SGD(params, lr=lr, momentum=momentum)
        v_chain = None  # CD-1: no persistent chain, negative phase comes from the data batch
    else:
        raise ValueError("unknown --recipe " + str(recipe))

    for _ in range(n_steps):
        idx = rng_np.integers(0, n_data, size=batch)
        v_data = data_bits_t[idx]

        h_data_p = rbm.p_h_given_v(v_data)
        pos_W = v_data.t() @ h_data_p / batch
        pos_bv = v_data.mean(0)
        pos_bh = h_data_p.mean(0)

        if recipe == "pcd_adam":
            h0, _ = rbm.sample_h(v_chain, torch_gen)
            v1, _ = rbm.sample_v(h0, torch_gen)
            neg_n = float(n_chains)
            v_chain = v1.detach()
        else:  # cd1_sgd
            h0, _ = rbm.sample_h(v_data, torch_gen)
            v1, _ = rbm.sample_v(h0, torch_gen)
            neg_n = float(batch)

        h1_p = rbm.p_h_given_v(v1)
        neg_W = v1.t() @ h1_p / neg_n
        neg_bv = v1.mean(0)
        neg_bh = h1_p.mean(0)

        dW = (pos_W - neg_W).detach()
        dbv = (pos_bv - neg_bv).detach()
        dbh = (pos_bh - neg_bh).detach()

        if latent_only:
            dW, dbv, dbh = zero_non_latent_grad(rbm, dW, dbv, dbh, latent)

        opt.zero_grad()
        rbm.W.grad = -dW
        rbm.b_v.grad = -dbv
        rbm.b_h.grad = -dbh
        opt.step()


def parse_args():
    p = argparse.ArgumentParser(description="RBM baseline-alignment variants (arXiv 2603.06440 replication)")
    p.add_argument("--data", choices=["synthetic", "d2"], default="synthetic")
    p.add_argument("--out", default="outputs/rbm_variant")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--latent", choices=["biases", "literal"], default="biases")
    p.add_argument("--recipe", choices=["pcd_adam", "cd1_sgd"], default="pcd_adam")
    p.add_argument("--n-tuples", type=int, default=50000, help="N_t tuples per anchor")
    p.add_argument("--steps-core", type=int, default=None,
                    help="default: 30000 for pcd_adam; (n_tuples // batch) * epochs-core for cd1_sgd")
    p.add_argument("--steps-lat", type=int, default=3000,
                    help="per-anchor latent-only steps -- SAME default for both recipes "
                         "(only the training algorithm changes, not this budget)")
    p.add_argument("--epochs-core", type=int, default=30,
                    help="cd1_sgd only: stage-1 epochs over --n-tuples (ignored if --steps-core given)")
    p.add_argument("--lr", type=float, default=None, help="default: 1e-3 for pcd_adam, 0.05 for cd1_sgd")
    p.add_argument("--momentum", type=float, default=0.0, help="cd1_sgd only (SGD momentum)")
    p.add_argument("--hidden", type=int, default=630)
    p.add_argument("--batch", type=int, default=None, help="default: 512 for pcd_adam, 64 for cd1_sgd")
    p.add_argument("--n-chains-pcd", type=int, default=100, help="pcd_adam only")
    p.add_argument("--gen-chains", type=int, default=10000)
    p.add_argument("--gen-sweeps", type=int, default=200)
    p.add_argument("--gen-keep", type=int, default=10)
    p.add_argument("--anchors-max", type=int, default=0, help="0 = use all 11 anchors")
    p.add_argument("--n-holdout", type=int, default=200)
    return p.parse_args()


def main():
    args = parse_args()
    lr = args.lr if args.lr is not None else RECIPE_DEFAULTS[args.recipe]["lr"]
    batch = args.batch if args.batch is not None else RECIPE_DEFAULTS[args.recipe]["batch"]
    if args.steps_core is not None:
        steps_core = args.steps_core
    elif args.recipe == "pcd_adam":
        steps_core = 30000
    else:
        steps_core = (args.n_tuples // batch) * args.epochs_core

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

    get_field, get_pool = load_data_accessors(args.data, args.seed)

    holdout_ts = None  # resolved below; synthetic resolves it here already (shared split)
    if args.data == "synthetic":
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

    if args.data == "d2":
        # a_z/b_z MUST be the exact bounds load_d2()/load_d2_snapshot() used to encode
        # pool_codes (data_d2.get_z_range, native z_val over the 11 anchors) -- see
        # INTERFACES.md / baseline_rbm.py for why NOT re-derived from `.field`.
        from data_d2 import get_z_range
        a_z, b_z = get_z_range("mid")
    else:
        all_vals = np.concatenate([f.ravel() for f in anchor_fields.values()])
        a_z, b_z = float(all_vals.min()), float(all_vals.max())
        if b_z <= a_z:
            b_z = a_z + 1.0

    rbm = make_rbm(args.latent, N_VISIBLE, args.hidden, device, args.seed)

    # Stage 1: train theta_core on anchors[0] (t=1 by default), theta_lat left at its
    # random initial value.
    t0 = anchors[0]
    codes0 = codec.sample_codes(get_pool(t0, a_z, b_z), args.n_tuples, np.random.default_rng(args.seed))
    train_rbm_variant(
        rbm, codes0, n_steps=steps_core, recipe=args.recipe, latent=args.latent,
        batch=min(batch, args.n_tuples), n_chains=args.n_chains_pcd, lr=lr, momentum=args.momentum,
        latent_only=False, rng_np=rng_np, torch_gen=torch_gen, device=device,
    )

    latent_by_t = {}
    latent_by_t[t0] = rbm.get_latent().cpu().numpy()
    prev_latent = rbm.get_latent()

    # Stage 2: adapt theta_lat only, per subsequent anchor, warm-started.
    for t in anchors[1:]:
        codes_t = codec.sample_codes(get_pool(t, a_z, b_z), args.n_tuples, np.random.default_rng(args.seed + t))
        rbm.set_latent(prev_latent)
        train_rbm_variant(
            rbm, codes_t, n_steps=args.steps_lat, recipe=args.recipe, latent=args.latent,
            batch=min(batch, args.n_tuples), n_chains=args.n_chains_pcd, lr=lr, momentum=args.momentum,
            latent_only=True, rng_np=rng_np, torch_gen=torch_gen, device=device,
        )
        prev_latent = rbm.get_latent()
        latent_by_t[t] = prev_latent.cpu().numpy()

    anchors_sorted = sorted(latent_by_t.keys())
    latents_arr = np.stack([latent_by_t[t] for t in anchors_sorted])

    if holdout_ts is None:
        # d2: use data_d2's own fixed 200-tau holdout split (INTERFACES.md), truncated to
        # --n-holdout -- the same source/convention every other arm uses.
        from data_d2 import get_holdout_taus
        holdout_ts = get_holdout_taus()[: args.n_holdout] if args.n_holdout > 0 else []

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
        "arm": "RBM-{}k-{}-{}".format(int(round(rbm.param_count() / 1000.0)), args.latent, args.recipe),
        "data": args.data,
        "hidden": args.hidden,
        "n_tuples": args.n_tuples,
        "steps_core": steps_core,
        "steps_lat": args.steps_lat,
        "lr": lr,
        "batch": batch,
        "momentum": args.momentum if args.recipe == "cd1_sgd" else None,
        "n_chains_pcd": args.n_chains_pcd if args.recipe == "pcd_adam" else None,
        "gen_chains": args.gen_chains,
        "gen_sweeps": args.gen_sweeps,
        "gen_keep": args.gen_keep,
        "a_z": a_z,
        "b_z": b_z,
        "n_holdout": len(holdout_ts),
        "holdout_ts": holdout_ts,
        "d_lat": D_LAT,
        "latent_mode": args.latent,
        "recipe": args.recipe,
        "data_seed": None,
    }
    if args.data == "synthetic":
        from synth_data import SYNTH_DATA_SEED as _sds
        meta["data_seed"] = int(_sds)
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("RBM variant done: arm={} param_count={} anchors={} holdout={} wallclock={:.1f}s".format(
        meta["arm"], meta["param_count"], anchors_sorted, len(holdout_ts), meta["wallclock_s"]
    ))


if __name__ == "__main__":
    main()
