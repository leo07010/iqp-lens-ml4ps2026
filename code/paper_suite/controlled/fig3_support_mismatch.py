"""Fig. 3 of arXiv 2603.06440 -- controlled QCLI-MMD support-mismatch probe.

Paper setting (Table 3, `paper.txt:2766-2789`; Sec. II.6, `paper.txt:877-928`):

    Target datasets        10^4 samples from n = 16 IQP circuits
    Data-generating circuit locality <= 4, G_n^data in {140,280,420,560,700,840,1050}
    Target diversity       SPSA-optimised parameters spanning a range of I_QCLI
    Learner circuits       restricted IQP, locality <= 2, G_n in {50,100,150}
    Training               train-on-classical / deploy-on-quantum; Adam, 5000 iters, lr 1e-4
    Evaluation             achieved MMD loss after training

Prediction under test: the UPPER ENVELOPE of the achieved MMD falls as the target's I_QCLI
rises, and shifts down as the learner gate count G_n rises.

What this script does, step by step:
  1. For each G_n^data it builds `--n-targets` independent <=4-local circuits (distinct random
     Pauli-Z strings), runs SPSA on their parameters with the objective of INCREASING the
     exact I_QCLI, and snapshots the trajectory.  Each target then keeps the snapshot whose
     I_QCLI is closest to its own assigned level in the pooled range, which is how the
     "spanning a range of I_QCLI" of Table 3 is realised.
  2. It draws `--n-samples` bitstrings from each selected circuit -- these ARE the target
     datasets -- and records their sample I_QCLI/I_CCI (paper protocol) next to the exact
     circuit I_QCLI/I_CCI (diagnostic).
  3. For each learner gate count it fixes ONE <=2-local architecture and fits every target
     with Adam on the exact closed-form MMD^2 of Appendix G (the 500-shot estimator of the
     paper replaced by its exact limit, as everywhere else in this replication), all targets
     in one batched tensor.
  4. It writes one jsonl row per (target, learner) plus meta and i.i.d.-uniform-null rows.

Deviations from the paper, all recorded in the output:
  * exact MMD instead of the shot-based estimator (repo-wide choice, `DESIGN_exact_iqp.md`);
  * the training kernel bandwidth is the repo's three-sigma rule (the paper never states one,
    spec item 8) computed per target from its own 10^4 samples;
  * `--n-targets` per gate count is our choice (the paper does not give a target count);
  * a learner asking for 150 <=2-local gates gets 136, the number that exists at n = 16
    (see `common.random_gate_masks`).

Python 3.9 compatible.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
from common import (BatchIQP, cci_batch, empirical_pmf, kernel_c_by_weight,  # noqa: E402
                    mmd2_from_expvals, popcount_index, qcli_torch, random_gate_masks,
                    resolve_sigmas_for_codes, spsa_maximize, uniform_null_qcli)

_DTYPES = {"float32": torch.float32, "float64": torch.float64}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Fig. 3: does target QCLI predict the achievable MMD?")
    p.add_argument("--n", type=int, default=16, help="qubits of the controlled study (paper: 16)")
    p.add_argument("--gates-target", default="140,280,420,560,700,840,1050",
                   help="G_n^data list (paper Table 3)")
    p.add_argument("--learner-gates", default="50,100,150", help="G_n list (paper Table 3)")
    p.add_argument("--locality-target", type=int, default=4)
    p.add_argument("--locality-learner", type=int, default=2)
    p.add_argument("--n-targets", type=int, default=20, help="targets per G_n^data")
    p.add_argument("--n-samples", type=int, default=10000, help="bitstrings per target dataset")
    p.add_argument("--target-steps", type=int, default=400,
                   help="optimisation steps used to spread the targets over I_QCLI.  Adam on "
                        "the exact I_QCLI converges in ~50 steps at n=16 (measured), so 400 is "
                        "already deep into the plateau.")
    p.add_argument("--target-snapshots", type=int, default=400,
                   help="trajectory snapshots kept; >= --target-steps means every step, which "
                        "is free (the value is the one Adam's forward already computed)")
    p.add_argument("--target-optimizer", choices=("adam", "spsa"), default="adam",
                   help="how the data-generating parameters are pushed to high I_QCLI.  "
                        "'spsa' is the paper's choice (Table 3) but its 2-evaluation gradient "
                        "estimate degrades as the parameter count grows, and at n=16 with "
                        "420-1050 4-local gates it stalls below I_QCLI = 0.04 (measured, run "
                        "356097), so the targets never span the paper's range.  'adam' (the "
                        "default) maximises the SAME exact I_QCLI by autograd, which the exact "
                        "FWHT simulator makes available and which is unaffected by the "
                        "parameter count.")
    p.add_argument("--target-lr", type=float, default=0.05, help="Adam lr for --target-optimizer adam")
    p.add_argument("--target-qcli-max", type=float, default=0.7,
                   help="top of the requested coverage band (the paper's targets reach ~0.7); "
                        "clipped to whatever the optimisation actually attains, which is "
                        "reported")
    p.add_argument("--spsa-a", type=float, default=1.0,
                   help="SPSA gain, used only with --target-optimizer spsa")
    p.add_argument("--spsa-c", type=float, default=0.2)
    p.add_argument("--target-init-std", type=float, default=0.05,
                   help="std of the N(0,.) init of the data-generating circuits")
    p.add_argument("--steps", type=int, default=5000, help="Adam iterations (paper Table 3)")
    p.add_argument("--lr", type=float, default=1e-4, help="Adam lr (paper Table 3)")
    p.add_argument("--learner-init-std", type=float, default=0.01, help="repo default")
    p.add_argument("--train-batch", type=int, default=70, help="targets fitted in one tensor")
    p.add_argument("--null-reps", type=int, default=5, help="i.i.d.-uniform null repetitions")
    p.add_argument("--bin-width-report", type=float, default=0.09,
                   help="bin width of the coverage histogram printed to the log (Appendix D Delta)")
    p.add_argument("--exact-target-control", dest="exact_target_control", action="store_true",
                   default=True,
                   help="also fit every learner against the EXACT target pmf (control column "
                        "final_mmd2_exacttrained), which separates finite-sample error in the "
                        "10^4-sample dataset from genuine support mismatch")
    p.add_argument("--no-exact-target-control", dest="exact_target_control", action="store_false")
    p.add_argument("--done-marker", default=None,
                   help="path of a marker file written after the jsonl is complete")
    p.add_argument("--dtype", choices=tuple(_DTYPES), default="float64")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=20260907)
    p.add_argument("--out", default="results_fig3.jsonl")
    p.add_argument("--progress-every", type=int, default=500)
    return p.parse_args(argv)


def _jsonable(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


class JsonlWriter(object):
    def __init__(self, path):
        d = os.path.dirname(os.path.abspath(path))
        if d:
            os.makedirs(d, exist_ok=True)
        self.f = open(path, "w")

    def write(self, row):
        self.f.write(json.dumps({k: _jsonable(v) for k, v in row.items()}) + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


# =======================================================================================
# Step 1 + 2: build the target datasets
# =======================================================================================
def _maximise_qcli(args, sim, theta0, n, pc_idx, gen, log, tag):
    """Push the data-generating parameters toward high exact I_QCLI and snapshot the way up.

    Returns (snaps_theta: list of (B, P) tensors, Q: (S, B) exact I_QCLI at each snapshot).
    The trajectory starts at a near-identity circuit, whose output is close to a point mass and
    therefore has I_QCLI ~ 0 (an exact corner case, see common.py), so the snapshots sweep the
    whole reachable band from ~0 upward -- that sweep IS the target diversity of Table 3.
    """
    B = theta0.shape[0]
    every = max(1, int(args.target_steps) // max(1, int(args.target_snapshots)))
    snaps_theta, snaps_q = [], []

    def snap(th, q=None):
        with torch.no_grad():
            if q is None:
                q = qcli_torch(sim.probs(th.detach()), n, pc_idx)
            # kept on the host: 400 snapshots x (20, 1050) float64 is 67 MB, which is worth
            # spending to keep the GPU free for the fits that follow.
            snaps_theta.append(th.detach().to("cpu").clone())
            snaps_q.append(q.detach().to("cpu").numpy().copy())

    t0 = time.time()
    if args.target_optimizer == "adam":
        th = theta0.clone().requires_grad_(True)
        opt = torch.optim.Adam([th], lr=float(args.target_lr))
        for k in range(int(args.target_steps)):
            opt.zero_grad(set_to_none=True)
            q = qcli_torch(sim.probs(th), n, pc_idx)
            # q is the value AT the current theta, so recording the pair here costs nothing:
            # the forward pass is the one Adam already needs.  With the default
            # --target-snapshots >= --target-steps every step is kept, which matters because
            # Adam covers the whole 0 -> plateau rise inside the first ~50 steps and an evenly
            # spaced schedule would land almost every snapshot on the plateau.
            if k % every == 0 or k == args.target_steps - 1:
                snap(th, q)
            (-q.sum()).backward()
            opt.step()
        snap(th)
    else:
        def objective(t):
            return qcli_torch(sim.probs(t), n, pc_idx)

        snap(theta0)

        def cb(step, th, val):  # noqa: ARG001
            if (step + 1) % every == 0 or step == args.target_steps - 1:
                snap(th)

        spsa_maximize(objective, theta0, args.target_steps, a=args.spsa_a, c=args.spsa_c,
                      generator=gen, callback=cb, maximize=True)
    dt = time.time() - t0
    Q = np.stack(snaps_q, axis=0)
    log("[targets %s] %s %d steps in %.1fs (%.1f ms/step); %d snapshots; exact I_QCLI "
        "reached %.4f (pooled range %.4f..%.4f)"
        % (tag, args.target_optimizer, args.target_steps, dt,
           1e3 * dt / max(1, args.target_steps), Q.shape[0], Q.max(), Q.min(), Q.max()))
    return snaps_theta, Q


def build_targets(args, n, g_data, rng, device, dtype, pc_idx, log):
    """Return (list of per-target dicts, the BatchIQP that owns their circuits) for one G^data.

    Selection (this is the "then filter the resulting circuits according to their measured
    I_QCLI" of Sec. II.6): pool EVERY snapshot of EVERY trajectory in this gate-count group,
    lay `--n-targets` levels uniformly across [0, min(--target-qcli-max, achieved max)], and
    give each level the nearest still-unused snapshot.  Pooling across trajectories (rather
    than making each trajectory serve one level, as v1 did) is what guarantees the requested
    coverage even when trajectories climb at very different rates.
    """
    B = int(args.n_targets)
    masks = np.stack([random_gate_masks(n, args.locality_target, g_data, rng)[0] for _ in range(B)])
    g_eff = int(masks.shape[1])
    sim = BatchIQP(n, masks, device=device, dtype=dtype)
    theta0 = sim.init_theta(B, rng, std=args.target_init_std)
    gen = torch.Generator(device=torch.device(device))
    gen.manual_seed(int(rng.integers(0, 2 ** 31 - 1)))

    snaps_theta, Q = _maximise_qcli(args, sim, theta0, n, pc_idx, gen, log, "G=%d" % g_data)
    S, Bq = Q.shape
    flat = Q.reshape(-1)                                    # index = snapshot * B + trajectory
    reached = float(flat.max())
    hi = min(float(args.target_qcli_max), reached)
    levels = np.linspace(0.0, hi, B)
    used = np.zeros(flat.size, dtype=bool)

    out = []
    for b in range(B):
        d = np.abs(flat - levels[b])
        d[used] = np.inf
        i = int(np.argmin(d))
        used[i] = True
        si, ti = i // Bq, i % Bq
        out.append({"b": b, "snap": si, "traj": ti,
                    "theta": snaps_theta[si][ti:ti + 1].to(device), "masks": masks[ti],
                    "gates_requested": int(g_data), "gates_effective": g_eff,
                    "qcli_level": float(levels[b]), "qcli_reached_pool": reached})
    sel = np.array([Q[t["snap"], t["traj"]] for t in out])
    log("[targets G=%d] levels 0..%.3f (requested max %.2f, reached %.3f); selected exact "
        "I_QCLI %.4f..%.4f, max |selected - level| = %.4f"
        % (g_data, hi, args.target_qcli_max, reached, sel.min(), sel.max(),
           float(np.abs(sel - levels).max())))

    # Pooling means target b's parameters may come from trajectory ti != b, so the simulator
    # handed back must carry the masks IN SELECTION ORDER -- otherwise every downstream
    # probs(theta) would evaluate one target's angles on another target's Pauli strings.
    sel_masks = np.stack([masks[t["traj"]] for t in out])
    sim_sel = BatchIQP(n, sel_masks, device=device, dtype=dtype)
    with torch.no_grad():
        theta_sel = torch.cat([t["theta"] for t in out], dim=0)
        q_check = qcli_torch(sim_sel.probs(theta_sel), n, pc_idx).cpu().numpy()
    drift = float(np.abs(q_check - sel).max())
    if drift > 1e-8:
        raise RuntimeError("target/mask realignment failed for G=%d: max |recomputed - "
                           "recorded| I_QCLI = %.3e" % (g_data, drift))
    for b, t in enumerate(out):
        t["masks"] = sel_masks[b]
    del sim
    return out, sim_sel


def characterise_targets(args, n, sim, targets, rng, device, dtype, pc_idx, log):
    """Sample each target's dataset and compute its exact / sample QCLI and CCI."""
    B = len(targets)
    theta = torch.cat([t["theta"] for t in targets], dim=0)    # (B, P)
    gen = torch.Generator(device=torch.device(device))
    gen.manual_seed(int(rng.integers(0, 2 ** 31 - 1)))
    with torch.no_grad():
        p_exact = sim.probs(theta)                             # (B, 2^n)
        q_exact = qcli_torch(p_exact, n, pc_idx).cpu().numpy()
        codes = sim.sample(theta, args.n_samples, generator=gen).cpu().numpy()
    c_exact = cci_batch(p_exact, n)

    pmfs = np.stack([empirical_pmf(codes[b], n) for b in range(B)])
    p_samp = torch.as_tensor(pmfs, dtype=dtype, device=device)
    with torch.no_grad():
        q_samp = qcli_torch(p_samp, n, pc_idx).cpu().numpy()
        ez_data = common.fwht(p_samp)                          # (B, 2^n) <Z_s>_data
    c_samp = cci_batch(p_samp, n)

    with torch.no_grad():
        ez_exact = common.fwht(p_exact)                        # (B, 2^n) <Z_s> of the EXACT pmf

    for b, t in enumerate(targets):
        sigmas, weights = resolve_sigmas_for_codes(n, codes[b], int(args.seed) + b)
        t["sigmas"] = [float(s) for s in sigmas]
        t["sigma_pauli_weights"] = [float(w) for w in weights]
        t["c_by_weight"] = kernel_c_by_weight(n, sigmas)
        t["I_QCLI_exact"] = float(q_exact[b])
        t["I_CCI_exact"] = float(c_exact["I_CCI"][b])
        t["I_QCLI_sample"] = float(q_samp[b])
        t["I_CCI_sample"] = float(c_samp["I_CCI"][b])
        t["I_TC_sample"] = float(c_samp["I_TC"][b])
        t["n_distinct_codes"] = int(np.unique(codes[b]).size)
    log("[targets G=%d] exact QCLI %.4f..%.4f -> sample QCLI %.4f..%.4f (M=%d), sample CCI "
        "%.3f..%.3f" % (targets[0]["gates_requested"], q_exact.min(), q_exact.max(),
                        q_samp.min(), q_samp.max(), args.n_samples,
                        c_samp["I_CCI"].min(), c_samp["I_CCI"].max()))
    return ez_data, ez_exact


# =======================================================================================
# Step 3: fit the restricted learner
# =======================================================================================
def fit_learner(args, n, masks_l, ez_data, c_by_weight, device, dtype, pc_idx, log, tag,
                theta0=None, ez_alt=None):
    """Batched Adam fit of one fixed <=2-local architecture to every target at once.

    `theta0` pins the initialisation so that the 10^4-sample fit and the exact-pmf control fit
    of the same target start from the same point.  `ez_alt`, when given, is a second set of
    data expectation values against which the FINAL theta is scored without further training
    (used to read the sample-trained learner against the exact target pmf, which separates
    finite-sample error from genuine support mismatch at no extra training cost).

    Returns dict of numpy arrays: final_mmd2, best_mmd2, init_mmd2, floor_proxy, final_mmd2_alt.
    """
    B = ez_data.shape[0]
    sim = BatchIQP(n, masks_l, device=device, dtype=dtype)
    if theta0 is None:
        rng = np.random.default_rng(int(args.seed) + 991)
        theta0 = sim.init_theta(B, rng, std=args.learner_init_std)
    theta = theta0.clone().requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=float(args.lr))

    # Support-mismatch floor proxy of Proposition II.2: the kernel-weighted spectral mass of
    # the DATA that sits outside the learner's tunable Pauli support S_A = masks u {empty}.
    with torch.no_grad():
        mass_all = (common.segment_by_weight(ez_data * ez_data, n, pc_idx) * c_by_weight).sum(dim=1)
        keep = torch.zeros(1 << n, dtype=torch.bool, device=ez_data.device)
        keep[torch.as_tensor(np.asarray(masks_l, dtype=np.int64), device=ez_data.device)] = True
        keep[0] = True
        cs = c_by_weight[:, pc_idx]                                       # (B, 2^n)
        mass_in = ((ez_data * ez_data) * cs * keep[None, :]).sum(dim=1)
        floor_proxy = (mass_all - mass_in).cpu().numpy()

    with torch.no_grad():
        init_mmd2 = mmd2_from_expvals(sim.expvals(theta), ez_data, c_by_weight, n, pc_idx).detach()

    best = None
    t0 = time.time()
    for step in range(int(args.steps)):
        opt.zero_grad(set_to_none=True)
        loss_vec = mmd2_from_expvals(sim.expvals(theta), ez_data, c_by_weight, n, pc_idx)
        loss_vec.sum().backward()
        opt.step()
        with torch.no_grad():
            lv = loss_vec.detach()
            best = lv.clone() if best is None else torch.minimum(best, lv)
        if args.progress_every and (step % args.progress_every == 0 or step == args.steps - 1):
            log("  [%s] step %5d/%d  mean MMD^2 %.6e  (%.1f ms/step)"
                % (tag, step, args.steps, float(lv.mean()),
                   1e3 * (time.time() - t0) / float(step + 1)))
    with torch.no_grad():
        ez_model = sim.expvals(theta)
        final = mmd2_from_expvals(ez_model, ez_data, c_by_weight, n, pc_idx).detach()
        best = torch.minimum(best, final)
        if ez_alt is None:
            final_alt = np.full(B, float("nan"))
        else:
            final_alt = mmd2_from_expvals(ez_model, ez_alt, c_by_weight, n,
                                          pc_idx).detach().cpu().numpy()
    dt = time.time() - t0
    log("  [%s] done: %d steps x batch %d in %.1fs (%.2f ms/step)" % (tag, args.steps, B, dt,
                                                                     1e3 * dt / float(args.steps)))
    return {"final_mmd2": final.cpu().numpy(), "best_mmd2": best.cpu().numpy(),
            "init_mmd2": init_mmd2.cpu().numpy(), "floor_proxy": floor_proxy,
            "final_mmd2_alt": final_alt, "ms_per_step": 1e3 * dt / float(args.steps)}


# =======================================================================================
def main(argv=None):
    args = parse_args(argv)
    if args.device == "cpu":
        torch.set_num_threads(min(8, os.cpu_count() or 1))
    device = torch.device(args.device)
    dtype = _DTYPES[args.dtype]
    n = int(args.n)
    torch.manual_seed(int(args.seed))
    rng = np.random.default_rng(int(args.seed))
    pc_idx = popcount_index(n, device)
    g_data_list = [int(x) for x in str(args.gates_target).split(",") if x.strip()]
    g_learn_list = [int(x) for x in str(args.learner_gates).split(",") if x.strip()]

    def log(msg):
        print(msg, flush=True)

    w = JsonlWriter(args.out)
    t_all = time.time()
    log("[fig3] n=%d dtype=%s device=%s targets/gate-count=%d samples/target=%d steps=%d"
        % (n, args.dtype, args.device, args.n_targets, args.n_samples, args.steps))

    # --- the finite-sample floor of the QCLI axis ---------------------------------------
    null_mean, null_std = uniform_null_qcli(n, args.n_samples, args.null_reps, int(args.seed) + 13)
    log("[fig3] i.i.d.-uniform null at M=%d: I_QCLI = %.4f +/- %.4f (this is the floor of the "
        "x-axis, not zero)" % (args.n_samples, null_mean, null_std))
    w.write({"record": "null", "n": n, "n_samples": int(args.n_samples),
             "null_reps": int(args.null_reps), "I_QCLI_uniform_mean": null_mean,
             "I_QCLI_uniform_std": null_std})

    # --- fixed learner architectures (one per G_n, shared across all targets) ------------
    rng_l = np.random.default_rng(int(args.seed) + 777)
    learners = []
    for g in g_learn_list:
        m, eff = random_gate_masks(n, args.locality_learner, g, rng_l)
        learners.append({"gates_requested": g, "gates_effective": eff, "masks": m})
        log("[fig3] learner G_n=%d -> %d distinct <=%d-local gates"
            % (g, eff, args.locality_learner))

    # --- targets -------------------------------------------------------------------------
    all_targets = []
    all_ez = []
    all_ez_exact = []
    for g_data in g_data_list:
        targets, sim = build_targets(args, n, g_data, rng, device, dtype, pc_idx, log)
        ez, ez_ex = characterise_targets(args, n, sim, targets, rng, device, dtype, pc_idx, log)
        for b, t in enumerate(targets):
            t["target_id"] = len(all_targets)
            all_targets.append(t)
        all_ez.append(ez)
        all_ez_exact.append(ez_ex)
        del sim
    ez_all = torch.cat(all_ez, dim=0)                                   # (T, 2^n) 10^4 samples
    ez_exact_all = torch.cat(all_ez_exact, dim=0)                       # (T, 2^n) exact pmf
    c_all = torch.as_tensor(np.stack([t["c_by_weight"] for t in all_targets]),
                            dtype=dtype, device=device)                # (T, n+1)
    T = len(all_targets)
    qx = np.array([t["I_QCLI_exact"] for t in all_targets])
    qs = np.array([t["I_QCLI_sample"] for t in all_targets])
    log("[fig3] %d targets built. EXACT I_QCLI span %.4f..%.4f (the x-axis); the 10^4-sample "
        "I_QCLI of the same targets spans %.4f..%.4f, against an i.i.d.-uniform null of %.4f "
        "-- the sample estimator at M=%d over 2^%d outcomes compresses the whole band below "
        "its own noise floor, which is why the level layout uses the exact value."
        % (T, qx.min(), qx.max(), qs.min(), qs.max(), null_mean, args.n_samples, n))
    hist, edges = np.histogram(qx, bins=np.arange(0.0, qx.max() + args.bin_width_report,
                                                  args.bin_width_report))
    log("[fig3] exact-QCLI coverage in %.2f-wide bins: %s"
        % (args.bin_width_report, " ".join("[%.2f)=%d" % (e, h) for e, h in zip(edges[:-1], hist))))

    for t in all_targets:
        w.write({"record": "target", "target_id": t["target_id"],
                 "gates_target": t["gates_requested"], "gates_target_effective": t["gates_effective"],
                 "snapshot": t["snap"], "qcli_level": t["qcli_level"],
                 "I_QCLI": t["I_QCLI_sample"], "I_CCI": t["I_CCI_sample"],
                 "I_QCLI_exact": t["I_QCLI_exact"], "I_CCI_exact": t["I_CCI_exact"],
                 "I_TC_sample": t["I_TC_sample"], "n_distinct_codes": t["n_distinct_codes"],
                 "traj": t["traj"], "qcli_reached_pool": t["qcli_reached_pool"],
                 "sigmas": t["sigmas"], "n_samples": int(args.n_samples)})

    # --- fits ----------------------------------------------------------------------------
    chunk = max(1, int(args.train_batch))
    arms = [("sample", ez_all, ez_exact_all)]
    if args.exact_target_control:
        arms.append(("exact", ez_exact_all, None))
    for L in learners:
        rng_th = np.random.default_rng(int(args.seed) + 991 + L["gates_requested"])
        simL = BatchIQP(n, L["masks"], device=device, dtype=dtype)
        theta0_all = simL.init_theta(T, rng_th, std=args.learner_init_std)
        res = {}
        for arm, ez_fit, ez_alt in arms:
            parts = []
            for lo in range(0, T, chunk):
                hi = min(T, lo + chunk)
                tag = "G%d/%s[%d:%d]" % (L["gates_requested"], arm, lo, hi)
                parts.append(fit_learner(args, n, L["masks"], ez_fit[lo:hi], c_all[lo:hi],
                                         device, dtype, pc_idx, log, tag,
                                         theta0=theta0_all[lo:hi],
                                         ez_alt=None if ez_alt is None else ez_alt[lo:hi]))
            res[arm] = {k: np.concatenate([p[k] for p in parts])
                        for k in ("final_mmd2", "best_mmd2", "init_mmd2", "floor_proxy",
                                  "final_mmd2_alt")}
        del simL
        for t in all_targets:
            i = t["target_id"]
            r = res["sample"]
            f2 = float(r["final_mmd2"][i])
            b2 = float(r["best_mmd2"][i])
            row = {"record": "fit", "target_id": i,
                   "gates_target": t["gates_requested"],
                   "gates_target_effective": t["gates_effective"],
                   "learner_gates": L["gates_requested"],
                   "learner_gates_effective": L["gates_effective"],
                   "I_QCLI": t["I_QCLI_sample"], "I_CCI": t["I_CCI_sample"],
                   "I_QCLI_exact": t["I_QCLI_exact"], "I_CCI_exact": t["I_CCI_exact"],
                   "final_mmd2": f2, "final_mmd": math.sqrt(max(f2, 0.0)),
                   "best_mmd2": b2, "best_mmd": math.sqrt(max(b2, 0.0)),
                   "init_mmd2": float(r["init_mmd2"][i]),
                   "mmd2_floor_proxy": float(r["floor_proxy"][i]),
                   "final_mmd2_vs_exact": float(r["final_mmd2_alt"][i]),
                   "steps": int(args.steps), "lr": float(args.lr),
                   "n_samples": int(args.n_samples), "n": n}
            if "exact" in res:
                e = res["exact"]
                e2 = float(e["final_mmd2"][i])
                row["final_mmd2_exacttrained"] = e2
                row["final_mmd_exacttrained"] = math.sqrt(max(e2, 0.0))
                row["best_mmd2_exacttrained"] = float(e["best_mmd2"][i])
                row["mmd2_floor_proxy_exact"] = float(e["floor_proxy"][i])
            w.write(row)
        msg = ("[fig3] learner G_n=%d: MMD^2 (10^4-sample target) median %.4e  min %.4e  "
               "max %.4e" % (L["gates_requested"], float(np.median(res["sample"]["final_mmd2"])),
                             float(res["sample"]["final_mmd2"].min()),
                             float(res["sample"]["final_mmd2"].max())))
        if "exact" in res:
            msg += ("; MMD^2 (exact target) median %.4e"
                    % float(np.median(res["exact"]["final_mmd2"])))
        log(msg)

    w.write({"record": "meta", "n": n, "seed": int(args.seed), "dtype": args.dtype,
             "device": str(args.device), "gates_target": g_data_list,
             "learner_gates": g_learn_list, "n_targets_per_gate_count": int(args.n_targets),
             "n_samples": int(args.n_samples), "steps": int(args.steps), "lr": float(args.lr),
             "target_steps": int(args.target_steps), "locality_target": int(args.locality_target),
             "locality_learner": int(args.locality_learner),
             "target_optimizer": args.target_optimizer, "target_lr": float(args.target_lr),
             "target_level_axis": "exact_pmf_I_QCLI",   # v2: fixed, no longer a CLI choice
             "target_qcli_max_requested": float(args.target_qcli_max),
             "target_qcli_max_reached": float(qx.max()),
             "exact_target_control": bool(args.exact_target_control),
             "I_QCLI_exact_span": [float(qx.min()), float(qx.max())],
             "I_QCLI_sample_span": [float(qs.min()), float(qs.max())],
             "I_QCLI_uniform_null": null_mean,
             "wall_seconds": time.time() - t_all})
    w.close()
    log("[fig3] wrote %s in %.1f s" % (args.out, time.time() - t_all))
    if args.done_marker:
        d = os.path.dirname(os.path.abspath(args.done_marker))
        if d:
            os.makedirs(d, exist_ok=True)
        with open(args.done_marker, "w") as f:
            f.write("fig3 v2 complete %s\n" % time.strftime("%Y-%m-%d %H:%M:%S"))
            f.write("jsonl=%s\n" % os.path.abspath(args.out))
            f.write("targets=%d exact_QCLI=%.4f..%.4f sample_QCLI=%.4f..%.4f null=%.4f\n"
                    % (T, qx.min(), qx.max(), qs.min(), qs.max(), null_mean))
            f.write("wall_seconds=%.1f\n" % (time.time() - t_all))
        log("[fig3] done marker -> %s" % args.done_marker)
    return 0


if __name__ == "__main__":
    sys.exit(main())
