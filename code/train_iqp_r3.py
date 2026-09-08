#!/usr/bin/env python3
"""Round-3 IQP generator on centroid-aligned 32x32 lens crops (10 qubits, ALL 1,024 Z-strings).

docs/IQP_LENS_ROUND3_PREREGISTRATION.md. NEW file: round 2's train_iqp.py is in use by a
running job and is neither imported for its main() nor modified -- only its three pure sigma
helpers (sigma3_from_codes / mean_pauli_weight_for_sigma / resolve_sigmas) are imported, and
the exact simulator comes from iqp_exact.py.

Model
-----
n = 10, K = 10 -> the parameter vector IS the full 1,024-entry Walsh spectrum of the phase.
50 of those coefficients are the LATENT block (latent_select_r3.py: `lowvar` or `hamming`),
the other 974 are the shared CORE. theta_i = core (shared) scattered together with
theta_lat[i] (per training image).

  theta_empty is a global phase: phi(y) -> phi(y) + theta_empty for every y, so q is
  unchanged. It is trainable but INERT. `hamming` includes it (the paper's block does),
  `lowvar` drops it (its cross-image variance is 0), so the Hamming block has 49 effective
  latent coordinates against lowvar's 50. Recorded in meta.json rather than silently fixed.

Protocols
---------
joint      (primary)   theta_core and ALL theta_lat trained together on sum_i loss_i,
                       Adam lr 1e-3, --steps (2e4) full-batch exact steps.
paper      (ablation)  the paper's sequential Algorithm 1 with the round-2 "fixed" stage 1:
                       stage 1 trains ALL parameters on training image 1 for --steps-core
                       (3e4) at lr 1e-4; then for each later image theta_core is frozen and
                       only theta_lat is trained for --steps-lat (3e3), warm-started from the
                       previous image.
pseudotime (addendum)  identical to `paper` except the images are visited in order of the
                       ellipticity e1 of their centred crop, and generation is CONDITIONAL:
                       the latent trajectory is piecewise-linear in e1 (clamped outside the
                       training range) and is evaluated at each held-out image's own e1.

Objectives
----------
mmd      closed-form MMD^2 = sum_s c_s (<Z_s>_theta - <Z_s>_data)^2 over all 1,024 subsets,
         with the Ref [38] three-bandwidth rule recomputed at n = 10 (train_iqp.resolve_sigmas).
mmdflat  addendum (2026-09-07): identical to mmd but with FLAT Walsh weights c_s = 2^-10 for
         every one of the 1,024 subsets (sums to 1; equivalently the MMD with a delta kernel).
         Still Sigma c_s (<Z_s>_theta - <Z_s>_data)^2, hence sample-estimable on hardware --
         a hardware-compatible counterpart to the (simulation-only) KL diagnostic.
kl       exact KL(p_hat || q) = sum_x p_hat(x) log((p_hat(x)+eps) / (q(x)+eps)), eps = 1e-12,
         with q the exact circuit probability -- no sampling anywhere.

Latent prior (--prior, addendum 2026-09-07, unconditional joint/paper only)
----------------------------------------------------------------------------
gauss     (default, unchanged) Gaussian in PCA-whitened coordinates, rank = min(N-1, 50).
gauss8    the same Gaussian truncated to rank 8 (the step-0 analysis's ~7-dim image manifold);
          at N=11 rank = min(8, N-1) = 8, at N=100 rank = min(8, N-1) = 8.
resample  non-parametric: a trained latent drawn WITH REPLACEMENT plus Gaussian jitter, std =
          0.2 x the median nearest-neighbour distance among the N trained latents.
gauss8 / resample are declared only for the primary IQP-lowvar-joint-MMD configuration; the
arm-name suffix (-prior8 / -resample) is added regardless of latent/protocol/objective so the
flag also works standalone in tests, but submit_lens_r3.sbatch only invokes it for that arm.

Generation
----------
Unconditional (joint / paper): the N trained latents feed --prior (see above); --n-gen draws
are mapped back, turned into exact q, multiplied by S (mean total intensity of the TRAINING
crops) and pasted into a black 64x64 frame at an offset drawn with replacement from the
training (dy, dx).

  Honest note on that rule: PCA-whitening is estimated from the SAME N latents, so the whitened
  sample covariance is the identity on every retained component and "full at N=100 / diagonal
  at N=11" are numerically the same fit. What actually differs between the two N is the RANK
  that survives whitening: min(N-1, 50), i.e. 10 at N = 11 and 50 at N = 100. The rule is
  implemented literally as pre-registered (the empirical whitened covariance is computed and
  its distance from I is logged), not quietly replaced.

Conditional (pseudotime): one sample per held-out image, in held-out order, from the latent
interpolated at that image's e1. meta.json carries conditional=true so evaluate_r3.py knows the
i-th generated image is paired with the i-th held-out image.

Outputs
-------
  <out>/generated_{i:03d}.npy        (64, 64) float32   pasted frame
  <out>/crops/generated_{i:03d}.npy  (32, 32) float32   the crop before pasting
  <out>/meta.json , theta_core.npy , theta_lat.npy , loss_curve.npy

Python 3.9 compatible (no `X | None`).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch

import codec10
import data_lens_r3
import latent_select_r3
from iqp_exact import IQPExact, fwht, popcount_table
from train_iqp import resolve_sigmas          # pure helper, no side effects on import

N_QUBITS = 10
K_MAX = 10                # every subset -> the complete Walsh basis
D_LAT = 50
NCODES = 1 << N_QUBITS
KL_EPS = 1e-12


# --------------------------------------------------------------------------------------
# batched exact circuit (iqp_exact.IQPExact is single-theta; round 3 needs N at once)
# --------------------------------------------------------------------------------------
def probs_batch(sim, theta):
    """q_theta(x) for a BATCH of parameter vectors. theta (B, P) -> (B, 2^n).

    Same three FWHTs as IQPExact.probs, applied along the last axis of a batched tensor
    (iqp_exact.fwht already supports leading batch dims). K = n here, so sim.masks is a
    permutation of 0..2^n-1 and index_add is an exact scatter.
    """
    if theta.dim() != 2:
        raise ValueError("probs_batch: expected (B, P), got %s" % (tuple(theta.shape),))
    base = torch.zeros(theta.shape[0], sim.size, dtype=theta.dtype, device=theta.device)
    dense = base.index_add(1, sim.masks.to(theta.device), theta)
    phi = fwht(dense)
    re = fwht(torch.cos(phi))
    im = fwht(torch.sin(phi))
    return (re * re + im * im) / float(sim.size * sim.size)


def expvals_batch(sim, theta):
    """<Z_s>_theta for a batch: fwht of the batched probabilities. (B, P) -> (B, 2^n)."""
    return fwht(probs_batch(sim, theta))


# --------------------------------------------------------------------------------------
# parameter plumbing: core (974) + latent (50) -> the full 1,024 vector
# --------------------------------------------------------------------------------------
class ParamLayout(object):
    """Positions of the latent / core coefficients inside IQPExact's parameter ordering."""

    def __init__(self, sim, latent_masks):
        masks_np = sim.masks.detach().cpu().numpy().astype(np.int64)
        pos_of_mask = np.full(sim.size, -1, dtype=np.int64)
        pos_of_mask[masks_np] = np.arange(masks_np.size, dtype=np.int64)
        lat = np.asarray(latent_masks, dtype=np.int64)
        if np.any(pos_of_mask[lat] < 0):
            raise ValueError("a latent mask is not in the parameter set")
        self.lat_pos = pos_of_mask[lat]
        allpos = np.ones(masks_np.size, dtype=bool)
        allpos[self.lat_pos] = False
        self.core_pos = np.flatnonzero(allpos)
        self.n_params = int(masks_np.size)
        self.d_lat = int(self.lat_pos.size)
        self.d_core = int(self.core_pos.size)
        self.latent_masks = lat
        # theta_empty (mask 0) is a global phase and cannot change q -- report where it sits.
        self.empty_in_latent = bool(0 in set(lat.tolist()))

    def assemble(self, theta_core, theta_lat, device):
        """(B, P) full parameter matrix from a shared core (d_core,) and per-image latents
        (B, d_lat). Uses index_copy so autograd flows to both pieces."""
        b = theta_lat.shape[0]
        out = torch.zeros(b, self.n_params, dtype=theta_core.dtype, device=device)
        core_idx = torch.as_tensor(self.core_pos, dtype=torch.long, device=device)
        lat_idx = torch.as_tensor(self.lat_pos, dtype=torch.long, device=device)
        out = out.index_copy(1, core_idx, theta_core.unsqueeze(0).expand(b, -1))
        out = out.index_copy(1, lat_idx, theta_lat)
        return out


# --------------------------------------------------------------------------------------
# losses
# --------------------------------------------------------------------------------------
def mmd_loss(sim, theta, ez_data, weights):
    """sum_i sum_s c_s (<Z_s>_i - <Z_s>_data,i)^2 -- exact, all 2^n subsets, batched."""
    d = expvals_batch(sim, theta) - ez_data
    return (weights.unsqueeze(0) * d * d).sum()


def kl_loss(sim, theta, pmf):
    """sum_i KL(p_hat_i || q_i) = sum_i sum_x p log((p+eps)/(q+eps)), eps = 1e-12."""
    q = probs_batch(sim, theta)
    return (pmf * (torch.log(pmf + KL_EPS) - torch.log(q + KL_EPS))).sum()


def flat_kernel_weights(size, dtype=torch.float64):
    """Addendum: c_s = 2^-n for every one of the `size` = 2^n subsets (--objective mmdflat).
    Constant vector that sums to exactly 1 -- the MMD with a delta kernel."""
    size = int(size)
    return torch.full((size,), 1.0 / float(size), dtype=dtype)


MMD_OBJECTIVES = ("mmd", "mmdflat")


def per_image_losses(sim, theta, ez_data, weights, pmf, objective):
    with torch.no_grad():
        if objective in MMD_OBJECTIVES:
            d = expvals_batch(sim, theta) - ez_data
            return (weights.unsqueeze(0) * d * d).sum(dim=1).detach().cpu().numpy()
        q = probs_batch(sim, theta)
        return (pmf * (torch.log(pmf + KL_EPS) - torch.log(q + KL_EPS))).sum(dim=1).detach().cpu().numpy()


# --------------------------------------------------------------------------------------
# latent prior: Gaussian in PCA-whitened coordinates
# --------------------------------------------------------------------------------------
def fit_latent_gaussian(latents, cov_mode, rank_tol=1e-10, max_rank=None):
    """Return a dict describing the pre-registered latent prior.

    mu     (d,)      latent mean
    comps  (k, d)    retained principal directions (rows)
    scale  (k,)      std along each retained direction
    chol   (k, k)    Cholesky factor of the fitted covariance IN WHITENED coordinates
    whitened_cov_dev  max |C_whitened - I|, logged because PCA-whitening on the same sample
                      forces C_whitened = I: it is why `full` and `diag` coincide here.
    max_rank  addendum (--prior gauss8): cap the retained rank to this many principal
              components (kept in descending singular-value order, i.e. the top `max_rank`).
              Applied AFTER the rank_tol cut, so it can only shrink the rank further.
    """
    X = np.asarray(latents, dtype=np.float64)
    n, d = X.shape
    mu = X.mean(axis=0)
    Xc = X - mu[None, :]
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    keep = s > (rank_tol * max(float(s.max()), 1e-300))
    if not np.any(keep):
        keep = np.zeros_like(s, dtype=bool)
        keep[0] = True
    if max_rank is not None:
        # s (and hence `keep`) is in descending order -- capping to the first max_rank
        # True entries keeps exactly the top max_rank principal components.
        keep_idx = np.flatnonzero(keep)[: int(max_rank)]
        keep = np.zeros_like(keep)
        keep[keep_idx] = True
    comps = Vt[keep]                                   # (k, d)
    scale = s[keep] / math.sqrt(float(n))              # population std along each PC
    scale = np.maximum(scale, 1e-300)
    Y = (Xc @ comps.T) / scale[None, :]                # whitened, sample cov == I
    C = np.cov(Y.T, bias=True) if Y.shape[1] > 1 else np.array([[float(np.var(Y[:, 0]))]])
    C = np.atleast_2d(C)
    dev = float(np.abs(C - np.eye(C.shape[0])).max())
    if cov_mode == "diag":
        C = np.diag(np.diag(C))
    jitter = 1e-10 * max(float(np.trace(C)) / max(C.shape[0], 1), 1e-12)
    chol = np.linalg.cholesky(C + jitter * np.eye(C.shape[0]))
    return {"mu": mu, "comps": comps, "scale": scale, "chol": chol,
            "rank": int(comps.shape[0]), "cov_mode": cov_mode,
            "whitened_cov_dev": dev}


def sample_latents(prior, n, rng):
    """(n, d) latent draws from the fitted Gaussian."""
    k = prior["chol"].shape[0]
    z = rng.standard_normal(size=(int(n), k))
    y = z @ prior["chol"].T
    return prior["mu"][None, :] + (y * prior["scale"][None, :]) @ prior["comps"]


def median_nn_spacing(latents):
    """Median nearest-neighbour Euclidean distance among the N trained latents
    (addendum, --prior resample's jitter scale)."""
    X = np.asarray(latents, dtype=np.float64)
    n = X.shape[0]
    if n < 2:
        return 0.0
    diffs = X[:, None, :] - X[None, :, :]
    dist = np.sqrt((diffs * diffs).sum(axis=-1))
    np.fill_diagonal(dist, np.inf)
    return float(np.median(dist.min(axis=1)))


def sample_latents_resample(latents, n, rng, jitter_scale=0.2):
    """Addendum (--prior resample): draw a trained latent WITH REPLACEMENT, then add
    isotropic Gaussian jitter whose typical MAGNITUDE (Euclidean distance moved) is
    jitter_scale * median_nn_spacing(latents) -- i.e. the reported `jitter_std` is in the
    same units as `spacing` (both are distances in the d-dim latent space), so the two are
    directly comparable, matching the pre-registration's "jitter of 0.2x the spacing".

    A d-dim isotropic Gaussian with PER-COORDINATE std sigma has ||noise|| concentrated
    around sigma * sqrt(d) (d = 50 here), so sigma is set to jitter_std / sqrt(d): the
    resulting draw typically lands about jitter_std away from its base training latent,
    not jitter_std * sqrt(d) away (that mismatch is why this is spelled out).

    Returns (draws (n, d), spacing, jitter_std).
    """
    X = np.asarray(latents, dtype=np.float64)
    d = X.shape[1]
    spacing = median_nn_spacing(X)
    jitter_std = float(jitter_scale) * spacing             # target perturbation DISTANCE
    per_coord_std = jitter_std / math.sqrt(max(d, 1))      # per-coordinate std achieving it
    idx = rng.integers(0, X.shape[0], size=int(n))
    base = X[idx]
    noise = rng.standard_normal(size=base.shape) * per_coord_std
    return base + noise, spacing, jitter_std


def interp_latents_in_e1(anchor_e1, anchor_lat, targets):
    """Piecewise-linear interpolation of each latent coordinate along e1, clamped outside
    the training range (the addendum's pseudo-time trajectory)."""
    a = np.asarray(anchor_e1, dtype=np.float64)
    L = np.asarray(anchor_lat, dtype=np.float64)
    order = np.argsort(a)
    a, L = a[order], L[order]
    t = np.asarray(targets, dtype=np.float64)
    out = np.empty((t.size, L.shape[1]), dtype=np.float64)
    for j in range(L.shape[1]):
        out[:, j] = np.interp(t, a, L[:, j])           # np.interp clamps at both ends
    return out


# --------------------------------------------------------------------------------------
# physics helper (only e1 is needed here; evaluate_r3.py owns the full set)
# --------------------------------------------------------------------------------------
def e1_of(img):
    """e1 = (Qxx - Qyy)/(Qxx + Qyy) from the intensity second moments of a 2-D field."""
    w = np.clip(np.asarray(img, dtype=np.float64), 0.0, None)
    tot = float(w.sum())
    if not (tot > 0):
        return 0.0
    ny, nx = w.shape
    ys = np.arange(ny, dtype=np.float64)[:, None]
    xs = np.arange(nx, dtype=np.float64)[None, :]
    cy = float((w * ys).sum() / tot)
    cx = float((w * xs).sum() / tot)
    dy = ys - cy
    dx = xs - cx
    qxx = float((w * dx * dx).sum() / tot)
    qyy = float((w * dy * dy).sum() / tot)
    den = qxx + qyy
    return float((qxx - qyy) / den) if den > 0 else 0.0


# --------------------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Round-3 IQP generator on aligned 32x32 lens crops.")
    p.add_argument("--latent", choices=("lowvar", "hamming"), default="lowvar")
    p.add_argument("--protocol", choices=("joint", "paper", "pseudotime"), default="joint")
    p.add_argument("--objective", choices=("mmd", "kl", "mmdflat"), default="mmd",
                   help="mmdflat = mmd with flat Walsh weights c_s = 2^-10 (addendum, hardware-"
                        "compatible counterpart to kl)")
    p.add_argument("--prior", choices=("gauss", "gauss8", "resample"), default="gauss",
                   help="unconditional (joint/paper) latent prior: gauss = pre-registered "
                        "PCA-whitened Gaussian (unchanged default); gauss8 = the same truncated "
                        "to rank 8; resample = draw a trained latent with replacement + Gaussian "
                        "jitter (addendum, declared for IQP-lowvar-joint-MMD only)")
    p.add_argument("--n-train", type=int, default=11)
    p.add_argument("--n-holdout", type=int, default=data_lens_r3.N_HOLDOUT)
    p.add_argument("--steps", type=int, default=20000, help="joint protocol: optimiser steps")
    p.add_argument("--steps-core", type=int, default=30000, help="paper/pseudotime stage 1")
    p.add_argument("--steps-lat", type=int, default=3000, help="paper/pseudotime stage 2, per image")
    p.add_argument("--lr", type=float, default=None,
                   help="default 1e-3 for --protocol joint, 1e-4 for paper/pseudotime")
    p.add_argument("--n-gen", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--sigmas", default="auto3",
                   help="'auto3' = the Ref [38] three-bandwidth rule RECOMPUTED at n = 10 from "
                        "the first training image's codes; or a comma-separated float list")
    p.add_argument("--init", choices=("random", "zeros"), default="random")
    p.add_argument("--init-std", type=float, default=0.01,
                   help="the pre-registration does not fix an initialisation; round 3 declares "
                        "N(0, init_std) on every coefficient for every protocol")
    p.add_argument("--arm-name", default=None)
    p.add_argument("--latent-dir", default=None, help="where r3_latent_*.npy live")
    p.add_argument("--log-every", type=int, default=2000)
    p.add_argument("--threads", type=int, default=4)
    return p.parse_args(argv)


OBJECTIVE_LABEL = {"mmd": "MMD", "kl": "KL", "mmdflat": "MMDflat"}

PRIOR_SUFFIX = {"gauss": "", "gauss8": "-prior8", "resample": "-resample"}


def default_arm_name(args):
    if args.arm_name:
        return args.arm_name
    obj_label = OBJECTIVE_LABEL.get(args.objective, args.objective.upper())
    if args.protocol == "pseudotime":
        name = "IQP-%s-pseudotime-%s" % (args.latent, obj_label)
    else:
        name = "IQP-%s-%s-%s" % (args.latent, args.protocol, obj_label)
    name += PRIOR_SUFFIX.get(getattr(args, "prior", "gauss"), "")
    return name


def _tensor(x, device, dtype=torch.float64):
    return torch.as_tensor(np.asarray(x), dtype=dtype, device=device)


def main(argv=None):
    args = parse_args(argv)
    t0 = time.time()
    if str(args.device) == "cpu":
        # Same reason as train_iqp.py:326-332 -- the FWHT butterfly is a chain of tiny ops and
        # torch's default all-core intra-op pool makes sync dominate. Capped, not tuned.
        torch.set_num_threads(min(int(args.threads), os.cpu_count() or 1))
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "crops"), exist_ok=True)

    lr = args.lr if args.lr is not None else (1e-3 if args.protocol == "joint" else 1e-4)

    # ---------------------------------------------------------------- data
    d = data_lens_r3.load_r3(args.n_train, seed=args.seed, n_holdout=args.n_holdout)
    n = d.n_train
    S = d.S
    print("[r3] N=%d  S=%.4f  crops %s  holdout %s"
          % (n, S, (d.train_crops.shape,), (d.holdout_images.shape,)))

    latent_masks = latent_select_r3.load_latent_masks(args.latent, out_dir=args.latent_dir)
    sim = IQPExact(N_QUBITS, K_MAX, device=device, dtype=torch.complex128)
    layout = ParamLayout(sim, latent_masks)
    print("[r3] latent=%s d_lat=%d d_core=%d n_params=%d empty_in_latent=%s "
          "(theta_empty is a global phase: %d effective latent coords)"
          % (args.latent, layout.d_lat, layout.d_core, layout.n_params,
             layout.empty_in_latent, layout.d_lat - int(layout.empty_in_latent)))

    pmf_t = _tensor(d.train_pmfs, device)                     # (n, 1024)
    ez_data = fwht(pmf_t)                                     # (n, 1024) exact <Z_s>_data

    # sigmas / kernel weights. mmdflat (addendum) skips the bandwidth rule entirely: c_s is a
    # flat constant for every one of the 1,024 subsets.
    if args.objective == "mmdflat":
        sigmas, mean_weights = [], []
        weights = flat_kernel_weights(sim.size).to(device=device, dtype=torch.float64)
        print("[r3] objective=mmdflat: flat Walsh weights c_s=%.6g for all %d subsets (sum=%.6f)"
              % (1.0 / float(sim.size), sim.size, float(weights.sum())))
    else:
        codes1 = codec10.sample_codes_from_pmf(d.train_pmfs[0], 50000, np.random.default_rng(args.seed + 11))
        sigmas, mean_weights = resolve_sigmas(args.sigmas, N_QUBITS, codes1, args.seed)
        weights = sim.kernel_weights(sigmas).to(torch.float64)
        print("[r3] sigmas=%s -> mean Pauli weights=%s"
              % (["%.4f" % s for s in sigmas], ["%.3f" % w for w in mean_weights]))

    # ---------------------------------------------------------------- init
    def init_vec(size, gen_seed):
        if args.init == "zeros":
            return np.zeros(size, dtype=np.float64)
        return np.random.default_rng(gen_seed).normal(0.0, args.init_std, size=size)

    theta_core = torch.tensor(init_vec(layout.d_core, args.seed * 1000 + 1),
                              dtype=torch.float64, device=device, requires_grad=True)
    theta_lat = torch.tensor(np.stack([init_vec(layout.d_lat, args.seed * 1000 + 100 + i)
                                       for i in range(n)], axis=0),
                             dtype=torch.float64, device=device, requires_grad=True)

    def total_loss(th):
        if args.objective in MMD_OBJECTIVES:
            return mmd_loss(sim, th, ez_data, weights)
        return kl_loss(sim, th, pmf_t)

    loss_curve = []

    # ---------------------------------------------------------------- training
    if args.protocol == "joint":
        opt = torch.optim.Adam([theta_core, theta_lat], lr=lr)
        for step in range(int(args.steps)):
            opt.zero_grad()
            loss = total_loss(layout.assemble(theta_core, theta_lat, device))
            loss.backward()
            opt.step()
            if step % max(int(args.log_every), 1) == 0 or step == int(args.steps) - 1:
                loss_curve.append([int(step), float(loss.item())])
                print("[r3] joint step %6d  loss=%.6e  (%.1f s)" % (step, float(loss.item()), time.time() - t0))
        visit_order = list(range(n))
    else:
        # sequential Algorithm 1. `paper` visits images in index order; `pseudotime` visits
        # them in ascending e1 of the centred crop (addendum).
        if args.protocol == "pseudotime":
            e1_train = np.asarray([e1_of(d.train_crops[i]) for i in range(n)], dtype=np.float64)
            visit_order = list(np.argsort(e1_train))
            print("[r3] pseudotime: e1 range [%.4f, %.4f], first image index %d"
                  % (float(e1_train.min()), float(e1_train.max()), visit_order[0]))
        else:
            e1_train = np.asarray([e1_of(d.train_crops[i]) for i in range(n)], dtype=np.float64)
            visit_order = list(range(n))

        first = visit_order[0]
        # stage 1: ALL parameters on the first image (round-2 "fixed" protocol)
        lat1 = theta_lat[first:first + 1].detach().clone().requires_grad_(True)
        opt = torch.optim.Adam([theta_core, lat1], lr=lr)
        ez1 = ez_data[first:first + 1]
        pmf1 = pmf_t[first:first + 1]
        for step in range(int(args.steps_core)):
            opt.zero_grad()
            th = layout.assemble(theta_core, lat1, device)
            loss = (mmd_loss(sim, th, ez1, weights) if args.objective in MMD_OBJECTIVES
                    else kl_loss(sim, th, pmf1))
            loss.backward()
            opt.step()
            if step % max(int(args.log_every), 1) == 0 or step == int(args.steps_core) - 1:
                loss_curve.append([int(step), float(loss.item())])
                print("[r3] stage1 step %6d  loss=%.6e  (%.1f s)" % (step, float(loss.item()), time.time() - t0))
        with torch.no_grad():
            theta_lat[first] = lat1[0]
        theta_core = theta_core.detach()          # frozen from here on

        # stage 2: per-image latent, warm-started from the previous image in visit order
        prev = lat1.detach().clone()
        for pos, i in enumerate(visit_order[1:], start=1):
            lat_i = prev.clone().requires_grad_(True)
            opt = torch.optim.Adam([lat_i], lr=lr)
            ezi = ez_data[i:i + 1]
            pmfi = pmf_t[i:i + 1]
            for step in range(int(args.steps_lat)):
                opt.zero_grad()
                th = layout.assemble(theta_core, lat_i, device)
                loss = (mmd_loss(sim, th, ezi, weights) if args.objective in MMD_OBJECTIVES
                        else kl_loss(sim, th, pmfi))
                loss.backward()
                opt.step()
            with torch.no_grad():
                theta_lat[i] = lat_i[0]
            prev = lat_i.detach().clone()
            loss_curve.append([int(args.steps_core + pos * args.steps_lat), float(loss.item())])
            if pos % max(n // 5, 1) == 0 or pos == len(visit_order) - 1:
                print("[r3] stage2 %3d/%3d (image %3d) loss=%.6e (%.1f s)"
                      % (pos, n - 1, i, float(loss.item()), time.time() - t0))

    theta_core_np = theta_core.detach().cpu().numpy().astype(np.float64)
    theta_lat_np = theta_lat.detach().cpu().numpy().astype(np.float64)
    with torch.no_grad():
        th_final = layout.assemble(theta_core.detach(), theta_lat.detach(), device)
        per_img = per_image_losses(sim, th_final, ez_data, weights, pmf_t, args.objective)
        final_total = float(per_img.sum())
    train_time = time.time() - t0
    print("[r3] training done in %.1f s, final total loss=%.6e (per-image mean %.6e)"
          % (train_time, final_total, float(per_img.mean())))

    # ---------------------------------------------------------------- generation
    conditional = (args.protocol == "pseudotime")
    gen_rng = np.random.default_rng(args.seed + 999)
    if conditional:
        e1_train = np.asarray([e1_of(d.train_crops[i]) for i in range(n)], dtype=np.float64)
        e1_hold = np.asarray([e1_of(d.holdout_crops[i]) for i in range(d.holdout_crops.shape[0])],
                             dtype=np.float64)
        targets = e1_hold[: int(args.n_gen)]
        lat_gen = interp_latents_in_e1(e1_train, theta_lat_np, targets)
        prior_info = {"kind": "piecewise-linear in e1 (clamped)",
                      "n_targets": int(targets.size),
                      "e1_train_min": float(e1_train.min()), "e1_train_max": float(e1_train.max()),
                      "e1_target_min": float(targets.min()), "e1_target_max": float(targets.max()),
                      "n_targets_clamped": int(np.sum((targets < e1_train.min()) |
                                                      (targets > e1_train.max())))}
        print("[r3] conditional generation: %d targets, %d outside the training e1 range (clamped)"
              % (prior_info["n_targets"], prior_info["n_targets_clamped"]))
    elif args.prior == "resample":
        # addendum: non-parametric prior, declared for IQP-lowvar-joint-MMD only.
        lat_gen, spacing, jitter_std = sample_latents_resample(theta_lat_np, int(args.n_gen), gen_rng,
                                                                jitter_scale=0.2)
        prior_info = {"kind": "resample-with-gaussian-jitter", "prior_mode": args.prior,
                      "jitter_scale": 0.2, "median_nn_spacing": spacing, "jitter_std": jitter_std}
        print("[r3] latent prior: resample (with replacement) + Gaussian jitter, median NN "
              "spacing=%.6g -> jitter std=%.6g" % (spacing, jitter_std))
    else:
        cov_mode = "full" if n >= 100 else "diag"
        max_rank = 8 if args.prior == "gauss8" else None
        prior = fit_latent_gaussian(theta_lat_np, cov_mode, max_rank=max_rank)
        lat_gen = sample_latents(prior, int(args.n_gen), gen_rng)
        prior_info = {"kind": "Gaussian in PCA-whitened coordinates", "prior_mode": args.prior,
                      "cov_mode": cov_mode, "max_rank_requested": max_rank,
                      "rank": prior["rank"],
                      "whitened_cov_max_dev_from_I": prior["whitened_cov_dev"],
                      "latent_mu_absmean": float(np.abs(prior["mu"]).mean()),
                      "latent_pc_scale_mean": float(prior["scale"].mean()),
                      "latent_pc_scale_min": float(prior["scale"].min()),
                      "latent_pc_scale_max": float(prior["scale"].max())}
        print("[r3] latent prior: %s cov, rank %d of %d (max_rank_requested=%s); whitened cov "
              "deviates from I by %.3g (0 => 'full' and 'diag' are the same fit, see the module "
              "docstring)"
              % (cov_mode, prior["rank"], layout.d_lat, max_rank, prior["whitened_cov_dev"]))

    offsets = data_lens_r3.sample_offsets(d.train_offsets, lat_gen.shape[0], gen_rng)
    with torch.no_grad():
        th_gen = layout.assemble(torch.as_tensor(theta_core_np, dtype=torch.float64, device=device),
                                 torch.as_tensor(lat_gen, dtype=torch.float64, device=device),
                                 device)
        q_gen = probs_batch(sim, th_gen).cpu().numpy().astype(np.float64)

    for i in range(q_gen.shape[0]):
        crop = codec10.pmf_to_image(q_gen[i], S)
        np.save(os.path.join(args.out, "crops", "generated_%03d.npy" % i), crop.astype(np.float32))
        frame = codec10.paste(crop, int(offsets[i, 0]), int(offsets[i, 1]))
        np.save(os.path.join(args.out, "generated_%03d.npy" % i), frame.astype(np.float32))

    # ---------------------------------------------------------------- meta
    meta = {
        "arm": default_arm_name(args),
        "script": "train_iqp_r3.py",
        "round": 3,
        "n_qubits": N_QUBITS, "K": K_MAX, "param_count": int(layout.n_params),
        "d_lat": int(layout.d_lat), "d_core": int(layout.d_core),
        "latent": args.latent, "latent_masks": [int(m) for m in latent_masks.tolist()],
        "latent_empty_included": layout.empty_in_latent,
        "latent_effective_dim": int(layout.d_lat - int(layout.empty_in_latent)),
        "protocol": args.protocol, "objective": args.objective, "prior_arg": args.prior,
        "n_train": int(n), "n_gen": int(q_gen.shape[0]), "n_holdout": int(args.n_holdout),
        "S": float(S), "seed": int(args.seed), "device": str(args.device),
        "lr": float(lr), "steps": int(args.steps), "steps_core": int(args.steps_core),
        "steps_lat": int(args.steps_lat), "init": args.init, "init_std": float(args.init_std),
        "sigmas": [float(s) for s in sigmas],
        "sigma_mean_pauli_weights": [float(w) for w in mean_weights],
        "loss_final_total": final_total,
        "loss_final_per_image_mean": float(per_img.mean()),
        "loss_final_per_image_max": float(per_img.max()),
        "loss_final_per_image_min": float(per_img.min()),
        "loss_first_logged": float(loss_curve[0][1]) if loss_curve else None,
        "conditional": bool(conditional),
        "prior": prior_info,
        "train_seconds": float(train_time),
        "wallclock_seconds": float(time.time() - t0),
        "gen_offsets": [[int(a), int(b)] for a, b in offsets.tolist()],
    }
    with open(os.path.join(args.out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True, default=str)
    np.save(os.path.join(args.out, "theta_core.npy"), theta_core_np)
    np.save(os.path.join(args.out, "theta_lat.npy"), theta_lat_np)
    np.save(os.path.join(args.out, "loss_curve.npy"), np.asarray(loss_curve, dtype=np.float64))
    print("[r3] wrote %d images to %s (arm=%s) in %.1f s"
          % (q_gen.shape[0], args.out, meta["arm"], time.time() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
