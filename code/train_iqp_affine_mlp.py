#!/usr/bin/env python3
"""NON-LINEAR latent map: theta_i = theta0 + g(z_i), g a small MLP, on the same 10 qubits.

DERIVED FILE.  A copy of seeds/train_iqp_affine_r3_snap.py in which the ONLY model change is
that the affine map theta = theta0 + Phi^T z is replaced by theta = theta0 + g(z),

    g(z) = c * [ tanh(z W1 + b1) - tanh(b1) ] W2n ,   W2n = W2 rows L2-normalised to PHI_SCALE

Everything else is byte-identical to the affine arm: same data, same theta0 KL warm start,
same objective, same joint Adam and schedule, same z penalty relative to L0, same
resample+jitter prior, same held-out z-only projection with the same restart rule, same
snapshots, same output contract.

WHY IT IS BUILT THIS WAY
    - g(0) = 0 exactly (the tanh(b1) subtraction), so z = 0 still means theta0 and the L0
      reference the z penalty is measured against is unchanged.
    - The output layer's rows are normalised exactly as normalise_phi normalises Phi's, and
      one scalar c is calibrated ONCE at initialisation so that the typical ||g(z)|| equals
      PHI_SCALE * ||z||, i.e. the affine map's magnitude law.  The comparison is therefore
      about the SHAPE of the map, not about handing the non-linear arm a bigger phase budget.
      (--gnorm zscale hard-locks ||g(z)|| = PHI_SCALE ||z|| for every z instead of only on
      average at init; c is then unused.)
    - --g-input-scale exists because of a trap.  z lives inside a coherence radius of about
      0.05 (v11 Sec. 4), so with unit-norm input weights every pre-activation is ~0.05 and
      tanh is linear to a part in 10^3 -- the "non-linear" map would then BE the affine map
      and the experiment would be null by construction.  The input weight vector of each
      hidden unit is therefore initialised with norm --g-input-scale (default 20 = 1/0.05),
      putting the typical pre-activation at O(1) where tanh actually bends.  meta.json
      reports pre_abs_mean / pre_frac_gt1 and nonlinearity_ratio so that whether the
      non-linearity was really used is measured, not assumed.
    - Phi.npy still holds an (M, 1,024) matrix: the least-squares BEST AFFINE APPROXIMATION
      of the learned map on the trained latents.  nonlinearity_ratio is the relative residual
      of that fit -- 0 means the MLP collapsed to an affine map.

Parameters: theta0 (1,024) + M*H + H + H*1,024 (the MLP; no output bias, which would be
exactly redundant with theta0) + N*M per-image latents.

Everything below this line is the parent file's docstring, unchanged.
--------------------------------------------------------------------------------------
COPY of train_iqp_affine_r3.py with ONE added flag: --save-every.

WHY THIS FORK EXISTS.  The classical baselines now pick their step count on a validation set
(seeds/select_checkpoint.py), while the IQP arms ran a fixed 4,000 steps.  That asymmetry
favours the baselines, and a reviewer is entitled to ask what the IQP arms would do under the
same rule.  Answering it needs intermediate checkpoints, and they cannot be faked by running
the trainer several times with different --steps: run_adam_scalar drives a CosineAnnealingLR
whose T_max IS the total step count (train_iqp_affine_r3.py:230), so the first 1,000 steps of
a 16,000-step run follow a completely different learning-rate trajectory from a 1,000-step
run.  The snapshots therefore have to come from ONE run.

--save-every takes a list of step counts.  The joint optimiser runs once to the largest of
them, and at each listed step the CURRENT BEST iterate -- the trainer's own convention, the
best-so-far rather than the latest (line 226) -- is put through the unchanged held-out
projection, generation and output-writing path.  Snapshot k lands in `<--out>_s<k/1000>k`,
the same layout baseline_rbm_sweep.py / baseline_dcgan_sweep.py use, so
seeds/select_checkpoint.py reads it with no changes at all.

With --save-every unset, or set to a single value equal to --steps, this file reproduces
train_iqp_affine_r3.py exactly.  The original is not modified.

--------------------------------------------------------------------------------------
Round-3b IQP generator with LEARNED latent directions in phase space.

docs/IQP_LENS_ROUND3_PREREGISTRATION.md, section "Round 3b -- learned latent directions".
NEW file. Round 3's 28-job sbatch is queued against train_iqp_r3.py / controls_r3.py /
evaluate_r3.py / data_lens_r3.py / codec10.py / test_r3.py, so none of those is modified:
this script only IMPORTS from them (the batched simulator, the flat/auto3 kernel weights,
the resample prior and the whole data layer), which keeps the circuit, the data split, the
crops, the scale S and the paste offsets bit-identical to round 3.

Model
-----
Round 3 partitioned the 1,024 Walsh coefficients into a shared CORE (974) and a fixed LATENT
BLOCK (50 chosen a priori). Round 3b instead LEARNS which directions of the 1,024-dimensional
phase space carry the image-to-image variation:

    theta_i = theta0 + Phi^T z_i ,   theta0 in R^1024,  Phi in R^{M x 1024},  z_i in R^M

with M in {2, 4, 8} (the amendment's grid). Every row of Phi is L2-normalised and rescaled to
sqrt(1024) = 32, so
z_m = 1 shifts the 1,024 phases by ~1 rad rms -- z is then in interpretable units and the
1e-3 L2 penalty on z means the same thing at every M. theta0, Phi and all N latents z_i are
trained JOINTLY on the mean of the per-image losses.

theta0 is initialised by fitting a single full 1,024-phase hologram to the MEAN training crop
(--theta0-steps, exact KL, Adam). Mechanism test (scratchpad mech_test2/affine_latent.py):
that single-image fit reaches corr 0.9997, and the joint fit then reaches train corr 0.958 /
held-out projection corr 0.911 at M = 4 on 20 images. The seed always uses KL even for the
MMDflat/auto3 arms, so all arms start from the SAME theta0 and differ only in what they train
on; meta.json records theta0_seed_objective.

Objectives (--objective)
------------------------
kl       exact KL(p_hat_i || q_i), averaged over images -- simulation-only diagnostic.
mmdflat  sum_s c_s (<Z_s>_i - <Z_s>_data,i)^2 with FLAT Walsh weights c_s = 2^-10
         (train_iqp_r3.flat_kernel_weights): hardware-compatible, sample-estimable.
auto3    the same quadratic form with the paper's three-bandwidth Gaussian kernel weights
         (train_iqp.resolve_sigmas at n = 10, iqp_exact.IQPExact.kernel_weights).

Honest note on the z penalty: --zpen (1e-3 by default, 0.3 for the amendment's M = 8 capacity
arm, whose name then carries a `-zpen` tag) is a RELATIVE penalty, not a literal add-on: it
multiplies L0, the chosen --objective's data loss at theta0 (post-seed) with z = 0, a detached
constant fixed once at the start of training (Phi is irrelevant to L0 -- z = 0 collapses
theta_of to theta0 regardless of Phi). The joint loss is
mean_i loss_i + zpen * L0 * mean_i ||z_i||^2 , so --zpen means "this fraction of the STARTING
loss" at every objective, instead of an absolute number that means different things at
different scales (the three objectives differ in absolute scale by orders of magnitude: KL ~
1e-1, the MMD forms ~ 1e-4 -- a literal additive penalty would bite harder on the MMD arms than
on KL, which is what an earlier version of this script did). The held-out projection inference
(infer_z_batched) reuses the SAME L0 rather than recomputing one from the held-out crops, so a
held-out image's penalty is on the same relative scale as a training image's. meta.json carries
zpen_L0, loss_final_data / loss_final_zpen and zpen_penalty_over_data (= loss_final_zpen /
loss_final_data), and the run still prints a WARNING when the final penalty exceeds 10% of the
final data term, so the reader can see the relative weight instead of guessing at it.

Held-out projection (the round-3b extra metric)
-----------------------------------------------
For each of the --n-holdout held-out crops, theta0 and Phi are FROZEN and only z is optimised
(amendment: --infer-steps 1000 at --infer-lr 0.005 -- a fifth of the training LR, because the
mechanism test measured the model's coherence radius in z at ~0.03-0.05 and a 0.02 step
overshoots it -- from --infer-restarts starting points). The restarts are, in order:

    0            zeros
    1            the trained z of the nearest TRAINING image (nearest in pmf L2)
    2 .. R-1     draws from the Gaussian fitted to the trained z

and the reported number for each image is the BEST image correlation over the restarts. The
mechanism test is why this exists: with a single zero start, M = 8 collapsed to corr 0.09 on
held-out images while M = 4 reached 0.91 -- an optimiser failure, not a subspace failure, and
the restarts are the pre-registered fix. The same report also fixes the reference points this
arm is read against: a RANDOM Phi reaches 0.67, i.e. no better than theta0 alone (0.67), so
anything above that is what LEARNING the directions bought. All (image x restart) problems are solved as ONE
batched Adam (the rows are independent given frozen theta0/Phi, so the batched sum is exactly
N_ho x R separate optimisations sharing only the LR schedule), chunked by --infer-chunk to
bound memory. The per-ROW best iterate is kept, not the per-batch one.

meta.json also carries two reference numbers computed on the same held-out crops -- corr to
the mean training crop and corr to the nearest training crop -- so P18 ("projection corr >=
0.90, above CTRL-mean ~ 0.74") can be read off a single arm's meta without re-deriving them.

Generation (--prior)
--------------------
gauss     Gaussian fitted to the N trained z (full covariance at N >= 100, diagonal below, the
          same rule round 3 uses for its latent prior).
resample  a trained z drawn WITH REPLACEMENT plus Gaussian jitter whose typical DISTANCE is
          0.2 x the median nearest-neighbour spacing of the trained z
          (train_iqp_r3.sample_latents_resample, imported unchanged).

--n-gen draws -> exact q -> x S (mean total intensity of the TRAINING crops) -> pasted into a
black 64x64 frame at an offset drawn with replacement from the training (dy, dx), i.e. exactly
round 3's output contract:

  <out>/generated_{i:03d}.npy        (64, 64) float32
  <out>/crops/generated_{i:03d}.npy  (32, 32) float32
  <out>/meta.json , theta0.npy , Phi.npy , z_train.npy , z_holdout.npy , loss_curve.npy

Arm name: IQP-affine-M{M}-{KL|MMDflat|auto3}-{gauss|resample}.

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

# imports live next to this file
import sys as _sys
_REPO = os.path.dirname(os.path.abspath(__file__))
if _REPO not in _sys.path:
    _sys.path.insert(0, _REPO)

import codec10
import data_lens_r3
from iqp_exact import IQPExact, fwht
# round-3 pieces reused verbatim so 3b and 3 share one simulator / one prior implementation
from train_iqp_r3 import (MMD_OBJECTIVES, flat_kernel_weights, probs_batch,
                          sample_latents_resample)
from train_iqp import resolve_sigmas          # pure helper, no side effects on import

N_QUBITS = 10
K_MAX = 10                       # every subset -> the complete 1,024-entry Walsh basis
NCODES = 1 << N_QUBITS
PHI_SCALE = math.sqrt(float(NCODES))     # ||Phi_m||_2 = 32  =>  z_m = 1 is ~1 rad rms of phase
KL_EPS = 1e-12
DT = torch.float64


# --------------------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------------------
# The MLP is carried in ONE flat tensor so that every piece of plumbing inherited from the
# affine arm -- the [theta0, phi_raw, z] parameter list, the snapshot state unpacking, the
# frozen-map held-out inference -- keeps working unchanged.  G_M / G_H / G_MODE / G_SCALE are
# set once in main() before any forward pass.
G_M = 4
G_H = 32
G_MODE = "rownorm"
G_SCALE = 1.0


def g_size(M, H):
    """Length of the flat MLP vector: W1 (M x H) + b1 (H) + W2 (H x NCODES)."""
    return int(M) * int(H) + int(H) + int(H) * NCODES


def unpack_g(vec):
    """(W1, b1, W2) views into the flat vector. Autograd-safe (pure reshapes/slices)."""
    M, H = G_M, G_H
    i = 0
    W1 = vec[i:i + M * H].reshape(M, H)
    i += M * H
    b1 = vec[i:i + H]
    i += H
    W2 = vec[i:i + H * NCODES].reshape(H, NCODES)
    return W1, b1, W2


def normalise_phi(w2_raw):
    """Rows L2-normalised then scaled to PHI_SCALE. Autograd-safe.

    Kept under its original name and applied to the MLP's OUTPUT layer: that is what "the
    output layer is normalised the same way Phi is" means here.
    """
    return w2_raw / w2_raw.norm(dim=1, keepdim=True).clamp_min(1e-12) * PHI_SCALE


def g_of(vec, z):
    """(B, 1024) phase offset g(z). g(0) = 0 exactly, as Phi^T 0 = 0."""
    W1, b1, W2 = unpack_g(vec)
    h = torch.tanh(z @ W1 + b1) - torch.tanh(b1)
    raw = h @ normalise_phi(W2)
    if G_MODE == "zscale":
        nz = z.norm(dim=1, keepdim=True)
        return raw / raw.norm(dim=1, keepdim=True).clamp_min(1e-12) * (PHI_SCALE * nz)
    return G_SCALE * raw


def theta_of(theta0, phi_raw, z):
    """(B, 1024) parameter matrix theta_i = theta0 + g(z_i)."""
    return theta0.unsqueeze(0) + g_of(phi_raw, z)


def init_gvec(M, H, input_scale, bias_std, generator, dtype):
    """Flat MLP init.  Each hidden unit's input weight vector gets L2 norm `input_scale` (see
    the module docstring: unit norm would leave tanh linear over the coherence radius and the
    arm would silently reduce to the affine one); biases spread the units' operating points."""
    W1 = torch.randn(M, H, dtype=dtype, generator=generator)
    W1 = W1 / W1.norm(dim=0, keepdim=True).clamp_min(1e-12) * float(input_scale)
    b1 = float(bias_std) * torch.randn(H, dtype=dtype, generator=generator)
    W2 = torch.randn(H, NCODES, dtype=dtype, generator=generator)
    return torch.cat((W1.reshape(-1), b1.reshape(-1), W2.reshape(-1)))


def effective_affine(vec, z_np):
    """Least-squares best affine approximation A of g on the trained latents, and the
    relative residual ||G - Z A||_F / ||G||_F.  A has Phi's shape (M, 1024) and Phi's units,
    so the two arms' maps can be compared directly; residual 0 = the MLP is affine."""
    Z = np.asarray(z_np, dtype=np.float64)
    with torch.no_grad():
        G = g_of(vec.detach(), torch.as_tensor(Z, dtype=vec.dtype,
                                               device=vec.device)).cpu().numpy().astype(np.float64)
    A, _res, _rank, _sv = np.linalg.lstsq(Z, G, rcond=None)
    den = float(np.linalg.norm(G))
    resid = float(np.linalg.norm(G - Z @ A) / den) if den > 0 else 0.0
    return A, resid, G


# --------------------------------------------------------------------------------------
# losses (per ROW, so the batched held-out inference can keep a per-image best)
# --------------------------------------------------------------------------------------
def kl_rows(pmf, q):
    """KL(p_hat_i || q_i) for every row. Same eps convention as train_iqp_r3.kl_loss."""
    return (pmf * (torch.log(pmf + KL_EPS) - torch.log(q + KL_EPS))).sum(dim=1)


def mmd_rows(q, ez_data, weights):
    """sum_s c_s (<Z_s>_i - <Z_s>_data,i)^2 for every row."""
    d = fwht(q) - ez_data
    return (weights.unsqueeze(0) * d * d).sum(dim=1)


def loss_rows(sim, theta, pmf, ez_data, weights, objective):
    q = probs_batch(sim, theta)
    if objective in MMD_OBJECTIVES or objective == "auto3":
        return mmd_rows(q, ez_data, weights)
    return kl_rows(pmf, q)


OBJECTIVE_LABEL = {"kl": "KL", "mmdflat": "MMDflat", "auto3": "auto3"}


# --------------------------------------------------------------------------------------
# correlation (the projection metric)
# --------------------------------------------------------------------------------------
def corr_rows(a, b):
    """Pearson correlation of matching rows of two (m, 1024) arrays.

    Computed in pmf space; corr is invariant to a positive rescaling of either argument, and
    codec10.pmf_to_image is exactly (reshape, transpose, x S > 0), so this equals the
    correlation of the 32x32 crops it decodes to. test_r3b.py checks that equality.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    num = (a * b).sum(axis=1)
    den = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
    return np.where(den > 0, num / np.maximum(den, 1e-300), 0.0)


def corr_matrix(a, b):
    """(ma, mb) matrix of correlations between every row of `a` and every row of `b`."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    na = np.sqrt((a * a).sum(axis=1))
    nb = np.sqrt((b * b).sum(axis=1))
    den = np.outer(np.maximum(na, 1e-300), np.maximum(nb, 1e-300))
    return (a @ b.T) / den


# --------------------------------------------------------------------------------------
# latent prior
# --------------------------------------------------------------------------------------
def fit_z_gaussian(z, cov_mode):
    """Gaussian on the trained z. M is small (4 or 8), so this is done directly in z space
    (no PCA whitening is needed to make it well posed) -- `full` at N >= 100, `diag` below,
    the same N rule round 3 applies to its own latent prior."""
    X = np.asarray(z, dtype=np.float64)
    n, m = X.shape
    mu = X.mean(axis=0)
    Xc = X - mu[None, :]
    cov = (Xc.T @ Xc) / float(max(n - 1, 1))
    if cov_mode == "diag":
        cov = np.diag(np.diag(cov))
    jitter = 1e-12 * max(float(np.trace(cov)) / max(m, 1), 1e-30)
    chol = np.linalg.cholesky(cov + jitter * np.eye(m))
    return {"mu": mu, "chol": chol, "cov_mode": cov_mode,
            "std": np.sqrt(np.maximum(np.diag(cov), 0.0))}


def sample_z_gaussian(prior, n, rng):
    g = rng.standard_normal(size=(int(n), prior["mu"].size))
    return prior["mu"][None, :] + g @ prior["chol"].T


# --------------------------------------------------------------------------------------
# optimisation helpers
# --------------------------------------------------------------------------------------
def run_adam_scalar(params, closure, steps, lr, log_every=0, tag="", t0=None):
    """Adam + cosine schedule on a scalar closure; returns (best_value, best_state, curve).
    The best ITERATE is kept, not the last (the mechanism test's convention)."""
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(int(steps), 1), eta_min=lr * 0.05)
    best = float("inf")
    best_state = [p.detach().clone() for p in params]
    curve = []
    for step in range(int(steps)):
        opt.zero_grad()
        loss = closure()
        loss.backward()
        opt.step()
        sch.step()
        v = float(loss.detach())
        if v < best:
            best = v
            best_state = [p.detach().clone() for p in params]
        if log_every and (step % log_every == 0 or step == int(steps) - 1):
            curve.append([int(step), v])
            print("[mlp] %s step %6d  loss=%.6e  (%.1f s)"
                  % (tag, step, v, (time.time() - t0) if t0 else 0.0), flush=True)
    return best, best_state, curve


def step_tag(steps):
    """20000 -> 's020k' -- the baseline sweeps' tag, so one selector reads both."""
    return "s%03dk" % int(round(int(steps) / 1000.0))


def run_adam_snapshots(params, closure, steps, lr, snaps, log_every=0, tag="", t0=None):
    """run_adam_scalar, but also hands back the best-so-far state at each step in `snaps`.

    Identical arithmetic to run_adam_scalar: same Adam, same CosineAnnealingLR with T_max =
    the FULL step count, same best-iterate bookkeeping.  Taking a snapshot does not perturb
    the optimiser -- it only clones the current best state -- so the final snapshot is
    bit-for-bit what run_adam_scalar would have returned.
    """
    want = sorted({int(s) for s in snaps if 0 < int(s) <= int(steps)})
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(int(steps), 1),
                                                     eta_min=lr * 0.05)
    best = float("inf")
    best_state = [p.detach().clone() for p in params]
    curve, shots = [], []
    for step in range(int(steps)):
        opt.zero_grad()
        loss = closure()
        loss.backward()
        opt.step()
        sch.step()
        v = float(loss.detach())
        if v < best:
            best = v
            best_state = [p.detach().clone() for p in params]
        if log_every and (step % log_every == 0 or step == int(steps) - 1):
            curve.append([int(step), v])
            print("[mlp] %s step %6d  loss=%.6e  (%.1f s)"
                  % (tag, step, v, (time.time() - t0) if t0 else 0.0), flush=True)
        if want and (step + 1) == want[0]:
            k = want.pop(0)
            shots.append((k, best, [b.clone() for b in best_state], list(curve)))
            print("[mlp] %s snapshot at %d steps (best loss %.6e)" % (tag, k, best), flush=True)
    return best, best_state, curve, shots


def zpen_penalty(z, zpen, L0, reduce="mean"):
    """zpen * L0 * ||z_i||^2, per image, reduced across images by `reduce`.

    `L0` is the SAME detached constant everywhere this is called (the --objective's data loss
    at theta0, z = 0 -- see the module docstring's "z penalty" note): it is what makes the same
    --zpen a comparable fraction of the starting loss at every --objective, instead of an
    absolute number that means different things at different loss scales.

    reduce='mean' is the joint-training loss's convention (pairs with rows.mean()).
    reduce='sum' is infer_z_batched's convention: a SUM keeps each row's gradient independent
    of the batch size B (needed for --infer-chunk invariance -- a mean would divide by B and
    make chunk size silently change the answer). Both reduce to the SAME per-image penalty,
    zpen * L0 * ||z_i||^2; they only differ in how the per-image values are combined.
    """
    per_image = (z * z).sum(dim=1)
    if reduce == "mean":
        return zpen * L0 * per_image.mean()
    if reduce == "sum":
        return zpen * L0 * per_image.sum()
    raise ValueError("zpen_penalty: reduce must be 'mean' or 'sum', got %r" % (reduce,))


def infer_z_batched(sim, theta0, phi_raw, pmf_t, ez_t, weights, objective, z_init,
                    steps, lr, zpen, L0):
    """Optimise z ONLY, for a batch of independent rows, keeping the per-ROW best iterate.

    theta0/Phi are frozen, so row i's loss depends only on z[i]: the summed objective below is
    exactly `z_init.shape[0]` separate optimisations that happen to share one LR schedule.
    The batch objective is a SUM (not a mean) over rows, and the z penalty with it (via
    zpen_penalty(..., reduce="sum")), so row i's gradient is exactly
    d/dz_i [loss_i + zpen * L0 * ||z_i||^2] whatever B is: results do not depend on
    --infer-chunk. (With a mean the per-row gradient would carry a 1/B factor; Adam is almost
    but not exactly invariant to that, and chunk size would silently change the answer.
    test_r3b.py checks chunk invariance.) L0 is the SAME detached constant the joint training
    loss uses (the --objective's data loss at theta0, z = 0), passed in rather than recomputed
    here, so a held-out image's penalty sits on the same relative scale as a training image's.

    Returns (best_z (B, M), best_row_loss (B,)).
    """
    z = z_init.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(int(steps), 1), eta_min=lr * 0.05)
    best_row = torch.full((z.shape[0],), float("inf"), dtype=z.dtype, device=z.device)
    best_z = z.detach().clone()
    th0 = theta0.detach()
    ph = phi_raw.detach()
    for _step in range(int(steps)):
        opt.zero_grad()
        rows = loss_rows(sim, theta_of(th0, ph, z), pmf_t, ez_t, weights, objective)
        total = rows.sum() + zpen_penalty(z, zpen, L0, reduce="sum")
        total.backward()
        opt.step()
        sch.step()
        with torch.no_grad():
            better = rows.detach() < best_row
            if bool(better.any()):
                best_row = torch.where(better, rows.detach(), best_row)
                best_z[better] = z.detach()[better]
    return best_z, best_row


# --------------------------------------------------------------------------------------
def nearest_train_index(ho_pmf, tr_pmf):
    """For each held-out pmf, the index of the nearest TRAINING pmf in L2."""
    ho = np.asarray(ho_pmf, dtype=np.float64)
    tr = np.asarray(tr_pmf, dtype=np.float64)
    d2 = ((ho * ho).sum(axis=1)[:, None] - 2.0 * (ho @ tr.T) + (tr * tr).sum(axis=1)[None, :])
    return np.argmin(d2, axis=1)


def build_restart_inits(z_train, nearest, restarts, rand_block=None):
    """(m, R, M) starting points for the held-out z search, in the declared order:
    restart 0 = zeros, restart 1 = the nearest TRAINING image's trained z, 2..R-1 = the rows of
    `rand_block` (draws from the Gaussian fitted to the trained z -- random but on the right
    scale: z lives inside a coherence radius of ~0.03-0.05, so N(0, 1) starts would be off the
    manifold entirely).

    `rand_block` is drawn ONCE for the whole held-out set and sliced per chunk by the caller,
    so --infer-chunk changes nothing about which starting points an image gets.
    """
    z_train = np.asarray(z_train, dtype=np.float64)
    nearest = np.asarray(nearest, dtype=np.int64)
    m = int(nearest.size)
    M = int(z_train.shape[1])
    R = max(int(restarts), 1)
    z0 = np.zeros((m, R, M), dtype=np.float64)
    if R > 1:
        z0[:, 1, :] = z_train[nearest]
    if R > 2:
        if rand_block is None:
            raise ValueError("build_restart_inits: R > 2 needs a rand_block")
        rb = np.asarray(rand_block, dtype=np.float64)
        if rb.shape != (m, R - 2, M):
            raise ValueError("build_restart_inits: rand_block %s, expected %s"
                             % (rb.shape, (m, R - 2, M)))
        z0[:, 2:, :] = rb
    return z0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Round-3b IQP with learned affine phase latents.")
    p.add_argument("--M", type=int, default=4, help="latent dimension (round 3b runs 4 and 8)")
    p.add_argument("--objective", choices=("kl", "mmdflat", "auto3"), default="kl")
    p.add_argument("--prior", choices=("gauss", "resample"), default="gauss")
    p.add_argument("--n-train", type=int, default=11)
    p.add_argument("--n-holdout", type=int, default=data_lens_r3.N_HOLDOUT)
    p.add_argument("--steps", type=int, default=4000, help="joint Adam steps on theta0/Phi/z")
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--zpen", type=float, default=1e-3,
                   help="L2 penalty on z, RELATIVE to L0 (the --objective's data loss at "
                        "theta0, z=0): loss + zpen*L0*mean_i||z_i||^2, so the same value is "
                        "the same fraction of the starting loss at every objective "
                        "(amendment: 0.3 for the M = 8 capacity arm, 1e-3 everywhere else)")
    p.add_argument("--theta0-steps", type=int, default=1000,
                   help="steps of the single-hologram KL fit to the mean training crop")
    p.add_argument("--theta0-lr", type=float, default=0.05)
    p.add_argument("--n-gen", type=int, default=200)
    p.add_argument("--save-every", type=int, nargs="+", default=None,
                   help="step counts at which to emit a COMPLETE output directory "
                        "(generated_*.npy, crops/, meta.json, the parameter arrays). One "
                        "training run covers all of them; --steps is forced up to the largest. "
                        "Snapshot k is written to <--out>_s<k/1000>k, the layout the baseline "
                        "sweeps use, so select_checkpoint.py reads it unchanged.")
    p.add_argument("--infer-restarts", type=int, default=8)
    p.add_argument("--infer-steps", type=int, default=1000,
                   help="held-out z-only steps (amendment: 1000 at lr 0.005)")
    p.add_argument("--infer-lr", type=float, default=0.005,
                   help="LR of the held-out z-only optimiser; deliberately smaller than --lr "
                        "because z lives inside the model's ~0.03-0.05 coherence radius")
    p.add_argument("--infer-chunk", type=int, default=50,
                   help="held-out images per batched inference chunk (memory bound)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--sigmas", default="auto3", help="--objective auto3 only")
    p.add_argument("--phi-init-std", type=float, default=1.0,
                   help="std of the raw output-layer init (immaterial: rows are renormalised)")
    p.add_argument("--hidden", type=int, default=32,
                   help="width H of the tanh hidden layer of g")
    p.add_argument("--gnorm", choices=("rownorm", "zscale"), default="rownorm",
                   help="rownorm: output rows normalised like Phi's and ONE scalar calibrated "
                        "at init so the typical ||g(z)|| = PHI_SCALE*||z||, free to move "
                        "afterwards.  zscale: ||g(z)|| = PHI_SCALE*||z|| enforced for every z, "
                        "the affine arm's magnitude law made exact")
    p.add_argument("--g-input-scale", type=float, default=20.0,
                   help="L2 norm of each hidden unit's input weights at init. The default "
                        "1/0.05 puts the typical pre-activation at O(1) over the model's "
                        "coherence radius; ~1 would leave tanh linear and the arm affine")
    p.add_argument("--g-bias-std", type=float, default=0.5,
                   help="std of the hidden biases at init (spreads the operating points)")
    p.add_argument("--z-init-std", type=float, default=1e-3)
    p.add_argument("--arm-name", default=None)
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--threads", type=int, default=4)
    return p.parse_args(argv)


ZPEN_DEFAULT = 1e-3


def default_arm_name(args):
    """IQP-affine-M{M}-{KL|MMDflat|auto3}[-zpen]-{gauss|resample}.

    The `-zpen` tag marks a NON-default z penalty (the amendment's M = 8 capacity arm uses
    0.3 instead of 1e-3), so two runs that differ only in --zpen cannot collide in the table.
    """
    if args.arm_name:
        return args.arm_name
    tag = "" if float(args.zpen) == ZPEN_DEFAULT else "-zpen"
    return "IQP-mlpH%d%s-M%d-%s%s-%s" % (int(args.hidden),
                                         "zs" if args.gnorm == "zscale" else "",
                                         int(args.M), OBJECTIVE_LABEL[args.objective], tag,
                                         args.prior)


def main(argv=None):
    args = parse_args(argv)
    t0 = time.time()
    if str(args.device) == "cpu":
        torch.set_num_threads(min(int(args.threads), os.cpu_count() or 1))
    device = torch.device(args.device)
    if int(args.M) < 1:
        raise SystemExit("train_iqp_affine_r3.py: --M must be >= 1, got %d" % args.M)
    if args.save_every:
        # one run has to reach the last snapshot, and the cosine schedule must be set for that
        # full length -- shortening it would change every earlier snapshot too
        args.steps = max(int(args.steps), max(int(s) for s in args.save_every))
        # step_tag rounds to whole thousands, so two snapshots under 500 steps apart would
        # write to the same directory and one would silently overwrite the other
        _tags = [step_tag(x) for x in sorted({int(x) for x in args.save_every})]
        if len(set(_tags)) != len(_tags):
            raise SystemExit("--save-every values collide once rounded to whole thousands: "
                             "%s -> %s" % (sorted({int(x) for x in args.save_every}), _tags))
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "crops"), exist_ok=True)

    # ---------------------------------------------------------------- data (identical to r3)
    d = data_lens_r3.load_r3(args.n_train, seed=args.seed, n_holdout=args.n_holdout)
    n = d.n_train
    S = d.S
    M = int(args.M)
    tr_pmf = np.asarray(d.train_pmfs, dtype=np.float64)                       # (N, 1024)
    n_ho = int(d.holdout_crops.shape[0])
    ho_pmf = (np.stack([codec10.image_to_pmf(d.holdout_crops[i]) for i in range(n_ho)], axis=0)
              if n_ho > 0 else np.zeros((0, NCODES), dtype=np.float64))
    print("[mlp] N=%d  M=%d  S=%.4f  crops %s  holdout %d  objective=%s prior=%s"
          % (n, M, S, (d.train_crops.shape,), n_ho, args.objective, args.prior), flush=True)

    sim = IQPExact(N_QUBITS, K_MAX, device=device, dtype=torch.complex128)
    tr_t = torch.as_tensor(tr_pmf, dtype=DT, device=device)
    ez_tr = fwht(tr_t)
    mean_pmf = tr_pmf.mean(axis=0)
    mean_pmf = mean_pmf / mean_pmf.sum()
    mean_t = torch.as_tensor(mean_pmf, dtype=DT, device=device).unsqueeze(0)

    # ---------------------------------------------------------------- kernel weights
    if args.objective == "kl":
        sigmas, mean_weights = [], []
        weights = torch.ones(sim.size, dtype=DT, device=device)      # unused by the KL path
    elif args.objective == "mmdflat":
        sigmas, mean_weights = [], []
        weights = flat_kernel_weights(sim.size).to(device=device, dtype=DT)
        print("[mlp] objective=mmdflat: flat Walsh weights c_s=%.6g (sum=%.6f)"
              % (1.0 / float(sim.size), float(weights.sum())), flush=True)
    else:
        codes1 = codec10.sample_codes_from_pmf(tr_pmf[0], 50000,
                                               np.random.default_rng(args.seed + 11))
        sigmas, mean_weights = resolve_sigmas(args.sigmas, N_QUBITS, codes1, args.seed)
        weights = sim.kernel_weights(sigmas).to(device=device, dtype=DT)
        print("[mlp] objective=auto3: sigmas=%s -> mean Pauli weights=%s"
              % (["%.4f" % s for s in sigmas], ["%.3f" % w for w in mean_weights]), flush=True)

    # ---------------------------------------------------------------- theta0 seed
    torch.manual_seed(args.seed)
    th0 = 0.01 * torch.randn(NCODES, dtype=DT, device=device)

    def seed_closure():
        # ALWAYS KL, for every objective: all arms then share one theta0 (see the docstring).
        q = probs_batch(sim, th0.unsqueeze(0))
        return kl_rows(mean_t, q).mean()

    seed_best, seed_state, _c = run_adam_scalar([th0], seed_closure, args.theta0_steps,
                                                args.theta0_lr)
    theta0 = seed_state[0].detach().clone()
    with torch.no_grad():
        q_seed = probs_batch(sim, theta0.unsqueeze(0)).cpu().numpy()
    seed_corr = float(corr_rows(q_seed, mean_pmf[None, :])[0])
    print("[mlp] theta0 seed: %d KL steps on the mean crop -> KL=%.6e corr=%.4f (%.1f s)"
          % (args.theta0_steps, seed_best, seed_corr, time.time() - t0), flush=True)

    # ------------------------------------------------------- zpen reference scale L0
    # L0 = the ACTUAL --objective's data loss at theta0 (post-seed), z = 0, mean over the n
    # training images -- a detached constant (z = 0 makes theta_of collapse to theta0
    # regardless of Phi, so this needs no phi_raw). See zpen_penalty() and the module
    # docstring's "z penalty" note: --zpen then always means "this fraction of L0", the same
    # fraction at KL (~1e-1) as at the MMD forms (~1e-4).
    with torch.no_grad():
        theta0_bcast = theta0.unsqueeze(0).repeat(n, 1)
        L0 = float(loss_rows(sim, theta0_bcast, tr_t, ez_tr, weights, args.objective)
                   .mean().item())
    print("[mlp] zpen reference L0 (%s data loss at theta0, z=0) = %.6e"
          % (args.objective, L0), flush=True)

    # ---------------------------------------------------------------- joint training
    gen_t = torch.Generator(device="cpu").manual_seed(int(args.seed) * 1000 + 7)
    global G_M, G_H, G_MODE, G_SCALE
    G_M, G_H, G_MODE = M, int(args.hidden), args.gnorm
    phi_raw = init_gvec(M, G_H, args.g_input_scale, args.g_bias_std, gen_t, DT).to(device)
    z = (args.z_init_std * torch.randn(n, M, dtype=DT, generator=gen_t)).to(device)
    theta0 = theta0.clone()

    # ONE calibration constant, fixed here and never trained: it makes the typical ||g(z)||
    # at initialisation equal PHI_SCALE*||z||, the affine map's magnitude.  Measured on the
    # initial z themselves, so no extra randomness enters.
    G_SCALE = 1.0
    with torch.no_grad():
        _raw = g_of(phi_raw, z)
        _ratio = (_raw.norm(dim=1) / (PHI_SCALE * z.norm(dim=1).clamp_min(1e-30))).mean()
        g_calib = float(_ratio)
    if args.gnorm == "rownorm":
        G_SCALE = 1.0 / max(g_calib, 1e-30)
    with torch.no_grad():
        _W1, _b1, _W2 = unpack_g(phi_raw)
        _pre = z @ _W1 + _b1
        g_init = {"g_calibration_raw_ratio": g_calib, "g_scale": float(G_SCALE),
                  "pre_abs_mean_init": float(_pre.abs().mean()),
                  "g_norm_over_phi_norm_init":
                      float((g_of(phi_raw, z).norm(dim=1)
                             / (PHI_SCALE * z.norm(dim=1).clamp_min(1e-30))).mean())}
    print("[mlp] g: H=%d mode=%s input-scale=%.3g -> c=%.4g, ||g(z)||/(PHI*||z||)=%.4f at init"
          % (G_H, G_MODE, args.g_input_scale, G_SCALE,
             g_init["g_norm_over_phi_norm_init"]), flush=True)

    def train_closure():
        rows = loss_rows(sim, theta_of(theta0, phi_raw, z), tr_t, ez_tr, weights, args.objective)
        return rows.mean() + zpen_penalty(z, args.zpen, L0, reduce="mean")

    _snap_steps = sorted({int(s) for s in (args.save_every or [args.steps]) if int(s) > 0})
    best, state, loss_curve, _shots = run_adam_snapshots(
        [theta0, phi_raw, z], train_closure, args.steps, args.lr, _snap_steps,
        log_every=max(int(args.log_every), 1), tag="joint", t0=t0)
    train_time = time.time() - t0
    if not _shots:
        _shots = [(int(args.steps), best, [p.detach().clone() for p in state], list(loss_curve))]
    print("[mlp] %d snapshot(s) to emit: %s"
          % (len(_shots), [s[0] for s in _shots]), flush=True)

    _snap_root = args.out
    _multi = len(_shots) > 1 or bool(args.save_every)
    for _snap_k, _snap_best, _snap_state, _snap_curve in _shots:
        theta0, phi_raw, z = (_snap_state[0].detach(), _snap_state[1].detach(),
                              _snap_state[2].detach())
        best = _snap_best
        loss_curve = _snap_curve
        args.out = ("%s_%s" % (_snap_root, step_tag(_snap_k))) if _multi else _snap_root
        os.makedirs(args.out, exist_ok=True)
        os.makedirs(os.path.join(args.out, "crops"), exist_ok=True)
        _snap_t = time.time()
        with torch.no_grad():
            th_tr = theta_of(theta0, phi_raw, z)
            q_tr = probs_batch(sim, th_tr).cpu().numpy()
            rows_tr = loss_rows(sim, th_tr, tr_t, ez_tr, weights, args.objective).cpu().numpy()
        data_loss = float(rows_tr.mean())
        zpen_term = float(zpen_penalty(z, args.zpen, L0, reduce="mean").item())
        corr_tr = corr_rows(q_tr, tr_pmf)
        print("[mlp] joint training done in %.1f s: loss=%.6e (data %.6e + zpen %.6e), "
              "train corr mean=%.4f min=%.4f"
              % (train_time, best, data_loss, zpen_term, float(corr_tr.mean()), float(corr_tr.min())),
              flush=True)
        if data_loss > 0 and zpen_term > 0.1 * data_loss:
            print("[mlp] WARNING: the z penalty is %.1f%% of the data term -- at this objective's "
                  "scale --zpen=%g is NOT a mild regulariser (declared value kept, see docstring)"
                  % (100.0 * zpen_term / data_loss, args.zpen), flush=True)

        z_np = z.cpu().numpy().astype(np.float64)
        theta0_np = theta0.cpu().numpy().astype(np.float64)
        # Phi.npy = the best AFFINE approximation of the learned map on the trained latents,
        # in Phi's own shape and units; the residual says how much of g is not affine.
        phi_np, nonlin_ratio, _G_tr = effective_affine(phi_raw, z_np)
        with torch.no_grad():
            _W1, _b1, _W2 = unpack_g(phi_raw)
            _pre = (z @ _W1 + _b1)
            g_stats = {"nonlinearity_ratio": float(nonlin_ratio),
                       "pre_abs_mean": float(_pre.abs().mean()),
                       "pre_frac_gt1": float((_pre.abs() > 1.0).to(DT).mean()),
                       "g_norm_mean": float(g_of(phi_raw, z).norm(dim=1).mean()),
                       "phi_eff_row_norm_mean": float(np.linalg.norm(phi_np, axis=1).mean())}
        print("[mlp] map at %s: nonlinearity ratio=%.4f  |pre| mean=%.3f (%.1f%% > 1)  "
              "||g(z)|| mean=%.4f  effective Phi row norm=%.3f"
              % (step_tag(_snap_k), g_stats["nonlinearity_ratio"], g_stats["pre_abs_mean"],
                 100.0 * g_stats["pre_frac_gt1"], g_stats["g_norm_mean"],
                 g_stats["phi_eff_row_norm_mean"]), flush=True)

        # ---------------------------------------------------------------- held-out projection
        proj = {"corrs": [], "restart": [], "loss": []}
        t_proj = time.time()
        if n_ho > 0:
            z_prior = fit_z_gaussian(z_np, "full" if n >= 100 else "diag")
            R = max(int(args.infer_restarts), 1)
            # restart 1 needs each held-out image's nearest TRAINING image, in pmf L2
            nearest_train = nearest_train_index(ho_pmf, tr_pmf)
            rng_r = np.random.default_rng(args.seed + 4321)
            # drawn once for the WHOLE held-out set (see build_restart_inits): chunk-invariant
            rand_all = (sample_z_gaussian(z_prior, n_ho * (R - 2), rng_r).reshape(n_ho, R - 2, M)
                        if R > 2 else None)
            chunk = max(int(args.infer_chunk), 1)
            all_corr = np.zeros(n_ho, dtype=np.float64)
            all_restart = np.zeros(n_ho, dtype=np.int64)
            all_loss = np.zeros(n_ho, dtype=np.float64)
            z_ho_best = np.zeros((n_ho, M), dtype=np.float64)
            for lo in range(0, n_ho, chunk):
                hi = min(lo + chunk, n_ho)
                m = hi - lo
                blk = ho_pmf[lo:hi]                                       # (m, 1024)
                z0 = build_restart_inits(z_np, nearest_train[lo:hi], R,
                                         None if rand_all is None else rand_all[lo:hi])
                flat_z = torch.as_tensor(z0.reshape(m * R, M), dtype=DT, device=device)
                rep = torch.as_tensor(np.repeat(blk, R, axis=0), dtype=DT, device=device)
                best_z, best_row = infer_z_batched(sim, theta0, phi_raw, rep, fwht(rep), weights,
                                                   args.objective, flat_z, args.infer_steps,
                                                   args.infer_lr, args.zpen, L0)
                with torch.no_grad():
                    q = probs_batch(sim, theta_of(theta0, phi_raw, best_z)).cpu().numpy()
                c = corr_rows(q, np.repeat(blk, R, axis=0)).reshape(m, R)
                pick = np.argmax(c, axis=1)                               # best CORR over restarts
                idx = np.arange(m) * R + pick
                all_corr[lo:hi] = c[np.arange(m), pick]
                all_restart[lo:hi] = pick
                all_loss[lo:hi] = best_row.cpu().numpy()[idx]
                z_ho_best[lo:hi] = best_z.cpu().numpy()[idx]
                print("[mlp] projection %3d-%3d/%d: corr mean=%.4f min=%.4f (%.1f s)"
                      % (lo, hi - 1, n_ho, float(all_corr[lo:hi].mean()),
                         float(all_corr[lo:hi].min()), time.time() - t0), flush=True)
            proj = {"corrs": all_corr.tolist(), "restart": all_restart.tolist(),
                    "loss": all_loss.tolist()}
            # reference rows: the same 200 crops against the mean / nearest training crop
            ref_mean = corr_rows(np.repeat(mean_pmf[None, :], n_ho, axis=0), ho_pmf)
            ref_near = corr_matrix(ho_pmf, tr_pmf).max(axis=1)
            print("[mlp] held-out projection corr mean=%.4f min=%.4f | restarts used %s | "
                  "ref: mean-crop %.4f, nearest-train %.4f (%.1f s)"
                  % (float(all_corr.mean()), float(all_corr.min()),
                     np.bincount(all_restart, minlength=R).tolist(),
                     float(ref_mean.mean()), float(ref_near.mean()), time.time() - t_proj), flush=True)
        else:
            z_ho_best = np.zeros((0, M), dtype=np.float64)
            ref_mean = np.zeros(0)
            ref_near = np.zeros(0)
        proj_time = time.time() - t_proj

        # ---------------------------------------------------------------- generation
        gen_rng = np.random.default_rng(args.seed + 999)
        if args.prior == "resample":
            z_gen, spacing, jitter_std = sample_latents_resample(z_np, int(args.n_gen), gen_rng,
                                                                 jitter_scale=0.2)
            prior_info = {"kind": "resample-with-gaussian-jitter", "prior_mode": "resample",
                          "jitter_scale": 0.2, "median_nn_spacing": float(spacing),
                          "jitter_std": float(jitter_std)}
            print("[mlp] latent prior: resample + jitter, median NN spacing=%.6g -> jitter std=%.6g"
                  % (spacing, jitter_std), flush=True)
        else:
            cov_mode = "full" if n >= 100 else "diag"
            g_prior = fit_z_gaussian(z_np, cov_mode)
            z_gen = sample_z_gaussian(g_prior, int(args.n_gen), gen_rng)
            prior_info = {"kind": "Gaussian on the trained z", "prior_mode": "gauss",
                          "cov_mode": cov_mode, "M": M,
                          "z_mu_absmean": float(np.abs(g_prior["mu"]).mean()),
                          "z_std_mean": float(g_prior["std"].mean()),
                          "z_std_min": float(g_prior["std"].min()),
                          "z_std_max": float(g_prior["std"].max())}
            print("[mlp] latent prior: %s-covariance Gaussian in R^%d, per-coord std %s"
                  % (cov_mode, M, ["%.4f" % v for v in g_prior["std"]]), flush=True)

        offsets = data_lens_r3.sample_offsets(d.train_offsets, z_gen.shape[0], gen_rng)
        with torch.no_grad():
            q_gen = probs_batch(sim, theta_of(theta0, phi_raw,
                                              torch.as_tensor(z_gen, dtype=DT, device=device))
                                ).cpu().numpy().astype(np.float64)
        for i in range(q_gen.shape[0]):
            crop = codec10.pmf_to_image(q_gen[i], S)
            np.save(os.path.join(args.out, "crops", "generated_%03d.npy" % i), crop.astype(np.float32))
            np.save(os.path.join(args.out, "generated_%03d.npy" % i),
                    codec10.paste(crop, int(offsets[i, 0]), int(offsets[i, 1])).astype(np.float32))

        # ---------------------------------------------------------------- meta
        pc = np.asarray(proj["corrs"], dtype=np.float64)
        meta = {
            # tag the arm with the snapshot exactly as the baseline sweeps do, so the two
            # protocols land in one selector and one table without colliding
            "arm": ("%s %s" % (default_arm_name(args), step_tag(_snap_k)) if _multi
                    else default_arm_name(args)),
            "arm_base": default_arm_name(args),
            "snapshot_steps": int(_snap_k),
            "sweep": "iqp-save-every",
            "script": "train_iqp_affine_mlp.py",
            "round": "3b",
            "n_qubits": N_QUBITS, "K": K_MAX,
            # shared theta0 (1,024) + MLP (M*H + H + H*1,024) + N x M per-image latents.
            # No output bias: it would be exactly redundant with theta0.
            "param_count": int(NCODES + g_size(M, G_H) + n * M),
            "param_count_shared": int(NCODES + g_size(M, G_H)),
            "hidden": int(G_H), "gnorm": str(G_MODE),
            "g_input_scale": float(args.g_input_scale), "g_bias_std": float(args.g_bias_std),
            "g_init": g_init, "g_stats": g_stats,
            "nonlinearity_ratio": g_stats["nonlinearity_ratio"],
            "M": M, "latent": "learned-mlp", "protocol": "joint-mlp",
            "objective": args.objective, "prior_arg": args.prior,
            "phi_row_norm": float(PHI_SCALE),
            "theta0_seed_objective": "kl", "theta0_seed_steps": int(args.theta0_steps),
            "theta0_seed_loss": float(seed_best), "theta0_seed_corr": seed_corr,
            "n_train": int(n), "n_gen": int(q_gen.shape[0]), "n_holdout": int(n_ho),
            "S": float(S), "seed": int(args.seed), "device": str(args.device),
            "lr": float(args.lr), "steps": int(_snap_k), "zpen": float(args.zpen),
            "zpen_L0": float(L0),
            "zpen_penalty_over_data": (float(zpen_term / data_loss) if data_loss > 0 else None),
            "sigmas": [float(s) for s in sigmas],
            "sigma_mean_pauli_weights": [float(w) for w in mean_weights],
            "loss_final_total": float(best),
            "loss_final_data": data_loss, "loss_final_zpen": zpen_term,
            "loss_final_per_image_mean": float(rows_tr.mean()),
            "loss_final_per_image_max": float(rows_tr.max()),
            "loss_final_per_image_min": float(rows_tr.min()),
            "loss_first_logged": float(loss_curve[0][1]) if loss_curve else None,
            "train_corr_mean": float(corr_tr.mean()), "train_corr_min": float(corr_tr.min()),
            "train_corrs": corr_tr.tolist(),
            "projection_corr_mean": float(pc.mean()) if pc.size else None,
            "projection_corr_min": float(pc.min()) if pc.size else None,
            "projection_corr_median": float(np.median(pc)) if pc.size else None,
            "projection_corrs": proj["corrs"],
            "projection_restart_used": proj["restart"],
            "projection_restart_hist": (np.bincount(np.asarray(proj["restart"], dtype=np.int64),
                                                    minlength=max(int(args.infer_restarts), 1))
                                        .tolist() if pc.size else []),
            "projection_restarts": int(args.infer_restarts),
            "projection_steps": int(args.infer_steps),
            "projection_lr": float(args.infer_lr),
            "projection_seconds": float(proj_time),
            "ref_projection_mean_crop_corr": float(ref_mean.mean()) if ref_mean.size else None,
            "ref_projection_nearest_train_corr": float(ref_near.mean()) if ref_near.size else None,
            "z_norm_mean": float(np.linalg.norm(z_np, axis=1).mean()),
            "z_norm_max": float(np.linalg.norm(z_np, axis=1).max()),
            "conditional": False,
            "prior": prior_info,
            "train_seconds": float(train_time),
            "wallclock_seconds": float(time.time() - t0),
            "gen_offsets": [[int(a), int(b)] for a, b in offsets.tolist()],
        }
        with open(os.path.join(args.out, "meta.json"), "w") as fh:
            json.dump(meta, fh, indent=2, sort_keys=True, default=str)
        np.save(os.path.join(args.out, "theta0.npy"), theta0_np)
        np.save(os.path.join(args.out, "Phi.npy"), phi_np)          # best affine fit of g
        np.save(os.path.join(args.out, "gmap.npy"), phi_raw.cpu().numpy().astype(np.float64))
        np.save(os.path.join(args.out, "z_train.npy"), z_np)
        np.save(os.path.join(args.out, "z_holdout.npy"), z_ho_best)
        np.save(os.path.join(args.out, "loss_curve.npy"), np.asarray(loss_curve, dtype=np.float64))
        print("[mlp] wrote %d images to %s (arm=%s) in %.1f s"
              % (q_gen.shape[0], args.out, meta["arm"], time.time() - t0), flush=True)
        args.out = _snap_root
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
