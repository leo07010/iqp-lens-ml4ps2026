# Pre-registration — round 3: IQP as a generator of plausible lens images (goal B)

Written 2026-09-06 before any run. Builds on round 1 (job 345400), round 2 (job 347344, running) and the step-0
image analysis (`docs/LENS_IMAGE_CHARACTERISTICS.md`). Goal chosen by <user>: (B) show whether the IQP model family can
generate plausible gravitational-lens images, judged on physical quantities, not on class conditioning. Single seed.

## Representation (decided from measurements)

- Each 64×64 image is aligned on its intensity centroid (integer shift) and cropped to the central 32×32 window.
  Measured on 3,000 val images: the aligned crop keeps 99.4% of the light (minimum 98.4%); centroid offset from the
  frame centre is 3.4 ± 1.6 px (max 7.8); rms radius 9.44 ± 0.13 px. The offset (dy, dx) is recorded per image.
- Image as pmf over the crop: p(kx, ky) = crop / sum(crop), 10 qubits (5 + 5 bits), code = (kx << 5) | ky.
  Bit layout is irrelevant because the full generator set is used.
- Model: IQP on 10 qubits with **all 1,024 Z-strings** as generators (every subset, including ∅), exact simulation.
- Latent block (d_lat = 50): primary = the 50 generators with |s| ≤ 2 whose Walsh coefficients vary most across a
  disjoint pool of 2,000 training images (unsupervised selection, declared here); ablation = the paper's
  Hamming-first 50 (∅, 10 singles, 39 pairs).
- Training protocol: primary = **joint** — θ_core shared and θ_lat,i per training image, all trained together on the
  sum of per-image losses (Adam, lr 1e-3, 2×10^4 steps, full-batch exact); ablation = the paper's sequential
  Algorithm 1 with all parameters trained in stage 1 (round-2 "fixed" protocol).
- Objective: primary = closed-form MMD with the three-bandwidth rule at n = 10; secondary = exact KL(p̂ ‖ q).
- Generation: fit a Gaussian to the N trained latents in PCA-whitened coordinates (full covariance at N = 100,
  diagonal at N = 11); draw 200 latents → exact q → 32×32 field × S (S = mean total intensity of the training crops)
  → pasted into a black 64×64 frame at an offset drawn with replacement from the training images' (dy, dx). Both the
  64×64 images and the centred crops are saved.

## Arms (classes pooled; N ∈ {11, 100} training images drawn with seed 0 from train_x; 200 held-out val images)

| Arm | Notes |
|---|---|
| IQP-lowvar-joint-MMD | primary |
| IQP-lowvar-joint-KL | objective ablation |
| IQP-hamming-joint-MMD | latent-selection ablation |
| IQP-lowvar-paper-MMD | protocol ablation (sequential Algorithm 1, all params in stage 1) |
| RBM-10bit | Bernoulli RBM on the 10-bit codes (≈12k params: 10 × 1,000 hidden), same latent/prior/paste pipeline |
| DCGAN-32 | Radford-style generator for 32×32 crops (≈0.2M params), 20k steps, samples pasted with sampled offsets |
| CTRL-copy | the N training images |
| CTRL-mean | mean centred crop, pasted at sampled offsets |
| CTRL-ring | zero-learning parametric baseline: the mean centred radial profile stretched by (flux, rms radius, e1, e2) drawn jointly (bootstrap) from the training images, pasted at sampled offsets |

## Metrics (generated 200 vs held-out 200)

Physical quantities per image from intensity moments: centroid offset (dx, dy), total flux, rms radius, ellipticity
e1 = (Qxx − Qyy)/(Qxx + Qyy), e2 = 2Qxy/(Qxx + Qyy). For each quantity the 1-D Wasserstein-1 distance between the
generated and held-out distributions, divided by the held-out standard deviation; **physics score = mean over the
five**. Also: centred radial-profile RMSE vs the held-out mean profile; and, for continuity, PDF–JS, Conv2D-MMD²,
NN-train and NN-holdout on the 64×64 images. HOLDOUT-vs-HOLDOUT split-half values give the floor of every metric.

## Predictions, recorded before running

- P10: IQP-lowvar-joint-MMD beats CTRL-mean on NN-holdout and on the physics score at N = 100.
- P11: the KL objective is at least as good as MMD on the physics score (the MMD kernel under-weights high orders).
- P12: the Hamming-first latent is clearly worse than the low-order-variance latent on the physics score.
- P13: CTRL-ring, which learns nothing, beats every IQP arm on the physics score; if so, "IQP generates plausible
  images" holds only in the weak sense that it beats the mean image, not a five-parameter physical model.
- P14: no IQP arm beats DCGAN-32 on NN-holdout at N = 100.

## Decision rule for goal B, fixed now

At N = 100: **success** = IQP-lowvar-joint (MMD or KL) beats CTRL-mean on both NN-holdout and the physics score AND
its physics score is within 1.2× of CTRL-ring's. **Weak success** = beats CTRL-mean only. **Failure** otherwise.

## Budget

18 jobs (9 arms × 2 N), all cheap at n = 10 (1,024 amplitudes); one 8gpus node, < 1 h.

## Addendum (before running) — pseudo-time arm, added at <user>'s request on 2026-09-06

The paper's Algorithm 1 needs a 1-D conditioning variable. Lens images have none, so two more arms use a measured
physical scalar as pseudo-time: the training images are sorted by ellipticity e1 (from the centred crop's second
moments) and the rank/e1 value plays the role of t.

| Arm | Notes |
|---|---|
| IQP-lowvar-pseudotime-MMD | Algorithm 1 along the e1 ordering: stage 1 = all parameters on the lowest-e1 image; then per-image latent (3000 steps, warm-started in e1 order); latent trajectory piecewise-linear in e1 (clamped at the ends) |
| CTRL-interp-e1 | zero-parameter control: for a target e1, linear interpolation of the two training crops nearest in e1 |

Generation for these arms is conditional: for each of the 200 held-out images the latent (or crop) is evaluated at
that image's e1. Paired metrics on the centred 32×32 crops: |e1_gen − e1_target|, |e2|, |flux|, |radius| errors
(mean over 200), and the paired L2 to the target crop. The distributional metrics of the main table are also
reported (crops pasted at sampled offsets). A "random pairing" reference (the unconditional IQP-lowvar-joint arm's
samples paired with held-out images at random) gives the floor of the paired metrics.

- P15: the pseudo-time IQP tracks e1 better than random pairing (|Δe1| clearly smaller), showing that latent
  adaptation carries the conditioning; but CTRL-interp-e1 tracks it better still, and on the other quantities the
  pseudo-time IQP is no better than the unconditional arm.
Total jobs: 22 (11 arms × 2 N).

## Addendum (before running) — hardware-compatible objective arm, added 2026-09-07 at <user>'s request

The KL arm needs the values q(x), which only exact classical simulation provides; a quantum device only yields
samples, and the paper's MMD is chosen because it is a sum of squared expectation-value differences, all estimable
from samples. To keep a hardware-compatible counterpart of the KL diagnostic, one more arm is added:

| Arm | Notes |
|---|---|
| IQP-lowvar-joint-MMDflat | identical to IQP-lowvar-joint-MMD but with flat Walsh weights c_s = 2^-10 for every s (equivalently the MMD with a delta kernel; still Σ c_s (⟨Z_s⟩_θ − ⟨Z_s⟩_data)², hence sample-estimable) |

- P16: MMDflat lies between MMD-auto3 and KL on the physics score; if it matches KL, the low-order kernel weighting is
  the cause of the haze and a hardware-compatible fix exists; if it matches MMD-auto3, the limitation is the
  phase-only parametrisation itself.
Total jobs: 24 (12 arms × 2 N).

## Addendum (before running) — latent-prior arms, added 2026-09-07 after the round-2 train-fit diagnostic

The round-2 diagnostic (scratchpad diag_r2; numbers recorded in the round-2 results document) showed two things: with
the paper's protocol every training image except the first reconstructs as noise (corr 0.47) because only 50 latent
parameters are free against a core fitted to one image; and a Gaussian fitted to 100 latents in 50 dimensions puts no
mass near any real latent (draws land 4.8× the anchors' own nearest-neighbour spacing). Round 3's joint protocol
addresses the first; the prior is addressed by two extra arms of the primary configuration:

| Arm | Prior over latents |
|---|---|
| IQP-lowvar-joint-MMD (unchanged) | PCA-whitened Gaussian, rank = min(N−1, 50) |
| IQP-lowvar-joint-MMD-prior8 | PCA-whitened Gaussian truncated to rank 8 (the ≈7-dimensional image manifold of the step-0 analysis) |
| IQP-lowvar-joint-MMD-resample | non-parametric: a trained latent drawn with replacement plus Gaussian jitter of 0.2× the median nearest-neighbour spacing |

- P17: prior8 and resample both beat the rank-50 Gaussian on NN-holdout at N = 100; resample is closest to
  CTRL-copy (it nearly reproduces training images) and its physics score is the best of the IQP arms.
Total jobs: 28 (14 arms × 2 N).

Scoring note added 2026-09-07, before running: the round-2 analysis showed that Conv2D-MMD² and NN distances are
sensitive to each arm's mean total intensity (the pmf arms' MMD² gain over round 1 largely disappeared once totals were
aligned). Round 3 therefore also reports an intensity-aligned secondary block — every generated image rescaled so that
its total equals the held-out mean total — next to the primary scores. The physics score is unaffected (flux is a
separate quantity; radius and ellipticity are scale-invariant). The decision rule uses the primary scores.

## Round 3b — learned latent directions (declared 2026-09-07 before running; mechanism tests in scratchpad mech_test*)

Two mechanism tests preceded this. (i) A single aligned crop is represented almost exactly by a full 1,024-phase
10-qubit IQP under every objective (corr: KL 0.9999, MMD-flat 1.000, paper auto3 0.996), so representability and the
kernel are not the bottleneck. (ii) Interpolating between two images along fixed-basis latents (paper's Hamming
block, or ancilla mixing angles) collapses mid-path unless the path is short, and even then produces a pixel
cross-fade, not a geometric morph; an ancilla marginal is an equal-weight mixture of holograms (verified to 1e-18),
so ancillas add nothing. (iii) Learning the latent directions — every gate angle an affine function of a small
latent, θ = θ⁰ + Φᵀz with Φ ∈ R^{M×1024} shared and z_i ∈ R^M per image, trained jointly — fits 20 training crops
(corr 0.96 at M = 4) and, decisively, reconstructs unseen crops through the learned subspace (held-out projection
corr 0.91 at M = 4 vs 0.74 for the mean image and 0.93 for the nearest training image). At M = 8/16 the held-out
inference failed (0.09/0.16) with the naive single-start optimiser, and Gaussian-sampled z gave weak rings.

Round 3b therefore adds, on the same data, crops, metrics and controls as round 3:

| Arm | Latent | Objective | Prior over z |
|---|---|---|---|
| IQP-affine-M4-KL-gauss | learned Φ, M = 4 | exact KL | full-covariance Gaussian on trained z |
| IQP-affine-M4-KL-resample | same | KL | bootstrap of trained z + Gaussian jitter 0.2× median NN spacing |
| IQP-affine-M4-MMDflat-resample | same | MMD, flat weights (hardware-compatible) | resample |
| IQP-affine-M4-auto3-resample | same | paper's MMD kernel | resample |
| IQP-affine-M8-KL-resample | M = 8, held-out inference with 8 restarts incl. nearest-training-z init | KL | resample |
| CTRL-pca4-gauss / CTRL-pca4-resample / CTRL-pca8-resample | pixel-space PCA of the centred crops (classical linear analogue) | – | as named |

Rows of Φ are L2-normalised; θ⁰ initialised from a single-hologram fit to the mean crop. Extra metric for every arm:
**held-out projection corr** — for each of the 200 held-out crops, the best reconstruction reachable inside the
model's latent space (z optimised, everything else frozen), mean and min; for PCA arms this is the orthogonal
projection. This measures whether the learned subspace covers unseen images, independently of the prior.

Predictions: P18 IQP-affine-M4 held-out projection corr ≥ 0.90 at N = 100, above CTRL-mean (≈ 0.74) and within
0.05 of CTRL-pca4. P19 resample priors beat Gaussian priors on NN-holdout and ringiness for both IQP and PCA.
P20 IQP-affine-M4-*-resample beats CTRL-mean on NN-holdout and on the physics score (the first IQP arm to do so),
but does not beat CTRL-pca4-resample by more than the split-half noise on any metric — i.e. the IQP behaves as a
linear model in phase space. P21 auto3 is worse than KL and MMD-flat on projection corr. Decision rule for 3b: "IQP
is a genuinely useful generator on this data" requires beating CTRL-pca4-resample on the physics score AND on
held-out projection corr by more than the split-half floor; matching it is "IQP = linear model"; losing to it is failure.
Total: 8 arms × 2 N = 16 jobs.

### Round 3b amendment (2026-09-07, before running) — settings from the completed mechanism test

The final mechanism-test report (scratchpad mech_test2) adds: M = 2 generates best (100% of Gaussian samples are
rings, radius 9.40 ± 0.21 vs data 9.21 ± 0.15, samples as far from the training set as real held-out images, but
under-dispersed: e1 sd 0.048 vs 0.099); M = 4 projects best (0.911); a random-direction control gives 0.67 (= no
latent), so the gain is from learning; M ≥ 6 collapses because ‖z‖ exceeds the model's coherence radius (≈ 0.03–0.05);
the paper's auto3 kernel lowers corr by ≈ 0.1 but gives better ellipticity dispersion (0.084). Arms are amended to:

| Arm | M | Objective | Prior | Note |
|---|---|---|---|---|
| IQP-affine-M2-KL-gauss / -resample | 2 | KL | full-cov Gaussian / resample | generation working point |
| IQP-affine-M2-auto3-gauss | 2 | paper kernel | Gaussian | dispersion check |
| IQP-affine-M4-KL-gauss / -resample | 4 | KL | Gaussian / resample | projection working point |
| IQP-affine-M4-MMDflat-resample | 4 | flat MMD | resample | hardware-compatible |
| IQP-affine-M4-auto3-resample | 4 | paper kernel | resample | |
| IQP-affine-M8-KL-zpen-resample | 8 | KL + z-penalty 0.3 | resample | capacity test: does regularisation lift the M ceiling? |
| CTRL-pca2-gauss, CTRL-pca4-gauss, CTRL-pca4-resample | – | – | as named | classical linear analogue |

Training: Adam lr 0.02, 4000 steps, θ⁰ from a mean-crop fit, Φ rows L2-normalised, z-penalty 1e-3 except the zpen arm.
Held-out projection: z-only, 1000 steps, lr 0.005, 8 restarts (zero, nearest-training z, random). Extra reported
columns: within-sample NN distance (diversity) and e1/e2 standard deviations, each next to the held-out value.
P22: M2 arms have the best ringiness and physics score among IQP arms but the lowest diversity; P23: the z-penalty does
not lift the M = 8 projection above M = 4's. Total: 11 arms × 2 N = 22 jobs.

Implementation notes for 3b, fixed before running: the z-penalty is expressed relative to the initial data loss
(penalty = zpen × L₀ × mean‖z‖², L₀ = data loss at z = 0 after the θ⁰ seed), so "1e-3" and "0.3" mean the same
fraction under every objective; the same relative penalty is used during held-out inference. θ⁰ is seeded with a KL
fit of the mean crop for every arm, so arms differ only in what they train on. Restart selection during held-out
inference is by best correlation. Round 3 and 3b are scored together only once both are complete.
