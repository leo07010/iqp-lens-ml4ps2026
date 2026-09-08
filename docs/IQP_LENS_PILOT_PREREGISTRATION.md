# Pre-registration — IQP generative pilot on DeepLense Model II (dark-matter substructure images)

Written 2026-09-06 before any run. Follows the replication pipeline in `experiments/13_iqp_turbulence_replication/`
(exact 18-qubit IQP, Algorithm 1 of arXiv 2603.06440, RBM / DCGAN baselines, controls, PDF–JS and Conv2D-MMD²).
Single seed (seed 0) throughout, at Leo's request. Purpose: see whether the paper's method does anything useful on
Leo's own data, with the controls that the turbulence replication showed to be necessary.

## Data

`<HOME>/mae-lensing/model_II.npz`: train_x (80,194, 64, 64) float32 in [0, 1], val_x (8,910, 64, 64),
labels 0/1/2 = `class_names` (axion, cdm, no_sub — order read from the file at run time and recorded in meta.json).
Images are mostly dark (mean intensity ≈ 0.03) with a bright arc.

Mapping to the paper's encoding, applied literally: one image = one "snapshot"; tuple (x, y, intensity) with
kx, ky = pixel index (6 bits each) and kz = floor(64 · intensity) clipped to 0..63 (6 bits), i.e. (a_z, b_z) = (0, 1)
fixed for every class so classes are comparable; code pool = the 4,096 pixels; ground-truth field = the quantized
image. Known consequence, measured on 2,000 val images before running: 83.7% of pixels fall in intensity bin 0 (< 1/64) and
86.5% in bins 0–1, so the 6-bit z-channel is dominated by the background; no log/sqrt transform in the primary arms
(a sqrt-intensity variant is a secondary arm). class_names order in the file: axion, cdm, no_sub.

## Design (per class, independently)

Training set = N images drawn from train_x of that class with seed 0, N ∈ {11, 100} (the paper's two regimes).
Held-out = 200 val images of the same class (seed 0), used for every arm.

| Arm | Synthesis of the 200 evaluation images |
|---|---|
| IQP-31k (Algorithm 1: core on image 1, per-image latent 3000 steps warm-started in index order) | latent ~ N(μ, diag σ²) fitted to the N trained latents; 10^5 samples per image, mean aggregation (primary) and last-write (secondary) |
| RBM-12k (same protocol) | same Gaussian-latent sampling |
| DCGAN 51k / 0.2M / 1.2M, 20k G-steps | 200 unconditional samples |
| CTRL-marginal | product-Bernoulli per image, Gaussian-latent sampling |
| CTRL-copy | the N training images, cycled to 200 (memorisation control) |
| CTRL-mean | the mean of the N training images, repeated 200 times |
| IQP-31k-sqrt (secondary, N = 11 only) | as IQP but intensity → sqrt(intensity) before quantisation |

Metrics on the 200 generated vs 200 held-out: PDF–JS (B = 50, log2), Conv2D-MMD² (random encoder seed 0, median
heuristic), plus two nearest-neighbour L2 distances per generated image (to the training set: memorisation; to the
held-out set: coverage), reported as means. QCLI / CCI per class: per-image (M = 4,096 ≪ 2^18, the regime where the
indicator's null is near 0) over 200 images, and pooled over N = 100 images, each with an i.i.d.-uniform null at
matched sample count.

## Predictions, recorded before running

- P1: IQP does not beat DCGAN-1.2M on either metric in either regime (the turbulence result carries over).
- P2: CTRL-copy scores best or near-best on both metrics in the N = 100 regime (the metrics reward memorisation);
  CTRL-mean scores well on Conv2D-MMD² and badly on PDF–JS.
- P3: IQP's generated images are blurry class-mean-like arcs (the model captures p(z | x, y) averaged over the class);
  its NN-distance to held-out is close to CTRL-mean's.
- P4: per-image QCLI for lens images is low (< 0.1) and not separable from the matched null; CCI is high because of the
  arc's spatial structure. Classes will not separate on the map.
- P5: the sqrt variant lowers PDF–JS for IQP but does not change the ordering.

## Decision rule

There is no paper claim to reproduce here; the deliverable is the table plus the controls. "IQP is useful on this data"
would require IQP to beat CTRL-copy and CTRL-mean on both metrics AND have a smaller NN-distance to held-out than
CTRL-mean, in the N = 100 regime. Anything less is reported as "not useful under this protocol".

## Budget

51 jobs (17 per class, incl. the sqrt secondary arm), 8gpus packed 8-wide; the N = 100 IQP/RBM arms (~3.3×10^5 steps ≈ 1 h each) dominate.

## Addendum 2026-09-06 — round 2, declared before running: image-specific fixes for IQP (requested by Leo)

Round 1 (job 345400) verdict: not useful under this protocol; IQP was the worst trained model. Three image-specific
mechanisms were identified: (i) 84% of pixels sit in intensity bin 0, so the tuple pool is mostly background;
(ii) the Hamming-first latent block (∅, 18 singles, 31 pairs) cannot move the arc geometry, so Gaussian-latent samples
are small distortions of the core image (NN-holdout 5.8–7.7 vs CTRL-mean 3.6–3.8); (iii) the paper's tuple-per-pixel
representation suits dense fields, not a sparse arc on a black background. Round 2 tests the fixes, single seed:

| Arm | Encoding | Generators | Stage 1 | Data pmf | Reconstruction |
|---|---|---|---|---|---|
| IQP-tuple-sqrt-fixed | tuple (x, y, sqrt-intensity), 18 qubits | \|s\| ≤ 6 | all parameters trained (latent not frozen) | full 4,096-pixel pmf | 10^5 samples, mean aggregation (as round 1) |
| IQP-pmf-full-fixed | **image as pmf**: pixels (x, y) drawn ∝ intensity, 12 qubits | all 4,096 Z-strings | all parameters | exact normalised image | exact q(kx, ky) × S, S = mean total intensity of the training images |
| IQP-pmf-K6-fixed | same | \|s\| ≤ 6 (2,510) | all parameters | exact | same |
| IQP-pmf-full-paper | same | all 4,096 | paper protocol (latent frozen) | 5×10^4 sampled codes | same |
| RBM-sqrt | tuple sqrt, 18 bits | – | as round 1 | as round 1 | as round 1 |

All arms keep Algorithm 1's per-image latent (d_lat = 50, Hamming order; for the 12-qubit model that is ∅ + 12 singles +
37 pairs), the 3000-step warm-started adaptation per training image, and Gaussian-latent synthesis of 200 images.
Round-1 arms (IQP raw, RBM raw, DCGAN ×3, CTRL-copy, CTRL-mean, HOLDOUT ref) are reused, not rerun. Same metrics
(PDF–JS, Conv2D-MMD², NN-train, NN-holdout), same 200 held-out images, three classes, N ∈ {11, 100}.

Predictions: P6 the sqrt+fixed-protocol tuple arm beats round-1 IQP on every metric and reaches RBM-raw's NN-holdout
(4.5–4.9). P7 the pmf-full-fixed arm has the best NN-holdout of all IQP arms and beats RBM-raw; it ends within ±0.5
of CTRL-mean (3.6–3.8) but does not beat it. P8 pmf-full-paper (frozen latent, sampled pmf) is clearly worse than
pmf-full-fixed, replicating the turbulence attribution on images. P9 no IQP arm beats the best DCGAN or CTRL-copy on
PDF–JS / MMD² (those reward memorisation). Decision rule for "IQP is a reasonable generative model on this data":
in the N = 100 regime, beat RBM-sqrt on NN-holdout and PDF–JS, and beat CTRL-mean on NN-holdout.

Scoring note for the pmf arms, added before results are seen: their fields are q(kx, ky) × S with S the mean total
intensity of the training images, so pixel values can exceed the real images' maximum of 1.0 if the model is more
peaked than the data. Primary scores use the raw scaled field; a secondary score with values clipped to [0, 1] is
also reported, and the mean total intensity of generated vs held-out images is printed next to every pmf row.
