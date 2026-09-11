# Pre-registration — replication of arXiv 2603.06440 (IQP generative model on the D2 turbulence dataset)

Drafted 2026-09-04 before any training run. Code lives in `experiments/13_iqp_turbulence_replication/`.
All dataset placeholders were filled in the Addendum on 2026-09-05, before the first run.

## What is being replicated, and why

Liu, Placidi, Brunner, Rinaldi (Quantinuum, arXiv 2603.06440 v2) train an 18-qubit IQP generator on tuples
(x, y, velocity magnitude) from the D2 wake-of-cylinder turbulence dataset (Khojasteh et al. 2022) and report, in
their Table 2, that with only 11 training snapshots the IQP model beats an RBM and three DCGANs on two metrics
(PDF–JS and Conv2D-feature MMD²), and that DCGAN only catches up with ~100 snapshots. The paper released no code.
The goal here is narrow: rebuild the pipeline from the paper text and test whether the same phenomenon appears on the
same data. No quantum-advantage claim is made or tested; the whole pipeline is classical (exact simulation at n = 18).

Two deliberate deviations from the paper, both disclosed here:
1. Exact simulation. The paper estimates the MMD with 500 model samples per step (Ref [38] estimator). At n = 18 the
   state has 2^18 amplitudes, so we compute q_θ, all ⟨Z_s⟩ and the closed-form MMD exactly with Walsh–Hadamard
   transforms (`DESIGN_exact_iqp.md`). This removes estimator noise; it cannot make the model family more expressive.
2. Two extra controls the paper lacks (see Arms), because "IQP beats RBM/DCGAN" is uninterpretable without them.

## Arms

Data-matched regime = the 11 anchor snapshots t ∈ {1, 100, 200, …, 1000}. Evaluation = 200 unseen τ drawn
uniformly from [1, 1000] excluding anchors (seed 0, fixed once), 200 generated vs 200 ground-truth fields.

| Arm | Model | Train snapshots | Seeds | Notes |
|---|---|---|---|---|
| IQP-31k | IQP, all Z-strings |s| ≤ 6 (31,180 params), d_lat = 50 | 11 | 3 | primary |
| IQP-63k | IQP, |s| ≤ 7 (63,004 params), d_lat = 50 | 11 | 3 | primary |
| RBM-12k | Bernoulli RBM 18×630 (11,988 params), latent = first 50 flattened params | 11 | 3 | paper baseline |
| DCGAN-51k / 0.2M / 1.2M | generator param count; latent 128 | 11 and 100 | 3 | paper baseline, two regimes |
| DCGAN scaling | same three tiers | N ∈ {10, 30, 50, 70, 90, 100} | 1 | paper Fig. (data-scaling) |
| CTRL-marginal | product-Bernoulli over the 18 bits fitted per anchor (18 params), same latent interpolation | 11 | 3 | control: is the win just marginals? (amended from 1 to 3 seeds on 2026-09-05, before any run; cost is negligible) |
| CTRL-interp | no model: piecewise-linear interpolation of the two neighbouring anchor ground-truth fields | 11 | – | control: do the metrics reward modelling at all? |

Sensitivity arms (secondary, single seed, IQP-31k only): random core init instead of data-dependent init; data-dependent
latent init (`--lat-init data`) instead of random; lr 1e-3; σ ∈ {0.6, 0.3} (Ref [38] 16-qubit table values) and
σ ∈ {1.4105, 0.6746} (mean Pauli weight 2 and 6 at n = 18) instead of the three-bandwidth rule; data pmf from the full
4096-cell grid instead of 5×10^4 sampled tuples.

## Design choices the paper does not state (fixed here, before running)

| Item | Paper | Our choice |
|---|---|---|
| Optimizer / lr (IQP core and latent) | not given (Adam 1e-4 only for the Fig. 3 probe) | Adam, lr 1e-4, both stages |
| Kernel bandwidths | "Gaussian kernel" | Ref [38]'s three-bandwidth rule for larger systems, computed from the t=1 data: σ1 such that the kernel's mean Pauli weight n(1−e^{−1/2σ²})/2 = 2, σ3 = sqrt(median pairwise Hamming distance), σ2 = sqrt((σ1²+σ3²)/2); coefficients averaged over the three. (At n = 18 the 16-qubit table values {0.6, 0.3} give mean weights 6.8 and 9.0, i.e. no low-order coverage.) |
| Core init | hints at Ref [38] data-dependent init | Ref [38] init ported: singles from marginals, pairs from ±1 covariance, rest N(0, 0.01) |
| θ_lat^(1) | "random" | N(0, 0.01), fixed during stage 1. Note this freezes all 18 single-qubit terms near 0 for the whole of stage 1 (they are inside the first-50 latent block), so the core must encode the marginals through higher-order terms; this is the paper's design, kept as is |
| Steps | 3×10^4 core, 3000 per latent | same |
| Data pmf per snapshot | N_t = 5×10^4 tuples | each snapshot has a code pool = every native grid point quantized to (6, 6, 6) bits (D2: ~10^5 points of the 308×328 slice); N_t = 5×10^4 codes drawn with replacement from the pool (seeded); the 64×64 ground-truth field is the mean z per (x, y) bin of the same pool. Time index t is 1-based throughout; one hold-out set of 200 τ shared by every arm |
| Synthesis | "e.g. 10^5 samples" then "aggregate" | 10^5 codes; mean z per (x, y) cell; empty cells = mean of non-empty 8-neighbours |
| Reconstruction grid | never stated (raw snapshot quoted as 64×64×64) | 64×64 (6 bits per axis) |
| Quantization interval [a, b] | "empirically from the dataset" | z: min/max over the 11 anchors; x, y: grid index |
| PDF–JS | B = 50, joint min/max, log base unstated | log base 2 (matches QCLI's eq. 53) |
| Conv2D encoder | 3 × Conv–ReLU stride 2 + GAP, random init | channels 16/32/64, k = 3, seed 0, inputs scaled to [0, 1] by joint min/max |
| MMD² estimator | biased, N = M = 200, median-heuristic σ | same |
| RBM training | not given | PCD-1, 100 chains, Adam 1e-3, batch 512, 3×10^4 / 3000 steps; block Gibbs generation |
| RBM latent block | "first 50 elements of a fixed flattened ordering (weights and biases)" | flattened order [b_v (18), b_h (630), W] so the block = all 18 visible biases + the first 32 hidden biases, mirroring the IQP block (∅ + 18 singles + 31 pairs). Note the IQP's θ_∅ is a global phase, so the IQP block has 49 effective degrees of freedom |
| DCGAN | "Radford et al." + param counts; trained on *quantized* 2D fields | 4×4→64×64 transposed-conv stack, BN+ReLU / LeakyReLU 0.2, non-saturating BCE, Adam 2e-4 (0.5, 0.999), batch min(16, N), 20k G-steps, no augmentation, last checkpoint; inputs are the 64×64 fields with z quantized to the same 6-bit bins (bin centres) |
| "100 snapshots" for data-rich DCGAN | which 100 not given | 100 evenly spaced t in [1, 1000] |
| DCGAN vs unseen τ | no time conditioning | 200 unconditional samples vs the same 200 GT fields |
| QCLI / CCI on D2 | protocol not given | exact enumeration on the pooled tuple pmf; plug-in entropies; Chow–Liu tree. Because I_QCLI of i.i.d. data is only ≈ 0 when the sample count M ≪ 2^n (it rises towards a degenerate maximum as M grows past ~C(n, n/2) ≈ 4.9×10^4 at n = 18), we report I_QCLI as a curve over M ∈ {10^3, 10^4, 5×10^4, 10^5, all} for D2, per snapshot and pooled, and state which M is used for the map point (5×10^4, the paper's N_t) |
| Seeds | not given (Table 2 looks single-run) | 3 seeds for primary arms; report mean ± std |
| D2 identity, field extraction | "different simulation domains" only | D2 = `2D_velocity_snapshot_sub_domain_2` (1000 snapshots), D1 = `..._sub_domain_1` (100 snapshots). Each D2 file holds three z-planes (z = 1.97, 3.0, 3.98; labels −1/0/+1) of a 308×328 (x, y) grid with u, v, w. Primary: the middle plane z = 3.0 only; z-channel = |v| = sqrt(u²+v²+w²). Sensitivity: the three planes pooled. x, y are binned to 6 bits over the grid extents (≈ 24 native points per 64×64 cell) |

## Tests (must pass before submission)

`test_iqp_exact.py`: exact simulator vs dense statevector (n ≤ 10, < 1e-10); closed-form MMD vs explicit kernel
matrix (n = 8); gradcheck; sum(q) = 1 at n = 18. `test_baselines.py`: parameter counts, smoke runs, output contract.
`test_diagnostics.py`: QCLI/CCI vs brute force (n = 6), uniform → 0, parity → high QCLI, tree → CCI ≈ 0.
A synthetic-data end-to-end smoke (all arms, tiny steps, CPU) must produce files satisfying `INTERFACES.md`.

## Decision rule, fixed now

Let m(arm) be the mean over seeds of a metric on the 200 unseen τ, and s(arm) the across-seed std.

- **Reproduced**: for both metrics, each IQP arm has m lower than every 11-snapshot baseline (RBM, three DCGANs) by
  more than max(s(IQP), s(baseline)), AND at least one 100-snapshot DCGAN tier matches or beats IQP-63k on at least
  one metric. This is the paper's regime-dependent claim.
- **Partially reproduced**: the ordering holds on one metric only, or holds but within seed noise.
- **Not reproduced**: an 11-snapshot baseline matches or beats IQP on both metrics.
- **Uninterpretable** (overrides the above): CTRL-interp beats every trained model on both metrics, or CTRL-marginal
  is within seed noise of IQP on both metrics. Either means the metrics do not test what the paper says they test.

Absolute numbers are secondary: our data processing is reconstructed from prose, so we compare orderings, and report
our Table 2 side by side with the paper's for the record. No arm is added, dropped, or re-tuned after seeing results.

## Predictions, recorded before running

- P1: H1 (IQP beats all 11-snapshot baselines) holds on PDF–JS — likely (~0.7). The IQP models the marginal z-pmf
  directly through the tuple distribution, and a GAN on 11 images overfits.
- P2: H1 on Conv2D-MMD² — coin flip (~0.5). IQP fields are aggregated from ~24 samples per cell and are noisy; a
  random-conv feature space may penalise that noise more than a blurry GAN sample.
- P3: 100-snapshot DCGAN-0.2M/1.2M matches or beats IQP on at least one metric — likely (~0.7), as in the paper.
- P4: CTRL-interp will be competitive on PDF–JS (marginal histograms drift slowly with t) and may beat everything
  there; if it does, PDF–JS is a weak metric for this task and the paper's headline number should be read accordingly.
- P5: D2 pooled QCLI lands near 0.2, as the paper says real datasets do; CCI unknown.

## Addendum (before submission) — dataset inspection and any change to the above

### 2026-09-04 — dataset identified, download blocked by a repository outage

The Khojasteh et al. 2022 data sits on the INRAE Recherche Data Gouv Dataverse (DOI 10.15454/GLNRHK; 24 files, 300 GB).
Sub-domain 2 is saved every DNS step and has exactly 1000 snapshots, sub-domain 1 every 10 steps with 100 snapshots, so the
paper's "D2 with t in {1, ..., 1000}" is `2D_velocity_snapshot_sub_domain_2.zip` (4.1 GB) and D1 is the sub-domain-1
counterpart (9.3 GB). Only D2 is needed for Table 2; D1 is only a point on the Fig. 5 map and is deferred (disk).
The repository answered HTTP 503 (maintenance) for the whole evening; a poller in
`<WORK>/data/turbulence_khojasteh2022/wait_and_download_d2.sh` downloads D2 when it returns.

Still open until the file is inspected: which velocity components the 2D snapshot carries (z-channel = |v| over the
components present), the native 2D grid (expected 308x328), and how it is binned to 64x64 (choice: mean over the
grid points falling in each 6-bit (x, y) bin; tuples for the IQP are drawn from the native grid points, not the binned
field). These will be written here before the first SLURM submission.

### 2026-09-05 — D2 downloaded and inspected (before the first run)

Download verified byte-exact (4,116,034,340 B). The zip holds 1000 text files `2D_sub_domain_2_z=[-1,0,1]_t={t}.txt`,
t = 1..1000 complete. Each file: 2 header lines then 303,072 rows `x y z u v w` = three z-planes (z = 1.96875, 3.0,
3.984375) × 308 x-values (3.997–7.995) × 328 y-values (3.0–4.995). The paper's "64×64×64 grid" remark does not match
this layout; we treat it as loose wording. Decisions, fixed now: primary arm uses the middle plane (z = 3.0) as "the 2D
snapshot"; z-channel = 3D velocity magnitude sqrt(u²+v²+w²); each snapshot's code pool = its 101,024 native points
quantized to (6, 6, 6) bits, N_t = 5×10^4 drawn from the pool; ground-truth 64×64 field = mean |v| per (x, y) bin of
the same pool. Sensitivity: the three planes pooled as one snapshot (303,072 points). The 200 hold-out τ are drawn once
with numpy seed 0 from [1, 1000] minus the anchors and shared by every arm.

### 2026-09-05 — pre-submission verification notes

Independent math verification (PennyLane statevector oracle, explicit-kernel MMD, finite-difference gradients,
brute-force QCLI/CCI, hand-computed metrics) passed all eight checks at machine precision. Two bugs were caught and
fixed before submission: (i) in D2 mode several arms re-derived the z quantization range from the mean-binned
64×64 fields instead of the native |v| range used to encode the code pools (an 18.8% decode-scale mismatch); every
arm now takes (a_z, b_z) = (0.0, 2.0348) from `data_d2.get_z_range()`; (ii) a NaN guard in the kernel-weight code
for β → 0. IQP arms run on GPU (2× faster than 4 CPU threads at n = 18; 11 ms per step). CTRL-marginal runs 3 seeds.

### 2026-09-05 — post-hoc, exploratory: reconstruction rule re-examined after the main run (job 342220)

The main run finished (55 arms, no failures). Outcome under the fixed decision rule: **not reproduced**, and the
"uninterpretable" override fires (CTRL-interp scores 0.001 / 0.002 on PDF-JS / MMD², beating every trained model;
CTRL-marginal ties IQP on MMD²). Before writing conclusions we re-read the paper: the synthesis step says samples
are aggregated "according to the same probability-aggregation/decoding pipeline used during training" and that
10^5–10^6 samples are drawn per synthesised time. Our pre-registered reading (10^5 samples, mean z per cell, ≈24
samples per cell) is the noisiest admissible one; "probability-aggregation" reads more naturally as the
probability-weighted conditional mean E[z | x, y], which we can compute exactly from q_θ with no sampling noise.

Exploratory follow-up, declared before its results are seen: re-synthesise every IQP arm's 200 hold-out fields from
the saved θ with (a) the exact conditional mean and (b) 10^6 samples; RBM and CTRL-marginal get matching variants.
No model is retrained. These numbers are reported as exploratory, alongside the pre-registered ones, and the
pre-registered decision stands as the confirmatory result. Prediction: IQP PDF-JS falls substantially under (a)
(aggregation noise is a large part of 0.24), but CTRL-interp still wins both metrics and the ordering against RBM
and DCGAN is unlikely to flip to the paper's.

### 2026-09-06 — post-hoc, exploratory: baseline alignment to the paper's literal wording (requested by <user>)

The paper specifies its baselines only at the architecture level (RBM: same 11 anchors, same 18-bit tuples, latent =
"first 50 entries of a fixed flattened parameter ordering (weights and biases)", rest fixed; DCGAN: Radford et al.,
51k / 0.2M / 1.2M, latent 128, quantized 2D fields, 11 vs 100 snapshots). Everything stated is already matched. What
is not stated (training recipe, steps, learning rates, RBM hidden size beyond the ≈12k count, DCGAN input size) was
chosen by us with standard defaults, and our baselines are far stronger than the paper's (RBM 0.169 / 1.472, GAN@11
0.24–0.37 / 1.40–1.45 in the paper). Declared before running:

- RBM-literal: latent block = first 50 entries of [W.flatten(), b_v, b_h] (= the 50 weights of visible unit 0), the
  most literal reading of the paper's sentence; two training recipes (our PCD-1 + Adam, and CD-1 + SGD
  lr 0.05 batch 64, 30 epochs-equivalent), each with and without the literal latent block. Single seed (seed 0) at
  <user>'s request on 2026-09-06; seeds can be added later.
- DCGAN training-budget sweep at 11 snapshots: generator steps ∈ {300, 1000, 3000, 20000} × 3 tiers, single seed
  (seed 0), last checkpoint, everything else unchanged.
Purpose: locate which (if any) of these settings reproduces the paper's baseline numbers. Prediction: the paper's
GAN@11 numbers correspond to an under-trained budget (≤ 1000 steps) and its RBM numbers to the crippled literal
latent block and/or a weak CD-1 recipe. Whatever is found, the confirmatory verdict above is unchanged; this section
only calibrates what "the paper's baselines" were.
