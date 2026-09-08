# Facts sheet for the ML4PS 2026 writeup

Every number below carries its provenance (`file:line` or a jsonl field query). Do not retype numbers
from memory when drafting the paper — copy from here or re-derive with the query shown. All experiments
are single-seed (seed 0) unless stated otherwise. Paths are relative to
`<HOME>/gsoc2026-deeplense-qml/` unless given absolute.

Framing agreed with Leo (see memory `ml4ps2026-paper.md`): paper-style comparison IQP vs RBM vs DCGAN
only (PDF–JS / Conv2D-MMD², N = 11 vs 100), no quantum-advantage claim, exact classical simulation
disclosed, single seed disclosed. Core contribution: learned affine latent directions
(θ = θ⁰ + Φᵀz) let a 10-qubit IQP generate and generalise on lens crops. Secondary: the
correlation–complexity map places lens images at high QCLI, with matched nulls showing most of that
position is an artifact of the encoding, not of lens structure — this is a genuinely new methodological
finding (permutation nulls), not in the original paper.

---

## 1. Dataset (DeepLense `model_II.npz`)

Source: `docs/IQP_LENS_ROUND3_PREREGISTRATION.md:7-11`; class/shape facts from memory
`dataset1-equals-model-ii.md` and `docs/LENS_IMAGE_CHARACTERISTICS.md:5`.

- File: `<HOME>/mae-lensing/model_II.npz` — train 80,194 / val 8,910 images, 64×64, float32, ∈ [0,1].
- 3 classes: `axion`, `cdm`, `no_sub` (dark-matter substructure morphology), pooled across classes for
  the generative experiments (goal is "plausible lens image", not class-conditional generation).
- `model_II.npz` is byte-identical to `dataset1.npz` — memory `dataset1-equals-model-ii.md`. Every image
  has exactly one pixel = 0 and one pixel = 1 (per-image min–max normalized) — `docs/LENS_IMAGE_CHARACTERISTICS.md:31`.
- Representation used for the IQP experiments: each 64×64 image is aligned on its intensity centroid
  (integer shift) and cropped to the central 32×32 window.
  - Measured on 3,000 val images (`docs/IQP_LENS_ROUND3_PREREGISTRATION.md:10-11`):
    - aligned crop keeps **99.4%** of the light (minimum 98.4%)
    - centroid offset from frame centre: **3.4 ± 1.6 px** (max 7.8 px)
    - rms radius: **9.44 ± 0.13 px**
  - The crop is turned into a 1,024-cell pmf `p(kx,ky) = crop / sum(crop)`, encoded as **10 qubits**
    (5+5 bits, `code = (kx << 5) | ky`), with **all 1,024 Z-strings** as IQP generators (exact simulation).
- TODO / not fully verified: `data/README.md` and `papers/README.md` do not explicitly attribute
  `model_II.npz` to a specific paper. Most likely candidate: Alexander et al. 2020, *Deep Learning the
  Morphology of Dark Matter Substructure*, ApJ 893, 15 (arXiv:1909.07346) — `bib key alexander_2020_deep`,
  the paper that established the axion/CDM/no_sub 3-class DeepLense simulation convention. Confirm against
  the ML4SCI DeepLense GitHub repo before submission.

---

## 2. Correlation–Complexity Map on our data (arXiv 2603.06440 Fig. 5 reproduction)

Source: `docs/IQP_LENS_MAP_RESULTS.md`, table at line 29 (lens rows) and line 33-37 (D2 rows); numbers
recomputed 2026-09-07, single seed, 63 s on CPU (login node).

| dataset | representation | n | M | I_QCLI | I_CCI | i.i.d.-uniform null QCLI | null CCI | permutation-null QCLI | permutation-null CCI |
|---|---|---|---|---|---|---|---|---|---|
| lens | 18-bit (x,y,z) tuples, per-image | 18 | 4,096 | 0.4715 ± 0.0075 | 0.8677 ± 0.0078 | 0.0077 | 0.9980 | **0.4154** | **0.8782** |
| lens | 10-bit crop pmf, per-image (exact pmf) | 10 | pmf | 0.4731 ± 0.0693 | 0.4313 ± 0.0507 | 0.9944 | 0.0000 | **0.1782** | **0.9633** |

(`docs/IQP_LENS_MAP_RESULTS.md:29,31` — full table has 15 rows including MNIST/BinaryBlobs/pooled variants.)

- Lens 18-bit per-image I_QCLI = **0.4715**, I_CCI = **0.8677**; permutation null (shuffle pixel positions,
  keep intensity histogram) gives QCLI = 0.4154, CCI = 0.8782 — i.e. most of the "IQP-compatible" position
  is an artifact of the (x,y,z)-tuple encoding, not of lens spatial structure (only +0.056 QCLI / −0.011 CCI
  is attributable to real structure). `docs/IQP_LENS_MAP_RESULTS.md:16` (TL;DR point 1).
- Same finding for the paper's own turbulence D2: per-snapshot QCLI/CCI (0.091, 0.909) vs permutation null
  (0.098, 0.878) — `docs/IQP_LENS_MAP_RESULTS.md:17`. The paper's claim that turbulence sits in the
  "IQP-compatible" quadrant cannot distinguish the real flow field from the same velocity values with
  spatial structure fully shuffled.
- The 10-bit crop-pmf representation (the one actually used to train round 3) gives a genuinely different
  CCI (0.41–0.50) than the 18-bit tuple encoding (~0.87) — CCI is a property of the representation, not
  just the data — `docs/IQP_LENS_MAP_RESULTS.md:18-19`.
- Map position has **no predictive power** for generation quality: lens ranks above D2 on the map, but in
  the actual round-3 generation experiment IQP beats RBM/DCGAN on physics/PDF–JS and loses on Conv2D-MMD²
  — `docs/IQP_LENS_MAP_RESULTS.md:21,72,74` (table reproduced in §3 below).

---

## 3. Paper-style Sec. III table — IQP vs RBM vs DCGAN only (N = 11, 100)

Sources: `experiments/13_iqp_turbulence_replication/results_r3.jsonl` (fixed-basis arms + RBM + DCGAN),
`experiments/13_iqp_turbulence_replication/results_r3b.jsonl` (learned-affine-direction arms). Query used:
`python3 -c "import json; [print(json.loads(l)) for l in open(f) if json.loads(l).get('arm') in ARMS]"`.
Metrics: `pdf_js`, `conv_mmd2` (raw), `conv_mmd2_aligned` (intensity-aligned secondary block), `nn_holdout`.
Full definitions and the complete 14+11-arm tables (with physics/W1 quantities) are in
`docs/IQP_LENS_ROUND3_RESULTS.md` Table A (line ~64, N=100) and Table B (N=11, line ~121 region).

| Arm | params | N | PDF–JS ↓ | MMD² (raw) ↓ | MMD² (aligned) ↓ | NN-holdout ↓ |
|---|---|---|---|---|---|---|
| IQP-lowvar-joint-MMD-resample | 1,024 | 100 | 0.000988 | 0.10407 | 0.06168 | 4.682 |
| IQP-lowvar-joint-MMD-resample | 1,024 | 11 | 0.001332 | 0.14161 | 0.11409 | 4.567 |
| IQP-lowvar-joint-KL | 1,024 | 100 | 0.006710 | 0.15399 | 0.12816 | 3.625 |
| IQP-lowvar-joint-KL | 1,024 | 11 | 0.005543 | 0.25929 | 0.17331 | 3.554 |
| IQP-lowvar-paper-MMD (sequential Algorithm 1) | 1,024 | 100 | 0.006966 | 0.21401 | 0.22330 | 5.772 |
| IQP-lowvar-paper-MMD (sequential Algorithm 1) | 1,024 | 11 | 0.006091 | 0.20315 | 0.18366 | 4.931 |
| IQP-affine-M2-KL-resample | 3,272 (N=100) / 3,094 (N=11) | 100 | 0.003021 | 0.13562 | 0.08660 | 3.686 |
| IQP-affine-M2-KL-resample | 3,272 / 3,094 | 11 | 0.001748 | 0.16727 | 0.05586 | 3.664 |
| IQP-affine-M4-KL-resample | 5,520 (N=100) / 5,164 (N=11) | 100 | 0.005434 | 0.16193 | 0.13057 | 3.626 |
| IQP-affine-M4-KL-resample | 5,520 / 5,164 | 11 | 0.001829 | 0.16280 | 0.05634 | 3.735 |
| IQP-affine-M4-MMDflat-resample (hardware-compatible) | 5,520 / 5,164 | 100 | 0.002387 | 0.08607 | 0.02745 | 3.963 |
| IQP-affine-M4-MMDflat-resample | 5,520 / 5,164 | 11 | 0.001963 | 0.11339 | 0.02786 | 3.915 |
| IQP-affine-M4-auto3-resample (paper's kernel) | 5,520 / 5,164 | 100 | 0.000922 | 0.11888 | 0.08762 | 5.057 |
| IQP-affine-M4-auto3-resample | 5,520 / 5,164 | 11 | 0.000947 | 0.11681 | 0.06787 | 4.451 |
| RBM-10bit | 11,010 | 100 | 0.001952 | 0.07520 | 0.11438 | 5.225 |
| RBM-10bit | 11,010 | 11 | 0.006324 | 0.07410 | 0.12400 | 6.808 |
| DCGAN-32 | 199,837 | 100 | 0.007631 | 0.03711 | 0.06937 | 4.005 |
| DCGAN-32 | 199,837 | 11 | 0.001833 | 0.21647 | 0.12408 | 4.251 |

HOLDOUT split-half floor (N=100): PDF–JS 3.41e-05, MMD² 0.004006, NN-holdout 0.949 —
`experiments/13_iqp_turbulence_replication/results_r3.jsonl`, arm=`HOLDOUT split-half`.

Best-IQP-vs-baseline summary at N=100 (`docs/IQP_LENS_MAP_RESULTS.md:70-74`): IQP wins physics score
(0.513 vs RBM 0.727, DCGAN 0.861) and PDF–JS (0.00065 vs RBM 0.0020, DCGAN 0.0076); DCGAN wins
Conv2D-MMD² (0.0371 vs RBM 0.0752 vs best IQP 0.0861) — **mixed result, not a clean IQP win**, contrary
to the target paper's Table 2 pattern (§10 below).

Held-out projection corr (3b arms only, `results_r3b.jsonl` field `projection_corr_mean`/`_min`), N=100:
IQP-affine-M4-MMDflat-resample **0.968** (min 0.851), IQP-affine-M4-KL-resample 0.874 (min 0.693),
IQP-affine-M4-auto3-resample 0.794 (min 0.633), IQP-affine-M2-KL-resample 0.914 (min 0.792).
Reference: nearest-training-image corr ≈ 0.970, CTRL-pca4-resample = 0.952 —
`docs/IQP_LENS_ROUND3_RESULTS.md:88` (table row) and `:16` (TL;DR point 3, "0.968 已經等於...最近那張...0.970").

---

## 4. Data-scaling (`results_scaling.jsonl`, N ∈ {10,30,50,70,90,100}, job 356077, 3 min 28 s)

Three arms compared at matching N: DCGAN-32, RBM-10bit, IQP-affine-M4-MMDflat-resample.

| N | DCGAN PDF–JS | DCGAN MMD² (raw/aligned) | RBM PDF–JS | RBM MMD² (raw/aligned) | IQP PDF–JS | IQP MMD² (raw/aligned) |
|---|---|---|---|---|---|---|
| 10 | 0.004754 | 0.04403 / 0.04481 | 0.002319 | 0.07661 / 0.09468 | 0.001913 | 0.11976 / 0.02908 |
| 30 | 0.001932 | 0.11282 / 0.06411 | 0.005215 | 0.07627 / 0.11979 | 0.002006 | 0.08399 / 0.02271 |
| 50 | 0.008906 | 0.07533 / 0.09049 | 0.002879 | 0.08887 / 0.07554 | 0.001992 | 0.08262 / 0.02227 |
| 70 | 0.010722 | 0.08219 / 0.10657 | 0.002252 | 0.06571 / 0.09449 | 0.001841 | 0.08697 / 0.02536 |
| 90 | 0.011309 | 0.04173 / 0.09729 | 0.003924 | 0.07269 / 0.12710 | 0.002305 | 0.08217 / 0.02178 |
| 100 | 0.007631 | 0.03711 / 0.06937 | 0.001952 | 0.07520 / 0.11438 | 0.002387 | 0.08607 / 0.02745 |

Reading (`docs/IQP_LENS_PAPER_SUITE_RESULTS.md:12`): IQP's raw MMD² is flat for N ≥ 30 (0.084–0.087); RBM
is also flat (0.066–0.089); DCGAN's single-seed MMD² swings 3× (0.113 → 0.037) non-monotonically — no
clean "DCGAN catches up as data grows" crossing like the target paper's Table 2 pattern
(`docs/IQP_LENS_PAPER_SUITE_RESULTS.md:201`, Q4: "half right" — flat-IQP confirmed, DCGAN-overtakes-at-30–50
not confirmed, DCGAN already leads at N=10).

---

## 5. Mechanism tests (why fixed-basis IQP fails and learned affine directions work)

Sources: `experiments/13_iqp_turbulence_replication/mech_tests/mech_test/results.json`,
`mech_tests/mech_test2/results.json`, `mech_tests/mech_test2/diag2.json`;
narrative in `docs/IQP_LENS_ROUND3_PREREGISTRATION.md:99-110` (mechanism tests) and `:163-170` (M-sweep).

- **Single-image fit (round 3b preamble, mech_test/results.json `a.fits`)**: a single aligned crop is fit
  almost exactly by a full 1,024-phase 10-qubit IQP under every objective: KL corr **0.9999**, MMD-flat
  corr **0.99997**, paper's auto3-kernel MMD corr **0.9962** — representability and kernel choice are not
  the bottleneck (`docs/IQP_LENS_ROUND3_PREREGISTRATION.md:100-101`).
- **Fixed-basis interpolation collapses**: interpolating between two images along fixed-basis latents
  (paper's Hamming block, or ancilla mixing angles) collapses mid-path except for short paths, and even
  then gives a pixel cross-fade, not a geometric morph; an ancilla marginal is an equal-weight mixture of
  holograms (verified to 1e-18) — ancillas add nothing (`docs/IQP_LENS_ROUND3_PREREGISTRATION.md:102-105`).
- **Learned affine directions (θ = θ⁰ + Φᵀz) generalise**: M=4 held-out projection corr **0.911**
  (`mech_tests/mech_test2/diag2.json`, key `holdout_random_vs_trained."trained Phi + trained theta0".corr_mean`
  = 0.9113), vs **0.672** for a random-direction Φ (same key, "random Phi(seed 99) + trained theta0") and
  **0.671** for no latent at all ("no latent (theta0 only)") — the gain is from *learning* the directions,
  not from having extra parameters (`docs/IQP_LENS_ROUND3_PREREGISTRATION.md:106,169-170`).
- **M sweep** (`docs/IQP_LENS_ROUND3_PREREGISTRATION.md:168-170`): M=2 generates the best samples (100% of
  Gaussian draws are ring-shaped, radius 9.40 ± 0.21 vs data 9.21 ± 0.15, as far from the training set as
  real held-out images, but under-dispersed: e1 sd 0.048 vs data 0.099); M=4 projects best (corr 0.911);
  M≥6 collapses because ‖z‖ exceeds the model's coherence radius (≈0.03–0.05).
- **Round-2 diagnostic (paper's own sequential Algorithm 1 protocol)**: every training image except the
  first reconstructs as pure noise, corr **0.47**, because only 50 latent parameters are free against a
  core fitted to a single anchor image (`docs/IQP_LENS_ROUND3_PREREGISTRATION.md:107`). This motivated
  round 3's joint-training protocol.

---

## 6. Round 3b paired projection-corr table (`results_r3b.jsonl`, N=100, mean/min)

| Arm | proj corr mean | proj corr min |
|---|---|---|
| IQP-affine-M2-KL-resample | 0.9140 | 0.7923 |
| IQP-affine-M4-KL-resample | 0.8741 | 0.6932 |
| IQP-affine-M4-MMDflat-resample | **0.9684** | 0.8514 |
| IQP-affine-M4-auto3-resample | 0.7940 | 0.6326 |
| CTRL-pca4-resample (classical linear analogue) | 0.952 | — (`docs/IQP_LENS_ROUND3_RESULTS.md:88` region) |

Reading: hardware-compatible MMD-flat kernel generalises **best** (M=4: 0.968 mean), the paper's own auto3
kernel generalises **worst** (0.794) — `docs/IQP_LENS_ROUND3_RESULTS.md:15` (TL;DR point 4).

---

## 7. Fig. 4 (QCLI-maximization → emergent CCI), `outputs_controlled/results_fig4.jsonl` (60,016 rows)

Setup: 8Q/12Q/16Q IQP circuits (2-local, 150 requested gates), optimized to **maximize QCLI only** (no CCI
term in the objective); 3 seeds per n, 20,000 steps logged. i.i.d.-uniform calibration QCLI (upper
reference, "≈0.98 for uniform"): n=8 → **0.9816**, n=12 → **0.9984**, n=16 → **0.9999**
(`results_fig4.jsonl`, `record="calibration"` rows).

Mean CCI by QCLI bin, pooled over all logged steps and seeds (computed from the raw jsonl, not a doc —
reproduce with the binning script used for this sheet, see note below):

| QCLI bin | n=8 mean CCI | n=12 mean CCI | n=16 mean CCI |
|---|---|---|---|
| [0.0, 0.2) | 0.622 | 0.647 | 0.749 |
| [0.2, 0.4) | 0.894 | 0.871 | 0.915 |
| [0.4, 0.6) | 0.908 | 0.797 | 0.868 |
| [0.6, 0.8) | 0.827 | 0.602 | 0.505 |
| [0.8, 1.0] | 0.995 | 0.498 | 0.277 |

Note on reading this table: the emergent-coupling claim (QCLI↑ ⟹ CCI↑, `docs/IQP_LITERATURE_SURVEY.md`
§3.2 / target paper Fig. 4 narrative) holds cleanly for the **low-to-mid QCLI regime** (bins [0,0.6)) at
all three qubit counts; at high QCLI (bin [0.8,1.0], approaching the uniform-null ceiling) CCI instead
*falls* for n=12 and n=16 as circuits saturate toward uniform (I_CCI→0 for exact uniform, per the n=8/12/16
calibration rows) — this saturation-regime reversal is not mentioned in the target paper's Fig. 4
discussion and is worth flagging as a caveat if this figure is used. Figure file:
`figures/iqp_fig4_qcli_cci.png`.

---

## 8. Turbulence replication (arXiv 2603.06440 D2, one-sentence TL;DR)

Source: `docs/IQP_TURBULENCE_REPLICATION_RESULTS.md:10-21` (TL;DR block), job 342220 (main, 55 arms, 0
failures, 31m55s) + job 342454/342514/342542/342685 (post-hoc analysis).

**Not reproduced** under the pre-registered evaluation convention: at the 11-snapshot regime, RBM and
DCGAN-1.2M beat our IQP on *both* metrics (PDF–JS, Conv2D-MMD²), triggering the pre-registration's
"not-reproduced" clause (`docs/IQP_TURBULENCE_REPLICATION_RESULTS.md:12`). However, under a *last-write-wins*
sample-aggregation convention (never specified by the paper), the paper's own IQP PDF–JS numbers **do**
match (0.0322 ± 0.0006 / 0.0476 ± 0.0004 vs. paper 0.0238 / 0.0221 — close but this doesn't flip the
not-reproduced verdict, since RBM still wins under the same convention) — `:13`. The deeper problem: our
IQP only fits its own training anchors to corr 0.69 (sampled field) / 0.77 (exact field), and root-cause
analysis (`docs/IQP_TURBULENCE_REPLICATION_RESULTS.md` §3(d)) attributes this gap to the paper's own
protocol (frozen θ_lat in stage 1, ≈ −0.08; 5×10⁴-sample pmf, ≈ −0.04) plus a ≈0.88–0.89 ceiling of the
model family + closed-form MMD objective — not to our implementation. Conv2D-MMD² is saturated/uninformative
across the whole comparison (ground truth shuffled in pixel space already scores 0.704, BELOW every model
in the paper's Table 2, whose column spans 0.896–1.472 — corrected 2026-09-08, the earlier wording
"inside the range of every model" was wrong; source TURB:20 and :477-478, PAPER §5.1). Held-out evaluation in the original paper's design is solvable by zero-parameter
interpolation (corr 0.9998), so its core "latent adaptation" claim is never actually exercised by its own
metrics (`docs/IQP_TURBULENCE_REPLICATION_RESULTS.md:16`, TL;DR point 4).

---

## 9. Compute

- **Exact simulation method**: fast Walsh–Hadamard transform (FWHT), no Monte Carlo, no sampling noise —
  `experiments/13_iqp_turbulence_replication/DESIGN_exact_iqp.md:1-6`.
- **Per-step timing, n=18 (turbulence)**: GPU **11 ms/step**, ≈2× faster than 4 CPU threads
  (`docs/IQP_TURBULENCE_REPLICATION_PREREGISTRATION.md:133`, confirmed in
  `docs/IQP_TURBULENCE_REPLICATION_RESULTS.md:61`: "IQP 每個 arm 約 10.4–11.5 分鐘...11 ms/step").
- **Per-step timing, n=10 (lens)**: **not found in local docs** — TODO, needs a fresh timing run before
  citing a number in the paper. Do not use "~4 ms" without re-measuring; that figure could not be traced
  to any file in this repo during this pass.
- **Job durations** (all single-seed unless noted):
  - Round 3 (lens, 14 arms × 2 N = 28 jobs): job 349702, dev partition, **46 min 02 s**, 0 failures.
  - Round 3b (lens, 11 arms × 2 N = 22 jobs): job 349854, dev partition, **3 min 19 s**, 0 failures.
  - Data-scaling (lens): job 356077, dev partition, **3 min 28 s**, 0 failures.
  - Correlation–complexity map recompute: login node, CPU, **63 s**, no job number (not sbatch'd).
  - Turbulence main replication: job 342220, 8gpus (8 GPU), **31m55s**, 55 arms, 0 failures.
- **Single seed** throughout (seed 0) for all lens and turbulence generative experiments; the RBM
  last-write-wins re-sampling in §3(c) of the turbulence results is explicitly flagged as non-deterministic
  (PCD sampling on GPU) — `docs/IQP_TURBULENCE_REPLICATION_RESULTS.md:178`.

---

## 10. Target paper's own numbers (arXiv 2603.06440 Table 2, for side-by-side reference)

Source: `docs/PAPER_2603.06440_SUMMARY.md` §5.1 (verbatim table, cross-checked against `paper.txt:1400-1463`,
"逐格核對結果：與 Leo 提供的數字完全一致，無差異").

| Model (Params.)(# training data) | PDF–JS ↓ | MMD² (Conv2D) ↓ |
|---|---|---|
| RBM (12k)(11) | 0.168770 | 1.472481 |
| **IQP (31k)(11)** | 0.023784 | 0.952420 |
| **IQP (63k)(11)** | 0.022084† | 0.942267† |
| GAN (51k)(100) | 0.135584 | 1.327760 |
| GAN (0.2M)(100) | 0.027695 | 0.895912 |
| GAN (1.2M)(100) | 0.012204 | 0.964874 |
| GAN (51k)(11) | 0.366758 | 1.397969 |
| GAN (0.2M)(11) | 0.328273 | 1.407441 |
| GAN (1.2M)(11) | 0.244852 | 1.446292 |

Note (`docs/PAPER_2603.06440_SUMMARY.md` §5.1): the † marks are unexplained in the paper's own text/HTML —
no footnote was found; likely (not confirmed) marks "best in column". Do not present our numbers next to
this table without noting the absolute-scale caveat: **our lens PDF–JS/MMD² values are not on the same
scale as the paper's turbulence values** (different dataset, different image size/channel structure) — any
side-by-side figure should compare *rankings* (IQP vs RBM vs DCGAN, which metric each wins), not absolute
magnitudes across the two datasets.

---

## Open TODOs / items needing verification before submission

1. **DeepLense/model_II.npz dataset citation** (§1): candidate is Alexander et al. 2020 (arXiv:1909.07346),
   not confirmed against the ML4SCI DeepLense repo or data/README.md provenance.
2. **n=10 exact-sim per-step timing** (§9): no number found in any local doc; needs a fresh measurement.
3. **Fig. 3 (support-mismatch) is explicitly v1, awaiting v2** — `docs/IQP_LENS_PAPER_SUITE_RESULTS.md:54`:
   "狀態：v1，待 v2 覆蓋。第一版（job 356097）的目標多樣性建構失敗...v2 正在重跑". The copied figure
   `figures/iqp_fig3_support_mismatch.png` is the v1 (broken/uninformative) version — **do not use it in
   the paper as-is**; re-copy once v2 lands in `docs/figures/`.
4. Several `refs.bib` entries have incomplete author lists (marked `note = {TODO: ...}` in the .bib file
   itself) — these come from the literature survey table, which only recorded "first author et al." for
   some 2026 preprints. Full author lists should be re-verified via WebFetch/arXiv before camera-ready.
5. The Fig. 4 CCI-vs-QCLI-bin table in §7 is a fresh aggregation over the raw jsonl (not copied from any
   existing doc) — worth an independent re-check of the binning script before quoting the saturation-regime
   reversal finding in the paper, since it is not corroborated by any existing markdown analysis.

## Addendum (2026-09-08, coordinator) — per-step timings not found in docs, taken from agent verification reports
- n = 10 (1,024 amplitudes), joint affine-latent model, CPU 4 threads, float64: 4.0 ms/step (N = 3, round-3b pre-submission verifier), 4.17 ms/step (N = 11, M = 4, KL) and 18.6 ms/step (N = 100) per the round-3b implementer's measurements; a full round-3b arm (4,000 steps + held-out inference) took ≈ 2.6–3.9 min on one H100.
- n = 16 (65,536 amplitudes), GPU probe inside job 356097: Adam MMD step with a batch of 70 targets = 14.5 ms (0.21 ms per target-step); SPSA step (batch 3) 4.6 ms.
- n = 18 (262,144 amplitudes), GPU: 11 ms/step (turbulence replication, docs/IQP_TURBULENCE_REPLICATION_PREREGISTRATION.md).
- Wall-clock of the lens jobs: round 3 (28 arms) 46 min on 8 H100; round 3b (22 arms) 3 min 19 s; scaling (15 arms) 3 min 28 s; Fig. 3/4 controlled study 10.5 min on 1 GPU.
- DeepLense Model II dataset attribution: to be confirmed by Leo (candidate: Alexander et al. 2020, arXiv:1909.07346, ApJ 893:15; data distributed via the ML4SCI DeepLense project).

## Addendum 2026-09-08: Fig. 3 v2 (controlled support-mismatch study) — COMPLETE
- Source: experiments/13_iqp_turbulence_replication/outputs_controlled/results_fig3_v2.jsonl (562 records), done marker outputs_controlled/fig3_v2.done (2026-09-08 07:07:27, wall 897 s, SLURM job 356221, dev partition).
- Targets: 140 IQP target circuits at n=16, ≤4-local generators, obtained as Adam snapshots along exact-QCLI ascent trajectories with G_target ∈ {40,140,280,420,560,700,840,1050}; exact I_QCLI of targets spans 0.000–0.408 (reachable ceiling ≈0.41 at n=16 ≤4-local; paper's Fig. 3 plots targets up to ≈0.7, which we could not reach); sample-QCLI of the same targets spans 0.000–0.064 (below the finite-sample null 0.0696), so the sample axis is uninformative.
- Learners: IQP students with G_n ∈ {50,100,150} random generators (all three learners cover 140/140 targets — 420 `record="fit"` rows, 140 per learner; the earlier "G_n=150 has 136/140 targets" was a misreading of `learner_gates_effective` = 136, i.e. the number of distinct generators left after dedup for the 150-gate learner; corrected 2026-09-08 from outputs_controlled/results_fig3_v2.jsonl), trained by MMD; achieved MMD² recorded per (target, learner).
- Qualitative finding: achieved MMD² increases with target exact I_QCLI for all three learner sizes (upper envelopes rise from ≈2e-3 to ≈1.5e-2 for G_n=50, ≈2e-3→5e-3 for G_n=100, ≈1.5e-3→3e-3 for G_n=150); larger learners fit better at every QCLI. This reproduces the paper's Fig. 3 trend within I_QCLI ≤ 0.41 only.
- Figure: docs/figures/iqp_fig3_support_mismatch.png (v2, overwrote v1); paper copy figures/iqp_fig3_support_mismatch.png; vector version pending as figures/figA_fig3.pdf.
