# Facts sheet for the ML4PS 2026 writeup

Every number below carries its provenance (`file:line` or a jsonl field query). Do not retype numbers
from memory when drafting the paper — copy from here or re-derive with the query shown. All experiments
are single-seed (seed 0) unless stated otherwise. Paths are relative to
`<HOME>/gsoc2026-deeplense-qml/` unless given absolute.

Framing agreed with <user> (see memory `ml4ps2026-paper.md`): paper-style comparison IQP vs RBM vs DCGAN
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
"逐格核對結果：與 <user> 提供的數字完全一致，無差異").

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
- DeepLense Model II dataset attribution: to be confirmed by <user> (candidate: Alexander et al. 2020, arXiv:1909.07346, ApJ 893:15; data distributed via the ML4SCI DeepLense project).

## Addendum 2026-09-08: Fig. 3 v2 (controlled support-mismatch study) — COMPLETE
- Source: experiments/13_iqp_turbulence_replication/outputs_controlled/results_fig3_v2.jsonl (562 records), done marker outputs_controlled/fig3_v2.done (2026-09-08 07:07:27, wall 897 s, SLURM job 356221, dev partition).
- Targets: 140 IQP target circuits at n=16, ≤4-local generators, obtained as Adam snapshots along exact-QCLI ascent trajectories with G_target ∈ {40,140,280,420,560,700,840,1050}; exact I_QCLI of targets spans 0.000–0.408 (reachable ceiling ≈0.41 at n=16 ≤4-local; paper's Fig. 3 plots targets up to ≈0.7, which we could not reach); sample-QCLI of the same targets spans 0.000–0.064 (below the finite-sample null 0.0696), so the sample axis is uninformative.
- Learners: IQP students with G_n ∈ {50,100,150} random generators (all three learners cover 140/140 targets — 420 `record="fit"` rows, 140 per learner; the earlier "G_n=150 has 136/140 targets" was a misreading of `learner_gates_effective` = 136, i.e. the number of distinct generators left after dedup for the 150-gate learner; corrected 2026-09-08 from outputs_controlled/results_fig3_v2.jsonl), trained by MMD; achieved MMD² recorded per (target, learner).
- Qualitative finding: achieved MMD² increases with target exact I_QCLI for all three learner sizes (upper envelopes rise from ≈2e-3 to ≈1.5e-2 for G_n=50, ≈2e-3→5e-3 for G_n=100, ≈1.5e-3→3e-3 for G_n=150); larger learners fit better at every QCLI. This reproduces the paper's Fig. 3 trend within I_QCLI ≤ 0.41 only.
- Figure: docs/figures/iqp_fig3_support_mismatch.png (v2, overwrote v1); paper copy figures/iqp_fig3_support_mismatch.png; vector version pending as figures/figA_fig3.pdf.

---

## Addendum: per-class generation (2026-09-08) — appendix material, COMPLETE

**Question.** Every result above pools the three dark-matter substructure classes. Does the
paper's best arm train and generate for **each class on its own**?

**Setup.** The arm is unchanged — `IQP-affine`, M = 4, MMD-flat objective, resample prior,
seed 0, lr 0.02, 4 000 joint steps, 1 000 θ₀ seed steps, 8 × 1 000-step projection restarts at
`infer-lr 0.005`, 200 generated frames, 200 held-out. The **only** change is which images it
sees: for class *c* it trains on 100 images of *c* and is scored against 200 held-out images of
*c*. Same codec (`codec10`), same metric functions (`evaluate_r3` / `evaluate_r3b`), same
`CONV_SEED`.

**Provenance.** SLURM job **358382**, partition `dev`, node `<node>`, account <YOUR_ACCOUNT>,
`/usr/bin/python3`, elapsed **00:07:24**, `COMPLETED`, exit 0:0, `fail=0`, 6/6 run dirs.
- submit script: `experiments/13_iqp_turbulence_replication/submit_lens_class.sbatch`
- log: `experiments/13_iqp_turbulence_replication/logs_lens/sbatch-class-358382.out`
- outputs: `experiments/13_iqp_turbulence_replication/outputs_lens_class/<class>_m4_mmdflat_resample_n{100,11}/`
- metrics: `experiments/13_iqp_turbulence_replication/results_class.jsonl` (18 rows)
- new scripts: `data_lens_class.py`, `train_iqp_affine_class.py`, `evaluate_class.py`, `test_class.py`
- figure: `paper/ml4ps2026/figures/figB_classes.pdf` (5.21 × 2.10 in) + `.png`

**Per-class split (the one deliberate difference from the pooled runs).** The pooled N = 100
holds only ≈ 33 images of each class, so a per-class N = 100 cannot be a subset of it. The same
"one permutation, take the prefix" rule is applied *within* a class, with the class index
appended to the split seed: train = `default_rng([0,3,0,c]).permutation(train indices of c)[:100]`,
latent pool = the next 2 000, held-out = `default_rng([0,3,1,c]).permutation(val indices of c)[:200]`.
Class pools are large enough (train 26 942 / 26 782 / 26 470; val 2 954 / 2 977 / 2 979 for
axion / cdm / no_sub). Purity and train-vs-latent disjointness are asserted at run time and
logged: all three report `pure(train,val)=(True,True) disjoint=True`
(`logs_lens/sbatch-class-358382.out`). Each arm uses its own class's flux scale
S (axion 127.19, cdm 123.30, no_sub 121.54; pooled 124.92).

### N = 100 (the working point of the paper's main table)

| training set | PDF–JS ↓ | MMD² ↓ | MMD²(flux-aligned) ↓ | held-out projection corr ↑ | NN-holdout |
|---|---|---|---|---|---|
| axion only | 0.00205 | 0.09012 | 0.02346 | 0.9720 | 3.851 |
| cdm only | 0.002275 | 0.09783 | 0.01848 | 0.9686 | 4.111 |
| no_sub only | 0.002001 | 0.10340 | 0.02773 | 0.9703 | 3.780 |
| **pooled (main paper)** | 0.002387 | 0.08607 | 0.02745 | 0.9684 | 3.963 |
| _axion held-out split-half (floor)_ | 7.633e-05 | 0.005915 | 0.006399 | — | 0.8186 |
| _cdm held-out split-half (floor)_ | 9.869e-05 | 0.01794 | 0.01871 | — | 1.019 |
| _no_sub held-out split-half (floor)_ | 0.0001097 | 0.01603 | 0.01823 | — | 0.7795 |

### N = 11

| training set | PDF–JS ↓ | MMD² ↓ | MMD²(flux-aligned) ↓ | held-out projection corr ↑ | NN-holdout |
|---|---|---|---|---|---|
| axion only | 0.002197 | 0.1015 | 0.02636 | 0.9232 | 3.874 |
| cdm only | 0.002174 | 0.1108 | 0.02031 | 0.9238 | 3.462 |
| no_sub only | 0.001102 | 0.08096 | 0.02379 | 0.8998 | 3.994 |
| **pooled (main paper)** | 0.001963 | 0.11340 | 0.02786 | 0.9092 | 3.915 |

Pooled reference rows are the `IQP-affine-M4-MMDflat-resample` rows of `results_r3b.jsonl` at
the matching N — the same arm, scored the same way, against the *pooled* 200 held-out images.

**What this supports (and what it does not).**
- *Supported:* each class trains and generates on its own at the pooled arm's working point.
  At N = 100 the three per-class projection correlations (0.9686–0.9720) bracket the pooled
  0.9684, and the three PDF–JS values (0.0020–0.0023) bracket the pooled 0.002387. Nothing
  degrades when the training set is restricted to one class.
- *Not supported:* any claim that one class is easier or harder. The per-class spread is
  ≤ 0.004 in projection corr and ≤ 0.0003 in PDF–JS — inside the noise band this project
  treats as uninformative — and each class is scored against a *different* held-out set with
  its own split-half floor (MMD² floors differ by 3× across classes: 0.0059 / 0.0179 / 0.0160),
  so the raw per-class MMD² values are not comparable to each other.
- *Expected, and observed:* the generated samples look alike across classes. The class signal
  lives from PC6 onward and accounts for only **3.0 %** of the pixel variance
  (`docs/LENS_IMAGE_CHARACTERISTICS.md:195`); nearest-centroid classification on raw pixels is
  38.3 % against a 33.3 % chance baseline (`docs/LENS_IMAGE_CHARACTERISTICS.md:205`). A
  pixel-reconstruction objective is not expected to reproduce it, and this run does not claim to.

---

# Addendum: baseline convergence sweep (2026-09-09, job 360684)

**The question.** Table 2 (`tab:main`) trains both classical baselines for a fixed 20,000 steps.
At that point the RBM's reconstruction MSE is still falling (0.0662 at 20k against 0.0041 at
160k for N = 100) and the DCGAN's losses are still oscillating, so a reader is entitled to ask
whether the baselines were simply stopped too early and whether a longer run would overtake the
IQP arms.

**What was run.** Each baseline was re-trained for longer and scored at several step counts with
the *unmodified* `evaluate_r3b.py` — same 200 held-out images, same seed-0 Conv2D encoder, same
flux-aligned addendum — so every number below is directly comparable to a Table 2 cell.

| arm | N | step counts scored | points |
|---|---|---|---|
| `RBM-10bit` | 11, 100 | 20k / 40k / 80k / 160k | 8 |
| `DCGAN-32` | 11, 100 | every 5k from 5k to 60k | 24 |

Each (arm, N) is **one** training run with snapshots, not one run per step count: both training
loops are sequential with a single RNG stream and no step-count-dependent schedule, so the state
after step k is what a standalone `--steps k` run ends with. This was verified before any real
training, on CPU, by byte-comparing a sweep snapshot against a standalone run of the same length
— **bit-identical for both models** (`logs_lens/sbatch-conv-360684.out:5-6`).

Scripts: `baseline_rbm_sweep.py`, `baseline_dcgan_sweep.py`, `aggregate_convergence.py`,
`submit_baseline_convergence.sbatch`. Data: `results_convergence.jsonl` (32 rows),
`outputs_convergence/` (32 run directories), evaluation table
`logs_lens/evaluate_convergence.log`, best-of-steps table `logs_lens/aggregate_convergence.log`.

**Reading rule (deliberately generous to the baselines).** "Best over steps" below picks, per
metric, the step count that scores best *on the held-out set used for scoring*. That is selection
on the evaluation set, not on a validation split, and no IQP arm in Table 2 is given it: every
IQP number there comes from one pre-registered step count. The comparison is therefore an upper
bound on what the baselines can do, not a like-for-like protocol.

## Does a longer run change Table 2?

Best IQP entry per column is recomputed from `results_r3.jsonl` / `results_r3b.jsonl` by
`aggregate_convergence.iqp_reference`, which asserts it still matches the published table.

| arm | N | metric | Table 2 (20k) | sweep @ 20k | best over steps | @ steps | best IQP | verdict |
|---|---|---|---|---|---|---|---|---|
| RBM | 11 | PDF–JS | 0.00632 | 0.006324 | 0.001274 | 160k | **0.00095** | IQP still lower |
| RBM | 11 | MMD² | **0.0741** | 0.074099 | 0.051279 | 160k | 0.1134 | baseline lower (already was) |
| RBM | 11 | MMD²al | 0.1240 | 0.12400 | 0.053769 | 160k | **0.0279** | IQP still lower |
| RBM | 100 | PDF–JS | 0.00195 | 0.001952 | 0.001245 | 80k | **0.00092** | IQP still lower |
| RBM | 100 | MMD² | 0.0752 | 0.075195 | 0.053830 | 80k | 0.0861 | baseline lower (was 0.0752 vs 0.0861) |
| RBM | 100 | MMD²al | 0.1144 | 0.114380 | 0.075711 | 80k | **0.0275** | IQP still lower |
| DCGAN | 11 | PDF–JS | 0.00183 | 0.002638 | 0.001600 | 30k | **0.00095** | IQP still lower |
| DCGAN | 11 | MMD² | 0.2165 | 0.276110 | 0.047555 | 55k | 0.1134 | **now lower than IQP** |
| DCGAN | 11 | MMD²al | 0.1241 | 0.100100 | 0.031707 | 30k | **0.0279** | IQP still lower, margin 4.5× → 1.14× |
| DCGAN | 100 | PDF–JS | 0.00763 | 0.009577 | **0.000436** | 5k | 0.00092 | **BOLD FLIPS to DCGAN** |
| DCGAN | 100 | MMD² | **0.0371** | 0.060113 | 0.021604 | 5k | 0.0861 | baseline lower (already was) |
| DCGAN | 100 | MMD²al | 0.0694 | 0.090767 | **0.017574** | 5k | 0.0275 | **BOLD FLIPS to DCGAN** |

Secondary columns (not in Table 2): best NN-holdout is RBM 4.342 @ 40k (N = 11) / 5.211 @ 80k
(N = 100) and DCGAN 3.482 @ 35k (N = 11) / 3.950 @ 20k (N = 100).

**Answer to the question as asked: no. Longer training does not let either baseline win a column
it was losing.** Every RBM column and every N = 11 column that IQP held at 20k, IQP still holds
at the baseline's best step count. The RBM's improvement is large in absolute terms (PDF–JS 5.0×
better at N = 11, 1.6× at N = 100) and still leaves it above the best IQP PDF–JS and MMD²al at
both N.

**But a step-count sweep does flip two of the six bold entries — via a *shorter* run, not a
longer one.** The DCGAN at N = 100 is at its best after **5k** steps, a quarter of the Table 2
budget: PDF–JS 0.000436 against the best IQP 0.00092, and MMD²al 0.01757 against 0.0275. Its
flux is also nearly exact there (mean total 124.82 against the held-out 123.87). The one N = 11
column that changes hands does so between two classical baselines (MMD², RBM 0.0741 → DCGAN
0.0476), a column IQP never held.

## Three caveats that limit how far the DCGAN result can be pushed

1. **The DCGAN is not reproducible run-to-run on this GPU, and the sweep's own 20k row does not
   reproduce Table 2's.** cuDNN's transposed-convolution backward accumulates with atomics.
   Running the *unmodified* `baseline_dcgan_r3.py` twice with identical arguments on the same GPU
   gives per-pixel max |Δ| = 0.855 and Δ lossG = 2.48 on a 2,000-step smoke; the sweep differs
   from a standalone run by less than that (0.506 / 0.564)
   (`logs_lens/sbatch-conv-360684.out:7-9`). Consequently the sweep's DCGAN @ 20k
   (N = 100: PDF–JS 0.00958, MMD² 0.0601, MMD²al 0.0908) is *not* Table 2's DCGAN @ 20k
   (0.00763 / 0.0371 / 0.0694), and a best-of-12-steps selection over a non-monotone noisy curve
   is partly a maximum over run-to-run noise. The **RBM has no such operator and reproduces
   Table 2's 20k row exactly** at both N (0.006324 / 0.074099 / 0.12400 and 0.001952 / 0.075195 /
   0.114380), which is what makes the RBM half of this sweep a clean measurement.
2. **The DCGAN at N = 100 collapses at ~30k.** From step 30,000 to 60,000 `lossD` sits at the
   BCE clamp (100.0) and `lossG` is exactly 0.0: the discriminator is saturated and the generator
   receives no gradient. The flat N = 100 plateau in `figA_convergence` panel (b) beyond 30k is
   a dead model, not a converged one.
3. **The RBM's late gains come partly from over-dispersion.** Its NN-within rises from 1.022 to
   3.335 (N = 11) and 1.674 to 5.523 (N = 100) between 20k and 160k, against a held-out NN-within
   of 0.949. By 160k the RBM's samples are 3.5–5.8× more spread out than the real held-out set,
   so part of the PDF–JS improvement is added noise rather than better structure. At N = 100 the
   curve also turns back up after 80k (PDF–JS 0.001245 → 0.001628, MMD²al 0.0757 → 0.1242).

## Suggested final step counts

* **RBM: 80k.** It is the best step on all three Table 2 metrics at N = 100 and within noise of
  160k at N = 11 (PDF–JS 0.001245 vs 0.001274), it is where the N = 100 curve turns around, and
  it avoids the worst of the over-dispersion. If one number per N is acceptable, 160k for N = 11
  and 80k for N = 100. 20k is defensibly *too short* for this arm and saying so is honest.
* **DCGAN: no single step count is defensible.** Its curve is non-monotone, its N = 100 run is
  dead after 30k, and its run-to-run spread on this GPU is comparable to the gaps being measured.
  Keeping the pre-registered 20k is as defensible as any other choice; the honest statement is
  that this arm's score is set by where training is stopped and by run-to-run noise, and that
  both are larger than the DCGAN-vs-IQP differences quoted in Table 2.

---

## Addendum: metric protocol v11 (2026-09-09)

Code: `experiments/13_iqp_turbulence_replication/metrics_v11.py` (definitions and floor) and
`evaluate_v11.py` (re-scoring CLI, `--outputs-dir ... --results ... [--floor]`). Both are new
files; `metrics.py`, `evaluate_r3.py`, `evaluate_r3b.py`, `evaluate_class.py` and every existing
`results_*.jsonl` are untouched, so every v10 number in this document stays reproducible.
Seed-0 re-scoring of all 106 existing runs: `results_v11_seed0.jsonl` (job 360844, dev, 1 min 26 s).
Floors: `results_v11_floor.jsonl` (200 vs 200, pooled and per class), `results_v11_floor_disjoint.jsonl`
(pooled, mutually exclusive draws), `results_v11_floor_n100.jsonl` (same rule at n = 100, job 360850).

### What the v10 protocol let a generator do to its own yardstick

1. **PDF–JS bins** (`metrics.py:32-36`). The 50 bins span the joint min/max of real union generated,
   so an arm whose brightest pixel sits at 2.7 (several IQP arms reach 1.4 to 2.8 while the data live
   in [0, 1]) is scored on a coarser histogram than an arm that stays inside [0, 1]. Setting one pixel
   of the DCGAN output to 4.0 divides its PDF–JS by about 13.
2. **Conv2D encoder input scale** (`metrics.py:94-95`). The same joint min/max normalises the encoder
   input, so each arm enters the feature extractor on its own scale.
3. **Conv2D kernel bandwidth** (`metrics.py:112-116`). Sigma comes from the median heuristic on the
   pooled (real plus generated) features, so every arm is scored with a different Gaussian kernel.
4. **Floor sample size** (`evaluate_r3.holdout_reference`). The floor row is 100 vs 100 while every
   model row is 200 vs 200, and the biased MMD² estimator's bias scales like 1/n.

### v11 definitions (what Setup should cite)

* **PDF–JS.** Both sets clipped to the data range [0, 1] (Model II images are per-image min-max
  normalised, so 0 and 1 are the data's exact extremes), 50 equal-width bins fixed on [0, 1],
  eps-smoothed, JS in bits. The bin grid depends on neither input. The generated mass the clip moves
  is reported per run (`gen_clip_frac`, `gen_max`) instead of silently widening the bins.
  For the flux-aligned column the range comes from the aligned held-out set (`ref_lo_aligned`,
  `ref_hi_aligned`), which is the same rule with the reference as the only source.
* **Conv2D-MMD².** Same fixed random encoder as v10 (three stride-2 conv blocks 1→16→32→64 plus
  global average pooling, `CONV_SEED = 0`). The encoder input range is the held-out set's min/max
  and generated pixels outside it are clipped; the Gaussian bandwidth is the median heuristic
  computed once on the held-out features and shared by every arm, every N, every class and every
  floor draw (pooled sigma 0.005547 raw, 0.002558 flux-aligned; the aligned reference range is [0, 1.662]). N = M = 200.
  Both estimators are stored: `conv_mmd2` (biased V-statistic, the one the main table quotes, so
  the column stays comparable with v10) and `conv_mmd2u` (unbiased U-statistic, both diagonals
  removed), each with a flux-aligned twin.
* **Floor.** 20 draws of 200 real vs 200 real from the validation images left over by the held-out
  split (perm[200:] of the permutation whose prefix perm[:200] is the held-out set, per class
  likewise), scored against the held-out set's encoder context like every model row, reported as
  mean ± sd. Pooled draws are also available mutually exclusive (`--draw-mode disjoint`, 20 x 400
  distinct images); a single class's leftover pool (~2,770) cannot supply 8,000, so the per-class
  floors use an independent permutation per draw (A and B disjoint within a draw).

### Floor, v10 against v11

| kind | v10 (100 v 100, one split) | v11 n = 100 (20 draws) | v11 n = 200 (20 draws) | v11 n = 200 unbiased |
|---|---|---|---|---|
| pooled PDF–JS | 3.41e-05 | 7.50e-05 ± 3.6e-05 | 4.08e-05 ± 2.1e-05 | n/a |
| pooled MMD² | 0.004006 | 0.00931 ± 0.0057 | 0.00481 ± 0.0028 | 0.00033 ± 0.0028 |
| pooled MMD²_al | 0.006923 | 0.00764 ± 0.0028 | 0.00398 ± 0.0022 | 0.00023 |
| axion MMD² | 0.005915 | 0.00947 ± 0.0060 | 0.00360 ± 0.0017 | -0.00074 |
| cdm MMD² | 0.017943 | 0.00804 ± 0.0044 | 0.00302 ± 0.0017 | -0.00121 |
| no_sub MMD² | 0.016027 | 0.00933 ± 0.0076 | 0.00446 ± 0.0027 | 0.00019 |

Reading. The biased floor halves when n doubles (pooled 0.00931 → 0.00481, ratio 1.94; PDF–JS
7.50e-05 → 4.08e-05, ratio 1.84), which is the 1/n bias the audit predicted, while the unbiased
floor is flat in n (0.00030 at n = 100, 0.00033 at n = 200) and consistent with zero, as a
U-statistic on two samples of one distribution should be. The v10 per-class floors were high by
2 to 6 times; the v10 pooled floor happened to land near the correct 200-vs-200 value because one
100-vs-100 split is a draw from a distribution whose sd (0.0028) is over half its own mean.
The mutually exclusive pooled draws give the same answer as the overlapping ones
(0.00524 ± 0.0025 against 0.00481 ± 0.0028), so the draw rule is not what sets the floor.

### Table 2 under v11 (seed 0, same runs, same held-out images)

| Model | N | PDF–JS | MMD² | MMD²_u | MMD²_al | MMD²_u,al | gen_max | clip frac |
|---|---|---|---|---|---|---|---|---|
| IQP, paper protocol (Alg. 1) | 11 | 0.00644 | 0.1793 | 0.1759 | 0.2027 | 0.1987 | 1.76 | 1.5e-04 |
| IQP, fixed basis, joint | 11 | 0.00137 | 0.1226 | 0.1193 | 0.1209 | 0.1169 | 1.59 | 9.8e-04 |
| IQP, learned dir. M=4 (flat MMD) | 11 | 0.00287 | **0.0894** | 0.0863 | **0.0259** | 0.0223 | 1.37 | 6.3e-04 |
| IQP, learned dir. M=4 (paper kernel) | 11 | **0.00081** | 0.0945 | 0.0912 | 0.0685 | 0.0646 | 1.57 | 1.1e-03 |
| RBM-10bit | 11 | 0.00607 | 0.0967 | 0.0929 | 0.1474 | 0.1423 | 2.50 | 4.8e-03 |
| DCGAN-32 | 11 | 0.00183 | 0.1865 | 0.1832 | 0.1214 | 0.1179 | 0.97 | 0 |
| IQP, paper protocol (Alg. 1) | 100 | 0.00792 | 0.2098 | 0.2061 | 0.2578 | 0.2537 | 2.78 | 3.0e-04 |
| IQP, fixed basis, joint | 100 | **0.00097** | 0.0921 | 0.0885 | 0.0655 | 0.0614 | 1.93 | 8.7e-04 |
| IQP, learned dir. M=4 (flat MMD) | 100 | 0.00330 | 0.0694 | 0.0659 | **0.0272** | 0.0234 | 1.51 | 5.3e-04 |
| IQP, learned dir. M=4 (paper kernel) | 100 | 0.00146 | 0.1088 | 0.1050 | 0.0977 | 0.0934 | 2.53 | 1.3e-03 |
| RBM-10bit | 100 | 0.00191 | 0.0975 | 0.0927 | 0.1764 | 0.1711 | 1.39 | 2.8e-04 |
| DCGAN-32 | 100 | 0.00763 | **0.0374** | 0.0330 | 0.0638 | 0.0604 | 1.00 | 0 |
| *Floor (200 v 200, 20 draws)* | n/a | *4.1e-05 ± 2.1e-05* | *0.0048 ± 0.0028* | *0.0003 ± 0.0028* | *0.0040 ± 0.0022* | *0.0002 ± 0.0021* | n/a | n/a |

Two winners move relative to v10. At N = 100 the PDF–JS winner changes from the learned M=4
(paper kernel) arm (v10 0.00092, v11 0.00146) to the fixed-basis joint arm (v10 0.00099,
v11 0.00097): the two were separated by 7e-05 under a bin grid that the arm with the higher
gen_max (2.53 against 1.93) was widening. At N = 11 the raw MMD² winner changes from the RBM
(v10 0.0741, v11 0.0967, rank 1 to 3) to the learned M=4 flat-MMD arm (0.0894), which is the
shared-bandwidth effect the audit measured, and is consistent with that RBM run having the largest
clip fraction in the table (4.8e-03, gen_max 2.50). Everything else keeps its v10 winner: the
flux-aligned MMD² column stays with the learned M=4 flat-MMD arm at both N, the N = 100 raw MMD²
column stays with the DCGAN, and the N = 11 PDF–JS column stays with the learned M=4 paper-kernel arm.
The qualitative claim of the section is unchanged and stays mixed: no arm wins all three columns,
the DCGAN still owns raw MMD² at N = 100, and the best IQP arm still owns the flux-aligned column.

### Other tables under v11

* **Per-class (Table 3).** No class ordering changes on PDF–JS at either N. Raw MMD² reorders at
  N = 11 only (cdm and axion swap 2nd and 3rd, 0.0814 against 0.0868, a gap of 0.005 against a floor
  sd of 0.0017); N = 100 keeps axion < cdm < no_sub. The flux-aligned column also swaps its 2nd and 3rd at
  N = 11 (axion 0.0262, no_sub 0.0266, previously 0.0264 and 0.0238) and is unchanged at N = 100.
  The per-class floors drop by 2 to 6 times
  (see the floor table), which matters more for the appendix's "how close to the floor" sentence
  than the arm ordering does.
* **Scaling.** The three-arm ordering per N is stable on the flux-aligned column (IQP first at every
  N) and on PDF–JS except at N = 10 and N = 50, where IQP and RBM swap 1st and 2nd with gaps under
  5e-05. Raw MMD² reorders at five of the six N (IQP moves ahead of the RBM at N = 30, 70, 90, 100
  and ahead of the DCGAN at N = 50),
  so the "IQP flat, RBM flat, DCGAN swings" reading is unchanged but the RBM's raw-MMD² advantage
  over IQP at large N was a bandwidth artefact.
* **Convergence sweep.** The best step per arm and metric is unchanged in 11 of 12 cases; only the
  RBM at N = 100 on raw MMD² moves (80k under v10, 160k under v11). The suggested step counts above
  (RBM 80k, DCGAN not defensible) do not change.

### Clipping (which arms v10 was flattering)

64 of the 106 seed-0 runs put pixels above 1. The clipped fractions are small (median 5.4e-04,
maximum 6.2e-03 for `rbm_n30`), so v11's clip removes a negligible amount of light, but the same
pixels were setting the histogram width and the encoder input scale for the whole comparison
under v10. The largest gen_max values are the RBM's late convergence checkpoints
(`rbm_n100_s160k` 12.0, `rbm_n11_s160k` 8.3), which is the same over-dispersion the convergence
section already flags. The DCGAN is the only arm that never exceeds 1 (its output passes through
`clip((x+1)/2, 0, None)` on a tanh generator), so it was the only arm never affected by the v10
bin-widening, which is worth one sentence if a reviewer asks why the DCGAN's PDF–JS is identical
under both protocols.
