# Provenance of every number in `main.tex`

Written 2026-09-08 alongside the ML4PS 2026 draft, and updated the same day for the final
submission draft (positive-framing restructure: new single-sentence title, hardness-first abstract
and introduction, §3.1 renamed and its permutation-null material compressed to two caveat sentences,
§3.4 reordered to lead with the win, `Findings.` / `Limitations.` paragraph names, Appendix A's
pipeline replaced by `fig_pipeline_v2`, Appendices C and D sharing one two-panel figure). Rows below
that changed in that pass carry an inline note. Every quantity that appears in the submitted PDF is
listed with the file it came from. Paths are relative to `<HOME>/gsoc2026-deeplense-qml/`.
`facts.md` = `paper/ml4ps2026/facts.md` (the number sheet, itself carrying `file:line` provenance).
Abbreviations: **SUITE** = `docs/IQP_LENS_PAPER_SUITE_RESULTS.md`, **R3** = `docs/IQP_LENS_ROUND3_RESULTS.md`,
**MAP** = `docs/IQP_LENS_MAP_RESULTS.md`, **TURB** = `docs/IQP_TURBULENCE_REPLICATION_RESULTS.md`,
**PAPER** = `docs/PAPER_2603.06440_SUMMARY.md`.

Everything is single seed (seed 0) and exact classical simulation; both facts are stated in the paper.

---

## Abstract

| Value | Source |
|---|---|
| `I_QCLI = 0.47`, `I_CCI = 0.87` (lens, 18-bit, K = 4,096) | facts.md §2 table (0.4715 / 0.8677) → MAP:29 |
| structure-matched permutation null `I_QCLI = 0.42` | MAP:16; `results_map.json` point 0 `perm_QCLI` = 0.4154 |
| reconstruction correlation `0.97` (best arm, N = 100) | `results_r3b.jsonl` arm `IQP-affine-M4-MMDflat-resample`, N=100, `projection_corr_mean` = 0.9684 |
| `2`–`39×` fewer parameters | Table 1: RBM 11,010 / 5,520 = 1.99×; DCGAN 199,837 / 5,164 = 38.7× |
| DCGAN leads raw Conv2D-MMD² at `N = 100` | Table 1 (`DCGAN-32` 0.0371 vs best IQP 0.0861) |
| `N = 11` / `N = 100` | facts.md §3 |

Revision note (2026-09-08, final draft): the abstract no longer quotes the i.i.d.-uniform null `0.008`
(it was mislabelled "matched null" in the 07:48 draft; the matched null is the permutation null 0.42)
and no longer quotes a correlation range, only the best arm's 0.97.

## Section 2 (Setup)

| Value | Source |
|---|---|
| 80,194 train / 8,910 val, 64×64, 3 classes pooled | facts.md §1 (memory `dataset1-equals-model-ii.md`, `docs/LENS_IMAGE_CHARACTERISTICS.md:5`) |
| centroid offset 3.4 ± 1.6 px | facts.md §1 → `docs/IQP_LENS_ROUND3_PREREGISTRATION.md:10-11` |
| crop keeps 99.4 % of light, min 98.4 %, over 3,000 val images | same |
| 1,024 cells, n = 10 qubits, all 2^10 generators, `code = (kx<<5)|ky` | facts.md §1 |
(The "≈ 4 ms/step, 4 CPU threads" claim was removed from the paper on 2026-09-08: it was not
traceable, and `outputs_lens_r3b/*/meta.json` records `device = cuda` with 9.3–10.9 ms/step.)
| fixed total flux `S = 124.92` | `experiments/13_iqp_turbulence_replication/outputs_lens_r3b/iqp_m4_mmdflat_resample_n100/meta.json`, key `S` = 124.92079504305893 |
| `d_lat = 50`, remaining 974 core coordinates frozen (Algorithm 1) | PAPER §4.6 (d_lat = 50, Hamming-ordered), R3:35 (arm definitions) |
| row of Φ normalised to √1024 = 32 | `meta.json` key `phi_row_norm` = 32.0; `train_iqp_affine_r3.py:123` (`PHI_SCALE`) |
| 10^-3 L2 penalty on z, raised to 0.3 for the M=8 arm | `meta.json` key `zpen` = 0.001; `outputs_lens_r3b/iqp_m8_kl_zpen_resample_n100/meta.json` `zpen` = 0.3 |
| Conv2D-MMD² = biased MMD in a fixed random 3-layer conv encoder, median-heuristic Gaussian kernel | `experiments/13_iqp_turbulence_replication/metrics.py:52-121` (`ConvEncoder` 1→16→32→64, stride 2, ReLU, global average pool; `conv_mmd2`) |
| RBM 10 visible / 1,000 hidden / PCD-1, 11,010 params | `results_r3.jsonl` arm `RBM-10bit`, `param_count` = 11010; `baseline_rbm_r3.py` |
| DCGAN-32, 199,837 params | `results_r3.jsonl` arm `DCGAN-32`, `param_count` = 199837 |
| 200 generated vs 200 held-out images | facts.md §3 / R3:60 |

## Section 2, "Departures from the reference setup" (added 2026-09-08)

| Value | Source |
|---|---|
| reference paper encodes 18-bit (x, y, intensity) tuples on 18 qubits | `docs/PAPER_2603.06440_SUMMARY.md:24` (per coordinate N = 6 bits, 18 bits total) |
| its generators truncated by subset size to "about 31k phases" | `docs/PAPER_2603.06440_SUMMARY.md:176` quotes its models as 31k and 63k parameters. At n = 18 the subset counts match exactly: sum of C(18,k) for k <= 6 = 31,180 and for k <= 7 = 63,004, so the truncation is by |s|. The paper text quotes only "about 31k", not the rule, on purpose. |
| its MMD estimated from "500 model samples per step" | `docs/PAPER_2603.06440_SUMMARY.md:142` (500 model samples per step, 3e4 steps, N_t = 5e4 data points) |
| lens "raw intensity axis carries barely more than one bit" | `docs/LENS_IMAGE_CHARACTERISTICS.md:45`, raw 64-level intensity quantisation carries **1.42 bits** of the 6 available. **Not** phrased as "84 % of pixels are near zero": the same file states the images are *not* sparse (one strictly zero pixel per image; effective support ~331 of 4,096 px), so a sparsity claim would be wrong. |
| 99.4 % of the light in the crop, flux S = 124.92, all 1,024 generators | already sourced in the Section 2 block above; the departures paragraph repeats no number that the pipeline paragraph states |

The paragraph adds no measurement of our own. It restates the reference paper's setup from the
summary document and contrasts it with choices already documented elsewhere in this file.

## Section 3.1 (correlation–complexity map) — all from facts.md §2 / MAP

| Value | Source |
|---|---|
| lens 18-bit `I_QCLI = 0.4715 ± 0.0075`, `I_CCI = 0.8677 ± 0.0078` | MAP:29; `paper_suite/map/results_map.json` point 0 |
| turbulence D2 at the same K = 4,096: `0.023` (CCI 0.929) | results_map.json point 8 ("per-snapshot (M=4096)", 0.0232 / 0.9294). The paper text quotes this row, not the M = 5×10^4 row (0.0910 / 0.9085, point 7), because K is not comparable across rows. |
| binarised MNIST `0.014` (784-bit) | results_map.json point 12 (0.0135) |
| BinaryBlobs `0.032` (p = 0.05) | results_map.json point 13 (0.0316) |
| lens 18-bit permutation null `0.4154` / CCI `0.8782`; “+0.056 attributable to structure” | MAP:16; results_map.json point 0 `perm_QCLI`/`perm_CCI` |
| D2 permutation null at K = 4,096: `0.011` (CCI 0.937) | results_map.json point 8 `perm_QCLI`/`perm_CCI` = 0.0113 / 0.9371. (Point 7, at M = 5×10^4, has perm 0.0984 / 0.8779; that row is no longer quoted.) |
| plug-in CCI null `0.975`–`0.9998` across datasets | SUITE §2 ("本地圖 13 列有限樣本 null 落在 0.975–0.9998") |
| QCLI null `0.008` at K = 4,096 → `0.405` at K = 4.1×10^5, n = 18 | SUITE §2; results_map.json points 0 and 1 (`null_QCLI` 0.0077, 0.4046). The paper calls the sample count K (M is reserved for the number of learned latent directions). |
| 10-bit crop pmf `0.473` vs `0.178`; CCI `0.431` vs `0.963` | MAP:29,31; results_map.json point 2. **Both quoted nulls are the permutation null** (facts.md §2 columns `permutation-null QCLI`/`permutation-null CCI` = 0.1782 / 0.9633); the i.i.d.-uniform null of that row is 0.9944 / 0.0000. The §3.1 sentence said "both nulls" and was corrected to "its permutation null" on 2026-09-08. |
| plug-in CCI null range and the MNIST/Blobs K = 10^4 note | SUITE §2; results_map.json points 12, 13. The CCI-null range row is no longer quoted in the paper (cut for space in the final draft). |

## Section 3.2 (why the reference protocol fails)

| Value | Source |
|---|---|
| single-crop fit corr `0.9999` (KL), `0.99997` (flat MMD), `0.9962` (paper kernel) | facts.md §5 → `mech_tests/mech_test/results.json` (`a.fits`), `docs/IQP_LENS_ROUND3_PREREGISTRATION.md:100-101` |
| other training images reconstructed at `0.47` | facts.md §5 (round-2 diagnostic) → `docs/IQP_LENS_ROUND3_PREREGISTRATION.md:107`. The companion "anchor at 0.999" half-claim was removed on 2026-09-08 because no source records it (do not confuse it with the single-crop full-parameter fit 0.9999 above). |
| "the paper's sequential protocol, our low-variance latent block" | Table 1 row 1 is `IQP-lowvar-paper-MMD` (latent = `lowvar`, not Hamming-ordered); the Hamming-ordered variant is a separate arm, reported in App. E |

## Section 3.3 (learned latent directions)

| Value | Source |
|---|---|
| best held-out projection corr `0.968`, min `0.851` | facts.md §6; `results_r3b.jsonl` arm `IQP-affine-M4-MMDflat-resample`, N=100 (`projection_corr_mean` 0.9684, `_min` 0.8514). Independently recomputed from the saved `theta0.npy`/`Phi.npy`/`z_holdout.npy` by `fig_src/make_paper_figs.py`, matching `meta.json` to 4e-16. |
| nearest-training-image reference `0.970` | `meta.json` key `ref_projection_nearest_train_corr` = 0.96995 |
| mean-training-image reference `0.779` (figure only) | `meta.json` key `ref_projection_mean_crop_corr` = 0.77886 |
| random Φ `0.672`, no latent `0.671` | `mech_tests/mech_test2/diag2.json` (0.67168 / 0.67053). `0.671` **removed from text 2026-09-08 dedup** (it is the trained model with its latent switched off, not the M = 0 point of Fig. 2(a), which is a from-scratch M = 0 model at 0.7438). |
| paper kernel `0.74`–`0.79` | `results_r3b.jsonl`, N=100: `IQP-affine-M2-auto3-gauss` 0.7437, `IQP-affine-M4-auto3-resample` 0.7940. Quoted to two decimals so that it is not confused with the mean-training-image reference 0.744 of the N=20 sweep in Fig. 2(a). |
| controlled N=20 sweep, trained Φ `0.911` | `mech_tests/mech_test2/diag2.json` (0.9113). Fig. 2(a) and the §3.3 sentence contrasting 0.911 / 0.672 / 0.671 are all this experiment; the 0.968 / 0.851 figures are the separate N=100 round-3b run. After the 2026-09-08 dedup the §3.3 sentence contrasts 0.911 against 0.672 only. |
| KL `0.914` (M=2), `0.874` (M=4) | `results_r3b.jsonl`, N=100: 0.9140, 0.8741 |
| M=2 samples: 100 % rings, rms radius `9.40 ± 0.21` px vs data `9.21 ± 0.15`, ellipticity sd `0.048` vs `0.099` | facts.md §5 → `docs/IQP_LENS_ROUND3_PREREGISTRATION.md:168-170` |
| N=20 sweep collapses to `0.095` by M=8 | `mech_tests/mech_test2/diag.json` `exp2_redo.M8.best.corr_mean` = 0.0952 (M=16 is 0.1470); §3.3 quoted `0.10` until 2026-09-08 and now quotes `0.095` to match the three-decimal figures in the same sentence |
| M=8 collapses to `0.636` | `results_r3b.jsonl` arm `IQP-affine-M8-KL-zpen-resample`, N=100, `projection_corr_mean` = 0.6360 |
| coherence radius ≈ `0.03`–`0.05` | facts.md §5; `meta.json` `z_norm_max` = 0.0414, `z_norm_mean` = 0.0174 |

## Table 1 and Section 3.4

All eight numeric columns come from two jsonl files, field-for-field. The paper-protocol row's N = 11
Conv2D-MMD² was corrected from `0.2032` to `0.2031` on 2026-09-08 (`results_r3.jsonl`
`IQP-lowvar-paper-MMD`, n_train 11, `conv_mmd2` = 0.20314624). Section 3.4 now names the winning arm
per metric (PDF--JS from the M = 4 paper-kernel arm, flux-aligned MMD² from the M = 4 flat-MMD arm)
because no single arm wins both. On 2026-09-08 the `IQP, learned dir. M=2 (KL)` row
(`IQP-affine-M2-KL-resample`, `results_r3b.jsonl`, PDF--JS 0.001754 / 0.003022, `conv_mmd2` 0.16729 /
0.13556, `conv_mmd2_aligned` 0.055913 / 0.086638, `param_count` 3094 / 3272) was **moved out of
Table 1 into App. E**, because exact KL needs the full simulated distribution and cannot be estimated
from hardware shots. The same move changes the §3.4 sentence from "two of five / three of five" to
"two of the four / two of four": at N=11 the arms above the DCGAN's 0.00183 are the paper protocol
(0.00609) and the M=4 flat-MMD arm (0.00196); at N=100 the arms above the RBM's 0.00195 are the paper
protocol (0.00697) and the M=4 flat-MMD arm (0.00239). Two values dropped from the text in the same
pass, the single-crop KL fit `0.9999` (§3.2) and the KL projection correlation `0.914` at M=2 (§3.3);
both survive in `facts.md` §5 and §6.

| Table row | jsonl arm | file |
|---|---|---|
| IQP, paper protocol (Alg. 1) | `IQP-lowvar-paper-MMD` | `results_r3.jsonl` |
| IQP, fixed basis, joint | `IQP-lowvar-joint-MMD-resample` | `results_r3.jsonl` |
| IQP, learned dir. M=4 (flat MMD) | `IQP-affine-M4-MMDflat-resample` | `results_r3b.jsonl` |
| IQP, learned dir. M=4 (paper kernel) | `IQP-affine-M4-auto3-resample` | `results_r3b.jsonl` |
| RBM | `RBM-10bit` | `results_r3.jsonl` |
| DCGAN | `DCGAN-32` | `results_r3.jsonl` |
| Held-out split-half (floor) | `HOLDOUT split-half` | either file (identical) |

Columns map to fields `pdf_js`, `conv_mmd2`, `conv_mmd2_aligned`; parameter counts to `param_count`
(the learned-direction arms differ between N=11 and N=100 because the per-image latents `z_i` scale
with N). The same table is reproduced in facts.md §3 and R3 Table A/B.

Text claims in §3.4 (final draft wording): “the **best** IQP arm wins PDF–JS at both sizes”
(0.00095 / 0.00092 vs RBM 0.00632 / 0.00195 and DCGAN 0.00183 / 0.00763), “wins flux-aligned MMD² at
both” (0.0279 / 0.0275), “two of five IQP arms fall behind the DCGAN on PDF–JS at N=11 and three of
five fall behind the RBM at N=100” (count the five IQP rows of Table 1 against the two baseline rows),
“the IQP arms lose raw MMD²” (RBM 0.0741 at N=11, DCGAN 0.0371 at N=100) — all read directly off
Table 1, quoted to the table's own precision. The DCGAN PDF–JS reversal (0.00183 at N=11 vs 0.00763
at N=100) is visible in the same table. Parameter ratios “2× / 39× fewer” are 11,010/5,520 = 1.99 and
199,837/5,164 = 38.7. Data-scaling values (IQP raw MMD² **0.082–0.087** flat for N ≥ 30, flux-aligned
0.022–0.027, RBM 0.066–0.089, DCGAN 0.113 → 0.037, DCGAN leads at N=10 with 0.044 vs 0.120) come from
`results_scaling.jsonl` / facts.md §4 / SUITE TL;DR 3 (the earlier “0.084–0.087” in this file was a
typo for the 0.082–0.087 span printed in the paper).

## Appendix B (turbulence replication) — facts.md §8 / TURB

| Value | Source |
|---|---|
| “not reproduced”; RBM and DCGAN-1.2M beat our IQP on both metrics at 11 snapshots | TURB:10-12 |
| our PDF–JS `0.0322 ± 0.0006`, `0.0476 ± 0.0004` vs paper `0.0238`, `0.0221` | TURB:13; paper values from PAPER §5.1 (Table 2 rows IQP 31k / 63k: 0.023784, 0.022084) |
| shuffled ground truth scores `0.704` on Conv2D-MMD², **below** every model in the paper's Table 2 (column span 0.90–1.47) | TURB:20 and :477-478; PAPER §5.1 (MMD² column min 0.8959, max 1.4720). facts.md §8 said “inside the range of every model”; corrected 2026-09-08 in both files. |
| zero-parameter interpolation reaches correlation `0.9998` | TURB:16 (TL;DR point 4) |
| anchor-fit ceiling `0.69` sampled / `0.77` exact; contributions `−0.08`, `−0.04`; family ceiling `0.88`–`0.89` | TURB §3(d) / facts.md §8 |

## Appendix C (support mismatch, reference paper Fig. 3 — v2 run)

All values supplied by the coordinator on 2026-09-08 from the v2 run
(`outputs_controlled/results_fig3_v2.jsonl`, 562 rows = 140 targets + 420 fits + null + meta, job
356221, 897 s wall) and cross-checked
against SUITE §3:

| Value | Source |
|---|---|
| 140 target circuits (≤ 4-local), n = 16; 420 fits; 5,000 Adam steps at lr 1e-4 | **verified**: results_fig3_v2.jsonl has 140 `record="target"` and 420 `record="fit"` rows, each with `steps`=5000, `lr`=1e-4 |
| exact target QCLI span `[0, 0.408]`, family ceiling ≈ `0.41` | **verified**: min/max of `I_QCLI_exact` = 0.0000 / 0.4079 |
| learners with 50 / 100 / 150 gates, exact MMD, Adam, 5,000 steps | SUITE §3 |
| median achieved MMD² `9.57e-3 → 2.15e-3 → 1.45e-3` (three significant figures since 2026-09-08; previously quoted as 9.6 / 2.2 / 1.45) | **recomputed independently 2026-09-08**: medians of `best_mmd2` per learner in `results_fig3_v2.jsonl` = 9.572e-3 / 2.153e-3 / 1.450e-3 (supersedes the v1 numbers 2.87e-3 → 1.11e-3 → 1.02e-3 in SUITE §3) |
| achieved MMD² **rises** with target QCLI (the paper says "rises", not "monotonically", because the binned lower envelope dips at the low end); Spearman `+0.98 / +0.90 / +0.81` | **recomputed independently 2026-09-08** from `results_fig3_v2.jsonl` (`best_mmd2` vs `I_QCLI_exact`, per learner): +0.979 / +0.901 / +0.809. The coordinator’s “≈2.1×” magnitude could **not** be reproduced (bin-median ratios give 3–5×), so it was removed from the text. |
| Proposition II.2 floor proxy falls with QCLI, `ρ ≤ −0.98` (the paper quotes the bound, not a single value) | **recomputed independently**: `mmd2_floor_proxy_exact` vs `I_QCLI_exact` gives −0.988 / −0.988 / −0.996 per learner |
| sample-QCLI compressed to `0`–`0.064`; i.i.d.-uniform null `0.0696` | **verified**: min/max of `I_QCLI` = 0.0000 / 0.0644; `record="null"` row `I_QCLI_uniform_mean` = 0.069597 |
| paper's range `0`–`≈0.7` | PAPER §5.2 |


## Appendix D (QCLI–CCI ceiling, reference paper Fig. 4)

| Value | Source |
|---|---|
| n = 8, 12, 16; 3 seeds and 2 gain settings per n; 3,333 steps each, ≈20,000 logged points per n | facts.md §7 (`outputs_controlled/results_fig4.jsonl`, 60,016 rows; meta `steps` = 3333, 3 seeds × 2 gains). The caption previously said “20,000 logged steps”, which misdescribed the run; corrected 2026-09-08. |
| Spearman `+0.83 / +0.76 / +0.95` for QCLI ≤ 0.3 | SUITE §4 table (+0.827 / +0.759 / +0.950) |
| global `ρ = −0.53, −0.57` for n = 12, 16 | SUITE §4 table (−0.527, −0.571) |
| reversal at QCLI ≈ `0.38` | SUITE §4 (binned-median CCI peak at 0.38 for n = 12, 16) |
| uniform-distribution QCLI `0.982 / 0.998 / 1.000`, CCI exactly 0 | facts.md §7 (calibration rows: 0.9816 / 0.9984 / 0.9999); SUITE §4 |
| paper's figure stops below QCLI ≈ 0.7 | PAPER §5.3 |

## Appendix E (additional arms)

| Value | Source |
|---|---|
| twenty-five conditions = 16 IQP arms + RBM + DCGAN + 7 zero-learning/PCA controls | round 3 (14 arms) + round 3b (11 arms), R3 §1; the paper's sentence counts conditions, not IQP arms alone |
| NN-holdout of every IQP arm `3.6`–`5.9`; floor `0.95` | `results_r3.jsonl` / `results_r3b.jsonl` field `nn_holdout` (min 3.554 `IQP-lowvar-joint-KL`, max 5.875 `IQP-affine-M8-KL-zpen-resample`); floor from `HOLDOUT split-half` (0.9489) |
| M=8 arm: PDF–JS `0.0103`, MMD² `0.244` at N=100 | `results_r3b.jsonl` arm `IQP-affine-M8-KL-zpen-resample` (0.010280, 0.24408) |
| Hamming-ordered block, joint protocol, PDF--JS `0.00137` (N=11) and `0.00139` (N=100) | `results_r3.jsonl` arm `IQP-hamming-joint-MMD`, `pdf_js` = 0.0013672497 (n_train 11) and 0.0013909022 (n_train 100). Added to App. E on 2026-09-08 so that the §3.2 pointer to the Hamming variant resolves to numbers. We did not run Hamming + sequential, and App. E says so. |
| scaling numbers | `results_scaling.jsonl` / facts.md §4 |

---

## Appendix A (pipeline figure, `fig_pipeline_v2`)

Every panel value in the Fig. 5 caption (held-out frame 55, crop, code 404 = 0110010100₂ for
(k_x,k_y) = (12,20), the M = 4 latent scatter, PDF–JS 0.0024, Conv2D-MMD² 0.086 / 0.075 / 0.037 at
N = 100, mean projection correlation 0.968) is listed panel by panel in
`figures/FIGURE_PROVENANCE.md`, section `fig_pipeline_v2.pdf`.

## Appendix F (per-class generation)

All eight table rows are read field-for-field from
`experiments/13_iqp_turbulence_replication/results_class.jsonl` (18 rows, SLURM job 358382, partition
`dev`, account <YOUR_ACCOUNT>, elapsed 00:07:24, exit 0:0) and the pooled reference rows from
`results_r3b.jsonl`, arm `IQP-affine-M4-MMDflat-resample`, at the matching N. Quoted to four
significant figures. Cross-checked against the `facts.md` addendum "per-class generation".

| Row | arm (`results_class.jsonl`) | N | pdf_js | conv_mmd2 | conv_mmd2_aligned | projection_corr_mean |
|---|---|---|---|---|---|---|
| axion only | `IQP-affine-M4-MMDflat-resample-axion` | 11 | 0.0021969147 | 0.10150149 | 0.026364098 | 0.92320602 |
| cdm only | `IQP-affine-M4-MMDflat-resample-cdm` | 11 | 0.0021735977 | 0.11081300 | 0.020309469 | 0.92383292 |
| no sub only | `IQP-affine-M4-MMDflat-resample-no_sub` | 11 | 0.0011019071 | 0.08096495 | 0.023794119 | 0.89977204 |
| pooled | `IQP-affine-M4-MMDflat-resample` (`results_r3b.jsonl`) | 11 | 0.0019630 | 0.11339 | 0.027860 | 0.90924 |
| axion only | `IQP-affine-M4-MMDflat-resample-axion` | 100 | 0.0020503075 | 0.09012000 | 0.023464523 | 0.97203498 |
| cdm only | `IQP-affine-M4-MMDflat-resample-cdm` | 100 | 0.0022750194 | 0.09782690 | 0.018483537 | 0.96864299 |
| no sub only | `IQP-affine-M4-MMDflat-resample-no_sub` | 100 | 0.0020014842 | 0.10335508 | 0.027729875 | 0.97034709 |
| pooled | `IQP-affine-M4-MMDflat-resample` (`results_r3b.jsonl`) | 100 | 0.0023870 | 0.08607 | 0.027450 | 0.96838 |

Two statements in the appendix text carry no number on purpose. "Each class is scored against its own
held-out set, which carries its own split-half floor" rests on the per-class `HOLDOUT split-half` rows
of the same jsonl (`conv_mmd2` 0.005915 / 0.017943 / 0.016027 for axion / cdm / no_sub, a 3x spread),
which is why the rows are not compared with one another. "The three differ in only a few per cent of
the pixel variance" rests on `docs/LENS_IMAGE_CHARACTERISTICS.md:195` (class signal from PC6 onward,
3.0 % of pixel variance) and `:205` (nearest-centroid on raw pixels 38.3 % against a 33.3 % chance
baseline); no figure is quoted in the paper.

Figure `figB_classes` provenance (arm, directories, selection rule, per-class split) is in
`figures/FIGURE_PROVENANCE.md`.

## Items that still need Leo's confirmation

1. **DeepLense dataset citation.** The paper cites `alexander_2020_deep` (Alexander et al. 2020,
   *Deep Learning the Morphology of Dark Matter Substructure*, ApJ 893:15, arXiv:1909.07346) as the
   source of `model_II.npz`. `data/README.md` and `papers/README.md` do not explicitly attribute the
   file to that paper; it is the most likely candidate (it established the axion/CDM/no_sub 3-class
   convention). **Confirm against the ML4SCI DeepLense repository before camera-ready.**
   (facts.md §1 TODO 1, and the coordinator's instruction of 2026-09-08.)
2. **Appendix C v2 aggregates** — recomputed independently from `results_fig3_v2.jsonl` on 2026-09-08 (see the table above); they are *not* yet written into `facts.md`, so add them there if that sheet is reused.
3. **`refs.bib` author lists.** Twelve entries had `..." placeholders for unverified co-authors; the
   placeholder tokens were removed on 2026-09-08 (backup: `refs.bib.backup-20260908`), so those
   entries now print surname-only for the affected authors. All `note = {...}` fields (internal
   annotations, several containing TODOs) were stripped so they do not leak into the printed
   bibliography. Full author lists should be re-verified via arXiv before camera-ready.
4. **Figure 2 panel description.** The caption describes the delivered vector figure
   `figures/fig2_mechanism_scaling.pdf` as of 2026-09-08 07:26. If that figure is redrawn, re-check
   the caption (in particular whether panel (b) shows samples or projections).
