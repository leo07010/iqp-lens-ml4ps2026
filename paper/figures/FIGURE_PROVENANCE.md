# Figure provenance — ML4PS 2026 submission

Every number and every pixel in `fig_pipeline`, `fig_pipeline_v2`, `fig1_map_samples`,
`fig2_mechanism_scaling`, `figA_fig3`, `figA_fig4` and `figB_classes` is traced to a file below.  Rebuild everything with

```
cd paper/ml4ps2026/figures && python3 make_figures.py          # ~15 s, CPU only
python3 make_figures.py fig1_map_samples                       # or one stem at a time
```

Scripts: `fig_style.py` (shared style), `fig_pipeline.py`, `fig_pipeline_v2.py`,
`fig1_map_samples.py`, `fig2_mechanism_scaling.py`, `figA_fig3.py`, `figA_fig4.py`,
`figB_classes.py`, driver `make_figures.py`.
Each writes `<stem>.pdf` (vector, TrueType fonts) and `<stem>.png` (300 dpi).

Experiment root (`EXP` below):
`<HOME>/gsoc2026-deeplense-qml/experiments/13_iqp_turbulence_replication`

## Conventions used everywhere

| item | choice |
|---|---|
| font sizes | 8 pt axis labels, 7 pt tick labels / legends / in-panel text; nothing below 7 pt |
| categorical colours | Okabe–Ito, fixed order: blue `#0072B2`, vermilion `#D55E00`, green `#009E73`, orange `#E69F00`, purple `#7B3FA0` |
| redundant encoding | every series carries its own marker shape as well as its colour |
| image panels | `magma`, **linear**, `vmin = 0`, `vmax = 1`, `origin="lower"`, `interpolation="nearest"` |
| widths | 5.5 in (`\textwidth`) for `fig_pipeline` / `fig1` / `fig2`; 2.7 in (half column) for the two appendix figures |
| PDF export | `bbox_inches="tight"`, `pad_inches=0.02`, `pdf.fonttype=42` |

Final PDF media boxes: `fig_pipeline` 5.50 × 1.82 in, `fig_pipeline_v2` 5.46 × 3.58 in,
`fig1_map_samples` 5.51 × 3.23 in, `fig2_mechanism_scaling` 5.45 × 3.15 in,
`figA_fig3` 2.63 × 2.19 in, `figA_fig4` 2.68 × 2.16 in, `figB_classes` 5.21 × 2.10 in.

**Intensity clipping.** All samples are rescaled to the same total flux `S = 124.92`
(`data_lens_r3.load_r3(100).S`), so a shared 0–1 scale is the honest one, but a handful of
IQP peak pixels exceed 1 and are clipped by `vmax = 1`:
Fig. 1(b) IQP paper-protocol 3 / 12 288 px (max 1.146), IQP learned-Φ 13 / 12 288 px (max 1.491);
Fig. 2(b) as shipped in `fig2_compact` M=4 (flat MMD) 13 / 12 288 px (max 1.491), M=4 (paper kernel)
15 px (max 1.602), M=8 (collapsed) 1 px (max 1.013). (The M=2 paper-kernel row, 28 px, max 1.826, is
no longer shipped.)
Held-out, RBM and DCGAN rows have no clipped pixels.

---

## `fig_pipeline.pdf` — method schematic (5.50 × 1.82 in)

Pure schematic, no data file is read.  `dot`/graphviz is **not installed** on this cluster
(`which dot` → not found), so the diagram is drawn with matplotlib `FancyArrowPatch` plus
auto-sized text bbox patches (measured after a draw pass, so no label can overflow its box).

Quantities quoted inside the boxes:

| box text | source |
|---|---|
| "64×64 lens image (DeepLense simulation)" | `EXP/data_lens_r3.py` (`R3Data.holdout_images` is `(200, 64, 64)`) |
| "centroid align + 32×32 crop (99.4% of light)" | `EXP/data_lens_r3.py:98` `crop_light_fraction`, `CROP = 32`; measured mean light kept = 0.9944 (train, N=100) / 0.9943 (held-out, N=200), min 0.9891 |
| "pmf over 1024 pixels = 10 qubits, code (kx≪5)\|ky" | `EXP/codec10.py` (`image_to_pmf`, code `(kx << 5) \| ky`) |
| "IQP: H⊗ⁿ e^{iΣ_s θ_s Z_s} H⊗ⁿ, all 1024 generators, θ = θ⁰ + Φᵀz, M = 2–4" | `EXP/iqp_exact.py`, `EXP/train_iqp_affine_r3.py` (`Phi.npy` is `(M, 1024)`) |
| "exact FWHT simulation; MMD (flat / paper kernel) or KL, joint over N" | `EXP/iqp_exact.py:fwht`, `EXP/train_iqp_affine_r3.py` objectives `mmdflat` / `mmd` / `kl` |
| "latent prior (resample / Gaussian) → q(x,y) × S → paste at sampled offset" | `EXP/train_iqp_affine_r3.py` (`prior` ∈ {`resample`, `gauss`}), `data_lens_r3.sample_offsets`, `meta.json:gen_offsets` |
| "metrics: PDF–JS, Conv2D-MMD², held-out projection corr." | `EXP/evaluate_r3.py`, `EXP/evaluate_r3b.py` |
| refuted branch: "paper protocol: fixed Hamming-first latent, sequential stage 1/2" | arm `IQP-lowvar-paper-MMD`, `EXP/results_r3.jsonl` (physics = 3.520 at N=100 vs 0.123 for the held-out split-half floor) |

---

## `fig1_map_samples.pdf` (5.51 × 3.23 in)

### Panel (a) — correlation–complexity map
Source: **`EXP/paper_suite/map/results_map.json`** (generated 2026-09-07, seed 0,
`L_subset_budget = 20000`, by `paper_suite/map/run_map.py`).  Filled marker = dataset,
hollow marker of the same colour = its i.i.d.-uniform null at the *same* (n, M) joined by a
thin line, small `×` on a dotted line = the structure-matched permutation null.

| legend label | `representation` field | n | M | I_QCLI | I_CCI | i.i.d. null | permutation null |
|---|---|---|---|---|---|---|---|
| lens 18-bit | `18-bit (x,y,z) tuples, per-image` | 18 | 4096 | 0.4715 | 0.8677 | (0.0077, 0.9980) | (0.4154, 0.8782) |
| lens 18-bit, pooled | `18-bit (x,y,z) tuples, pooled 100 train` | 18 | 409 600 | 0.7095 | 0.8013 | (0.4046, 0.9998) | — |
| lens 10-bit pmf | `10-bit crop pmf, per-image (exact pmf)` | 10 | exact pmf | 0.4731 | 0.4313 | (0.9944, 0.0000) | (0.1782, 0.9633) |
| turbulence D2 | `18-bit tuples, per-snapshot (M=4096)` | 18 | 4096 | 0.0232 | 0.9294 | (0.0077, 0.9985) | (0.0113, 0.9371) |
| turb. D2, pooled | `18-bit (x,y,\|u\|) tuples, pooled 11 anchors` | 18 | 550 000 | 0.1619 | 0.8837 | (0.4691, 0.9997) | — |
| MNIST-16 | `16-bit (7x7 mean-pool to 4x4, thr 0.5)` | 16 | 10 000 | 0.0157 | 0.1237 | (0.0696, 0.9984) | — |
| MNIST-784 | `784-bit (thr 0.5)` | 784 | 10 000 | 0.0135 | 0.5684 | (0.0129, 0.9992) | — |
| BinaryBlobs | `16-bit, 8 patterns, flip p=0.05` | 16 | 10 000 | 0.0316 | 0.5417 | (0.0694, 0.9986) | — |

Rendering decisions (deviations from the brief, all deliberate):

* **The 10-bit point's own null is annotated, not drawn.** Its i.i.d.-uniform null sits at
  (0.9944, 0.0000); connecting it would put a full-width diagonal across the panel. The
  marker is drawn filled like every other dataset (a hollow marker would collide with the
  "null" convention) and the saturation is stated in text: *"saturated at n = 10:
  null = (0.99, 0.00)"*. This is the same information, stated rather than drawn.
* **The "IQP-compatible" region has no hard edge.** The paper states only a qualitative
  "upper-right quadrant" (`docs/PAPER_2603.06440_SUMMARY.md:197`) and gives no threshold, so
  the panel uses a soft alpha wash increasing towards (1, 1) rather than inventing a cut.
* **Legend moved below both panels.** With 8 datasets + 3 marker conventions, an in-panel
  legend at 7 pt could not avoid covering data; the 4-column legend under the figure keeps
  every label at 7 pt with no overlap.
* The `484-bit` MNIST point and the `p = 0.01` BinaryBlobs variant that exist in
  `results_map.json` are not drawn (redundant with the 784-bit / p = 0.05 entries).
  The per-image `M = 4096` and the pooled `M = 409600` 10-bit rows are likewise omitted.

### Panel (b) — samples
Rows, 3 samples each (`generated_000/001/002.npy`, all 64 × 64 float32):

| row | source |
|---|---|
| held-out (real) | `data_lens_r3.load_r3(100, n_holdout=200).holdout_images[0:3]` |
| IQP (paper protocol) | `EXP/outputs_lens_r3/iqp_paper_n100/` — arm `IQP-lowvar-paper-MMD` |
| IQP, learned Φ (M = 4, MMD-flat) | `EXP/outputs_lens_r3b/iqp_m4_mmdflat_resample_n100/` — arm `IQP-affine-M4-MMDflat-resample` |
| RBM | `EXP/outputs_lens_r3/rbm_n100/` — arm `RBM-10bit` |
| DCGAN | `EXP/outputs_lens_r3/dcgan_n100/` — arm `DCGAN-32` |

Arm ↔ directory mapping verified against `EXP/results_r3.jsonl` and `EXP/results_r3b.jsonl`
(`run_dir` field).  All five rows are N = 100 training images.

---

## Layout change 2026-09-08 (compact figures shipped in the paper)

`main.tex` no longer includes `fig1_map_samples` or `fig2_mechanism_scaling`. It ships three compact
figures instead, so that every figure is placed at its natural size with 7 pt fonts rather than being
scaled down to fit the four-page main text.

| in the paper | file | media box | included at |
|---|---|---|---|
| Fig. 1 | `fig1_compact.pdf` | 5.44 x 2.28 in | `width=1.0\textwidth` (5.50 in), i.e. a 1.1 % upscale |
| Fig. 2 | `fig2_compact.pdf` | 5.34 x 1.97 in | `width=1.0\textwidth` (5.50 in), i.e. a 3.0 % upscale |
| Fig. 5 (App. E) | `figA_scaling.pdf` | 2.58 x 2.06 in | `width=0.48\textwidth` (2.64 in), inside a `wrapfigure` |

**Appendix figures 2026-09-08.** The appendix now ships the native-size variants
`fig_pipeline_v2_c` (width 0.92), `figA_fig3_c` and `figA_fig4_c` (0.38 minipages) and
`figB_classes_c` (width 0.70), so that appendix text also stays at 7 pt. The non-`_c` files remain
for reference and are no longer included by `main.tex`.

`fig2_compact` carries only panels (a) and (b); the old panel (c) became the standalone
`figA_scaling` in App. E. Its panel (b) rows are held-out, M = 4 (flat MMD), M = 4 (paper kernel) and
the collapsed M = 8 arm, so the M = 2 row discussed in the section below is no longer shipped.

**`figA_scaling` label fix (2026-09-08).** Its dashed series was labelled `centroid-aligned`, which
contradicts the paper, where `conv_mmd2_aligned` is the **flux-aligned** variant (Sec. 2 defines the
shared flux scalar S; `centroid-aligned` in the paper refers to the integer-shift image alignment,
a different operation). The label in `figA_scaling.py` was corrected to `flux-aligned` and the figure
regenerated (backup `figA_scaling.py.backup-20260908`). Any regeneration of that figure must keep
`flux-aligned`.

The two sections below describe the superseded full-size figures and are kept because the compact
scripts reuse their data loaders.

## `fig2_mechanism_scaling.pdf` (5.45 × 3.15 in)

### Panel (a) — held-out projection correlation vs M
Controlled mechanism test, **N = 20 training images**, joint training of
{θ⁰, Φ, z} (`EXP/mech_tests/mech_test2/affine_latent.py`).

| M | held-out corr | file : json path |
|---|---|---|
| 0 | 0.7438 | `mech_test2/results.json` : `configs/KL_M0/holdout/corr_mean` |
| 2 | 0.9087 | `mech_test2/diag3.json` : `KL_M2/holdout/corr_mean` |
| 4 | 0.9113 | `mech_test2/results.json` : `configs/KL_M4/holdout/corr_mean` |
| 6 | 0.7705 | `mech_test2/diag3.json` : `KL_M6/holdout/corr_mean` |
| 8 | 0.0952 | `mech_test2/diag.json` : `exp2_redo/M8/best/corr_mean` |
| 16 | 0.1470 | `mech_test2/diag.json` : `exp2_redo/M16/best/corr_mean` |

M = 8 and M = 16 take the **best over the learning-rate sweep** in `diag.json`
(`lr ∈ {0.020, 0.005, 0.001} × {zero, zmean}` init), i.e. the value most favourable to the
model, so the collapse claim is the conservative one.  The single-config values in
`results.json` are 0.0910 (M = 8) and 0.1568 (M = 16) — same conclusion.

Reference levels (horizontal lines):

| line | value | source |
|---|---|---|
| nearest training image | 0.9339 | `mech_test2/results.json` : `controls/holdout_vs_nearest_train_corr/mean` |
| mean training image | 0.7439 | `mech_test2/results.json` : `controls/holdout_vs_mean_image_corr/mean` |
| random directions | 0.6719 | mean of `mech_test2/diag2.json` : `holdout_random_vs_trained/random Phi(seed 99)…/corr_mean` = 0.6717 and `…seed 123…` = 0.6721 |

Note for the caption: M = 0 (0.7438) lands on the mean-training-image level (0.7439) to
within 1e-4 — with no latent, the model *is* one shared distribution.  `diag2.json`'s
"no latent (θ⁰ only)" control, which reuses the trained-Φ run's θ⁰, gives 0.6705.

### Panel (b) — samples (round 3b, N = 100 — a *different* run from panel (a))
Rows match Table 1.  `generated_000/001/002.npy` from `EXP/outputs_lens_r3b/<tag>/`:

| row | tag | arm | objective | projection corr (`results_r3b.jsonl`) |
|---|---|---|---|---|
| held-out (real) | — | — | — | — |
| IQP, M = 4 (flat MMD) | `iqp_m4_mmdflat_resample_n100` | `IQP-affine-M4-MMDflat-resample` | `mmdflat` | 0.968 |
| IQP, M = 4 (paper kernel) | `iqp_m4_auto3_resample_n100` | `IQP-affine-M4-auto3-resample` | `auto3` | 0.794 |
| IQP, M = 8 (collapsed) | `iqp_m8_kl_zpen_resample_n100` | `IQP-affine-M8-KL-zpen-resample` | `kl` | 0.636 |

The two M = 4 rows are the same model size and the same resample prior, differing only in the
training kernel, so the row pair isolates the objective; M = 8 is the collapsed regime.

Round-3b reference values for that column: mean crop 0.779, nearest training image 0.970
(`docs/IQP_LENS_ROUND3_RESULTS.md:93`).  Panels (a) and (b) come from two different
experiments (N = 20 controlled sweep vs N = 100 round 3b) and the caption must say so.

### Panel (c) — Conv2D-MMD² vs training-set size
Source: **`EXP/results_scaling.jsonl`** (30 rows).  Solid = `conv_mmd2`, dashed (open
markers) = `conv_mmd2_aligned`.  N ∈ {10, 30, 50, 70, 90, 100}.

| arm | params | `conv_mmd2` | `conv_mmd2_aligned` |
|---|---|---|---|
| `IQP-affine-M4-MMDflat-resample` | 5 160–5 520 | 0.11976, 0.08399, 0.08262, 0.08697, 0.08217, 0.08607 | 0.02908, 0.02271, 0.02227, 0.02536, 0.02178, 0.02745 |
| `RBM-10bit` | 11 010 | 0.07661, 0.07627, 0.08887, 0.06571, 0.07269, 0.07519 | 0.09468, 0.11979, 0.07554, 0.09449, 0.12710, 0.11438 |
| `DCGAN-32` | 199 837 | 0.04403, 0.11282, 0.07533, 0.08219, 0.04173, 0.03711 | 0.04481, 0.06411, 0.09049, 0.10657, 0.09729, 0.06937 |

Grey band = `HOLDOUT split-half` floor: `conv_mmd2` 0.00401, `conv_mmd2_aligned` 0.00692
(N-independent by construction; the same row is repeated for every N in the jsonl).
`RANDOM-PAIRING (ref)` rows carry `null` for these metrics and are not plotted.
y-axis is log so the floor stays visible.  The x ticks are the six actual training-set sizes
(10, 30, 50, 70, 90, 100), read off the jsonl rather than hard-coded.

---

## `figA_fig3.pdf` — support-mismatch control (2.63 × 2.19 in)

Source: **`EXP/outputs_controlled/results_fig3_v2.jsonl`** (562 lines: 1 `meta`, 1 `null`,
140 `target`, 420 `fit`).  Run meta: `n = 16`, `locality_target = 4`,
`locality_learner = 2`, 20 targets per target-gate-count, `gates_target ∈ {140 … 1050}`,
5 000 Adam steps at lr 1e-4, seed 20260907, wall 897 s.

* **x** = `I_QCLI_exact` of the target pmf; span `[0.0000, 0.4079]` (`meta.I_QCLI_exact_span`).
* **y** = `final_mmd2` of the trained learner (log axis).
* **colour** = `learner_gates` `G_n ∈ {50, 100, 150}` — **140 fit records each**, not 136;
  no record was dropped for non-finite or non-positive `final_mmd2`.  (The brief said 136
  for G = 150; the file contains 140.)
* **envelope** = per-bin maximum over 8 equal bins on [0, 0.4079]:
  * G = 50 : 0.00611, 0.00734, 0.00791, 0.00950, 0.01005, 0.01270, 0.01528, 0.01523
  * G = 100: 0.00185, 0.00149, 0.00157, 0.00214, 0.00262, 0.00351, 0.00505, 0.00508
  * G = 150: 0.00170, 0.00076, 0.00086, 0.00133, 0.00150, 0.00222, 0.00346, 0.00346
* **No null line is drawn.**  The `null` record's `I_QCLI_uniform_mean = 0.06960` is on the
  *sample*-QCLI scale (M = 10 000 draws), while this axis is the *exact* I_QCLI of the target
  pmf; the two are not comparable.  Worth stating in the text instead:
  the sample-QCLI of the same targets never exceeds 0.0644
  (`meta.I_QCLI_sample_span`), i.e. it never clears its own null of 0.0696.
* Annotation: "targets: n = 16, ≤4-local; learners: ≤2-local, G_n gates;
  reachable I_QCLI ≤ 0.41".

---

## `figA_fig4.pdf` — QCLI vs CCI beyond the paper's range (2.68 × 2.16 in)

Source: **`EXP/outputs_controlled/results_fig4.jsonl`** (60 016 lines: 1 `meta`,
3 `calibration`, 60 012 `traj`).  Run meta: sizes {8, 12, 16}, 3 seeds, 2 gain arms
(0.5 / 2.0), 3 333 steps recorded every step, `gates_requested = 150`
(effective 36 / 78 / 136), locality 2, seed 20260907, float64 on CUDA.

Reduction (both gain arms and all 3 seeds pooled per size, 20 004 circuits each):
18 equal bins on [0, 1]; a bin is kept only with ≥ 20 circuits (all 18 survive for every n);
the line is the per-bin **median** `I_CCI`, the band the 10th–90th percentile,
plotted at the per-bin median `I_QCLI`.

| n | circuits | median I_CCI, first bin | median I_CCI, last bin | exact i.i.d.-uniform calibration (★) |
|---|---|---|---|---|
| 8 | 20 004 | 0.594 | 1.000 | (0.9816, 0.0000) |
| 12 | 20 004 | 0.623 | 0.462 | (0.9984, 0.0000) |
| 16 | 20 004 | 0.724 | 0.472 | (0.9999, 0.0000) |

Calibration points are the `record = "calibration"`, `which = "exact_uniform"` rows.
The shaded band marks `I_QCLI ≤ 0.7`, the range the paper plots; the divergence between
n = 8 and n = 12/16 and the crossing towards the ★ both happen outside it.

---

## `fig_pipeline_v2.pdf` — illustrated method figure (5.46 × 3.58 in)

The detailed alternative to `fig_pipeline.pdf`; **both are kept**, `fig_pipeline.*` is not
overwritten.  Every stage carries a real thumbnail, all on the shared linear magma 0–1 scale.
`dot`/graphviz is not installed on this cluster, so the circuit is hand-drawn with matplotlib
primitives (no pennylane dependency).  Nothing in this figure is below 7 pt.

Fixed choices, all deterministic:

| constant | value | why |
|---|---|---|
| `HOLDOUT_IDX` | 55 | most azimuthally complete ring among `holdout_images[:60]` (min/mean sector intensity on the peak annulus = 0.904) |
| `GEN_IDX` | 43 | same criterion over this arm's `crops/generated_*.npy[:60]` (0.923) |
| `REF_IDX` | 0 | reference-protocol sample; every sample from that arm is noise |
| `KX, KY` | 12, 20 | the pixel whose 10-bit code is spelled out |

| panel | content | source |
|---|---|---|
| (a) | held-out 64×64 frame, white `+` at the intensity centroid, dashed 32×32 crop window | `data_lens_r3.load_r3(100).holdout_images[55]`; centroid (31.43, 31.26) and crop offset (16, 16) from `data_lens_r3.centroid` / `align_crop` |
| (b) | aligned 32×32 crop; the same array as a pmf with 4×4 block gridlines, the example pixel boxed; the code arithmetic beside it | `data_lens_r3.align_crop`; `codec10.encode(12, 20) = 404`, `format(404, "010b") = 0110010100`, convention `code = (k_x≪5)\|k_y`, bit *j* → qubit *j* (`codec10.py:9`) |
| (c) | scatter of the first two learned latent coordinates, 100 training points; $\theta = \theta^0 + \Phi^\top z$, $M = 4$ | `outputs_lens_r3b/iqp_m4_mmdflat_resample_n100/z_train.npy` `(100, 4)`; `Phi.npy` is `(4, 1024)` |
| (d) | hand-drawn IQP circuit: 5 of 10 wires (`⋮` for the rest), H layer, one diagonal block $\exp(i\sum_s\theta_s Z_s)$ labelled "1024 generators, all subsets", H layer, measurement meters, output $x\in\{0,1\}^{10}$ | schematic; the 1024-generator / 10-qubit facts from `codec10.py` and `iqp_exact.py` |
| (e) | training box $q_\theta = \lvert\mathcal{W}e^{i\theta}\rvert^2/2^n$, MMD (flat / reference kernel) or KL, joint over $N$; dashed feedback arrow updating $\theta^0, \Phi, z$ | `iqp_exact.py:fwht`, `train_iqp_affine_r3.py` |
| (f) | $q(x,y)\times S$ thumbnail → "× S, paste at offset" → the pasted 64×64 sample | `…/crops/generated_043.npy` (sums to $S = 124.92$) and `…/generated_043.npy` |
| (g) | three metric minis: (i) pixel-intensity PDF of the 200 held-out frames vs the 200 generated frames, 50 bins on the joint range — **exactly the binning `metrics.pdf_js` uses** (`metrics.py:pdf_js`), log-*y*; (ii) horizontal Conv2D-MMD² bars at $N = 100$, IQP 0.0861 / RBM 0.0752 / DCGAN 0.0371; (iii) histogram of the 200 per-image held-out projection correlations (mean 0.968) with the mean-crop (0.779, grey) and nearest-training (0.970, vermilion) reference lines | `results_r3b.jsonl` (IQP), `results_r3.jsonl` (RBM, DCGAN); `…/meta.json` `projection_corrs`, `ref_projection_mean_crop_corr`, `ref_projection_nearest_train_corr` |
| bottom bar | reference-protocol sample (✗) vs this work's $M = 4$ sample (✓) | `outputs_lens_r3/iqp_paper_n100/generated_000.npy` and `…r3b/iqp_m4_mmdflat_resample_n100/generated_043.npy` |

**No pixel in this figure is clipped**: the maxima of the five thumbnails are 1.000 (held-out
frame), 1.000 (crop), 0.614 (`q×S`), 0.614 (generated) and 0.875 (reference-protocol sample),
all ≤ `vmax = 1`.

---

## `figB_classes.pdf` — appendix: per-class generation (5.21 × 2.10 in)

**What it shows.** The paper's best arm trained *separately* on each of the three dark-matter
substructure classes.  Three rows (`axion`, `cdm`, `no sub`); each row is two held-out images
of that class followed by three images generated by a run trained on 100 images of that class
alone.  Every panel uses the same `magma`, linear, `vmin = 0`, `vmax = 1` scale as the other
image panels, with one shared colorbar.

**Why it exists.** Every result in the main paper pools the three classes.  This figure and
the table in `facts.md` § "Addendum: per-class generation" answer the narrower question — does
the same arm train and generate for each class on its own?

**Arm.** Unchanged from the main result: IQP-affine, `M = 4`, MMD-flat objective, resample
prior, seed 0, `lr 0.02`, 4 000 joint steps, 1 000 θ₀ seed steps, 8 × 1 000-step projection
restarts at `infer-lr 0.005`, `N = 100`, 200 generated frames, 200 held-out.  Byte-for-byte the
`IQP-affine-M4-MMDflat-resample` command line of `submit_lens_r3b.sbatch:160-162`; the *only*
change is which images it sees.

| row | held-out source (2 images) | generated source (3 images) |
|---|---|---|
| axion  | `data_lens_class.install("axion")` → `data_lens_r3.load_r3(100, n_holdout=200).holdout_images[0:2]` | `EXP/outputs_lens_class/axion_m4_mmdflat_resample_n100/generated_000/001/002.npy` |
| cdm    | `data_lens_class.install("cdm")` → same call | `EXP/outputs_lens_class/cdm_m4_mmdflat_resample_n100/generated_000/001/002.npy` |
| no sub | `data_lens_class.install("no_sub")` → same call | `EXP/outputs_lens_class/no_sub_m4_mmdflat_resample_n100/generated_000/001/002.npy` |

Arm ↔ directory mapping verified against `EXP/results_class.jsonl` (`run_dir` / `arm` fields);
the arm names are class-tagged (`IQP-affine-M4-MMDflat-resample-<class>`) so the three runs
cannot collide in the results table.

**Selection rule (fixed in advance, no cherry-picking).** The *first two* held-out images and
the *first three* generated frames of each run, in index order — held-out `i = 0, 1` and
`generated_000/001/002.npy`.  Nothing is reordered, screened for ring completeness, or swapped
between rows.  The rule is stated in `figB_classes.py`'s docstring and is the same rule
`fig1_map_samples.py` uses for its sample rows.

**Per-class split (this is the one thing that differs from the pooled runs).**
`data_lens_class.py` builds a cache per class and `install(<class>)` swaps three functions
(`_arrays`, `build_cache`, `get_cache_meta`) inside the imported `data_lens_r3` module, so
`load_r3` transparently returns that class's images.  No existing script is edited on disk.

| | pooled (main paper) | per class (this figure) |
|---|---|---|
| training set | `default_rng([0,3,0]).permutation(80194)[:100]` | `default_rng([0,3,0,c]).permutation(indices with label c)[:100]` |
| latent pool | same permutation, `[100:2100]` | same per-class permutation, `[100:2100]` |
| held-out | `default_rng([0,3,1]).permutation(8910)[:200]` | `default_rng([0,3,1,c]).permutation(val indices with label c)[:200]` |
| flux scale `S` | 124.92 | axion 127.19 (per class; each arm uses its own class's `S`) |

The per-class training sets are **not** subsets of the pooled one: the pooled `N = 100`
contains only ≈ 33 images of each class, so a per-class `N = 100` cannot be carved out of it.
The class index is appended to the seed so the three classes get independent permutations.
Purity and disjointness are asserted at run time (`test_class.py`) and re-asserted on the real
dataset in the job log (`logs_lens/sbatch-class-358382.out`): all three classes report
`pure(train,val)=(True,True) disjoint=True`.

**Provenance of the pixels.** SLURM job **358382**, partition `dev`, node `25a-hgpn024`,
account <YOUR_ACCOUNT>, `/usr/bin/python3`.  Scripts: `data_lens_class.py`,
`train_iqp_affine_class.py`, `evaluate_class.py`, `test_class.py`, `submit_lens_class.sbatch`
(all in `EXP/`).  Metrics in `EXP/results_class.jsonl`.

---

## Body-column variants — `fig1_compact`, `fig2_compact`, `figA_scaling`

Written because the full-height Fig. 1 / Fig. 2 had to be scaled to 0.58 / 0.66 of
`\linewidth` in the body, which drops their 7 pt text to 4.5–5 pt.  These three are laid out
to be included at `width=\linewidth` at close to their natural size, so 7 pt stays essentially 7 pt.
The residual upscale is **+1.1 % for `fig1_compact`** (5.44 in drawn into 5.50 in) and **+3.0 % for
`fig2_compact`** (5.34 in into 5.50 in), measured on the delivered `main.pdf` as 108.0-504.0 pt on
both pages.  They add no
new numbers: each imports its loaders from the full-size script, so the pair cannot drift.

| file | size | replaces | what changed |
|---|---|---|---|
| `fig1_compact.pdf` | 5.44 × 2.28 in | `fig1_map_samples.pdf` | same panel (a) map at a full 2.26 in wide; the dataset legend moves to a thin 0.95 in column between the panels with shortened labels; panel (b) grid uses 0.33 in cells with one-line row labels and a horizontal colour bar beneath |
| `fig2_compact.pdf` | 5.34 × 1.97 in | `fig2_mechanism_scaling.pdf` panels (a)+(b) | panel (c) removed (it becomes `figA_scaling`); 0.40 in cells |
| `figA_scaling.pdf` | 2.58 × 2.06 in | `fig2_mechanism_scaling.pdf` panel (c) | stands alone at half-column width, legend inside the axes |

### `fig1_compact.pdf`
* Panel (a): identical data to **Fig. 1(a)** above — `fig1_compact.py` calls
  `fig1_map_samples.load_map()`, so the 8 datasets, their i.i.d. nulls and their permutation
  nulls are the same rows of `paper_suite/map/results_map.json`.  The same two documented
  deviations apply (the 10-bit null is annotated rather than drawn; the "IQP-compatible"
  wash has no hard edge).  Legend labels are shortened to fit the thin column:
  `lens 18b`, `lens 18b pool`, `lens 10b`, `turb. D2`, `D2 pooled`, `MNIST-16`,
  `MNIST-784`, `Blobs`.
* Panel (b): identical rows to **Fig. 1(b)** — `fig1_map_samples.load_images()`, i.e.
  held-out / `iqp_paper_n100` / `iqp_m4_mmdflat_resample_n100` / `rbm_n100` / `dcgan_n100`,
  samples 000–002, N = 100.  Row labels shortened to one line: `held-out`, `IQP ref.`,
  `IQP M=4`, `RBM`, `DCGAN`.

### `fig2_compact.pdf`
* Panel (a): identical to **Fig. 2(a)** — `fig2_mechanism_scaling.load_mech()`, so the six
  M values and the three reference levels come from the same
  `mech_tests/mech_test2/{results,diag,diag2,diag3}.json` paths listed above.
* Panel (b): identical to **Fig. 2(b)** — `fig2_mechanism_scaling.load_images()` with that
  module's `ARMS_B`, i.e. held-out, `iqp_m4_mmdflat_resample_n100` (projection corr 0.968),
  `iqp_m4_auto3_resample_n100` (0.794, the paper's reference kernel) and
  `iqp_m8_kl_zpen_resample_n100` (0.636), all from `results_r3b.jsonl` at n_train = 100 —
  the same four rows as Table 1.

### `figA_scaling.pdf`
* Identical to **Fig. 2(c)** — `fig2_mechanism_scaling.load_scaling()` and its
  `SCALING_ARMS`, i.e. the same three arms and the same `HOLDOUT split-half` floor from
  `results_scaling.jsonl` (values tabulated in the Fig. 2(c) section above).  x ticks are the
  six actual training-set sizes read off the jsonl.

---

## Native-size appendix variants — `*_c`

Written because the appendix figures were being scaled to 0.92 / 0.38 / 0.70 of
`\textwidth`, which drops their 7 pt text to 4–5.5 pt.  These four are laid out for exactly
that target width, so they are included at `width=<target>` with **at most a few percent of
upscaling** — 7 pt never shrinks.  They introduce **no new data**: each imports its loader
from the full-size script, so the pair cannot drift.  Originals are untouched.

| file | target width | actual PDF | source of every number |
|---|---|---|---|
| `fig_pipeline_v2_c.pdf` | 5.06 in (0.92 `\textwidth`) | 4.91 × 3.29 in | `fig_pipeline_v2.load()` — identical panels (a)–(g), see the `fig_pipeline_v2` section above |
| `figA_fig3_c.pdf` | 2.09 in (0.38 `\textwidth`) | 2.06 × 1.85 in | `figA_fig3.load()` + `figA_fig3.envelope()` — same `results_fig3_v2.jsonl`, same 8-bin envelopes |
| `figA_fig4_c.pdf` | 2.09 in (0.38 `\textwidth`) | 1.99 × 1.80 in | `figA_fig4.load()` + `figA_fig4.binned()` — same `results_fig4.jsonl`, same 18-bin medians and 10–90 bands |
| `figB_classes_c.pdf` | 3.85 in (0.70 `\textwidth`) | 3.40 × 1.57 in | `figB_classes.load_rows()` — same three classes, same "first two held-out, first three generated" selection rule |

What had to change to fit 7 pt at these widths (content, never font size):

* **`fig_pipeline_v2_c`** — thumbnails 0.66 → 0.58 in, circuit 4.92 → 3.74 in, the (a) caption
  shortened to "held-out", the phase-block subtitle to "1024 generators, all subsets $s$",
  and the three metric minis lose their axis decoration: the Conv2D-MMD² panel now carries
  rotated in-panel bar labels (IQP / RBM / GAN) instead of a y axis, and the projection title
  is "projection 0.968" rather than "held-out proj. mean 0.968".  Same values throughout
  (PDF–JS 0.0024; MMD² 0.0861 / 0.0752 / 0.0371; projection mean 0.968).
* **`figA_fig3_c`** — the three-line note becomes two lines,
  "$n = 16$, ≤4-local targets; ≤2-local learners".  It does not fit on one line at 7 pt in
  2.09 in (that string is ~2.3 in wide), so it is wrapped rather than shrunk.
  x ticks thinned to 0, 0.2, 0.4.
* **`figA_fig4_c`** — "paper's plotted range" → "paper's range"; the calibration note →
  "uniform calib. (★)"; ticks thinned to 0, 0.5, 1.0; the per-*n* labels moved inward.
* **`figB_classes_c`** — cells 0.40 in, the right-hand group header shortened from
  "generated, trained on this class only" to "generated (this class)", and the colour-bar
  label from "pixel intensity (a.u.)" to "intensity (a.u.)".

Smallest font in all four scripts is exactly 7.0 pt (verified by grepping every `fontsize=`
and `labelsize=`); the modest upscaling at include time only makes text larger.
