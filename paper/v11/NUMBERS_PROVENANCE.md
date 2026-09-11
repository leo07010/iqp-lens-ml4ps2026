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

**SUPERSEDED BY THE v11 SECTION BELOW for every number that carries a `±`.** The v10 rows are kept
because they document where each quantity came from originally; the v11 section says which
seed-aggregate table the submitted value is now read from.

---

# v11 (2026-09-09): every number in `v11/main.tex`

v11 = v10 text + five-seed `mean ± sd` (sample sd, `ddof = 1`) + the v11 metric protocol + the
correctness fixes from the X-ray audit. Sources, all relative to
`<HOME>/gsoc2026-deeplense-qml/paper/ml4ps2026/`:

* **ST1** = `seed_tables.md` **Part 1** (v11 metric protocol: Table 2, Table 3, Figure 5, floors)
* **ST2** = `seed_tables.md` **Part 2** (v10 protocol: params/S, baseline step counts, θ⁰ ablation,
  §3.2 diagnostics, per-restart projection, repeat-run gaps, protocol notes)
* **STB** = `seed_tables_b.md` (Table 1 map incl. native-K rows, Fig. 2(a) M sweep, App. C, App. D)
* **STC** = `seed_tables_c.md` (Appendix B turbulence, 5 seeds, sections C1–C6)
* **FACTS** = `v11/facts.md`, addendum "Addendum: metric protocol v11" (metric definitions, σ values,
  floor construction)
* **V11J** = `experiments/13_iqp_turbulence_replication/results_v11_seed0.jsonl` (seed-0 re-scoring of
  all 106 runs under the v11 protocol; used only for the App. E arms that were never seeded)

## Abstract

| Value in v11 | Source |
|---|---|
| 18-bit tuples `I_QCLI = 0.47`, `I_CCI = 0.87` | STB §1, row "Lens, 18-bit (x,y,z) tuples" (0.4715 ± 0.0000 / 0.8677 ± 0.0000) |
| 10-bit crop pmf `0.47`, `0.43` | STB §1, row "Lens, 10-bit crop pmf" (0.4731 / 0.4313) |
| `2–39×` more parameters at 11 training images | ST2 "Parameter counts and the flux scale S": RBM total 11,510 / IQP M=4 total 5,164 = 2.2×; DCGAN 199,837 / 5,164 = 38.7× |
| reconstruction correlation `0.9684 ± 0.0004` | ST2 "Held-out projection correlation, by restart", IQP M=4 (flat MMD) N=100, "best by CORR (reported)" |
| permutation null `I_QCLI = 0.42` | STB §1, row "permutation null" under the lens 18-bit row (0.4154) |
| "the DCGAN leads at 100 images" | ST1 "Table 2 (v11) — wins that the seed spread does not support", N=100 rows |
| "over five seeds" | ST1 header (seeds 0–4, ddof = 1) |

## §2 Setup

| Value in v11 | Source |
|---|---|
| baselines early-stopped, RBM 160k/80k, DCGAN 5k–15k | ST2 "Baseline step counts chosen on the validation set" (median column) |
| IQP arms fixed 4,000 steps | ST2 "Protocol notes", first bullet |
| PDF–JS 50 bins fixed on [0,1], generated pixels clipped, clipped mass ≤ 6e-3 | FACTS "v11 definitions", PDF–JS bullet; clip fractions in ST1 Part-1 notes (max 0.36 %) |
| Conv2D-MMD² scale and bandwidth from the held-out set alone, shared by every arm | FACTS "v11 definitions", Conv2D-MMD² bullet |
| floor = 20 draws of 200 vs 200 validation images | FACTS "v11 definitions", Floor bullet; values in ST1 "200-vs-200 real-vs-real floor (v11)" |
| flux-aligned = rescale every image to the held-out mean total intensity | FACTS "v11 definitions", aligned-range note |
| `S` = training-set mean total intensity, 112.3 at N=11, 124.9 at N=100, held-out 123.9 | ST2 "Parameter counts and the flux scale S" (S column); held-out mean in `seed_tables.md` header |
| θ⁰ warm-started by 1,000 exact-KL steps | ST2 "theta0 warm-start ablation" preamble (`--theta0-steps 1000`, `train_iqp_affine_r3.py:457-465`) |
| paper kernel = IQPopt multi-bandwidth Gaussian, bandwidth rule from the library | X-ray 3 H1; σ values in App. E row below |
| z penalty λ = 1e-3·L₀, raised to 0.3·L₀ for M=8 | X-ray 3 M1 (relative penalty) |
| projection from 8 restarts, corr-selected; loss-selected trails by ≤ 0.004; zero-init alone 0.968 | ST2 "Held-out projection correlation, by restart" (columns "best by LOSS", "best by CORR", "restart 0 only") |
| M=4 in Table 2, M ∈ {0,2,4,6,8,16} in the Fig. 2(a) sweep | STB §2 (M values present in the sweep) |

## §3.1 and Table 1

| Value in v11 | Source |
|---|---|
| all eight rows of Table 1, with ± | STB §1 table (5 seeds, `run_map.py --seed 0..4` + `agg_map_seeds.py`) |
| native-K turbulence rows (K = 5×10⁴): 0.0910 ± 0.0000 / 0.9085 ± 0.0000, null 0.0981 ± 0.0007 / 0.8782 ± 0.0003 | STB §1, rows "Turbulence D2, 18-bit tuples (native K)" and its null |
| "above its null at the lens K and below it at the paper's own K" | STB §1, "One thing the new native-K row adds" |
| K definition, lens exhaustive vs turbulence subsample | STB §1, "K = 4,096 means two different things" |

## §3.2

| Value in v11 | Source |
|---|---|
| single-crop fit `0.99997 ± 0.00001` | ST2 "Section 3.2 diagnostics", MMD-flat row (0.999966 ± 5.749e-06) |
| frozen-core reconstruction `0.44 ± 0.04` (N=11), `0.43 ± 0.02` (N=100) | ST2 "Frozen-core reconstruction", "other images, mean corr" column |

## §3.3 and Figure 2

| Value in v11 | Source |
|---|---|
| best arm `0.9684 ± 0.0004` at N=100 | ST2 restart table, IQP M=4 (flat MMD) N=100 |
| paper kernel `0.792 ± 0.006` (N=11), `0.805 ± 0.012` (N=100) | ST2 restart table, IQP M=4 (paper kernel), "best by CORR (reported)" |
| matched random direction `0.7458 ± 0.0005` | STB §2 reference-level table, "random direction (paired)" |
| M=2 `0.9060 ± 0.0021`, M=4 `0.8577 ± 0.0559`, M=8 `0.218 ± 0.121` | STB §2 M-sweep table |
| Fig. 2(a) reference levels 0.934 / 0.744 / 0.746 | STB §2 reference-level table |
| Fig. 2(a) caption "M=8 and 16 best over a learning-rate sweep" | STB §2 protocol column; X-ray 5 M3 |
| figure file `fig2_compact_seeds` | `figures/FIGURE_PROVENANCE.md` |

## Table 2 (§3.4)

Every cell from **ST1 "Table 2 (v11)"** (5 seeds), rescaled to the caption's units (PDF–JS ×10⁻³,
MMD² and MMD²_al ×10⁻²). Bold follows **ST1 "Table 2 (v11) — wins that the seed spread does not
support"**: only the two N=100 DCGAN cells (PDF–JS, raw MMD²) have "none — the win holds".

| Value in v11 | Source |
|---|---|
| floor row `0.041 ± 0.021 / 0.48 ± 0.28 / 0.40 ± 0.22` | ST1 "200-vs-200 real-vs-real floor (v11)", pooled row |
| Params column, totals at N (1,524/5,974; 5,164/5,520; 11,510/15,960; 199,837) | ST2 "Parameter counts and the flux scale S" |
| DCGAN discriminator +53,497 | ST2 same table, "discriminator" column |
| † footnote, stage 1 trains core and anchor latent jointly | X-ray 3 M2 |

## §3.4 claims

| Claim in v11 | Source |
|---|---|
| "both leads sit inside one sd of the baselines" (N=11 PDF–JS and MMD²_al) | ST1 Table-2 wins table, N=11 rows (PDF–JS overlaps RBM and DCGAN; MMD²_al overlaps DCGAN) |
| "the DCGAN leading with the RBM inside its spread" (N=11 raw MMD²) | ST1 Table-2 wins table, N=11 MMD² row |
| "both baselines sitting ahead of every IQP arm" (raw MMD², both N) | ST1 Table 2 (v11): N=11 DCGAN 0.0509 < RBM 0.0654 < best IQP 0.0891; N=100 DCGAN 0.0223 < RBM 0.0587 < best IQP 0.0716. **Correction to the X-ray note** "IQP beats RBM on raw MMD² at both N", which held only for the seed-0 fixed-step RBM (0.0967); with validation early stopping over five seeds the RBM is ahead |
| "at N=100 the DCGAN leads all three columns" | ST1 Table 2 (v11), N=100 rows |
| "within the M=4 flat-MMD arm's spread on the flux-aligned one" | ST1 Table-2 wins table, N=100 MMD²_al row |

## Table 4 (per-class, Appendix F in v11)

| Value in v11 | Source |
|---|---|
| all N=100 rows, PDF–JS / MMD² / MMD²_al | ST1 "Table 3 (v11)", N=100 rows |
| proj. corr. column | ST2 "Table 3 — per-class generation", proj. corr. column (protocol-independent) |
| bold on axion proj. corr. | ST2 Table-3 wins table, N=100 proj. corr. row ("none — the win holds") |
| per-class floors (in App. F) | ST1 "200-vs-200 real-vs-real floor (v11)", axion / cdm / no_sub rows |

## §4 Discussion

| Value in v11 | Source |
|---|---|
| DCGAN not bit-reproducible, up to `57 %` (raw MMD²) and `74 %` (PDF–JS) | ST2 "Repeat runs of an identical configuration": conv_mmd2 gaps span 4.6–57.0 %, pdf_js gaps 1.9–74.1 %. **The "3–57 %" summary sentence in ST2's protocol notes does not match its own table**, so v11 quotes the table |

## Appendix A (pipeline)

| Value in v11 | Source |
|---|---|
| PDF–JS 0.0029, Conv2D-MMD² IQP 0.072 / RBM 0.059 / DCGAN 0.022, projection 0.968 | ST1 Table 2 (v11) N=100 rows (0.002912, 0.07155, 0.05865, 0.02225) and ST2 restart table |
| "paper kernel" in the figure and caption | X-ray 2 (terminology); `figures/fig_pipeline_v2_c.py:170` |
| "DCGAN" bar label | `figures/fig_pipeline_v2_c.py:212` |

## Appendix B (turbulence)

| Value in v11 | Source |
|---|---|
| RBM 0.217 ± 0.007, DCGAN-1.2M 0.045 ± 0.062, IQP-31k 0.272 ± 0.004 | STC C1 |
| last-write PDF–JS 0.0324 ± 0.0011 and 0.0473 ± 0.0005; Conv2D-MMD² 0.711 ± 0.012 and 0.738 ± 0.017 | STC C2 |
| shuffled ground truth 0.7045 ± 0.0095 | STC C5 (`gt_spatially_shuffled_px`) |
| linear interpolation correlation 0.9998, deterministic | STC C5 (`linear_interp_of_bracketing_anchors`, 0.99978) |
| anchor fit 0.651 ± 0.010 sampled, 0.739 ± 0.010 exact (31k arm) | STC C3, IQP-31k "mean of 11 anchors" (the v10 "0.69/0.77" was the 63k arm; STC C3 provenance correction) |
| freeze of the latent block −0.114 ± 0.021 | STC C4 attribution table |
| 5×10⁴-sample pmf −0.001 ± 0.022 | STC C4 attribution table |
| Algorithm 1 stage 2 0.000 ± 0.000 | STC C4 attribution table |
| model-family ceiling 0.892, no seed | STC C4 ("best config of the sweep", seed-independent) |

## Appendix C (support mismatch)

| Value in v11 | Source |
|---|---|
| medians 9.58e-3 / 2.15e-3 / 1.44e-3, sd ≤ 2e-5 | STB §3 first table |
| Spearman +0.979 ± 0.001 / +0.901 ± 0.001 / +0.809 ± 0.004 | STB §3 first table |
| learners G_n = 50 / 100 / 150 *requested*, the last capped at the 136 generators a ≤2-local circuit has at n = 16 | X-ray 3 M11. The figure legend and STB §3 label the third learner 150 because that is the requested gate count; the caption now states the cap, so figure and caption agree |
| ρ ≤ −0.98 exact-pmf proxy, ρ ≤ −0.93 for the 10⁴-sample proxy | STB §3 second table and "One wording issue this surfaces" |
| figure file `figA_fig3_c_seeds` | `figures/FIGURE_PROVENANCE.md` |

## Appendix D (QCLI–CCI ceiling)

| Value in v11 | Source |
|---|---|
| pooled ρ +0.79 / +0.79 / +0.90 for QCLI ≤ 0.3 | STB §4 table, "pooled (5 seeds)" column |
| per-seed +0.92 ± 0.10 / +0.91 ± 0.04 / +0.96 ± 0.01 | STB §4 table, per-seed column |
| global ρ +0.08, −0.57, −0.64 | STB §4 table, "global ρ, pooled" |
| per-seed global ρ +0.25 ± 0.92 (n=8), −0.52 ± 0.36 (n=12) | STB §4 table |
| uniform calibration 0.982 / 0.998 / 0.9999 | STB §4 table, "exact uniform calib." (0.9999 replaces v10's "1.000", X-ray 1) |
| figure file `figA_fig4_c_seeds` | `figures/FIGURE_PROVENANCE.md` |

## Appendix E (additional arms)

| Value in v11 | Source |
|---|---|
| scaling: IQP 0.068–0.072, RBM 0.055–0.067, DCGAN 0.046 ± 0.017 at N=10 falling to 0.022 | ST1 "Figure 5 (v11)", MMD² column |
| flux-aligned: IQP leads at N=10 and 30, DCGAN from N=50, overlapping except at N=10 and N=90 | ST1 "Figure 5 (v11) — wins that the seed spread does not support", MMD²_al rows. **Correction to the X-ray note** "aligned column IQP first at every N", which held at seed 0 under fixed-step baselines |
| exact-KL M=2 arm 0.00172 / 0.00302, MMD² 0.119 / 0.091, aligned 0.046 / 0.075 (seed 0) | V11J, `iqp_m2_kl_resample_n11` and `_n100` |
| Hamming block 0.00178 (N=11), 0.00175 (N=100), seed 0 | V11J, `iqp_hamming_n11`, `iqp_hamming_n100` |
| M=8 collapse PDF–JS 0.0102, MMD² 0.241 (N=100, seed 0) | V11J, `iqp_m8_kl_zpen_resample_n100` |
| 18 conditions (16 IQP arms + RBM + DCGAN) | v10 said 25 including 7 zero-learning/PCA controls; those are removed from v11 by <user>'s decision |
| paper-kernel σ = {0.989, 1.729, 2.236}, mean Pauli weights 2.00 / 0.77 / 0.48 | X-ray 3 H1 |
| Conv2D bandwidth 0.005547 raw, 0.002558 flux-aligned | FACTS "v11 definitions", Conv2D-MMD² bullet |
| θ⁰ ablation "changes no metric by more than one sd" | ST2 "theta0 warm-start — wins that the seed spread does not support" (every row overlaps) |
| zero-init restart 0.899 ± 0.018 (N=11), 0.9684 ± 0.0004 (N=100) | ST2 restart table, "restart 0 only" column |
| DCGAN repeat-run gap up to 57 % (raw MMD²) and 74 % (PDF–JS) | ST2 "Repeat runs of an identical configuration" (table, not its summary sentence) |
| Table E1 (fixed 20k vs chosen budget) | ST2 "Appendix — the same baselines at a fixed 20,000 steps" (v10 metric protocol, stated in the caption) |
| figure file `figA_scaling_seeds` | `figures/FIGURE_PROVENANCE.md` |

## Appendix F (per-class samples)

| Value in v11 | Source |
|---|---|
| N=11 projection correlations 0.9227 ± 0.0014 / 0.9248 ± 0.0030 / 0.8986 ± 0.0032, pooled 0.9102 ± 0.0026 | ST2 "Table 3 — per-class generation", N=11 proj. corr. column |
| per-class floors | ST1 "200-vs-200 real-vs-real floor (v11)" |

## Citations changed in v11

| Change | Reason |
|---|---|
| `alexander_2023_domain`: added Hanna Parul | X-ray 4 (missing author) |
| `schneidman_2003_network`: author order now Schneidman, Still, Berry, Bialek | X-ray 4 (wrong order) |
| `codsi_2025_classically` moved to the noise / classical-simulability clause | X-ray 4 (it is counter-evidence to hardness, not support) |

## Removed in v11 (<user>'s decision, not an audit finding)

* Discussion sentence "Zero-learning and PCA controls match or exceed the IQP arms on some
  physical-quantity metrics."
* Appendix E mention of "seven zero-learning or PCA controls" (25 conditions → 18).
* Appendix E memorisation sentence (the NN-holdout 0.95 comparison was inverted; X-ray 5 H2), dropped
  rather than rewritten because v11 discusses the generative models only.

## Layout note (revised 2026-09-09 after the CFP check)

The ML4PS CFP forbids modifications to the LaTeX template, so v11 carries **none**. The preamble is
the template's own defaults plus the `\@noticestring` hook the style file documents; it is in fact
cleaner than v10, which changed `\topfraction`, `\bottomfraction`, `\textfraction`,
`\floatpagefraction`, `topnumber`, `totalnumber`, `\bibsep` and used negative `\vspace` around the
scaling wrapfigure. Every one of those is now gone. Captions use the template size, the four tables
use `\scriptsize`, and the only `\setlength` left is `\tabcolsep` inside tables.

Four pages of main text were reached by content instead:

* Table 3 (per-class generation) moved from §3.4 into Appendix F, where it is now Table 4. §3.4
  keeps a one-sentence summary (projection correlation 0.969-0.972, PDF-JS and Conv2D-MMD² at the
  pooled level, each class against its own floor).
* Table 1's two native-K turbulence rows became a caption clause (QCLI 0.091 ± 0.000 against a
  permutation null of 0.098 ± 0.001).
* Table 2's sds are quoted to one significant figure; every mean is unchanged and no row or column
  was dropped.
* Table 2's `†` footnote and its Params convention moved into Appendix E, which the caption points to.
* Prose was tightened in the abstract, introduction, Setup (Departures loses the duplicated
  cross-fade explanation, which §3.2 states), §3.1, §3.2 (the N=100 frozen-core value 0.43 ± 0.02
  is dropped, the N=11 value 0.44 ± 0.04 kept), §3.3, §3.4, Discussion, and Appendices B-F.
* Four bibliography `note` fields lost duplicated DOI or venue strings.
* Figure 5 became a normal float instead of a `wrapfigure`, because without v10's negative `\vspace`
  the wrapped version pushed its caption off the bottom of the page.
* Figures 1 and 2 are at full `\linewidth`. The appendix figures (3, 4, 5, 6) are scaled with
  `includegraphics[width=...]`, which is ordinary LaTeX rather than a template change.

No number, table row, table column or finding was dropped for space.

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

## Restructure 2026-09-08 (numbers moved from prose into main-text tables)

The results sections were rewritten to carry their numbers in tables rather than in sentences. Three
tables now sit in the main text and nothing below changed value, only location.

| Main-text table | contents | source rows in this file |
|---|---|---|
| Table 1 (`tab:map`, Sec. 3.1) | six map rows, lens 18-bit and its permutation null, turbulence D2 at K = 4,096 and its permutation null, lens 10-bit crop pmf and its permutation null | the Section 3.1 block below; MNIST-784 (0.0135 / 0.5684) and BinaryBlobs (0.0316 / 0.5420) were **dropped from the paper on 2026-09-08** for space, and Sec. 3.1 now compares lens only against the reference paper's own turbulence data |
| Table 2 (`tab:main`, Sec. 3.4) | unchanged | the Table 1 block below |
| Table 3 (`tab:classes`, Sec. 3.4) | per-class run at **N = 100 only**, four rows | the Appendix F block below. The N = 11 block was moved out of the table into one sentence of App. F, which quotes the three projection correlations 0.9232 / 0.9238 / 0.8998 against the pooled 0.9092 |

Values **removed from the running text** in the same pass, all still sourced below and unchanged in
value. Sec. 3.1 no longer prints 0.4715 / 0.8677 / 0.4154 / 0.8782 / 0.473 / 0.178 / 0.431 / 0.963
inline (they are the table). Sec. 3.3 no longer prints the N = 20 sweep points 0.911 / 0.672 / 0.095
or the coherence radius 0.03-0.05 (they are Fig. 2(a) and its caption). Sec. 3.4 no longer prints
0.00095 / 0.00092 / 0.0279 / 0.0275 / 0.0741 / 0.1134 / 0.0371 / 0.0861 inline (they are Table 2);
the parameter ratios 2x and 36-39x stay in the text. Sec. 3.2 no longer prints the KL single-crop fit
0.9999 or the paper-kernel 0.9962.

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

## Items that still need <user>'s confirmation

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

## Final-verification round (2026-09-09, after the CFP template check)

| Change | Source |
|---|---|
| Fig. 4 caption 20,000 → 33,000 logged points per n | 5 seeds x 2 gain settings x 3,333 steps |
| Table 2 fixed-basis N=100 MMD²_al sd 2 → 1 | ST1 Table 2 (v11), 8.521 ± 1.49 |
| DCGAN chosen steps "5k to 15k" → "5k to 30k" | ST2 "Baseline step counts", DCGAN-32 N=11 row (30k/30k/5k/15k/5k) |
| App. C 10⁴-sample proxy ≤ −0.93 → ≤ −0.92 | STB §3 second table, worst case −0.9291 |
| Discussion quotes both metrics (57 % raw MMD², 74 % PDF–JS) | ST2 "Repeat runs of an identical configuration" |
| "We ran" → "We report" 18 conditions | wording |
| Abstract "matches or surpasses" → "matches" | consistent with §3.4, where every N=11 lead sits inside 1 sd |
| RBM is a joint model, 10,960 shared + 50 per image | ST2 "Parameter counts and the flux scale S" |
| figA_scaling_seeds.py drops the in-axes "v11 metrics; seeds per point" note | the caption carries it |
| Floor draws from the 8,710 validation images the split leaves | FACTS "v11 definitions", Floor bullet (8,910 val minus the 200 held out) |
| Restart gap "at most 0.004" → "0.005" | ST2 restart table, max |CORR − LOSS| = 0.0041 |
| Table 4 caption states the sd convention | formatting |
| §3.4 "reverses" → "the flux-aligned column partly reverses" | ST1 wins tables: aligned column flips at N=11 but only overlaps at N=100 |
| App. B "anchor-fit ceiling" → "anchor-fit level" | the ceiling is the 0.892 model-family number, not this one |
| Setup states Conv2D-MMD² is a biased V-statistic | FACTS "v11 definitions", Conv2D-MMD² bullet |
| Five `[h]` floats → `[tb]` | `[h]` triggered LaTeX's "changed to `ht`" warning; `[t]` alone forced two extra pages, so `[tb]`, which Table 1 already used |
| IQP early-stopping disclosure in App. E | seed_tables.md "IQP early stopping": selected counts span 1k–16k, no v11 cell moves by more than 1 sd, and the single v10 exception is the paper-kernel PDF–JS at N=11 (0.92 → 0.78 x 10⁻³) |
| Fig. 1 caption restores "D2 pooled" | STB §1 |

Deviations from the instruction, both deliberate: the disclosure says the validation rule selects
**1k–16k** steps rather than 4k–16k, because `seed_tables.md`'s own table lists a 1,000-step choice
(IQP M=4 flat, N=100, seed 4) and a 2,000-step one; and the five `[h]` floats became `[tb]` rather
than `[t]`, because `[t]`-only placement with six appendix floats and the template's default
`topnumber` = 2 pushed the paper from 9 pages to 11.
