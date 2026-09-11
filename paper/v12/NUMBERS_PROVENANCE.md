# v12 numbers provenance

Every number printed in `v12/main.tex`, with the file, section and row it comes from.
Source tables live in `paper/ml4ps2026/` one level up. "part G" = `seed_tables_g.md`,
"part F" = `seed_tables_f.md`, "part E" = `seed_tables_e.md`, "v11" = `seed_tables.md`
and `seed_tables_b.md` / `seed_tables_c.md`.

Units in the paper: PDF-JS in 1e-3, both MMD2 columns in 1e-2. Sources print raw values.

---

## Abstract

| number | source |
|---|---|
| correlation 0.977 (held-out projection, MLP M=2, N=100) | part G section 5b, row `restart picked by loss, not correlation`, N=100, `proj corr` 0.9773 +- 0.0006483 |
| "matches a DCGAN of six times as many parameters" | 199,837 / 33,910 = 5.9, part G section 1, N=11 params column |
| "leads it on a flux-aligned variant, leads an RBM on all three" | part G section 1, N=11 resampled-S block (see section 3.3 rows below) |
| "past M=6 directions" | mech_test2 `results_seeds.json`, `M_sweep_holdout_corr`, M=6 -> 0.7378 |

## Section 2 (Setup)

| number | source |
|---|---|
| 1,024 Walsh phases, n=10 | encoding, `data_lens_r3` / App. A |
| block of 50 latent phases | reference paper Algorithm 1 |
| H = 32 hidden width | part F section 2, chosen where the H curve flattens |
| fixed 4,000 joint steps | part F header (protocol frozen to v11 / part D) |
| 200 validation, 200 held-out, 200 generated | `evaluate_v11.py` protocol, unchanged from v11 |
| 99.4% of the light in the 32x32 crop | v11 `NUMBERS_PROVENANCE.md`, encoding section (unchanged) |

## Table 1 (correlation-complexity map)

Unchanged from v11 Table 1. Source `seed_tables_c.md` / v11 `NUMBERS_PROVENANCE.md`.

| row | QCLI | CCI |
|---|---|---|
| Lens 18-bit tuples | 0.4715 | 0.8677 |
| its permutation null | 0.4154 | 0.8783 |
| Turbulence D2 18-bit | 0.0232 | 0.9285 |
| its permutation null | 0.0114 | 0.9371 |
| Lens 10-bit crop pmf | 0.4731 | 0.4313 |
| its permutation null | 0.1782 | 0.9634 |
| turbulence QCLI at K = 5e4 | 0.091 against null 0.098 |

## Section 3.2 (latent map)

| number | source |
|---|---|
| single-crop fit 0.99997 +- 0.00001 | v11, unchanged |
| fixed block reconstructs others at 0.44 +- 0.04 | v11, unchanged |
| affine M=2 0.9060 +- 0.0021 | `mech_tests/mech_test2/results_seeds.json`, `M_sweep_holdout_corr["2"]` |
| affine M=6 0.7378 +- 0.0672 | same file, `["6"]` |
| affine M=8 0.2184 +- 0.1213 | same file, `["8"]` |
| nearest-training-image 0.9339 | same file, `references.nearest` |
| MLP 0.9368 to 0.9519 over M = 1..16 | tabulated in part F section 1b; `results_seeds_mlp_suite.jsonl`, n_train = 20, arms `IQP-mlpH32-M{1,2,3,4,6,8,16}-MMDflat-resample`, run-dir `projection_corr_mean`, means 0.9381 / 0.9429 / 0.9463 / 0.9519 / 0.9486 / 0.9397 / 0.9368 |
| MLP M=2 projection 0.9013 +- 0.0012 (N=11), 0.9773 +- 0.0006 (N=100) | part G section 5b, `restart picked by loss, not correlation` rows |

## Table 2 (main, resampled S) -- part G section 1, "resampled S" columns

N = 11 (paper units: PDF-JS x1e-3, MMD2 x1e-2)

| row | params | PDF-JS | MMD2 | MMD2_al | source cell (raw) |
|---|---|---|---|---|---|
| IQP reference protocol (Alg. 1) | 1,524 / 5,974 | 6.5 +- 0.8 | 14.8 +- 1.6 | 22.5 +- 2.2 | 0.00655 +- 0.0007681 / 0.1477 +- 0.01552 / 0.2251 +- 0.02187 |
| IQP fixed basis, joint | 1,524 / 5,974 | 1.8 +- 0.4 | 8.7 +- 1.4 | 12.5 +- 1.3 | 0.001773 +- 0.0003711 / 0.08733 +- 0.01363 / 0.1253 +- 0.01326 |
| IQP linear M=4 (flat MMD) | 5,164 / 5,520 | 2.8 +- 0.1 | 5.6 +- 1.9 | 2.7 +- 0.2 | 0.002834 +- 0.0001348 / 0.05643 +- 0.01939 / 0.02737 +- 0.002194 |
| IQP linear M=4 (paper kernel) | 5,164 / 5,520 | 1.0 +- 0.05 | 6.3 +- 1.5 | 7.4 +- 0.6 | 0.001034 +- 4.555e-05 / 0.06266 +- 0.01543 / 0.07432 +- 0.006323 |
| IQP non-linear M=2 (flat MMD) | 33,910 / 34,088 | 1.2 +- 0.07 | 5.2 +- 1.8 | **2.2 +- 0.2** | 0.001176 +- 7.025e-05 / 0.05153 +- 0.01762 / 0.02247 +- 0.001811 |
| IQP non-linear M=2 (paper kernel) | 33,910 / 34,088 | 0.75 +- 0.07 | 5.4 +- 1.5 | 5.1 +- 0.3 | 0.0007499 +- 6.542e-05 / 0.05363 +- 0.01547 / 0.0514 +- 0.003457 |
| RBM | 11,510 / 15,960 | 1.9 +- 1.0 | 7.9 +- 1.9 | 11.1 +- 3.9 | 0.001887 +- 0.0009503 / 0.07866 +- 0.01932 / 0.1106 +- 0.03881 |
| DCGAN | 199,837 | 1.1 +- 0.6 | 5.1 +- 1.5 | 3.9 +- 1.0 | 0.001106 +- 0.0005576 / 0.05094 +- 0.01514 / 0.03883 +- 0.0104 |
| floor | n/a | 0.04 +- 0.02 | 0.5 +- 0.3 | 0.4 +- 0.2 | 4.079e-05 / 0.004806 / 0.003979 |

N = 100

| row | PDF-JS | MMD2 | MMD2_al | source cell (raw) |
|---|---|---|---|---|
| IQP reference protocol | 7.7 +- 0.4 | 14.7 +- 0.3 | 26.6 +- 0.7 | 0.007687 +- 0.0004039 / 0.1465 +- 0.003232 / 0.2659 +- 0.006753 |
| IQP fixed basis, joint | 1.1 +- 0.2 | 5.4 +- 0.4 | 8.5 +- 1.5 | 0.001113 +- 0.000154 / 0.05381 +- 0.003639 / 0.08521 +- 0.0149 |
| IQP linear M=4 (flat MMD) | 2.6 +- 0.2 | 2.8 +- 0.5 | 2.6 +- 0.4 | 0.0026 +- 0.0002413 / 0.02752 +- 0.005375 / 0.02551 +- 0.004413 |
| IQP linear M=4 (paper kernel) | 1.5 +- 0.2 | 5.9 +- 0.4 | 10.0 +- 0.8 | 0.001462 +- 0.000179 / 0.05894 +- 0.003623 / 0.09991 +- 0.007786 |
| IQP non-linear M=2 (flat MMD) | 1.7 +- 0.2 | 2.4 +- 0.5 | 1.8 +- 0.2 | 0.001705 +- 0.0001888 / 0.02421 +- 0.00496 / 0.01812 +- 0.00171 |
| IQP non-linear M=2 (paper kernel) | 0.99 +- 0.04 | 4.2 +- 0.4 | 6.6 +- 0.6 | 0.0009874 +- 3.601e-05 / 0.04215 +- 0.004002 / 0.06607 +- 0.005518 |
| RBM | 1.1 +- 0.2 | 5.1 +- 0.8 | 8.1 +- 2.2 | 0.001105 +- 0.0001995 / 0.05105 +- 0.007834 / 0.08067 +- 0.02205 |
| DCGAN | **0.55 +- 0.12** | 2.2 +- 0.3 | 2.1 +- 0.4 | 0.0005534 +- 0.0001211 / 0.02225 +- 0.003118 / 0.02122 +- 0.004081 |

Bold cells are exactly the ones part G section 1 marks as wins the seed spread supports
("wins the seed spread does not support" tables, rows reading "none -- the win holds"):
N=11 resampled MMD2_al (non-linear M=2 flat) and N=100 resampled PDF-JS (DCGAN). Table 3 adds
N=100 fixed-S MMD2 (DCGAN) on the same criterion.

The caption states the rule as "a best mean leading **every other arm** by more than both sds",
which is the criterion part G's overlap tables actually apply. Part G's own raw table bolds
one further cell (MLP M=2 paper kernel, N=11 PDF-JS, both conventions) under a looser
runner-up-only reading; that cell is NOT bold here because the DCGAN (0.001106 +- 0.0005576)
overlaps it, exactly as part G's "wins the seed spread does not support" table records and as
Section 3.3 states in words ("which leaves the two inside one sd"). Recomputed independently:
under the every-other-arm rule the bold set is N=11 MMD2_al and N=100 PDF-JS under the
resampled S, plus N=100 MMD2 under the fixed S, and nothing else.

Parameter counts: IQP rows and the reference-protocol / RBM / DCGAN conventions are v11's
(v11 `NUMBERS_PROVENANCE.md`, "Table 2 conventions"); the MLP rows are part F section 1 and
part G section 1 (33,910 at N=11, 34,088 at N=100).

## Section 3.3 sd margins

Computed from the part G section 1 resampled-S cells above, in units of the baseline's own
seed sd (`(baseline mean - arm mean) / baseline sd`, positive = the arm leads):

| comparison | PDF-JS | MMD2 | MMD2_al |
|---|---|---|---|
| non-linear M=2 flat vs DCGAN, N=11 | -0.13 | -0.04 | +1.57 |
| non-linear M=2 flat vs RBM, N=11 | +0.75 | +1.40 | +2.27 |

Other quoted comparisons:
- paper-kernel arm PDF-JS 0.75e-3 against DCGAN 1.11 +- 0.56 e-3, gap 0.36e-3 < 0.56e-3, hence "inside one sd" (part G section 1, "wins the seed spread does not support", N=11 resampled PDF-JS row lists DCGAN as overlapping).
- "No arm beats the DCGAN on all three columns by more than a seed sd at either size": part G section 1, "which arms beat BOTH DCGAN and RBM on all three columns", every row `no`.
- N=100 flux-aligned 1.8 +- 0.2 vs DCGAN 2.1 +- 0.4: part G section 1 N=100 overlap table, MMD2_al row lists DCGAN as overlapping.

## Section 3.3 caveats

| number | source |
|---|---|
| largest paired MMD2_al difference 6e-9 | part G header note, "the largest \|resampled - fixed\| difference in MMD2_al is 6.44e-09" |
| 325 runs | part G section 7, "650 flux rows over 325 run directories" |
| raw MMD2 improves 22 to 39% at N=11 | part G section 1c table, recomputed per arm from the fixed-S and resampled-S means: -21.6% (reference protocol), -28.8% (fixed basis), -36.7%, -34.9%, -39.4%, -38.7%. Part G section 1c prose says "29-37%"; the table it prints supports 22-39%, which is what the paper quotes |
| RBM 20% worse at N=11 | part G section 1c, RBM N=11 MMD2 delta +0.0133 on 0.06537 |
| RBM 13% better at N=100 | part G section 1c, RBM N=100 MMD2 delta -0.007603 on 0.05865 |
| training frames 9% dimmer (112.3 vs 123.2) | part G section 1d table, N=11 row |

## Section 3.4 (scaling, per class)

| number | source |
|---|---|
| non-linear raw MMD2 0.062 +- 0.016 (N=10) -> 0.024 +- 0.005 (N=100) | part G section 3, MLP MMD2 column, N=10 0.06171 +- 0.01554 and N=100 0.02421 +- 0.00496 |
| RBM 0.091 +- 0.017 -> 0.051 +- 0.008 | part G section 3, RBM MMD2 column, 0.09117 +- 0.01745 and 0.05105 +- 0.007834 |
| DCGAN 0.046 +- 0.017 at N=10 | v11 scaling curve / part F section 4 |
| non-linear lowest aligned mean at every N, separated only at N=10 and N=70 | part F section 4, `MLP vs DCGAN (MMD2_al)` column |
| per class aligned 0.017 / 0.015 / 0.021 at N=100 | part G section 4, MLP MMD2_al, axion 0.0166, cdm 0.01462, no_sub 0.02058 |
| class floors 0.0031 / 0.0032 / 0.0045 | part G section 4 floor rows, MMD2_al 0.003068 / 0.003166 / 0.004463 |

## Section 4 (discussion)

| number | source |
|---|---|
| DCGAN spread up to 57% on raw Conv2D-MMD2 | v11, unchanged |
| "a sixth of the DCGAN's parameters" | 33,910 / 199,837 |

## Appendix B (fixed S)

Table 3 rows are part G section 1, "fixed S" columns, same arms as Table 2 with the two
fixed-block arms omitted. Paired deltas in the text are part G section 1c:

| arm | N=11 raw MMD2 delta (resampled - fixed) |
|---|---|
| reference protocol | -0.04055 +- 0.0158 |
| fixed basis, joint | -0.0354 +- 0.01223 |
| linear M=4 flat | -0.0327 +- 0.01954 |
| linear M=4 paper | -0.03366 +- 0.01512 |
| non-linear M=2 flat | -0.03355 +- 0.01755 |
| non-linear M=2 paper | -0.03384 +- 0.01574 |
| RBM | +0.0133 +- 0.01696 |
| RBM at N=100 | -0.007603 +- 0.007703 |

Quoted as "improves every IQP arm by 0.033 to 0.041 with a paired sd of 0.012 to 0.020".
PDF-JS paired effect max magnitude 0.0006957 (reference protocol, N=100), quoted as
"at most 7e-4". Flux-aligned paired difference below 1e-8 (part G header, 6.44e-09).

## Appendix C (Table 4)

Latent-width rows: part G section 2 (resampled S), M = 2, 4, 16 at both N.

| M | N | PDF-JS | MMD2 | MMD2_al |
|---|---|---|---|---|
| 2 | 11 | 0.001176 | 0.05153 | 0.02247 |
| 4 | 11 | 0.001168 | 0.05359 | 0.02494 |
| 16 | 11 | 0.00767 | 0.1239 | 0.1732 |
| 2 | 100 | 0.001705 | 0.02421 | 0.01812 |
| 4 | 100 | 0.001623 | 0.02477 | 0.0193 |
| 16 | 100 | 0.008814 | 0.133 | 0.2451 |

Per-class rows: part G section 4 (resampled S). Class floors quoted in the text are the
same table's floor rows. The H sweep quoted in the text (0.029 / 0.022 / 0.020 / 0.019 at
M=4, N=100) is part F section 2, `MMD2_al` column for H = 4, 8, 16, 32.

## Appendix D (turbulence)

Unchanged from v11 Appendix B, source `docs/IQP_TURBULENCE_REPLICATION_RESULTS.md` and
v11 `NUMBERS_PROVENANCE.md`.

## Appendix E

| number | source |
|---|---|
| DCGAN discriminator 53,497 parameters, RBM 10,960 + 50 per image | v11, unchanged |
| RBM 160k / 80k steps, DCGAN 5k to 30k | v11 Table 2 protocol, unchanged |
| sigma = {0.989, 1.729, 2.236} | v11, IQPopt bandwidth rule |
| Conv2D-MMD2 bandwidth 0.005547 raw, 0.002558 aligned | v11, unchanged |
| 8,710 validation images, 20 floor draws | v11, unchanged |
| Silverman h 14.64 (N=11), 10.36 (N=100) | part G section 1d table |
| validation rule on IQP arms selects 2k to 16k, no cell moves by more than one sd | part G section 1e. 1k appears only for an M=4 variant that is not in Table 2, so the paper says 2k to 16k |
| 20,000-step baselines much worse | v11 Table 3 (baselines at a fixed budget) |
| flat-MMD warm start 0.0231 +- 0.0022 against 0.0225 +- 0.0018 | part G section 5b, `theta0 warm start on MMD, not KL` vs baseline, N=11 MMD2_al |
| pick-by-loss moves the correlation by 0.0003 | part G section 5b, N=11 proj corr 0.9016 (baseline) vs 0.9013 (pick by loss) |
| two restarts match eight | part G section 5b, `restart R2` proj corr 0.9010 (N=11), 0.9772 (N=100) |
| zero-initialised restart alone 0.838 and 0.887 | part G section 5b, `restart R1 (1 restart)` proj corr 0.8381 +- 0.002979 (N=11), 0.8871 +- 0.01539 (N=100) |

## Numbers deliberately NOT in v12

- Lever attribution (flux resampling, raw MMD2 -41%, 0.0894 -> 0.05316) from part G
  section 5a. Cut for space; the same effect is reported as the paired deltas in App. B.
- v11 Appendices C and D (support mismatch, QCLI-CCI ceiling) and their figures.
- `fig_samples_v12.png` (held-out against the MLP arm under both conventions). Built and
  available in `figures/`, left out of v12 for space.

---

## Layout deviations from the v12 brief (recorded so they are not silent)

1. `figures/fig_pipeline_v2_c` is included at `0.66\textwidth`, not `\linewidth`. The three
   body figures (`fig1_compact`, `fig2_v12_msweep`, `figA_scaling_v12`) are all at
   `\linewidth`. At `\linewidth` the appendix schematic alone costs about a third of a page
   and the paper does not fit in seven.
2. `fig_samples_v12` is not included. It exists in `figures/` and can be dropped into
   App. B if a page is added.
3. Appendix A holds the pipeline figure and no body text, for the same page reason.
4. Section 3.4 is a run-in `\paragraph` inside Section 3.3 rather than its own subsection.
5. Table 3 (fixed S) omits the two fixed-block IQP rows; their paired deltas are in the
   App. B text. Table 4 shows M = 2, 4 and 16 rather than the full seven-point sweep, and
   the per-class floors are given in the App. C text rather than as table rows.
6. `fig2_v12_msweep.py` and `figA_scaling_v12.py` had their in-figure notes reworded (no
   "controlled", no "label: explanation" colon) and were regenerated; the plotted data are
   unchanged. Backups of the two scripts are `*.py.backup-20260909` in `figures/`.
   `fig1_compact` is reused from v11 unchanged and still carries the in-figure annotation
   "saturated at n=10: null = (0.99, 0.00)".

## Mechanical check run against this file's sources

A parser compared every printed cell of Tables 2, 3 and 4 in `main.tex` against
`seed_tables_g.md` sections 1, 2 and 4 (Table 2 against the resampled-S columns, Table 3
against the fixed-S columns), accepting a cell only when it is within half of its own last
printed digit. Result: 0 mismatches across all three tables. The six sd margins quoted in
Section 3.3 were recomputed from the same cells as (baseline mean - arm mean) / baseline sd
and reproduce -0.13, -0.04, +1.57 against the DCGAN and 0.75, 1.40, 2.27 against the RBM.

## Independent review, and what it changed

A fresh-context agent re-derived every table cell and every quoted number from
`seed_tables_g.md`, `seed_tables_f.md` and `seed_tables_e.md`. Outcome:

- Tables 2, 3 and 4 cell values, Table 1, the Section 3.3 sd margins, the App. B paired
  deltas, the App. C H sweep and class floors, App. D and the App. E constants all verified.
- One real defect, now fixed: the bold-rule wording in the Table 2 caption said "the
  runner-up" where the bold cells were in fact chosen by the stricter every-other-arm
  criterion. The caption now states that criterion; no bold cell changed.
- Section 2 now says the held-out projection is taken over eight restarts at the lowest fit
  loss, so the 0.9013 / 0.9773 quoted in Section 3.2 is traceable to part G section 5b's
  "restart picked by loss" row rather than to its correlation-picked baseline (0.9016).
- Reported as a defect but not one: the Section 3.4 claim that the non-linear arm leads the
  DCGAN by more than a seed sd only at N=10 and N=70 is `seed_tables_f.md` section 4's
  "MLP vs DCGAN (MMD2_al)" verdict column, which reads SEPARATED at exactly those two N.
  The reviewer compared the MLP arm with the affine arm instead.
- Also reported: the "29 to 37%" figure, already corrected to "22 to 39%" before the review
  landed (see the row above), and the Section 3.2 sweep numbers, whose sources are named in
  this file and lie outside the three files the reviewer was given.

---

## 2026-09-11 revision (three reviews applied)

Conventions and corrections.

| item | resolution |
|---|---|
| Table 2 reference-protocol PDF-JS at N=11 | raw `0.00655 +- 0.0007681` (part G section 1, resampled S) rounds to **6.5 +- 0.8**, not 6.6; the table in `main.tex` was already 6.5 and this file is now corrected |
| Params column | NOT taken from part G, whose Params column still carries the old one-image convention (e.g. RBM 11,010). App. E states the single convention used, shared parameters (theta0 included) plus N per-image latents, so RBM = 10,960 + 50N = 11,510 / 15,960, fixed-basis arms = 974 + 50N = 1,524 / 5,974, learned-direction arms = 5,120 + MN, DCGAN = generator only (199,837). Totals cross-checked against `seed_tables.md`, "Parameter counts and the flux scale S" |
| DCGAN sd in Sec. 3.3 | quoted as 1.1 +- 0.6 to match Table 2's digits (raw 0.001106 +- 0.0005576) |
| duplicate `\label{app:sweeps}` | removed (the stray second one after the scaling figure) |

New numbers introduced by this revision.

| number | where | source |
|---|---|---|
| permutation null within 14% of the lens value | Sec. 3.1 | Table 1, lens 18-bit 0.4715 vs null 0.4154 = 1.135x |
| turbulence within a factor of two of its null | Sec. 3.1 | Table 1, turbulence 18-bit 0.0232 vs null 0.0114 = 2.04x |
| 10-bit crop pmf sits at I_CCI = 0.43 | Sec. 3.1 | Table 1, row "Lens, 10-bit crop pmf", 0.4313 |
| turbulence QCLI 0.091 vs null 0.098 at N_t = 5e4 | Sec. 3.1 body (moved out of the Table 1 caption) | unchanged from v12 caption |
| nearest-training-image projection 0.892 (N=11), 0.970 (N=100) | Sec. 3.2 | `results_seeds_mlp_suite_es.jsonl`, arm `IQP-mlpH32-M2-MMDflat-resample`, field `ref_projection_nearest_train_corr` = 0.8919 / 0.9700 (identical across seeds) |
| 0.901 / 0.977 (projection, main arm) | Sec. 3.2 | same as before (0.9013 +- 0.0012, 0.9773 +- 0.0006), printed to three digits to save a line |
| RBM margins 0.75 / 1.40 / 2.27 sd | Sec. 3.3 | part G section 1, N=11 resampled S. (0.001887-0.001176)/0.0009503 = 0.75; (0.07866-0.05153)/0.01932 = 1.40; (0.1106-0.02247)/0.03881 = 2.27 |
| RBM 6.5 +- 0.8 vs non-linear 8.5 +- 0.2 on raw MMD2 under fixed S | Sec. 3.3 (H5) | part G section 1, N=11 fixed S, 0.06537 +- 0.008124 and 0.08508 +- 0.001855; the same cells already appear in Table 3 |
| nearest-training-image L2 4.21 +- 0.10 against 2.90 for real held-out frames | Sec. 4 (memorisation check) | `results_seeds_mlp_suite_es.jsonl`, `kind = mlpsuite_fixed_steps`, arm `IQP-mlpH32-M2-MMDflat-resample`, N=11, `nn_train` mean 4.2086 +- 0.1000 over seeds 0-4; the real reference is `results_seeds_table2.jsonl`, arm `HOLDOUT split-half`, N=11, `nn_train` = 2.904. Both computed on frames decoded under the fixed intensity (`evaluate_lens.py::nn_l2`) |
| Table 4 M=8 row (3.0 +- 1.28 / 6.9 +- 2.3 / 5.0 +- 2.8 and 2.5 +- 0.29 / 2.5 +- 0.4 / 2.1 +- 0.3) | App. C | part G section 2, M=8 rows at N=11 and N=100, resampled S |
| flux-aligned scaling series 0.026, 0.023, 0.020, 0.016, 0.019, 0.018 and RBM 0.077 to 0.127 | App. C (replaces the deleted scaling figure) | part G section 3, MLP MMD2_al and RBM MMD2_al columns at N = 10, 30, 50, 70, 90, 100 |
| mean \|z\| 0.101 at M=2 and 0.120 at M=8; affine residual 0.70 to 0.17 | App. C | part F section 1, H=32, N=11 rows, columns `||z||` and `nonlin` |
| RBM 10 visible, 1,000 hidden, 50 per image = b_v(10) + b_h[:40] | App. E | `baseline_rbm_sweep.py:131-132` and `baseline_rbm.py` (D_LAT = 50); shared 10x1000 + 960 = 10,960 |
| DCGAN discriminator 53,497; 128-dimensional prior; four transposed convolutions | App. E | `seed_tables.md` parameter table; `baseline_dcgan_r3.py:38-46`; run meta `nz = 128`, `ngf = ndf = 18`, `param_count = 199837` |
| paper-kernel bandwidth rule | App. E | `docs/REF38_IQPOPT_METHOD_NOTES.md` sections 6.1-6.3. sigma_1 from the mean-Pauli-weight heuristic, sigma_3 = sqrt(median heuristic), sigma_2 = sqrt((sigma_1^2+sigma_3^2)/2); check sqrt((0.989^2+2.236^2)/2) = 1.729 |

Content removed to hold 7 pages (4 + references + 2).

| removed | why it is safe |
|---|---|
| Figure 2 (latent width, `fig2_v12_msweep`) | every number it plotted is quoted in Sec. 3.2 (linear 0.906 / 0.738 / 0.218, non-linear 0.9368 to 0.9519, reference 0.9339) and panel (b) repeated Table 2's baseline levels |
| Figure 4 (data scaling, `figA_scaling_v12`) | endpoints are in Sec. 3.3 and the full six-point flux-aligned series is now written out in App. C |
| `\cite{hinton_2002_training}` and `\cite{alexander_2021_decoding}` | the RBM is trained with PCD-1 (Tieleman 2008), and DeepLense provenance is carried by `alexander_2020_deep` plus the dataset citation `ml4sci_deeplense`. Dropping two entries was needed to keep the bibliography on page 5 |
| `\cite{codsi_2025_classically}` | replaced by `maslov_2024_fast` on the citation review's finding that Codsi does not restore classical simulability; the bib entry is kept (with number and doi added) but is no longer cited |
