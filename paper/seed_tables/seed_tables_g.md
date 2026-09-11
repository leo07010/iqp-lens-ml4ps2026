# Seed tables - part G: the v12 tables, under both flux conventions

Mean +- sample sd (ddof = 1) over seeds 0-4. Metrics, held-out set (the same 200
frames), flux-alignment target and Conv2D encoder seed are `evaluate_v11.py`'s and are
unchanged. Lower is better in every metric column.

The two blocks of every table differ ONLY in how the per-image total intensity is set
when a generated pmf is decoded into a frame:

| convention | rule |
|---|---|
| **fixed S** | `S_i` = mean of the N training crop totals, for every image -- what every published IQP / MLP / RBM row uses. |
| **resampled S** | `S_i` drawn from a Gaussian KDE of the same N totals, Silverman bandwidth `h = 0.9 min(sd, IQR/1.34) N^(-1/5)`, clipped positive. |

No model was retrained for the second block. Each arm's generator emits a normalised
pmf and multiplies it by S; `evaluate_v12_flux.py` recovers the pmf exactly from the
run's saved `crops/`, re-fluxes it and re-pastes it at the run's own saved offsets. Its
`--check` asserts that the fixed-S pass reproduces the published frames -- measured
`max |saved - rebuilt| = 0` over every run in this file. DCGAN is the one arm with no
fixed S to swap (it learns a per-image flux), so its published row stands in both
blocks and is marked *(learns own flux)*.

> **The aligned column cannot move.** `MMD2_al` rescales every generated frame to the
> held-out mean total before scoring, which divides the per-image flux back out. So the
> flux convention is invisible to it *by construction*, not approximately: over the 325
> paired runs in this file the largest |resampled - fixed| difference in `MMD2_al` is
> `6.44e-09`. The aligned block is repeated below anyway so each table can be read on its
> own, but a reader should not read it as a second, independent win.

## 1. Table 2 (v12)

### N = 11

| Model | Params | PDF-JS (fixed S) | MMD2 (fixed S) | MMD2_al (fixed S) | PDF-JS (resampled S) | MMD2 (resampled S) | MMD2_al (resampled S) | seeds |
|---|---|---|---|---|---|---|---|---|
| IQP, paper protocol (Alg. 1) | 1,024 | 0.007199 +- 0.0006771 | 0.1883 +- 0.0103 | 0.2251 +- 0.02187 | 0.00655 +- 0.0007681 | 0.1477 +- 0.01552 | 0.2251 +- 0.02187 | 5 |
| IQP, fixed basis, joint | 1,024 | 0.001629 +- 0.0004399 | 0.1227 +- 0.003883 | 0.1253 +- 0.01326 | 0.001773 +- 0.0003711 | 0.08733 +- 0.01363 | 0.1253 +- 0.01326 | 5 |
| IQP, learned dir. M=4 (flat MMD) | 5,164 | 0.002979 +- 0.0001767 | 0.08914 +- 0.002443 | 0.02737 +- 0.002194 | 0.002834 +- 0.0001348 | 0.05643 +- 0.01939 | 0.02737 +- 0.002194 | 5 |
| IQP, learned dir. M=4 (paper kernel) | 5,164 | 0.0008062 +- 5.176e-05 | 0.09632 +- 0.002696 | 0.07432 +- 0.006323 | 0.001034 +- 4.555e-05 | 0.06266 +- 0.01543 | 0.07432 +- 0.006323 | 5 |
| IQP, MLP map M=2 (flat MMD) | 33,910 | 0.0009595 +- 9.236e-05 | 0.08508 +- 0.001855 | **0.02247 +- 0.001811** | 0.001176 +- 7.025e-05 | 0.05153 +- 0.01762 | **0.02247 +- 0.001811** | 5 |
| IQP, MLP map M=2 (paper kernel) | 33,910 | **0.0004921 +- 6.911e-05** | 0.08747 +- 0.001736 | 0.0514 +- 0.003457 | **0.0007499 +- 6.542e-05** | 0.05363 +- 0.01547 | 0.0514 +- 0.003457 | 5 |
| RBM | 11,010 | 0.001669 +- 0.0008203 | 0.06537 +- 0.008124 | 0.1106 +- 0.03881 | 0.001887 +- 0.0009503 | 0.07866 +- 0.01932 | 0.1106 +- 0.03881 | 5 |
| DCGAN *(learns own flux)* | 199,837 | 0.001106 +- 0.0005576 | 0.05094 +- 0.01514 | 0.03883 +- 0.0104 | 0.001106 +- 0.0005576 | 0.05094 +- 0.01514 | 0.03883 +- 0.0104 | 5 |
| Held-out 200-vs-200 floor | -- | 4.079e-05 +- 2.057e-05 | 0.004806 +- 0.002783 | 0.003979 +- 0.00221 | 4.079e-05 +- 2.057e-05 | 0.004806 +- 0.002783 | 0.003979 +- 0.00221 | 20 |

**bold** = the best mean in that column whose lead over the runner-up exceeds the
sum of the two seed sds. A column with no bold cell has no win the seeds support.

#### N = 11 -- wins the seed spread does not support

| convention | metric | best arm (mean) | overlaps it within 1 sd |
|---|---|---|---|
| fixed S | PDF-JS | IQP, MLP map M=2 (paper kernel) (0.0004921 +- 6.911e-05) | DCGAN *(learns own flux)* |
| fixed S | MMD2 | DCGAN *(learns own flux)* (0.05094 +- 0.01514) | RBM |
| fixed S | MMD2_al | IQP, MLP map M=2 (flat MMD) (0.02247 +- 0.001811) | **none -- the win holds** |
| resampled S | PDF-JS | IQP, MLP map M=2 (paper kernel) (0.0007499 +- 6.542e-05) | DCGAN *(learns own flux)* |
| resampled S | MMD2 | DCGAN *(learns own flux)* (0.05094 +- 0.01514) | IQP, MLP map M=2 (flat MMD), IQP, MLP map M=2 (paper kernel), IQP, learned dir. M=4 (flat MMD), IQP, learned dir. M=4 (paper kernel), RBM |
| resampled S | MMD2_al | IQP, MLP map M=2 (flat MMD) (0.02247 +- 0.001811) | **none -- the win holds** |

#### N = 11 -- which arms beat BOTH DCGAN and RBM on all three columns

An arm passes only if its mean is better than **both** baselines on PDF-JS, raw
MMD2 and MMD2_al. `> base sd` = every one of the six gaps exceeds the baseline's own
seed sd (Table 2 (v11)'s rule); `> both sds` = every gap also exceeds the arm's own
sd. Margins are in units of the baseline sd, smallest of the six shown.

| convention | arm | beats both on all 3? | > base sd | > both sds | smallest margin (base sd) |
|---|---|---|---|---|---|
| fixed S | IQP, paper protocol (Alg. 1) | no | no | no | -17.91 |
| fixed S | IQP, fixed basis, joint | no | no | no | -8.31 |
| fixed S | IQP, learned dir. M=4 (flat MMD) | no | no | no | -3.36 |
| fixed S | IQP, learned dir. M=4 (paper kernel) | no | no | no | -3.81 |
| fixed S | IQP, MLP map M=2 (flat MMD) | no | no | no | -2.43 |
| fixed S | IQP, MLP map M=2 (paper kernel) | no | no | no | -2.72 |
| resampled S | IQP, paper protocol (Alg. 1) | no | no | no | -17.91 |
| resampled S | IQP, fixed basis, joint | no | no | no | -8.31 |
| resampled S | IQP, learned dir. M=4 (flat MMD) | no | no | no | -3.10 |
| resampled S | IQP, learned dir. M=4 (paper kernel) | no | no | no | -3.41 |
| resampled S | IQP, MLP map M=2 (flat MMD) | no | no | no | -0.13 |
| resampled S | IQP, MLP map M=2 (paper kernel) | no | no | no | -1.21 |

### N = 100

| Model | Params | PDF-JS (fixed S) | MMD2 (fixed S) | MMD2_al (fixed S) | PDF-JS (resampled S) | MMD2 (resampled S) | MMD2_al (resampled S) | seeds |
|---|---|---|---|---|---|---|---|---|
| IQP, paper protocol (Alg. 1) | 1,024 | 0.008383 +- 0.0004439 | 0.2176 +- 0.006072 | 0.2659 +- 0.006753 | 0.007687 +- 0.0004039 | 0.1465 +- 0.003232 | 0.2659 +- 0.006753 | 5 |
| IQP, fixed basis, joint | 1,024 | 0.001042 +- 0.0001092 | 0.102 +- 0.007668 | 0.08521 +- 0.0149 | 0.001113 +- 0.000154 | 0.05381 +- 0.003639 | 0.08521 +- 0.0149 | 5 |
| IQP, learned dir. M=4 (flat MMD) | 5,520 | 0.002912 +- 0.0002788 | 0.07155 +- 0.002939 | 0.02551 +- 0.004413 | 0.0026 +- 0.0002413 | 0.02752 +- 0.005375 | 0.02551 +- 0.004413 | 5 |
| IQP, learned dir. M=4 (paper kernel) | 5,520 | 0.001363 +- 0.0001801 | 0.1132 +- 0.004813 | 0.09991 +- 0.007786 | 0.001462 +- 0.000179 | 0.05894 +- 0.003623 | 0.09991 +- 0.007786 | 5 |
| IQP, MLP map M=2 (flat MMD) | 34,088 | 0.001902 +- 0.0002152 | 0.06844 +- 0.001485 | 0.01812 +- 0.00171 | 0.001705 +- 0.0001888 | 0.02421 +- 0.00496 | 0.01812 +- 0.00171 | 5 |
| IQP, MLP map M=2 (paper kernel) | 34,088 | 0.0008594 +- 7.899e-05 | 0.0978 +- 0.003107 | 0.06607 +- 0.005518 | 0.0009874 +- 3.601e-05 | 0.04215 +- 0.004002 | 0.06607 +- 0.005518 | 5 |
| RBM | 11,010 | 0.0009171 +- 0.000132 | 0.05865 +- 0.004363 | 0.08067 +- 0.02205 | 0.001105 +- 0.0001995 | 0.05105 +- 0.007834 | 0.08067 +- 0.02205 | 5 |
| DCGAN *(learns own flux)* | 199,837 | **0.0005534 +- 0.0001211** | **0.02225 +- 0.003118** | 0.02122 +- 0.004081 | **0.0005534 +- 0.0001211** | 0.02225 +- 0.003118 | 0.02122 +- 0.004081 | 5 |
| Held-out 200-vs-200 floor | -- | 4.079e-05 +- 2.057e-05 | 0.004806 +- 0.002783 | 0.003979 +- 0.00221 | 4.079e-05 +- 2.057e-05 | 0.004806 +- 0.002783 | 0.003979 +- 0.00221 | 20 |

**bold** = the best mean in that column whose lead over the runner-up exceeds the
sum of the two seed sds. A column with no bold cell has no win the seeds support.

#### N = 100 -- wins the seed spread does not support

| convention | metric | best arm (mean) | overlaps it within 1 sd |
|---|---|---|---|
| fixed S | PDF-JS | DCGAN *(learns own flux)* (0.0005534 +- 0.0001211) | **none -- the win holds** |
| fixed S | MMD2 | DCGAN *(learns own flux)* (0.02225 +- 0.003118) | **none -- the win holds** |
| fixed S | MMD2_al | IQP, MLP map M=2 (flat MMD) (0.01812 +- 0.00171) | DCGAN *(learns own flux)* |
| resampled S | PDF-JS | DCGAN *(learns own flux)* (0.0005534 +- 0.0001211) | **none -- the win holds** |
| resampled S | MMD2 | DCGAN *(learns own flux)* (0.02225 +- 0.003118) | IQP, MLP map M=2 (flat MMD), IQP, learned dir. M=4 (flat MMD) |
| resampled S | MMD2_al | IQP, MLP map M=2 (flat MMD) (0.01812 +- 0.00171) | DCGAN *(learns own flux)* |

#### N = 100 -- which arms beat BOTH DCGAN and RBM on all three columns

An arm passes only if its mean is better than **both** baselines on PDF-JS, raw
MMD2 and MMD2_al. `> base sd` = every one of the six gaps exceeds the baseline's own
seed sd (Table 2 (v11)'s rule); `> both sds` = every gap also exceeds the arm's own
sd. Margins are in units of the baseline sd, smallest of the six shown.

| convention | arm | beats both on all 3? | > base sd | > both sds | smallest margin (base sd) |
|---|---|---|---|---|---|
| fixed S | IQP, paper protocol (Alg. 1) | no | no | no | -64.66 |
| fixed S | IQP, fixed basis, joint | no | no | no | -25.57 |
| fixed S | IQP, learned dir. M=4 (flat MMD) | no | no | no | -19.48 |
| fixed S | IQP, learned dir. M=4 (paper kernel) | no | no | no | -29.16 |
| fixed S | IQP, MLP map M=2 (flat MMD) | no | no | no | -14.82 |
| fixed S | IQP, MLP map M=2 (paper kernel) | no | no | no | -24.23 |
| resampled S | IQP, paper protocol (Alg. 1) | no | no | no | -59.95 |
| resampled S | IQP, fixed basis, joint | no | no | no | -15.68 |
| resampled S | IQP, learned dir. M=4 (flat MMD) | no | no | no | -16.90 |
| resampled S | IQP, learned dir. M=4 (paper kernel) | no | no | no | -19.28 |
| resampled S | IQP, MLP map M=2 (flat MMD) | no | no | no | -9.51 |
| resampled S | IQP, MLP map M=2 (paper kernel) | no | no | no | -10.99 |

### 1c. What the flux swap does, seed by seed

The two conventions are evaluated on the SAME trained run, so the comparison is paired
and the sd of the per-seed DIFFERENCE is the informative number -- not the sd of either
column on its own. `delta` is (resampled - fixed), so negative is an improvement.

| arm | N | metric | fixed S | resampled S | delta (mean +- sd of the paired diff) |
|---|---|---|---|---|---|
| IQP, paper protocol (Alg. 1) | 11 | PDF-JS | 0.007199 +- 0.0006771 | 0.00655 +- 0.0007681 | -0.000649 +- 0.0002581 |
| IQP, paper protocol (Alg. 1) | 11 | MMD2 | 0.1883 +- 0.0103 | 0.1477 +- 0.01552 | -0.04055 +- 0.0158 |
| IQP, paper protocol (Alg. 1) | 100 | PDF-JS | 0.008383 +- 0.0004439 | 0.007687 +- 0.0004039 | -0.0006957 +- 0.0001128 |
| IQP, paper protocol (Alg. 1) | 100 | MMD2 | 0.2176 +- 0.006072 | 0.1465 +- 0.003232 | -0.07109 +- 0.004069 |
| | | | | | |
| IQP, fixed basis, joint | 11 | PDF-JS | 0.001629 +- 0.0004399 | 0.001773 +- 0.0003711 | 0.0001443 +- 8.567e-05 |
| IQP, fixed basis, joint | 11 | MMD2 | 0.1227 +- 0.003883 | 0.08733 +- 0.01363 | -0.0354 +- 0.01223 |
| IQP, fixed basis, joint | 100 | PDF-JS | 0.001042 +- 0.0001092 | 0.001113 +- 0.000154 | 7.082e-05 +- 5.959e-05 |
| IQP, fixed basis, joint | 100 | MMD2 | 0.102 +- 0.007668 | 0.05381 +- 0.003639 | -0.04815 +- 0.005091 |
| | | | | | |
| IQP, learned dir. M=4 (flat MMD) | 11 | PDF-JS | 0.002979 +- 0.0001767 | 0.002834 +- 0.0001348 | -0.0001457 +- 5.589e-05 |
| IQP, learned dir. M=4 (flat MMD) | 11 | MMD2 | 0.08914 +- 0.002443 | 0.05643 +- 0.01939 | -0.0327 +- 0.01954 |
| IQP, learned dir. M=4 (flat MMD) | 100 | PDF-JS | 0.002912 +- 0.0002788 | 0.0026 +- 0.0002413 | -0.000312 +- 0.0001042 |
| IQP, learned dir. M=4 (flat MMD) | 100 | MMD2 | 0.07155 +- 0.002939 | 0.02752 +- 0.005375 | -0.04403 +- 0.006494 |
| | | | | | |
| IQP, learned dir. M=4 (paper kernel) | 11 | PDF-JS | 0.0008062 +- 5.176e-05 | 0.001034 +- 4.555e-05 | 0.0002274 +- 1.76e-05 |
| IQP, learned dir. M=4 (paper kernel) | 11 | MMD2 | 0.09632 +- 0.002696 | 0.06266 +- 0.01543 | -0.03366 +- 0.01512 |
| IQP, learned dir. M=4 (paper kernel) | 100 | PDF-JS | 0.001363 +- 0.0001801 | 0.001462 +- 0.000179 | 9.851e-05 +- 3.907e-05 |
| IQP, learned dir. M=4 (paper kernel) | 100 | MMD2 | 0.1132 +- 0.004813 | 0.05894 +- 0.003623 | -0.05421 +- 0.005039 |
| | | | | | |
| IQP, MLP map M=2 (flat MMD) | 11 | PDF-JS | 0.0009595 +- 9.236e-05 | 0.001176 +- 7.025e-05 | 0.0002163 +- 6.273e-05 |
| IQP, MLP map M=2 (flat MMD) | 11 | MMD2 | 0.08508 +- 0.001855 | 0.05153 +- 0.01762 | -0.03355 +- 0.01755 |
| IQP, MLP map M=2 (flat MMD) | 100 | PDF-JS | 0.001902 +- 0.0002152 | 0.001705 +- 0.0001888 | -0.000197 +- 7.944e-05 |
| IQP, MLP map M=2 (flat MMD) | 100 | MMD2 | 0.06844 +- 0.001485 | 0.02421 +- 0.00496 | -0.04423 +- 0.005803 |
| | | | | | |
| IQP, MLP map M=2 (paper kernel) | 11 | PDF-JS | 0.0004921 +- 6.911e-05 | 0.0007499 +- 6.542e-05 | 0.0002579 +- 4.393e-05 |
| IQP, MLP map M=2 (paper kernel) | 11 | MMD2 | 0.08747 +- 0.001736 | 0.05363 +- 0.01547 | -0.03384 +- 0.01574 |
| IQP, MLP map M=2 (paper kernel) | 100 | PDF-JS | 0.0008594 +- 7.899e-05 | 0.0009874 +- 3.601e-05 | 0.0001279 +- 6.284e-05 |
| IQP, MLP map M=2 (paper kernel) | 100 | MMD2 | 0.0978 +- 0.003107 | 0.04215 +- 0.004002 | -0.05565 +- 0.003804 |
| | | | | | |
| RBM | 11 | PDF-JS | 0.001669 +- 0.0008203 | 0.001887 +- 0.0009503 | 0.0002182 +- 0.0001414 |
| RBM | 11 | MMD2 | 0.06537 +- 0.008124 | 0.07866 +- 0.01932 | 0.0133 +- 0.01696 |
| RBM | 100 | PDF-JS | 0.0009171 +- 0.000132 | 0.001105 +- 0.0001995 | 0.000188 +- 0.0001183 |
| RBM | 100 | MMD2 | 0.05865 +- 0.004363 | 0.05105 +- 0.007834 | -0.007603 +- 0.007703 |
| | | | | | |

> **The swap is not neutral between arms.** At N = 11 it improves every IQP/MLP arm's
> raw MMD$^2$ by 29-37% and makes the RBM's *worse* by 20%; at N = 100 it improves the
> RBM too (-13%). So part of the MLP arm's apparent gain over the RBM at N = 11 comes
> from the baseline moving down, not the arm moving up. The convention is applied
> identically to all of them -- the same KDE of the same training totals -- but a reader
> comparing arms across conventions should look at the paired deltas above, not only at
> the resampled column.

### 1d. Why the raw column stops where it does

The flux swap can only fix the SHAPE of the generated flux distribution. It cannot fix
its location, because the location is set by which images landed in the training split:

| N | training totals (mean +- sd) | held-out totals (mean +- sd) | Silverman h | sampled flux sd | part E's 0.2 x NN-spacing h | its sampled flux sd |
|---|---|---|---|---|---|---|
| 11 | 112.34 +- 26.96 | 123.20 +- 27.80 | 14.64 | 29.58 | 0.366 | 25.71 |
| 100 | 124.92 +- 28.91 | 123.20 +- 27.80 | 10.36 | 30.57 | 0.054 | 28.77 |

At N = 11 the eleven training frames are 9% dimmer on average than the held-out set
(112.3 vs 123.2). Resampling reproduces the training spread, so it removes the
*zero-variance* artefact of a single S but keeps that 9% offset -- which is why the raw
column lands next to DCGAN rather than near the floor, and why the swap costs seed
stability (each run now carries a 200-draw Monte-Carlo flux sample: the raw MMD2 sd of
the MLP M=2 arm at N = 11 goes from 0.0019 to 0.0176). The Silverman bandwidth
over-widens the sampled flux by ~6% at N = 11; part E's 0.2 x nearest-neighbour rule
under-widens it by ~8%. Neither error is the reason the raw column stops where it does.

### 1e. The validation-selected checkpoint, for symmetry with the baselines

Table 2 above reports the MLP arms at a FIXED 4,000 joint steps. The classical
baselines are validation-early-stopped, so for symmetry the MLP arms were also run on a
16,000-step schedule with snapshots at {1k, 2k, 4k, 8k, 16k} and one picked per seed on
the validation block (`seeds/select_checkpoint.py`). Those picked runs are re-fluxed
here the same way. If the selected rows moved the story, the fixed-4k convention would
be the wrong one to publish; they do not.

| arm | N | steps | PDF-JS (fixed S) | MMD2 (fixed S) | MMD2_al | PDF-JS (resampled S) | MMD2 (resampled S) | seeds |
|---|---|---|---|---|---|---|---|---|
| MLP M=2 (flat MMD) | 11 | 4,000 (fixed) | 0.0009595 +- 9.236e-05 | 0.08508 +- 0.001855 | 0.02247 +- 0.001811 | 0.001176 +- 7.025e-05 | 0.05153 +- 0.01762 | 5 |
| MLP M=2 (flat MMD) | 11 | validation-selected: 4000, 4000, 4000, 8000, 16000 | 0.0007825 +- 0.0001195 | 0.08488 +- 0.001612 | 0.02232 +- 0.001686 | 0.001009 +- 0.0001379 | 0.05126 +- 0.01754 | 5 |
| MLP M=2 (flat MMD) | 100 | 4,000 (fixed) | 0.001902 +- 0.0002152 | 0.06844 +- 0.001485 | 0.01812 +- 0.00171 | 0.001705 +- 0.0001888 | 0.02421 +- 0.00496 | 5 |
| MLP M=2 (flat MMD) | 100 | validation-selected: 4000, 8000, 16000, 16000, 16000 | 0.001553 +- 8.548e-05 | 0.06836 +- 0.001348 | 0.01729 +- 0.001358 | 0.001401 +- 6.475e-05 | 0.02375 +- 0.004702 | 5 |
| | | | | | | | | |
| MLP M=2 (paper kernel) | 11 | 4,000 (fixed) | 0.0004921 +- 6.911e-05 | 0.08747 +- 0.001736 | 0.0514 +- 0.003457 | 0.0007499 +- 6.542e-05 | 0.05363 +- 0.01547 | 5 |
| MLP M=2 (paper kernel) | 11 | validation-selected: 2000, 4000, 4000, 8000, 16000 | 0.000461 +- 2.712e-05 | 0.0868 +- 0.003487 | 0.04768 +- 0.009313 | 0.0007141 +- 6.818e-05 | 0.05316 +- 0.0164 | 5 |
| MLP M=2 (paper kernel) | 100 | 4,000 (fixed) | 0.0008594 +- 7.899e-05 | 0.0978 +- 0.003107 | 0.06607 +- 0.005518 | 0.0009874 +- 3.601e-05 | 0.04215 +- 0.004002 | 5 |
| MLP M=2 (paper kernel) | 100 | validation-selected: 8000, 8000, 16000, 16000, 16000 | 0.0006384 +- 3.386e-05 | 0.09058 +- 0.003249 | 0.05146 +- 0.005301 | 0.0007814 +- 5.442e-05 | 0.03636 +- 0.004475 | 5 |
| | | | | | | | | |

## 2. Latent-width sweep (M), MLP map, H = 32, flat MMD -- resampled S

| M | N | Params | PDF-JS | MMD2 | MMD2_al | seeds | MMD2 (fixed S, for reference) |
|---|---|---|---|---|---|---|---|
| 1 | 11 | 33,867 | 0.001128 +- 6.546e-05 | 0.05169 +- 0.01756 | 0.02366 +- 0.004649 | 5 | 0.08593 +- 0.002553 |
| 2 | 11 | 33,910 | 0.001176 +- 7.025e-05 | 0.05153 +- 0.01762 | 0.02247 +- 0.001811 | 5 | 0.08508 +- 0.001855 |
| 3 | 11 | 33,953 | 0.001184 +- 9.29e-05 | 0.05304 +- 0.01767 | 0.02398 +- 0.002142 | 5 | 0.0854 +- 0.001762 |
| 4 | 11 | 33,996 | 0.001168 +- 3.865e-05 | 0.05359 +- 0.01909 | 0.02494 +- 0.0019 | 5 | 0.08663 +- 0.002438 |
| 6 | 11 | 34,082 | 0.001432 +- 0.0004445 | 0.05551 +- 0.01608 | 0.02837 +- 0.007931 | 5 | 0.08758 +- 0.002824 |
| 8 | 11 | 34,168 | 0.00296 +- 0.00128 | 0.06867 +- 0.02279 | 0.05034 +- 0.02831 | 5 | 0.09973 +- 0.01412 |
| 16 | 11 | 34,512 | 0.00767 +- 0.0009074 | 0.1239 +- 0.02505 | 0.1732 +- 0.02507 | 5 | 0.1621 +- 0.01314 |
| | | | | | | | |
| 1 | 100 | 33,956 | 0.00127 +- 9.108e-05 | 0.0233 +- 0.004082 | 0.01876 +- 0.002453 | 5 | 0.06938 +- 0.0014 |
| 2 | 100 | 34,088 | 0.001705 +- 0.0001888 | 0.02421 +- 0.00496 | 0.01812 +- 0.00171 | 5 | 0.06844 +- 0.001485 |
| 3 | 100 | 34,220 | 0.001669 +- 0.0001375 | 0.02569 +- 0.003669 | 0.01884 +- 0.004644 | 5 | 0.06924 +- 0.003094 |
| 4 | 100 | 34,352 | 0.001623 +- 9.664e-05 | 0.02477 +- 0.004506 | 0.0193 +- 0.003701 | 5 | 0.06998 +- 0.003063 |
| 6 | 100 | 34,616 | 0.001769 +- 5.693e-05 | 0.0263 +- 0.003793 | 0.02049 +- 0.003698 | 5 | 0.0706 +- 0.002103 |
| 8 | 100 | 34,880 | 0.002535 +- 0.00029 | 0.02462 +- 0.004246 | 0.02068 +- 0.002966 | 5 | 0.06842 +- 0.000413 |
| 16 | 100 | 35,936 | 0.008814 +- 0.0001383 | 0.133 +- 0.002981 | 0.2451 +- 0.008245 | 5 | 0.2083 +- 0.007475 |
| | | | | | | | |

## 3. Data scaling -- resampled S

`MLP` = `IQP-mlpH32-M2-MMDflat-resample`, `affine` = `IQP-affine-M4-MMDflat-resample`.
DCGAN is quoted from Table 2 (v11) / the published scaling curve: it learns its own
per-image flux, so the resampled-S column does not apply to it.

| N | MLP PDF-JS | MLP MMD2 | MLP MMD2_al | affine MMD2 | affine MMD2_al | RBM MMD2 | RBM MMD2_al |
|---|---|---|---|---|---|---|---|
| 10 | 0.001231 +- 0.0001054 | 0.06171 +- 0.01554 | 0.02599 +- 0.004446 | 0.06396 +- 0.01635 | 0.02712 +- 0.003066 | 0.09117 +- 0.01745 | 0.07722 +- 0.01446 |
| 30 | 0.001388 +- 9.34e-05 | 0.02846 +- 0.003613 | 0.02279 +- 0.002938 | 0.02906 +- 0.004492 | 0.02584 +- 0.004968 | 0.06454 +- 0.01035 | 0.103 +- 0.0218 |
| 50 | 0.001639 +- 0.0001136 | 0.02533 +- 0.005668 | 0.01982 +- 0.005044 | 0.02502 +- 0.004788 | 0.02479 +- 0.002751 | 0.06373 +- 0.01114 | 0.1149 +- 0.03428 |
| 70 | 0.001703 +- 0.0001385 | 0.02421 +- 0.003993 | 0.01605 +- 0.001457 | 0.02614 +- 0.001442 | 0.02262 +- 0.002206 | 0.07126 +- 0.01578 | 0.1272 +- 0.02807 |
| 90 | 0.00176 +- 7.307e-05 | 0.02954 +- 0.005053 | 0.0188 +- 0.002804 | 0.02856 +- 0.005917 | 0.02296 +- 0.001042 | 0.05012 +- 0.0105 | 0.08166 +- 0.01382 |
| 100 | 0.001705 +- 0.0001888 | 0.02421 +- 0.00496 | 0.01812 +- 0.00171 | 0.02752 +- 0.005375 | 0.02551 +- 0.004413 | 0.05105 +- 0.007834 | 0.08067 +- 0.02205 |

The same three arms under the published fixed S, for the delta:

| N | MLP MMD2 | affine MMD2 | RBM MMD2 |
|---|---|---|---|
| 10 | 0.09159 +- 0.002582 | 0.0952 +- 0.001527 | 0.07649 +- 0.01655 |
| 30 | 0.0673 +- 0.001954 | 0.06775 +- 0.001577 | 0.06108 +- 0.005951 |
| 50 | 0.06636 +- 0.003434 | 0.06793 +- 0.00183 | 0.06269 +- 0.008308 |
| 70 | 0.06979 +- 0.001353 | 0.07242 +- 0.0008608 | 0.06742 +- 0.007281 |
| 90 | 0.06977 +- 0.003077 | 0.06985 +- 0.003029 | 0.05469 +- 0.002378 |
| 100 | 0.06844 +- 0.001485 | 0.07155 +- 0.002939 | 0.05865 +- 0.004363 |

## 4. Per class -- resampled S

Each class is scored against ITS OWN 200 held-out frames and its own 200-vs-200
real-vs-real floor, so the columns are comparable down a class but not across classes.

| class | N | MLP PDF-JS | MLP MMD2 | MLP MMD2_al | affine PDF-JS | affine MMD2 | affine MMD2_al |
|---|---|---|---|---|---|---|---|
| axion | 11 | 0.001101 +- 0.0001014 | 0.04576 +- 0.006455 | 0.0295 +- 0.003794 | 0.003366 +- 0.000197 | 0.04727 +- 0.007288 | 0.02823 +- 0.005261 |
| axion | 100 | 0.001826 +- 0.000172 | 0.02861 +- 0.001861 | 0.0166 +- 0.002033 | 0.002574 +- 0.0001051 | 0.0274 +- 0.002164 | 0.02081 +- 0.002261 |
| *axion floor (200 vs 200)* | -- | 3.165e-05 +- 1.277e-05 | 0.003599 +- 0.001722 | 0.003068 +- 0.00162 | -- | -- | -- |
| | | | | | | | |
| cdm | 11 | 0.0009426 +- 4.954e-05 | 0.04418 +- 0.002353 | 0.02655 +- 0.003228 | 0.00242 +- 9.354e-05 | 0.04282 +- 0.003407 | 0.02569 +- 0.004515 |
| cdm | 100 | 0.001707 +- 6.082e-05 | 0.02139 +- 0.00301 | 0.01462 +- 0.003116 | 0.002651 +- 0.0002463 | 0.02319 +- 0.004482 | 0.01835 +- 0.002811 |
| *cdm floor (200 vs 200)* | -- | 3.41e-05 +- 1.308e-05 | 0.003024 +- 0.001663 | 0.003166 +- 0.001323 | -- | -- | -- |
| | | | | | | | |
| no_sub | 11 | 0.0009809 +- 9.981e-05 | 0.03141 +- 0.008833 | 0.02423 +- 0.005823 | 0.002603 +- 0.0002249 | 0.03489 +- 0.007929 | 0.02606 +- 0.00103 |
| no_sub | 100 | 0.001701 +- 9.525e-05 | 0.02347 +- 0.00417 | 0.02058 +- 0.002953 | 0.002687 +- 0.0003076 | 0.02734 +- 0.00409 | 0.02939 +- 0.002721 |
| *no_sub floor (200 vs 200)* | -- | 3.519e-05 +- 1.378e-05 | 0.004462 +- 0.002691 | 0.004463 +- 0.002394 | -- | -- | -- |
| | | | | | | | |

## 5. Ablation summary

### 5a. IQP-side levers (part E, `seed_tables_e.md`, N = 11, fixed 4,000 steps)

Each lever is read off a PAIR of v12 arms that differ in that one setting, both at
seeds 0-4, so the number is a within-configuration delta rather than a rank in a table.

| lever | pair | MMD2 | MMD2_al | verdict |
|---|---|---|---|---|
| C: flux resampled at generation | `M4-multi-resample` -> `M4-multi-fluxres-resample` | 0.0894 -> 0.05316 (-41%) | 0.02825 -> 0.02825 (+0%) | SEPARATED |
| C: flux resampled (M=2) | `M2-multi-resample` -> `M2-multi-fluxres-resample` | 0.09299 -> 0.06138 (-34%) | 0.03279 -> 0.03279 (-0%) | SEPARATED |
| C: flux resampled (MLP H=8) | `mlpH8-M4-MMDflat-resample` -> `mlpH8-M4-MMDflat-fluxres-resample` | 0.0881 -> 0.05208 (-41%) | 0.0282 -> 0.0282 (-0%) | SEPARATED |
| B: Walsh low-pass mu=0.01, k0=3 | `M4-multi-fluxres-resample` -> `M4-multi-mu0.01-k3-fluxres-resample` | 0.05316 -> 0.05525 (+4%) | 0.02825 -> 0.02773 (-2%) | within 1 sd |
| encoding: Gray code | `M4-MMDflat-fluxres-resample` -> `gray10-M4-MMDflat-fluxres-resample` | 0.05278 -> 0.05288 (+0%) | 0.02737 -> 0.02793 (+2%) | within 1 sd |
| A: spatial kernel (+flat) | `M4-MMDflat-fluxres-resample` -> `M4-spatial+flat-fluxres-resample` | 0.05278 -> 0.0508 (-4%) | 0.02737 -> 0.02921 (+7%) | within 1 sd |

Reading: only lever C moves the raw column, and it moves nothing else -- the aligned
column is invariant to it by construction (see the note at the top). The Walsh low-pass
and the Gray-code re-ordering are null within the seed spread; the spatial kernel is a
trade, not a gain.

### 5a-bis. Does the post-hoc swap reproduce part E's RETRAINED arms?

Part E paid GPU time to retrain eight configurations with `--flux resample`. Each of
those has a twin trained with the fixed S. Re-fluxing the fixed-S twin here must land
on the retrained arm's numbers -- it does, within the seed spread, for all eight. This
is the end-to-end evidence that the post-hoc swap is the same operation, not an
approximation of it.

| configuration (N = 11) | fixed S | this file's KDE re-flux | part E's retrained arm |
|---|---|---|---|
| `M2-multi-mu0.01-k3-resample` | 0.09452 +- 0.00123 | 0.06121 +- 0.01936 | 0.06299 +- 0.01385 |
| `M2-multi-resample` | 0.09299 +- 0.001268 | 0.05946 +- 0.01939 | 0.06138 +- 0.01315 |
| `M4-multi-mu0.01-k3-resample` | 0.09184 +- 0.002708 | 0.05874 +- 0.01992 | 0.05525 +- 0.008336 |
| `M4-multi-resample` | 0.0894 +- 0.003041 | 0.05706 +- 0.01961 | 0.05316 +- 0.007259 |
| `M4-spatial+flat-resample` | 0.08702 +- 0.002806 | 0.05478 +- 0.01868 | 0.0508 +- 0.007389 |
| `gray10-M4-MMDflat-resample` | 0.08936 +- 0.002276 | 0.05637 +- 0.01921 | 0.05288 +- 0.007737 |
| `lowres8-M2-spatial+flat-resample` | 0.1332 +- 0.002492 | 0.08169 +- 0.0252 | 0.08747 +- 0.01728 |
| `mlpH8-M4-MMDflat-resample` | 0.0881 +- 0.003253 | 0.05586 +- 0.01909 | 0.05208 +- 0.008108 |

(raw Conv2D-MMD$^2$, mean +- sd over seeds 0-4. The re-fluxed sd is the wider of the
two because the Silverman bandwidth is wider than the trainer's 0.2 x NN-spacing one;
the means agree to well inside 1 sd in every row.)

### 5b. MLP-side ablations (part F section 6, N = 11 and 100, fixed S)

Recomputed from `results_seeds_mlp_suite.jsonl` rather than copied out of
`seed_tables_f.md`, so these cells cannot drift from that file's source.

| ablation | N | PDF-JS | MMD2 | MMD2_al | proj corr | vs baseline (MMD2_al) |
|---|---|---|---|---|---|---|
| baseline (8 restarts, KL warm start, pick by corr.) | 11 | 0.0009595 +- 9.236e-05 | 0.08508 +- 0.001855 | 0.02247 +- 0.001811 | 0.9016 +- 0.001305 | -- |
| restart R1 (1 restart) | 11 | 0.0009595 +- 9.236e-05 | 0.08508 +- 0.001855 | 0.02247 +- 0.001811 | 0.8381 +- 0.002979 | within 1 sd |
| restart R2 (2 restarts) | 11 | 0.0009595 +- 9.236e-05 | 0.08508 +- 0.001855 | 0.02247 +- 0.001811 | 0.901 +- 0.001314 | within 1 sd |
| restart picked by loss, not correlation | 11 | 0.0009595 +- 9.236e-05 | 0.08508 +- 0.001855 | 0.02247 +- 0.001811 | 0.9013 +- 0.001178 | within 1 sd |
| theta0 warm start on MMD, not KL | 11 | 0.0009484 +- 3.324e-05 | 0.08499 +- 0.001308 | 0.0231 +- 0.002219 | 0.9022 +- 0.001075 | within 1 sd |
| | | | | | | |
| baseline (8 restarts, KL warm start, pick by corr.) | 100 | 0.001902 +- 0.0002152 | 0.06844 +- 0.001485 | 0.01812 +- 0.00171 | 0.9773 +- 0.0006473 | -- |
| restart R1 (1 restart) | 100 | 0.001902 +- 0.0002152 | 0.06844 +- 0.001485 | 0.01812 +- 0.00171 | 0.8871 +- 0.01539 | within 1 sd |
| restart R2 (2 restarts) | 100 | 0.001902 +- 0.0002152 | 0.06844 +- 0.001485 | 0.01812 +- 0.00171 | 0.9772 +- 0.0007208 | within 1 sd |
| restart picked by loss, not correlation | 100 | 0.001902 +- 0.0002152 | 0.06844 +- 0.001485 | 0.01812 +- 0.00171 | 0.9773 +- 0.0006483 | within 1 sd |
| theta0 warm start on MMD, not KL | 100 | 0.001844 +- 6.061e-05 | 0.06872 +- 0.001554 | 0.01775 +- 0.001388 | 0.9773 +- 0.0004157 | within 1 sd |
| | | | | | | |

## 6. What changed from v11 to v12

Left column: the published Table 2 (v11) cell, parsed out of `seed_tables.md`. Middle:
the same arm here under the fixed S -- it must agree, and any disagreement is a bug in
this pipeline, not a result. Right: the same arm under the resampled S.

| Model | N | metric | v11 published | v12 fixed S | v12 resampled S | change (res. vs v11) |
|---|---|---|---|---|---|---|
| IQP, paper protocol (Alg. 1) | 11 | PDF-JS | 0.007199 +- 0.0006771 | 0.007199 +- 0.0006771 | 0.00655 +- 0.0007681 | -9% |
| IQP, paper protocol (Alg. 1) | 11 | MMD2 | 0.1883 +- 0.0103 | 0.1883 +- 0.0103 | 0.1477 +- 0.01552 | -22% |
| IQP, paper protocol (Alg. 1) | 11 | MMD2_al | 0.2251 +- 0.02187 | 0.2251 +- 0.02187 | 0.2251 +- 0.02187 | -0% |
| IQP, paper protocol (Alg. 1) | 100 | PDF-JS | 0.008383 +- 0.0004439 | 0.008383 +- 0.0004439 | 0.007687 +- 0.0004039 | -8% |
| IQP, paper protocol (Alg. 1) | 100 | MMD2 | 0.2176 +- 0.006072 | 0.2176 +- 0.006072 | 0.1465 +- 0.003232 | -33% |
| IQP, paper protocol (Alg. 1) | 100 | MMD2_al | 0.2659 +- 0.006753 | 0.2659 +- 0.006753 | 0.2659 +- 0.006753 | -0% |
| | | | | | | |
| IQP, fixed basis, joint | 11 | PDF-JS | 0.001629 +- 0.0004399 | 0.001629 +- 0.0004399 | 0.001773 +- 0.0003711 | +9% |
| IQP, fixed basis, joint | 11 | MMD2 | 0.1227 +- 0.003883 | 0.1227 +- 0.003883 | 0.08733 +- 0.01363 | -29% |
| IQP, fixed basis, joint | 11 | MMD2_al | 0.1253 +- 0.01326 | 0.1253 +- 0.01326 | 0.1253 +- 0.01326 | -0% |
| IQP, fixed basis, joint | 100 | PDF-JS | 0.001042 +- 0.0001092 | 0.001042 +- 0.0001092 | 0.001113 +- 0.000154 | +7% |
| IQP, fixed basis, joint | 100 | MMD2 | 0.102 +- 0.007668 | 0.102 +- 0.007668 | 0.05381 +- 0.003639 | -47% |
| IQP, fixed basis, joint | 100 | MMD2_al | 0.08521 +- 0.0149 | 0.08521 +- 0.0149 | 0.08521 +- 0.0149 | +0% |
| | | | | | | |
| IQP, learned dir. M=4 (flat MMD) | 11 | PDF-JS | 0.002979 +- 0.0001767 | 0.002979 +- 0.0001767 | 0.002834 +- 0.0001348 | -5% |
| IQP, learned dir. M=4 (flat MMD) | 11 | MMD2 | 0.08914 +- 0.002443 | 0.08914 +- 0.002443 | 0.05643 +- 0.01939 | -37% |
| IQP, learned dir. M=4 (flat MMD) | 11 | MMD2_al | 0.02737 +- 0.002194 | 0.02737 +- 0.002194 | 0.02737 +- 0.002194 | +0% |
| IQP, learned dir. M=4 (flat MMD) | 100 | PDF-JS | 0.002912 +- 0.0002788 | 0.002912 +- 0.0002788 | 0.0026 +- 0.0002413 | -11% |
| IQP, learned dir. M=4 (flat MMD) | 100 | MMD2 | 0.07155 +- 0.002939 | 0.07155 +- 0.002939 | 0.02752 +- 0.005375 | -62% |
| IQP, learned dir. M=4 (flat MMD) | 100 | MMD2_al | 0.02551 +- 0.004413 | 0.02551 +- 0.004413 | 0.02551 +- 0.004413 | -0% |
| | | | | | | |
| IQP, learned dir. M=4 (paper kernel) | 11 | PDF-JS | 0.0008062 +- 5.176e-05 | 0.0008062 +- 5.176e-05 | 0.001034 +- 4.555e-05 | +28% |
| IQP, learned dir. M=4 (paper kernel) | 11 | MMD2 | 0.09632 +- 0.002696 | 0.09632 +- 0.002696 | 0.06266 +- 0.01543 | -35% |
| IQP, learned dir. M=4 (paper kernel) | 11 | MMD2_al | 0.07432 +- 0.006323 | 0.07432 +- 0.006323 | 0.07432 +- 0.006323 | +0% |
| IQP, learned dir. M=4 (paper kernel) | 100 | PDF-JS | 0.001363 +- 0.0001801 | 0.001363 +- 0.0001801 | 0.001462 +- 0.000179 | +7% |
| IQP, learned dir. M=4 (paper kernel) | 100 | MMD2 | 0.1132 +- 0.004813 | 0.1132 +- 0.004813 | 0.05894 +- 0.003623 | -48% |
| IQP, learned dir. M=4 (paper kernel) | 100 | MMD2_al | 0.09991 +- 0.007786 | 0.09991 +- 0.007786 | 0.09991 +- 0.007786 | +0% |
| | | | | | | |
| RBM | 11 | PDF-JS | 0.001669 +- 0.0008203 | 0.001669 +- 0.0008203 | 0.001887 +- 0.0009503 | +13% |
| RBM | 11 | MMD2 | 0.06537 +- 0.008124 | 0.06537 +- 0.008124 | 0.07866 +- 0.01932 | +20% |
| RBM | 11 | MMD2_al | 0.1106 +- 0.03881 | 0.1106 +- 0.03881 | 0.1106 +- 0.03881 | +0% |
| RBM | 100 | PDF-JS | 0.0009171 +- 0.000132 | 0.0009171 +- 0.000132 | 0.001105 +- 0.0001995 | +20% |
| RBM | 100 | MMD2 | 0.05865 +- 0.004363 | 0.05865 +- 0.004363 | 0.05105 +- 0.007834 | -13% |
| RBM | 100 | MMD2_al | 0.08067 +- 0.02205 | 0.08067 +- 0.02205 | 0.08067 +- 0.02205 | -0% |
| | | | | | | |

Largest relative disagreement between a v11 published cell and this file's fixed-S
recomputation of it: **0.00042**. `seed_tables.md` prints its cells to four significant
figures, and that rounding alone allows up to ~5e-4 relative -- so this is display
rounding, not drift. The underlying frames are byte-identical (`evaluate_v12_flux.py
--check` reports `max |saved - rebuilt| = 0` over all 325 runs).

New in v12 relative to Table 2 (v11):

1. Two more IQP rows -- the MLP latent map at M = 2 under both objectives (part F).
2. Every IQP / MLP / RBM row is now quoted under both flux conventions.
3. The 16k-step schedule with validation-selected checkpoints exists for the MLP arms
   (part F section 3) and for the best v12 IQP arm (`seed_tables_e.md`); the main table
   still reports the fixed 4,000 steps, which is the harder convention for the IQP side
   because the classical baselines ARE validation-early-stopped.

## 7. Provenance

| what | where |
|---|---|
| both flux conventions, every run | `experiments/13_iqp_turbulence_replication/results_v12_fluxres.jsonl` |
| the re-scorer | `experiments/13_iqp_turbulence_replication/evaluate_v12_flux.py` |
| its SLURM wrapper | `experiments/13_iqp_turbulence_replication/seeds/submit_v12_fluxres.sbatch` |
| DCGAN, and the v11 published cells | `results_seeds_table2_v11.jsonl`, `seed_tables.md` |
| MLP ablations | `results_seeds_mlp_suite.jsonl` |
| part E levers | `results_seeds_v12.jsonl`, `seed_tables_e.md` |
| floors | `results_v11_floor.jsonl` |
| this file | `paper/ml4ps2026/make_seed_tables_g.py` |

650 flux rows over 325 run directories.
