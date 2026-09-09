# v12 IQP-side improvements -- seed statistics

Every number is mean +- sample sd (ddof = 1) over the seeds listed in the last column, scored under the UNCHANGED v11 protocol (`experiments/13_iqp_turbulence_replication/evaluate_v11.py`, same 200 held-out images, same `CONV_SEED`). The baselines, the data split and the metrics were not touched; the rows in **bold** are copied by this script straight out of `seed_tables.md`'s Table 2 (v11) so they cannot drift.

The four levers (`train_iqp_affine_v12.py`): **A** `--objective multi` = flat MMD + lam x paper-kernel MMD with lam = L0_flat / L0_paper fixed at theta0; **B** `--mu`, `--walsh-k0` = a penalty on the Walsh energy above |s| = k0, relative to L0 and to its own value at theta0; **C** `--flux resample` = each generated image's total intensity drawn from the N training crop totals with replacement plus jitter, instead of the single training mean S; **D** `--M`. `flux=mean` + `objective=flat` + `mu=0` reproduces the published arm bit-for-bit (checked on seed 0, N = 11: max |difference| over all 200 generated images = 0).

All arms run a FIXED 4,000 joint steps, the same count the published IQP arms use; the baselines they are compared against are validation early-stopped, which if anything favours the baselines.

> **Selection caveat.** mu, k0 and lam-scale are swept here and the best value is read off the held-out score, so the best row's margin is optimistic by the usual selection amount. The mu = 0, lam-scale = 1 rows carry no such selection.

> **Projection-correlation caveat.** `proj. corr.` is computed in the arm's OWN pmf space, so the `lowres8` rows measure a 16x16 correlation and the 10-qubit rows a 32x32 one. They are not comparable across encodings; the four metric columns are (they are computed on the decoded 64x64 frames by the unchanged evaluator). `lowres8`'s encode->decode ceiling on real crops is 0.990.

## Headline: the 8 best v12 arms at N = 11 (mean rank over the three verdict columns)

| Arm | PDF-JS | MMD2 | MMD2_al | Params | beats DCGAN+RBM on all 3 by > baseline sd? |
|---|---|---|---|---|---|
| mlpH8, M=4, multi, flux=resample, 16k steps | 0.001162 +- 0.000125 | 0.0508 +- 0.006664 | 0.02689 +- 0.001429 | 9,300 | no |
| mlpH8, M=4, multi, flux=resample, selected | 0.001217 +- 0.0001823 | 0.0508 +- 0.006596 | 0.0267 +- 0.001283 | 9,300 | no |
| mlpH8, M=4, multi, flux=resample, 12k steps | 0.001181 +- 0.0001255 | 0.05083 +- 0.006615 | 0.02685 +- 0.001308 | 9,300 | no |
| mlpH8, M=4, multi, flux=resample, 8k steps | 0.001234 +- 0.0001105 | 0.05091 +- 0.006668 | 0.0268 +- 0.001214 | 9,300 | no |
| mlpH8, M=4, multi, flux=resample, 4k steps | 0.001405 +- 7.379e-05 | 0.05108 +- 0.006877 | 0.02713 +- 0.001605 | 9,300 | no |
| mlpH8, M=4, multi, flux=resample | 0.001545 +- 4.807e-05 | 0.05105 +- 0.007075 | 0.02722 +- 0.001877 | 9,300 | no |
| M=4, spatial+flatx0.25, flux=resample | 0.001104 +- 6.543e-05 | 0.05 +- 0.007085 | 0.03213 +- 0.001059 | 5,164 | no |
| mlpH8, M=2, flat, flux=resample | 0.001851 +- 0.0001456 | 0.05373 +- 0.01234 | 0.02443 +- 0.002838 | 9,262 | no |
| **DCGAN** (Table 2, v11) | 0.001106 +- 0.0005576 | 0.05094 +- 0.01514 | 0.03883 +- 0.0104 | 199,837 | -- |
| **RBM** (Table 2, v11) | 0.001669 +- 0.0008203 | 0.06537 +- 0.008124 | 0.1106 +- 0.03881 | 11,010 | -- |

## N = 5

| Arm | N | PDF-JS | MMD2 | MMD2_u | MMD2_al | proj. corr. | gen_max | clip frac | gen total (mean/sd) | Params | seeds |
|---|---|---|---|---|---|---|---|---|---|---|---|
| M=4, spatial+flatx0.25, flux=resample | 5 | 0.0005037 +- 4.704e-05 | 0.02558 +- 0.003135 | 0.02102 +- 0.00314 | 0.03058 +- 0.002525 | 0.8266 +- 0.001514 | 1.97 +- 0.0531 | 0.00112 +- 4.52e-05 | 122.9 / 26.7 | 5,140 | 5 |
| mlpH8, M=2, flat, flux=resample | 5 | 0.000642 +- 4.895e-05 | 0.0277 +- 0.002632 | 0.02305 +- 0.002577 | 0.03135 +- 0.002229 | 0.8353 +- 0.001077 | 1.9 +- 0.0901 | 0.00116 +- 0.000124 | 124.7 / 27.6 | 9,250 | 5 |
| mlpH8, M=4, flat, flux=resample | 5 | 0.0008982 +- 0.0001684 | 0.02744 +- 0.003038 | 0.02289 +- 0.003013 | 0.03287 +- 0.00325 | 0.8404 +- 0.003728 | 1.98 +- 0.177 | 0.00103 +- 7.42e-05 | 122.9 / 26.7 | 9,276 | 5 |
| mlpH8, M=4, multi, flux=resample | 5 | 0.001064 +- 0.0004344 | 0.02968 +- 0.005865 | 0.02513 +- 0.005867 | 0.03611 +- 0.006868 | 0.8255 +- 0.002385 | 2.06 +- 0.0878 | 0.00104 +- 8.81e-05 | 122.9 / 26.7 | 9,276 | 5 |
| mlpH8, M=4, spatial+flat, flux=resample | 5 | 0.000694 +- 0.0001323 | 0.02703 +- 0.003605 | 0.02248 +- 0.003614 | 0.03301 +- 0.00285 | 0.8334 +- 0.003052 | 1.94 +- 0.153 | 0.00105 +- 7.22e-05 | 122.9 / 26.7 | 9,276 | 5 |
| gray10, M=4, spatial+flatx4, flux=resample | 5 | 0.0006148 +- 5.379e-05 | 0.02584 +- 0.003298 | 0.0213 +- 0.003299 | 0.02962 +- 0.002775 | 0.8438 +- 0.001913 | 1.84 +- 0.0441 | 0.00112 +- 7.34e-05 | 122.9 / 26.7 | 5,140 | 5 |

### N = 5 -- wins that the seed spread does not support

| metric | best arm (mean) | overlaps it within 1 sd |
|---|---|---|
| PDF-JS | M=4, spatial+flatx0.25, flux=resample (0.0005037 +- 4.704e-05) | **none -- the win holds** |
| MMD2 | M=4, spatial+flatx0.25, flux=resample (0.02558 +- 0.003135) | gray10, M=4, spatial+flatx4, flux=resample, mlpH8, M=4, spatial+flat, flux=resample, mlpH8, M=4, flat, flux=resample, mlpH8, M=2, flat, flux=resample, mlpH8, M=4, multi, flux=resample |
| MMD2_u | M=4, spatial+flatx0.25, flux=resample (0.02102 +- 0.00314) | gray10, M=4, spatial+flatx4, flux=resample, mlpH8, M=4, spatial+flat, flux=resample, mlpH8, M=4, flat, flux=resample, mlpH8, M=2, flat, flux=resample, mlpH8, M=4, multi, flux=resample |
| MMD2_al | gray10, M=4, spatial+flatx4, flux=resample (0.02962 +- 0.002775) | M=4, spatial+flatx0.25, flux=resample, mlpH8, M=2, flat, flux=resample, mlpH8, M=4, flat, flux=resample, mlpH8, M=4, spatial+flat, flux=resample, mlpH8, M=4, multi, flux=resample |

### N = 5 -- the pre-registered "clear win" test

**Not applicable at N = 5:** Table 2 (v11) has no DCGAN or RBM row at this N, so the clear-win test has nothing to test against. The rows above are reported for the N-scaling trend only.

## N = 11 (headline cell)

| Arm | N | PDF-JS | MMD2 | MMD2_u | MMD2_al | proj. corr. | gen_max | clip frac | gen total (mean/sd) | Params | seeds |
|---|---|---|---|---|---|---|---|---|---|---|---|
| M=2, paper, flux=resample | 11 | 0.001475 +- 0.0001354 | 0.07061 +- 0.006698 | 0.06624 +- 0.006782 | 0.1014 +- 0.006898 | 0.7539 +- 0.008774 | 2.39 +- 0.2 | 0.00143 +- 0.000163 | 112.3 / 25.7 | 3,094 | 5 |
| M=2, flat, flux=resample | 11 | 0.004286 +- 0.0003034 | 0.06113 +- 0.01262 | 0.05689 +- 0.01271 | 0.03044 +- 0.002126 | 0.8853 +- 0.004307 | 1.96 +- 0.0536 | 0.000875 +- 0.000148 | 112.3 / 25.7 | 3,094 | 5 |
| M=2, multi, flux=mean | 11 | 0.003757 +- 0.0001553 | 0.09299 +- 0.001268 | 0.08981 +- 0.00129 | 0.03279 +- 0.00249 | 0.8718 +- 0.003948 | 1.36 +- 0.0382 | 0.000526 +- 3.51e-05 | 112.3 / 0.0 | 3,094 | 5 |
| M=2, multi, flux=resample | 11 | 0.003656 +- 0.0001556 | 0.06138 +- 0.01315 | 0.05713 +- 0.01325 | 0.03279 +- 0.00249 | 0.8718 +- 0.003948 | 2.01 +- 0.0486 | 0.00089 +- 0.000158 | 112.3 / 25.7 | 3,094 | 5 |
| M=2, multi, mu=0.01,k0=3, flux=mean | 11 | 0.003759 +- 0.0002587 | 0.09452 +- 0.00123 | 0.09136 +- 0.001265 | 0.03193 +- 0.002312 | 0.8754 +- 0.002358 | 1.35 +- 0.0328 | 0.000465 +- 3.75e-05 | 112.3 / 0.0 | 3,094 | 5 |
| M=2, multi, mu=0.01,k0=3, flux=resample | 11 | 0.003645 +- 0.000273 | 0.06299 +- 0.01385 | 0.05874 +- 0.01394 | 0.03193 +- 0.002312 | 0.8754 +- 0.002358 | 2.01 +- 0.0738 | 0.000821 +- 0.000149 | 112.3 / 25.7 | 3,094 | 5 |
| M=2, spatial+flat, flux=resample | 11 | 0.002824 +- 0.0002505 | 0.05772 +- 0.01176 | 0.05345 +- 0.01182 | 0.03231 +- 0.00295 | 0.8798 +- 0.003815 | 1.99 +- 0.0405 | 0.000909 +- 0.000149 | 112.3 / 25.7 | 3,094 | 5 |
| M=4, paper, flux=resample | 11 | 0.0009558 +- 5.896e-05 | 0.05811 +- 0.007209 | 0.05379 +- 0.007212 | 0.07432 +- 0.006323 | 0.7921 +- 0.005793 | 2.32 +- 0.202 | 0.00134 +- 0.000104 | 112.2 / 25.7 | 5,164 | 5 |
| M=4, flat, flux=resample | 11 | 0.002883 +- 0.0001122 | 0.05278 +- 0.007422 | 0.04854 +- 0.007444 | 0.02737 +- 0.002194 | 0.9102 +- 0.002598 | 2.03 +- 0.031 | 0.00105 +- 0.000135 | 112.2 / 25.7 | 5,164 | 5 |
| M=4, multi, flux=mean | 11 | 0.002588 +- 0.0001641 | 0.0894 +- 0.003041 | 0.08623 +- 0.003078 | 0.02825 +- 0.002022 | 0.8951 +- 0.001246 | 1.39 +- 0.018 | 0.000674 +- 4.59e-05 | 112.3 / 0.0 | 5,164 | 5 |
| M=4, multi, flux=resample | 11 | 0.002569 +- 0.0001623 | 0.05316 +- 0.007259 | 0.04891 +- 0.007273 | 0.02825 +- 0.002022 | 0.8951 +- 0.001246 | 2.06 +- 0.0163 | 0.00105 +- 0.000129 | 112.2 / 25.7 | 5,164 | 5 |
| M=4, multi(lam x0.25), flux=resample | 11 | 0.002898 +- 0.0001004 | 0.0528 +- 0.007499 | 0.04856 +- 0.00752 | 0.02771 +- 0.002449 | 0.9061 +- 0.002926 | 2.06 +- 0.0384 | 0.00104 +- 0.000132 | 112.2 / 25.7 | 5,164 | 5 |
| M=4, multi(lam x4), flux=resample | 11 | 0.001989 +- 4.341e-05 | 0.05305 +- 0.007774 | 0.04881 +- 0.007789 | 0.03006 +- 0.002376 | 0.8757 +- 0.008145 | 2.08 +- 0.0814 | 0.00105 +- 0.000144 | 112.2 / 25.7 | 5,164 | 5 |
| M=4, multi, mu=0.01,k0=3, flux=mean | 11 | 0.002705 +- 0.0001582 | 0.09184 +- 0.002708 | 0.08868 +- 0.002732 | 0.02773 +- 0.003231 | 0.8964 +- 0.003256 | 1.35 +- 0.0216 | 0.000595 +- 3.51e-05 | 112.3 / 0.0 | 5,164 | 5 |
| M=4, multi, mu=0.01,k0=3, flux=resample | 11 | 0.0027 +- 0.0001854 | 0.05525 +- 0.008336 | 0.05101 +- 0.008357 | 0.02773 +- 0.003231 | 0.8964 +- 0.003256 | 2 +- 0.0364 | 0.000983 +- 0.00013 | 112.2 / 25.7 | 5,164 | 5 |
| M=4, spatial+flat, flux=mean | 11 | 0.001752 +- 3.193e-05 | 0.08702 +- 0.002806 | 0.08383 +- 0.002834 | 0.02921 +- 0.002165 | 0.9034 +- 0.002508 | 1.44 +- 0.0392 | 0.000698 +- 3.69e-05 | 112.3 / 0.0 | 5,164 | 5 |
| M=4, spatial, flux=resample | 11 | 0.001017 +- 5.353e-05 | 0.05855 +- 0.00693 | 0.0541 +- 0.006955 | 0.1229 +- 0.01652 | 0.7957 +- 0.007436 | 2.78 +- 0.141 | 0.00184 +- 7.7e-05 | 112.2 / 25.7 | 5,164 | 5 |
| M=4, spatial+flatx0.25, flux=resample | 11 | 0.001104 +- 6.543e-05 | 0.05 +- 0.007085 | 0.04573 +- 0.007105 | 0.03213 +- 0.001059 | 0.8922 +- 0.00235 | 2.12 +- 0.0448 | 0.00109 +- 0.000136 | 112.2 / 25.7 | 5,164 | 5 |
| M=4, spatial+flatx16, flux=resample | 11 | 0.002881 +- 0.0001278 | 0.05276 +- 0.007545 | 0.04852 +- 0.007568 | 0.02802 +- 0.002176 | 0.9086 +- 0.002769 | 2.06 +- 0.0536 | 0.00104 +- 0.000128 | 112.2 / 25.7 | 5,164 | 5 |
| M=4, spatial+flat, flux=resample | 11 | 0.001791 +- 4.665e-05 | 0.0508 +- 0.007389 | 0.04654 +- 0.007412 | 0.02921 +- 0.002165 | 0.9034 +- 0.002508 | 2.13 +- 0.0294 | 0.00107 +- 0.000129 | 112.2 / 25.7 | 5,164 | 5 |
| M=4, spatial+flatx4, flux=resample | 11 | 0.002597 +- 0.0002022 | 0.05221 +- 0.007722 | 0.04797 +- 0.007743 | 0.02853 +- 0.002169 | 0.9069 +- 0.00172 | 2.07 +- 0.0264 | 0.00104 +- 0.000142 | 112.2 / 25.7 | 5,164 | 5 |
| M=4, spatial+flat, mu=0.01,k0=3, flux=resample | 11 | 0.001937 +- 6.892e-05 | 0.05238 +- 0.007393 | 0.04814 +- 0.00741 | 0.02793 +- 0.002301 | 0.9031 +- 0.001216 | 2.06 +- 0.0578 | 0.00102 +- 0.000139 | 112.2 / 25.7 | 5,164 | 5 |
| mlpH4, M=2, flat, flux=resample | 11 | 0.003013 +- 0.0002334 | 0.05577 +- 0.01285 | 0.05151 +- 0.01294 | 0.02557 +- 0.001913 | 0.8989 +- 0.001543 | 2.06 +- 0.0736 | 0.000977 +- 0.000157 | 112.3 / 25.7 | 5,154 | 5 |
| mlpH4, M=4, flat, flux=resample | 11 | 0.004073 +- 0.0008637 | 0.05945 +- 0.007623 | 0.05518 +- 0.007623 | 0.03851 +- 0.0098 | 0.8938 +- 0.01174 | 2.22 +- 0.145 | 0.00091 +- 0.000141 | 112.2 / 25.7 | 5,184 | 5 |
| mlpH4, M=4, spatial+flat, flux=resample | 11 | 0.002469 +- 0.0005858 | 0.05633 +- 0.006898 | 0.05206 +- 0.0069 | 0.03719 +- 0.008614 | 0.8974 +- 0.004812 | 2.14 +- 0.116 | 0.000983 +- 0.000105 | 112.2 / 25.7 | 5,184 | 5 |
| mlpH8, M=2, flat, flux=resample | 11 | 0.001851 +- 0.0001456 | 0.05373 +- 0.01234 | 0.04948 +- 0.01243 | 0.02443 +- 0.002838 | 0.902 +- 0.001252 | 2.09 +- 0.0406 | 0.00104 +- 0.000164 | 112.3 / 25.7 | 9,262 | 5 |
| mlpH8, M=4, flat, flux=mean | 11 | 0.001727 +- 0.0001421 | 0.0881 +- 0.003253 | 0.08493 +- 0.003275 | 0.0282 +- 0.002717 | 0.9087 +- 0.005225 | 1.52 +- 0.0865 | 0.000671 +- 5.24e-05 | 112.3 / 0.0 | 9,300 | 5 |
| mlpH8, M=4, flat, flux=resample | 11 | 0.001814 +- 0.0001286 | 0.05208 +- 0.008108 | 0.04783 +- 0.008127 | 0.0282 +- 0.002717 | 0.9087 +- 0.005225 | 2.2 +- 0.198 | 0.00106 +- 0.000122 | 112.2 / 25.7 | 9,300 | 5 |
| mlpH8, M=4, multi, flux=resample | 11 | 0.001545 +- 4.807e-05 | 0.05105 +- 0.007075 | 0.0468 +- 0.007097 | 0.02722 +- 0.001877 | 0.8983 +- 0.004071 | 2.18 +- 0.121 | 0.00108 +- 0.000106 | 112.2 / 25.7 | 9,300 | 5 |
| mlpH8, M=4, multi, flux=resample, 12k steps | 11 | 0.001181 +- 0.0001255 | 0.05083 +- 0.006615 | 0.04658 +- 0.006641 | 0.02685 +- 0.001308 | 0.8948 +- 0.002291 | 2.13 +- 0.0743 | 0.00108 +- 0.000104 | 112.2 / 25.7 | 9,300 | 5 |
| mlpH8, M=4, multi, flux=resample, 16k steps | 11 | 0.001162 +- 0.000125 | 0.0508 +- 0.006664 | 0.04655 +- 0.00669 | 0.02689 +- 0.001429 | 0.8945 +- 0.002404 | 2.14 +- 0.0779 | 0.00108 +- 0.000112 | 112.2 / 25.7 | 9,300 | 5 |
| mlpH8, M=4, multi, flux=resample, 4k steps | 11 | 0.001405 +- 7.379e-05 | 0.05108 +- 0.006877 | 0.04683 +- 0.006902 | 0.02713 +- 0.001605 | 0.8968 +- 0.003656 | 2.15 +- 0.1 | 0.00107 +- 0.000112 | 112.2 / 25.7 | 9,300 | 5 |
| mlpH8, M=4, multi, flux=resample, 8k steps | 11 | 0.001234 +- 0.0001105 | 0.05091 +- 0.006668 | 0.04666 +- 0.006692 | 0.0268 +- 0.001214 | 0.8956 +- 0.002925 | 2.12 +- 0.0791 | 0.00108 +- 0.000105 | 112.2 / 25.7 | 9,300 | 5 |
| mlpH8, M=4, multi, flux=resample, selected | 11 | 0.001217 +- 0.0001823 | 0.0508 +- 0.006596 | 0.04656 +- 0.006621 | 0.0267 +- 0.001283 | 0.8952 +- 0.003091 | 2.16 +- 0.0903 | 0.00107 +- 0.000111 | 112.2 / 25.7 | 9,300 | 5 |
| mlpH8, M=4, spatial+flat, flux=resample | 11 | 0.001279 +- 0.0001153 | 0.05126 +- 0.006873 | 0.047 +- 0.006893 | 0.0294 +- 0.003285 | 0.9039 +- 0.00189 | 2.21 +- 0.139 | 0.00108 +- 9.44e-05 | 112.2 / 25.7 | 9,300 | 5 |
| mlpH8, M=4, spatial+flatx4, flux=resample | 11 | 0.001692 +- 0.0001959 | 0.05282 +- 0.008853 | 0.04856 +- 0.008865 | 0.03096 +- 0.004117 | 0.9065 +- 0.002994 | 2.23 +- 0.126 | 0.00104 +- 0.000159 | 112.2 / 25.7 | 9,300 | 5 |
| gray10, M=4, flat, flux=mean | 11 | 0.003182 +- 0.0001964 | 0.08936 +- 0.002276 | 0.08621 +- 0.0023 | 0.02793 +- 0.002019 | 0.9077 +- 0.002382 | 1.38 +- 0.0327 | 0.000665 +- 4.12e-05 | 112.3 / 0.0 | 5,164 | 5 |
| gray10, M=4, flat, flux=resample | 11 | 0.003049 +- 0.0001768 | 0.05288 +- 0.007737 | 0.04864 +- 0.007761 | 0.02793 +- 0.002019 | 0.9077 +- 0.002382 | 2.06 +- 0.0255 | 0.00104 +- 0.000127 | 112.2 / 25.7 | 5,164 | 5 |
| gray10, M=4, spatial+flat, flux=resample | 11 | 0.001839 +- 7.498e-05 | 0.05078 +- 0.007629 | 0.04653 +- 0.007648 | 0.02896 +- 0.002619 | 0.9049 +- 0.0007289 | 2.13 +- 0.0797 | 0.00107 +- 0.000133 | 112.2 / 25.7 | 5,164 | 5 |
| gray10, M=4, spatial+flatx4, flux=resample | 11 | 0.002439 +- 5.572e-05 | 0.05142 +- 0.006855 | 0.04718 +- 0.006872 | 0.02779 +- 0.002185 | 0.907 +- 0.002763 | 2.06 +- 0.0142 | 0.00105 +- 0.000142 | 112.2 / 25.7 | 5,164 | 5 |
| lowres8, M=2, flat, flux=resample | 11 | 0.006394 +- 0.0004536 | 0.0926 +- 0.01785 | 0.08859 +- 0.01795 | 0.07094 +- 0.004417 | 0.8968 +- 0.002038 | 1.46 +- 0.0397 | 0.000264 +- 9.1e-05 | 112.3 / 25.7 | 790 | 5 |
| lowres8, M=2, spatial+flat, flux=mean | 11 | 0.00596 +- 0.0002315 | 0.1332 +- 0.002492 | 0.1303 +- 0.002513 | 0.06234 +- 0.004252 | 0.8935 +- 0.003185 | 1 +- 0.0102 | 1.71e-06 +- 3.18e-06 | 112.3 / 0.0 | 790 | 5 |
| lowres8, M=2, spatial+flat, flux=resample | 11 | 0.005239 +- 0.000127 | 0.08747 +- 0.01728 | 0.08343 +- 0.01738 | 0.06234 +- 0.004252 | 0.8935 +- 0.003185 | 1.51 +- 0.0302 | 0.0003 +- 0.000105 | 112.3 / 25.7 | 790 | 5 |
| lowres8, M=4, flat, flux=resample | 11 | 0.003237 +- 0.0003097 | 0.07897 +- 0.009328 | 0.07492 +- 0.009355 | 0.05719 +- 0.003709 | 0.9192 +- 0.002724 | 1.53 +- 0.0172 | 0.000422 +- 9e-05 | 112.2 / 25.7 | 1,324 | 5 |
| lowres8, M=4, spatial+flat, flux=resample | 11 | 0.002619 +- 0.0002059 | 0.07684 +- 0.009368 | 0.07279 +- 0.009398 | 0.05413 +- 0.003954 | 0.9213 +- 0.003166 | 1.54 +- 0.0154 | 0.000431 +- 8.31e-05 | 112.2 / 25.7 | 1,324 | 5 |
| lowres8, mlpH8, M=2, flat, flux=resample | 11 | 0.00167 +- 5.052e-05 | 0.08007 +- 0.01673 | 0.07603 +- 0.01684 | 0.05109 +- 0.003043 | 0.9105 +- 0.0008315 | 1.52 +- 0.0213 | 0.000355 +- 0.000118 | 112.3 / 25.7 | 2,350 | 5 |
| lowres8, mlpH8, M=2, spatial+flat, flux=resample | 11 | 0.001394 +- 0.0002079 | 0.0791 +- 0.01696 | 0.07505 +- 0.01706 | 0.04906 +- 0.002757 | 0.9091 +- 0.003457 | 1.53 +- 0.0326 | 0.000368 +- 0.000128 | 112.3 / 25.7 | 2,350 | 5 |
| lowres8, mlpH8, M=4, flat, flux=resample | 11 | 0.00197 +- 0.0002224 | 0.0777 +- 0.01042 | 0.07365 +- 0.01045 | 0.0546 +- 0.0041 | 0.9241 +- 0.005797 | 1.57 +- 0.0511 | 0.000441 +- 0.000107 | 112.2 / 25.7 | 2,388 | 5 |
| lowres8, mlpH8, M=4, spatial+flat, flux=resample | 11 | 0.001661 +- 0.0001811 | 0.07612 +- 0.009716 | 0.07206 +- 0.009748 | 0.0526 +- 0.004468 | 0.9192 +- 0.003057 | 1.62 +- 0.0718 | 0.000432 +- 9.49e-05 | 112.2 / 25.7 | 2,388 | 5 |
| **DCGAN** (Table 2, v11) | 11 | 0.001106 +- 0.0005576 | 0.05094 +- 0.01514 | 0.04704 +- 0.01527 | 0.03883 +- 0.0104 | -- | -- | -- | -- | -- | 5 |
| **RBM** (Table 2, v11) | 11 | 0.001669 +- 0.0008203 | 0.06537 +- 0.008124 | 0.06097 +- 0.008183 | 0.1106 +- 0.03881 | -- | -- | -- | -- | -- | 5 |
| **IQP, learned dir. M=4 (flat MMD)** (Table 2, v11) | 11 | 0.002979 +- 0.0001767 | 0.08914 +- 0.002443 | 0.08597 +- 0.002479 | 0.02737 +- 0.002194 | -- | -- | -- | -- | -- | 5 |
| **IQP, learned dir. M=4 (paper kernel)** (Table 2, v11) | 11 | 0.0008062 +- 5.176e-05 | 0.09632 +- 0.002696 | 0.09304 +- 0.002712 | 0.07432 +- 0.006323 | -- | -- | -- | -- | -- | 5 |

### N = 11 -- wins that the seed spread does not support

| metric | best arm (mean) | overlaps it within 1 sd |
|---|---|---|
| PDF-JS | IQP, learned dir. M=4 (paper kernel) (Table 2) (0.0008062 +- 5.176e-05) | DCGAN (Table 2), RBM (Table 2) |
| MMD2 | M=4, spatial+flatx0.25, flux=resample (0.05 +- 0.007085) | gray10, M=4, spatial+flat, flux=resample, M=4, spatial+flat, flux=resample, mlpH8, M=4, multi, flux=resample, 16k steps, mlpH8, M=4, multi, flux=resample, selected, mlpH8, M=4, multi, flux=resample, 12k steps, mlpH8, M=4, multi, flux=resample, 8k steps, DCGAN (Table 2), mlpH8, M=4, multi, flux=resample, mlpH8, M=4, multi, flux=resample, 4k steps, mlpH8, M=4, spatial+flat, flux=resample, gray10, M=4, spatial+flatx4, flux=resample, mlpH8, M=4, flat, flux=resample, M=4, spatial+flatx4, flux=resample, M=4, spatial+flat, mu=0.01,k0=3, flux=resample, M=4, spatial+flatx16, flux=resample, M=4, flat, flux=resample, M=4, multi(lam x0.25), flux=resample, mlpH8, M=4, spatial+flatx4, flux=resample, gray10, M=4, flat, flux=resample, M=4, multi(lam x4), flux=resample, M=4, multi, flux=resample, mlpH8, M=2, flat, flux=resample, M=4, multi, mu=0.01,k0=3, flux=resample, mlpH4, M=2, flat, flux=resample, mlpH4, M=4, spatial+flat, flux=resample, M=2, spatial+flat, flux=resample, M=4, paper, flux=resample, M=4, spatial, flux=resample, mlpH4, M=4, flat, flux=resample, M=2, flat, flux=resample, M=2, multi, flux=resample, M=2, multi, mu=0.01,k0=3, flux=resample |
| MMD2_u | M=4, spatial+flatx0.25, flux=resample (0.04573 +- 0.007105) | gray10, M=4, spatial+flat, flux=resample, M=4, spatial+flat, flux=resample, mlpH8, M=4, multi, flux=resample, 16k steps, mlpH8, M=4, multi, flux=resample, selected, mlpH8, M=4, multi, flux=resample, 12k steps, mlpH8, M=4, multi, flux=resample, 8k steps, mlpH8, M=4, multi, flux=resample, mlpH8, M=4, multi, flux=resample, 4k steps, mlpH8, M=4, spatial+flat, flux=resample, DCGAN (Table 2), gray10, M=4, spatial+flatx4, flux=resample, mlpH8, M=4, flat, flux=resample, M=4, spatial+flatx4, flux=resample, M=4, spatial+flat, mu=0.01,k0=3, flux=resample, M=4, spatial+flatx16, flux=resample, M=4, flat, flux=resample, M=4, multi(lam x0.25), flux=resample, mlpH8, M=4, spatial+flatx4, flux=resample, gray10, M=4, flat, flux=resample, M=4, multi(lam x4), flux=resample, M=4, multi, flux=resample, mlpH8, M=2, flat, flux=resample, M=4, multi, mu=0.01,k0=3, flux=resample, mlpH4, M=2, flat, flux=resample, mlpH4, M=4, spatial+flat, flux=resample, M=2, spatial+flat, flux=resample, M=4, paper, flux=resample, M=4, spatial, flux=resample, mlpH4, M=4, flat, flux=resample, M=2, flat, flux=resample, M=2, multi, flux=resample, M=2, multi, mu=0.01,k0=3, flux=resample, RBM (Table 2) |
| MMD2_al | mlpH8, M=2, flat, flux=resample (0.02443 +- 0.002838) | mlpH4, M=2, flat, flux=resample, mlpH8, M=4, multi, flux=resample, selected, mlpH8, M=4, multi, flux=resample, 8k steps, mlpH8, M=4, multi, flux=resample, 12k steps, mlpH8, M=4, multi, flux=resample, 16k steps, mlpH8, M=4, multi, flux=resample, 4k steps, mlpH8, M=4, multi, flux=resample, IQP, learned dir. M=4 (flat MMD) (Table 2), M=4, flat, flux=resample, M=4, multi(lam x0.25), flux=resample, M=4, multi, mu=0.01,k0=3, flux=mean, M=4, multi, mu=0.01,k0=3, flux=resample, gray10, M=4, spatial+flatx4, flux=resample, gray10, M=4, flat, flux=resample, gray10, M=4, flat, flux=mean, M=4, spatial+flat, mu=0.01,k0=3, flux=resample, M=4, spatial+flatx16, flux=resample, mlpH8, M=4, flat, flux=resample, mlpH8, M=4, flat, flux=mean, M=4, multi, flux=mean, M=4, multi, flux=resample, M=4, spatial+flatx4, flux=resample, gray10, M=4, spatial+flat, flux=resample, M=4, spatial+flat, flux=resample, M=4, spatial+flat, flux=mean, mlpH8, M=4, spatial+flat, flux=resample, mlpH8, M=4, spatial+flatx4, flux=resample |

### N = 11 -- the pre-registered "clear win" test

A combination passes only if its MEAN beats BOTH baselines on all three of PDF-JS, raw MMD$^2$ and aligned MMD$^2$ by more than the baseline's own seed sd. `gap/sd_base` is the margin in units of the baseline's sd (the literal test); `gap/sd_iqp` is the same gap in units of the IQP arm's own sd (the stricter reading). A negative number means the baseline is still ahead.

| Arm | metric | vs DCGAN gap/sd_base | vs DCGAN gap/sd_iqp | vs RBM gap/sd_base | vs RBM gap/sd_iqp |
|---|---|---|---|---|---|
| M=2, paper, flux=resample | PDF-JS | -0.66 | -2.73 | 0.24 | 1.43 |
| M=2, paper, flux=resample | MMD2 | -1.30 | -2.94 | -0.65 | -0.78 |
| M=2, paper, flux=resample | MMD2_al | -6.01 | -9.07 | 0.24 | 1.34 |
| M=2, flat, flux=resample | PDF-JS | -5.70 | -10.48 | -3.19 | -8.62 |
| M=2, flat, flux=resample | MMD2 | -0.67 | -0.81 | 0.52 | 0.34 |
| M=2, flat, flux=resample | MMD2_al | 0.81 | 3.95 | 2.07 | 37.71 |
| M=2, multi, flux=mean | PDF-JS | -4.75 | -17.07 | -2.55 | -13.45 |
| M=2, multi, flux=mean | MMD2 | -2.78 | -33.16 | -3.40 | -21.78 |
| M=2, multi, flux=mean | MMD2_al | 0.58 | 2.42 | 2.00 | 31.25 |
| M=2, multi, flux=resample | PDF-JS | -4.57 | -16.39 | -2.42 | -12.77 |
| M=2, multi, flux=resample | MMD2 | -0.69 | -0.79 | 0.49 | 0.30 |
| M=2, multi, flux=resample | MMD2_al | 0.58 | 2.42 | 2.00 | 31.25 |
| M=2, multi, mu=0.01,k0=3, flux=mean | PDF-JS | -4.76 | -10.26 | -2.55 | -8.08 |
| M=2, multi, mu=0.01,k0=3, flux=mean | MMD2 | -2.88 | -35.45 | -3.59 | -23.71 |
| M=2, multi, mu=0.01,k0=3, flux=mean | MMD2_al | 0.66 | 2.99 | 2.03 | 34.03 |
| M=2, multi, mu=0.01,k0=3, flux=resample | PDF-JS | -4.55 | -9.30 | -2.41 | -7.24 |
| M=2, multi, mu=0.01,k0=3, flux=resample | MMD2 | -0.80 | -0.87 | 0.29 | 0.17 |
| M=2, multi, mu=0.01,k0=3, flux=resample | MMD2_al | 0.66 | 2.99 | 2.03 | 34.03 |
| M=2, spatial+flat, flux=resample | PDF-JS | -3.08 | -6.86 | -1.41 | -4.61 |
| M=2, spatial+flat, flux=resample | MMD2 | -0.45 | -0.58 | 0.94 | 0.65 |
| M=2, spatial+flat, flux=resample | MMD2_al | 0.63 | 2.21 | 2.02 | 26.54 |
| M=4, paper, flux=resample | PDF-JS | 0.27 | 2.55 | 0.87 | 12.10 |
| M=4, paper, flux=resample | MMD2 | -0.47 | -0.99 | 0.89 | 1.01 |
| M=4, paper, flux=resample | MMD2_al | -3.41 | -5.61 | 0.93 | 5.74 |
| M=4, flat, flux=resample | PDF-JS | -3.19 | -15.84 | -1.48 | -10.82 |
| M=4, flat, flux=resample | MMD2 | -0.12 | -0.25 | 1.55 | 1.70 |
| M=4, flat, flux=resample | MMD2_al | 1.10 | 5.22 | 2.14 | 37.93 |
| M=4, multi, flux=mean | PDF-JS | -2.66 | -9.03 | -1.12 | -5.60 |
| M=4, multi, flux=mean | MMD2 | -2.54 | -12.65 | -2.96 | -7.90 |
| M=4, multi, flux=mean | MMD2_al | 1.02 | 5.23 | 2.12 | 40.73 |
| M=4, multi, flux=resample | PDF-JS | -2.62 | -9.01 | -1.10 | -5.54 |
| M=4, multi, flux=resample | MMD2 | -0.15 | -0.31 | 1.50 | 1.68 |
| M=4, multi, flux=resample | MMD2_al | 1.02 | 5.23 | 2.12 | 40.73 |
| M=4, multi(lam x0.25), flux=resample | PDF-JS | -3.21 | -17.84 | -1.50 | -12.23 |
| M=4, multi(lam x0.25), flux=resample | MMD2 | -0.12 | -0.25 | 1.55 | 1.68 |
| M=4, multi(lam x0.25), flux=resample | MMD2_al | 1.07 | 4.54 | 2.14 | 33.85 |
| M=4, multi(lam x4), flux=resample | PDF-JS | -1.58 | -20.34 | -0.39 | -7.37 |
| M=4, multi(lam x4), flux=resample | MMD2 | -0.14 | -0.27 | 1.52 | 1.58 |
| M=4, multi(lam x4), flux=resample | MMD2_al | 0.84 | 3.69 | 2.08 | 33.90 |
| M=4, multi, mu=0.01,k0=3, flux=mean | PDF-JS | -2.87 | -10.11 | -1.26 | -6.55 |
| M=4, multi, mu=0.01,k0=3, flux=mean | MMD2 | -2.70 | -15.10 | -3.26 | -9.77 |
| M=4, multi, mu=0.01,k0=3, flux=mean | MMD2_al | 1.07 | 3.43 | 2.14 | 25.65 |
| M=4, multi, mu=0.01,k0=3, flux=resample | PDF-JS | -2.86 | -8.60 | -1.26 | -5.56 |
| M=4, multi, mu=0.01,k0=3, flux=resample | MMD2 | -0.28 | -0.52 | 1.25 | 1.21 |
| M=4, multi, mu=0.01,k0=3, flux=resample | MMD2_al | 1.07 | 3.43 | 2.14 | 25.65 |
| M=4, spatial+flat, flux=mean | PDF-JS | -1.16 | -20.22 | -0.10 | -2.59 |
| M=4, spatial+flat, flux=mean | MMD2 | -2.38 | -12.86 | -2.66 | -7.72 |
| M=4, spatial+flat, flux=mean | MMD2_al | 0.92 | 4.44 | 2.10 | 37.59 |
| M=4, spatial, flux=resample | PDF-JS | 0.16 | 1.67 | 0.80 | 12.18 |
| M=4, spatial, flux=resample | MMD2 | -0.50 | -1.10 | 0.84 | 0.98 |
| M=4, spatial, flux=resample | MMD2_al | -8.08 | -5.09 | -0.32 | -0.74 |
| M=4, spatial+flatx0.25, flux=resample | PDF-JS | 0.00 | 0.02 | 0.69 | 8.63 |
| M=4, spatial+flatx0.25, flux=resample | MMD2 | 0.06 | 0.13 | 1.89 | 2.17 |
| M=4, spatial+flatx0.25, flux=resample | MMD2_al | 0.64 | 6.32 | 2.02 | 74.08 |
| M=4, spatial+flatx16, flux=resample | PDF-JS | -3.18 | -13.89 | -1.48 | -9.48 |
| M=4, spatial+flatx16, flux=resample | MMD2 | -0.12 | -0.24 | 1.55 | 1.67 |
| M=4, spatial+flatx16, flux=resample | MMD2_al | 1.04 | 4.97 | 2.13 | 37.96 |
| M=4, spatial+flat, flux=resample | PDF-JS | -1.23 | -14.69 | -0.15 | -2.62 |
| M=4, spatial+flat, flux=resample | MMD2 | 0.01 | 0.02 | 1.79 | 1.97 |
| M=4, spatial+flat, flux=resample | MMD2_al | 0.92 | 4.44 | 2.10 | 37.59 |
| M=4, spatial+flatx4, flux=resample | PDF-JS | -2.67 | -7.38 | -1.13 | -4.59 |
| M=4, spatial+flatx4, flux=resample | MMD2 | -0.08 | -0.16 | 1.62 | 1.70 |
| M=4, spatial+flatx4, flux=resample | MMD2_al | 0.99 | 4.75 | 2.11 | 37.84 |
| M=4, spatial+flat, mu=0.01,k0=3, flux=resample | PDF-JS | -1.49 | -12.05 | -0.33 | -3.88 |
| M=4, spatial+flat, mu=0.01,k0=3, flux=resample | MMD2 | -0.10 | -0.20 | 1.60 | 1.76 |
| M=4, spatial+flat, mu=0.01,k0=3, flux=resample | MMD2_al | 1.05 | 4.74 | 2.13 | 35.93 |
| mlpH4, M=2, flat, flux=resample | PDF-JS | -3.42 | -8.17 | -1.64 | -5.76 |
| mlpH4, M=2, flat, flux=resample | MMD2 | -0.32 | -0.38 | 1.18 | 0.75 |
| mlpH4, M=2, flat, flux=resample | MMD2_al | 1.28 | 6.93 | 2.19 | 44.46 |
| mlpH4, M=4, flat, flux=resample | PDF-JS | -5.32 | -3.43 | -2.93 | -2.78 |
| mlpH4, M=4, flat, flux=resample | MMD2 | -0.56 | -1.12 | 0.73 | 0.78 |
| mlpH4, M=4, flat, flux=resample | MMD2_al | 0.03 | 0.03 | 1.86 | 7.36 |
| mlpH4, M=4, spatial+flat, flux=resample | PDF-JS | -2.44 | -2.33 | -0.98 | -1.37 |
| mlpH4, M=4, spatial+flat, flux=resample | MMD2 | -0.36 | -0.78 | 1.11 | 1.31 |
| mlpH4, M=4, spatial+flat, flux=resample | MMD2_al | 0.16 | 0.19 | 1.89 | 8.52 |
| mlpH8, M=2, flat, flux=resample | PDF-JS | -1.34 | -5.12 | -0.22 | -1.25 |
| mlpH8, M=2, flat, flux=resample | MMD2 | -0.18 | -0.23 | 1.43 | 0.94 |
| mlpH8, M=2, flat, flux=resample | MMD2_al | 1.38 | 5.07 | 2.22 | 30.36 |
| mlpH8, M=4, flat, flux=mean | PDF-JS | -1.11 | -4.37 | -0.07 | -0.41 |
| mlpH8, M=4, flat, flux=mean | MMD2 | -2.45 | -11.42 | -2.80 | -6.99 |
| mlpH8, M=4, flat, flux=mean | MMD2_al | 1.02 | 3.91 | 2.12 | 30.33 |
| mlpH8, M=4, flat, flux=resample | PDF-JS | -1.27 | -5.51 | -0.18 | -1.13 |
| mlpH8, M=4, flat, flux=resample | MMD2 | -0.08 | -0.14 | 1.64 | 1.64 |
| mlpH8, M=4, flat, flux=resample | MMD2_al | 1.02 | 3.91 | 2.12 | 30.33 |
| mlpH8, M=4, multi, flux=resample | PDF-JS | -0.79 | -9.13 | 0.15 | 2.59 |
| mlpH8, M=4, multi, flux=resample | MMD2 | -0.01 | -0.02 | 1.76 | 2.02 |
| mlpH8, M=4, multi, flux=resample | MMD2_al | 1.12 | 6.18 | 2.15 | 44.42 |
| mlpH8, M=4, multi, flux=resample, 12k steps | PDF-JS | -0.13 | -0.59 | 0.60 | 3.89 |
| mlpH8, M=4, multi, flux=resample, 12k steps | MMD2 | 0.01 | 0.02 | 1.79 | 2.20 |
| mlpH8, M=4, multi, flux=resample, 12k steps | MMD2_al | 1.15 | 9.16 | 2.16 | 64.03 |
| mlpH8, M=4, multi, flux=resample, 16k steps | PDF-JS | -0.10 | -0.45 | 0.62 | 4.06 |
| mlpH8, M=4, multi, flux=resample, 16k steps | MMD2 | 0.01 | 0.02 | 1.79 | 2.19 |
| mlpH8, M=4, multi, flux=resample, 16k steps | MMD2_al | 1.15 | 8.36 | 2.16 | 58.59 |
| mlpH8, M=4, multi, flux=resample, 4k steps | PDF-JS | -0.54 | -4.05 | 0.32 | 3.57 |
| mlpH8, M=4, multi, flux=resample, 4k steps | MMD2 | -0.01 | -0.02 | 1.76 | 2.08 |
| mlpH8, M=4, multi, flux=resample, 4k steps | MMD2_al | 1.12 | 7.29 | 2.15 | 52.01 |
| mlpH8, M=4, multi, flux=resample, 8k steps | PDF-JS | -0.23 | -1.16 | 0.53 | 3.93 |
| mlpH8, M=4, multi, flux=resample, 8k steps | MMD2 | 0.00 | 0.00 | 1.78 | 2.17 |
| mlpH8, M=4, multi, flux=resample, 8k steps | MMD2_al | 1.16 | 9.91 | 2.16 | 69.04 |
| mlpH8, M=4, multi, flux=resample, selected | PDF-JS | -0.20 | -0.61 | 0.55 | 2.48 |
| mlpH8, M=4, multi, flux=resample, selected | MMD2 | 0.01 | 0.02 | 1.79 | 2.21 |
| mlpH8, M=4, multi, flux=resample, selected | MMD2_al | 1.17 | 9.45 | 2.16 | 65.40 |
| mlpH8, M=4, spatial+flat, flux=resample | PDF-JS | -0.31 | -1.50 | 0.48 | 3.38 |
| mlpH8, M=4, spatial+flat, flux=resample | MMD2 | -0.02 | -0.05 | 1.74 | 2.05 |
| mlpH8, M=4, spatial+flat, flux=resample | MMD2_al | 0.91 | 2.87 | 2.09 | 24.72 |
| mlpH8, M=4, spatial+flatx4, flux=resample | PDF-JS | -1.05 | -2.99 | -0.03 | -0.12 |
| mlpH8, M=4, spatial+flatx4, flux=resample | MMD2 | -0.12 | -0.21 | 1.54 | 1.42 |
| mlpH8, M=4, spatial+flatx4, flux=resample | MMD2_al | 0.76 | 1.91 | 2.05 | 19.34 |
| gray10, M=4, flat, flux=mean | PDF-JS | -3.72 | -10.57 | -1.84 | -7.70 |
| gray10, M=4, flat, flux=mean | MMD2 | -2.54 | -16.88 | -2.95 | -10.54 |
| gray10, M=4, flat, flux=mean | MMD2_al | 1.05 | 5.40 | 2.13 | 40.94 |
| gray10, M=4, flat, flux=resample | PDF-JS | -3.48 | -10.99 | -1.68 | -7.80 |
| gray10, M=4, flat, flux=resample | MMD2 | -0.13 | -0.25 | 1.54 | 1.61 |
| gray10, M=4, flat, flux=resample | MMD2_al | 1.05 | 5.40 | 2.13 | 40.94 |
| gray10, M=4, spatial+flat, flux=resample | PDF-JS | -1.32 | -9.78 | -0.21 | -2.27 |
| gray10, M=4, spatial+flat, flux=resample | MMD2 | 0.01 | 0.02 | 1.80 | 1.91 |
| gray10, M=4, spatial+flat, flux=resample | MMD2_al | 0.95 | 3.77 | 2.10 | 31.18 |
| gray10, M=4, spatial+flatx4, flux=resample | PDF-JS | -2.39 | -23.92 | -0.94 | -13.82 |
| gray10, M=4, spatial+flatx4, flux=resample | MMD2 | -0.03 | -0.07 | 1.72 | 2.03 |
| gray10, M=4, spatial+flatx4, flux=resample | MMD2_al | 1.06 | 5.05 | 2.13 | 37.90 |
| lowres8, M=2, flat, flux=resample | PDF-JS | -9.48 | -11.66 | -5.76 | -10.41 |
| lowres8, M=2, flat, flux=resample | MMD2 | -2.75 | -2.33 | -3.35 | -1.53 |
| lowres8, M=2, flat, flux=resample | MMD2_al | -3.09 | -7.27 | 1.02 | 8.98 |
| lowres8, M=2, spatial+flat, flux=mean | PDF-JS | -8.71 | -20.97 | -5.23 | -18.54 |
| lowres8, M=2, spatial+flat, flux=mean | MMD2 | -5.43 | -33.00 | -8.35 | -27.21 |
| lowres8, M=2, spatial+flat, flux=mean | MMD2_al | -2.26 | -5.53 | 1.24 | 11.35 |
| lowres8, M=2, spatial+flat, flux=resample | PDF-JS | -7.41 | -32.56 | -4.35 | -28.12 |
| lowres8, M=2, spatial+flat, flux=resample | MMD2 | -2.41 | -2.11 | -2.72 | -1.28 |
| lowres8, M=2, spatial+flat, flux=resample | MMD2_al | -2.26 | -5.53 | 1.24 | 11.35 |
| lowres8, M=4, flat, flux=resample | PDF-JS | -3.82 | -6.88 | -1.91 | -5.06 |
| lowres8, M=4, flat, flux=resample | MMD2 | -1.85 | -3.01 | -1.67 | -1.46 |
| lowres8, M=4, flat, flux=resample | MMD2_al | -1.77 | -4.95 | 1.38 | 14.40 |
| lowres8, M=4, spatial+flat, flux=resample | PDF-JS | -2.71 | -7.35 | -1.16 | -4.61 |
| lowres8, M=4, spatial+flat, flux=resample | MMD2 | -1.71 | -2.77 | -1.41 | -1.22 |
| lowres8, M=4, spatial+flat, flux=resample | MMD2_al | -1.47 | -3.87 | 1.45 | 14.28 |
| lowres8, mlpH8, M=2, flat, flux=resample | PDF-JS | -1.01 | -11.17 | -0.00 | -0.02 |
| lowres8, mlpH8, M=2, flat, flux=resample | MMD2 | -1.92 | -1.74 | -1.81 | -0.88 |
| lowres8, mlpH8, M=2, flat, flux=resample | MMD2_al | -1.18 | -4.03 | 1.53 | 19.55 |
| lowres8, mlpH8, M=2, spatial+flat, flux=resample | PDF-JS | -0.52 | -1.39 | 0.33 | 1.32 |
| lowres8, mlpH8, M=2, spatial+flat, flux=resample | MMD2 | -1.86 | -1.66 | -1.69 | -0.81 |
| lowres8, mlpH8, M=2, spatial+flat, flux=resample | MMD2_al | -0.98 | -3.71 | 1.59 | 22.33 |
| lowres8, mlpH8, M=4, flat, flux=resample | PDF-JS | -1.55 | -3.89 | -0.37 | -1.35 |
| lowres8, mlpH8, M=4, flat, flux=resample | MMD2 | -1.77 | -2.57 | -1.52 | -1.18 |
| lowres8, mlpH8, M=4, flat, flux=resample | MMD2_al | -1.52 | -3.85 | 1.44 | 13.66 |
| lowres8, mlpH8, M=4, spatial+flat, flux=resample | PDF-JS | -1.00 | -3.06 | 0.01 | 0.05 |
| lowres8, mlpH8, M=4, spatial+flat, flux=resample | MMD2 | -1.66 | -2.59 | -1.32 | -1.11 |
| lowres8, mlpH8, M=4, spatial+flat, flux=resample | MMD2_al | -1.32 | -3.08 | 1.49 | 12.98 |

**Verdict at N = 11:** no combination passes. No v12 arm beats both baselines on all three columns by more than the baselines' seed sd.

## N = 100

| Arm | N | PDF-JS | MMD2 | MMD2_u | MMD2_al | proj. corr. | gen_max | clip frac | gen total (mean/sd) | Params | seeds |
|---|---|---|---|---|---|---|---|---|---|---|---|
| M=4, spatial+flatx0.25, flux=resample | 100 | 0.001007 +- 6.842e-05 | 0.02309 +- 0.001968 | 0.01844 +- 0.001934 | 0.02521 +- 0.002625 | 0.9599 +- 0.0007359 | 2.16 +- 0.105 | 0.00112 +- 8.95e-05 | 125.0 / 27.5 | 5,520 | 5 |
| mlpH8, M=2, flat, flux=resample | 100 | 0.002409 +- 6.98e-05 | 0.02316 +- 0.0009813 | 0.01847 +- 0.0009441 | 0.02123 +- 0.00173 | 0.9699 +- 0.0006758 | 2.12 +- 0.122 | 0.00124 +- 7.66e-05 | 126.4 / 29.7 | 9,440 | 5 |
| mlpH8, M=4, flat, flux=resample | 100 | 0.002142 +- 0.0001848 | 0.02105 +- 0.002828 | 0.01647 +- 0.002803 | 0.02189 +- 0.003829 | 0.976 +- 0.0008443 | 2.17 +- 0.0952 | 0.00113 +- 8.21e-05 | 125.0 / 27.5 | 9,656 | 5 |
| mlpH8, M=4, multi, flux=resample | 100 | 0.001928 +- 0.0001809 | 0.02147 +- 0.002354 | 0.01688 +- 0.002339 | 0.02288 +- 0.00314 | 0.9728 +- 0.00306 | 2.08 +- 0.105 | 0.00111 +- 9.95e-05 | 125.0 / 27.5 | 9,656 | 5 |
| mlpH8, M=4, spatial+flat, flux=resample | 100 | 0.001285 +- 5.44e-05 | 0.02102 +- 0.00231 | 0.01641 +- 0.002279 | 0.02118 +- 0.003781 | 0.9759 +- 9.771e-05 | 2.13 +- 0.0992 | 0.00119 +- 9.24e-05 | 125.0 / 27.5 | 9,656 | 5 |
| gray10, M=4, spatial+flatx4, flux=resample | 100 | 0.00218 +- 0.0001492 | 0.0223 +- 0.00302 | 0.0177 +- 0.003013 | 0.02455 +- 0.004031 | 0.9682 +- 0.0005552 | 2.1 +- 0.0371 | 0.00102 +- 6.57e-05 | 125.0 / 27.5 | 5,520 | 5 |
| **DCGAN** (Table 2, v11) | 100 | 0.0005534 +- 0.0001211 | 0.02225 +- 0.003118 | 0.01752 +- 0.003006 | 0.02122 +- 0.004081 | -- | -- | -- | -- | -- | 5 |
| **RBM** (Table 2, v11) | 100 | 0.0009171 +- 0.000132 | 0.05865 +- 0.004363 | 0.05425 +- 0.004423 | 0.08067 +- 0.02205 | -- | -- | -- | -- | -- | 5 |
| **IQP, learned dir. M=4 (flat MMD)** (Table 2, v11) | 100 | 0.002912 +- 0.0002788 | 0.07155 +- 0.002939 | 0.06811 +- 0.002961 | 0.02551 +- 0.004413 | -- | -- | -- | -- | -- | 5 |
| **IQP, learned dir. M=4 (paper kernel)** (Table 2, v11) | 100 | 0.001363 +- 0.0001801 | 0.1132 +- 0.004813 | 0.1095 +- 0.004808 | 0.09991 +- 0.007786 | -- | -- | -- | -- | -- | 5 |

### N = 100 -- wins that the seed spread does not support

| metric | best arm (mean) | overlaps it within 1 sd |
|---|---|---|
| PDF-JS | DCGAN (Table 2) (0.0005534 +- 0.0001211) | **none -- the win holds** |
| MMD2 | mlpH8, M=4, spatial+flat, flux=resample (0.02102 +- 0.00231) | mlpH8, M=4, flat, flux=resample, mlpH8, M=4, multi, flux=resample, DCGAN (Table 2), gray10, M=4, spatial+flatx4, flux=resample, M=4, spatial+flatx0.25, flux=resample, mlpH8, M=2, flat, flux=resample |
| MMD2_u | mlpH8, M=4, spatial+flat, flux=resample (0.01641 +- 0.002279) | mlpH8, M=4, flat, flux=resample, mlpH8, M=4, multi, flux=resample, DCGAN (Table 2), gray10, M=4, spatial+flatx4, flux=resample, M=4, spatial+flatx0.25, flux=resample, mlpH8, M=2, flat, flux=resample |
| MMD2_al | mlpH8, M=4, spatial+flat, flux=resample (0.02118 +- 0.003781) | DCGAN (Table 2), mlpH8, M=2, flat, flux=resample, mlpH8, M=4, flat, flux=resample, mlpH8, M=4, multi, flux=resample, gray10, M=4, spatial+flatx4, flux=resample, M=4, spatial+flatx0.25, flux=resample, IQP, learned dir. M=4 (flat MMD) (Table 2) |

### N = 100 -- the pre-registered "clear win" test

A combination passes only if its MEAN beats BOTH baselines on all three of PDF-JS, raw MMD$^2$ and aligned MMD$^2$ by more than the baseline's own seed sd. `gap/sd_base` is the margin in units of the baseline's sd (the literal test); `gap/sd_iqp` is the same gap in units of the IQP arm's own sd (the stricter reading). A negative number means the baseline is still ahead.

| Arm | metric | vs DCGAN gap/sd_base | vs DCGAN gap/sd_iqp | vs RBM gap/sd_base | vs RBM gap/sd_iqp |
|---|---|---|---|---|---|
| M=4, spatial+flatx0.25, flux=resample | PDF-JS | -3.75 | -6.63 | -0.68 | -1.32 |
| M=4, spatial+flatx0.25, flux=resample | MMD2 | -0.27 | -0.43 | 8.15 | 18.07 |
| M=4, spatial+flatx0.25, flux=resample | MMD2_al | -0.98 | -1.52 | 2.51 | 21.13 |
| mlpH8, M=2, flat, flux=resample | PDF-JS | -15.33 | -26.59 | -11.31 | -21.38 |
| mlpH8, M=2, flat, flux=resample | MMD2 | -0.29 | -0.93 | 8.13 | 36.16 |
| mlpH8, M=2, flat, flux=resample | MMD2_al | -0.00 | -0.00 | 2.70 | 34.35 |
| mlpH8, M=4, flat, flux=resample | PDF-JS | -13.12 | -8.60 | -9.28 | -6.63 |
| mlpH8, M=4, flat, flux=resample | MMD2 | 0.38 | 0.42 | 8.62 | 13.29 |
| mlpH8, M=4, flat, flux=resample | MMD2_al | -0.16 | -0.17 | 2.67 | 15.35 |
| mlpH8, M=4, multi, flux=resample | PDF-JS | -11.35 | -7.60 | -7.65 | -5.59 |
| mlpH8, M=4, multi, flux=resample | MMD2 | 0.25 | 0.33 | 8.52 | 15.79 |
| mlpH8, M=4, multi, flux=resample | MMD2_al | -0.41 | -0.53 | 2.62 | 18.40 |
| mlpH8, M=4, spatial+flat, flux=resample | PDF-JS | -6.04 | -13.44 | -2.78 | -6.76 |
| mlpH8, M=4, spatial+flat, flux=resample | MMD2 | 0.39 | 0.53 | 8.62 | 16.29 |
| mlpH8, M=4, spatial+flat, flux=resample | MMD2_al | 0.01 | 0.01 | 2.70 | 15.73 |
| gray10, M=4, spatial+flatx4, flux=resample | PDF-JS | -13.43 | -10.90 | -9.57 | -8.46 |
| gray10, M=4, spatial+flatx4, flux=resample | MMD2 | -0.01 | -0.02 | 8.33 | 12.04 |
| gray10, M=4, spatial+flatx4, flux=resample | MMD2_al | -0.81 | -0.82 | 2.55 | 13.92 |

**Verdict at N = 100:** no combination passes. No v12 arm beats both baselines on all three columns by more than the baselines' seed sd.

## Provenance

- runs: `outputs_v12/s{0..4}/*` (`train_iqp_affine_v12.py`)
- scores: `results_seeds_v12.jsonl` (`evaluate_v11.py --no-legacy`)
- the `selected` rows: which snapshot the validation block picked comes from `results_seeds_v12_es.jsonl` (`seeds/select_checkpoint.py`); the NUMBERS are the same `results_seeds_v12.jsonl` rows as everywhere else in this file
- baselines: parsed from `seed_tables.md`, section `## Table 2 (v11)`
- this file: `experiments/13_iqp_turbulence_replication/v12_aggregate.py`
