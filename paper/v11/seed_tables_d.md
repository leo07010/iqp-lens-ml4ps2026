# Seed tables - part D: finer grid (12 qubits) and non-linear latent map

Every number is mean +- sd over seeds 0-4, scored by `evaluate_v11.py` against the same
200 held-out frames Table 2 uses, on the same `data_lens_r3` split (`R3_SPLIT_SEED = 0`;
`data_lens_n12` takes its images from that same loader, so the 12-qubit arm sees exactly
the same frames).  The three Table 2 rows are read from
`results_seeds_table2_v11.jsonl`, not recomputed.  Lower is better for PDF-JS and both
Conv-MMD2 columns; higher is better for the projection correlation.

| arm | what it changes | parameters (N=100) |
|---|---|---|
| IQP-n12-affine-M4 (fixed 4k) | 12 qubits, whole 64x64 frame on 4,096 codes, no 32x32 crop | 20880 |
| IQP-n12-affine-M4 (val-selected) | the same, checkpoint chosen on the validation block | 20880 |
| IQP-mlpH32-M4 (fixed 4k) | theta = theta0 + g(z), g a tanh MLP (H=32), magnitude calibrated to the affine map's at init | 34352 |
| IQP-mlpH32-M4 (val-selected) | the same, checkpoint chosen on the validation block | 34352 |
| IQP-mlpH32-M4 zscale (fixed 4k) | the same MLP with ||g(z)|| = PHI*||z|| enforced for every z | 34352 |
| IQP-mlpH32-M2 (fixed 4k) | the same MLP with a 2-dimensional latent | 34088 |
| IQP-n12-affine-M4 z-init 0.05 (fixed 4k) | diagnostic: the 12-qubit arm with z started at the 10-qubit arm's own latent scale (0.05) | 20880 |
| IQP-affine-M4 (Table 2, fixed 4k) | -- (reference) | 5520 |
| RBM-10bit (Table 2) | -- (reference) | 11010 |
| DCGAN-32 (Table 2) | -- (reference) | 199837 |

## 1. Main table (v11 metrics, 5 seeds)

| arm | N | params | PDF-JS | Conv-MMD2 | Conv-MMD2 (flux-aligned) | proj. corr | steps |
|---|---|---|---|---|---|---|---|
| IQP-n12-affine-M4 (fixed 4k) | 11 | 20524 | 0.0433 +- 0.0300 | 0.2328 +- 0.0879 | 0.1841 +- 0.1128 | 0.8959 +- 0.0105 | 4000 |
| IQP-n12-affine-M4 (val-selected) | 11 | 20524 | 0.0422 +- 0.0309 | 0.2323 +- 0.0911 | 0.1846 +- 0.1166 | 0.8987 +- 0.0109 | 2000 x1, 8000 x2, 16000 x2 |
| IQP-mlpH32-M4 (fixed 4k) | 11 | 33996 | 0.0009 +- 0.0000 | 0.0866 +- 0.0024 | 0.0249 +- 0.0019 | 0.9076 +- 0.0015 | 4000 |
| IQP-mlpH32-M4 (val-selected) | 11 | 33996 | 0.0010 +- 0.0004 | 0.0867 +- 0.0024 | 0.0248 +- 0.0017 | 0.9059 +- 0.0049 | 1000 x1, 4000 x3, 8000 x1 |
| IQP-mlpH32-M4 zscale (fixed 4k) | 11 | 33996 | 0.0011 +- 0.0000 | 0.0874 +- 0.0021 | 0.0259 +- 0.0021 | 0.9092 +- 0.0027 | 4000 |
| IQP-mlpH32-M2 (fixed 4k) | 11 | 33910 | 0.0010 +- 0.0001 | 0.0851 +- 0.0019 | 0.0225 +- 0.0018 | 0.9016 +- 0.0013 | 4000 |
| IQP-n12-affine-M4 z-init 0.05 (fixed 4k) | 11 | 20524 | 0.1115 +- 0.0053 | 0.4578 +- 0.0272 | 0.4897 +- 0.0411 | 0.8818 +- 0.0042 | 4000 |
| IQP-affine-M4 (Table 2, fixed 4k) | 11 | 5164 | 0.0030 +- 0.0002 | 0.0891 +- 0.0024 | 0.0274 +- 0.0022 | 0.9102 +- 0.0026 | 4000 |
| RBM-10bit (Table 2) | 11 | 11010 | 0.0017 +- 0.0008 | 0.0654 +- 0.0081 | 0.1106 +- 0.0388 | -- | 160000 |
| DCGAN-32 (Table 2) | 11 | 199837 | 0.0011 +- 0.0006 | 0.0509 +- 0.0151 | 0.0388 +- 0.0104 | -- | 5000 x2, 15000 x1, 30000 x2 |
| | | | | | | | |
| IQP-n12-affine-M4 (fixed 4k) | 100 | 20880 | 0.0921 +- 0.0711 | 0.2913 +- 0.1993 | 0.4303 +- 0.3084 | 0.7565 +- 0.1925 | 4000 |
| IQP-n12-affine-M4 (val-selected) | 100 | 20880 | 0.0900 +- 0.0733 | 0.2888 +- 0.2016 | 0.4253 +- 0.3132 | 0.7569 +- 0.1929 | 1000 x2, 2000 x1, 8000 x1, 16000 x1 |
| IQP-mlpH32-M4 (fixed 4k) | 100 | 34352 | 0.0018 +- 0.0001 | 0.0700 +- 0.0031 | 0.0193 +- 0.0037 | 0.9840 +- 0.0003 | 4000 |
| IQP-mlpH32-M4 (val-selected) | 100 | 34352 | 0.0015 +- 0.0001 | 0.0698 +- 0.0031 | 0.0186 +- 0.0039 | 0.9842 +- 0.0002 | 8000 x3, 16000 x2 |
| IQP-mlpH32-M4 zscale (fixed 4k) | 100 | 34352 | 0.0022 +- 0.0002 | 0.0703 +- 0.0031 | 0.0200 +- 0.0038 | 0.9822 +- 0.0002 | 4000 |
| IQP-mlpH32-M2 (fixed 4k) | 100 | 34088 | 0.0019 +- 0.0002 | 0.0684 +- 0.0015 | 0.0181 +- 0.0017 | 0.9773 +- 0.0006 | 4000 |
| IQP-n12-affine-M4 z-init 0.05 (fixed 4k) | 100 | 20880 | 0.1573 +- 0.0041 | 0.4345 +- 0.0155 | 0.6638 +- 0.0246 | 0.5946 +- 0.0450 | 4000 |
| IQP-affine-M4 (Table 2, fixed 4k) | 100 | 5520 | 0.0029 +- 0.0003 | 0.0716 +- 0.0029 | 0.0255 +- 0.0044 | 0.9684 +- 0.0004 | 4000 |
| RBM-10bit (Table 2) | 100 | 11010 | 0.0009 +- 0.0001 | 0.0587 +- 0.0044 | 0.0807 +- 0.0221 | -- | 80000 |
| DCGAN-32 (Table 2) | 100 | 199837 | 0.0006 +- 0.0001 | 0.0222 +- 0.0031 | 0.0212 +- 0.0041 | -- | 5000 |
| | | | | | | | |

`steps` is the number of joint Adam steps behind the reported checkpoint: 4,000 for the
fixed protocol (Table 2's own number) and, for the val-selected rows, whatever
`select_checkpoint.py` picked per seed out of {1k, 2k, 4k, 8k, 16k} using only the
validation block, which is disjoint from the 200 held-out frames.

## 2. Does either change beat the current best IQP arm by more than 1 sd?

`separated` = the two mean +- 1 sd intervals do not overlap; `overlap` = they do, i.e.
the seed spread already covers the difference and the arms are not distinguishable at
this sample size.  `better`/`worse` is the direction of the means.

| arm | N | metric | this arm | IQP-affine-M4 (Table 2, fixed 4k) | verdict |
|---|---|---|---|---|---|
| IQP-n12-affine-M4 (fixed 4k) | 11 | PDF-JS | 0.0433 +- 0.0300 | 0.0030 +- 0.0002 | worse, separated by >1 sd |
| IQP-n12-affine-M4 (fixed 4k) | 11 | Conv-MMD2 | 0.2328 +- 0.0879 | 0.0891 +- 0.0024 | worse, separated by >1 sd |
| IQP-n12-affine-M4 (fixed 4k) | 11 | Conv-MMD2 (flux-aligned) | 0.1841 +- 0.1128 | 0.0274 +- 0.0022 | worse, separated by >1 sd |
| IQP-n12-affine-M4 (fixed 4k) | 11 | held-out projection corr | 0.8959 +- 0.0105 | 0.9102 +- 0.0026 | worse, separated by >1 sd |
| IQP-n12-affine-M4 (val-selected) | 11 | PDF-JS | 0.0422 +- 0.0309 | 0.0030 +- 0.0002 | worse, separated by >1 sd |
| IQP-n12-affine-M4 (val-selected) | 11 | Conv-MMD2 | 0.2323 +- 0.0911 | 0.0891 +- 0.0024 | worse, separated by >1 sd |
| IQP-n12-affine-M4 (val-selected) | 11 | Conv-MMD2 (flux-aligned) | 0.1846 +- 0.1166 | 0.0274 +- 0.0022 | worse, separated by >1 sd |
| IQP-n12-affine-M4 (val-selected) | 11 | held-out projection corr | 0.8987 +- 0.0109 | 0.9102 +- 0.0026 | worse, overlap (not distinguishable) |
| IQP-mlpH32-M4 (fixed 4k) | 11 | PDF-JS | 0.0009 +- 0.0000 | 0.0030 +- 0.0002 | better, separated by >1 sd |
| IQP-mlpH32-M4 (fixed 4k) | 11 | Conv-MMD2 | 0.0866 +- 0.0024 | 0.0891 +- 0.0024 | better, overlap (not distinguishable) |
| IQP-mlpH32-M4 (fixed 4k) | 11 | Conv-MMD2 (flux-aligned) | 0.0249 +- 0.0019 | 0.0274 +- 0.0022 | better, overlap (not distinguishable) |
| IQP-mlpH32-M4 (fixed 4k) | 11 | held-out projection corr | 0.9076 +- 0.0015 | 0.9102 +- 0.0026 | worse, overlap (not distinguishable) |
| IQP-mlpH32-M4 (val-selected) | 11 | PDF-JS | 0.0010 +- 0.0004 | 0.0030 +- 0.0002 | better, separated by >1 sd |
| IQP-mlpH32-M4 (val-selected) | 11 | Conv-MMD2 | 0.0867 +- 0.0024 | 0.0891 +- 0.0024 | better, overlap (not distinguishable) |
| IQP-mlpH32-M4 (val-selected) | 11 | Conv-MMD2 (flux-aligned) | 0.0248 +- 0.0017 | 0.0274 +- 0.0022 | better, overlap (not distinguishable) |
| IQP-mlpH32-M4 (val-selected) | 11 | held-out projection corr | 0.9059 +- 0.0049 | 0.9102 +- 0.0026 | worse, overlap (not distinguishable) |
| IQP-mlpH32-M4 zscale (fixed 4k) | 11 | PDF-JS | 0.0011 +- 0.0000 | 0.0030 +- 0.0002 | better, separated by >1 sd |
| IQP-mlpH32-M4 zscale (fixed 4k) | 11 | Conv-MMD2 | 0.0874 +- 0.0021 | 0.0891 +- 0.0024 | better, overlap (not distinguishable) |
| IQP-mlpH32-M4 zscale (fixed 4k) | 11 | Conv-MMD2 (flux-aligned) | 0.0259 +- 0.0021 | 0.0274 +- 0.0022 | better, overlap (not distinguishable) |
| IQP-mlpH32-M4 zscale (fixed 4k) | 11 | held-out projection corr | 0.9092 +- 0.0027 | 0.9102 +- 0.0026 | worse, overlap (not distinguishable) |
| IQP-mlpH32-M2 (fixed 4k) | 11 | PDF-JS | 0.0010 +- 0.0001 | 0.0030 +- 0.0002 | better, separated by >1 sd |
| IQP-mlpH32-M2 (fixed 4k) | 11 | Conv-MMD2 | 0.0851 +- 0.0019 | 0.0891 +- 0.0024 | better, overlap (not distinguishable) |
| IQP-mlpH32-M2 (fixed 4k) | 11 | Conv-MMD2 (flux-aligned) | 0.0225 +- 0.0018 | 0.0274 +- 0.0022 | better, separated by >1 sd |
| IQP-mlpH32-M2 (fixed 4k) | 11 | held-out projection corr | 0.9016 +- 0.0013 | 0.9102 +- 0.0026 | worse, separated by >1 sd |
| IQP-n12-affine-M4 z-init 0.05 (fixed 4k) | 11 | PDF-JS | 0.1115 +- 0.0053 | 0.0030 +- 0.0002 | worse, separated by >1 sd |
| IQP-n12-affine-M4 z-init 0.05 (fixed 4k) | 11 | Conv-MMD2 | 0.4578 +- 0.0272 | 0.0891 +- 0.0024 | worse, separated by >1 sd |
| IQP-n12-affine-M4 z-init 0.05 (fixed 4k) | 11 | Conv-MMD2 (flux-aligned) | 0.4897 +- 0.0411 | 0.0274 +- 0.0022 | worse, separated by >1 sd |
| IQP-n12-affine-M4 z-init 0.05 (fixed 4k) | 11 | held-out projection corr | 0.8818 +- 0.0042 | 0.9102 +- 0.0026 | worse, separated by >1 sd |
| | | | | | |
| IQP-n12-affine-M4 (fixed 4k) | 100 | PDF-JS | 0.0921 +- 0.0711 | 0.0029 +- 0.0003 | worse, separated by >1 sd |
| IQP-n12-affine-M4 (fixed 4k) | 100 | Conv-MMD2 | 0.2913 +- 0.1993 | 0.0716 +- 0.0029 | worse, separated by >1 sd |
| IQP-n12-affine-M4 (fixed 4k) | 100 | Conv-MMD2 (flux-aligned) | 0.4303 +- 0.3084 | 0.0255 +- 0.0044 | worse, separated by >1 sd |
| IQP-n12-affine-M4 (fixed 4k) | 100 | held-out projection corr | 0.7565 +- 0.1925 | 0.9684 +- 0.0004 | worse, separated by >1 sd |
| IQP-n12-affine-M4 (val-selected) | 100 | PDF-JS | 0.0900 +- 0.0733 | 0.0029 +- 0.0003 | worse, separated by >1 sd |
| IQP-n12-affine-M4 (val-selected) | 100 | Conv-MMD2 | 0.2888 +- 0.2016 | 0.0716 +- 0.0029 | worse, separated by >1 sd |
| IQP-n12-affine-M4 (val-selected) | 100 | Conv-MMD2 (flux-aligned) | 0.4253 +- 0.3132 | 0.0255 +- 0.0044 | worse, separated by >1 sd |
| IQP-n12-affine-M4 (val-selected) | 100 | held-out projection corr | 0.7569 +- 0.1929 | 0.9684 +- 0.0004 | worse, separated by >1 sd |
| IQP-mlpH32-M4 (fixed 4k) | 100 | PDF-JS | 0.0018 +- 0.0001 | 0.0029 +- 0.0003 | better, separated by >1 sd |
| IQP-mlpH32-M4 (fixed 4k) | 100 | Conv-MMD2 | 0.0700 +- 0.0031 | 0.0716 +- 0.0029 | better, overlap (not distinguishable) |
| IQP-mlpH32-M4 (fixed 4k) | 100 | Conv-MMD2 (flux-aligned) | 0.0193 +- 0.0037 | 0.0255 +- 0.0044 | better, overlap (not distinguishable) |
| IQP-mlpH32-M4 (fixed 4k) | 100 | held-out projection corr | 0.9840 +- 0.0003 | 0.9684 +- 0.0004 | better, separated by >1 sd |
| IQP-mlpH32-M4 (val-selected) | 100 | PDF-JS | 0.0015 +- 0.0001 | 0.0029 +- 0.0003 | better, separated by >1 sd |
| IQP-mlpH32-M4 (val-selected) | 100 | Conv-MMD2 | 0.0698 +- 0.0031 | 0.0716 +- 0.0029 | better, overlap (not distinguishable) |
| IQP-mlpH32-M4 (val-selected) | 100 | Conv-MMD2 (flux-aligned) | 0.0186 +- 0.0039 | 0.0255 +- 0.0044 | better, overlap (not distinguishable) |
| IQP-mlpH32-M4 (val-selected) | 100 | held-out projection corr | 0.9842 +- 0.0002 | 0.9684 +- 0.0004 | better, separated by >1 sd |
| IQP-mlpH32-M4 zscale (fixed 4k) | 100 | PDF-JS | 0.0022 +- 0.0002 | 0.0029 +- 0.0003 | better, separated by >1 sd |
| IQP-mlpH32-M4 zscale (fixed 4k) | 100 | Conv-MMD2 | 0.0703 +- 0.0031 | 0.0716 +- 0.0029 | better, overlap (not distinguishable) |
| IQP-mlpH32-M4 zscale (fixed 4k) | 100 | Conv-MMD2 (flux-aligned) | 0.0200 +- 0.0038 | 0.0255 +- 0.0044 | better, overlap (not distinguishable) |
| IQP-mlpH32-M4 zscale (fixed 4k) | 100 | held-out projection corr | 0.9822 +- 0.0002 | 0.9684 +- 0.0004 | better, separated by >1 sd |
| IQP-mlpH32-M2 (fixed 4k) | 100 | PDF-JS | 0.0019 +- 0.0002 | 0.0029 +- 0.0003 | better, separated by >1 sd |
| IQP-mlpH32-M2 (fixed 4k) | 100 | Conv-MMD2 | 0.0684 +- 0.0015 | 0.0716 +- 0.0029 | better, overlap (not distinguishable) |
| IQP-mlpH32-M2 (fixed 4k) | 100 | Conv-MMD2 (flux-aligned) | 0.0181 +- 0.0017 | 0.0255 +- 0.0044 | better, separated by >1 sd |
| IQP-mlpH32-M2 (fixed 4k) | 100 | held-out projection corr | 0.9773 +- 0.0006 | 0.9684 +- 0.0004 | better, separated by >1 sd |
| IQP-n12-affine-M4 z-init 0.05 (fixed 4k) | 100 | PDF-JS | 0.1573 +- 0.0041 | 0.0029 +- 0.0003 | worse, separated by >1 sd |
| IQP-n12-affine-M4 z-init 0.05 (fixed 4k) | 100 | Conv-MMD2 | 0.4345 +- 0.0155 | 0.0716 +- 0.0029 | worse, separated by >1 sd |
| IQP-n12-affine-M4 z-init 0.05 (fixed 4k) | 100 | Conv-MMD2 (flux-aligned) | 0.6638 +- 0.0246 | 0.0255 +- 0.0044 | worse, separated by >1 sd |
| IQP-n12-affine-M4 z-init 0.05 (fixed 4k) | 100 | held-out projection corr | 0.5946 +- 0.0450 | 0.9684 +- 0.0004 | worse, separated by >1 sd |
| | | | | | |

## 3. Diagnostics: was the change actually exercised?

`nonlinearity ratio` is the relative residual of the least-squares best AFFINE fit to
the learned map on the trained latents: 0 means the MLP collapsed to exactly the affine
map the reference arm already has, so a null result would be a null test rather than a
finding.  `|pre|` is the mean absolute tanh pre-activation (tanh is linear to a part in
10^3 below ~0.05).  `train corr` and `proj. corr` are the fit to the training frames and
to the held-out frames through the z-only projection.

`fit gain` is 1 - (data loss at the reported checkpoint) / L0, where L0 is the same
objective's loss at theta0 with z = 0: the fraction of the fittable loss the latent map
removed.  Near 0 means the arm is barely doing better than one shared hologram.

| arm | N | fit gain | nonlinearity ratio | mean \|pre\| | \|\|z\|\| | train corr | proj. corr |
|---|---|---|---|---|---|---|---|
| IQP-n12-affine-M4 (fixed 4k) | 11 | 0.8471 +- 0.0336 | -- | -- | 0.0719 +- 0.0080 | 0.9824 +- 0.0042 | 0.8959 +- 0.0105 |
| IQP-n12-affine-M4 (val-selected) | 11 | 0.8636 +- 0.0376 | -- | -- | 0.0738 +- 0.0113 | 0.9842 +- 0.0046 | 0.8987 +- 0.0109 |
| IQP-mlpH32-M4 (fixed 4k) | 11 | 0.9972 +- 0.0003 | 0.4915 +- 0.0393 | 1.350 +- 0.043 | 0.0802 +- 0.0088 | 0.9994 +- 0.0001 | 0.9076 +- 0.0015 |
| IQP-mlpH32-M4 (val-selected) | 11 | 0.9966 +- 0.0029 | 0.4777 +- 0.0337 | 1.435 +- 0.094 | 0.0876 +- 0.0200 | 0.9993 +- 0.0006 | 0.9059 +- 0.0049 |
| IQP-mlpH32-M4 zscale (fixed 4k) | 11 | 0.9958 +- 0.0004 | 0.2415 +- 0.0496 | 1.100 +- 0.101 | 0.0387 +- 0.0058 | 0.9991 +- 0.0001 | 0.9092 +- 0.0027 |
| IQP-mlpH32-M2 (fixed 4k) | 11 | 0.9965 +- 0.0002 | 0.6961 +- 0.0269 | 1.963 +- 0.025 | 0.1010 +- 0.0046 | 0.9993 +- 0.0001 | 0.9016 +- 0.0013 |
| IQP-n12-affine-M4 z-init 0.05 (fixed 4k) | 11 | 0.7711 +- 0.0150 | -- | -- | 0.1262 +- 0.0130 | 0.9741 +- 0.0022 | 0.8818 +- 0.0042 |
| IQP-affine-M4 (Table 2, fixed 4k) | 11 | 0.9665 +- 0.0008 | -- | -- | 0.0257 +- 0.0016 | 0.9926 +- 0.0001 | 0.9102 +- 0.0026 |
| RBM-10bit (Table 2) | 11 | -- | -- | -- | -- | -- | -- |
| DCGAN-32 (Table 2) | 11 | -- | -- | -- | -- | -- | -- |
| | | | | | | | |
| IQP-n12-affine-M4 (fixed 4k) | 100 | -0.9583 +- 1.4731 | -- | -- | 0.0583 +- 0.0180 | 0.5289 +- 0.4774 | 0.7565 +- 0.1925 |
| IQP-n12-affine-M4 (val-selected) | 100 | -0.9466 +- 1.4848 | -- | -- | 0.0592 +- 0.0189 | 0.5298 +- 0.4782 | 0.7569 +- 0.1929 |
| IQP-mlpH32-M4 (fixed 4k) | 100 | 0.9656 +- 0.0010 | 0.5596 +- 0.0213 | 1.705 +- 0.073 | 0.0932 +- 0.0035 | 0.9931 +- 0.0002 | 0.9840 +- 0.0003 |
| IQP-mlpH32-M4 (val-selected) | 100 | 0.9793 +- 0.0018 | 0.6191 +- 0.0185 | 2.095 +- 0.102 | 0.1391 +- 0.0105 | 0.9958 +- 0.0004 | 0.9842 +- 0.0002 |
| IQP-mlpH32-M4 zscale (fixed 4k) | 100 | 0.9513 +- 0.0007 | 0.2944 +- 0.0282 | 1.341 +- 0.086 | 0.0333 +- 0.0048 | 0.9902 +- 0.0001 | 0.9822 +- 0.0002 |
| IQP-mlpH32-M2 (fixed 4k) | 100 | 0.9508 +- 0.0025 | 0.7322 +- 0.0340 | 2.767 +- 0.255 | 0.1481 +- 0.0141 | 0.9899 +- 0.0006 | 0.9773 +- 0.0006 |
| IQP-n12-affine-M4 z-init 0.05 (fixed 4k) | 100 | -1.2698 +- 0.0525 | -- | -- | 0.1316 +- 0.0079 | 0.5796 +- 0.0144 | 0.5946 +- 0.0450 |
| IQP-affine-M4 (Table 2, fixed 4k) | 100 | 0.8715 +- 0.0012 | -- | -- | 0.0173 +- 0.0002 | 0.9746 +- 0.0002 | 0.9684 +- 0.0004 |
| RBM-10bit (Table 2) | 100 | -- | -- | -- | -- | -- | -- |
| DCGAN-32 (Table 2) | 100 | -- | -- | -- | -- | -- | -- |
| | | | | | | | |

## 4. Provenance

| what | where |
|---|---|
| new arms, v11 metrics | `experiments/13_iqp_turbulence_replication/results_seeds_gridmlp.jsonl` |
| checkpoint selection | `experiments/13_iqp_turbulence_replication/results_seeds_gridmlp_es.jsonl` |
| Table 2 reference rows | `experiments/13_iqp_turbulence_replication/results_seeds_table2_v11.jsonl` |
| trainer (A) | `experiments/13_iqp_turbulence_replication/train_iqp_affine_n12.py` |
| trainer (B) | `experiments/13_iqp_turbulence_replication/train_iqp_affine_mlp.py` |
| loader (A) | `experiments/13_iqp_turbulence_replication/data_lens_n12.py` |
| grid | `experiments/13_iqp_turbulence_replication/seeds/submit_seeds_grid_mlp.sbatch` |

100 scored runs across 20 (arm, N) cells.
