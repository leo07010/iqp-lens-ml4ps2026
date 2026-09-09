# Seed tables - part F: the non-linear (MLP) IQP arm, full suite

`theta_i = theta0 + g(z_i)`, g a tanh MLP (`train_iqp_affine_mlp.py`).  Protocol frozen
to v11 and part D: `data_lens_r3` split (`R3_SPLIT_SEED = 0`), the same 200 held-out
frames, seeds 0-4, `--prior resample`, `--gnorm rownorm`, `--g-input-scale 20`,
1,000-step theta0 KL warm start, 4,000 fixed joint Adam steps, `evaluate_v11.py`.
Lower is better for PDF-JS and both MMD2 columns; higher is better for the projection
correlation.  Verdict cells read `direction / 1-sd`: `SEPARATED` = the mean +- 1 sd
intervals do not overlap, `overlap` = they do and the arms are not distinguishable.

## 0. Where this sits

| arm | N | PDF-JS | MMD2 | MMD2_al | proj corr |
|---|---|---|---|---|---|
| v11 affine IQP (Table 2) | 11 | 0.0030 +- 0.0002 | 0.0891 +- 0.0024 | 0.0274 +- 0.0022 | 0.9102 +- 0.0026 |
| part D MLP, M=4 H=32 | 11 | 0.0009 +- 0.0000 | 0.0866 +- 0.0024 | 0.0249 +- 0.0019 | 0.9076 +- 0.0015 |
| part D2 softmax control, H=32 (no circuit) | 11 | 0.0018 +- 0.0001 | 0.1009 +- 0.0017 | 0.0328 +- 0.0018 | 0.9276 +- 0.0058 |
| RBM-10bit (Table 2) | 11 | 0.0017 +- 0.0008 | 0.0654 +- 0.0081 | 0.1106 +- 0.0388 | -- |
| DCGAN-32 (Table 2) | 11 | 0.0011 +- 0.0006 | 0.0509 +- 0.0151 | 0.0388 +- 0.0104 | -- |
| | | | | | |
| v11 affine IQP (Table 2) | 100 | 0.0029 +- 0.0003 | 0.0716 +- 0.0029 | 0.0255 +- 0.0044 | 0.9684 +- 0.0004 |
| part D MLP, M=4 H=32 | 100 | 0.0018 +- 0.0001 | 0.0700 +- 0.0031 | 0.0193 +- 0.0037 | 0.9840 +- 0.0003 |
| part D2 softmax control, H=32 (no circuit) | 100 | 0.0017 +- 0.0002 | 0.0778 +- 0.0031 | 0.0372 +- 0.0038 | 0.9590 +- 0.0045 |
| RBM-10bit (Table 2) | 100 | 0.0009 +- 0.0001 | 0.0587 +- 0.0044 | 0.0807 +- 0.0221 | -- |
| DCGAN-32 (Table 2) | 100 | 0.0006 +- 0.0001 | 0.0222 +- 0.0031 | 0.0212 +- 0.0041 | -- |
| | | | | | |

## 1. Latent-width sweep (M), flat MMD

`nonlin` is the relative residual of the best affine fit to the learned map on the
trained latents.  It is meaningless where M >= N (the fit is then exact by
construction) and is printed as `n/a` there.

| M | H | N | params | PDF-JS | MMD2 | MMD2_al | proj corr | \|\|z\|\| | nonlin | vs RBM (MMD2_al) | vs DCGAN (MMD2_al) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 32 | 11 | 33867 | 0.0009 +- 0.0001 | 0.0859 +- 0.0026 | 0.0237 +- 0.0046 | 0.9001 +- 0.0007 | 0.1777 +- 0.0148 | 0.7813 +- 0.0291 | better / SEPARATED | better / SEPARATED |
| 2 | 32 | 11 | 33910 | 0.0010 +- 0.0001 | 0.0851 +- 0.0017 | 0.0225 +- 0.0017 | 0.9016 +- 0.0012 | 0.1010 +- 0.0044 | 0.6961 +- 0.0254 | better / SEPARATED | better / SEPARATED |
| 3 | 32 | 11 | 33953 | 0.0009 +- 0.0001 | 0.0854 +- 0.0018 | 0.0240 +- 0.0021 | 0.9059 +- 0.0014 | 0.0829 +- 0.0019 | 0.5541 +- 0.0356 | better / SEPARATED | better / SEPARATED |
| 4 | 32 | 11 | 33996 | 0.0009 +- 0.0000 | 0.0866 +- 0.0023 | 0.0249 +- 0.0018 | 0.9076 +- 0.0014 | 0.0802 +- 0.0083 | 0.4915 +- 0.0371 | better / SEPARATED | better / SEPARATED |
| 6 | 32 | 11 | 34082 | 0.0013 +- 0.0006 | 0.0876 +- 0.0028 | 0.0284 +- 0.0079 | 0.9054 +- 0.0026 | 0.0823 +- 0.0041 | 0.3083 +- 0.0330 | better / SEPARATED | better / overlap |
| 8 | 32 | 11 | 34168 | 0.0029 +- 0.0014 | 0.0997 +- 0.0141 | 0.0503 +- 0.0283 | 0.8986 +- 0.0051 | 0.1202 +- 0.0095 | 0.1725 +- 0.0591 | better / overlap | worse / overlap |
| 16 | 32 | 11 | 34512 | 0.0085 +- 0.0009 | 0.1621 +- 0.0131 | 0.1732 +- 0.0251 | 0.8953 +- 0.0006 | 0.2245 +- 0.0110 | n/a | worse / overlap | worse / SEPARATED |
| 1 | 8 | 11 | 9243 | 0.0019 +- 0.0004 | 0.0865 +- 0.0027 | 0.0243 +- 0.0046 | 0.8970 +- 0.0029 | 0.2003 +- 0.0272 | 0.7932 +- 0.0273 | better / SEPARATED | better / overlap |
| 2 | 8 | 11 | 9262 | 0.0018 +- 0.0002 | 0.0860 +- 0.0011 | 0.0244 +- 0.0028 | 0.9020 +- 0.0013 | 0.1150 +- 0.0163 | 0.7003 +- 0.0373 | better / SEPARATED | better / SEPARATED |
| 3 | 8 | 11 | 9281 | 0.0016 +- 0.0001 | 0.0860 +- 0.0015 | 0.0256 +- 0.0020 | 0.9090 +- 0.0027 | 0.1056 +- 0.0189 | 0.5309 +- 0.0413 | better / SEPARATED | better / SEPARATED |
| 4 | 8 | 11 | 9300 | 0.0017 +- 0.0001 | 0.0881 +- 0.0033 | 0.0282 +- 0.0027 | 0.9087 +- 0.0052 | 0.1084 +- 0.0097 | 0.4529 +- 0.0428 | better / SEPARATED | better / overlap |
| 6 | 8 | 11 | 9338 | 0.0032 +- 0.0009 | 0.0980 +- 0.0075 | 0.0489 +- 0.0105 | 0.9027 +- 0.0040 | 0.1358 +- 0.0110 | 0.3144 +- 0.0656 | better / SEPARATED | worse / overlap |
| 8 | 8 | 11 | 9376 | 0.0050 +- 0.0009 | 0.1164 +- 0.0086 | 0.0867 +- 0.0158 | 0.8949 +- 0.0047 | 0.1669 +- 0.0179 | 0.2449 +- 0.0402 | better / overlap | worse / SEPARATED |
| 16 | 8 | 11 | 9528 | 0.0063 +- 0.0006 | 0.1278 +- 0.0067 | 0.1072 +- 0.0143 | 0.8881 +- 0.0008 | 0.2825 +- 0.0254 | n/a | better / overlap | worse / SEPARATED |
| | | | | | | | | | | | |
| 1 | 32 | 100 | 33956 | 0.0015 +- 0.0001 | 0.0694 +- 0.0014 | 0.0188 +- 0.0025 | 0.9624 +- 0.0016 | 0.3549 +- 0.0094 | 0.8326 +- 0.0183 | better / SEPARATED | better / overlap |
| 2 | 32 | 100 | 34088 | 0.0019 +- 0.0002 | 0.0684 +- 0.0014 | 0.0181 +- 0.0016 | 0.9773 +- 0.0006 | 0.1481 +- 0.0130 | 0.7322 +- 0.0315 | better / SEPARATED | better / overlap |
| 3 | 32 | 100 | 34220 | 0.0018 +- 0.0002 | 0.0692 +- 0.0031 | 0.0188 +- 0.0046 | 0.9807 +- 0.0004 | 0.1175 +- 0.0056 | 0.6498 +- 0.0318 | better / SEPARATED | better / overlap |
| 4 | 32 | 100 | 34352 | 0.0018 +- 0.0001 | 0.0700 +- 0.0029 | 0.0193 +- 0.0035 | 0.9840 +- 0.0003 | 0.0932 +- 0.0033 | 0.5596 +- 0.0201 | better / SEPARATED | better / overlap |
| 6 | 32 | 100 | 34616 | 0.0019 +- 0.0001 | 0.0706 +- 0.0021 | 0.0205 +- 0.0037 | 0.9855 +- 0.0011 | 0.0941 +- 0.0195 | 0.4363 +- 0.0762 | better / SEPARATED | better / overlap |
| 8 | 32 | 100 | 34880 | 0.0028 +- 0.0004 | 0.0684 +- 0.0004 | 0.0207 +- 0.0030 | 0.9752 +- 0.0023 | 0.2168 +- 0.0105 | 0.4909 +- 0.0304 | better / SEPARATED | better / overlap |
| 16 | 32 | 100 | 35936 | 0.0098 +- 0.0001 | 0.2083 +- 0.0075 | 0.2451 +- 0.0082 | 0.9120 +- 0.0013 | 0.4750 +- 0.0165 | 0.5417 +- 0.0108 | worse / SEPARATED | worse / SEPARATED |
| 1 | 8 | 100 | 9332 | 0.0024 +- 0.0001 | 0.0705 +- 0.0022 | 0.0230 +- 0.0025 | 0.9488 +- 0.0017 | 0.2351 +- 0.0188 | 0.8214 +- 0.0286 | better / SEPARATED | worse / overlap |
| 2 | 8 | 100 | 9440 | 0.0026 +- 0.0001 | 0.0696 +- 0.0017 | 0.0212 +- 0.0017 | 0.9699 +- 0.0007 | 0.1173 +- 0.0151 | 0.7158 +- 0.0269 | better / SEPARATED | worse / overlap |
| 3 | 8 | 100 | 9548 | 0.0025 +- 0.0002 | 0.0696 +- 0.0030 | 0.0211 +- 0.0049 | 0.9737 +- 0.0006 | 0.0988 +- 0.0165 | 0.5941 +- 0.0286 | better / SEPARATED | better / overlap |
| 4 | 8 | 100 | 9656 | 0.0023 +- 0.0002 | 0.0706 +- 0.0029 | 0.0219 +- 0.0038 | 0.9760 +- 0.0008 | 0.0701 +- 0.0085 | 0.4865 +- 0.0388 | better / SEPARATED | worse / overlap |
| 6 | 8 | 100 | 9872 | 0.0037 +- 0.0003 | 0.0753 +- 0.0026 | 0.0351 +- 0.0066 | 0.9588 +- 0.0065 | 0.1691 +- 0.0242 | 0.4603 +- 0.0592 | better / SEPARATED | worse / SEPARATED |
| 8 | 8 | 100 | 10088 | 0.0056 +- 0.0007 | 0.0858 +- 0.0049 | 0.0660 +- 0.0114 | 0.9156 +- 0.0126 | 0.2282 +- 0.0085 | 0.4646 +- 0.0341 | better / overlap | worse / SEPARATED |
| 16 | 8 | 100 | 10952 | 0.0098 +- 0.0007 | 0.1905 +- 0.0642 | 0.2212 +- 0.0816 | 0.6720 +- 0.1192 | 0.2132 +- 0.1845 | 0.3241 +- 0.1319 | worse / SEPARATED | worse / SEPARATED |
| | | | | | | | | | | | |

## 2. Hidden-width sweep (H) at M = 4, flat MMD

| H | N | params | PDF-JS | MMD2 | MMD2_al | proj corr | nonlin | vs affine IQP (MMD2_al) |
|---|---|---|---|---|---|---|---|---|
| 4 | 11 | 5184 | 0.0042 +- 0.0009 | 0.0955 +- 0.0061 | 0.0385 +- 0.0098 | 0.8938 +- 0.0117 | 0.3488 +- 0.0595 | worse / overlap |
| 8 | 11 | 9300 | 0.0017 +- 0.0001 | 0.0881 +- 0.0033 | 0.0282 +- 0.0027 | 0.9087 +- 0.0052 | 0.4529 +- 0.0428 | worse / overlap |
| 16 | 11 | 17532 | 0.0010 +- 0.0001 | 0.0866 +- 0.0023 | 0.0253 +- 0.0013 | 0.9093 +- 0.0028 | 0.5026 +- 0.0492 | better / overlap |
| 32 | 11 | 33996 | 0.0009 +- 0.0000 | 0.0866 +- 0.0023 | 0.0249 +- 0.0018 | 0.9076 +- 0.0014 | 0.4915 +- 0.0371 | better / overlap |
| | | | | | | | | |
| 4 | 100 | 5540 | 0.0033 +- 0.0006 | 0.0722 +- 0.0046 | 0.0290 +- 0.0104 | 0.9625 +- 0.0123 | 0.4542 +- 0.0390 | worse / overlap |
| 8 | 100 | 9656 | 0.0023 +- 0.0002 | 0.0706 +- 0.0029 | 0.0219 +- 0.0038 | 0.9760 +- 0.0008 | 0.4865 +- 0.0388 | better / overlap |
| 16 | 100 | 17888 | 0.0020 +- 0.0001 | 0.0701 +- 0.0034 | 0.0197 +- 0.0037 | 0.9826 +- 0.0004 | 0.5331 +- 0.0143 | better / overlap |
| 32 | 100 | 34352 | 0.0018 +- 0.0001 | 0.0700 +- 0.0029 | 0.0193 +- 0.0035 | 0.9840 +- 0.0003 | 0.5596 +- 0.0201 | better / overlap |
| | | | | | | | | |

## 3. Table 2 targets: the best combination under both objectives

`steps` is 4,000 (the protocol Table 2 reports).  `selected` is what
`seeds/select_checkpoint.py` picked per seed out of {1k, 2k, 4k, 8k, 16k} on the
validation block, the symmetry check the classical baselines get.

| arm | objective | N | params | PDF-JS | MMD2 | MMD2_al | proj corr | selected steps | vs affine | vs RBM | vs DCGAN |
|---|---|---|---|---|---|---|---|---|---|---|---|
| M=2 H=32 | MMDflat | 11 | 33910 | 0.0010 +- 0.0001 | 0.0851 +- 0.0017 | 0.0225 +- 0.0017 | 0.9016 +- 0.0012 | 4000, 4000, 4000, 8000, 16000 | better / SEPARATED | better / SEPARATED | better / SEPARATED |
| M=2 H=32 | auto3 | 11 | 33910 | 0.0005 +- 0.0001 | 0.0875 +- 0.0017 | 0.0514 +- 0.0035 | 0.8215 +- 0.0091 | 2000, 4000, 4000, 8000, 16000 | worse / SEPARATED | better / SEPARATED | worse / overlap |
| M=4 H=32 | MMDflat | 11 | 33996 | 0.0009 +- 0.0000 | 0.0866 +- 0.0023 | 0.0249 +- 0.0018 | 0.9076 +- 0.0014 | 1000, 4000, 4000, 4000, 8000 | better / overlap | better / SEPARATED | better / SEPARATED |
| M=4 H=32 | auto3 | 11 | 33996 | 0.0005 +- 0.0000 | 0.0883 +- 0.0015 | 0.0478 +- 0.0008 | 0.8184 +- 0.0059 | 2000, 2000, 4000, 4000, 4000 | worse / SEPARATED | better / SEPARATED | worse / overlap |
| | | | | | | | | | | | |
| M=2 H=32 | MMDflat | 100 | 34088 | 0.0019 +- 0.0002 | 0.0684 +- 0.0014 | 0.0181 +- 0.0016 | 0.9773 +- 0.0006 | 4000, 8000, 16000, 16000, 16000 | better / SEPARATED | better / SEPARATED | better / overlap |
| M=2 H=32 | auto3 | 100 | 34088 | 0.0009 +- 0.0001 | 0.0978 +- 0.0031 | 0.0661 +- 0.0055 | 0.8661 +- 0.0076 | 8000, 8000, 16000, 16000, 16000 | worse / SEPARATED | better / overlap | worse / SEPARATED |
| M=4 H=32 | MMDflat | 100 | 34352 | 0.0018 +- 0.0001 | 0.0700 +- 0.0029 | 0.0193 +- 0.0035 | 0.9840 +- 0.0003 | 8000, 8000, 8000, 16000, 16000 | better / overlap | better / SEPARATED | better / overlap |
| M=4 H=32 | auto3 | 100 | 34352 | 0.0008 +- 0.0001 | 0.0966 +- 0.0018 | 0.0615 +- 0.0046 | 0.8859 +- 0.0118 | 4000, 4000, 16000, 16000, 16000 | worse / SEPARATED | better / overlap | worse / SEPARATED |
| | | | | | | | | | | | |

Per-metric 1-sd verdicts for the same rows, against DCGAN-32:

| arm | objective | N | PDF-JS | MMD2 | MMD2_al | proj corr |
|---|---|---|---|---|---|---|
| M=2 H=32 | MMDflat | 11 | better / overlap | worse / SEPARATED | better / SEPARATED | -- |
| M=2 H=32 | auto3 | 11 | better / overlap | worse / SEPARATED | worse / overlap | -- |
| M=4 H=32 | MMDflat | 11 | better / overlap | worse / SEPARATED | better / SEPARATED | -- |
| M=4 H=32 | auto3 | 11 | better / SEPARATED | worse / SEPARATED | worse / overlap | -- |
| M=2 H=32 | MMDflat | 100 | worse / SEPARATED | worse / SEPARATED | better / overlap | -- |
| M=2 H=32 | auto3 | 100 | worse / SEPARATED | worse / SEPARATED | worse / SEPARATED | -- |
| M=4 H=32 | MMDflat | 100 | worse / SEPARATED | worse / SEPARATED | better / overlap | -- |
| M=4 H=32 | auto3 | 100 | worse / overlap | worse / SEPARATED | worse / SEPARATED | -- |

## 4. Data scaling

| N | MLP arm PDF-JS | MLP MMD2 | MLP MMD2_al | affine MMD2_al | RBM MMD2_al | DCGAN MMD2_al | MLP vs DCGAN (MMD2_al) |
|---|---|---|---|---|---|---|---|
| 10 | 0.0010 +- 0.0001 | 0.0916 +- 0.0026 | 0.0260 +- 0.0044 | 0.0271 +- 0.0031 | 0.0772 +- 0.0145 | 0.0452 +- 0.0085 | better / SEPARATED |
| 11 | 0.0010 +- 0.0001 | 0.0851 +- 0.0017 | 0.0225 +- 0.0017 | -- | -- | -- | -- |
| 20 | 0.0013 +- 0.0001 | 0.0696 +- 0.0029 | 0.0184 +- 0.0027 | -- | -- | -- | -- |
| 30 | 0.0014 +- 0.0001 | 0.0673 +- 0.0020 | 0.0228 +- 0.0029 | 0.0258 +- 0.0050 | 0.1030 +- 0.0218 | 0.0324 +- 0.0070 | better / overlap |
| 50 | 0.0017 +- 0.0001 | 0.0664 +- 0.0034 | 0.0198 +- 0.0050 | 0.0248 +- 0.0028 | 0.1149 +- 0.0343 | 0.0244 +- 0.0094 | better / overlap |
| 70 | 0.0019 +- 0.0001 | 0.0698 +- 0.0014 | 0.0160 +- 0.0015 | 0.0226 +- 0.0022 | 0.1272 +- 0.0281 | 0.0214 +- 0.0037 | better / SEPARATED |
| 90 | 0.0019 +- 0.0002 | 0.0698 +- 0.0031 | 0.0188 +- 0.0028 | 0.0230 +- 0.0010 | 0.0817 +- 0.0138 | 0.0191 +- 0.0018 | better / overlap |
| 100 | 0.0019 +- 0.0002 | 0.0684 +- 0.0014 | 0.0181 +- 0.0016 | 0.0255 +- 0.0044 | 0.0807 +- 0.0221 | 0.0212 +- 0.0041 | better / overlap |

The scaling arm is `IQP-mlpH32-M2-MMDflat-resample`.  N = 11, 20 and 100 come from other sections of this suite and
are shown where the published curves also have a point.

## 5. Per class

The published per-class rows exist only for the affine IQP arm, so RBM and DCGAN have
no per-class counterpart to compare against.

| class | N | MLP PDF-JS | MLP MMD2 | MLP MMD2_al | affine MMD2_al | vs affine (MMD2_al) |
|---|---|---|---|---|---|---|
| axion | 11 | 0.0010 +- 0.0001 | 0.0841 +- 0.0022 | 0.0295 +- 0.0038 | 0.0282 +- 0.0053 | worse / overlap |
| axion | 100 | 0.0019 +- 0.0001 | 0.0695 +- 0.0015 | 0.0166 +- 0.0020 | 0.0208 +- 0.0023 | better / overlap |
| cdm | 11 | 0.0009 +- 0.0001 | 0.0850 +- 0.0022 | 0.0265 +- 0.0032 | 0.0257 +- 0.0045 | worse / overlap |
| cdm | 100 | 0.0018 +- 0.0001 | 0.0767 +- 0.0021 | 0.0146 +- 0.0031 | 0.0184 +- 0.0028 | better / overlap |
| no_sub | 11 | 0.0008 +- 0.0001 | 0.0687 +- 0.0018 | 0.0242 +- 0.0058 | 0.0261 +- 0.0010 | better / overlap |
| no_sub | 100 | 0.0018 +- 0.0001 | 0.0749 +- 0.0008 | 0.0206 +- 0.0030 | 0.0294 +- 0.0027 | better / SEPARATED |

## 6. Ablations

Baseline = the same combination with the suite's own settings (KL warm start, 8
restarts, restart picked by correlation).

| ablation | N | PDF-JS | MMD2 | MMD2_al | proj corr | vs baseline (proj corr) | vs baseline (MMD2_al) |
|---|---|---|---|---|---|---|---|
| baseline (IQP-mlpH32-M2-MMDflat-resample) | 11 | 0.0010 +- 0.0001 | 0.0851 +- 0.0017 | 0.0225 +- 0.0017 | 0.9016 +- 0.0012 | -- | -- |
| IQP-mlpabl-R1-H32-M2 | 11 | 0.0010 +- 0.0001 | 0.0851 +- 0.0019 | 0.0225 +- 0.0018 | 0.8381 +- 0.0030 | worse / SEPARATED | worse / overlap |
| IQP-mlpabl-R2-H32-M2 | 11 | 0.0010 +- 0.0001 | 0.0851 +- 0.0019 | 0.0225 +- 0.0018 | 0.9010 +- 0.0013 | worse / overlap | worse / overlap |
| IQP-mlpabl-pickloss-H32-M2 | 11 | 0.0010 +- 0.0001 | 0.0851 +- 0.0019 | 0.0225 +- 0.0018 | 0.9013 +- 0.0012 | worse / overlap | worse / overlap |
| IQP-mlpabl-th0mmd-H32-M2 | 11 | 0.0009 +- 0.0000 | 0.0850 +- 0.0013 | 0.0231 +- 0.0022 | 0.9022 +- 0.0011 | better / overlap | worse / overlap |
| | | | | | | | |
| baseline (IQP-mlpH32-M2-MMDflat-resample) | 100 | 0.0019 +- 0.0002 | 0.0684 +- 0.0014 | 0.0181 +- 0.0016 | 0.9773 +- 0.0006 | -- | -- |
| IQP-mlpabl-R1-H32-M2 | 100 | 0.0019 +- 0.0002 | 0.0684 +- 0.0015 | 0.0181 +- 0.0017 | 0.8871 +- 0.0154 | worse / SEPARATED | worse / overlap |
| IQP-mlpabl-R2-H32-M2 | 100 | 0.0019 +- 0.0002 | 0.0684 +- 0.0015 | 0.0181 +- 0.0017 | 0.9772 +- 0.0007 | worse / overlap | worse / overlap |
| IQP-mlpabl-pickloss-H32-M2 | 100 | 0.0019 +- 0.0002 | 0.0684 +- 0.0015 | 0.0181 +- 0.0017 | 0.9773 +- 0.0006 | worse / overlap | worse / overlap |
| IQP-mlpabl-th0mmd-H32-M2 | 100 | 0.0018 +- 0.0001 | 0.0687 +- 0.0016 | 0.0178 +- 0.0014 | 0.9773 +- 0.0004 | worse / overlap | better / overlap |
| | | | | | | | |

## 7. Provenance

| what | where |
|---|---|
| suite rows | `experiments/13_iqp_turbulence_replication/results_seeds_mlp_suite.jsonl` |
| checkpoint selection | `experiments/13_iqp_turbulence_replication/results_seeds_mlp_suite_es.jsonl` |
| Table 2 rows | `experiments/13_iqp_turbulence_replication/results_seeds_table2_v11.jsonl` |
| scaling rows | `experiments/13_iqp_turbulence_replication/results_seeds_scaling_v11.jsonl` |
| per-class rows | `experiments/13_iqp_turbulence_replication/results_seeds_class_v11.jsonl` |
| part D | `experiments/13_iqp_turbulence_replication/results_seeds_gridmlp.jsonl` |
| part D2 control | `experiments/13_iqp_turbulence_replication/results_seeds_mlpctrl.jsonl` |
| trainer | `experiments/13_iqp_turbulence_replication/train_iqp_affine_mlp.py` |
| ablation trainer | `experiments/13_iqp_turbulence_replication/train_iqp_affine_mlp_abl.py` |
| per-class wrapper | `experiments/13_iqp_turbulence_replication/seeds/run_mlp_class.py` |
| phase 1 / 2 / 3 | `seeds/submit_mlp_suite_sweep.sbatch`, `seeds/submit_mlp_suite_phase2.sbatch`, `seeds/submit_mlp_suite_phase3.sbatch` |
| run outputs | `<WORK>/gsoc_iqp_mlpsuite/` |

535 scored suite rows (335 at the fixed 4,000 steps, 200 snapshots).
