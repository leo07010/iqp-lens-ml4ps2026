# Seed tables - part D2: the same latent model with the circuit removed

Internal control, not paper material.  The question is narrow: the non-linear latent
arm (`train_iqp_affine_mlp.py`) beats the published affine IQP arm -- how much of that
survives if the IQP generator is deleted and the SAME machinery writes the pmf
directly?

| | quantum arm | this control |
|---|---|---|
| generator | `q = \|<x\| U(theta) \|0>\|^2` | `q = softmax(theta)` |
| theta | `theta0 + g(z)`, g a 4->H->1024 tanh MLP, no output bias | identical |
| latents, z penalty, prior, projection, S, paste, steps, Adam | round-3b protocol | identical |
| objective | flat MMD from the pmf's Walsh coefficients | identical (plus a KL variant) |

Everything is mean +- sd over seeds 0-4, at 4,000 fixed steps, scored by
`evaluate_v11.py` against the same 200 held-out frames Table 2 uses.  Lower is better
for PDF-JS and both Conv-MMD2 columns; higher is better for the projection correlation.

## 1. Main table

| arm | N | params | PDF-JS | Conv-MMD2 | Conv-MMD2 (flux-aligned) | proj. corr |
|---|---|---|---|---|---|---|
| MLPctrl-H32 flat-MMD (no circuit) | 11 | 33996 | 0.0018 +- 0.0001 | 0.1009 +- 0.0017 | 0.0328 +- 0.0018 | 0.9276 +- 0.0058 |
| MLPctrl-H4 flat-MMD (no circuit) | 11 | 5184 | 0.0040 +- 0.0008 | 0.1389 +- 0.0095 | 0.0769 +- 0.0121 | 0.8702 +- 0.0123 |
| MLPctrl-H32 KL (no circuit) | 11 | 33996 | 0.0014 +- 0.0001 | 0.1171 +- 0.0035 | 0.0467 +- 0.0042 | 0.9281 +- 0.0045 |
| MLPctrl-H4 KL (no circuit) | 11 | 5184 | 0.0047 +- 0.0010 | 0.1691 +- 0.0109 | 0.1204 +- 0.0169 | 0.8720 +- 0.0122 |
| IQP-mlpH32-M4 (circuit, same latent) | 11 | 33996 | 0.0009 +- 0.0000 | 0.0866 +- 0.0024 | 0.0249 +- 0.0019 | 0.9076 +- 0.0015 |
| IQP-affine-M4 (Table 2) | 11 | 5164 | 0.0030 +- 0.0002 | 0.0891 +- 0.0024 | 0.0274 +- 0.0022 | 0.9102 +- 0.0026 |
| RBM-10bit (Table 2) | 11 | 11010 | 0.0017 +- 0.0008 | 0.0654 +- 0.0081 | 0.1106 +- 0.0388 | -- |
| DCGAN-32 (Table 2) | 11 | 199837 | 0.0011 +- 0.0006 | 0.0509 +- 0.0151 | 0.0388 +- 0.0104 | -- |
| | | | | | | |
| MLPctrl-H32 flat-MMD (no circuit) | 100 | 34352 | 0.0017 +- 0.0002 | 0.0778 +- 0.0031 | 0.0372 +- 0.0038 | 0.9590 +- 0.0045 |
| MLPctrl-H4 flat-MMD (no circuit) | 100 | 5540 | 0.0049 +- 0.0010 | 0.1031 +- 0.0082 | 0.0900 +- 0.0152 | 0.8919 +- 0.0161 |
| MLPctrl-H32 KL (no circuit) | 100 | 34352 | 0.0015 +- 0.0001 | 0.0848 +- 0.0034 | 0.0591 +- 0.0051 | 0.9566 +- 0.0033 |
| MLPctrl-H4 KL (no circuit) | 100 | 5540 | 0.0059 +- 0.0012 | 0.1136 +- 0.0080 | 0.1230 +- 0.0152 | 0.8864 +- 0.0170 |
| IQP-mlpH32-M4 (circuit, same latent) | 100 | 34352 | 0.0018 +- 0.0001 | 0.0700 +- 0.0031 | 0.0193 +- 0.0037 | 0.9840 +- 0.0003 |
| IQP-affine-M4 (Table 2) | 100 | 5520 | 0.0029 +- 0.0003 | 0.0716 +- 0.0029 | 0.0255 +- 0.0044 | 0.9684 +- 0.0004 |
| RBM-10bit (Table 2) | 100 | 11010 | 0.0009 +- 0.0001 | 0.0587 +- 0.0044 | 0.0807 +- 0.0221 | -- |
| DCGAN-32 (Table 2) | 100 | 199837 | 0.0006 +- 0.0001 | 0.0222 +- 0.0031 | 0.0212 +- 0.0041 | -- |
| | | | | | | |

## 2. Control vs its own quantum counterpart, at 1 sd

`separated` = the mean +- 1 sd intervals do not overlap.  The comparison is against
**IQP-mlpH32-M4 (circuit, same latent)**, which differs from the controls only by the generator.

| arm | N | metric | this arm | IQP-mlpH32-M4 (circuit, same latent) | verdict |
|---|---|---|---|---|---|
| MLPctrl-H32 flat-MMD (no circuit) | 11 | PDF-JS | 0.0018 +- 0.0001 | 0.0009 +- 0.0000 | worse, separated by >1 sd |
| MLPctrl-H32 flat-MMD (no circuit) | 11 | Conv-MMD2 | 0.1009 +- 0.0017 | 0.0866 +- 0.0024 | worse, separated by >1 sd |
| MLPctrl-H32 flat-MMD (no circuit) | 11 | Conv-MMD2 (flux-aligned) | 0.0328 +- 0.0018 | 0.0249 +- 0.0019 | worse, separated by >1 sd |
| MLPctrl-H32 flat-MMD (no circuit) | 11 | held-out projection corr | 0.9276 +- 0.0058 | 0.9076 +- 0.0015 | better, separated by >1 sd |
| MLPctrl-H4 flat-MMD (no circuit) | 11 | PDF-JS | 0.0040 +- 0.0008 | 0.0009 +- 0.0000 | worse, separated by >1 sd |
| MLPctrl-H4 flat-MMD (no circuit) | 11 | Conv-MMD2 | 0.1389 +- 0.0095 | 0.0866 +- 0.0024 | worse, separated by >1 sd |
| MLPctrl-H4 flat-MMD (no circuit) | 11 | Conv-MMD2 (flux-aligned) | 0.0769 +- 0.0121 | 0.0249 +- 0.0019 | worse, separated by >1 sd |
| MLPctrl-H4 flat-MMD (no circuit) | 11 | held-out projection corr | 0.8702 +- 0.0123 | 0.9076 +- 0.0015 | worse, separated by >1 sd |
| MLPctrl-H32 KL (no circuit) | 11 | PDF-JS | 0.0014 +- 0.0001 | 0.0009 +- 0.0000 | worse, separated by >1 sd |
| MLPctrl-H32 KL (no circuit) | 11 | Conv-MMD2 | 0.1171 +- 0.0035 | 0.0866 +- 0.0024 | worse, separated by >1 sd |
| MLPctrl-H32 KL (no circuit) | 11 | Conv-MMD2 (flux-aligned) | 0.0467 +- 0.0042 | 0.0249 +- 0.0019 | worse, separated by >1 sd |
| MLPctrl-H32 KL (no circuit) | 11 | held-out projection corr | 0.9281 +- 0.0045 | 0.9076 +- 0.0015 | better, separated by >1 sd |
| MLPctrl-H4 KL (no circuit) | 11 | PDF-JS | 0.0047 +- 0.0010 | 0.0009 +- 0.0000 | worse, separated by >1 sd |
| MLPctrl-H4 KL (no circuit) | 11 | Conv-MMD2 | 0.1691 +- 0.0109 | 0.0866 +- 0.0024 | worse, separated by >1 sd |
| MLPctrl-H4 KL (no circuit) | 11 | Conv-MMD2 (flux-aligned) | 0.1204 +- 0.0169 | 0.0249 +- 0.0019 | worse, separated by >1 sd |
| MLPctrl-H4 KL (no circuit) | 11 | held-out projection corr | 0.8720 +- 0.0122 | 0.9076 +- 0.0015 | worse, separated by >1 sd |
| | | | | | |
| MLPctrl-H32 flat-MMD (no circuit) | 100 | PDF-JS | 0.0017 +- 0.0002 | 0.0018 +- 0.0001 | better, overlap (not distinguishable) |
| MLPctrl-H32 flat-MMD (no circuit) | 100 | Conv-MMD2 | 0.0778 +- 0.0031 | 0.0700 +- 0.0031 | worse, separated by >1 sd |
| MLPctrl-H32 flat-MMD (no circuit) | 100 | Conv-MMD2 (flux-aligned) | 0.0372 +- 0.0038 | 0.0193 +- 0.0037 | worse, separated by >1 sd |
| MLPctrl-H32 flat-MMD (no circuit) | 100 | held-out projection corr | 0.9590 +- 0.0045 | 0.9840 +- 0.0003 | worse, separated by >1 sd |
| MLPctrl-H4 flat-MMD (no circuit) | 100 | PDF-JS | 0.0049 +- 0.0010 | 0.0018 +- 0.0001 | worse, separated by >1 sd |
| MLPctrl-H4 flat-MMD (no circuit) | 100 | Conv-MMD2 | 0.1031 +- 0.0082 | 0.0700 +- 0.0031 | worse, separated by >1 sd |
| MLPctrl-H4 flat-MMD (no circuit) | 100 | Conv-MMD2 (flux-aligned) | 0.0900 +- 0.0152 | 0.0193 +- 0.0037 | worse, separated by >1 sd |
| MLPctrl-H4 flat-MMD (no circuit) | 100 | held-out projection corr | 0.8919 +- 0.0161 | 0.9840 +- 0.0003 | worse, separated by >1 sd |
| MLPctrl-H32 KL (no circuit) | 100 | PDF-JS | 0.0015 +- 0.0001 | 0.0018 +- 0.0001 | better, separated by >1 sd |
| MLPctrl-H32 KL (no circuit) | 100 | Conv-MMD2 | 0.0848 +- 0.0034 | 0.0700 +- 0.0031 | worse, separated by >1 sd |
| MLPctrl-H32 KL (no circuit) | 100 | Conv-MMD2 (flux-aligned) | 0.0591 +- 0.0051 | 0.0193 +- 0.0037 | worse, separated by >1 sd |
| MLPctrl-H32 KL (no circuit) | 100 | held-out projection corr | 0.9566 +- 0.0033 | 0.9840 +- 0.0003 | worse, separated by >1 sd |
| MLPctrl-H4 KL (no circuit) | 100 | PDF-JS | 0.0059 +- 0.0012 | 0.0018 +- 0.0001 | worse, separated by >1 sd |
| MLPctrl-H4 KL (no circuit) | 100 | Conv-MMD2 | 0.1136 +- 0.0080 | 0.0700 +- 0.0031 | worse, separated by >1 sd |
| MLPctrl-H4 KL (no circuit) | 100 | Conv-MMD2 (flux-aligned) | 0.1230 +- 0.0152 | 0.0193 +- 0.0037 | worse, separated by >1 sd |
| MLPctrl-H4 KL (no circuit) | 100 | held-out projection corr | 0.8864 +- 0.0170 | 0.9840 +- 0.0003 | worse, separated by >1 sd |
| | | | | | |

## 3. Diagnostics

| arm | N | fit gain | nonlinearity ratio | \|\|z\|\| | train corr | proj. corr |
|---|---|---|---|---|---|---|
| MLPctrl-H32 flat-MMD (no circuit) | 11 | 0.9227 +- 0.0178 | 0.4651 +- 0.0550 | 0.7010 +- 0.0198 | 0.9848 +- 0.0036 | 0.9276 +- 0.0058 |
| MLPctrl-H4 flat-MMD (no circuit) | 11 | 0.5947 +- 0.0717 | 0.3741 +- 0.0962 | 0.3632 +- 0.0234 | 0.9124 +- 0.0160 | 0.8702 +- 0.0123 |
| MLPctrl-H32 KL (no circuit) | 11 | 0.8407 +- 0.0212 | 0.3838 +- 0.0656 | 0.5703 +- 0.0382 | 0.9822 +- 0.0034 | 0.9281 +- 0.0045 |
| MLPctrl-H4 KL (no circuit) | 11 | 0.5079 +- 0.0709 | 0.3939 +- 0.1532 | 0.3116 +- 0.0346 | 0.9089 +- 0.0175 | 0.8720 +- 0.0122 |
| IQP-mlpH32-M4 (circuit, same latent) | 11 | 0.9972 +- 0.0003 | 0.4915 +- 0.0393 | 0.0802 +- 0.0088 | 0.9994 +- 0.0001 | 0.9076 +- 0.0015 |
| IQP-affine-M4 (Table 2) | 11 | 0.9665 +- 0.0008 | -- | 0.0257 +- 0.0016 | 0.9926 +- 0.0001 | 0.9102 +- 0.0026 |
| RBM-10bit (Table 2) | 11 | -- | -- | -- | -- | -- |
| DCGAN-32 (Table 2) | 11 | -- | -- | -- | -- | -- |
| | | | | | | |
| MLPctrl-H32 flat-MMD (no circuit) | 100 | 0.8280 +- 0.0213 | 0.5751 +- 0.0228 | 0.9991 +- 0.0753 | 0.9680 +- 0.0042 | 0.9590 +- 0.0045 |
| MLPctrl-H4 flat-MMD (no circuit) | 100 | 0.4810 +- 0.0742 | 0.4045 +- 0.0383 | 0.3391 +- 0.0084 | 0.8958 +- 0.0162 | 0.8919 +- 0.0161 |
| MLPctrl-H32 KL (no circuit) | 100 | 0.7611 +- 0.0190 | 0.5229 +- 0.0161 | 0.7774 +- 0.0140 | 0.9640 +- 0.0034 | 0.9566 +- 0.0033 |
| MLPctrl-H4 KL (no circuit) | 100 | 0.4279 +- 0.0678 | 0.3472 +- 0.0342 | 0.2867 +- 0.0150 | 0.8894 +- 0.0166 | 0.8864 +- 0.0170 |
| IQP-mlpH32-M4 (circuit, same latent) | 100 | 0.9656 +- 0.0010 | 0.5596 +- 0.0213 | 0.0932 +- 0.0035 | 0.9931 +- 0.0002 | 0.9840 +- 0.0003 |
| IQP-affine-M4 (Table 2) | 100 | 0.8715 +- 0.0012 | -- | 0.0173 +- 0.0002 | 0.9746 +- 0.0002 | 0.9684 +- 0.0004 |
| RBM-10bit (Table 2) | 100 | -- | -- | -- | -- | -- |
| DCGAN-32 (Table 2) | 100 | -- | -- | -- | -- | -- |
| | | | | | | |

`fit gain` = 1 - (data loss at 4,000 steps) / L0, L0 being the same objective's loss at
theta0 with z = 0.  It is comparable only WITHIN one objective and one generator: the
KL and flat-MMD losses are different quantities, and so are the softmax's and the
circuit's.  `nonlinearity ratio` is the relative residual of the best affine fit to the
learned map on the trained latents (0 = the MLP collapsed to an affine map).

## 4. Answer

Across the 32 (control arm, N, metric) comparisons in section 2: **3 separated wins**
for the circuit-free control, **28 separated losses**, and 1 not distinguishable at
1 sd.

## 5. Provenance

| what | where |
|---|---|
| control rows | `experiments/13_iqp_turbulence_replication/results_seeds_mlpctrl.jsonl` |
| quantum counterpart | `experiments/13_iqp_turbulence_replication/results_seeds_gridmlp.jsonl` |
| Table 2 rows | `experiments/13_iqp_turbulence_replication/results_seeds_table2_v11.jsonl` |
| trainer | `experiments/13_iqp_turbulence_replication/train_mlpctrl.py` |
| grid | `experiments/13_iqp_turbulence_replication/seeds/submit_seeds_mlpctrl.sbatch` |
| run outputs | `<WORK>/gsoc_iqp_gridmlp/ctrl/` |

80 scored runs across 16 (arm, N) cells.
