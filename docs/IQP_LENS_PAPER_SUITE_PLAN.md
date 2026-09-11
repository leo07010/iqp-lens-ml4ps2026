# Plan — the full analysis suite of arXiv 2603.06440, applied to the DeepLense lens images

Written 2026-09-07 before running. Scope set by <user>: compare only against the baselines the paper used (RBM, DCGAN);
zero-learning controls and the PCA generator are not part of these comparisons. Single seed.

| Paper analysis | Here | Job |
|---|---|---|
| Fig. 5 map: (I_QCLI, I_CCI) of datasets, "IQP-compatible" = high/high | lens in both training representations (18-bit (x,y,z) tuples; 10-bit centred-crop pmf), turbulence D2, binarized MNIST (Ref [38]'s 16-qubit and 484-qubit versions if reproducible), BinaryBlobs (Ref [38] definition), each with an i.i.d.-uniform null at matched sample count; D-Wave and RCS omitted (no data) | CPU |
| Fig. 3: synthetic IQP targets, learner MMD vs target QCLI | 16 qubits, data-generating 4-local IQP (140–1050 gates, SPSA to spread QCLI), 10^4 samples per target, learner 2-local with 50/100/150 gates, Adam 5000 steps lr 1e-4, exact simulation | dev |
| Fig. 4: circuits optimised for high QCLI, CCI follows | 8/12/16 qubits, 150 random 2-local gates, maximise I_QCLI, record I_CCI | dev |
| Sec. III generative experiment (Algorithm 1, 11 vs 100 samples, RBM/DCGAN, PDF–JS + Conv2D-MMD²) | done: rounds 3 / 3b (`docs/IQP_LENS_ROUND3_RESULTS.md`), paper-style table restricted to IQP vs RBM vs DCGAN | done |
| Table 2 qualitative samples + unseen-time synthesis by latent interpolation | pseudo-time arm (ellipticity e1 as the time axis): generated crops along e1 vs the nearest real crops; latent trajectory plot | CPU |
| DCGAN data-scaling N ∈ {10, 30, 50, 70, 90, 100} | DCGAN-32 on the crops, plus IQP-affine-M4-MMDflat-resample and RBM-10bit at the same N | dev |
| Table 1 (literature numbers), Appendix H theorem, hardware deployment | not experiments / no hardware | omitted |

Predictions: Q1 the lens data sit in the paper's upper-right "IQP-compatible" region in both representations and
above every reference dataset on I_QCLI. Q2 the Fig. 3 relation (higher target QCLI → lower learner MMD envelope) is
reproduced in our exact simulator. Q3 Fig. 4's QCLI–CCI coupling is reproduced. Q4 in the scaling study IQP's MMD²
is flat in N while DCGAN improves with N and overtakes IQP between N = 30 and 50 (raw MMD²).
