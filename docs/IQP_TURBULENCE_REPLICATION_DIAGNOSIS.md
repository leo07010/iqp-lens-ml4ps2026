> **Correction added 2026-09-05 (after jobs 342542 / 342685; see IQP_TURBULENCE_REPLICATION_RESULTS.md §3(d)).**
> The "anchor corr 0.69" quoted in Q2 is the correlation of the *sampled-and-mean-aggregated* field; the same model's
> *exact* conditional-mean field reaches 0.77 on anchor t = 1. The statement "trained IQP sits exactly on the
> order-≤3 truncation" holds only for sampled fields; on exact fields the model (0.77) is clearly better than the
> truncation (0.69). Protocol attribution shows the main-run deficit comes from Algorithm 1 itself: freezing the
> latent block (all 18 single-qubit terms) during stage 1 costs ≈ 0.08, the 5×10^4-tuple subsample ≈ 0.04, and the
> stage-2 latent adaptation recovers nothing (+0.0001); with all parameters trained the ceiling is ≈ 0.88–0.89 for
> every kernel tried, so the kernel's low-order weighting explains the *shape* of the high-order error, not the bulk
> of the correlation gap. Everything else in this document stands.

# Root-cause diagnosis: why our arXiv 2603.06440 replication (SLURM 342220) does not reproduce Table 2

Scope: analysis only. No file in
`<HOME>/gsoc2026-deeplense-qml/experiments/13_iqp_turbulence_replication/` was modified.
All numbers below come from actual runs of the scripts in this directory (CPU, `torch.set_num_threads(4)`,
python 3.9.21), reading `outputs_d2_main/`, `results_d2_main.jsonl` and the D2 cache
`<WORK>/data/turbulence_khojasteh2022/d2_cache/`.

Scripts: `prep.py`, `common.py`, `q1_ceiling.py`, `q1b_tuple_level_js.py`, `q1c_agg_convention.py`,
`q2_anchor_fit.py`, `q2b_loss_decomp.py`, `q2c_order_cutoff.py`, `q3_sens.py`,
`q4_task_triviality.py`, `q5_dcgan.py`, `q6_figure.py`, `q6b_metric_scale.py`.

---

## Q1. Pipeline ceiling — sampling/aggregation is NOT the bottleneck

Perfect model = the ground-truth empirical pmf of each held-out snapshot's native code pool;
draw `n` codes with `codec.sample_codes`, aggregate with `codec.aggregate`, score with
`metrics.pdf_js` / `metrics.conv_mmd2` against the 200 held-out GT fields (identical to the real
evaluation path). `fast_aggregate` in `common.py` was verified **bit-identical** to `codec.aggregate`
(max abs diff 0.000e+00).

| n_synth | PDF–JS | Conv-MMD² | RMSE | mean corr | field std (GT 0.4460) |
|---|---|---|---|---|---|
| 10⁴ | 0.00209 | 0.01102 | 0.0424 | 0.99548 | 0.4464 |
| **10⁵ (the setting actually used)** | **0.00004** | **0.00022** | 0.0115 | 0.99967 | 0.4462 |
| 10⁶ | 0.00002 | 0.00001 | 0.0036 | 0.99997 | 0.4460 |
| 10⁷ | 0.00002 | 0.00000 | 0.0011 | 1.00000 | 0.4460 |

Sanity: GT vs GT = (0.0, 0.0).

Why the ceiling is so good: the native D2 grid is 328×308 = 101,024 points binned into 64×64,
i.e. **every one of the 4096 cells is covered**, count min/max/mean = 16 / 30 / 24.66. The
*within-cell* z spread is 0.0303 while the *between-cell* spread is 0.4487 — the field is smooth
at 64×64 resolution, so a 24-sample cell mean has standard error ≈ 0.006, 1.4 % of the signal.

**Conclusion Q1.** At n_synth = 10⁵ the pipeline ceiling is PDF–JS 4e-5 and MMD² 2e-4. IQP's
0.235 / 0.834 is **4 orders of magnitude** above the ceiling. Aggregation noise explains nothing;
increasing n_synth cannot help. (CTRL-interp's 0.0012 / 0.0022 is ~30× the ceiling, i.e. genuinely
near-perfect.)

### Q1b/Q1c. But one *unspecified* convention does explain the PDF–JS gap

`docs/IQP_REPLICATION_SPEC_2603.06440.md` §11 unknown #10 records that the paper never defines the
aggregation rule ("平均/眾數/最後一筆覆蓋"). We implemented **mean over the samples in a cell**.
Re-scoring the *same* IQP samples under **last-write-wins** (one sample per cell, empties filled the
same way), 60 held-out τ:

| tuple source | MEAN agg PDF–JS | MEAN agg MMD² | LAST-WRITE PDF–JS | LAST-WRITE MMD² |
|---|---|---|---|---|
| perfect (GT pmf) | 0.00006 | 0.00158 | 0.00438 | 0.08618 |
| IQP-63k | 0.23632 | 0.83505 | **0.04463** | 0.69406 |
| IQP-31k | 0.26457 | 0.83515 | **0.02812** | 0.69484 |

Paper Table 2: IQP-31k 0.023784, IQP-63k 0.022084. Equivalently, the PDF–JS of the model's **raw
tuple z-marginal** (no field at all) is 0.0141 (IQP-31k) / 0.0294 (IQP-63k) — also bracketing 0.024.

**So the PDF–JS column is a convention mismatch, not a defect of our IQP.** Caveat: this convention
would also improve the RBM (whose tuples we did not save), and it does **not** rescue the MMD² column
(0.69 is still at "structure destroyed" level, see Q4/Q6b).

---

## Q2. The IQP fails on its own TRAINING anchors — this is not a latent-interpolation problem

Reconstructed each anchor from `theta_core.npy` + that anchor's `theta_lat_anchors.npy` row, sampled
10⁵ codes, aggregated identically.

| arm | anchor-set PDF–JS | anchor-set MMD² | held-out PDF–JS | held-out MMD² |
|---|---|---|---|---|
| IQP-63k (K=7, seed 0) | **0.2338** | **0.7786** | 0.2354 | 0.8340 |
| IQP-31k (K=6, seed 0) | **0.2668** | **0.7623** | 0.2660 | 0.8188 |

Training and held-out scores are **the same to 1 %**. Piecewise-linear θ_lat interpolation is
essentially free of cost; the model simply never fit the data.

Per-anchor detail (IQP-63k):

| t | corr(sampled field, GT) | corr(exact E[z|x,y], GT) | MMD² vs data | model p(x,y)·4096 min/max (uniform = 1) | model cv | data cv |
|---|---|---|---|---|---|---|
| 1 | 0.711 | 0.791 | 8.70e-05 | 0.334 / 3.932 | 0.295 | 0.134 |
| 500 | 0.695 | 0.781 | 1.35e-04 | 0.459 / 3.216 | 0.278 | 0.134 |
| 1000 | 0.662 | 0.742 | 2.68e-04 | 0.425 / 3.603 | 0.259 | 0.134 |

mean over 11 anchors: corr(sampled) 0.691, corr(exact conditional mean) 0.776.
Using the model's **exact** conditional mean (no sampling noise at all) only lifts corr from 0.69 to
0.78 — again confirming Q1: sampling is not the problem.

(a) **z-marginal is fine.** ⟨Z_j⟩ for the six z qubits, t=1: model [0.128, 0.125, 0.154, −0.023,
0.294, 0.262] vs data [0.120, 0.115, 0.162, −0.027, 0.305, 0.267]. 64-bin k_z histogram JS(model,data)
= 0.0142 (K6) … 0.0466 (K7, t=1000). The x/y qubit ⟨Z_j⟩ are ~0 in both (the one non-zero,
y-bit 6 = −0.0305 from the 328-row grid not dividing into 64 bins, is reproduced as −0.030).
So the codec/bit-order/scale plumbing is correct.

(b) **The joint is not.** ⟨Z_s⟩ RMS error vs data RMS by Pauli weight (IQP-63k, t=1):

| \|s\| | 1 | 2 | 3 | 4 | 6 |
|---|---|---|---|---|---|
| RMS error | 0.0056 | 0.0144 | 0.0190 | 0.0182 | 0.0134 |
| data RMS | 0.1107 | 0.0610 | 0.0456 | 0.0287 | 0.0156 |
| relative error | **5 %** | 24 % | 42 % | 63 % | **86 %** |

### Q2b. Why: the closed-form MMD objective is blind to exactly the orders that carry the field

Kernel Walsh weights c_s = mean over the three `auto3` bandwidths of α^(n−|s|)β^|s|
(σ = 1.411, 2.344, 3.000 → mean Pauli weights 2.00, 0.78, 0.49):

| \|s\| | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| c_s | 3.93e-01 | 1.75e-02 | 1.09e-03 | 9.66e-05 | 1.05e-05 | 1.25e-06 | 1.54e-07 | 1.91e-08 |

**Kernel-weighted** share of the data's spectral energy: |s|=1 82.5 %, 2 13.3 %, 3 3.5 %,
**|s|≥4 together 0.68 %**.
**Unweighted** share of the same data's ⟨Z_s⟩² energy: |s|=1 0.6 %, 2 1.7 %, 3 4.9 %, 4 7.4 %,
5 10.0 %, 6 13.1 %, 7 15.6 %, … **46.8 % lies above |s| = 7**.
The objective spends 99.3 % of its weight on the 7.2 % of the data that is low-order.

MMD² against t=1 under the training kernel: trained IQP-63k 8.70e-05; **exact product-Bernoulli
(1-body only) 8.52e-04**; uniform 4.67e-03. Optimizer noise floor (MMD² between two independent
50k-tuple empirical pmfs of the same snapshot) = 1.12e-05. So the trained model is 10× better than a
product-Bernoulli and 7× above the estimator's own noise floor — it is doing roughly what the
objective asks, and the objective is nearly satisfied.

Loss split by |s| at the optimum: |s|=1 11 %, 2 40 %, 3 33 %, 4 12 %, 5 3 %, ≥6 <1 %.

### Q2c. Decisive check (no training required)

Project the **exact data pmf** onto Walsh order ≤ L, clip/renormalize, then push it through the
identical 10⁵-sample + aggregate + score pipeline (200 held-out τ):

| order L | PDF–JS | Conv-MMD² | corr(paired, vs GT at same τ) | field std |
|---|---|---|---|---|
| 1 | 0.4428 | 0.828 | 0.001 | 0.113 |
| 2 | 0.3902 | 0.860 | 0.130 | 0.129 |
| **3** | **0.1824** | **0.932** | **0.688** | **0.240** |
| 4 | 0.1976 | 0.901 | 0.795 | 0.243 |
| 5 | 0.1691 | 0.925 | 0.870 | 0.260 |
| 7 | 0.1569 | 0.916 | 0.916 | 0.267 |
| 9 | 0.1339 | 0.926 | 0.946 | 0.291 |
| 12 | 0.0690 | 0.972 | 0.981 | 0.365 |
| 18 (exact) | 0.00006 | 0.00025 | 0.9997 | 0.446 |
| — trained IQP-63k | 0.2354 | 0.834 | 0.693 | 0.219 |
| — trained IQP-31k | 0.2660 | 0.819 | 0.669 | 0.202 |
| — RBM-12k | 0.2062 | 0.706 | 0.751 | 0.270 |

The trained IQP sits **exactly on the order-≤3 truncation of the true data** (corr 0.69 vs 0.688,
std 0.22 vs 0.24). That is the signature of an optimum of *this* objective, not of a bug.

Also note the second column: Conv-MMD² is **non-monotone and anti-correlated with fidelity** over the
whole range these models occupy — the order-1 model (corr 0.001) scores *better* (0.828) than the
order-12 model (corr 0.981, MMD² 0.972). See Q6b.

---

## Q3. Sensitivity arms — none of them rescue the result

| run | PDF–JS | MMD² | key change | final stage-1 loss |
|---|---|---|---|---|
| iqp_K6_seed0/1/2 | 0.266 / 0.271 / 0.273 | 0.819 / 0.823 / 0.830 | main K=6 | 6.1e-05 |
| iqp_K7_seed0/1/2 | 0.235 / 0.238 / 0.237 | 0.834 / 0.811 / 0.804 | main K=7 | 7.7e-05 |
| **iqp_sens_lat_init_data** | **0.188** | 0.811 | θ_lat⁽¹⁾ from data-init instead of N(0,0.01) | **2.37e-05** |
| iqp_sens_sigma_ref38_16q | 0.211 | 0.885 | σ = 0.6, 0.3 (mean Pauli weight 6.8, 9.0) | 1.19e-04 |
| iqp_sens_full_grid | 0.244 | 0.826 | full native pool instead of 50k draws | 4.54e-05 |
| iqp_sens_random_core_init | 0.250 | 0.796 | random core init | 4.45e-05 |
| iqp_sens_sigma_meanweight2_6 | 0.248 | 0.816 | σ mean weights 2.0, 6.0 | 1.64e-04 |
| iqp_sens_lr1e-3 | 0.287 | 0.819 | lr 1e-3 | 2.48e-04 (worse) |

Best arm = `iqp_sens_lat_init_data`, PDF–JS 0.188 (−29 % vs its K=6 baseline 0.266), MMD² unchanged.
It also has the lowest training loss (2.4e-05), so the ranking is consistent. Second best is the
high-order-kernel arm `iqp_sens_sigma_ref38_16q` (0.211), which is the direction Q2b predicts,
but it costs MMD² (0.885) and is still 5000× above the pipeline ceiling.

Convergence (stage-1 loss curve): K7 dropped only 0.28 % over the last 5000 of 30000 steps
(K6 3.96 %, lr1e-3 0.03 %) — the optimizer is converged, more steps will not help.

---

## Q4. The evaluation design cannot distinguish models — this is the paper's problem, not ours

**(i) The D2 field is nearly frozen over the whole 1000-snapshot window.**

| adjacent anchors | corr | RMS diff | RMS / field std |
|---|---|---|---|
| t=1→100 | 0.99302 | 0.0530 | 0.119 |
| t=500→600 | 0.99254 | 0.0545 | 0.122 |
| t=900→1000 | 0.99033 | 0.0620 | 0.139 |

Correlation vs lag (all pairs of the 211 loaded snapshots): lag ≤5 → 1.0000; 21–50 → 0.9986;
101–200 → 0.9852; 501–1000 → **0.9293**. Even the two most distant snapshots in the dataset are
93 % correlated.

**(ii) Therefore "unseen-time synthesis" is trivially solvable with no model.**

| predictor for a held-out τ | mean corr | min corr | RMS | RMS as % of field std |
|---|---|---|---|---|
| linear interp of bracketing anchors (= CTRL-interp) | 0.99978 | 0.99936 | 0.0085 | **1.9 %** |
| nearest anchor, copied verbatim | 0.99917 | 0.99693 | 0.0159 | 3.6 % |
| constant global mean | 0.0 | — | 0.4460 | 100 % |

**(iii) Worse: neither metric is time-aware at all.** Both are set-level distributional statistics
(`pdf_js` pools all pixels of all 200 fields; `conv_mmd2` compares two 200-element sets).

| "generator" | PDF–JS | Conv-MMD² |
|---|---|---|
| the 200 GT fields, **randomly permuted across τ** | **0.00000** | **0.00000** |
| nearest-anchor copy (11 distinct images reused) | 0.00006 | 0.00052 |
| copy anchor t=1 for every τ (one image, 200×) | 0.00259 | 0.51127 |

A model that emits the right *set* of fields with completely scrambled time labels scores a perfect
0 / 0. Time conditioning — the entire point of Algorithm 1's latent adaptation — is **not measured**.

---

## Q5. Why DCGAN@11 did not collapse for us

Logs (`logs_d2/dcgan_*_n11_*.log`) carry only 2 lines each (arch line + done line); no loss curve was
written, so this is inferred from the outputs. All DCGAN arms: 20000 generator steps, lr 2e-4,
batch = min(16, n_train), 200 unconditional samples, ~90 s wall-clock, identical for n11 and n100.

| arm | self-corr among its 200 outputs | field std | mean max-corr to any of the 11 training anchors | corr to GT at the same τ |
|---|---|---|---|---|
| GT held-out | 0.9640 | 0.4460 | 0.9992 | 1.0000 |
| dcgan_1.2m_n11_seed0 | 0.9633 | 0.4453 | **0.9944** | 0.9639 |
| dcgan_1.2m_n11_seed2 | 0.8468 | 0.2674 | 0.3255 | 0.3038 |
| dcgan_1.2m_n100_seed0 | 0.9663 | 0.4291 | 0.9980 | 0.9655 |
| dcgan_0.2m_n11_seed0 | **0.9989** | 0.5269 | 0.9333 | 0.9210 |
| dcgan_51k_n11_seed0 | 0.9964 | 0.4548 | 0.8905 | 0.8797 |
| iqp_K7_seed0 | 0.7556 | 0.2194 | 0.6987 | 0.6925 |
| ctrl_interp | 0.9662 | 0.4456 | 0.9992 | 0.9925 |
| ctrl_marginal_seed0 | 0.0001 | 0.1182 | 0.0061 | 0.0016 |

The 11 training snapshots are 99.3 % correlated with each other **and** 99.9 % correlated with every
held-out target (Q4). So memorising the 11 images *is* the optimal solution, and the small DCGANs'
near-total mode collapse (self-corr 0.9989 for 0.2M@11) is invisible to a metric that only compares
the two sets' feature distributions. DCGAN@11 "works" here because the task has no diversity to
capture. One of the three seeds (1.2M@11 seed 2) *did* destabilise (max-corr to anchors 0.33,
PDF–JS 0.153, MMD² 0.898) — the paper's instability is present, just not in the median seed.

---

## Q6b. Metric-scale calibration (what the numbers actually mean)

Reference "generators" scored against the same 200 held-out GT fields:

| reference generator | PDF–JS | Conv-MMD² |
|---|---|---|
| identical to GT | 0.00000 | 0.00000 |
| GT shuffled across τ | 0.00000 | 0.00000 |
| GT + N(0, 0.05) noise | 0.03466 | 0.02734 |
| **GT with each field's pixels spatially shuffled** | **0.00000** | **0.70388** |
| constant = global mean | 0.92290 | 0.80949 |
| iid N(GT mean, GT std) | 0.10814 | 0.42936 |
| iid uniform[a_z, b_z] noise | 0.19205 | 0.92664 |

* **PDF–JS is exactly blind to spatial structure** (shuffled pixels → 0.00000). It is a pure
  contrast/histogram test. GT field std 0.4460 vs IQP 0.219, RBM 0.270, CTRL-marginal 0.118,
  DCGAN-1.2M 0.445, CTRL-interp 0.446 — the PDF–JS ranking is almost entirely explained by how much
  contrast survives cell-averaging.
* **Conv-MMD² saturates above ≈0.70.** IQP 0.83, CTRL-marginal 0.82, RBM 0.71 all sit at or past the
  "spatially destroyed" level, so the ordering "IQP 0.82 vs RBM 0.73" carries no information. It is
  also non-monotone in true fidelity (Q2c).
* For reference, **every value in the paper's own Table 2 MMD² column (0.895 – 1.472, best IQP 0.942)
  is above our spatially-shuffled reference of 0.704 and close to our iid-uniform-noise reference of
  0.927**, whereas our CTRL-interp reaches 0.002. Caveat: absolute MMD² is implementation-dependent
  (encoder init, normalization), so this is indicative, not a proof — but *within our implementation
  of their spec* the metric clearly can reach ~0, and none of the paper's models do.

---

## Root-cause verdict

**Both, and they are separable.**

1. **Not our bug (evidence).** `pytest test_iqp_exact.py` → 8 passed. `fast_aggregate` is bit-identical
   to `codec.aggregate`. Pipeline ceiling 4e-5 / 2e-4 at the settings used. a_z/b_z = (0, 2.034849),
   the native anchor bounds (the earlier `.field`-derived-range bug is fixed and visible in the metas).
   The six z-qubit ⟨Z_j⟩ and the k_z histogram match the data (JS 0.014–0.047), so codec bit order and
   scale are right. Training is converged (0.28 % drop over the last 5k steps) and reaches 10× below a
   product-Bernoulli and 7× above the estimator noise floor.

2. **Our IQP is a faithful optimum of the paper's stated objective — which cannot fit this data.**
   The closed-form Gaussian-kernel MMD puts 99.3 % of its weight on |s| ≤ 3, while 93 % of D2's Walsh
   energy (47 % above |s| = 7) lives higher. The trained model reproduces the order-≤3 projection of
   the data almost exactly (corr 0.693 vs 0.688; std 0.219 vs 0.240). The one lever the paper leaves
   unspecified (§11 unknown #8: the training kernel's σ) moves PDF–JS only 0.266 → 0.211.

3. **The PDF–JS discrepancy specifically is a convention mismatch, not a modelling failure.** The paper
   never defines the aggregation rule (§11 unknown #10). Under last-write-wins our IQP-31k scores
   0.028 and IQP-63k 0.045 vs the paper's 0.024 / 0.022. Under mean-aggregation it scores 0.26 / 0.24.
   Same samples, same metric code, 9× difference.

4. **The paper's evaluation design cannot support its claim, under either convention.** The held-out
   task is solved to corr 0.9998 by zero-parameter interpolation; both metrics are invariant to
   permuting τ (0.00000 / 0.00000 for shuffled GT), so time conditioning is never tested; PDF–JS is
   exactly invariant to spatial shuffling; Conv-MMD² saturates above 0.70 and is anti-correlated with
   fidelity across the range every model in Table 2 occupies.

**Net:** our replication is sound; the numbers differ from Table 2 for one identifiable reason
(aggregation convention) on PDF–JS, and are simply not comparable on MMD² because the metric is
saturated for every model in both the paper's table and ours. The scientifically defensible claim is
about the benchmark, not about IQP vs RBM vs DCGAN.

---

## Figure

`<HOME>/gsoc2026-deeplense-qml/docs/figures/iqp_turbulence_replication.png`
(rows: GT / IQP-63k / RBM-12k / DCGAN-1.2M@11 / CTRL-interp / CTRL-marginal at τ = 101, 502, 947,
shared colour scale 0–1.70; right panels: pixel-intensity histogram over all 200 held-out τ, and a
metric-scale calibration bar chart with the 0.704 "structure destroyed" reference line.)
