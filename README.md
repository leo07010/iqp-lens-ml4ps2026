> **Current version: [`paper/v12/main.pdf`](paper/v12/main.pdf)** (9 pages: 4 body, references, 3 appendix; non-linear latent map as the main arm, sampled flux, five seeds; claims, citations and figures checked in three independent review passes, see paper/v12/CITATION_CHECK.md and REFS_VERIFICATION.md). `paper/v11/main.pdf` is the previous five-seed version and (five seeds, mean ± sd, metric protocol v11, baselines with validation early stopping). `paper/main.pdf` is the superseded **v10** (single seed, legacy metric protocol, baselines at a fixed 20,000 steps) and is kept only for the record; its Table 2 differs from v11 for those three reasons.

# IQP generative modeling of strong-lensing images — code release

Code, results, and documentation accompanying an ML4PS 2026 (NeurIPS workshop)
submission that replicates and stress-tests the IQP (instantaneous quantum
polynomial-time circuit) generative-modeling protocol of arXiv:2603.06440 on
two datasets: 64x64 DeepLense strong-lensing images (the paper's main results)
and the 2D turbulence dataset of Khojasteh et al. 2022 (an independent
replication attempt, reported as a negative result in the paper).

All experiments are single-seed, exact classical simulation of the quantum
circuits (via the fast Walsh–Hadamard transform, `code/iqp_exact.py`) — no
hardware and no sampling-noise approximation is involved anywhere in this
repository.

## What is (and is not) in this folder

```
paper/      the submission itself, in all three revisions that were compiled:
            paper/          v10 (superseded, single seed, legacy metrics)
            paper/v11/      v11 (five seeds, metric protocol v11)
            paper/v12/      v12 (current: non-linear latent map, sampled flux)
            each holds main.tex, main.pdf, refs.bib, the NeurIPS style files,
            NUMBERS_PROVENANCE.md and its own figures/ directory
            paper/seed_tables/  the seed-aggregated tables (seed_tables*.md)
                            and the make_seed_tables_*.py scripts that build
                            them from results/
            paper/figures_all/  the complete figure set: every figure script
                            (*.py) with its rendered *.pdf/*.png, including
                            panels that were cut from the final PDFs, plus
                            FIGURE_PROVENANCE.md
code/       every script used to run the experiments (data loading, IQP
            training/inference, classical baselines (RBM/DCGAN), evaluation,
            diagnostics, and the SLURM submission scripts used to launch them)
results/    the numeric outputs those scripts produced: per-arm metric jsonl
            logs, per-run meta.json files, and a 20-image sample of generated
            output for the six arms shown in the paper's figures.
            results/work/ holds the two sweeps that ran out of the scratch
            filesystem (the MLP suite and the MLP/affine grid)
docs/       the running research log: pre-registrations (written and
            committed BEFORE each experiment ran, with numbered predictions),
            results write-ups, the Chinese-language long-form paper and
            method-difference documents (v10/v11/v12, .md + .pdf), and a
            literature/method summary of the replicated paper
data/       README only — raw data is not redistributed here, see data/README.md
```

`results/` deliberately does **not** contain the full output trees (several
GB of generated images per experiment round); it contains everything needed
to reproduce every number and figure in the paper (jsonl metric logs and
per-arm `meta.json`), plus a small illustrative sample of generated images.
Re-running the code (see "Reproducing" below) regenerates the full trees.

## Environment

- Python 3.9+ (the original runs used 3.9.21 on the login node's system
  Python; the code has no `X | None` type-hint syntax or `match` statements,
  so it also runs unmodified on 3.11+).
- Dependencies: see `requirements.txt` (numpy, torch, matplotlib, scipy — all
  CPU- or single-GPU-only; nothing here needs a torch build newer than what
  `pip install torch` gives you, and nothing needs multi-node/distributed
  training).

```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

All non-quantum baselines (RBM, DCGAN) and the IQP simulator itself are
ordinary numpy/torch code; no quantum SDK or simulator library is required
because the IQP circuits are simulated exactly via closed-form Walsh–Hadamard
transforms rather than through a general-purpose statevector/tensor-network
backend.

## Reproducing the paper's numbers

1. **Get the data.** See `data/README.md` for where each dataset comes from
   and which two path constants / environment variables (`LENS_NPZ`,
   `LENS_CACHE_DIR` for the lensing data; `--zip`/cache-dir constants in
   `code/preprocess_d2.py` for the turbulence data) you need to point at your
   own filesystem — the code ships with the original authors' cluster paths
   as defaults and will not find your data until you change those.

2. **Build the cache.** The lensing loaders (`code/data_lens.py`,
   `code/data_lens_r3.py`) build their own small cache automatically on first
   call. The turbulence loader needs one explicit pass first:
   ```
   python3 code/preprocess_d2.py --zip /your/path/2D_velocity_snapshot_sub_domain_2.zip /your/cache/dir
   ```

3. **Run the round 3b main arm** (the learned-latent-direction IQP model that
   the paper's Table 1 / Fig. 2 report as the primary result, arm
   `IQP-affine-M4-MMDflat-resample`, N=100 training images):
   ```
   cd code
   python3 train_iqp_affine_r3.py \
       --n-holdout 200 --n-gen 200 --device cuda --seed 0 --lr 0.02 --steps 4000 \
       --theta0-steps 1000 --infer-restarts 8 --infer-steps 1000 --infer-lr 0.005 \
       --infer-chunk 50 --log-every 500 \
       --n-train 100 --M 4 --objective mmdflat --prior resample \
       --out outputs_lens_r3b/iqp_m4_mmdflat_resample_n100
   ```
   `code/submit_lens_r3b.sbatch` (a SLURM job — edit `<REPO_ROOT>` and
   `<YOUR_ACCOUNT>` at the top of it before submitting) runs this arm plus the
   other ten round-3b arms (M in {2,4,8}, three training objectives, two
   latent priors) at both N=11 and N=100 back to back; `submit_lens_r3.sbatch`
   runs the 28-arm round-3 sweep (fixed-basis IQP, the RBM/DCGAN baselines,
   and four ablation controls) that the same table also draws from.

   Placeholders. For double-blind release the copies in this folder replace the original cluster paths and account with `<HOME>` (the researcher's home directory), `<WORK>` (the scratch/data root; default cache dirs such as `<WORK>/data/lens_cache` and `<WORK>/data/turbulence_khojasteh2022/d2_cache` live there), `<user>` and `<YOUR_ACCOUNT>` (the SLURM account in the `.sbatch` files). Replace them with your own paths, or override the `CACHE_DIR_DEFAULT` constants in `code/data_lens.py`, `code/data_lens_r3.py` and `code/data_d2.py`.

4. **Evaluate.** Score one or more output directories into the results jsonl
   used by the paper (PDF–JS, Conv2D-MMD², held-out projection correlation,
   nearest-neighbour-holdout check):
   ```
   python3 evaluate_r3b.py --outputs-dir outputs_lens_r3b outputs_lens_r3 \
       --n-holdout 200 --n-train-max 100 \
       --random-pair-arm IQP-affine-M4-KL-gauss --seed 0 --skip-missing \
       --results results_r3b.jsonl
   ```
   (`evaluate_r3.py` is the round-3-only equivalent that produced
   `results/results_r3.jsonl`.)

5. **Regenerate the figures.** Each figure is a standalone script that reads
   `results/` and writes its own `.pdf`/`.png` next to itself:
   ```
   cd paper/figures_all
   python3 fig1_compact.py        # and fig2_*.py, figA_*.py, figB_*.py, ...
   ```
   `paper/figures_all/FIGURE_PROVENANCE.md` lists, panel by panel, which
   `results/*.jsonl` file, `meta.json` key, or `code/` script each figure and
   number is computed from. `code/analysis_lens/make_figures.py` regenerates
   the dataset-characterisation figures in `docs/figures/`.

The other pipelines in this repository (the correlation–complexity "map" of
Fig. 1a, the two controlled-synthetic falsification studies of Appendix C/D,
the mechanism tests in `code/mech_tests/`, and the earlier pilot/round-1/2
lensing sweeps) follow the same pattern — a `train_*`/`run_*` or `fig*.py`
script writes a jsonl log or `meta.json`, an `evaluate_*.py` or `plot_*.py`
script reads it back. See `code/INTERFACES.md` for the full data-contract
reference and `docs/IQP_LENS_PAPER_SUITE_PLAN.md` for how the pieces fit
together.

## Where each paper number comes from

`paper/NUMBERS_PROVENANCE.md` (and its `paper/v11/`, `paper/v12/` revisions)
traces every number in the submitted PDF to a `docs/*.md` line or a
`results/` file+field; `paper/figures_all/FIGURE_PROVENANCE.md` does the same
for every figure panel, plus the exact rendering decisions. Short version:

| Paper section | Primary results file(s) |
|---|---|
| Abstract / correlation–complexity map (Fig. 1a) | `results/paper_suite/map/results_map.json` |
| Fig. 1b samples, Table 1, mechanism/scaling (Fig. 2) | `results/results_r3.jsonl`, `results/results_r3b.jsonl`, `results/results_scaling.jsonl`, `results/mech_tests/mech_test2/*.json` |
| Appendix B (turbulence "not reproduced") | `results/results_d2_*.jsonl` |
| Appendix C (support-mismatch control, ref. Fig. 3) | `results/outputs_controlled/results_fig3_v2.jsonl` |
| Appendix D (QCLI–CCI ceiling, ref. Fig. 4) | `results/outputs_controlled/results_fig4.jsonl` |

`docs/IQP_LENS_ROUND3_RESULTS.md`, `docs/IQP_LENS_MAP_RESULTS.md`,
`docs/IQP_LENS_PAPER_SUITE_RESULTS.md`, and `docs/IQP_TURBULENCE_REPLICATION_RESULTS.md`
are the write-ups those tables were taken from; each pre-registration
(`docs/*_PREREGISTRATION.md`) states the falsifiable predictions the
corresponding results file was scored against, written before the run.

## v11 (five seeds, metric protocol v11)

A later campaign re-scored the main IQP-vs-baseline comparison under a
revised metric protocol (`code/metrics_v11.py`, `code/evaluate_v11.py`)
across five random seeds instead of one, with an early-stopping variant and
a convergence/floor sweep for the classical baselines:

- `code/seeds/` — the five-seed training and diagnostic scripts
  (`train_iqp_affine_r3_snap.py`, `train_iqp_affine_r3_theta0obj.py`,
  `select_checkpoint.py`, `rescore_extra_metrics.py`, `floor_bootstrap.py`,
  `val_extra.py`, plus the `diag_*_seeds.py` robustness probes) and their
  `submit_seeds_*.sbatch` launchers, including the early-stopping variant
  (`submit_seeds_iqp_es.sbatch`, `submit_seeds_iqp_es_v11.sbatch`).
- `code/metrics_v11.py`, `code/evaluate_v11.py`, `code/compare_v11_tables.py`
  — the v11 metric protocol and the script that turns the seed results into
  the paper's seed tables.
- `code/aggregate_seeds.py`, `code/baseline_rbm_sweep.py`,
  `code/baseline_dcgan_sweep.py`, `code/aggregate_convergence.py`,
  `code/submit_baseline_convergence.sbatch`, `code/submit_v11_rescore.sbatch`
  — the classical-baseline convergence/floor sweep and its aggregation, and
  the batch re-scoring job for existing checkpoints under metrics_v11.
- `code/turb_aggregate_seeds.py`, `code/turb_seed_extras.py`,
  `code/turb_facts_addendum.py` — the matching five-seed pass on the
  turbulence-replication track (Appendix B).
- `code/paper_suite/map/agg_map_seeds.py`, `code/mech_tests/mech_test2/mech_seeds.py`
  + `agg_mech_seeds.py`, and `code/paper_suite/controlled/fig3_learner_seeds.py`
  + `agg_fig34_seeds.py` — the five-seed variants of the correlation-complexity
  map, mechanism test, and controlled-synthetic (Fig. 3/4) studies, each with
  its `submit_*_seeds.sbatch` launcher.
- `results/results_seeds_*.jsonl`, `results/results_v11_*.jsonl`,
  `results/results_convergence.jsonl`, `results/results_d2_seeds.jsonl`,
  `results/paper_suite/map/results_map_seed*.json`,
  `results/mech_tests/mech_test2/results_seeds.json`,
  `results/outputs_controlled/results_fig34_seeds.json` /
  `results_fig4_seeds34.jsonl` — the raw per-seed results, plus `meta.json`
  files (checkpoint-selection and run metadata, no weights) preserved under
  `results/outputs_seeds*/` and `results/outputs_convergence/`.
- `paper/seed_tables/` — the seed-aggregated tables as they appear in the
  paper (`seed_tables.md` through `seed_tables_g.md`, covering the v11 and
  v12 campaigns) together with the `make_seed_tables_*.py` scripts that build
  them. `paper/v11/` holds the frozen v11 copies of the tables it cites.

## Everything in this repository

This release is a complete record of the project, not only the runs that made
the paper: every experiment batch that was launched is here, including the
ones that were refuted, abandoned, or superseded. The table below maps each
batch to where its outputs live.

| Batch | What it was | Results |
|---|---|---|
| Pilot (round 1) | first per-class IQP fit on lensing images, 3 classes | `results/outputs_lens_pilot/`, `results/results_lens_{axion,cdm,no_sub}.jsonl` |
| Round 2 | per-class re-run with the corrected aggregation and a clipped-intensity variant | `results/outputs_lens_round2/`, `results/results_lens_round2_*.jsonl` |
| Round 3 | 28-arm sweep: fixed-basis IQP, RBM/DCGAN baselines, four zero-learning controls | `results/outputs_lens_r3/`, `results/results_r3.jsonl`, `results/r3_latent_select.json` |
| Round 3b | learned-latent-direction IQP, M ∈ {2,4,8} × 3 objectives × 2 priors, N ∈ {11,100} — the paper's main arm | `results/outputs_lens_r3b/`, `results/results_r3b.jsonl` |
| Paper suite — map | correlation–complexity map (Fig. 1a), single seed and five seeds | `results/paper_suite/map/results_map*.json` |
| Paper suite — controlled | the two controlled-synthetic falsification studies (support mismatch, QCLI–CCI ceiling; ref. Figs. 3–4) | `results/outputs_controlled/results_fig3*.jsonl`, `results_fig4*.jsonl`, `results_fig34_seeds.json` |
| Paper suite — scaling | N-scaling of IQP vs RBM vs DCGAN, N = 10…100 | `results/outputs_lens_scaling/`, `results/results_scaling.jsonl`, `results/results_seeds_scaling*.jsonl` |
| Convergence | baseline-fairness sweep: are RBM/DCGAN simply stopped too early? | `results/outputs_convergence/`, `results/results_convergence.jsonl`, `results/results_convergence_eval.jsonl` |
| Mechanism tests | what the IQP arm is actually doing (interpolation, affine latent, branch checks) | `results/mech_tests/` |
| Seeds (v11) | five-seed re-run of the main comparison under metric protocol v11, with an early-stopping variant, a floor/bootstrap pass and robustness probes | `results/outputs_seeds*/`, `results/results_seeds_*.jsonl`, `results/results_v11_*.jsonl`, `results/results_seeds_raw/` |
| θ₀ objective | ablation on the θ₀ pre-training objective | `results/outputs_seeds_theta0/`, `results/results_seeds_theta0.jsonl` |
| Class-conditional | per-class (axion / cdm / no_sub) arms under v11 metrics | `results/outputs_lens_class/`, `results/results_class.jsonl`, `results/results_seeds_class*.jsonl` |
| v12 arms | the v12 campaign: non-linear (MLP) latent map, sampled flux, scheduled variants, 5 seeds + smoke tests | `results/outputs_v12/`, `results/outputs_v12_smoke{,2}/`, `results/results_seeds_v12.jsonl`, `results/results_seeds_v12_es.jsonl` |
| Grid-MLP | latent-map capacity grid (M × hidden width) plus its zero-learning controls | `results/work/gsoc_iqp_gridmlp/`, `results/results_seeds_gridmlp.jsonl`, `results/results_seeds_gridmlp_es.jsonl` |
| MLP control | classical MLP decoder trained on the same targets, as a null model for the quantum core | `results/results_seeds_mlpctrl.jsonl`, `results/work/gsoc_iqp_gridmlp/ctrl/` |
| MLP suite | the large MLP sweep and its phase-2 ablations (pick-loss, R1/R2, θ₀-MMD, per-class) | `results/work/gsoc_iqp_mlpsuite/`, `results/results_seeds_mlp_suite.jsonl`, `results/results_seeds_mlp_suite_es.jsonl` |
| Flux residual | v12 flux-residual diagnostic (`docs/figures/diag_v12_flux_*.png`) | `results/results_v12_fluxres.jsonl` |
| Turbulence (Appendix B) | the independent replication attempt on Khojasteh et al. 2022 D2 — reported as a negative result | `results/outputs_d2*/`, `results/results_d2_*.jsonl`, `results/results_smoke_d2.jsonl` |
| Anchor fit | the anchor-fitting protocol sweeps that diagnosed the turbulence failure (5 seeds × several protocols) | `results/outputs_anchor_fit*/`, `results/outputs_anchor_fit_protocol*/` |
| Dataset characterisation | the measurements behind `docs/LENS_IMAGE_CHARACTERISTICS.md` | `results/analysis_lens/` |

Pre-registrations and write-ups for these batches are in `docs/`: every
`docs/*_PREREGISTRATION.md` was committed before its run and states numbered,
falsifiable predictions; the matching `docs/*_RESULTS.md` scores the run
against them. `docs/IQP_LENS_METHOD_DIFFERENCES_ZH*.md` (v10/v11/v12, with
PDFs) is the line-by-line account of how this replication departs from
arXiv:2603.06440, and `docs/IQP_LENS_PAPER_ZH*.md` is the long-form Chinese
version of the paper.

## Anonymization

All filesystem paths and account identifiers in this repository have been
replaced with placeholders, in every text file (`.md`, `.py`, `.sbatch`,
`.sh`, `.json`, `.jsonl`, `.tex`, `.bib`, `.txt`) and in the compiled PDFs:

| Placeholder | Was |
|---|---|
| `<HOME>` | the researcher's home directory on the cluster |
| `<WORK>` | the scratch/data root (caches such as `<WORK>/data/lens_cache` and `<WORK>/data/turbulence_khojasteh2022/d2_cache` live there) |
| `<user>` | the cluster username, and the author's name where it appeared in the research log |
| `<YOUR_ACCOUNT>` | the SLURM charge account in the `.sbatch` files |
| `<node>` | individual compute-node hostnames in run logs |

Nothing else was edited: SLURM job IDs, timestamps, hyperparameters and all
numeric results are the originals. Before running anything you will need to
substitute your own values — either in place, or by overriding the
`CACHE_DIR_DEFAULT` constants in `code/data_lens.py`, `code/data_lens_r3.py`
and `code/data_d2.py` and the header of each `.sbatch` file.

`docs/` remains a running lab notebook rather than polished prose: it keeps
the first-person reasoning exactly as written at the time, and it includes
the honest negative results (e.g. `docs/IQP_TURBULENCE_REPLICATION_RESULTS.md`)
alongside the positive ones.
