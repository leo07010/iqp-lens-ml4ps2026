# Experiments in progress (2026-09-09 evening)

Two exploratory runs were launched after v11 was frozen. Neither changes `paper/v11/`.
Their outputs are synced into this repo when they finish; if you are reading this before
that happened, the raw results live on the cluster under
`experiments/13_iqp_turbulence_replication/` (paths below are relative to that directory).

| Experiment | SLURM job | What it tests | Where results land |
|---|---|---|---|
| 12-qubit finer grid (aligned 64x64 image as a 4,096-cell pmf, all generators) and a nonlinear latent map (theta = theta0 + g(z), g = 4 -> 32 tanh -> 1024) | 362840 (dev, 8 GPU, 70 runs) | Does removing the 32x32 crop, or letting the latent path curve in phase space, improve generation? Same protocol as Table 2: seeds 0-4, N = 11 and 100, v11 metrics, validation-selected checkpoints | `results_seeds_gridmlp.jsonl`, `paper/ml4ps2026/seed_tables_d.md`, `paper/ml4ps2026/figures/diag_gridmlp_*.png` -> copied to `results/` and `paper/v11/` here |
| IQP small-data improvements: multi-kernel objective (flat MMD + paper kernel), Walsh low-pass regulariser, flux sampled from the training distribution, M in {2, 4} | submitted after this note | Can an IQP-side change win PDF-JS, raw MMD2 and flux-aligned MMD2 at N = 11 by more than the baselines' sd? | `results_seeds_v12.jsonl`, `paper/ml4ps2026/seed_tables_e.md`, `figures/diag_v12_*.png` -> same places |

Reading the results: every table lists mean +- sd over five seeds under the v11 metric protocol
(`code/metrics_v11.py`), next to the Table 2 baselines from `paper/v11/seed_tables.md`.
A win only counts if the gap exceeds both arms' sd.

Open decisions left for the author:
1. Whether the primary small-data protocol should forbid the 200-image validation set for
   baseline checkpoint selection (only the N training images may be used); the current v11
   table gives the baselines that validation set.
2. Deleting the 180 unselected IQP early-stopping snapshot directories (about 1 GB) on the cluster.
3. Anonymised code link (anonymous.4open.science) for the OpenReview submission, due 2026-09-12 23:59 AoE.
