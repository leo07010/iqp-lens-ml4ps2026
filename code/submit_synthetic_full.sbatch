#!/bin/bash
#SBATCH -J iqp_turb_synth_full
#SBATCH -A <YOUR_ACCOUNT>
#SBATCH -p dev
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 03:50:00
#SBATCH -o <REPO_ROOT>/code/logs_synth/sbatch-%j.out
#SBATCH -e <REPO_ROOT>/code/logs_synth/sbatch-%j.err
#
# Full-SCALE (real step counts, NOT the smoke budget of run_smoke_all.sh), full-11-anchor
# arms on SYNTHETIC data, 1 seed each, all on a single dev-partition GPU, serial.
# Purpose: measure wall-clock and exercise the whole pipeline (train -> synthesize ->
# ground truth -> evaluate -> Table 2) end-to-end before ever touching real D2 data or
# spending 8gpus-partition budget. NOT a scientific run (synthetic data, 1 seed): do not
# read the printed Table 2 numbers as results.
#
# Estimated wall-clock (see task report for the derivation):
#   IQP-31k  (CPU, 4 threads; test_iqp_exact.py measured 0.0246 s/step at n=18,K=6)  ~27 min
#   IQP-63k  (CPU, 4 threads; same per-step cost, K only changes param count)        ~30 min
#   RBM-12k  (GPU; batched (N,18)x(18,630) matmuls + 200 tau x 200-sweep x 1e4-chain
#             block-Gibbs generation)                                                ~20 min
#   DCGAN x6 (3 tiers x {11,100} snapshots, GPU, 20k G-steps each)                    ~48 min
#   CTRL-marginal, CTRL-interp (numpy only)                                          ~2 min
#   make_ground_truth + evaluate                                                     ~1 min
#   TOTAL                                                                         ~2h8m, dev's
#   4h cap leaves ~1h50m headroom for GPU/thread variance from these CPU-measured estimates.
# IQP runs on CPU (device=cpu) because its per-step cost is EMPIRICALLY measured (not
# estimated) at 4 threads via test_iqp_exact.py::test_6_timing; the FWHT butterfly is 18
# sequential tiny kernel launches per transform and GPU benefit for that shape is
# unverified. RBM/DCGAN run on the allocated GPU (device=cuda) where large batched
# matmuls / conv layers reliably benefit. If GPU underperforms for RBM/DCGAN too, they
# already have the same CPU thread cap as IQP and can be resubmitted with --device cpu.
set -uo pipefail
module purge; module load cuda/13.0

REPO=<REPO_ROOT>/code
cd "$REPO"
mkdir -p logs_synth
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4

PY=/usr/bin/python3
OUT="$REPO/outputs_synth_full"
ts() { date +"%Y-%m-%d %H:%M:%S"; }

echo "[SLURM] job=$SLURM_JOB_ID node=$(hostname) part=$SLURM_JOB_PARTITION gpus=$($PY -c 'import torch;print(torch.cuda.device_count())')"

# ---------------------------------------------------------------- fail-fast pre-checks
echo "[$(ts)] pre-check: py_compile all .py"
for f in *.py; do
    "$PY" -m py_compile "$f" || { echo "[FATAL] py_compile failed for $f"; exit 1; }
done

echo "[$(ts)] pre-check: unit tests"
for t in test_iqp_exact.py test_baselines.py test_controls.py test_diagnostics.py test_data_d2.py; do
    "$PY" "$t" > "logs_synth/precheck_${t%.py}.log" 2>&1 \
        || { echo "[FATAL] $t failed -- see logs_synth/precheck_${t%.py}.log"; tail -20 "logs_synth/precheck_${t%.py}.log"; exit 1; }
done
echo "[$(ts)] pre-checks passed"

rm -rf "$OUT"
mkdir -p "$OUT"

# ---------------------------------------------------------------- ground truth (200 held-out tau)
echo "[$(ts)] START make_ground_truth"
"$PY" make_ground_truth.py --data synthetic --out "$OUT/ground_truth" --n-holdout 200 \
    > logs_synth/ground_truth.log 2>&1 || { echo "[FATAL] make_ground_truth failed"; tail -30 logs_synth/ground_truth.log; exit 1; }
echo "[$(ts)] OK    make_ground_truth"

# ---------------------------------------------------------------- IQP (K=6, K=7), CPU
for K in 6 7; do
    tag="iqp_K${K}"
    echo "[$(ts)] START $tag (device=cpu, seed=0)"
    "$PY" train_iqp.py --data synthetic --out "$OUT/$tag" --device cpu --seed 0 \
        --K "$K" --steps-core 30000 --steps-lat 3000 --n-holdout 200 \
        > "logs_synth/$tag.log" 2>&1 || { echo "[FATAL] $tag failed"; tail -30 "logs_synth/$tag.log"; exit 1; }
    echo "[$(ts)] OK    $tag"
done

# ---------------------------------------------------------------- RBM, GPU
echo "[$(ts)] START rbm (device=cuda, seed=0)"
CUDA_VISIBLE_DEVICES=0 "$PY" baseline_rbm.py --data synthetic --out "$OUT/rbm" --device cuda --seed 0 \
    --n-holdout 200 \
    > logs_synth/rbm.log 2>&1 || { echo "[FATAL] rbm failed"; tail -30 logs_synth/rbm.log; exit 1; }
echo "[$(ts)] OK    rbm"

# ---------------------------------------------------------------- DCGAN, 3 tiers x {11, 100}, GPU
for tier in 51k 0.2m 1.2m; do
    for n_train in 11 100; do
        tag="dcgan_${tier}_n${n_train}"
        echo "[$(ts)] START $tag (device=cuda, seed=0)"
        CUDA_VISIBLE_DEVICES=0 "$PY" baseline_dcgan.py --data synthetic --out "$OUT/$tag" --device cuda --seed 0 \
            --tier "$tier" --n-train "$n_train" --steps 20000 --n-gen 200 \
            > "logs_synth/$tag.log" 2>&1 || { echo "[FATAL] $tag failed"; tail -30 "logs_synth/$tag.log"; exit 1; }
        echo "[$(ts)] OK    $tag"
    done
done

# ---------------------------------------------------------------- controls (numpy only)
echo "[$(ts)] START ctrl-marginal (seed=0)"
"$PY" controls.py --arm marginal --data synthetic --out "$OUT/marginal" --seed 0 --n-holdout 200 \
    > logs_synth/marginal.log 2>&1 || { echo "[FATAL] ctrl-marginal failed"; tail -30 logs_synth/marginal.log; exit 1; }
echo "[$(ts)] OK    ctrl-marginal"

echo "[$(ts)] START ctrl-interp (seed=0)"
"$PY" controls.py --arm interp --data synthetic --out "$OUT/interp" --seed 0 --n-holdout 200 \
    > logs_synth/interp.log 2>&1 || { echo "[FATAL] ctrl-interp failed"; tail -30 logs_synth/interp.log; exit 1; }
echo "[$(ts)] OK    ctrl-interp"

# ---------------------------------------------------------------- evaluate
echo "[$(ts)] START evaluate"
"$PY" evaluate.py --outputs-dir "$OUT" --ground-truth "$OUT/ground_truth" \
    --results "$REPO/results_synth_full.jsonl" \
    > logs_synth/evaluate.log 2>&1 || { echo "[FATAL] evaluate failed"; tail -30 logs_synth/evaluate.log; exit 1; }
cat logs_synth/evaluate.log
echo "[$(ts)] DONE submit_synthetic_full.sbatch"
