#!/bin/bash
#SBATCH -J iqp_lens_pilot
#SBATCH -A <YOUR_ACCOUNT>
#SBATCH -p 8gpus
#SBATCH --gres=gpu:8
#SBATCH -c 32
#SBATCH --mem=128G
#SBATCH -t 06:00:00
#SBATCH -o <REPO_ROOT>/code/logs_lens/sbatch-%j.out
#SBATCH -e <REPO_ROOT>/code/logs_lens/sbatch-%j.err
#
# IQP generative pilot on DeepLense Model II (dark-matter substructure images).
# Pre-registration: docs/IQP_LENS_PILOT_PREREGISTRATION.md (written 2026-09-06, before
# any run). Single seed (seed 0) throughout, at Leo's request.
#
# ---------------------------------------------------------------------------- job list
# Per class (axion, cdm, no_sub) x per regime N in {11, 100}:
#   IQP-31k, RBM-12k, DCGAN-51k, DCGAN-0.2M, DCGAN-1.2M, CTRL-marginal, CTRL-copy,
#   CTRL-mean                                                       = 8 arms x 2 N = 16
# plus the secondary IQP-31k-sqrt arm (N = 11 only)                  = 1
#   -> 17 jobs per class x 3 classes = 51 jobs.
# The two tuple-based generative arms (IQP, RBM) and CTRL-marginal each ALSO write a
# `<out>_last/` sibling holding the same samples under the LAST-WRITE-WINS aggregation
# (agg_variants.py); those are extra output directories, not extra jobs.
#
# ------------------------------------------------------------------------- time budget
# IQP: 11 ms/step measured on H100 (submit_d2_main.sbatch, 2026-09-05). Algorithm 1 runs
#      --steps-core 30000 + (N - 1) x --steps-lat 3000 steps:
#        N = 11  ->  6.0e4 steps ~ 11 min      N = 100 -> 3.27e5 steps ~ 60 min
# RBM: d2 measured ~20 min for the same 6.0e4-step schedule -> <= ~110 min at N = 100
#      (that d2 figure is the conservative bound; PCD steps on 18x630 tensors are
#      launch-bound, so the real GPU time is expected well below it).
# DCGAN: 20k generator steps ~ 8 min per job (d2 measurement).
# CTRL-marginal ~ 2-4 min (200 x 1e5 product-Bernoulli draws + aggregation); CTRL-copy /
# CTRL-mean are seconds.
# Jobs are ordered LONGEST FIRST and run 8-wide with `wait` between batches, so a batch
# costs its slowest member:
#   batch 1 (RBM x3 @N100, IQP x3 @N100, RBM x2 @N11)          ~110 min
#   batch 2 (RBM @N11, IQP x3 @N11, IQP-sqrt x3, DCGAN)         ~20 min
#   batches 3-5 (DCGAN 17 jobs + CTRL-marginal)                 ~24 min
#   batches 6-7 (CTRL-marginal/copy/mean)                        ~2 min
#   + pre-checks (~5 min), evaluate_lens x3 (~3 min), qcli_cci_lens (~6 min)
#   TOTAL ~ 2.8 h.  Requesting 6 h leaves headroom for the RBM bound being real.
#
# IQP/RBM/DCGAN run on GPU (device=cuda) via CUDA_VISIBLE_DEVICES round-robin across the
# 8 allocated GPUs; the controls are CPU-only.
#
# NOTE on N_t: every tuple-based arm keeps the paper's N_t = 5e4 training tuples drawn
# (with replacement) from the image's 4,096-pixel code pool -- deliberately NOT
# --full-grid. The pool IS the full grid here, so --full-grid would give the exact
# empirical pmf; keeping 5e4 draws keeps the data-side sampling noise identical to what
# every d2 arm saw, and identical across IQP / RBM / CTRL-marginal.
set -uo pipefail

REPO=<REPO_ROOT>/code
cd "$REPO"
mkdir -p logs_lens

module purge; module load cuda/13.0
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
PY=/usr/bin/python3
OUT="$REPO/outputs_lens_pilot"
CLASSES="axion cdm no_sub"
ts() { date +"%Y-%m-%d %H:%M:%S"; }

echo "[SLURM] job=$SLURM_JOB_ID node=$(hostname) part=$SLURM_JOB_PARTITION gpus=$($PY -c 'import torch;print(torch.cuda.device_count())')"

# ---------------------------------------------------------------- fail-fast: dataset + cache
"$PY" -c "import data_lens; m = data_lens.build_cache(); print('[lens] cache ok, classes =', m['class_names'])" \
    || { echo "[FATAL] data_lens cache could not be built (dataset missing?). Aborting before using any 8gpus budget."; exit 1; }

# ---------------------------------------------------------------- fail-fast pre-checks
echo "[$(ts)] pre-check: py_compile all .py"
for f in *.py; do
    "$PY" -m py_compile "$f" || { echo "[FATAL] py_compile failed for $f"; exit 1; }
done

echo "[$(ts)] pre-check: unit tests"
for t in test_iqp_exact.py test_baselines.py test_controls.py test_diagnostics.py \
         test_data_d2.py test_d2_arms_smoke.py test_lens.py; do
    "$PY" "$t" > "logs_lens/precheck_${t%.py}.log" 2>&1 \
        || { echo "[FATAL] $t failed -- see logs_lens/precheck_${t%.py}.log"; tail -20 "logs_lens/precheck_${t%.py}.log"; exit 1; }
done
echo "[$(ts)] pre-checks passed"

rm -rf "$OUT"
mkdir -p "$OUT"
for c in $CLASSES; do mkdir -p "$OUT/$c"; done

# ---------------------------------------------------------------- build the job list: TAG|DEVICE|CMD
JOBS=()
add_job() { JOBS+=("$1|$2|$3"); }

# --- longest first: RBM @ N=100, then IQP @ N=100 -------------------------------------
for c in $CLASSES; do
    add_job "${c}_rbm_n100" cuda \
        "$PY baseline_rbm.py --data lens --lens-class $c --n-train 100 --n-holdout 200 \
         --synth-mode gaussian-latent --n-gen 200 --device cuda --seed 0 --out $OUT/$c/rbm_n100"
done
for c in $CLASSES; do
    add_job "${c}_iqp31k_n100" cuda \
        "$PY train_iqp.py --data lens --lens-class $c --n-train 100 --n-holdout 200 \
         --synth-mode gaussian-latent --n-gen 200 --K 6 --steps-core 30000 --steps-lat 3000 \
         --n-synth 100000 --device cuda --seed 0 --out $OUT/$c/iqp31k_n100"
done

# --- N = 11 tuple-based arms ----------------------------------------------------------
for c in $CLASSES; do
    add_job "${c}_rbm_n11" cuda \
        "$PY baseline_rbm.py --data lens --lens-class $c --n-train 11 --n-holdout 200 \
         --synth-mode gaussian-latent --n-gen 200 --device cuda --seed 0 --out $OUT/$c/rbm_n11"
done
for c in $CLASSES; do
    add_job "${c}_iqp31k_n11" cuda \
        "$PY train_iqp.py --data lens --lens-class $c --n-train 11 --n-holdout 200 \
         --synth-mode gaussian-latent --n-gen 200 --K 6 --steps-core 30000 --steps-lat 3000 \
         --n-synth 100000 --device cuda --seed 0 --out $OUT/$c/iqp31k_n11"
done
# secondary arm (pre-registration P5): sqrt-intensity quantization, N = 11 only
for c in $CLASSES; do
    add_job "${c}_iqp31k_sqrt_n11" cuda \
        "$PY train_iqp.py --data lens --lens-class $c --n-train 11 --n-holdout 200 --lens-sqrt \
         --synth-mode gaussian-latent --n-gen 200 --K 6 --steps-core 30000 --steps-lat 3000 \
         --n-synth 100000 --device cuda --seed 0 --out $OUT/$c/iqp31k_sqrt_n11"
done

# --- DCGAN: 3 tiers x 2 regimes x 3 classes ------------------------------------------
for c in $CLASSES; do
    for tier in 51k 0.2m 1.2m; do
        for n in 11 100; do
            add_job "${c}_dcgan_${tier}_n${n}" cuda \
                "$PY baseline_dcgan.py --data lens --lens-class $c --n-train $n --tier $tier \
                 --steps 20000 --n-gen 200 --device cuda --seed 0 --out $OUT/$c/dcgan_${tier}_n${n}"
        done
    done
done

# --- controls -------------------------------------------------------------------------
for c in $CLASSES; do
    for n in 11 100; do
        add_job "${c}_ctrl_marginal_n${n}" cpu \
            "$PY controls.py --arm marginal --data lens --lens-class $c --n-train $n \
             --synth-mode gaussian-latent --n-gen 200 --n-synth 100000 --seed 0 \
             --out $OUT/$c/ctrl_marginal_n${n}"
    done
done
for c in $CLASSES; do
    for n in 11 100; do
        add_job "${c}_ctrl_copy_n${n}" cpu \
            "$PY controls_lens.py --arm copy --lens-class $c --n-train $n --n-gen 200 \
             --seed 0 --out $OUT/$c/ctrl_copy_n${n}"
        add_job "${c}_ctrl_mean_n${n}" cpu \
            "$PY controls_lens.py --arm mean --lens-class $c --n-train $n --n-gen 200 \
             --seed 0 --out $OUT/$c/ctrl_mean_n${n}"
    done
done

echo "[$(ts)] ${#JOBS[@]} jobs queued (expected 51), running 8-wide across the allocated GPUs"
if [ "${#JOBS[@]}" -ne 51 ]; then
    echo "[FATAL] job list is ${#JOBS[@]}, expected 51 -- the arm matrix changed, refusing to run"
    exit 1
fi

# ---------------------------------------------------------------- run 8-wide, wait between batches
GPU_SLOTS=8
i=0
pids=()
tags=()
fail=0
for job in "${JOBS[@]}"; do
    tag="${job%%|*}"
    rest="${job#*|}"
    device="${rest%%|*}"
    cmd="${rest#*|}"
    slot=$((i % GPU_SLOTS))
    echo "[$(ts)] START $tag (slot $slot, device=$device)"
    if [ "$device" = "cuda" ]; then
        (CUDA_VISIBLE_DEVICES=$slot bash -c "$cmd" > "logs_lens/${tag}.log" 2>&1) &
    else
        (bash -c "$cmd" > "logs_lens/${tag}.log" 2>&1) &
    fi
    pids+=($!)
    tags[$i]="$tag"
    i=$((i + 1))
    if [ $((i % GPU_SLOTS)) -eq 0 ]; then
        for j in "${!pids[@]}"; do
            wait "${pids[$j]}" || { echo "[WARN] job ${tags[$((i - GPU_SLOTS + j))]} failed, see logs_lens/${tags[$((i - GPU_SLOTS + j))]}.log"; fail=1; }
        done
        pids=()
    fi
done
for j in "${!pids[@]}"; do
    idx=$((i - ${#pids[@]} + j))
    wait "${pids[$j]}" || { echo "[WARN] job ${tags[$idx]} failed, see logs_lens/${tags[$idx]}.log"; fail=1; }
done

echo "[$(ts)] all ${#JOBS[@]} jobs finished (fail=$fail)"

# ---------------------------------------------------------------- evaluate: one table per class
for c in $CLASSES; do
    echo "[$(ts)] START evaluate_lens ($c)"
    "$PY" evaluate_lens.py --lens-class "$c" --outputs-dir "$OUT/$c" --n-holdout 200 \
        --n-train-max 100 --results "$REPO/results_lens_${c}.jsonl" \
        > "logs_lens/evaluate_${c}.log" 2>&1 \
        || { echo "[WARN] evaluate_lens failed for $c"; tail -30 "logs_lens/evaluate_${c}.log"; fail=1; }
    cat "logs_lens/evaluate_${c}.log"
done

# ---------------------------------------------------------------- QCLI / CCI (pre-registration P4)
echo "[$(ts)] START qcli_cci_lens (all classes)"
"$PY" qcli_cci_lens.py --n-images 200 --n-train 100 --out "$OUT/qcli_cci_lens.json" \
    > logs_lens/qcli_cci_lens.log 2>&1 \
    || { echo "[WARN] qcli_cci_lens failed"; tail -30 logs_lens/qcli_cci_lens.log; fail=1; }
tail -20 logs_lens/qcli_cci_lens.log

echo "[$(ts)] DONE submit_lens_pilot.sbatch (fail=$fail)"
exit "$fail"
