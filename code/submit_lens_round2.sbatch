#!/bin/bash
#SBATCH -J iqp_lens_round2
#SBATCH -A <YOUR_ACCOUNT>
#SBATCH -p 8gpus
#SBATCH --gres=gpu:8
#SBATCH -c 32
#SBATCH --mem=128G
#SBATCH -t 05:00:00
#SBATCH -o <REPO_ROOT>/code/logs_lens/sbatch-round2-%j.out
#SBATCH -e <REPO_ROOT>/code/logs_lens/sbatch-round2-%j.err
#
# Round 2 of the IQP generative pilot on DeepLense Model II.
# Pre-registration: docs/IQP_LENS_PILOT_PREREGISTRATION.md, "Addendum 2026-09-06 -- round 2",
# written BEFORE this ran (predictions P6-P9 and the decision rule are in it).
#
# Round 1 (job 345400) verdict: IQP was the worst trained model. Three image-specific
# mechanisms were named; round 2 tests the three fixes, single seed (0) throughout.
#
# ---------------------------------------------------------------------------- job list
# 5 arms x 3 classes (axion, cdm, no_sub) x N in {11, 100} = 30 jobs.
#
#   arm label              | script         | what is different from round 1
#   -----------------------+----------------+--------------------------------------------
#   IQP-tuple-sqrt-fixed   | train_iqp.py   | sqrt intensity + ALL parameters trained in
#                          |                | stage 1 + exact 4,096-pixel data pmf
#                          |                | (--lens-sqrt --stage1-train-all --full-grid)
#   IQP-pmf-full-fixed     | train_iqp.py   | 12-qubit image-as-pmf encoding, all 4,096
#                          |                | Z-strings (--encoding pmf --K 12), fixed
#                          |                | protocol, exact data pmf
#   IQP-pmf-K6-fixed       | train_iqp.py   | same, |s| <= 6 (2,510 generators)
#   IQP-pmf-full-paper     | train_iqp.py   | same encoding, PAPER protocol: theta_lat
#                          |                | frozen in stage 1, data pmf from 5e4 sampled
#                          |                | codes (no --stage1-train-all, no --full-grid)
#   RBM-sqrt               | baseline_rbm.py| round-1 RBM on sqrt intensity
#
# Round-1 arms (IQP-31k raw, RBM-12k raw, DCGAN x3, CTRL-copy, CTRL-mean, CTRL-marginal,
# HOLDOUT ref) are REUSED from outputs_lens_pilot/, not rerun; the evaluation step at the
# bottom merges the two output trees into one table per class.
#
# Extra output directories (not extra jobs):
#   tuple arms  -> <out>_last/      LAST-WRITE-WINS aggregation of the same tuples
#   pmf arms    -> <out>_sampled/   counting density of 1e5 circuit samples, next to the
#                                   PRIMARY exact reconstruction q(kx, ky) x S
#
# ------------------------------------------------------------------------- time budget
# Measured, not guessed:
#   n = 18, K = 6 (tuple arms) on this node type, round 1 meta.json wallclock:
#       N = 11  (6.0e4 steps) = 10.6-11.2 min      -> ~11.0 ms/step
#       N = 100 (3.27e5 steps) = 56.0-59.5 min     -> ~10.5 ms/step
#   RBM, round 1 meta.json wallclock:  N = 11 = 0.9-1.0 min, N = 100 = 4.2 min
#   n = 12 (pmf arms), CPU 4-thread measurement on the login node, 2026-09-06:
#       K = 12 (4,096 params) = 4.11 ms/step ; K = 6 (2,510 params) = 4.18 ms/step
#       (same script's n = 18, K = 6 CPU figure is 29.0 ms/step, so n = 12 is 7.1x cheaper
#        on CPU).  On GPU the FWHT butterfly is LAUNCH-bound -- 12 passes instead of 18 --
#        so the honest GPU extrapolation is 10.5 x 12/18 ~ 7.0 ms/step, with 10.5 ms/step
#        (i.e. no speed-up at all vs n = 18) as the conservative bound used below.
#   => per pmf job:  N = 11  ~  7 min (bound 11) ;  N = 100  ~ 38 min (bound 57)
#
# Jobs run 8-wide with `wait` between batches (a batch costs its slowest member), ordered
# LONGEST FIRST:
#   batch 1  3x tuple N=100 (57) + 3x pmf-full-fixed N=100 + 2x pmf-K6 N=100   ~57 min
#   batch 2  1x pmf-K6 N=100 + 3x pmf-paper N=100 + 3x tuple N=11 + 1x RBM N=100 ~57 min
#   batch 3  2x RBM N=100 + 3x RBM N=11 + 3x pmf-full-fixed N=11               ~11 min
#   batch 4  3x pmf-K6 N=11 + 3x pmf-paper N=11                                ~11 min
#   + pre-checks + evaluate_lens x3
#   TOTAL ~ 2.5 h at the conservative bound.  Requesting 5 h.
#
# Pre-check cost, MEASURED on the login node 2026-09-06 (not asserted): py_compile of all
# .py = 2 s; test_lens.py = 39 s; test_lens_round2.py = 40 s; test_d2_arms_smoke.py = 18 s
# -> ~1.7 min total. (Those suites do launch real train_iqp/baseline_rbm subprocesses, but
# at 3 anchors / 20 steps / CPU, nowhere near their 900 s per-subprocess timeout.)
# evaluate_lens on the 24 real round-1 run dirs measured 5 s per class in round 1.
#
# IQP / RBM run on GPU (device=cuda) via CUDA_VISIBLE_DEVICES round-robin over the 8
# allocated GPUs, exactly as in round 1, so the round-1 rows in the merged table were
# produced on the same hardware path.
set -uo pipefail

REPO=<REPO_ROOT>/code
cd "$REPO"
mkdir -p logs_lens

module purge; module load cuda/13.0
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
PY=/usr/bin/python3
OUT="$REPO/outputs_lens_round2"
PILOT="$REPO/outputs_lens_pilot"
CLASSES="axion cdm no_sub"
ts() { date +"%Y-%m-%d %H:%M:%S"; }

echo "[SLURM] job=$SLURM_JOB_ID node=$(hostname) part=$SLURM_JOB_PARTITION gpus=$($PY -c 'import torch;print(torch.cuda.device_count())')"

# ---------------------------------------------------------------- fail-fast: dataset + cache
"$PY" -c "import data_lens; m = data_lens.build_cache(); print('[lens] cache ok, classes =', m['class_names'])" \
    || { echo "[FATAL] data_lens cache could not be built (dataset missing?). Aborting before using any 8gpus budget."; exit 1; }

# ---------------------------------------------------------------- fail-fast: round-1 outputs
# The deliverable is ONE table per class holding round-1 AND round-2 arms; without the
# round-1 tree there is nothing to compare against and the run is pointless.
for c in $CLASSES; do
    [ -d "$PILOT/$c" ] || { echo "[FATAL] missing round-1 outputs $PILOT/$c"; exit 1; }
    n1=$(find "$PILOT/$c" -mindepth 1 -maxdepth 1 -type d | wc -l)
    [ "$n1" -ge 10 ] || { echo "[FATAL] $PILOT/$c holds only $n1 run dirs (expected >= 10)"; exit 1; }
    echo "[round1] $c: $n1 run directories to merge into the table"
done

# ---------------------------------------------------------------- fail-fast pre-checks
echo "[$(ts)] pre-check: py_compile all .py"
for f in *.py; do
    "$PY" -m py_compile "$f" || { echo "[FATAL] py_compile failed for $f"; exit 1; }
done

echo "[$(ts)] pre-check: unit tests"
for t in test_lens.py test_lens_round2.py test_d2_arms_smoke.py; do
    "$PY" "$t" > "logs_lens/precheck_r2_${t%.py}.log" 2>&1 \
        || { echo "[FATAL] $t failed -- see logs_lens/precheck_r2_${t%.py}.log"; tail -25 "logs_lens/precheck_r2_${t%.py}.log"; exit 1; }
done
echo "[$(ts)] pre-checks passed"

rm -rf "$OUT"
mkdir -p "$OUT"
for c in $CLASSES; do mkdir -p "$OUT/$c"; done

# ---------------------------------------------------------------- build the job list: TAG|DEVICE|CMD
JOBS=()
add_job() { JOBS+=("$1|$2|$3"); }

IQP_COMMON="--data lens --n-holdout 200 --synth-mode gaussian-latent --n-gen 200 \
 --steps-core 30000 --steps-lat 3000 --n-synth 100000 --device cuda --seed 0"
RBM_COMMON="--data lens --n-holdout 200 --synth-mode gaussian-latent --n-gen 200 \
 --device cuda --seed 0"

# --- longest first: the 18-qubit tuple arm at N = 100 (~57 min each) ------------------
for c in $CLASSES; do
    add_job "${c}_iqp_tuple_sqrt_fixed_n100" cuda \
        "$PY train_iqp.py $IQP_COMMON --lens-class $c --n-train 100 \
         --encoding tuple --lens-sqrt --stage1-train-all --full-grid --K 6 \
         --arm-name IQP-tuple-sqrt-fixed --out $OUT/$c/iqp_tuple_sqrt_fixed_n100"
done
# --- 12-qubit pmf arms at N = 100 (~38 min each, bound 57) ----------------------------
for c in $CLASSES; do
    add_job "${c}_iqp_pmf_full_fixed_n100" cuda \
        "$PY train_iqp.py $IQP_COMMON --lens-class $c --n-train 100 \
         --encoding pmf --K 12 --stage1-train-all --full-grid \
         --arm-name IQP-pmf-full-fixed --out $OUT/$c/iqp_pmf_full_fixed_n100"
done
for c in $CLASSES; do
    add_job "${c}_iqp_pmf_k6_fixed_n100" cuda \
        "$PY train_iqp.py $IQP_COMMON --lens-class $c --n-train 100 \
         --encoding pmf --K 6 --stage1-train-all --full-grid \
         --arm-name IQP-pmf-K6-fixed --out $OUT/$c/iqp_pmf_k6_fixed_n100"
done
# paper protocol: theta_lat frozen in stage 1, data pmf from --n-tuples 5e4 sampled codes
for c in $CLASSES; do
    add_job "${c}_iqp_pmf_full_paper_n100" cuda \
        "$PY train_iqp.py $IQP_COMMON --lens-class $c --n-train 100 \
         --encoding pmf --K 12 --n-tuples 50000 \
         --arm-name IQP-pmf-full-paper --out $OUT/$c/iqp_pmf_full_paper_n100"
done
# --- tuple arm at N = 11 (~11 min each) -----------------------------------------------
for c in $CLASSES; do
    add_job "${c}_iqp_tuple_sqrt_fixed_n11" cuda \
        "$PY train_iqp.py $IQP_COMMON --lens-class $c --n-train 11 \
         --encoding tuple --lens-sqrt --stage1-train-all --full-grid --K 6 \
         --arm-name IQP-tuple-sqrt-fixed --out $OUT/$c/iqp_tuple_sqrt_fixed_n11"
done
# --- RBM-sqrt (fast: 4.2 min at N = 100, 1 min at N = 11) -----------------------------
for c in $CLASSES; do
    add_job "${c}_rbm_sqrt_n100" cuda \
        "$PY baseline_rbm.py $RBM_COMMON --lens-class $c --n-train 100 --lens-sqrt \
         --arm-name RBM-sqrt --out $OUT/$c/rbm_sqrt_n100"
done
for c in $CLASSES; do
    add_job "${c}_rbm_sqrt_n11" cuda \
        "$PY baseline_rbm.py $RBM_COMMON --lens-class $c --n-train 11 --lens-sqrt \
         --arm-name RBM-sqrt --out $OUT/$c/rbm_sqrt_n11"
done
# --- 12-qubit pmf arms at N = 11 (~7 min each, bound 11) ------------------------------
for c in $CLASSES; do
    add_job "${c}_iqp_pmf_full_fixed_n11" cuda \
        "$PY train_iqp.py $IQP_COMMON --lens-class $c --n-train 11 \
         --encoding pmf --K 12 --stage1-train-all --full-grid \
         --arm-name IQP-pmf-full-fixed --out $OUT/$c/iqp_pmf_full_fixed_n11"
done
for c in $CLASSES; do
    add_job "${c}_iqp_pmf_k6_fixed_n11" cuda \
        "$PY train_iqp.py $IQP_COMMON --lens-class $c --n-train 11 \
         --encoding pmf --K 6 --stage1-train-all --full-grid \
         --arm-name IQP-pmf-K6-fixed --out $OUT/$c/iqp_pmf_k6_fixed_n11"
done
for c in $CLASSES; do
    add_job "${c}_iqp_pmf_full_paper_n11" cuda \
        "$PY train_iqp.py $IQP_COMMON --lens-class $c --n-train 11 \
         --encoding pmf --K 12 --n-tuples 50000 \
         --arm-name IQP-pmf-full-paper --out $OUT/$c/iqp_pmf_full_paper_n11"
done

echo "[$(ts)] ${#JOBS[@]} jobs queued (expected 30 = 5 arms x 3 classes x 2 N), running 8-wide"
if [ "${#JOBS[@]}" -ne 30 ]; then
    echo "[FATAL] job list is ${#JOBS[@]}, expected 30 -- the arm matrix changed, refusing to run"
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

# ---------------------------------------------------------------- evaluate: ONE merged table per class
# evaluate_lens.py takes several --outputs-dir and merges them, so round 1 is scored in the
# same table (same 200 held-out images, same metrics, same encoder seed) without rerunning
# it -- every round-1 row is RE-SCORED here from its generated_*.npy, nothing is carried
# over from the round-1 jsonl. Every arm carries its own `arm` label in meta.json, so the
# rows do not collide (checked against the round-1 labels on disk: disjoint).
#
# Each row also records gen_min / gen_max / gen_mean_total against holdout_min / holdout_max
# / holdout_mean_total. Read them before quoting PDF-JS or MMD2: metrics.pdf_js bins on the
# JOINT range of (held-out, that arm's output) and metrics.conv_mmd2 normalises the same
# way, so an arm that leaves the data's range is scored on a stretched scale. The run also
# prints a NOTE line per out-of-range arm. Every pmf arm's images total exactly S by
# construction, so a residual NN-holdout gap can come from that rather than the spatial fit.
for c in $CLASSES; do
    echo "[$(ts)] START evaluate_lens round2 ($c): $PILOT/$c + $OUT/$c"
    "$PY" evaluate_lens.py --lens-class "$c" \
        --outputs-dir "$PILOT/$c" "$OUT/$c" \
        --n-holdout 200 --n-train-max 100 \
        --results "$REPO/results_lens_round2_${c}.jsonl" \
        > "logs_lens/evaluate_round2_${c}.log" 2>&1 \
        || { echo "[WARN] evaluate_lens failed for $c"; tail -30 "logs_lens/evaluate_round2_${c}.log"; fail=1; }
    cat "logs_lens/evaluate_round2_${c}.log"
done

echo "[$(ts)] DONE submit_lens_round2.sbatch (fail=$fail)"
exit "$fail"
