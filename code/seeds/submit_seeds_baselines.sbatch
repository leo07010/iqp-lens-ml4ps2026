#!/bin/bash
#SBATCH -J iqp_seeds_base
#SBATCH -A <YOUR_ACCOUNT>
#SBATCH -p dev
#SBATCH --gres=gpu:8
#SBATCH -c 32
#SBATCH --mem=128G
#SBATCH -t 03:00:00
#SBATCH -o <HOME>/gsoc2026-deeplense-qml/experiments/13_iqp_turbulence_replication/logs_lens/sbatch-seeds-base-%j.out
#SBATCH -e <HOME>/gsoc2026-deeplense-qml/experiments/13_iqp_turbulence_replication/logs_lens/sbatch-seeds-base-%j.err
#
# Seed replicates 1-4 of the two classical baselines, for Table 2 and Figure 5.
# Companion to seeds/submit_seeds_iqp.sbatch, which covers every IQP arm and Table 3.
#
# ------------------------------------------------------------------------ HOW TO SUBMIT
# The training step count is NOT hard-coded, because the convergence sweep
# (results_convergence.jsonl) is re-deriving it.  Pass it in:
#
#     sbatch --export=ALL,RBM_STEPS=<n>,DCGAN_STEPS=<n> seeds/submit_seeds_baselines.sbatch
#
# With no --export the script uses the round-3 values (20,000 each), i.e. it reproduces the
# protocol seed 0 already ran.  Whatever it is given, it prints it and stamps it into every
# run's meta.json, and it REFUSES to run if the two step counts do not match the ones seed 0
# used unless SEED0_STEPS_OVERRIDE=1 is also exported -- because a Table 2 whose seed-0 column
# was trained for 20k steps and whose seeds 1-4 were trained for something else is not a seed
# study, it is four different models.  Exporting the override means "seed 0's baselines will
# be re-run at the new count too"; do that only when that re-run is actually planned.
#
# ------------------------------------------------------------------------------ job list
# 2 arms x 7 training-set sizes x 4 seeds = 56 runs.
#   RBM-10bit   baseline_rbm_r3.py     N in {10, 11, 30, 50, 70, 90, 100}
#   DCGAN-32    baseline_dcgan_r3.py   N in {10, 11, 30, 50, 70, 90, 100}
# N = 11 and N = 100 feed Table 2; N = 10, 30, 50, 70, 90, 100 feed Figure 5 (N = 100 is one
# run serving both, exactly as it did at seed 0).
#
# The data does not move with --seed: data_lens_r3.load_r3 fixes the split at R3_SPLIT_SEED
# (data_lens_r3.py:264-267), so --seed reaches only the RBM's weight init / PCD chains / Gibbs
# generation (baseline_rbm_r3.py:66-67, 109, 159, 238) and the DCGAN's init / noise / paste
# offsets (baseline_dcgan_r3.py:97-98, 120, 158).
#
# --------------------------------------------------------------------------- time budget
# MEASURED at 20k steps on one GPU (logs_lens/r3_rbm_n100.log, r3_dcgan_n100.log):
#   RBM   N = 100  ~25 s      DCGAN  N = 100  ~80 s
# 56 runs ~ 2,940 GPU-seconds = 6 minutes on 8 GPUs.  The wall clock scales linearly with the
# step count, so even a 10x larger count fits inside the 3 h requested.
# ---------------------------------------------------------------------------------------
set -u

ts() { date "+%F %T"; }
REPO=<HOME>/gsoc2026-deeplense-qml/experiments/13_iqp_turbulence_replication
cd "$REPO" || exit 1
PY=/usr/bin/python3
OUT_ROOT="$REPO/outputs_seeds"
RAW="$REPO/results_seeds_raw"
SEEDS=(1 2 3 4)
NS=(100 90 70 50 30 11 10)          # descending: longest first for the work queue

RBM_STEPS="${RBM_STEPS:-20000}"
DCGAN_STEPS="${DCGAN_STEPS:-20000}"
SEED0_STEPS_OVERRIDE="${SEED0_STEPS_OVERRIDE:-0}"
SEED0_RBM_STEPS=20000
SEED0_DCGAN_STEPS=20000

mkdir -p logs_lens "$OUT_ROOT" "$RAW"
echo "[$(ts)] START submit_seeds_baselines.sbatch  job=${SLURM_JOB_ID:-none}  node=$(hostname)"
echo "[$(ts)] RBM_STEPS=$RBM_STEPS  DCGAN_STEPS=$DCGAN_STEPS  (seed 0 used $SEED0_RBM_STEPS / $SEED0_DCGAN_STEPS)"

# ---------------------------------------------------------------- fail-fast: fair comparison
if [ "$RBM_STEPS" -ne "$SEED0_RBM_STEPS" ] || [ "$DCGAN_STEPS" -ne "$SEED0_DCGAN_STEPS" ]; then
    if [ "$SEED0_STEPS_OVERRIDE" != "1" ]; then
        echo "[FATAL] step counts ($RBM_STEPS / $DCGAN_STEPS) differ from seed 0's"
        echo "        ($SEED0_RBM_STEPS / $SEED0_DCGAN_STEPS).  Averaging those together would"
        echo "        mix two different models into one 'mean +- sd'.  Re-export with"
        echo "        SEED0_STEPS_OVERRIDE=1 only if seed 0's baselines are being re-run at the"
        echo "        new count as well, and update SEED0_*_STEPS in this file when they are."
        exit 1
    fi
    echo "[$(ts)] WARNING: running at step counts seed 0 did not use, on an explicit override."
    echo "[$(ts)] Seed 0's baseline rows MUST be re-run at $RBM_STEPS / $DCGAN_STEPS before"
    echo "        aggregate_seeds.py's Table 2 baseline rows mean anything."
fi

# ---------------------------------------------------------------- fail-fast: syntax
echo "[$(ts)] pre-check: py_compile"
for f in baseline_rbm_r3.py baseline_dcgan_r3.py data_lens_r3.py evaluate_r3b.py \
         aggregate_seeds.py; do
    "$PY" -m py_compile "$f" || { echo "[FATAL] py_compile failed for $f"; exit 1; }
done
command -v flock >/dev/null 2>&1 || { echo "[FATAL] flock not available"; exit 1; }

# ---------------------------------------------------------------- fail-fast: CUDA smoke
echo "[$(ts)] pre-check: 5-step CUDA smoke of both baselines at a non-zero seed"
rm -rf logs_lens/seeds_base_cuda_smoke
CUDA_VISIBLE_DEVICES=0 "$PY" baseline_rbm_r3.py --n-train 3 --n-holdout 4 --n-gen 2 --seed 7 \
    --steps 5 --log-every 5 --n-pool 1000 --gen-sweeps 5 --gen-keep 2 --gen-chunk 2 \
    --device cuda --out logs_lens/seeds_base_cuda_smoke/rbm \
    > logs_lens/precheck_seeds_base_cuda.log 2>&1 \
    || { echo "[FATAL] baseline_rbm_r3.py does not run"; tail -30 logs_lens/precheck_seeds_base_cuda.log; exit 1; }
CUDA_VISIBLE_DEVICES=0 "$PY" baseline_dcgan_r3.py --n-train 3 --n-holdout 4 --n-gen 2 --seed 7 \
    --steps 5 --log-every 5 --device cuda --out logs_lens/seeds_base_cuda_smoke/dcgan \
    >> logs_lens/precheck_seeds_base_cuda.log 2>&1 \
    || { echo "[FATAL] baseline_dcgan_r3.py does not run"; tail -30 logs_lens/precheck_seeds_base_cuda.log; exit 1; }
rm -rf logs_lens/seeds_base_cuda_smoke
echo "[$(ts)] pre-checks passed"

# ---------------------------------------------------------------- build the job list
JOBS=()
add_job() { JOBS+=("$1|$2"); }
COMMON="--n-holdout 200 --n-gen 200 --device cuda"

for N in "${NS[@]}"; do
    for S in "${SEEDS[@]}"; do
        O="$OUT_ROOT/base_s$S"
        mkdir -p "$O"
        add_job "s${S}_dcgan_n$N" \
            "$PY baseline_dcgan_r3.py $COMMON --seed $S --n-train $N --steps $DCGAN_STEPS \
             --out $O/dcgan_n$N"
        add_job "s${S}_rbm_n$N" \
            "$PY baseline_rbm_r3.py $COMMON --seed $S --n-train $N --steps $RBM_STEPS \
             --out $O/rbm_n$N"
    done
done

NJOBS=${#JOBS[@]}
EXPECTED=$(( 2 * ${#NS[@]} * ${#SEEDS[@]} ))
echo "[$(ts)] $NJOBS jobs queued (expected $EXPECTED = 2 arms x ${#NS[@]} sizes x ${#SEEDS[@]} seeds)"
if [ "$NJOBS" -ne "$EXPECTED" ]; then
    echo "[FATAL] job list is $NJOBS, expected $EXPECTED -- refusing to run"
    exit 1
fi

# ---------------------------------------------------------------- run: 8-slot work queue
QUEUE="$OUT_ROOT/base_queue.txt"
IDXFILE="$OUT_ROOT/base_next_idx"
LOCKFILE="$OUT_ROOT/base_queue.lock"
FAILFILE="$OUT_ROOT/base_failed.txt"
printf '%s\n' "${JOBS[@]}" > "$QUEUE"
echo 0 > "$IDXFILE"
: > "$LOCKFILE"
: > "$FAILFILE"

worker() {
    slot="$1"
    while true; do
        idx=$(flock "$LOCKFILE" -c "i=\$(cat '$IDXFILE'); echo \$((i+1)) > '$IDXFILE'; echo \$i")
        if [ -z "$idx" ] || [ "$idx" -ge "$NJOBS" ]; then break; fi
        line=$(sed -n "$((idx + 1))p" "$QUEUE")
        tag="${line%%|*}"
        cmd="${line#*|}"
        t_start=$(date +%s)
        echo "[$(ts)] slot $slot START [$((idx + 1))/$NJOBS] $tag"
        if CUDA_VISIBLE_DEVICES="$slot" bash -c "$cmd" > "logs_lens/seedsbase_${tag}.log" 2>&1; then
            echo "[$(ts)] slot $slot OK    $tag ($(( $(date +%s) - t_start )) s)"
        else
            echo "[$(ts)] slot $slot FAIL  $tag ($(( $(date +%s) - t_start )) s) -- logs_lens/seedsbase_${tag}.log"
            tail -15 "logs_lens/seedsbase_${tag}.log"
            echo "$tag" >> "$FAILFILE"
        fi
    done
}

GPU_SLOTS=8
for slot in $(seq 0 $((GPU_SLOTS - 1))); do worker "$slot" & done
wait
NFAIL=$(wc -l < "$FAILFILE")
echo "[$(ts)] all $NJOBS training jobs finished, $NFAIL failed"
[ "$NFAIL" -gt 0 ] && { echo "[$(ts)] failed tags:"; cat "$FAILFILE"; }
fail=$([ "$NFAIL" -gt 0 ] && echo 1 || echo 0)

# ---------------------------------------------------------------- evaluate, one pass per seed
for S in "${SEEDS[@]}"; do
    got=$(find "$OUT_ROOT/base_s$S" -mindepth 2 -maxdepth 2 -name meta.json 2>/dev/null | wc -l)
    echo "[$(ts)] seed $S: $got/${#NS[@]} x 2 baseline run dirs"
    echo "[$(ts)] START evaluate_r3b for baseline seed $S"
    "$PY" evaluate_r3b.py --outputs-dir "$OUT_ROOT/base_s$S" --n-holdout 200 --n-train-max 100 \
        --random-pair-arm "RBM-10bit" --seed 0 --skip-missing \
        --results "$RAW/base_s$S.jsonl" > "logs_lens/seeds_evaluate_base_s$S.log" 2>&1 \
        || { echo "[WARN] evaluate_r3b failed for baseline seed $S"; tail -30 "logs_lens/seeds_evaluate_base_s$S.log"; fail=1; }
    tail -1 "logs_lens/seeds_evaluate_base_s$S.log"
done

# ---------------------------------------------------------------- re-aggregate with baselines
echo "[$(ts)] START aggregate_seeds.py"
"$PY" -u aggregate_seeds.py > logs_lens/seeds_aggregate_with_base.log 2>&1 \
    || { echo "[WARN] aggregate_seeds.py failed"; tail -40 logs_lens/seeds_aggregate_with_base.log; fail=1; }
tail -40 logs_lens/seeds_aggregate_with_base.log

echo "[$(ts)] START figA_scaling_seeds"
"$PY" <HOME>/gsoc2026-deeplense-qml/paper/ml4ps2026/figures/make_figures.py \
    figA_scaling_seeds > logs_lens/seeds_figA.log 2>&1 \
    || { echo "[WARN] figA_scaling_seeds failed"; tail -20 logs_lens/seeds_figA.log; fail=1; }

echo "[$(ts)] DONE submit_seeds_baselines.sbatch (fail=$fail)"
exit "$fail"
