#!/bin/bash
#SBATCH -J iqp_seeds_base_es
#SBATCH -A <YOUR_ACCOUNT>
#SBATCH -p dev
#SBATCH --gres=gpu:8
#SBATCH -c 32
#SBATCH --mem=128G
#SBATCH -t 03:30:00
#SBATCH -o <REPO_ROOT>/code/logs_lens/sbatch-seeds-base-es-%j.out
#SBATCH -e <REPO_ROOT>/code/logs_lens/sbatch-seeds-base-es-%j.err
#
# The classical baselines with VALIDATION-BASED EARLY STOPPING, seeds 0-4.
# Supersedes seeds/submit_seeds_baselines.sbatch, which trained both baselines for a fixed
# 20,000 steps.  That file is kept for the appendix's fixed-step comparison but is not the
# protocol any more, because the convergence sweep (results_convergence.jsonl) showed a fixed
# count cannot be defended for either arm:
#
#   RBM    still improving well past 20k -- 20k is undertrained, 80k is much better
#   DCGAN  not monotone at all, and at N = 100 it degrades after roughly 30k; on GPU it does
#          not even reproduce run to run (cuDNN conv-transpose is non-deterministic)
#
# So each baseline trains once to the top of its sweep, snapshots along the way, and the step
# count is chosen on a SELECTION set of 200 real images that is disjoint from the 200 held-out
# images (seeds/val_extra.py).  The held-out set chooses nothing.  seeds/select_checkpoint.py
# holds the rule and applies it.  The IQP arms are NOT early-stopped -- they keep their fixed
# 4,000 steps -- so this procedure can only favour the baselines.
#
# ------------------------------------------------------------------------------ job list
# arms x sizes x seeds, minus what the convergence sweep already produced:
#   RBM-10bit    baseline_rbm_sweep.py    snapshots 20k / 40k / 80k / 160k
#   DCGAN-32     baseline_dcgan_sweep.py  snapshots every 5k to 60k
#   N in {10, 11, 30, 50, 70, 90, 100}   (11 and 100 feed Table 2; the rest feed Figure 5)
#   seeds 0-4
# seed 0 at N = 11 and N = 100 is REUSED from outputs_convergence/ for both arms (4 cells),
# so 2 x 7 x 5 - 4 = 66 runs.
#
# --------------------------------------------------------------------------- time budget
# MEASURED (outputs_convergence/*/meta.json, one GPU, includes generation at every snapshot):
#   RBM sweep to 160k, 4 snapshots    298 s
#   DCGAN sweep to 60k, 12 snapshots  389 s
# 33 x 298 + 33 x 389 = 22,671 GPU-seconds = 6.3 GPU-hours -> ~47 min on 8 GPUs.
# Plus the selection pass: ~70 prefixes x up to 12 candidates scored on 200 validation
# images, then the winner and the 20k reference scored on the held-out set (~20 min, CPU).
# Requesting 3.5 h.
#
# Partition: dev.  8gpus caps submissions at 5 per user and this account is at that cap.
# ---------------------------------------------------------------------------------------
set -u

ts() { date "+%F %T"; }
REPO=<REPO_ROOT>/code
cd "$REPO" || exit 1
PY=/usr/bin/python3
OUT_ROOT="$REPO/outputs_seeds_es"
CONV="$REPO/outputs_convergence"
RESULTS="$REPO/results_seeds_baselines_es.jsonl"
SEEDS=(0 1 2 3 4)
NS=(100 90 70 50 30 11 10)
RBM_SNAPS="20000 40000 80000 160000"
DCGAN_TOTAL=60000
DCGAN_EVERY=5000

mkdir -p logs_lens "$OUT_ROOT"
echo "[$(ts)] START submit_seeds_baselines_es.sbatch job=${SLURM_JOB_ID:-none} node=$(hostname)"

# ---------------------------------------------------------------- fail-fast
echo "[$(ts)] pre-check: py_compile"
for f in baseline_rbm_sweep.py baseline_dcgan_sweep.py seeds/val_extra.py \
         seeds/select_checkpoint.py evaluate_r3.py evaluate_r3b.py aggregate_seeds.py; do
    "$PY" -m py_compile "$f" || { echo "[FATAL] py_compile failed for $f"; exit 1; }
done
command -v flock >/dev/null 2>&1 || { echo "[FATAL] flock not available"; exit 1; }

# The selection block must exist and must not overlap the held-out block.  If that assertion
# ever fails, the step counts chosen below would be chosen on the test set.
echo "[$(ts)] pre-check: the selection block is built and disjoint from the held-out set"
"$PY" seeds/val_extra.py || { echo "[FATAL] the selection block could not be built/verified"; exit 1; }

echo "[$(ts)] pre-check: 5-step CUDA smoke of both sweep scripts"
rm -rf logs_lens/seeds_es_smoke
CUDA_VISIBLE_DEVICES=0 "$PY" baseline_rbm_sweep.py --n-train 3 --n-holdout 4 --n-gen 2 \
    --seed 7 --snapshots 5 --n-pool 1000 --gen-sweeps 5 --gen-keep 2 --gen-chunk 2 \
    --log-every 5 --device cuda --out-prefix logs_lens/seeds_es_smoke/rbm \
    > logs_lens/precheck_seeds_es.log 2>&1 \
    || { echo "[FATAL] baseline_rbm_sweep.py does not run"; tail -30 logs_lens/precheck_seeds_es.log; exit 1; }
CUDA_VISIBLE_DEVICES=0 "$PY" baseline_dcgan_sweep.py --n-train 3 --n-holdout 4 --n-gen 2 \
    --seed 7 --steps 10 --snap-every 5 --log-every 5 --device cuda \
    --out-prefix logs_lens/seeds_es_smoke/dcgan >> logs_lens/precheck_seeds_es.log 2>&1 \
    || { echo "[FATAL] baseline_dcgan_sweep.py does not run"; tail -30 logs_lens/precheck_seeds_es.log; exit 1; }
rm -rf logs_lens/seeds_es_smoke
echo "[$(ts)] pre-checks passed"

# ---------------------------------------------------------------- build the job list
JOBS=()
PREFIXES=()
add_job() { JOBS+=("$1|$2"); }

for N in "${NS[@]}"; do
    for S in "${SEEDS[@]}"; do
        O="$OUT_ROOT/s$S"
        mkdir -p "$O"
        # seed 0 at N = 11 / 100 already exists under outputs_convergence: reuse, do not re-run
        if [ "$S" -eq 0 ] && { [ "$N" -eq 11 ] || [ "$N" -eq 100 ]; }; then
            PREFIXES+=("$CONV/dcgan_n$N" "$CONV/rbm_n$N")
            continue
        fi
        add_job "s${S}_dcgan_n$N" \
            "$PY baseline_dcgan_sweep.py --n-train $N --n-holdout 200 --n-gen 200 --seed $S \
             --steps $DCGAN_TOTAL --snap-every $DCGAN_EVERY --device cuda \
             --out-prefix $O/dcgan_n$N"
        add_job "s${S}_rbm_n$N" \
            "$PY baseline_rbm_sweep.py --n-train $N --n-holdout 200 --n-gen 200 --seed $S \
             --snapshots $RBM_SNAPS --device cuda --out-prefix $O/rbm_n$N"
        PREFIXES+=("$O/dcgan_n$N" "$O/rbm_n$N")
    done
done

NJOBS=${#JOBS[@]}
EXPECTED=$(( 2 * ${#NS[@]} * ${#SEEDS[@]} - 4 ))
echo "[$(ts)] $NJOBS sweeps queued (expected $EXPECTED), ${#PREFIXES[@]} prefixes to select over"
if [ "$NJOBS" -ne "$EXPECTED" ]; then
    echo "[FATAL] job list is $NJOBS, expected $EXPECTED -- refusing to run"
    exit 1
fi

# ---------------------------------------------------------------- run: 8-slot work queue
QUEUE="$OUT_ROOT/queue.txt"; IDXFILE="$OUT_ROOT/next_idx"
LOCKFILE="$OUT_ROOT/queue.lock"; FAILFILE="$OUT_ROOT/failed.txt"
printf '%s\n' "${JOBS[@]}" > "$QUEUE"
echo 0 > "$IDXFILE"; : > "$LOCKFILE"; : > "$FAILFILE"

worker() {
    slot="$1"
    while true; do
        idx=$(flock "$LOCKFILE" -c "i=\$(cat '$IDXFILE'); echo \$((i+1)) > '$IDXFILE'; echo \$i")
        if [ -z "$idx" ] || [ "$idx" -ge "$NJOBS" ]; then break; fi
        line=$(sed -n "$((idx + 1))p" "$QUEUE")
        tag="${line%%|*}"; cmd="${line#*|}"
        t_start=$(date +%s)
        echo "[$(ts)] slot $slot START [$((idx + 1))/$NJOBS] $tag"
        if CUDA_VISIBLE_DEVICES="$slot" bash -c "$cmd" > "logs_lens/seedses_${tag}.log" 2>&1; then
            echo "[$(ts)] slot $slot OK    $tag ($(( $(date +%s) - t_start )) s)"
        else
            echo "[$(ts)] slot $slot FAIL  $tag ($(( $(date +%s) - t_start )) s) -- logs_lens/seedses_${tag}.log"
            tail -15 "logs_lens/seedses_${tag}.log"
            echo "$tag" >> "$FAILFILE"
        fi
    done
}

for slot in $(seq 0 7); do worker "$slot" & done
wait
NFAIL=$(wc -l < "$FAILFILE")
echo "[$(ts)] all $NJOBS sweeps finished, $NFAIL failed"
[ "$NFAIL" -gt 0 ] && { echo "[$(ts)] failed tags:"; cat "$FAILFILE"; }
fail=$([ "$NFAIL" -gt 0 ] && echo 1 || echo 0)

# ---------------------------------------------------------------- select on the validation set
echo "[$(ts)] START select_checkpoint.py over ${#PREFIXES[@]} prefixes"
"$PY" -u seeds/select_checkpoint.py --prefix "${PREFIXES[@]}" --n-holdout 200 \
    --results "$RESULTS" > logs_lens/seeds_select_checkpoint.log 2>&1 \
    || { echo "[WARN] select_checkpoint.py failed"; tail -40 logs_lens/seeds_select_checkpoint.log; fail=1; }
grep -E "^\[select\] .* -> " logs_lens/seeds_select_checkpoint.log || true
tail -3 logs_lens/seeds_select_checkpoint.log

echo "[$(ts)] START aggregate_seeds.py"
"$PY" -u aggregate_seeds.py > logs_lens/seeds_aggregate_es.log 2>&1 \
    || { echo "[WARN] aggregate_seeds.py failed"; tail -40 logs_lens/seeds_aggregate_es.log; fail=1; }
tail -50 logs_lens/seeds_aggregate_es.log

echo "[$(ts)] DONE submit_seeds_baselines_es.sbatch (fail=$fail)"
exit "$fail"
