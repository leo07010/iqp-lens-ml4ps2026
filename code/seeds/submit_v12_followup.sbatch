#!/bin/bash
#SBATCH -J v12_follow
#SBATCH -A <YOUR_ACCOUNT>
#SBATCH -p dev
#SBATCH --gres=gpu:8
#SBATCH -c 32
#SBATCH --mem=128G
#SBATCH -t 03:00:00
#SBATCH -o <REPO_ROOT>/code/logs_lens/sbatch-v12follow-%j.out
#SBATCH -e <REPO_ROOT>/code/logs_lens/sbatch-v12follow-%j.err
#
# Follow-up to seeds/submit_v12_grid.sbatch:
#   (a) the data-scaling cells, N = 5 and N = 100, for the TOP_K arms of the N = 11 grid;
#   (b) the 16,000-step schedule with snapshots at 4/8/12/16k for the top arm, so
#       seeds/select_checkpoint.py can apply to the IQP side the SAME validation-based rule
#       the classical baselines get -- the main table still reports the fixed 4,000 steps.
#
# The arms are not hand-copied: v12_select_arms.py ranks the grid's own rows and rebuilds each
# arm's CLI from its meta.json, so what runs here is what ran there.
#
# ~60 short runs + 5 long ones ~ 6,000 GPU-s / 8 = 13 min, plus evaluation and selection.
# Disk ~ 450 MB.  Partition dev.
# ---------------------------------------------------------------------------------------
set -u
ts() { date "+%F %T"; }
REPO=<REPO_ROOT>/code
cd "$REPO" || exit 1
PY=/usr/bin/python3
OUT_ROOT="$REPO/outputs_v12"
SEEDS=(0 1 2 3 4)
TOP_K=${TOP_K:-6}
mkdir -p logs_lens "$OUT_ROOT"
echo "[$(ts)] START submit_v12_followup.sbatch job=${SLURM_JOB_ID:-none} node=$(hostname)"

"$PY" -m py_compile train_iqp_affine_v12.py v12_select_arms.py v12_aggregate.py \
    seeds/select_checkpoint.py || { echo "[FATAL] py_compile"; exit 1; }
# PRECONDITION.  This job is chained with --dependency=afterany, not afterok, deliberately:
# the grid job exits 1 if ANY of its 220 runs or its three post-steps warns, and with afterok a
# single transient failure would leave this job permanently DependencyNeverSatisfied and the
# unattended run would deliver nothing.  afterany + this explicit check gives the same
# guarantee without that failure mode: proceed only if the grid actually produced a full
# N = 11 grid (>= 200 of the 220 rows).
N_ROWS=$("$PY" - <<'PYEOF2'
import json, os
p = "<REPO_ROOT>/code/results_seeds_v12.jsonl"
n = 0
if os.path.isfile(p):
    for line in open(p):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if str(r.get("arm", "")).startswith("IQP-v12") and int(r.get("n_train", -1)) == 11:
            n += 1
print(n)
PYEOF2
)
echo "[$(ts)] grid produced $N_ROWS IQP-v12 rows at N=11 (need >= 200)"
[ "${N_ROWS:-0}" -lt 200 ] && { echo "[FATAL] the grid did not finish -- refusing to run"; exit 1; }
AVAIL_KB=$(df -Pk "$REPO" | tail -1 | awk '{print $4}')
echo "[$(ts)] free space: $((AVAIL_KB / 1024)) MB"
[ "$AVAIL_KB" -lt 1500000 ] && { echo "[FATAL] < 1.5 GB free"; exit 1; }

mapfile -t SEL < <("$PY" v12_select_arms.py --n-train 11 --top "$TOP_K" 2>logs_lens/v12_select.log)
cat logs_lens/v12_select.log
echo "[$(ts)] ${#SEL[@]} arms selected"
[ "${#SEL[@]}" -eq 0 ] && { echo "[FATAL] no arms selected"; exit 1; }

COMMON="--n-holdout 200 --n-gen 200 --device cuda --prior resample --lr 0.02 \
 --theta0-steps 1000 --infer-restarts 8 --infer-steps 1000 --infer-lr 0.005 \
 --infer-chunk 50 --log-every 2000 --sigmas auto3"

JOBS=(); add_job() { JOBS+=("$1|$2"); }
for S in "${SEEDS[@]}"; do
    O="$OUT_ROOT/s$S"; mkdir -p "$O"
    for N in 5 100; do
        for entry in "${SEL[@]}"; do
            tag="${entry%%|*}"; opt="${entry#*|}"
            T="n${N}_${tag}"
            add_job "s${S}_$T" "$PY -u train_iqp_affine_v12.py $COMMON --steps 4000 \
                --seed $S --n-train $N $opt --out $O/$T"
        done
    done
done
# the 16,000-step schedule, top arm only, N = 11 (longest jobs first in the queue)
TOPTAG="${SEL[0]%%|*}"; TOPOPT="${SEL[0]#*|}"
SCHED=()
for S in "${SEEDS[@]}"; do
    O="$OUT_ROOT/sched_s$S"; mkdir -p "$O"
    SCHED+=("s${S}_sched_${TOPTAG}|$PY -u train_iqp_affine_v12.py $COMMON --steps 16000 \
        --save-every 4000 8000 12000 16000 --seed $S --n-train 11 $TOPOPT --out $O/$TOPTAG")
done
JOBS=("${SCHED[@]}" "${JOBS[@]}")
NJOBS=${#JOBS[@]}
echo "[$(ts)] $NJOBS runs queued (${#SCHED[@]} schedule + $(( NJOBS - ${#SCHED[@]} )) scaling)"

QUEUE="$OUT_ROOT/queue2.txt"; IDXFILE="$OUT_ROOT/next_idx2"
LOCKFILE="$OUT_ROOT/queue2.lock"; FAILFILE="$OUT_ROOT/failed2.txt"
printf '%s\n' "${JOBS[@]}" > "$QUEUE"; echo 0 > "$IDXFILE"; : > "$LOCKFILE"; : > "$FAILFILE"
worker() {
    slot="$1"
    while true; do
        idx=$(flock "$LOCKFILE" -c "i=\$(cat '$IDXFILE'); echo \$((i+1)) > '$IDXFILE'; echo \$i")
        if [ -z "$idx" ] || [ "$idx" -ge "$NJOBS" ]; then break; fi
        line=$(sed -n "$((idx + 1))p" "$QUEUE")
        tag="${line%%|*}"; cmd="${line#*|}"
        t0=$(date +%s)
        if CUDA_VISIBLE_DEVICES="$slot" bash -c "$cmd" > "logs_lens/v12f_${tag}.log" 2>&1; then
            echo "[$(ts)] slot $slot OK   [$((idx + 1))/$NJOBS] $tag ($(( $(date +%s) - t0 )) s)"
        else
            echo "[$(ts)] slot $slot FAIL [$((idx + 1))/$NJOBS] $tag ($(( $(date +%s) - t0 )) s)"
            tail -15 "logs_lens/v12f_${tag}.log"; echo "$tag" >> "$FAILFILE"
        fi
    done
}
for slot in $(seq 0 7); do worker "$slot" & done
wait
NFAIL=$(wc -l < "$FAILFILE"); echo "[$(ts)] $NJOBS runs finished, $NFAIL failed"
[ "$NFAIL" -gt 0 ] && cat "$FAILFILE"
fail=$([ "$NFAIL" -gt 0 ] && echo 1 || echo 0)
du -sh "$OUT_ROOT"; df -Pk "$REPO" | tail -1

# ---------------------------------------------------------------- score everything, once
echo "[$(ts)] evaluate_v11 over the whole v12 tree (grid + follow-up + schedule snapshots)"
# written to a temp file and moved into place only on success, so a crash here cannot
# truncate the grid's results file and leave the tables unbuildable
cp "$REPO/results_seeds_v12.jsonl" "$REPO/results_seeds_v12.jsonl.grid-backup"
if "$PY" -u evaluate_v11.py --outputs-dir "$OUT_ROOT" --n-holdout 200 --no-legacy --quiet \
        --results "$REPO/results_seeds_v12.jsonl.tmp" > logs_lens/v12_eval2.log 2>&1 \
        && [ -s "$REPO/results_seeds_v12.jsonl.tmp" ]; then
    mv "$REPO/results_seeds_v12.jsonl.tmp" "$REPO/results_seeds_v12.jsonl"
else
    echo "[WARN] evaluate_v11 failed -- keeping the grid's results file"; tail -40 logs_lens/v12_eval2.log
    rm -f "$REPO/results_seeds_v12.jsonl.tmp"; fail=1
fi
tail -3 logs_lens/v12_eval2.log
echo "[$(ts)] rows: $(wc -l < "$REPO/results_seeds_v12.jsonl")"

# ------------------------------------- IQP-side validation-based early stopping (symmetry)
# The SAME rule the classical baselines get (seeds/select_checkpoint.py): candidates are the
# 4/8/12/16k snapshots, scored on the 200-image SELECTION block, which is disjoint from the
# held-out 200; the winner is the smaller mean rank of PDF-JS and raw Conv2D-MMD^2, ties to
# fewer steps.  Reported as an appendix -- the main table keeps the fixed 4,000 steps.
echo "[$(ts)] select_checkpoint over the 16k schedules"
PREFIXES=()
for S in "${SEEDS[@]}"; do PREFIXES+=("$OUT_ROOT/sched_s$S/$TOPTAG"); done
"$PY" -u seeds/select_checkpoint.py --prefix "${PREFIXES[@]}" --label iqp_v12 \
    --fixed-reference-steps 4000 --n-holdout 200 \
    --results "$REPO/results_seeds_v12_es.jsonl" > logs_lens/v12_select_ckpt.log 2>&1 \
    || { echo "[WARN] select_checkpoint failed"; tail -40 logs_lens/v12_select_ckpt.log; fail=1; }
tail -12 logs_lens/v12_select_ckpt.log

# The tables and figures are the deliverable of the whole chain, so they are rebuilt
# unconditionally from whatever results file survived above.
echo "[$(ts)] v12_aggregate.py"
"$PY" -u v12_aggregate.py > logs_lens/v12_aggregate2.log 2>&1 \
    || { echo "[WARN] aggregate failed"; tail -40 logs_lens/v12_aggregate2.log; fail=1; }
tail -90 logs_lens/v12_aggregate2.log
for N in 11 100; do
    "$PY" -u v12_figs.py --n-train "$N" >> logs_lens/v12_figs.log 2>&1 \
        || echo "[WARN] figs N=$N failed"
done
tail -8 logs_lens/v12_figs.log
echo "[$(ts)] DONE (fail=$fail)"
exit "$fail"
