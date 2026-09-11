#!/bin/bash
#SBATCH -J iqp_seeds_iqpes
#SBATCH -A <YOUR_ACCOUNT>
#SBATCH -p dev
#SBATCH --gres=gpu:8
#SBATCH -c 32
#SBATCH --mem=128G
#SBATCH -t 02:30:00
#SBATCH -o <REPO_ROOT>/code/logs_lens/sbatch-seeds-iqpes-%j.out
#SBATCH -e <REPO_ROOT>/code/logs_lens/sbatch-seeds-iqpes-%j.err
#
# The IQP arms under the SAME validation early-stopping rule the classical baselines get.
#
# WHY.  seeds/submit_seeds_baselines_es.sbatch gives RBM and DCGAN a step count chosen on a
# validation set, because no fixed count could be defended for either.  The IQP arms kept a
# fixed 4,000 steps.  That asymmetry favours the baselines, and it is the first thing a
# reviewer will ask about, so this job removes it: every IQP arm now gets checkpoints and the
# same selection rule, and the comparison becomes symmetric.
#
# HOW.  The checkpoints cannot come from separate short runs -- run_adam_scalar drives a
# CosineAnnealingLR whose T_max is the total step count, so a 1,000-step run and the first
# 1,000 steps of a 16,000-step run follow different learning-rate trajectories.  So each cell
# is ONE 16,000-step run that emits a complete output directory at 1k / 2k / 4k / 8k / 16k
# (seeds/train_iqp_affine_r3_snap.py --save-every).  The theta0 warm-start is unchanged.
# Verified before running: with --save-every unset the fork reproduces train_iqp_affine_r3.py
# bit-for-bit, and its final snapshot is bit-for-bit a plain run of the same length.
#
# The snapshot layout (<out>_s<k/1000>k) and the arm tagging ("... s004k") are the baseline
# sweeps', so seeds/select_checkpoint.py selects over them with no change beyond two new
# flags: --label iqp (keeps these rows out of the baseline tables) and
# --fixed-reference-steps 4000 (the published protocol, kept for the appendix comparison --
# note 4,000 is itself one of the snapshots, so the comparison is exact, not interpolated).
#
# ------------------------------------------------------------------------------ job list
#   IQP-affine-M4-MMDflat-resample   N in {10, 11, 30, 50, 70, 90, 100}  x seeds 0-4  = 35
#   IQP-affine-M4-auto3-resample     N in {11, 100}                      x seeds 0-4  = 10
# 45 runs, each emitting 5 checkpoints = 225 scored output directories.
#
# --------------------------------------------------------------------------- time budget
# MEASURED at 4,000 steps (logs_lens/r3b_*.log): joint step 25.6 ms at N = 100, 4.9 ms at
# N = 11; projection 26 s; generation < 1 s.  At 16,000 steps with 5 snapshots:
#   N = 100  16k x 25.6 ms + 5 x 27 s + overhead ~ 9 min
#   N = 11   16k x 4.9 ms  + 5 x 27 s + overhead ~ 4 min
# ~280 GPU-minutes over 45 runs -> ~35 min on 8 GPUs.  Requesting 2.5 h.
# ---------------------------------------------------------------------------------------
set -u

ts() { date "+%F %T"; }
REPO=<REPO_ROOT>/code
cd "$REPO" || exit 1
PY=/usr/bin/python3
OUT_ROOT="$REPO/outputs_seeds_iqp_es"
RESULTS="$REPO/results_seeds_iqp_es.jsonl"
SEEDS=(0 1 2 3 4)
FLAT_N=(100 90 70 50 30 11 10)
AUTO3_N=(100 11)
SNAPS="1000 2000 4000 8000 16000"

mkdir -p logs_lens "$OUT_ROOT"
echo "[$(ts)] START submit_seeds_iqp_es.sbatch job=${SLURM_JOB_ID:-none} node=$(hostname)"

# ---------------------------------------------------------------- fail-fast
echo "[$(ts)] pre-check: py_compile"
for f in seeds/train_iqp_affine_r3_snap.py seeds/select_checkpoint.py seeds/val_extra.py \
         evaluate_r3.py evaluate_r3b.py aggregate_seeds.py; do
    "$PY" -m py_compile "$f" || { echo "[FATAL] py_compile failed for $f"; exit 1; }
done
command -v flock >/dev/null 2>&1 || { echo "[FATAL] flock not available"; exit 1; }
"$PY" seeds/val_extra.py || { echo "[FATAL] the selection block is not available"; exit 1; }

# The whole point is that these snapshots are the same computation the published runs did, so
# check it rather than assert it: with --save-every unset the fork must reproduce the original.
echo "[$(ts)] pre-check: the fork reproduces train_iqp_affine_r3.py with --save-every unset"
rm -rf logs_lens/iqpes_check
SMOKE="--n-train 5 --n-holdout 6 --n-gen 3 --seed 0 --M 4 --objective mmdflat --prior resample \
 --steps 60 --theta0-steps 40 --infer-restarts 2 --infer-steps 20 --infer-chunk 3 \
 --log-every 100 --device cuda"
CUDA_VISIBLE_DEVICES=0 "$PY" train_iqp_affine_r3.py $SMOKE --out logs_lens/iqpes_check/orig \
    > logs_lens/precheck_iqpes.log 2>&1 || { echo "[FATAL] original failed"; tail -20 logs_lens/precheck_iqpes.log; exit 1; }
CUDA_VISIBLE_DEVICES=0 "$PY" seeds/train_iqp_affine_r3_snap.py $SMOKE --out logs_lens/iqpes_check/fork \
    >> logs_lens/precheck_iqpes.log 2>&1 || { echo "[FATAL] fork failed"; tail -20 logs_lens/precheck_iqpes.log; exit 1; }
"$PY" - <<'PYEOF' || { echo "[FATAL] the fork does NOT reproduce the original"; exit 1; }
import numpy as np, os
b = "logs_lens/iqpes_check"
bad = []
for f in ("theta0.npy", "Phi.npy", "z_train.npy", "z_holdout.npy", "generated_000.npy"):
    a = np.load(os.path.join(b, "orig", f)); c = np.load(os.path.join(b, "fork", f))
    if not np.array_equal(a, c):
        bad.append("%s max|d|=%.3e" % (f, float(np.abs(a - c).max())))
if bad:
    raise SystemExit("fork != original: " + "; ".join(bad))
print("[precheck] fork reproduces the original bit-for-bit with --save-every unset")
PYEOF
rm -rf logs_lens/iqpes_check
echo "[$(ts)] pre-checks passed"

# ---------------------------------------------------------------- build the job list
JOBS=()
PREFIXES=()
add_job() { JOBS+=("$1|$2"); }
COMMON="--n-holdout 200 --n-gen 200 --device cuda --lr 0.02 --steps 16000 \
 --theta0-steps 1000 --infer-restarts 8 --infer-steps 1000 --infer-lr 0.005 --infer-chunk 50 \
 --log-every 4000 --M 4 --prior resample --save-every $SNAPS"

for N in "${FLAT_N[@]}"; do
    for S in "${SEEDS[@]}"; do
        O="$OUT_ROOT/s$S"; mkdir -p "$O"
        add_job "s${S}_m4flat_n$N" \
            "$PY seeds/train_iqp_affine_r3_snap.py $COMMON --objective mmdflat --seed $S \
             --n-train $N --out $O/m4flat_n$N"
        PREFIXES+=("$O/m4flat_n$N")
    done
done
for N in "${AUTO3_N[@]}"; do
    for S in "${SEEDS[@]}"; do
        O="$OUT_ROOT/s$S"; mkdir -p "$O"
        add_job "s${S}_m4auto3_n$N" \
            "$PY seeds/train_iqp_affine_r3_snap.py $COMMON --objective auto3 --sigmas auto3 \
             --seed $S --n-train $N --out $O/m4auto3_n$N"
        PREFIXES+=("$O/m4auto3_n$N")
    done
done

NJOBS=${#JOBS[@]}
EXPECTED=$(( ${#FLAT_N[@]} * ${#SEEDS[@]} + ${#AUTO3_N[@]} * ${#SEEDS[@]} ))
echo "[$(ts)] $NJOBS runs queued (expected $EXPECTED), each emitting 5 checkpoints"
[ "$NJOBS" -ne "$EXPECTED" ] && { echo "[FATAL] job list is $NJOBS, expected $EXPECTED"; exit 1; }

# ---------------------------------------------------------------- run: 8-slot work queue
QUEUE="$OUT_ROOT/queue.txt"; IDXFILE="$OUT_ROOT/next_idx"
LOCKFILE="$OUT_ROOT/queue.lock"; FAILFILE="$OUT_ROOT/failed.txt"
printf '%s\n' "${JOBS[@]}" > "$QUEUE"; echo 0 > "$IDXFILE"; : > "$LOCKFILE"; : > "$FAILFILE"

worker() {
    slot="$1"
    while true; do
        idx=$(flock "$LOCKFILE" -c "i=\$(cat '$IDXFILE'); echo \$((i+1)) > '$IDXFILE'; echo \$i")
        if [ -z "$idx" ] || [ "$idx" -ge "$NJOBS" ]; then break; fi
        line=$(sed -n "$((idx + 1))p" "$QUEUE")
        tag="${line%%|*}"; cmd="${line#*|}"
        t_start=$(date +%s)
        echo "[$(ts)] slot $slot START [$((idx + 1))/$NJOBS] $tag"
        if CUDA_VISIBLE_DEVICES="$slot" bash -c "$cmd" > "logs_lens/iqpes_${tag}.log" 2>&1; then
            echo "[$(ts)] slot $slot OK    $tag ($(( $(date +%s) - t_start )) s)"
        else
            echo "[$(ts)] slot $slot FAIL  $tag ($(( $(date +%s) - t_start )) s)"
            tail -15 "logs_lens/iqpes_${tag}.log"
            echo "$tag" >> "$FAILFILE"
        fi
    done
}
for slot in $(seq 0 7); do worker "$slot" & done
wait
NFAIL=$(wc -l < "$FAILFILE")
echo "[$(ts)] all $NJOBS runs finished, $NFAIL failed"
[ "$NFAIL" -gt 0 ] && { echo "[$(ts)] failed:"; cat "$FAILFILE"; }
fail=$([ "$NFAIL" -gt 0 ] && echo 1 || echo 0)
echo "[$(ts)] checkpoint dirs written: $(find "$OUT_ROOT" -mindepth 2 -maxdepth 2 -name meta.json | wc -l) (expected $((NJOBS * 5)))"

# ---------------------------------------------------------------- select on the validation set
echo "[$(ts)] START select_checkpoint.py over ${#PREFIXES[@]} prefixes"
"$PY" -u seeds/select_checkpoint.py --prefix "${PREFIXES[@]}" --n-holdout 200 \
    --label iqp --fixed-reference-steps 4000 --results "$RESULTS" \
    > logs_lens/seeds_select_iqp.log 2>&1 \
    || { echo "[WARN] select_checkpoint failed"; tail -40 logs_lens/seeds_select_iqp.log; fail=1; }
grep -E "^\[select\] .* -> " logs_lens/seeds_select_iqp.log || true
tail -3 logs_lens/seeds_select_iqp.log

echo "[$(ts)] START aggregate_seeds.py"
"$PY" -u aggregate_seeds.py > logs_lens/seeds_aggregate_iqpes.log 2>&1 \
    || { echo "[WARN] aggregate failed"; tail -40 logs_lens/seeds_aggregate_iqpes.log; fail=1; }
grep -E "^\[aggregate\]" logs_lens/seeds_aggregate_iqpes.log | tail -20

echo "[$(ts)] DONE submit_seeds_iqp_es.sbatch (fail=$fail)"
exit "$fail"
