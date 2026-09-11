#!/bin/bash
#SBATCH -J iqp_seeds_restart
#SBATCH -A <YOUR_ACCOUNT>
#SBATCH -p dev
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -o <REPO_ROOT>/code/logs_lens/sbatch-seeds-restart-%j.out
#SBATCH -e <REPO_ROOT>/code/logs_lens/sbatch-seeds-restart-%j.err
#
# Per-restart decomposition of the held-out projection correlation, for the reviewer objection
# that restart 1 starts at the nearest training image's trained z.  See
# seeds/diag_projection_restarts.py for what it computes and why it is post-hoc rather than a
# fork of the trainer.
#
# Submit with SEEDS0_ONLY=1 to score only the four seed-0 run directories, which already exist
# -- that is the validation pass, and it also produces seed 0's real rows.  Submit with no
# override (and a dependency on the seed sweep) to score seeds 1-4 as well.
#
#   sbatch --export=ALL,SEEDS0_ONLY=1 seeds/submit_seeds_restarts.sbatch
#   sbatch --dependency=afterany:<seed sweep job> seeds/submit_seeds_restarts.sbatch
#
# TRUNCATE=1 empties results_seeds_restarts.jsonl first; without it rows are appended.
#
# Cost: one replay of a run's projection stage, ~26 s per run directory at R = 8 x 200
# held-out images (measured: meta.json's projection_seconds).  20 run dirs ~ 9 minutes.
# ---------------------------------------------------------------------------------------
set -u

ts() { date "+%F %T"; }
REPO=<REPO_ROOT>/code
cd "$REPO" || exit 1
PY=/usr/bin/python3
OUT_ROOT="$REPO/outputs_seeds"
RESULTS="$REPO/results_seeds_restarts.jsonl"
SEEDS0_ONLY="${SEEDS0_ONLY:-0}"
TRUNCATE="${TRUNCATE:-0}"

mkdir -p logs_lens
echo "[$(ts)] START submit_seeds_restarts.sbatch job=${SLURM_JOB_ID:-none} node=$(hostname)"
echo "[$(ts)] SEEDS0_ONLY=$SEEDS0_ONLY TRUNCATE=$TRUNCATE"

"$PY" -m py_compile seeds/diag_projection_restarts.py aggregate_seeds.py \
    || { echo "[FATAL] py_compile failed"; exit 1; }

[ "$TRUNCATE" = "1" ] && : > "$RESULTS"

# The two arms the paper quotes a projection correlation for, at both training-set sizes.
DIRS=("$REPO/outputs_lens_r3b/iqp_m4_mmdflat_resample_n11"
      "$REPO/outputs_lens_r3b/iqp_m4_mmdflat_resample_n100"
      "$REPO/outputs_lens_r3b/iqp_m4_auto3_resample_n11"
      "$REPO/outputs_lens_r3b/iqp_m4_auto3_resample_n100")
if [ "$SEEDS0_ONLY" != "1" ]; then
    for S in 1 2 3 4; do
        for A in iqp_m4_mmdflat_resample iqp_m4_auto3_resample; do
            for N in 11 100; do
                DIRS+=("$OUT_ROOT/s$S/${A}_n$N")
            done
        done
    done
fi

echo "[$(ts)] scoring ${#DIRS[@]} run directories"
missing=0
for d in "${DIRS[@]}"; do
    [ -f "$d/meta.json" ] || { echo "[WARN] missing $d/meta.json"; missing=$((missing + 1)); }
done
[ "$missing" -gt 0 ] && echo "[$(ts)] $missing directories are missing and will be skipped"

fail=0
CUDA_VISIBLE_DEVICES=0 "$PY" -u seeds/diag_projection_restarts.py --run-dir "${DIRS[@]}" \
    --results "$RESULTS" --device cuda --infer-chunk 50 \
    2>&1 | tee "logs_lens/seeds_diag_restarts_${SLURM_JOB_ID:-local}.log"
rc=${PIPESTATUS[0]}
[ "$rc" -ne 0 ] && { echo "[WARN] diag_projection_restarts.py exited $rc"; fail=1; }

if grep -q "MISMATCH" "logs_lens/seeds_diag_restarts_${SLURM_JOB_ID:-local}.log"; then
    echo "[$(ts)] WARNING: at least one replay did not reproduce the recorded"
    echo "        projection_corr_mean.  Those rows carry replay_matches_recorded=false and"
    echo "        aggregate_seeds.py marks them; do not quote them until the cause is known."
    fail=1
fi

if [ "$SEEDS0_ONLY" != "1" ]; then
    echo "[$(ts)] START aggregate_seeds.py"
    "$PY" -u aggregate_seeds.py > logs_lens/seeds_aggregate_restarts.log 2>&1 \
        || { echo "[WARN] aggregate_seeds.py failed"; tail -40 logs_lens/seeds_aggregate_restarts.log; fail=1; }
    tail -50 logs_lens/seeds_aggregate_restarts.log
fi

echo "[$(ts)] DONE submit_seeds_restarts.sbatch (fail=$fail)"
exit "$fail"
