#!/bin/bash
#SBATCH -J iqp_seeds_floor
#SBATCH -A <YOUR_ACCOUNT>
#SBATCH -p dev
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 01:00:00
#SBATCH -o <REPO_ROOT>/code/logs_lens/sbatch-seeds-floor-%j.out
#SBATCH -e <REPO_ROOT>/code/logs_lens/sbatch-seeds-floor-%j.err
#
# The 200-vs-200 real-vs-real floor for the POOLED set AND each substructure class.
#
# The first floor pass (inside submit_seeds_v11.sbatch) was given only outputs_lens_r3, whose
# runs are all class-pooled, so evaluate_v11 built one context and produced one floor row.
# Table 3 is per class and needs a per-class floor, which means the run list has to contain a
# run from each class -- evaluate_v11 derives the set of "kinds" from the runs it is given.
# ---------------------------------------------------------------------------------------
set -u
ts() { date "+%F %T"; }
REPO=<REPO_ROOT>/code
cd "$REPO" || exit 1
PY=/usr/bin/python3
mkdir -p logs_lens
echo "[$(ts)] START submit_seeds_floor.sbatch job=${SLURM_JOB_ID:-none}"
"$PY" -m py_compile evaluate_v11.py metrics_v11.py aggregate_seeds.py || exit 1

fail=0
"$PY" -u evaluate_v11.py --outputs-dir "$REPO/outputs_lens_r3" "$REPO/outputs_lens_class" \
    --n-holdout 200 --floor --n-boot 20 --n-side 200 \
    --floor-results "$REPO/results_v11_floor.jsonl" \
    --results "$REPO/results_v11_floorpass.jsonl" \
    > logs_lens/seeds_v11_floor_classes.log 2>&1 \
    || { echo "[WARN] floor pass failed"; tail -40 logs_lens/seeds_v11_floor_classes.log; fail=1; }
grep -E "floor|kind" logs_lens/seeds_v11_floor_classes.log | tail -12
echo "[$(ts)] floor rows: $(wc -l < "$REPO/results_v11_floor.jsonl" 2>/dev/null || echo 0)"

echo "[$(ts)] START aggregate_seeds.py"
"$PY" -u aggregate_seeds.py > logs_lens/seeds_aggregate_floor.log 2>&1 \
    || { echo "[WARN] aggregate failed"; tail -40 logs_lens/seeds_aggregate_floor.log; fail=1; }
grep -E "^\[aggregate\]" logs_lens/seeds_aggregate_floor.log | tail -20
echo "[$(ts)] DONE (fail=$fail)"
exit "$fail"
