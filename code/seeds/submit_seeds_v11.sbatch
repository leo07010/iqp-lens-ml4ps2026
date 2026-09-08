#!/bin/bash
#SBATCH -J iqp_seeds_v11
#SBATCH -A <YOUR_ACCOUNT>
#SBATCH -p dev
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 02:00:00
#SBATCH -o <HOME>/gsoc2026-deeplense-qml/experiments/13_iqp_turbulence_replication/logs_lens/sbatch-seeds-v11-%j.out
#SBATCH -e <HOME>/gsoc2026-deeplense-qml/experiments/13_iqp_turbulence_replication/logs_lens/sbatch-seeds-v11-%j.err
#
# Re-score every seed's generated images under the v11 metric protocol, then aggregate.
#
# The training runs are untouched by this: v11 changes only how the SAME generated_*.npy are
# scored (fixed [0, 1] PDF-JS grid; Conv2D-MMD^2 normalisation and bandwidth taken from the
# held-out set alone, so the kernel no longer moves with the arm being measured).  Every row
# evaluate_v11.py writes carries the v10 value beside the v11 one, so one pass over one set of
# images produces both protocols and they cannot drift apart.
#
# WHICH RUN DIRECTORIES.  Whole trees for the IQP arms.  For the baselines, only the
# checkpoints their validation set actually selected -- pointing evaluate_v11 at the sweep
# trees would score all twelve DCGAN snapshots per cell and put eleven step counts that were
# never chosen into the table.  The selected directories are read out of
# results_seeds_baselines_es.jsonl, together with the fixed-20k reference dirs for the
# appendix comparison.
#
# --floor adds the 200-vs-200 real-vs-real floor.  The published floor is a 100-vs-100 split
# of the held-out set while every model row is 200 vs 200, and the biased MMD^2 estimator is
# sensitive to sample size, so the two were never on the same scale.
# ---------------------------------------------------------------------------------------
set -u

ts() { date "+%F %T"; }
REPO=<HOME>/gsoc2026-deeplense-qml/experiments/13_iqp_turbulence_replication
cd "$REPO" || exit 1
PY=/usr/bin/python3
RESULTS="$REPO/results_v11_all.jsonl"
FLOOR="$REPO/results_v11_floor.jsonl"

mkdir -p logs_lens
echo "[$(ts)] START submit_seeds_v11.sbatch job=${SLURM_JOB_ID:-none} node=$(hostname)"

echo "[$(ts)] pre-check: py_compile"
for f in evaluate_v11.py metrics_v11.py aggregate_seeds.py; do
    "$PY" -m py_compile "$f" || { echo "[FATAL] py_compile failed for $f"; exit 1; }
done

# ---------------------------------------------------------------- the trees
TREES=("$REPO/outputs_lens_r3" "$REPO/outputs_lens_r3b" "$REPO/outputs_lens_class"
       "$REPO/outputs_lens_scaling" "$REPO/outputs_seeds_theta0")
for S in 1 2 3 4; do
    TREES+=("$REPO/outputs_seeds/s$S" "$REPO/outputs_seeds/cls_s$S")
done
KEEP=()
for t in "${TREES[@]}"; do
    if [ -d "$t" ]; then KEEP+=("$t"); else echo "[WARN] tree missing, skipping: $t"; fi
done

# ---------------------------------------------------------------- the selected baseline dirs
BASE_JSONL="$REPO/results_seeds_baselines_es.jsonl"
RUNS_FILE="$REPO/outputs_seeds/v11_baseline_runs.txt"
mkdir -p "$REPO/outputs_seeds"
: > "$RUNS_FILE"
if [ -f "$BASE_JSONL" ]; then
    "$PY" - "$BASE_JSONL" "$RUNS_FILE" <<'PYEOF'
import json, os, sys
src, dst = sys.argv[1], sys.argv[2]
seen, n = set(), 0
with open(dst, "w") as out:
    for line in open(src):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("kind") not in ("baseline_early_stopped", "baseline_fixed_steps"):
            continue
        rd = r.get("run_dir")
        if rd and os.path.isdir(rd) and rd not in seen:
            seen.add(rd)
            out.write(rd + "\n")
            n += 1
print("[v11] %d selected baseline run dirs -> %s" % (n, dst))
PYEOF
else
    echo "[WARN] $BASE_JSONL not found -- the baselines will be MISSING from the v11 tables"
fi

mapfile -t BASE_RUNS < "$RUNS_FILE"
echo "[$(ts)] ${#KEEP[@]} trees + ${#BASE_RUNS[@]} explicit baseline run dirs"

# ---------------------------------------------------------------- score
fail=0
echo "[$(ts)] START evaluate_v11 over the trees"
"$PY" -u evaluate_v11.py --outputs-dir "${KEEP[@]}" --n-holdout 200 \
    --results "$RESULTS" > logs_lens/seeds_v11_trees.log 2>&1 \
    || { echo "[WARN] evaluate_v11 over trees failed"; tail -40 logs_lens/seeds_v11_trees.log; fail=1; }
tail -2 logs_lens/seeds_v11_trees.log

if [ "${#BASE_RUNS[@]}" -gt 0 ]; then
    echo "[$(ts)] START evaluate_v11 over the selected baseline runs (append)"
    "$PY" -u evaluate_v11.py --runs "${BASE_RUNS[@]}" --n-holdout 200 --append \
        --results "$RESULTS" > logs_lens/seeds_v11_baselines.log 2>&1 \
        || { echo "[WARN] evaluate_v11 over baselines failed"; tail -40 logs_lens/seeds_v11_baselines.log; fail=1; }
    tail -2 logs_lens/seeds_v11_baselines.log
fi

echo "[$(ts)] START evaluate_v11 --floor (200-vs-200 real-vs-real)"
"$PY" -u evaluate_v11.py --outputs-dir "$REPO/outputs_lens_r3" --n-holdout 200 --floor \
    --n-boot 20 --n-side 200 --floor-results "$FLOOR" \
    --results "$REPO/results_v11_floorpass.jsonl" > logs_lens/seeds_v11_floor.log 2>&1 \
    || { echo "[WARN] evaluate_v11 --floor failed"; tail -40 logs_lens/seeds_v11_floor.log; fail=1; }
tail -3 logs_lens/seeds_v11_floor.log

echo "[$(ts)] rows: all=$(wc -l < "$RESULTS" 2>/dev/null || echo 0) floor=$(wc -l < "$FLOOR" 2>/dev/null || echo 0)"

# ---------------------------------------------------------------- aggregate
echo "[$(ts)] START aggregate_seeds.py"
"$PY" -u aggregate_seeds.py > logs_lens/seeds_aggregate_v11.log 2>&1 \
    || { echo "[WARN] aggregate_seeds.py failed"; tail -60 logs_lens/seeds_aggregate_v11.log; fail=1; }
tail -60 logs_lens/seeds_aggregate_v11.log

echo "[$(ts)] START figA_scaling_seeds"
"$PY" <HOME>/gsoc2026-deeplense-qml/paper/ml4ps2026/figures/make_figures.py \
    figA_scaling_seeds > logs_lens/seeds_figA.log 2>&1 \
    || { echo "[WARN] figA_scaling_seeds failed"; tail -20 logs_lens/seeds_figA.log; fail=1; }

echo "[$(ts)] DONE submit_seeds_v11.sbatch (fail=$fail)"
exit "$fail"
