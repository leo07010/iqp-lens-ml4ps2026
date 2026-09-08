#!/bin/bash
#SBATCH -J iqp_es_v11
#SBATCH -A <YOUR_ACCOUNT>
#SBATCH -p dev
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -o <HOME>/gsoc2026-deeplense-qml/experiments/13_iqp_turbulence_replication/logs_lens/sbatch-iqpes-v11-%j.out
#SBATCH -e <HOME>/gsoc2026-deeplense-qml/experiments/13_iqp_turbulence_replication/logs_lens/sbatch-iqpes-v11-%j.err
#
# Score the IQP early-stopping checkpoints under the v11 metric protocol.
#
# The SELECTION already happened, on the validation set, using the same two v10 metrics the
# baselines were selected with (seeds/select_checkpoint.py) -- that symmetry is the point of
# the exercise and must not be disturbed.  What is missing is the v11 numbers for the
# checkpoints it picked, so the IQP early-stopping table can be read in the same protocol as
# the rest of Part 1.
#
# Only the rows the selector actually emitted are scored: the chosen checkpoint and the
# step-4,000 reference for each (arm, N, seed).  Scoring all 225 snapshots would put 180 step
# counts nobody selected into the file.
# ---------------------------------------------------------------------------------------
set -u
ts() { date "+%F %T"; }
REPO=<HOME>/gsoc2026-deeplense-qml/experiments/13_iqp_turbulence_replication
cd "$REPO" || exit 1
PY=/usr/bin/python3
mkdir -p logs_lens
echo "[$(ts)] START job=${SLURM_JOB_ID:-none}"
"$PY" -m py_compile evaluate_v11.py aggregate_seeds.py || exit 1

RUNS_FILE="$REPO/outputs_seeds_iqp_es/v11_runs.txt"
"$PY" - "$REPO/results_seeds_iqp_es.jsonl" "$RUNS_FILE" <<'PYEOF'
import json, os, sys
src, dst = sys.argv[1], sys.argv[2]
seen = []
for line in open(src):
    line = line.strip()
    if not line:
        continue
    r = json.loads(line)
    if r.get("kind") not in ("iqp_early_stopped", "iqp_fixed_steps"):
        continue
    rd = r.get("run_dir")
    if rd and os.path.isdir(rd) and rd not in seen:
        seen.append(rd)
with open(dst, "w") as fh:
    fh.write("\n".join(seen) + "\n")
print("[iqpes-v11] %d distinct run dirs -> %s" % (len(seen), dst))
PYEOF
mapfile -t RUNS < "$RUNS_FILE"
echo "[$(ts)] scoring ${#RUNS[@]} run dirs"

fail=0
"$PY" -u evaluate_v11.py --runs "${RUNS[@]}" --n-holdout 200 \
    --results "$REPO/results_v11_iqp_es.jsonl" > logs_lens/iqpes_v11.log 2>&1 \
    || { echo "[WARN] evaluate_v11 failed"; tail -40 logs_lens/iqpes_v11.log; fail=1; }
tail -2 logs_lens/iqpes_v11.log
echo "[$(ts)] rows: $(wc -l < "$REPO/results_v11_iqp_es.jsonl" 2>/dev/null || echo 0)"

echo "[$(ts)] START aggregate_seeds.py"
"$PY" -u aggregate_seeds.py > logs_lens/seeds_aggregate_iqpes_v11.log 2>&1 \
    || { echo "[WARN] aggregate failed"; tail -40 logs_lens/seeds_aggregate_iqpes_v11.log; fail=1; }
grep -E "^\[aggregate\]" logs_lens/seeds_aggregate_iqpes_v11.log | tail -15
echo "[$(ts)] DONE (fail=$fail)"
exit "$fail"
