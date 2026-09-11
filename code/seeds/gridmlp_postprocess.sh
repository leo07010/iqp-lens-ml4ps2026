#!/bin/bash
# Scoring and tables for the grid/MLP experiment.  Split out of the sbatch so it can be re-run
# on its own without repeating the training.
#
#   1. select_checkpoint.py on the --save-every sweeps (validation block, never the held-out
#      200), the same selector and the same rule the classical baselines use.
#   2. evaluate_v11.py on every fixed-4,000-step run and on the checkpoints step 1 selected,
#      so every number in the table is in the v11 protocol.
#   3. the table and the sample figures.
set -u
ts() { date "+%F %T"; }
REPO=<REPO_ROOT>/code
PAPER=<REPO_ROOT>/paper
cd "$REPO" || exit 1
PY=/usr/bin/python3
OUT_ROOT=<WORK>/gsoc_iqp_gridmlp/seeds
ES_JSONL="$REPO/results_seeds_gridmlp_es.jsonl"
V11_JSONL="$REPO/results_seeds_gridmlp.jsonl"
mkdir -p logs_gridmlp "$PAPER/figures"
fail=0

echo "[$(ts)] STEP 1  checkpoint selection on the validation block"
PREFIXES=()
for seed in 0 1 2 3 4; do
  for N in 11 100; do
    PREFIXES+=("$OUT_ROOT/s$seed/n12es_m4flat_n$N" "$OUT_ROOT/s$seed/mlpes_m4flat_n$N")
  done
done
"$PY" -u seeds/select_checkpoint.py --prefix "${PREFIXES[@]}" --results "$ES_JSONL" \
    --label gridmlp --fixed-reference-steps 4000 --n-holdout 200 \
    > logs_gridmlp/select.log 2>&1 \
    || { echo "[WARN] select_checkpoint failed"; tail -40 logs_gridmlp/select.log; fail=1; }
grep -E "^\[select\] .* -> " logs_gridmlp/select.log | tail -25
echo "[$(ts)] selection rows: $(wc -l < "$ES_JSONL" 2>/dev/null || echo 0)"

echo "[$(ts)] STEP 2  v11 scoring"
RUNS_FILE="$OUT_ROOT/v11_runs.txt"
"$PY" - "$OUT_ROOT" "$ES_JSONL" "$RUNS_FILE" <<'PYEOF'
import json, os, sys
root, es_jsonl, dst = sys.argv[1], sys.argv[2], sys.argv[3]
runs = []
# every fixed-4,000-step run
for seed in range(5):
    for N in (11, 100):
        for name in ("n12_m4flat_n%d" % N, "n12z_m4flat_n%d" % N, "mlp_m4flat_n%d" % N,
                     "mlpzs_m4flat_n%d" % N, "mlpm2_m2flat_n%d" % N):
            d = os.path.join(root, "s%d" % seed, name)
            if os.path.isdir(d):
                runs.append(d)
            else:
                print("[v11] MISSING %s" % d)
# plus whatever the selector picked (and its own 4k reference row)
if os.path.isfile(es_jsonl):
    for line in open(es_jsonl):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        d = r.get("run_dir")
        if d and os.path.isdir(d) and d not in runs:
            runs.append(d)
open(dst, "w").write("\n".join(runs) + "\n")
print("[v11] %d run dirs -> %s" % (len(runs), dst))
PYEOF
mapfile -t RUNS < "$RUNS_FILE"
"$PY" -u evaluate_v11.py --runs "${RUNS[@]}" --n-holdout 200 --results "$V11_JSONL" \
    > logs_gridmlp/v11.log 2>&1 \
    || { echo "[WARN] evaluate_v11 failed"; tail -40 logs_gridmlp/v11.log; fail=1; }
tail -3 logs_gridmlp/v11.log
echo "[$(ts)] v11 rows: $(wc -l < "$V11_JSONL" 2>/dev/null || echo 0)"

echo "[$(ts)] STEP 3  table and figures"
"$PY" -u "$PAPER/make_seed_tables_d.py" > logs_gridmlp/table_d.log 2>&1 \
    || { echo "[WARN] make_seed_tables_d failed"; tail -40 logs_gridmlp/table_d.log; fail=1; }
tail -5 logs_gridmlp/table_d.log
"$PY" -u "$PAPER/fig_src/make_gridmlp_samples.py" > logs_gridmlp/figs.log 2>&1 \
    || { echo "[WARN] figure script failed"; tail -40 logs_gridmlp/figs.log; fail=1; }
tail -4 logs_gridmlp/figs.log
echo "[$(ts)] postprocess finished (fail=$fail)"
exit "$fail"
