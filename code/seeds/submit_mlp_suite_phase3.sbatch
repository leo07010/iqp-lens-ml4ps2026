#!/bin/bash
#SBATCH -J mlpsuite3
#SBATCH -A <YOUR_ACCOUNT>
#SBATCH -p dev
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -o <HOME>/gsoc2026-deeplense-qml/experiments/13_iqp_turbulence_replication/logs_mlpsuite/phase3-%j.out
#SBATCH -e <HOME>/gsoc2026-deeplense-qml/experiments/13_iqp_turbulence_replication/logs_mlpsuite/phase3-%j.err
#
# MLP suite, phase 3: checkpoint selection, the tables and the figures.  No training.
#
#   1. seeds/select_checkpoint.py on every 16,000-step sweep phase 2 wrote, choosing the step
#      count on the validation block (disjoint from the 200 held-out frames) exactly as the
#      classical baselines are selected.  Phase 2 already scored every snapshot under v11, so
#      nothing is re-scored here.
#   2. paper/ml4ps2026/make_seed_tables_f.py -> seed_tables_f.md
#   3. the three figures.  Each is allowed to fail without taking the table down with it: a
#      missing panel is recoverable, a missing table is the deliverable.
set -u
ts() { date "+%F %T"; }
REPO=<HOME>/gsoc2026-deeplense-qml/experiments/13_iqp_turbulence_replication
PAPER=<HOME>/gsoc2026-deeplense-qml/paper/ml4ps2026
cd "$REPO" || exit 1
PY=/usr/bin/python3
OUT=<WORK>/gsoc_iqp_mlpsuite/phase2
ES="$REPO/results_seeds_mlp_suite_es.jsonl"
mkdir -p logs_mlpsuite "$PAPER/figures"
echo "[$(ts)] START job=${SLURM_JOB_ID:-none} node=$(hostname)"
"$PY" -m py_compile "$PAPER/make_seed_tables_f.py" "$PAPER/figures/_mlpsuite_common.py" \
    "$PAPER/figures/fig2_mlp_sweep.py" "$PAPER/figures/figA_scaling_mlp.py" \
    "$PAPER/figures/fig_samples_mlp.py" seeds/select_checkpoint.py || exit 1
fail=0

echo "[$(ts)] STEP 1  checkpoint selection"
mapfile -t PREF < <("$PY" - "$OUT" <<'PYEOF'
import glob, os, re, sys
root = sys.argv[1]
pref = sorted({re.sub(r"_s\d+k$", "", d) for d in glob.glob(os.path.join(root, "s*", "t2es_*"))
               if os.path.isdir(d) and re.search(r"_s\d+k$", d)})
print("\n".join(pref))
PYEOF
)
echo "[$(ts)] ${#PREF[@]} snapshot sweeps found"
if [ "${#PREF[@]}" -gt 0 ]; then
  "$PY" -u seeds/select_checkpoint.py --prefix "${PREF[@]}" --results "$ES" \
      --label mlpsuite --fixed-reference-steps 4000 --n-holdout 200 \
      > logs_mlpsuite/select_f.log 2>&1 \
      || { echo "[WARN] select_checkpoint failed"; tail -40 logs_mlpsuite/select_f.log; fail=1; }
  grep -E "^\[select\] .* -> " logs_mlpsuite/select_f.log | tail -20
  echo "[$(ts)] selection rows: $(wc -l < "$ES" 2>/dev/null || echo 0)"
else
  echo "[WARN] no t2es_* snapshot sweeps found -- selection skipped"; fail=1
fi

echo "[$(ts)] STEP 2  seed_tables_f.md"
"$PY" -u "$PAPER/make_seed_tables_f.py" > logs_mlpsuite/table_f.log 2>&1 \
    || { echo "[FATAL] make_seed_tables_f failed"; tail -40 logs_mlpsuite/table_f.log; exit 1; }
tail -4 logs_mlpsuite/table_f.log

echo "[$(ts)] STEP 3  figures"
cd "$PAPER/figures" || exit 1
for f in fig2_mlp_sweep.py figA_scaling_mlp.py fig_samples_mlp.py; do
  "$PY" -u "$f" > "$REPO/logs_mlpsuite/${f%.py}.log" 2>&1 \
      && grep -E "^\[fig" "$REPO/logs_mlpsuite/${f%.py}.log" | tail -4 \
      || { echo "[WARN] $f failed"; tail -20 "$REPO/logs_mlpsuite/${f%.py}.log"; fail=1; }
done
echo "[$(ts)] DONE (fail=$fail)"
exit 0
