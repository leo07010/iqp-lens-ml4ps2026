#!/usr/bin/env bash
# CPU, end-to-end smoke test for the arXiv 2603.06440 replication (pre-registration
# "Tests" section: "A synthetic-data end-to-end smoke (all arms, tiny steps, CPU) must
# produce files satisfying INTERFACES.md"). Runs every arm with a tiny anchor/step
# budget so the whole pipeline finishes in well under a minute on a login node's CPU.
# This is ONLY meant to prove the plumbing end-to-end (same ground truth, same held-out
# tau, correct output contract, evaluate.py produces Table 2) -- the printed numbers
# are meaningless (few training steps, few held-out points).
#
# DATA=synthetic (default) or DATA=d2 selects --data for every arm below. DATA=d2 needs
# the real cache from preprocess_d2.py (INTERFACES.md / data_d2.py CACHE_DIR_DEFAULT) --
# still a tiny anchors-max/n-holdout smoke, NOT a real training run, but exercises the
# actual d2_cache-backed a_z/b_z (data_d2.get_z_range) and holdout_taus
# (data_d2.get_holdout_taus) instead of synth_data's. DATA=d2 writes to outputs_d2/ and
# results_smoke_d2.jsonl so it never clobbers a synthetic run's outputs.
#
# Repo rule (INTERFACES.md): login node = seconds-to-minutes CPU only. Do NOT sbatch
# this, and do NOT raise the step counts below to anything resembling a real run.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PY=/usr/bin/python3
DATA="${DATA:-synthetic}"
if [ "$DATA" = "synthetic" ]; then
    OUT="$HERE/outputs"
    RESULTS="$HERE/results_smoke.jsonl"
else
    OUT="$HERE/outputs_${DATA}"
    RESULTS="$HERE/results_smoke_${DATA}.jsonl"
fi
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

ANCHORS_MAX=3
N_HOLDOUT=4

ts() { date +"%Y-%m-%d %H:%M:%S"; }
fail=0
run_step() {
    local label="$1"; shift
    echo "[$(ts)] START  $label"
    if "$@"; then
        echo "[$(ts)] OK     $label"
    else
        echo "[$(ts)] FAILED $label (exit $?)"
        fail=1
    fi
}

echo "=== run_smoke_all.sh: CPU --data=$DATA smoke, all arms ==="
echo "anchors-max=$ANCHORS_MAX n-holdout=$N_HOLDOUT out=$OUT results=$RESULTS"

rm -rf "$OUT"
mkdir -p "$OUT"

run_step "make_ground_truth" \
    "$PY" make_ground_truth.py --data "$DATA" --out "$OUT/ground_truth" \
    --anchors-max "$ANCHORS_MAX" --n-holdout "$N_HOLDOUT"

run_step "IQP K=6 (steps 50/10)" \
    "$PY" train_iqp.py --data "$DATA" --out "$OUT/iqp" --device cpu --seed 0 \
    --K 6 --steps-core 50 --steps-lat 10 --n-tuples 2000 --n-synth 2000 \
    --anchors-max "$ANCHORS_MAX" --n-holdout "$N_HOLDOUT"

run_step "RBM (steps 30/5)" \
    "$PY" baseline_rbm.py --data "$DATA" --out "$OUT/rbm" --device cpu --seed 0 \
    --n-tuples 2000 --steps-core 30 --steps-lat 5 --batch 64 --n-chains-pcd 20 \
    --gen-chains 50 --gen-sweeps 10 --gen-keep 3 \
    --anchors-max "$ANCHORS_MAX" --n-holdout "$N_HOLDOUT"

run_step "DCGAN 51k (20 steps, n-train 3, n-gen 8)" \
    "$PY" baseline_dcgan.py --data "$DATA" --out "$OUT/dcgan" --device cpu --seed 0 \
    --tier 51k --n-train 3 --steps 20 --n-gen 8

run_step "CTRL-marginal" \
    "$PY" controls.py --arm marginal --data "$DATA" --out "$OUT/marginal" --seed 0 \
    --anchors-max "$ANCHORS_MAX" --n-holdout "$N_HOLDOUT" --n-synth 1000 --n-tuples 1000

run_step "CTRL-interp" \
    "$PY" controls.py --arm interp --data "$DATA" --out "$OUT/interp" --seed 0 \
    --anchors-max "$ANCHORS_MAX" --n-holdout "$N_HOLDOUT"

if [ "$fail" -ne 0 ]; then
    echo "[$(ts)] one or more arms FAILED -- see above; running evaluate.py on whatever completed"
fi

run_step "evaluate (Table 2)" \
    "$PY" evaluate.py --outputs-dir "$OUT" --ground-truth "$OUT/ground_truth" \
    --results "$RESULTS"

echo "=== run_smoke_all.sh done (fail=$fail) ==="
exit "$fail"
