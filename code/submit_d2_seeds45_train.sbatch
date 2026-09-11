#!/bin/bash
#SBATCH -J iqp_turb_seeds45_train
#SBATCH -A <YOUR_ACCOUNT>
#SBATCH -p dev
#SBATCH --gres=gpu:8
#SBATCH -c 32
#SBATCH --mem=128G
#SBATCH -t 03:00:00
#SBATCH -o <REPO_ROOT>/code/logs_d2/seeds45train-%j.out
#SBATCH -e <REPO_ROOT>/code/logs_d2/seeds45train-%j.err
#
# 5-seed completion for the ML4PS-2026 Appendix-B numbers, part 1 of 3: TRAINING.
#
# NEW FILE, NEW OUTPUT DIRS. Nothing under outputs_d2_main/, outputs_d2_last/,
# outputs_anchor_fit*/ is written, moved or deleted here, and the existing
# results_d2_{main,last,resynth,baseline_align}.jsonl are untouched -- this job writes only
# outputs_d2_seeds/, outputs_d2_seeds_last/ and results_d2_seeds_*.jsonl.
#
# Phases: 0 anchor-corr on the EXISTING seeds 0,1,2 (also the fail-fast validation of
# turb_seed_extras.py against the published 0.691/0.776); A train seeds 3,4; A2 evaluate under
# the pre-registered mean aggregation; A3 anchor-corr for seeds 3,4; A4 last-write-wins
# resynthesis + evaluate for seeds 3,4; D metric-calibration references over 5 shuffle seeds.
#
# Wall-clock basis, measured from job 342220's per-arm logs: train_iqp K=6 660 s, K=7 634 s,
# baseline_rbm 55 s, baseline_dcgan 98-160 s, controls.py marginal ~60 s. Packed 8-wide the
# 20 training runs are 3 batches whose slowest is an IQP run, ~25 min; resynthesize_last on 8
# arms (no IQP retraining, RBM PCD retrain) ~15 min; the CPU phases ~20 min. TOTAL ~60-70 min,
# so dev's 4 h cap has ample headroom and -t 03:00:00 is set well inside it.
set -uo pipefail

REPO=<REPO_ROOT>/code
cd "$REPO"
mkdir -p logs_d2

MAIN="$REPO/outputs_d2_main"
GT="$MAIN/ground_truth"
OUT="$REPO/outputs_d2_seeds"
OUT_LAST="$REPO/outputs_d2_seeds_last"
EXTRAS="$REPO/results_d2_seeds_extras.jsonl"
NEW_SEEDS="3 4"

ts() { date +"%Y-%m-%d %H:%M:%S"; }
module purge; module load cuda/13.0
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
PY=/usr/bin/python3

echo "[SLURM] job=$SLURM_JOB_ID node=$(hostname) part=$SLURM_JOB_PARTITION gpus=$($PY -c 'import torch;print(torch.cuda.device_count())')"
echo "[$(ts)] disk free on \$HOME: $(df -h "$REPO" | tail -1 | awk '{print $4}')"

echo "[$(ts)] pre-check: py_compile"
for f in turb_seed_extras.py turb_aggregate_seeds.py train_iqp.py baseline_rbm.py \
         baseline_dcgan.py controls.py evaluate.py resynthesize_last.py \
         anchor_fit_experiment.py metrics.py codec.py data_d2.py iqp_exact.py; do
    "$PY" -m py_compile "$f" || { echo "[FATAL] py_compile failed for $f"; exit 1; }
done
echo "[$(ts)] pre-check: existing artefacts this job depends on"
for d in "$GT" "$MAIN/iqp_K6_seed0" "$MAIN/iqp_K6_seed1" "$MAIN/iqp_K6_seed2" \
         "$MAIN/iqp_K7_seed0" "$MAIN/iqp_K7_seed1" "$MAIN/iqp_K7_seed2"; do
    [ -d "$d" ] || { echo "[FATAL] missing $d"; exit 1; }
done
"$PY" -c "
import data_d2
snap = data_d2.load_d2_snapshot(1, plane='mid')
print('[precheck] t=1 pool_codes=', snap.pool_codes.size, 'field shape=', snap.field.shape)
" || { echo "[FATAL] data_d2 pre-check failed"; exit 1; }
echo "[$(ts)] pre-checks passed"

rm -f "$EXTRAS"

# ---------------------------------------------------------------- PHASE 0
echo "[$(ts)] PHASE 0: anchor-corr for the existing seeds 0,1,2 (CPU)"
"$PY" -u turb_seed_extras.py --task anchor-corr --device cpu --results "$EXTRAS" \
    --run-dirs "$MAIN"/iqp_K6_seed0 "$MAIN"/iqp_K6_seed1 "$MAIN"/iqp_K6_seed2 \
               "$MAIN"/iqp_K7_seed0 "$MAIN"/iqp_K7_seed1 "$MAIN"/iqp_K7_seed2 \
    2>&1 | tee logs_d2/seeds45_anchor_corr_012.log
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "[FATAL] phase 0 anchor-corr failed"; exit 1; }

"$PY" - "$EXTRAS" <<'PYCHK' || { echo "[FATAL] phase 0 did not reproduce the published seed-0 anchor corr"; exit 1; }
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
by = {x["arm"]: x for x in rows if x.get("task") == "anchor_corr"}

# Provenance note (found 2026-09-09 while adding seeds 3,4):
# RESULTS.md 4(2) reports the 11-anchor mean anchor corr as "0.691 sampled / 0.776 exact"
# and attributes it to the K=6 main run.  Recomputation shows that pair is the K=7 arm's
# mean (0.6913 / 0.7755); the K=6 main run's 11-anchor mean is 0.6653 / 0.7538.  What IS
# the K=6 main run is the ANCHOR t=1 exact corr, 0.7738, independently reproduced by
# anchor_fit_protocol.py --summarize in job 342685 ("main run iqp_K6_seed0 (ref) 0.7738").
# So the fail-fast pins the two quantities that are actually verifiable against a second,
# independent implementation.
k7 = by["iqp_K7_seed0"]
k6 = by["iqp_K6_seed0"]
print("[precheck] iqp_K7_seed0 11-anchor mean: sampled=%.4f (RESULTS.md 0.691), exact=%.4f (RESULTS.md 0.776)"
      % (k7["anchor_corr_sampled_mean"], k7["anchor_corr_exact_mean"]))
print("[precheck] iqp_K6_seed0 anchor t=1 exact=%.4f (job 342685 independent recompute 0.7738)"
      % k6["anchor_corr_exact_t1"])
print("[precheck] iqp_K6_seed0 11-anchor mean: sampled=%.4f exact=%.4f (NOT the 0.691/0.776 pair)"
      % (k6["anchor_corr_sampled_mean"], k6["anchor_corr_exact_mean"]))
assert abs(k7["anchor_corr_sampled_mean"] - 0.691) < 0.005, k7["anchor_corr_sampled_mean"]
assert abs(k7["anchor_corr_exact_mean"] - 0.776) < 0.005, k7["anchor_corr_exact_mean"]
assert abs(k6["anchor_corr_exact_t1"] - 0.7738) < 0.005, k6["anchor_corr_exact_t1"]
print("[precheck] turb_seed_extras.py reproduces both independently verifiable seed-0 anchor-fit numbers -- OK")
PYCHK

# ---------------------------------------------------------------- 8-wide runner
GPU_SLOTS=8
run_batched() {
    local phase="$1"
    local i=0 slot fail_local=0
    local pids=() tags=()
    for job in "${JOBS[@]}"; do
        local tag="${job%%|*}"; local rest="${job#*|}"
        local device="${rest%%|*}"; local cmd="${rest#*|}"
        slot=$((i % GPU_SLOTS))
        echo "[$(ts)] [$phase] START $tag (slot $slot, device=$device)"
        if [ "$device" = "cuda" ]; then
            (CUDA_VISIBLE_DEVICES=$slot bash -c "$cmd" > "logs_d2/${tag}.log" 2>&1) &
        else
            (bash -c "$cmd" > "logs_d2/${tag}.log" 2>&1) &
        fi
        pids+=($!); tags[$i]="$tag"; i=$((i + 1))
        if [ $((i % GPU_SLOTS)) -eq 0 ]; then
            for j in "${!pids[@]}"; do
                wait "${pids[$j]}" || { echo "[WARN] ${tags[$((i - GPU_SLOTS + j))]} failed, see logs_d2/${tags[$((i - GPU_SLOTS + j))]}.log"; fail_local=1; }
            done
            pids=()
        fi
    done
    for j in "${!pids[@]}"; do
        local idx=$((i - ${#pids[@]} + j))
        wait "${pids[$j]}" || { echo "[WARN] ${tags[$idx]} failed, see logs_d2/${tags[$idx]}.log"; fail_local=1; }
    done
    echo "[$(ts)] [$phase] all ${#JOBS[@]} runs finished (fail=$fail_local)"
    return "$fail_local"
}

fail=0

# ---------------------------------------------------------------- PHASE A
rm -rf "$OUT"; mkdir -p "$OUT"
JOBS=()
for seed in $NEW_SEEDS; do
    for K in 6 7; do
        JOBS+=("iqp_K${K}_seed${seed}|cuda|$PY train_iqp.py --data d2 --out $OUT/iqp_K${K}_seed${seed} --device cuda --seed $seed --K $K --steps-core 30000 --steps-lat 3000 --n-holdout 200")
    done
done
for seed in $NEW_SEEDS; do
    JOBS+=("rbm_seed${seed}|cuda|$PY baseline_rbm.py --data d2 --out $OUT/rbm_seed${seed} --device cuda --seed $seed --n-holdout 200")
done
for seed in $NEW_SEEDS; do
    for tier in 51k 0.2m 1.2m; do
        for n_train in 11 100; do
            JOBS+=("dcgan_${tier}_n${n_train}_seed${seed}|cuda|$PY baseline_dcgan.py --data d2 --out $OUT/dcgan_${tier}_n${n_train}_seed${seed} --device cuda --seed $seed --tier $tier --n-train $n_train --steps 20000 --n-gen 200")
        done
    done
done
for seed in $NEW_SEEDS; do
    JOBS+=("ctrl_marginal_seed${seed}|cpu|$PY controls.py --arm marginal --data d2 --out $OUT/ctrl_marginal_seed${seed} --seed $seed --n-holdout 200")
done
echo "[$(ts)] PHASE A: ${#JOBS[@]} training runs for seeds $NEW_SEEDS"
run_batched A || fail=1

# ---------------------------------------------------------------- PHASE A2
echo "[$(ts)] PHASE A2: evaluate $OUT -> results_d2_seeds_main.jsonl"
"$PY" evaluate.py --outputs-dir "$OUT" --ground-truth "$GT" \
    --results "$REPO/results_d2_seeds_main.jsonl" > logs_d2/seeds45_evaluate_main.log 2>&1 \
    || { echo "[FATAL] evaluate (main) failed"; tail -30 logs_d2/seeds45_evaluate_main.log; exit 1; }
cat logs_d2/seeds45_evaluate_main.log

# ---------------------------------------------------------------- PHASE A3
echo "[$(ts)] PHASE A3: anchor-corr for seeds $NEW_SEEDS (CPU)"
"$PY" -u turb_seed_extras.py --task anchor-corr --device cpu --results "$EXTRAS" \
    --run-dirs "$OUT"/iqp_K6_seed3 "$OUT"/iqp_K6_seed4 "$OUT"/iqp_K7_seed3 "$OUT"/iqp_K7_seed4 \
    2>&1 | tee logs_d2/seeds45_anchor_corr_34.log
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "[WARN] phase A3 anchor-corr failed"; fail=1; }

# ---------------------------------------------------------------- PHASE A4
rm -rf "$OUT_LAST"; mkdir -p "$OUT_LAST"
echo "[$(ts)] PHASE A4: resynthesize_last (IQP x4 + CTRL-marginal x2), n_synth=100000/tau"
"$PY" resynthesize_last.py --arms-glob "$OUT/iqp_*,$OUT/ctrl_marginal_*" --out "$OUT_LAST" \
    --device cuda --n-synth 100000 > logs_d2/seeds45_resynth_last_iqp_ctrl.log 2>&1 \
    || { echo "[FATAL] resynthesize_last (iqp+ctrl) failed"; tail -60 logs_d2/seeds45_resynth_last_iqp_ctrl.log; exit 1; }
tail -20 logs_d2/seeds45_resynth_last_iqp_ctrl.log

echo "[$(ts)] PHASE A4: resynthesize_last (RBM x2, --rbm-retrain)"
"$PY" resynthesize_last.py --arms-glob "$OUT/rbm_*" --out "$OUT_LAST" \
    --device cuda --rbm-retrain > logs_d2/seeds45_resynth_last_rbm.log 2>&1 \
    || { echo "[FATAL] resynthesize_last (rbm) failed"; tail -60 logs_d2/seeds45_resynth_last_rbm.log; exit 1; }
tail -20 logs_d2/seeds45_resynth_last_rbm.log

echo "[$(ts)] PHASE A4: evaluate $OUT_LAST -> results_d2_seeds_last.jsonl"
"$PY" evaluate.py --outputs-dir "$OUT_LAST" --ground-truth "$GT" \
    --results "$REPO/results_d2_seeds_last.jsonl" > logs_d2/seeds45_evaluate_last.log 2>&1 \
    || { echo "[FATAL] evaluate (last) failed"; tail -30 logs_d2/seeds45_evaluate_last.log; exit 1; }
cat logs_d2/seeds45_evaluate_last.log

echo "[$(ts)] PHASE A4: dropping codes_t*.npy (evaluate only needs generated_t*.npy)"
find "$OUT_LAST" -name 'codes_t*.npy' -delete
echo "[$(ts)] disk free on \$HOME: $(df -h "$REPO" | tail -1 | awk '{print $4}')"

# ---------------------------------------------------------------- PHASE D
echo "[$(ts)] PHASE D: metric-calibration references (shuffle seeds 0..4, Conv2D encoder seed fixed at 0)"
"$PY" -u turb_seed_extras.py --task calib --ground-truth "$GT" --results "$EXTRAS" \
    --calib-seeds 0 1 2 3 4 2>&1 | tee logs_d2/seeds45_calib.log
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "[WARN] phase D calib failed"; fail=1; }

echo "[$(ts)] DONE submit_d2_seeds45_train.sbatch (fail=$fail)"
exit "$fail"
