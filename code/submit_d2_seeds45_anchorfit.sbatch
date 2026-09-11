#!/bin/bash
#SBATCH -J iqp_turb_seeds45_af
#SBATCH -A <YOUR_ACCOUNT>
#SBATCH -p dev
#SBATCH --gres=gpu:8
#SBATCH -c 32
#SBATCH --mem=128G
#SBATCH -t 03:30:00
#SBATCH -o <REPO_ROOT>/code/logs_d2/seeds45af-%j.out
#SBATCH -e <REPO_ROOT>/code/logs_d2/seeds45af-%j.err
#
# 5-seed completion for the ML4PS-2026 Appendix-B numbers, part 2 of 3: the ANCHOR-FIT
# sweeps that produce the 0.88-0.89 model-family ceiling and the -0.08 / -0.04 protocol
# attribution.  Both currently exist for seed 0 only; this job adds seeds 1, 2, 3, 4.
#
# NEW OUTPUT DIRS ONLY: outputs_anchor_fit_s{1,2,3,4}/ and
# outputs_anchor_fit_protocol_s{1,2,3,4}/.  The seed-0 dirs outputs_anchor_fit/ and
# outputs_anchor_fit_protocol/ (jobs 342542 / 342685) are read by the aggregator but never
# written here.  Independent of the training job: neither sweep reads outputs_d2_seeds/.
#
# Wall-clock basis, measured: anchor_fit_experiment 1048 s/config at 1e5 steps (job 342542
# ran all 8 configs 8-wide in 19 min); anchor_fit_protocol 319 s/config at 3e4 steps (job
# 342685 ran all 4 4-wide in 6 min).  Packed 8-wide: 32 experiment runs = 4 batches x ~18
# min = ~72 min, 16 protocol runs = 2 batches x ~6 min = ~12 min.  TOTAL ~85-95 min, inside
# dev's 4 h cap with headroom; -t 03:30:00.
set -uo pipefail

REPO=<REPO_ROOT>/code
cd "$REPO"
mkdir -p logs_d2
AF_SEEDS="1 2 3 4"

ts() { date +"%Y-%m-%d %H:%M:%S"; }
module purge; module load cuda/13.0
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
PY=/usr/bin/python3

echo "[SLURM] job=$SLURM_JOB_ID node=$(hostname) part=$SLURM_JOB_PARTITION gpus=$($PY -c 'import torch;print(torch.cuda.device_count())')"

echo "[$(ts)] pre-check: py_compile"
for f in anchor_fit_experiment.py anchor_fit_protocol.py train_iqp.py iqp_exact.py codec.py \
         data_d2.py metrics.py; do
    "$PY" -m py_compile "$f" || { echo "[FATAL] py_compile failed for $f"; exit 1; }
done
"$PY" -c "
import data_d2
snap = data_d2.load_d2_snapshot(1, plane='mid')
print('[precheck] t=1 pool_codes=', snap.pool_codes.size, 'field shape=', snap.field.shape)
" || { echo "[FATAL] data_d2 pre-check failed"; exit 1; }
echo "[$(ts)] pre-check: seed-0 reference sweeps present (read-only)"
for d in outputs_anchor_fit/auto3 outputs_anchor_fit_protocol/paper_stage1; do
    [ -f "$REPO/$d/meta.json" ] || { echo "[FATAL] missing $REPO/$d/meta.json"; exit 1; }
done
echo "[$(ts)] pre-checks passed"

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

# ---------------------------------------------------------------- PHASE B: model-family ceiling sweep
AF_CONFIGS=(auto3 sigma0.3 sigma0.15 flat flat_lr1e-3 flat_K7 sigma0.3_K7_lr1e-3 auto3_K7)
JOBS=()
for seed in $AF_SEEDS; do
    for cfg in "${AF_CONFIGS[@]}"; do
        JOBS+=("anchor_fit_s${seed}_${cfg}|cuda|$PY -u anchor_fit_experiment.py --config $cfg --device cuda --seed $seed --out-root $REPO/outputs_anchor_fit_s${seed}")
    done
done
echo "[$(ts)] PHASE B: ${#JOBS[@]} anchor_fit_experiment runs (8 configs x seeds $AF_SEEDS)"
run_batched B || fail=1

# ---------------------------------------------------------------- PHASE C: protocol attribution
AFP_CONFIGS=(paper_stage1 paper_stage1_plus_stage2 allparams_subsample5e4 paper_stage1_subsample5e4)
JOBS=()
for seed in $AF_SEEDS; do
    for cfg in "${AFP_CONFIGS[@]}"; do
        JOBS+=("anchor_fit_protocol_s${seed}_${cfg}|cuda|$PY -u anchor_fit_protocol.py --config $cfg --device cuda --seed $seed --out-root $REPO/outputs_anchor_fit_protocol_s${seed}")
    done
done
echo "[$(ts)] PHASE C: ${#JOBS[@]} anchor_fit_protocol runs (4 configs x seeds $AF_SEEDS)"
run_batched C || fail=1

echo "[$(ts)] summary of what landed:"
for seed in 0 $AF_SEEDS; do
    r1="$REPO/outputs_anchor_fit"; r2="$REPO/outputs_anchor_fit_protocol"
    [ "$seed" != "0" ] && { r1="${r1}_s${seed}"; r2="${r2}_s${seed}"; }
    echo "  seed $seed: $(ls -d $r1/*/ 2>/dev/null | wc -l) experiment configs, $(ls -d $r2/*/ 2>/dev/null | wc -l) protocol configs"
done

echo "[$(ts)] DONE submit_d2_seeds45_anchorfit.sbatch (fail=$fail)"
exit "$fail"
