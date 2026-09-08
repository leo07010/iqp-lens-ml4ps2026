#!/bin/bash
#SBATCH -J iqp_turb_d2_main
#SBATCH -A <YOUR_ACCOUNT>
#SBATCH -p 8gpus
#SBATCH --gres=gpu:8
#SBATCH -c 32
#SBATCH --mem=128G
#SBATCH -t 06:00:00
#SBATCH -o <REPO_ROOT>/code/logs_d2/sbatch-%j.out
#SBATCH -e <REPO_ROOT>/code/logs_d2/sbatch-%j.err
#
# Main D2 replication run: 3 seeds (0,1,2) for every primary/paper-baseline arm, the
# single-seed sensitivity arms (IQP-31k only), and the single-seed DCGAN data-scaling
# study, all on real D2 data (data_d2.py -- NOT written yet as of this draft; the
# fail-fast check below refuses to run until it exists).
#
# Job count and time estimate (see submit_synthetic_full.sbatch for the per-arm
# wall-clock basis -- D2 field VALUES differ from synthetic but shapes/step-counts are
# identical, so those per-arm estimates carry over; D2's native-grid code pool is larger
# than synthetic's 4096-cell pool, expected to add at most a few % to data-prep time):
#   IQP-31k x3 seeds + IQP-63k x3 seeds + 6 sensitivity arms (IQP-31k only)  = 12 jobs x ~30 min
#   RBM-12k x3 seeds                                                        =  3 jobs x ~20 min
#   DCGAN 3 tiers x {11,100} x3 seeds                                       = 18 jobs x ~8 min
#   DCGAN scaling: 3 tiers x {10,30,50,70,90,100} x1 seed                   = 18 jobs x ~8 min
#   CTRL-marginal x3 seeds, CTRL-interp x1                                  =  4 jobs x ~1 min
#   TOTAL 55 jobs. Serial-on-1-GPU this is ~11-12 h (would NOT fit dev's 4h, and is a
#   poor use of the 2-day 8gpus budget too); packed 8-wide across this partition's 8
#   GPUs (~7 batches of <=8 jobs, batch time = its slowest job) gives ~2-2.5 h. Requesting
#   6 h leaves ample headroom for D2 being slower than synthetic in ways not yet known
#   (dataset not inspected -- pre-registration Addendum) without touching the 2-day cap.
# IQP jobs run on GPU (device=cuda, measured on H100: ~11 ms/step vs ~23 ms/step on 4
# CPU threads -- 2026-09-05 update, also keeps all 8 GPUs busy instead of leaving them
# idle while only RBM/DCGAN use them); RBM/DCGAN also run on GPU (device=cuda). All GPU
# jobs go via CUDA_VISIBLE_DEVICES round-robin across the 8 allocated GPUs, 8-way
# concurrent, `wait` between batches.
set -uo pipefail

REPO=<REPO_ROOT>/code
cd "$REPO"
mkdir -p logs_d2

# ---------------------------------------------------------------- fail-fast: data_d2.py must exist
/usr/bin/python3 -c "import data_d2" || {
    echo "[FATAL] data_d2.py is not importable -- dataset not ready / not written yet. Aborting before using any 8gpus budget."
    exit 1
}

module purge; module load cuda/13.0
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
PY=/usr/bin/python3
OUT="$REPO/outputs_d2_main"
ts() { date +"%Y-%m-%d %H:%M:%S"; }

echo "[SLURM] job=$SLURM_JOB_ID node=$(hostname) part=$SLURM_JOB_PARTITION gpus=$($PY -c 'import torch;print(torch.cuda.device_count())')"

# ---------------------------------------------------------------- fail-fast pre-checks
echo "[$(ts)] pre-check: py_compile all .py"
for f in *.py; do
    "$PY" -m py_compile "$f" || { echo "[FATAL] py_compile failed for $f"; exit 1; }
done

echo "[$(ts)] pre-check: unit tests"
for t in test_iqp_exact.py test_baselines.py test_controls.py test_diagnostics.py test_data_d2.py test_d2_arms_smoke.py; do
    "$PY" "$t" > "logs_d2/precheck_${t%.py}.log" 2>&1 \
        || { echo "[FATAL] $t failed -- see logs_d2/precheck_${t%.py}.log"; tail -20 "logs_d2/precheck_${t%.py}.log"; exit 1; }
done
echo "[$(ts)] pre-checks passed"

rm -rf "$OUT"
mkdir -p "$OUT"

# ---------------------------------------------------------------- ground truth (200 held-out tau, d2's own fixed split)
echo "[$(ts)] START make_ground_truth (d2)"
"$PY" make_ground_truth.py --data d2 --out "$OUT/ground_truth" --n-holdout 200 \
    > logs_d2/ground_truth.log 2>&1 || { echo "[FATAL] make_ground_truth failed"; tail -30 logs_d2/ground_truth.log; exit 1; }
echo "[$(ts)] OK    make_ground_truth"

# ---------------------------------------------------------------- build the job list: TAG|DEVICE|CMD
JOBS=()
add_job() { JOBS+=("$1|$2|$3"); }

for seed in 0 1 2; do
    for K in 6 7; do
        tag="iqp_K${K}_seed${seed}"
        add_job "$tag" cuda \
            "$PY train_iqp.py --data d2 --out $OUT/$tag --device cuda --seed $seed --K $K --steps-core 30000 --steps-lat 3000 --n-holdout 200"
    done
done
# Sensitivity arms (single seed=0, IQP-31k/K=6 only) -- pre-registration "Sensitivity arms"
add_job "iqp_sens_random_core_init" cuda \
    "$PY train_iqp.py --data d2 --out $OUT/iqp_sens_random_core_init --device cuda --seed 0 --K 6 --steps-core 30000 --steps-lat 3000 --n-holdout 200 --init random"
add_job "iqp_sens_lat_init_data" cuda \
    "$PY train_iqp.py --data d2 --out $OUT/iqp_sens_lat_init_data --device cuda --seed 0 --K 6 --steps-core 30000 --steps-lat 3000 --n-holdout 200 --lat-init data"
add_job "iqp_sens_lr1e-3" cuda \
    "$PY train_iqp.py --data d2 --out $OUT/iqp_sens_lr1e-3 --device cuda --seed 0 --K 6 --steps-core 30000 --steps-lat 3000 --n-holdout 200 --lr 1e-3"
add_job "iqp_sens_sigma_ref38_16q" cuda \
    "$PY train_iqp.py --data d2 --out $OUT/iqp_sens_sigma_ref38_16q --device cuda --seed 0 --K 6 --steps-core 30000 --steps-lat 3000 --n-holdout 200 --sigmas 0.6,0.3"
add_job "iqp_sens_sigma_meanweight2_6" cuda \
    "$PY train_iqp.py --data d2 --out $OUT/iqp_sens_sigma_meanweight2_6 --device cuda --seed 0 --K 6 --steps-core 30000 --steps-lat 3000 --n-holdout 200 --sigmas 1.4105,0.6746"
add_job "iqp_sens_full_grid" cuda \
    "$PY train_iqp.py --data d2 --out $OUT/iqp_sens_full_grid --device cuda --seed 0 --K 6 --steps-core 30000 --steps-lat 3000 --n-holdout 200 --full-grid"

for seed in 0 1 2; do
    add_job "rbm_seed${seed}" cuda \
        "$PY baseline_rbm.py --data d2 --out $OUT/rbm_seed${seed} --device cuda --seed $seed --n-holdout 200"
done

for seed in 0 1 2; do
    for tier in 51k 0.2m 1.2m; do
        for n_train in 11 100; do
            tag="dcgan_${tier}_n${n_train}_seed${seed}"
            add_job "$tag" cuda \
                "$PY baseline_dcgan.py --data d2 --out $OUT/$tag --device cuda --seed $seed --tier $tier --n-train $n_train --steps 20000 --n-gen 200"
        done
    done
done

# DCGAN data-scaling study (paper Fig.): 3 tiers x N in {10,30,50,70,90,100}, seed 0 only
for tier in 51k 0.2m 1.2m; do
    for n_train in 10 30 50 70 90 100; do
        tag="dcgan_scaling_${tier}_n${n_train}"
        add_job "$tag" cuda \
            "$PY baseline_dcgan.py --data d2 --out $OUT/$tag --device cuda --seed 0 --tier $tier --n-train $n_train --steps 20000 --n-gen 200"
    done
done

for seed in 0 1 2; do
    add_job "ctrl_marginal_seed${seed}" cpu \
        "$PY controls.py --arm marginal --data d2 --out $OUT/ctrl_marginal_seed${seed} --seed $seed --n-holdout 200"
done
add_job "ctrl_interp" cpu \
    "$PY controls.py --arm interp --data d2 --out $OUT/ctrl_interp --seed 0 --n-holdout 200"

echo "[$(ts)] ${#JOBS[@]} jobs queued, running 8-wide across the allocated GPUs"

# ---------------------------------------------------------------- run 8-wide, wait between batches
GPU_SLOTS=8
i=0
pids=()
fail=0
for job in "${JOBS[@]}"; do
    tag="${job%%|*}"
    rest="${job#*|}"
    device="${rest%%|*}"
    cmd="${rest#*|}"
    slot=$((i % GPU_SLOTS))
    echo "[$(ts)] START $tag (slot $slot, device=$device)"
    if [ "$device" = "cuda" ]; then
        (CUDA_VISIBLE_DEVICES=$slot bash -c "$cmd" > "logs_d2/${tag}.log" 2>&1) &
    else
        (bash -c "$cmd" > "logs_d2/${tag}.log" 2>&1) &
    fi
    pids+=($!)
    tags[$i]="$tag"
    i=$((i + 1))
    if [ $((i % GPU_SLOTS)) -eq 0 ]; then
        for j in "${!pids[@]}"; do
            wait "${pids[$j]}" || { echo "[WARN] job ${tags[$((i - GPU_SLOTS + j))]} failed, see logs_d2/${tags[$((i - GPU_SLOTS + j))]}.log"; fail=1; }
        done
        pids=()
    fi
done
for j in "${!pids[@]}"; do
    idx=$((i - ${#pids[@]} + j))
    wait "${pids[$j]}" || { echo "[WARN] job ${tags[$idx]} failed, see logs_d2/${tags[$idx]}.log"; fail=1; }
done

echo "[$(ts)] all ${#JOBS[@]} jobs finished (fail=$fail)"

# ---------------------------------------------------------------- evaluate
echo "[$(ts)] START evaluate"
"$PY" evaluate.py --outputs-dir "$OUT" --ground-truth "$OUT/ground_truth" \
    --results "$REPO/results_d2_main.jsonl" \
    > logs_d2/evaluate.log 2>&1 || { echo "[FATAL] evaluate failed"; tail -30 logs_d2/evaluate.log; exit 1; }
cat logs_d2/evaluate.log
echo "[$(ts)] DONE submit_d2_main.sbatch (fail=$fail)"
exit "$fail"
