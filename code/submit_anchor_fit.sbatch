#!/bin/bash
#SBATCH -J iqp_anchor_fit
#SBATCH -A <YOUR_ACCOUNT>
#SBATCH -p 8gpus
#SBATCH --gres=gpu:8
#SBATCH -c 32
#SBATCH --mem=128G
#SBATCH -t 02:00:00
#SBATCH -o <REPO_ROOT>/code/logs_d2/sbatch-anchorfit-%j.out
#SBATCH -e <REPO_ROOT>/code/logs_d2/sbatch-anchorfit-%j.err
#
# Decisive anchor-fit experiment (see anchor_fit_experiment.py docstring): does the
# IQP model family + exact-MMD objective fit its own training anchor (t=1) once the
# kernel bandwidth/weighting is varied? 8 configs (sigma_mode x K x lr), 1e5 Adam steps
# each, one per GPU. Estimated ~11-15 ms/step on H100 (measured in submit_d2_main.sbatch)
# -> ~18-25 min/config; all 8 run concurrently on the 8 allocated GPUs, so the dominant
# cost is the single slowest config, well inside the 2h request (dev partition is full,
# per <user>).
#
# Also runs the D2 QCLI/CCI diagnostic (pre-registration item P5, missed by
# submit_d2_main.sbatch's main run) as a 9th, CPU-only, background job alongside the 8
# GPU configs -- see qcli_cci_d2.py docstring. Exact QCLI/CCI at n=18 measured
# ~30ms/~150ms per call, so this finishes in well under a minute; it just rides along
# instead of costing a separate allocation.
set -uo pipefail

REPO=<REPO_ROOT>/code
cd "$REPO"
mkdir -p logs_d2 outputs_anchor_fit

# ---------------------------------------------------------------- fail-fast pre-checks
module purge; module load cuda/13.0
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
PY=/usr/bin/python3
ts() { date +"%Y-%m-%d %H:%M:%S"; }

echo "[SLURM] job=$SLURM_JOB_ID node=$(hostname) part=$SLURM_JOB_PARTITION gpus=$($PY -c 'import torch;print(torch.cuda.device_count())')"

echo "[$(ts)] pre-check: py_compile"
for f in anchor_fit_experiment.py qcli_cci_d2.py train_iqp.py iqp_exact.py codec.py data_d2.py metrics.py diagnostics_qcli_cci.py; do
    "$PY" -m py_compile "$f" || { echo "[FATAL] py_compile failed for $f"; exit 1; }
done

echo "[$(ts)] pre-check: data_d2 importable + anchor t=1 loads"
"$PY" -c "
import data_d2
snap = data_d2.load_d2_snapshot(1, plane='mid')
print('[precheck] t=1 pool_codes=', snap.pool_codes.size, 'field shape=', snap.field.shape)
" || { echo "[FATAL] data_d2 pre-check failed"; exit 1; }
echo "[$(ts)] pre-checks passed"

# ---------------------------------------------------------------- launch: 8 GPU configs + 1 CPU diagnostic
CONFIGS=(auto3 sigma0.3 sigma0.15 flat flat_lr1e-3 flat_K7 sigma0.3_K7_lr1e-3 auto3_K7)

echo "[$(ts)] START qcli_cci_d2 (CPU, background)"
"$PY" -u qcli_cci_d2.py --plane mid --seed 0 --out "$REPO/outputs_anchor_fit/qcli_cci_d2.json" \
    > logs_d2/qcli_cci_d2.log 2>&1 &
qcli_pid=$!

pids=()
for i in "${!CONFIGS[@]}"; do
    cfg="${CONFIGS[$i]}"
    echo "[$(ts)] START anchor_fit config=$cfg (gpu $i)"
    (CUDA_VISIBLE_DEVICES=$i "$PY" -u anchor_fit_experiment.py --config "$cfg" --device cuda --seed 0 \
        > "logs_d2/anchor_fit_${cfg}.log" 2>&1) &
    pids+=($!)
done

fail=0
for j in "${!pids[@]}"; do
    wait "${pids[$j]}" || { echo "[WARN] config ${CONFIGS[$j]} failed, see logs_d2/anchor_fit_${CONFIGS[$j]}.log"; fail=1; }
done
wait "$qcli_pid" || { echo "[WARN] qcli_cci_d2 failed, see logs_d2/qcli_cci_d2.log"; fail=1; }

echo "[$(ts)] all anchor-fit configs + qcli_cci_d2 finished (fail=$fail)"

# ---------------------------------------------------------------- summarize
echo "[$(ts)] START summarize"
"$PY" -u anchor_fit_experiment.py --summarize > logs_d2/anchor_fit.log 2>&1
cat logs_d2/anchor_fit.log
echo "[$(ts)] DONE submit_anchor_fit.sbatch (fail=$fail)"
exit "$fail"
