#!/bin/bash
#SBATCH -J iqp_anchor_protocol
#SBATCH -A <YOUR_ACCOUNT>
#SBATCH -p 8gpus
#SBATCH --gres=gpu:4
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -o <REPO_ROOT>/code/logs_d2/sbatch-anchorprotocol-%j.out
#SBATCH -e <REPO_ROOT>/code/logs_d2/sbatch-anchorprotocol-%j.err
#
# Attribution protocol (see anchor_fit_protocol.py docstring): isolate whether the main
# run's paper-protocol anchor-fit gap (corr~0.69-0.77 at anchor t=1, outputs_d2_main/
# iqp_K6_seed0 -- see anchor_fit_protocol.py's --summarize, which recomputes this number
# directly from the saved theta rather than trusting a hardcoded literal) vs the
# anchor_fit_experiment.py "auto3" reference (all params trainable, full-pool exact pmf,
# corr~0.876, job 342542) is caused by (A) freezing theta_lat^(1) (Algorithm 1 stage 1)
# or (B) subsampling the anchor's pmf to 5e4 tuples (train_iqp.py's --n-tuples default),
# via 4 configs that each isolate (A), (B), both, or neither-plus-stage-2. All auto3
# kernel, K=6, Adam lr=1e-4, 30000 steps (+3000 stage-2 steps for one config). One
# config per GPU; ~30-40k total training steps/config at ~10ms/step (measured by the
# concurrent anchor_fit_experiment.py auto3 run) -> a few minutes/config, well inside
# the 1h request.
#
# Does NOT touch anchor_fit_experiment.py / train_iqp.py / iqp_exact.py / codec.py /
# data_d2.py -- anchor_fit_protocol.py only imports their functions. Safe to run
# alongside job 342542 (submit_anchor_fit.sbatch), which uses the same files read-only.
set -uo pipefail

REPO=<REPO_ROOT>/code
cd "$REPO"
mkdir -p logs_d2 outputs_anchor_fit_protocol

# ---------------------------------------------------------------- fail-fast pre-checks
module purge; module load cuda/13.0
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
PY=/usr/bin/python3
ts() { date +"%Y-%m-%d %H:%M:%S"; }

echo "[SLURM] job=$SLURM_JOB_ID node=$(hostname) part=$SLURM_JOB_PARTITION gpus=$($PY -c 'import torch;print(torch.cuda.device_count())')"

echo "[$(ts)] pre-check: py_compile"
for f in anchor_fit_protocol.py anchor_fit_experiment.py train_iqp.py iqp_exact.py codec.py data_d2.py metrics.py; do
    "$PY" -m py_compile "$f" || { echo "[FATAL] py_compile failed for $f"; exit 1; }
done

echo "[$(ts)] pre-check: data_d2 importable + anchor t=1 loads"
"$PY" -c "
import data_d2
snap = data_d2.load_d2_snapshot(1, plane='mid')
print('[precheck] t=1 pool_codes=', snap.pool_codes.size, 'field shape=', snap.field.shape)
" || { echo "[FATAL] data_d2 pre-check failed"; exit 1; }

echo "[$(ts)] pre-check: main-run artefacts for the --summarize reference exist"
"$PY" -c "
import os
d = '$REPO/outputs_d2_main/iqp_K6_seed0'
for f in ('theta_core.npy', 'theta_lat_anchors.npy'):
    p = os.path.join(d, f)
    assert os.path.exists(p), 'missing ' + p
print('[precheck] main-run reference artefacts OK')
" || { echo "[FATAL] main-run reference pre-check failed"; exit 1; }
echo "[$(ts)] pre-checks passed"

# ---------------------------------------------------------------- launch: 4 configs, 1 GPU each
CONFIGS=(paper_stage1 paper_stage1_plus_stage2 allparams_subsample5e4 paper_stage1_subsample5e4)

pids=()
for i in "${!CONFIGS[@]}"; do
    cfg="${CONFIGS[$i]}"
    echo "[$(ts)] START anchor_fit_protocol config=$cfg (gpu $i)"
    (CUDA_VISIBLE_DEVICES=$i "$PY" -u anchor_fit_protocol.py --config "$cfg" --device cuda --seed 0 \
        > "logs_d2/anchor_fit_protocol_${cfg}.log" 2>&1) &
    pids+=($!)
done

fail=0
for j in "${!pids[@]}"; do
    wait "${pids[$j]}" || { echo "[WARN] config ${CONFIGS[$j]} failed, see logs_d2/anchor_fit_protocol_${CONFIGS[$j]}.log"; fail=1; }
done
echo "[$(ts)] all anchor_fit_protocol configs finished (fail=$fail)"

# ---------------------------------------------------------------- summarize
echo "[$(ts)] START summarize"
"$PY" -u anchor_fit_protocol.py --summarize > logs_d2/anchor_fit_protocol.log 2>&1
cat logs_d2/anchor_fit_protocol.log
echo "[$(ts)] DONE submit_anchor_fit_protocol.sbatch (fail=$fail)"
exit "$fail"
