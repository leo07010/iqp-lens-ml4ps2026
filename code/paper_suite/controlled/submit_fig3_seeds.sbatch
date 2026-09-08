#!/bin/bash
#SBATCH -J iqp_f3seed
#SBATCH -A <YOUR_ACCOUNT>
#SBATCH -p dev
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o <HOME>/gsoc2026-deeplense-qml/experiments/13_iqp_turbulence_replication/logs_controlled/f3seed-%j.out
#SBATCH -e <HOME>/gsoc2026-deeplense-qml/experiments/13_iqp_turbulence_replication/logs_controlled/f3seed-%j.err
#
# ===========================================================================================
# App. C (reference paper Fig. 3) over learner seeds, targets held fixed.
#
# Submit with the seeds to run, e.g.
#     sbatch --export=ALL,LS=0:1 submit_fig3_seeds.sbatch
#     sbatch --export=ALL,LS=2:3:4 submit_fig3_seeds.sbatch
# (colon-separated: sbatch splits its own --export argument on commas)
# Learner seed 0 re-derives results_fig3_v2.jsonl and is the correctness check for the whole
# seed table (agg_fig34_seeds.py verifies it row by row); 1-4 are new.
#
# --seed stays at the published 20260907, which is what keeps the 140 targets, their 10^4-
# sample datasets, their kernel bandwidths and the three learner architectures bit-identical
# to the published run.  Only np.random.default_rng(seed + 991 + G_n + 1e6*learner_seed),
# the learner's initial angles, moves.
#
# Cost, measured: the published single-seed run was 897 s wall (outputs_controlled/fig3_v2.done),
# of which ~875 s are the fits (3 learners x 2 arms x 2 chunks of 70 x 5000 Adam steps at
# 14.6 ms/step).  Targets are built once and shared, so k learner seeds cost
# ~2 min + k x 15 min.  With LS=2,3,4 that is ~47 min; 2 h requested = ~2.5x headroom.
# ===========================================================================================
set -uo pipefail

REPL=<HOME>/gsoc2026-deeplense-qml/experiments/13_iqp_turbulence_replication
DIR="$REPL/paper_suite/controlled"
OUT="$REPL/outputs_controlled"
LOGS="$REPL/logs_controlled"
cd "$DIR"
mkdir -p "$OUT" "$LOGS"

# LS is COLON-separated on the command line: sbatch --export splits its own argument on
# commas, so "LS=0,1" would export LS=0 and a stray "1".
LS=$(echo "${LS:-1:2:3:4}" | tr ':' ',')
TAG=$(echo "$LS" | tr ',' '-')

module purge; module load cuda/13.0
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
PY=/usr/bin/python3
DONE="$OUT/fig3_ls${TAG}.done"
ts() { date +"%Y-%m-%d %H:%M:%S"; }

rm -f "$DONE"
echo "[SLURM] $(ts) job=${SLURM_JOB_ID:-none} node=$(hostname) part=${SLURM_JOB_PARTITION:-none} LS=$LS"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

"$PY" - <<'PYEOF' || { echo "[FATAL] python environment broken. Aborting before using dev budget."; exit 1; }
import sys
import numpy, torch, scipy, matplotlib
print("[env] python %s | numpy %s | torch %s | scipy %s | matplotlib %s"
      % (sys.version.split()[0], numpy.__version__, torch.__version__,
         scipy.__version__, matplotlib.__version__))
assert torch.cuda.is_available(), "no CUDA device visible"
print("[env] cuda device: %s" % torch.cuda.get_device_name(0))
PYEOF

echo "[pre] $(ts) py_compile"
"$PY" -m py_compile common.py fig3_support_mismatch.py fig3_learner_seeds.py fig4_qcli_max.py \
    plot_fig34.py test_controlled.py || { echo "[FATAL] py_compile failed"; exit 1; }

echo "[pre] $(ts) test_controlled.py"
"$PY" test_controlled.py 2>&1 | tail -20
if [ "${PIPESTATUS[0]}" -ne 0 ]; then echo "[FATAL] test_controlled.py failed"; exit 1; fi

echo "[pre] $(ts) end-to-end smoke of fig3_learner_seeds.py (tiny, throwaway)"
SMOKE=$(mktemp -d)
"$PY" fig3_learner_seeds.py --gates-target 140 --learner-gates 50 --n-targets 3 \
    --n-samples 500 --target-steps 20 --target-snapshots 20 --steps 30 \
    --train-batch 3 --null-reps 1 --learner-seeds 0,1 --device cuda \
    --out-dir "$SMOKE" --progress-every 0 > "$LOGS/f3seed-smoke-${SLURM_JOB_ID:-0}.log" 2>&1
if [ $? -ne 0 ]; then echo "[FATAL] smoke failed:"; tail -40 "$LOGS/f3seed-smoke-${SLURM_JOB_ID:-0}.log"; rm -rf "$SMOKE"; exit 1; fi
echo "[pre] smoke OK ($(wc -l < "$SMOKE/results_fig3_v2_ls0.jsonl") + $(wc -l < "$SMOKE/results_fig3_v2_ls1.jsonl") rows)"
rm -rf "$SMOKE"

echo "[fig3ls] $(ts) start, learner seeds $LS"
"$PY" fig3_learner_seeds.py \
    --n 16 --gates-target 140,280,420,560,700,840,1050 --learner-gates 50,100,150 \
    --locality-target 4 --locality-learner 2 \
    --n-targets 20 --n-samples 10000 \
    --target-optimizer adam --target-lr 0.05 --target-steps 400 --target-snapshots 400 \
    --target-qcli-max 0.7 --target-init-std 0.05 \
    --steps 5000 --lr 1e-4 --learner-init-std 0.01 --train-batch 70 \
    --exact-target-control --null-reps 5 \
    --dtype float64 --device cuda --seed 20260907 \
    --bin-width-report 0.09 --progress-every 2000 \
    --learner-seeds "$LS" --out-dir "$OUT" --done-marker "$DONE"
RC=$?
echo "[fig3ls] $(ts) exit=$RC"
echo "--- done marker ---"; cat "$DONE" 2>&1
ls -la "$OUT"/results_fig3_v2_ls*.jsonl 2>&1
exit "$RC"
