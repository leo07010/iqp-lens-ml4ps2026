#!/bin/bash
#SBATCH -J iqp_f4seed
#SBATCH -A <YOUR_ACCOUNT>
#SBATCH -p dev
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:40:00
#SBATCH -o <REPO_ROOT>/code/logs_controlled/f4seed-%j.out
#SBATCH -e <REPO_ROOT>/code/logs_controlled/f4seed-%j.err
#
# ===========================================================================================
# App. D (reference paper Fig. 4), seeds 3 and 4.
#
# results_fig4.jsonl holds 3 random initialisations per system size (--seeds 3, master
# --seed 20260907).  This job adds two more with a fresh master seed, written to a separate
# file so the published one is never touched; agg_fig34_seeds.py relabels their seed_idx
# 0,1 -> 3,4 and concatenates.
#
# A "seed" here is one SPSA trajectory: its own random <=2-local gate mask set, its own
# N(0, 0.05) initial angles, and its own SPSA perturbation stream, run for both gain arms
# (a = 0.5 and a = 2.0) at n = 8, 12, 16.  Everything else is identical to the published run.
#
# Cost, measured: the published 3-seed run was 127 s wall on GPU (meta.wall_seconds), so two
# more seeds are ~85 s.  40 min requested = ~25x headroom (the pre-checks dominate).
# ===========================================================================================
set -uo pipefail

REPL=<REPO_ROOT>/code
DIR="$REPL/paper_suite/controlled"
OUT="$REPL/outputs_controlled"
LOGS="$REPL/logs_controlled"
cd "$DIR"
mkdir -p "$OUT" "$LOGS"

module purge; module load cuda/13.0
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
PY=/usr/bin/python3
F4="$OUT/results_fig4_seeds34.jsonl"
ts() { date +"%Y-%m-%d %H:%M:%S"; }

echo "[SLURM] $(ts) job=${SLURM_JOB_ID:-none} node=$(hostname) part=${SLURM_JOB_PARTITION:-none}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

"$PY" - <<'PYEOF' || { echo "[FATAL] python environment broken"; exit 1; }
import sys
import numpy, torch, scipy, matplotlib
print("[env] python %s | numpy %s | torch %s" % (sys.version.split()[0], numpy.__version__, torch.__version__))
assert torch.cuda.is_available(), "no CUDA device visible"
print("[env] cuda device: %s" % torch.cuda.get_device_name(0))
PYEOF

echo "[pre] $(ts) py_compile"
"$PY" -m py_compile common.py fig4_qcli_max.py || { echo "[FATAL] py_compile failed"; exit 1; }

echo "[pre] $(ts) smoke (n=8 only, 20 steps, throwaway)"
"$PY" fig4_qcli_max.py --sizes 8 --seeds 2 --steps 20 --record-every 10 --spsa-a 0.5 \
    --device cuda --seed 20260908 --progress-every 0 --out /tmp/f4_smoke_$$.jsonl > /dev/null 2>&1
if [ $? -ne 0 ]; then echo "[FATAL] smoke failed"; "$PY" fig4_qcli_max.py --sizes 8 --seeds 2 --steps 20 \
    --record-every 10 --spsa-a 0.5 --device cuda --seed 20260908 --out /tmp/f4_smoke_$$.jsonl 2>&1 | tail -20; exit 1; fi
rm -f /tmp/f4_smoke_$$.jsonl
echo "[pre] smoke OK"

echo "[fig4] $(ts) start -> $F4"
"$PY" fig4_qcli_max.py \
    --sizes 8,12,16 --gates 150 --locality 2 \
    --seeds 2 --steps 3333 --record-every 1 --spsa-a 0.5,2.0 --spsa-c 0.2 \
    --init-std 0.05 --dtype float64 --device cuda --seed 20260908 \
    --progress-every 1000 --out "$F4"
RC=$?
echo "[fig4] $(ts) exit=$RC rows=$(wc -l < "$F4" 2>/dev/null || echo 0)"
ls -la "$F4" 2>&1
exit "$RC"
