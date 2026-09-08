#!/bin/bash
#SBATCH -J iqp_mechseed
#SBATCH -A <YOUR_ACCOUNT>
#SBATCH -p dev
#SBATCH --gres=gpu:1
#SBATCH -c 12
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o <HOME>/gsoc2026-deeplense-qml/experiments/13_iqp_turbulence_replication/logs_controlled/mechseed-%j.out
#SBATCH -e <HOME>/gsoc2026-deeplense-qml/experiments/13_iqp_turbulence_replication/logs_controlled/mechseed-%j.err
#
# ===========================================================================================
# Fig. 2(a): the M sweep {0, 2, 4, 6, 8, 16} over 5 model seeds.
#
# mech_seeds.py reproduces, in one process, the three held-out projection protocols the
# seed-0 sweep is spread over (affine_latent.run_config for M = 0, 4; diag3.joint for
# M = 2, 6; diag.py's exp2_redo lr x init sweep for M = 8, 16), plus diag2.py's random-
# direction reference.  Seed 0 is re-run so that agg_mech_seeds.py can check it reproduces
# results.json / diag3.json / diag.json / diag2.json exactly before any seed table is quoted.
#
# The DATA SPLIT does not move with the seed: data_lens_r3.load_r3 pins it at R3_SPLIT_SEED
# and documents that it must not follow an arm's seed.  Only the model does: theta0's random
# start, Phi's draw and z's draw.
#
# CPU only (the GPU is requested because dev schedules on GPU nodes; dev caps CPUs at 12 per
# GPU).  torch.set_num_threads(4) is NOT cosmetic here: at 2 threads the joint Adam fit for
# M >= 6 lands on a slightly different optimum (the reduction order of the z @ Phi matmul
# changes), and seed 0 then misses the published M = 6/8/16 values by ~7e-3 while M = 0/2/4
# still match to 1e-10.  Four threads is what affine_latent.py sets, so four threads is what
# reproduces.  Five seeds x 4 threads oversubscribe the 12 cores ~1.7x; ~8 min wall, 2 h
# requested.
# ===========================================================================================
set -uo pipefail

REPL=<HOME>/gsoc2026-deeplense-qml/experiments/13_iqp_turbulence_replication
DIR="$REPL/mech_tests/mech_test2"
LOGS="$REPL/logs_controlled"
cd "$DIR"
mkdir -p "$LOGS"

export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=/usr/bin/python3
ts() { date +"%Y-%m-%d %H:%M:%S"; }

echo "[SLURM] $(ts) job=${SLURM_JOB_ID:-none} node=$(hostname) part=${SLURM_JOB_PARTITION:-none} cpus=${SLURM_CPUS_PER_TASK:-?}"

"$PY" - <<'PYEOF' || { echo "[FATAL] python environment broken"; exit 1; }
import sys, numpy, torch, scipy
print("[env] python %s | numpy %s | torch %s | scipy %s"
      % (sys.version.split()[0], numpy.__version__, torch.__version__, scipy.__version__))
PYEOF

echo "[pre] $(ts) py_compile"
"$PY" -m py_compile mech_seeds.py agg_mech_seeds.py || { echo "[FATAL] py_compile failed"; exit 1; }

echo "[pre] $(ts) smoke (--smoke, all budgets /100, throwaway output)"
if ! "$PY" mech_seeds.py --seed 0 --smoke --threads 4 > "$LOGS/mechseed-smoke-${SLURM_JOB_ID:-0}.log" 2>&1; then
    echo "[FATAL] smoke failed:"; tail -40 "$LOGS/mechseed-smoke-${SLURM_JOB_ID:-0}.log"; exit 1
fi
echo "[pre] smoke OK"; grep -E "^\[s0\] (M=|refs|M sweep)" "$LOGS/mechseed-smoke-${SLURM_JOB_ID:-0}.log" | tail -5

echo "[run] $(ts) five seeds in parallel"
declare -a PIDS=()
for S in 0 1 2 3 4; do
    "$PY" mech_seeds.py --seed "$S" --threads 4 > "$LOGS/mechseed-s${S}-${SLURM_JOB_ID:-0}.log" 2>&1 &
    PIDS+=($!)
done
RC=0
for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}" || { echo "[ERROR] seed $i exited non-zero"; RC=1; }
done
for S in 0 1 2 3 4; do
    echo "--- seed $S tail ---"
    tail -12 "$LOGS/mechseed-s${S}-${SLURM_JOB_ID:-0}.log"
done
if [ "$RC" -ne 0 ]; then echo "[FATAL] at least one seed failed"; exit "$RC"; fi

echo "[agg] $(ts) aggregating"
"$PY" agg_mech_seeds.py
RCA=$?
echo "[done] $(ts) seeds=$RC agg=$RCA"
ls -la "$DIR"/mech_seeds_s*.json "$DIR/results_seeds.json" 2>&1
exit 0
