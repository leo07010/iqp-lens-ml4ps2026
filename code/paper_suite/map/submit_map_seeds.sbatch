#!/bin/bash
#SBATCH -J iqp_mapseed
#SBATCH -A <YOUR_ACCOUNT>
#SBATCH -p dev
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 01:00:00
#SBATCH -o <HOME>/gsoc2026-deeplense-qml/experiments/13_iqp_turbulence_replication/logs_controlled/mapseed-%j.out
#SBATCH -e <HOME>/gsoc2026-deeplense-qml/experiments/13_iqp_turbulence_replication/logs_controlled/mapseed-%j.err
#
# ===========================================================================================
# Table 1 (correlation-complexity map) over 5 seeds.
#
# run_map.py is re-run UNCHANGED at --seed 0..4 with --with-d2 (the flag that produces the
# turbulence per-snapshot K = 4,096 row Table 1 quotes).  Seed 0 is re-run as well, purely as
# a reproduction check against the committed results_map.json; agg_map_seeds.py reports the
# max |diff| and refuses to hide it.
#
# What the seed moves (see agg_map_seeds.py docstring): the K-sample draws, the permutation
# nulls and the i.i.d.-uniform nulls.  It does NOT move the data split -- data_lens_r3 and
# data_lens fix their splits at their own SPLIT_SEED and explicitly decouple them from any
# arm's seed, so the "200 held-out images" are the same 200 in every seed by design.
#
# Cost: the committed seed-0 run took 63 s wall (meta.wallclock_s).  5 seeds ~ 6 min.
# CPU only; the GPU is requested because dev schedules on GPU nodes.  1 h = ~10x headroom.
# ===========================================================================================
set -uo pipefail

REPL=<HOME>/gsoc2026-deeplense-qml/experiments/13_iqp_turbulence_replication
DIR="$REPL/paper_suite/map"
LOGS="$REPL/logs_controlled"
cd "$DIR"
mkdir -p "$LOGS"

export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=/usr/bin/python3
ts() { date +"%Y-%m-%d %H:%M:%S"; }

echo "[SLURM] $(ts) job=${SLURM_JOB_ID:-none} node=$(hostname) part=${SLURM_JOB_PARTITION:-none}"

"$PY" - <<'PYEOF' || { echo "[FATAL] python environment broken"; exit 1; }
import sys, numpy, scipy
print("[env] python %s | numpy %s | scipy %s" % (sys.version.split()[0], numpy.__version__, scipy.__version__))
PYEOF

echo "[pre] $(ts) py_compile"
"$PY" -m py_compile run_map.py map_estimators.py agg_map_seeds.py || { echo "[FATAL] py_compile failed"; exit 1; }

echo "[pre] $(ts) smoke (--quick, seed 7, throwaway output)"
"$PY" run_map.py --quick --with-d2 --seed 7 --out /tmp/map_smoke_$$.json >/dev/null 2>&1
if [ $? -ne 0 ]; then echo "[FATAL] --quick smoke failed"; "$PY" run_map.py --quick --with-d2 --seed 7 --out /tmp/map_smoke_$$.json 2>&1 | tail -30; exit 1; fi
rm -f /tmp/map_smoke_$$.json
echo "[pre] smoke OK"

RC=0
for S in 0 1 2 3 4; do
    OUTJ="$DIR/results_map_seed${S}.json"
    echo "[map] $(ts) seed=$S -> $OUTJ"
    "$PY" run_map.py --seed "$S" --with-d2 --out "$OUTJ"
    r=$?
    echo "[map] $(ts) seed=$S exit=$r"
    [ "$r" -ne 0 ] && RC=$r
done
if [ "$RC" -ne 0 ]; then echo "[FATAL] at least one seed failed"; exit "$RC"; fi

echo "[agg] $(ts) aggregating"
"$PY" agg_map_seeds.py
RCA=$?
echo "[done] $(ts) map_seeds=$RC agg=$RCA"
ls -la "$DIR"/results_map_seed*.json "$DIR/results_map_seeds.json" 2>&1
exit 0
