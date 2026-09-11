#!/bin/bash
#SBATCH -J gridmlp_time
#SBATCH -A <YOUR_ACCOUNT>
#SBATCH -p dev
#SBATCH --gres=gpu:2
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 00:50:00
#SBATCH -o <REPO_ROOT>/code/logs_gridmlp/time-%j.out
#SBATCH -e <REPO_ROOT>/code/logs_gridmlp/time-%j.err
#
# Time the WORST-CASE cell of the grid (N = 100 with all five snapshots) for each arm, so the
# grid's walltime is set from a measurement.  Also checks that evaluate_v11.py accepts the two
# new arms' output directories, which the smoke could not check (its timing wrapper died).
set -u
ts() { date "+%F %T"; }
REPO=<REPO_ROOT>/code
cd "$REPO" || exit 1
PY=/usr/bin/python3
OUT=<WORK>/gsoc_iqp_gridmlp/timing
mkdir -p "$OUT" logs_gridmlp
echo "[$(ts)] START job=${SLURM_JOB_ID:-none} node=$(hostname)"
C="--objective mmdflat --prior resample --M 4 --seed 0 --device cuda --n-train 100 --n-gen 200 --n-holdout 200"
SN="--save-every 1000 2000 4000 8000 16000"

t0=$SECONDS
CUDA_VISIBLE_DEVICES=0 "$PY" -u train_iqp_affine_n12.py $C $SN --out "$OUT/n12es_n100" \
    > logs_gridmlp/time_n12es.log 2>&1 &
p1=$!
CUDA_VISIBLE_DEVICES=1 "$PY" -u train_iqp_affine_mlp.py $C $SN --gnorm rownorm \
    --out "$OUT/mlpes_n100" > logs_gridmlp/time_mlpes.log 2>&1 &
p2=$!
wait $p1; r1=$?
t1=$SECONDS
wait $p2; r2=$?
t2=$SECONDS
echo "[$(ts)] n12 ES N=100 rc=$r1 : $((t1 - t0)) s   |   mlp ES N=100 rc=$r2 : $((t2 - t0)) s"
tail -3 logs_gridmlp/time_n12es.log
tail -3 logs_gridmlp/time_mlpes.log
[ "$r1" = 0 ] && [ "$r2" = 0 ] || exit 1

echo "[$(ts)] evaluate_v11 on the snapshots (contract check)"
"$PY" -u evaluate_v11.py --runs "$OUT"/n12es_n100_s0*k "$OUT"/mlpes_n100_s0*k \
    --n-holdout 200 --results "$OUT/timing_v11.jsonl" > logs_gridmlp/time_v11.log 2>&1 \
    || { echo "[WARN] evaluate_v11 failed"; tail -30 logs_gridmlp/time_v11.log; exit 1; }
tail -16 logs_gridmlp/time_v11.log
echo "[$(ts)] DONE"
