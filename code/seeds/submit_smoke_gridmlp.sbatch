#!/bin/bash
#SBATCH -J gridmlp_smoke
#SBATCH -A <YOUR_ACCOUNT>
#SBATCH -p dev
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:40:00
#SBATCH -o <REPO_ROOT>/code/logs_gridmlp/smoke-%j.out
#SBATCH -e <REPO_ROOT>/code/logs_gridmlp/smoke-%j.err
#
# Smoke test for the two new arms BEFORE the 5-seed grid is submitted:
#   (A) train_iqp_affine_n12.py   12 qubits, whole 64x64 frame, 4,096 codes
#   (B) train_iqp_affine_mlp.py   10 qubits, theta = theta0 + g(z), g a tanh MLP
# Checks that the loss falls, that held-out projection runs, that the output contract is
# intact, and TIMES one full-size run of each so the grid's walltime can be set from a
# measurement instead of a guess.
set -u
ts() { date "+%F %T"; }
REPO=<REPO_ROOT>/code
cd "$REPO" || exit 1
PY=/usr/bin/python3
OUT=<WORK>/gsoc_iqp_gridmlp/smoke
mkdir -p "$OUT" logs_gridmlp
echo "[$(ts)] START job=${SLURM_JOB_ID:-none} node=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$PY" -m py_compile data_lens_n12.py train_iqp_affine_n12.py train_iqp_affine_mlp.py || exit 1
echo "[$(ts)] py_compile OK"

fail=0
COMMON="--objective mmdflat --prior resample --M 4 --seed 0 --device cuda --n-train 11"
TINY="--steps 300 --theta0-steps 100 --n-gen 8 --n-holdout 8 --infer-steps 60 --infer-restarts 3 --log-every 100"

echo "[$(ts)] --- tiny A: n12 ---"
"$PY" -u train_iqp_affine_n12.py $COMMON $TINY --out "$OUT/tiny_n12" || fail=1
echo "[$(ts)] --- tiny B1: mlp rownorm ---"
"$PY" -u train_iqp_affine_mlp.py $COMMON $TINY --gnorm rownorm --out "$OUT/tiny_mlp" || fail=1
echo "[$(ts)] --- tiny B2: mlp zscale ---"
"$PY" -u train_iqp_affine_mlp.py $COMMON $TINY --gnorm zscale --out "$OUT/tiny_mlpzs" || fail=1
echo "[$(ts)] --- tiny B3: mlp with LINEAR-regime input scale (control: should look affine) ---"
"$PY" -u train_iqp_affine_mlp.py $COMMON $TINY --gnorm rownorm --g-input-scale 1.0 \
    --out "$OUT/tiny_mlplin" || fail=1
[ "$fail" = 0 ] || { echo "[$(ts)] TINY RUNS FAILED"; exit 1; }

echo "[$(ts)] --- checks ---"
"$PY" -u seeds/check_gridmlp_smoke.py "$OUT" || fail=1

echo "[$(ts)] --- timing: one FULL-SIZE run of each (N=11, the grid's own settings) ---"
/usr/bin/time -f "A n12 full N=11: %e s" "$PY" -u train_iqp_affine_n12.py \
    --objective mmdflat --prior resample --M 4 --seed 0 --device cuda --n-train 11 \
    --steps 4000 --n-gen 200 --out "$OUT/full_n12_n11" > logs_gridmlp/smoke_full_n12.log 2>&1 \
    || { echo "[WARN] full n12 failed"; tail -20 logs_gridmlp/smoke_full_n12.log; fail=1; }
tail -3 logs_gridmlp/smoke_full_n12.log
/usr/bin/time -f "B mlp full N=11: %e s" "$PY" -u train_iqp_affine_mlp.py \
    --objective mmdflat --prior resample --M 4 --seed 0 --device cuda --n-train 11 \
    --steps 4000 --n-gen 200 --gnorm rownorm --out "$OUT/full_mlp_n11" \
    > logs_gridmlp/smoke_full_mlp.log 2>&1 \
    || { echo "[WARN] full mlp failed"; tail -20 logs_gridmlp/smoke_full_mlp.log; fail=1; }
tail -3 logs_gridmlp/smoke_full_mlp.log

echo "[$(ts)] --- v11 scoring of the two full runs (checks evaluate_v11 accepts them) ---"
"$PY" -u evaluate_v11.py --runs "$OUT/full_n12_n11" "$OUT/full_mlp_n11" --n-holdout 200 \
    --results "$OUT/smoke_v11.jsonl" > logs_gridmlp/smoke_v11.log 2>&1 \
    || { echo "[WARN] evaluate_v11 failed"; tail -30 logs_gridmlp/smoke_v11.log; fail=1; }
tail -12 logs_gridmlp/smoke_v11.log
echo "[$(ts)] DONE (fail=$fail)"
exit "$fail"
