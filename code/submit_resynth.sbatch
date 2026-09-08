#!/bin/bash
#SBATCH -J iqp_turb_resynth
#SBATCH -A <YOUR_ACCOUNT>
#SBATCH -p dev
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -o <REPO_ROOT>/code/logs_d2/resynth-%j.out
#SBATCH -e <REPO_ROOT>/code/logs_d2/resynth-%j.err
#
# Resynthesize the 12 IQP arms + 3 CTRL-marginal arms of outputs_d2_main/ at two
# aggregation modes (exact conditional expectation, and sample1e6 = the paper's upper
# 1e6-sample range instead of job 342220's 1e5) FROM THE ALREADY-TRAINED theta_core.npy /
# theta_lat_anchors.npy -- no retraining, see resynthesize.py's module docstring.
#
# RBM (3 seeds) is deliberately left OFF by default: baseline_rbm.py's main() never
# checkpoints W/b_v/b_h/latents_arr, so there is no stored model to resynthesize sample1e6
# from -- getting it would require --rbm-retrain (a genuine full PCD retrain, and not even
# guaranteed bit-identical to the original run since baseline_rbm.py trains on CUDA without
# torch.use_deterministic_algorithms). Left as an opt-in below (commented out) so this does
# not silently burn GPU time on an actual retrain without a deliberate decision to do so.
#
# Time budget: this is CHEAP compared to training -- exact/sample1e6 are forward-only
# (probs()/sample()) on an already-optimized theta, no gradient steps. The CPU smoke test
# below (test_resynthesize.py, 3 tau, exact+sample1e6) took ~90s wall UNCAPPED (no
# OMP_NUM_THREADS) on this repo's login node; on the allocated GPU here it will be far
# faster per tau (H100 vs CPU FWHT, see train_iqp.py's own device-comparison note), so
# even 12 IQP arms x 200 tau x 2 modes should be a small fraction of the 1h requested.
# CTRL-marginal's sample1e6 (pure numpy: 1e6-draw multinomial + codec.aggregate) is the
# next cheapest thing here; RBM retrain, if ever enabled, is the only piece that could
# threaten the 1h budget (30000 + 10*3000 = 60000 PCD steps per seed, x3 seeds) -- not
# exercised by default.
set -uo pipefail

REPO=<REPO_ROOT>/code
cd "$REPO"
mkdir -p logs_d2

OUT="$REPO/outputs_d2_main"
OUT_RESYNTH="$REPO/outputs_d2_resynth"
ts() { date +"%Y-%m-%d %H:%M:%S"; }

# ---------------------------------------------------------------- fail-fast: source outputs must exist
if [ ! -d "$OUT/iqp_K6_seed0" ]; then
    echo "[FATAL] $OUT/iqp_K6_seed0 not found -- outputs_d2_main/ must already contain the trained IQP/RBM/CTRL-marginal arms."
    exit 1
fi

module purge; module load cuda/13.0
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
PY=/usr/bin/python3

echo "[SLURM] job=$SLURM_JOB_ID node=$(hostname) part=$SLURM_JOB_PARTITION gpus=$($PY -c 'import torch;print(torch.cuda.device_count())')"

# ---------------------------------------------------------------- fail-fast pre-checks
echo "[$(ts)] pre-check: py_compile all .py"
for f in *.py; do
    "$PY" -m py_compile "$f" || { echo "[FATAL] py_compile failed for $f"; exit 1; }
done

echo "[$(ts)] pre-check: test_resynthesize.py (quick: reshape-vs-brute-force + 3 real tau on CPU)"
"$PY" test_resynthesize.py > logs_d2/precheck_test_resynthesize.log 2>&1 \
    || { echo "[FATAL] test_resynthesize.py failed -- see logs_d2/precheck_test_resynthesize.log"; tail -40 logs_d2/precheck_test_resynthesize.log; exit 1; }
tail -10 logs_d2/precheck_test_resynthesize.log
echo "[$(ts)] pre-checks passed"

rm -rf "$OUT_RESYNTH"
mkdir -p "$OUT_RESYNTH"

# ---------------------------------------------------------------- resynthesize
# IQP (12 arms) + CTRL-marginal (3 seeds) x {exact, sample1e6}. RBM is matched by the
# default --arms-glob too but is skipped with a printed warning (no --rbm-retrain here --
# see header comment). To also retrain+resynthesize RBM at sample1e6, uncomment:
#   "$PY" resynthesize.py --arms-glob "$OUT/rbm_*" --modes sample1e6 --out "$OUT_RESYNTH" \
#       --device cuda --rbm-retrain > logs_d2/resynth_rbm.log 2>&1
echo "[$(ts)] START resynthesize (IQP x12, CTRL-marginal x3, modes=exact,sample1e6)"
"$PY" resynthesize.py \
    --arms-glob "$OUT/iqp_*,$OUT/rbm_*,$OUT/ctrl_marginal_*" \
    --modes exact,sample1e6 \
    --out "$OUT_RESYNTH" \
    --device cuda \
    > logs_d2/resynth.log 2>&1 || { echo "[FATAL] resynthesize failed"; tail -60 logs_d2/resynth.log; exit 1; }
tail -40 logs_d2/resynth.log
echo "[$(ts)] DONE resynthesize"

# ---------------------------------------------------------------- evaluate
echo "[$(ts)] START evaluate (resynthesized arms vs the original ground_truth)"
"$PY" evaluate.py --outputs-dir "$OUT_RESYNTH" --ground-truth "$OUT/ground_truth" \
    --results "$REPO/results_d2_resynth.jsonl" \
    > logs_d2/evaluate_resynth.log 2>&1 || { echo "[FATAL] evaluate failed"; tail -30 logs_d2/evaluate_resynth.log; exit 1; }
cat logs_d2/evaluate_resynth.log
echo "[$(ts)] DONE submit_resynth.sbatch"
