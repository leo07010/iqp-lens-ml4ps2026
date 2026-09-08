#!/bin/bash
#SBATCH -J iqp_turb_resynth_last
#SBATCH -A <YOUR_ACCOUNT>
#SBATCH -p 8gpus
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 01:30:00
#SBATCH -o <REPO_ROOT>/code/logs_d2/resynth_last-%j.out
#SBATCH -e <REPO_ROOT>/code/logs_d2/resynth_last-%j.err
#
# Re-evaluate D2 Table 2 under LAST-WRITE-WINS per-cell aggregation (agg_variants.aggregate_last)
# instead of codec.aggregate's MEAN, for EVERY tuple-based arm: IQP (12), CTRL-marginal (3),
# RBM (3). diag_scripts/q1c_agg_convention.py showed this swap alone moves IQP's D2 PDF-JS from
# ~0.24 (mean) to ~0.028/0.045 (last), close to the paper's reported ~0.024/0.022 -- the question
# this job answers is whether the paper's apparent IQP-vs-baselines Table-2 gap survives once
# RBM and CTRL-marginal ALSO get the same convention (they are tuple-based too, so nothing about
# "last" is IQP-specific). DCGAN and CTRL-interp are NOT tuple-based (no per-cell aggregation
# step at all) so they are untouched and reused verbatim from outputs_d2_main/ via symlink below.
#
# Sample counts: every arm uses ITS OWN ORIGINAL n_synth (~1e5/tau: IQP/CTRL-marginal via
# meta["config"]["n_synth"]/--n-synth default 100000; RBM via its own recorded
# gen_chains*gen_keep, no x10 bump) -- this isolates the AGGREGATION CONVENTION as the only
# variable, not sample count. No model is retrained for IQP/CTRL-marginal (verbatim
# theta_core.npy/theta_lat_anchors.npy reuse / closed-form p_vec replay, same as
# resynthesize.py). RBM genuinely has no checkpoint (baseline_rbm.py never saves W/b_v/b_h), so
# getting RBM codes at all requires a full PCD retrain -- gated behind --rbm-retrain, same
# opt-in convention as resynthesize.py, exercised explicitly below (not left off) because this
# job's whole point is a Table 2 that includes RBM under "last" too.
#
# Time budget: IQP+CTRL-marginal resynthesis is CHEAP -- 1e5 samples/tau (smaller than
# resynthesize.py's 1e6 sample1e6 mode, which the 1h submit_resynth.sbatch budget already
# covered comfortably per its own header note) is forward-only sampling on an already-optimized
# theta / already-fit p_vec, no gradient steps. RBM retrain: outputs_d2_main/rbm_seed0/meta.json
# records wallclock_s=54.66 for ONE seed's FULL 60000 PCD steps (30000 core + 10*3000 latent)
# PLUS generating all 200 held-out tau at ~1e5 samples/tau on GPU -- this script's RBM path does
# the IDENTICAL amount of work (same steps_core/steps_lat/gen_chains/gen_keep, no x10 bump), so
# 3 seeds should total ~3 min, not the ~20 min upper estimate in the task spec. If any RBM
# ms/pcd-step printed below is far above that implied rate ((54.66s)/60000steps ~= 0.91 ms/step),
# this job's own [rbm] log lines will show it -- check before trusting "well under budget".
set -uo pipefail

REPO=<REPO_ROOT>/code
cd "$REPO"
mkdir -p logs_d2

OUT="$REPO/outputs_d2_main"
OUT_LAST="$REPO/outputs_d2_last"
OUT_MERGED="$REPO/outputs_d2_last_merged"
ts() { date +"%Y-%m-%d %H:%M:%S"; }

# ---------------------------------------------------------------- fail-fast: source outputs must exist
if [ ! -d "$OUT/iqp_K6_seed0" ]; then
    echo "[FATAL] $OUT/iqp_K6_seed0 not found -- outputs_d2_main/ must already contain the trained IQP/RBM/CTRL-marginal arms."
    exit 1
fi
if [ ! -d "$OUT/ground_truth" ]; then
    echo "[FATAL] $OUT/ground_truth not found -- need the original D2 ground truth to evaluate against."
    exit 1
fi

module purge; module load cuda/13.0
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
PY=/usr/bin/python3

echo "[SLURM] job=$SLURM_JOB_ID node=$(hostname) part=$SLURM_JOB_PARTITION gpus=$($PY -c 'import torch;print(torch.cuda.device_count())')"
echo "[$(ts)] disk free on \$HOME: $(df -h "$REPO" | tail -1 | awk '{print $4}') (codes_t*.npy for all arms is an estimated ~1.4GB total -- abort manually if this is too tight)"

# ---------------------------------------------------------------- fail-fast pre-checks
echo "[$(ts)] pre-check: py_compile all .py"
for f in *.py; do
    "$PY" -m py_compile "$f" || { echo "[FATAL] py_compile failed for $f"; exit 1; }
done

echo "[$(ts)] pre-check: test_agg_variants.py (CPU, <1s: mean/last/first/random-one agreement + brute-force checks)"
"$PY" test_agg_variants.py > logs_d2/precheck_test_agg_variants.log 2>&1 \
    || { echo "[FATAL] test_agg_variants.py failed -- see logs_d2/precheck_test_agg_variants.log"; tail -40 logs_d2/precheck_test_agg_variants.log; exit 1; }
tail -10 logs_d2/precheck_test_agg_variants.log
echo "[$(ts)] pre-checks passed"

rm -rf "$OUT_LAST" "$OUT_MERGED"
mkdir -p "$OUT_LAST" "$OUT_MERGED"

# ---------------------------------------------------------------- resynthesize: IQP + CTRL-marginal first
echo "[$(ts)] START resynthesize_last: IQP x12 + CTRL-marginal x3 (n_synth=100000/tau, no retraining)"
"$PY" resynthesize_last.py \
    --arms-glob "$OUT/iqp_*,$OUT/ctrl_marginal_*" \
    --out "$OUT_LAST" \
    --device cuda \
    --n-synth 100000 \
    > logs_d2/resynth_last_iqp_ctrl.log 2>&1 || { echo "[FATAL] resynthesize_last (iqp+ctrl) failed"; tail -60 logs_d2/resynth_last_iqp_ctrl.log; exit 1; }
tail -40 logs_d2/resynth_last_iqp_ctrl.log
echo "[$(ts)] DONE resynthesize_last: IQP + CTRL-marginal"

# ---------------------------------------------------------------- resynthesize: RBM (retrain, opt-in flag exercised deliberately)
echo "[$(ts)] START resynthesize_last: RBM x3 (--rbm-retrain: full PCD retrain, no checkpoint exists to reuse)"
"$PY" resynthesize_last.py \
    --arms-glob "$OUT/rbm_*" \
    --out "$OUT_LAST" \
    --device cuda \
    --rbm-retrain \
    > logs_d2/resynth_last_rbm.log 2>&1 || { echo "[FATAL] resynthesize_last (rbm) failed"; tail -60 logs_d2/resynth_last_rbm.log; exit 1; }
tail -40 logs_d2/resynth_last_rbm.log
echo "[$(ts)] DONE resynthesize_last: RBM"

# ---------------------------------------------------------------- evaluate: outputs_d2_last alone (12 IQP + 3 CTRL-marginal + 3 RBM, all "last")
echo "[$(ts)] START evaluate: outputs_d2_last alone (18 arm dirs expected: 12 IQP + 3 CTRL-marginal + 3 RBM)"
n_last_dirs=$(find "$OUT_LAST" -mindepth 1 -maxdepth 1 -type d | wc -l)
echo "[$(ts)] $OUT_LAST has $n_last_dirs arm dirs (expected 18)"
"$PY" evaluate.py --outputs-dir "$OUT_LAST" --ground-truth "$OUT/ground_truth" \
    --results "$REPO/results_d2_last.jsonl" \
    > logs_d2/evaluate_last.log 2>&1 || { echo "[FATAL] evaluate (last-only) failed"; tail -30 logs_d2/evaluate_last.log; exit 1; }
cat logs_d2/evaluate_last.log
echo "[$(ts)] DONE evaluate: outputs_d2_last"

# ---------------------------------------------------------------- merged table: outputs_d2_last (tuple-based, "last") + outputs_d2_main's DCGAN/CTRL-interp (not tuple-based, untouched)
echo "[$(ts)] building $OUT_MERGED: symlink DCGAN + ctrl_interp from outputs_d2_main (not tuple-based, unaffected by aggregation convention) + everything from outputs_d2_last"
for d in "$OUT"/dcgan_* "$OUT/ctrl_interp"; do
    [ -d "$d" ] && ln -s "$d" "$OUT_MERGED/$(basename "$d")"
done
for d in "$OUT_LAST"/*/; do
    d="${d%/}"
    ln -s "$d" "$OUT_MERGED/$(basename "$d")"
done
n_merged=$(find "$OUT_MERGED" -mindepth 1 -maxdepth 1 | wc -l)
echo "[$(ts)] $OUT_MERGED has $n_merged entries (symlinks)"

echo "[$(ts)] START evaluate: outputs_d2_last_merged (full Table 2 under the 'last' convention)"
"$PY" evaluate.py --outputs-dir "$OUT_MERGED" --ground-truth "$OUT/ground_truth" \
    --results "$REPO/results_d2_last_merged.jsonl" \
    > logs_d2/evaluate_last_merged.log 2>&1 || { echo "[FATAL] evaluate (merged) failed"; tail -30 logs_d2/evaluate_last_merged.log; exit 1; }
cat logs_d2/evaluate_last_merged.log
echo "[$(ts)] DONE submit_resynth_last.sbatch"
