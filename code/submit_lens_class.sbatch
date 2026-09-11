#!/bin/bash
#SBATCH -J iqp_lens_class
#SBATCH -A <YOUR_ACCOUNT>
#SBATCH -p dev
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o <REPO_ROOT>/code/logs_lens/sbatch-class-%j.out
#SBATCH -e <REPO_ROOT>/code/logs_lens/sbatch-class-%j.err
#
# APPENDIX RUN: the paper's best arm, trained SEPARATELY on each of the three dark-matter
# substructure classes (axion / cdm / no_sub).
#
# Every result in the paper is trained on the three classes POOLED.  The appendix answers
# the narrower question "does the same arm train and generate for each class on its own?".
# The arm is unchanged -- IQP-affine, M = 4, MMD-flat objective, resample prior, seed 0,
# lr 0.02, 4000 joint steps, 1000 theta0 steps, 8 x 1000-step projection restarts -- the
# ONLY change is which 100 images it trains on and which 200 held-out images it is scored
# against.  Expect the three classes to look alike: the class-conditional image statistics
# differ by ~3% (docs/LENS_IMAGE_CHARACTERISTICS.md).  The point is that each class trains
# and is scored on its own, not that the samples differ.
#
# NEW FILES ONLY.  This script calls data_lens_class.py, train_iqp_affine_class.py,
# evaluate_class.py and test_class.py.  Those four IMPORT data_lens_r3 / train_iqp_affine_r3
# / evaluate_r3 / evaluate_r3b and modify none of them on disk: the per-class split is
# installed at run time by swapping three functions inside the imported data_lens_r3 module.
# Output tree (outputs_lens_class), results file (results_class.jsonl) and log names
# (class_*) are separate from every other round.
#
# ------------------------------------------------------------------------- time budget
# From the MEASURED round-3b numbers (same arm, same N, same steps; those are the
# conservative CPU bound and the GPU is never slower):
#   IQP N = 100 MMDflat  ~ 3.9 min  x 3 classes = 11.7 min
#   IQP N = 11  MMDflat  ~ 2.5 min  x 3 classes =  7.5 min
#   evaluate (2 runs per class, conv_mmd2 twice per row)  ~ 1.5 min x 3 =  4.5 min
#   pre-checks: class cache build (reads the 1.4 GB npz once) ~1 min, test_class.py ~1 min,
#               5-step CUDA smoke ~1 min
# => ~28 min.  Requesting 2 h on dev (4 h limit), ~4x headroom.  One GPU: the six arms run
#    sequentially, which is simpler than sharing and still fits many times over.
set -uo pipefail

REPO=<REPO_ROOT>/code
cd "$REPO"
mkdir -p logs_lens

module purge; module load cuda/13.0
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
PY=/usr/bin/python3
OUT="$REPO/outputs_lens_class"
RESULTS="$REPO/results_class.jsonl"
CLASSES=(axion cdm no_sub)
ts() { date +"%Y-%m-%d %H:%M:%S"; }

echo "[SLURM] job=$SLURM_JOB_ID node=$(hostname) part=$SLURM_JOB_PARTITION gpus=$($PY -c 'import torch;print(torch.cuda.device_count())')"

# ---------------------------------------------------------------- fail-fast: py_compile
echo "[$(ts)] pre-check: py_compile the four new files"
for f in data_lens_class.py train_iqp_affine_class.py evaluate_class.py test_class.py; do
    "$PY" -m py_compile "$f" || { echo "[FATAL] py_compile failed for $f"; exit 1; }
done

# ---------------------------------------------------------------- fail-fast: class caches
# Reads <HOME>/mae-lensing/model_II.npz ONCE for all three classes.
echo "[$(ts)] pre-check: build the three per-class caches"
"$PY" data_lens_class.py || { echo "[FATAL] the per-class caches could not be built"; exit 1; }

# ---------------------------------------------------------------- fail-fast: unit test
echo "[$(ts)] pre-check: test_class.py (synthetic dataset, cpu)"
"$PY" test_class.py > logs_lens/precheck_class_test.log 2>&1 \
    || { echo "[FATAL] test_class.py failed -- see logs_lens/precheck_class_test.log"; tail -40 logs_lens/precheck_class_test.log; exit 1; }
tail -2 logs_lens/precheck_class_test.log

# ---------------------------------------------------------------- fail-fast: CUDA smoke
echo "[$(ts)] pre-check: 5-step CUDA smoke of the real arm on the real per-class data"
rm -rf logs_lens/class_cuda_smoke
"$PY" train_iqp_affine_class.py --class-name axion --M 4 --objective mmdflat \
    --prior resample --n-train 3 --n-holdout 4 --n-gen 2 --steps 5 --theta0-steps 5 \
    --infer-restarts 3 --infer-steps 5 --infer-chunk 2 --log-every 5 --device cuda \
    --out logs_lens/class_cuda_smoke/axion > logs_lens/precheck_class_cuda.log 2>&1 \
    || { echo "[FATAL] train_iqp_affine_class.py does not run on cuda"; tail -30 logs_lens/precheck_class_cuda.log; exit 1; }
rm -rf logs_lens/class_cuda_smoke
echo "[$(ts)] pre-checks passed"

rm -rf "$OUT"
mkdir -p "$OUT"

# ---------------------------------------------------------------- the arms
# Spelled out rather than left to the defaults, so the log records exactly what ran.
# These are the round-3b amendment settings, identical to the pooled best arm's.
IQP_COMMON="--n-holdout 200 --n-gen 200 --device cuda --seed 0 --lr 0.02 --steps 4000 \
 --theta0-steps 1000 --infer-restarts 8 --infer-steps 1000 --infer-lr 0.005 --infer-chunk 50 \
 --log-every 500 --M 4 --objective mmdflat --prior resample"

fail=0
n_run=0
for N in 100 11; do
    for C in "${CLASSES[@]}"; do
        TAG="${C}_m4_mmdflat_resample_n${N}"
        echo "[$(ts)] START $TAG"
        "$PY" train_iqp_affine_class.py --class-name "$C" $IQP_COMMON \
            --n-train "$N" --out "$OUT/$TAG" > "logs_lens/class_${TAG}.log" 2>&1 \
            || { echo "[WARN] $TAG failed, see logs_lens/class_${TAG}.log"; tail -20 "logs_lens/class_${TAG}.log"; fail=1; }
        n_run=$((n_run + 1))
    done
done

echo "[$(ts)] all $n_run arms finished (fail=$fail)"
echo "[$(ts)] run dirs written: $(find "$OUT" -mindepth 1 -maxdepth 1 -type d | wc -l) (expected 6)"

# ---------------------------------------------------------------- evaluate
# Each class's runs are scored against THAT class's 200 held-out images, by the unchanged
# evaluate_r3b/evaluate_r3 functions and the same CONV_SEED.
echo "[$(ts)] START evaluate_class"
"$PY" evaluate_class.py --outputs-dir "$OUT" --classes "${CLASSES[@]}" \
    --n-holdout 200 --n-train-max 100 --seed 0 --results "$RESULTS" \
    > logs_lens/evaluate_class.log 2>&1 \
    || { echo "[WARN] evaluate_class failed"; tail -40 logs_lens/evaluate_class.log; fail=1; }
tail -60 logs_lens/evaluate_class.log

echo "[$(ts)] rows in $RESULTS: $(wc -l < "$RESULTS" 2>/dev/null || echo 0)"
echo "[$(ts)] DONE submit_lens_class.sbatch (fail=$fail)"
exit "$fail"
