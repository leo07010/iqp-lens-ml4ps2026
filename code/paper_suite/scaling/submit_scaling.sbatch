#!/bin/bash
#SBATCH -J iqp_lens_scaling
#SBATCH -A <YOUR_ACCOUNT>
#SBATCH -p dev
#SBATCH --gres=gpu:8
#SBATCH -c 32
#SBATCH --mem=128G
#SBATCH -t 02:30:00
#SBATCH -o <REPO_ROOT>/code/logs_lens/sbatch-scaling-%j.out
#SBATCH -e <REPO_ROOT>/code/logs_lens/sbatch-scaling-%j.err
#
# Data-scaling study for arXiv 2603.06440 applied to the DeepLense lens crops.
# docs/IQP_LENS_PAPER_SUITE_PLAN.md, row "DCGAN data-scaling N in {10, 30, 50, 70, 90, 100}":
# DCGAN-32 on the crops, plus IQP-affine-M4-MMDflat-resample and RBM-10bit at the same N.
# Prediction Q4: IQP's MMD^2 is flat in N while DCGAN improves with N and overtakes IQP
# between N = 30 and 50 (raw MMD^2).
#
# Touches NOTHING existing: this calls baseline_dcgan_r3.py / baseline_rbm_r3.py /
# train_iqp_affine_r3.py / evaluate_r3b.py / data_lens_r3.py exactly as round 3 / 3b already
# do, unmodified, with a NEW --n-train grid and a NEW output tree (outputs_lens_scaling,
# results_scaling.jsonl, logs_lens/scaling_*). Only the three files under paper_suite/scaling/
# (this script, plot_scaling.py, test_scaling.py -- plus plot_pseudotime.py, which this script
# does not call) are new.
#
# ---------------------------------------------------------------------------- job list
# 3 arms x N in {10, 30, 50, 70, 90, 100} = 18 (arm, N) cells.
#
#   arm label                        | script                  | flags beyond --n-train/--out
#   ----------------------------------+-------------------------+------------------------------
#   DCGAN-32                          | baseline_dcgan_r3.py    | round-3 defaults (steps 20000)
#   RBM-10bit                         | baseline_rbm_r3.py      | round-3 defaults (steps 20000)
#   IQP-affine-M4-MMDflat-resample    | train_iqp_affine_r3.py  | --M 4 --objective mmdflat
#                                     |                         | --prior resample, round-3b's
#                                     |                         | hyperparameters (IQP_COMMON)
#
# N = 100 REUSE (not a 19th-25th job): DCGAN-32, RBM-10bit and IQP-affine-M4-MMDflat-resample
# at N = 100 with n_holdout=200, n_gen=200, seed=0 were already run byte-for-byte identically
# in round 3 (outputs_lens_r3/{dcgan,rbm}_n100) and round 3b
# (outputs_lens_r3b/iqp_m4_mmdflat_resample_n100) -- verified complete (200/200 generated files
# each) before this script was written. Re-running them would waste the three single longest
# jobs in this grid (RBM-10bit N=100 alone is the ~18.6 min longest job here). This script
# copies them into outputs_lens_scaling/ if, AND ONLY IF, their meta.json is verified to match
# this grid's exact config (n_train/n_gen/n_holdout/seed/steps/arm); otherwise it falls back to
# running that (arm, N=100) cell fresh, so there is no HARD dependency on another round's tree.
#
# ------------------------------------------------------------------------- time budget
# MEASURED on GPU (cuda), round 3's own sbatch (submit_lens_r3.sbatch) and round 3b's
# (submit_lens_r3b.sbatch) header comments:
#   DCGAN-32   N=11  4.8 min   N=100  9.2 min   (batch = min(64, N): N in {70,90,100} all use
#                                                the same batch=64 as the N=100 measurement)
#   RBM-10bit  N=11  4.8 min   N=100 18.6 min   (PCD-1 processes all N images every step, so
#                                                cost is roughly linear in N, NOT capped)
#   IQP-affine (M4, any objective, round-3b hyperparameters)
#              N=11  ~2.5 min  N=100 ~3.9 min   (~2.2 min of this is the FIXED 200-held-out
#                                                projection cost, ~1.7 min is N-dependent)
# The task order's own assumption ("IQP and RBM short, DCGAN ~8 min") undercounts RBM: at
# N=100 RBM-10bit is the single LONGEST job here (18.6 min), longer than DCGAN (9.2 min) and
# IQP-affine (3.9 min) at the same N. Flagged in this run's report; kept as round 3's own
# --steps 20000 default (no shortcut) so this is the SAME modelling problem round 3 already
# validated at N=11/100, just swept over N.
#
# INTERPOLATED for N in {30, 50, 70, 90} (NOT independently measured -- this script's own logs
# after it runs are the real numbers; the mechanistic model behind each estimate is in the
# module docstrings of the two scripts it drives):
#   N            10     30     50     70     90    100
#   DCGAN-32    4.8    6.4    8.1    9.2    9.2    9.2   min
#   RBM-10bit   4.6    7.6   10.6   13.6   16.6   18.6   min
#   IQP-affine  2.5    2.8    3.1    3.4    3.8    3.9   min
#
# With the three N=100 cells REUSED (copied, not run) and the remaining 15 jobs ordered N
# descending / RBM,DCGAN,IQP within each N (see JOBS below) and packed 8-wide with `wait`
# between waves of 8 (a wave costs its slowest member):
#   wave 1 (8): RBM90(16.6) DCGAN90(9.2) IQP90(3.8) RBM70(13.6) DCGAN70(9.2) IQP70(3.4)
#               RBM50(10.6) DCGAN50(8.1)                                     ~16.6 min <- longest
#   wave 2 (7): IQP50(3.1) RBM30(7.6) DCGAN30(6.4) IQP30(2.8) RBM10(4.6) DCGAN10(4.8) IQP10(2.5)
#                                                                              ~7.6 min
#   + pre-checks (cache + py_compile + test_scaling.py, CPU, ~15 s measured locally) + a
#     5-step CUDA smoke of all three arms (~1 min) + evaluate_r3b over 18 run dirs + 2
#     reference-row kinds (smaller than round 3b's 22-50 dir table, ~1-2 min) + plot_scaling.py
#     (CPU, seconds)
#   TOTAL (three N=100 reused) ~ 16.6 + 7.6 + ~4 ~= 28 min.
#   TOTAL (fallback: all 18 run fresh, reuse verification fails) ~ 34 min (wave 1 becomes
#     RBM100 at 18.6 min instead of RBM90 at 16.6 min).
#   Requesting 2.5 h (150 min) either way -- ~5.3x to ~4.4x headroom.
set -uo pipefail

REPO=<REPO_ROOT>/code
cd "$REPO"
mkdir -p logs_lens

module purge; module load cuda/13.0
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
PY=/usr/bin/python3
OUT="$REPO/outputs_lens_scaling"
R3OUT="$REPO/outputs_lens_r3"
R3BOUT="$REPO/outputs_lens_r3b"
ts() { date +"%Y-%m-%d %H:%M:%S"; }

echo "[SLURM] job=$SLURM_JOB_ID node=$(hostname) part=$SLURM_JOB_PARTITION gpus=$($PY -c 'import torch;print(torch.cuda.device_count())')"

# ---------------------------------------------------------------- fail-fast: dataset + cache
"$PY" -c "import data_lens_r3 as d; m = d.build_cache(); print('[scaling] cache ok, N=11 class counts =', m['train11_class_counts'])" \
    || { echo "[FATAL] the round-3 cache could not be built (dataset missing?). Aborting before using any dev/8gpu budget."; exit 1; }

# ---------------------------------------------------------------- fail-fast pre-checks
# Every file below is READ-ONLY to this script: the three arm scripts and evaluate_r3b.py are
# round 3 / 3b's, unmodified; only the three paper_suite/scaling/ files are new.
echo "[$(ts)] pre-check: py_compile"
for f in codec10.py data_lens_r3.py train_iqp_r3.py baseline_dcgan_r3.py baseline_rbm_r3.py \
         train_iqp_affine_r3.py evaluate_r3.py evaluate_r3b.py \
         paper_suite/scaling/plot_scaling.py paper_suite/scaling/plot_pseudotime.py \
         paper_suite/scaling/test_scaling.py; do
    "$PY" -m py_compile "$f" || { echo "[FATAL] py_compile failed for $f"; exit 1; }
done

echo "[$(ts)] pre-check: paper_suite/scaling/test_scaling.py"
"$PY" paper_suite/scaling/test_scaling.py > logs_lens/precheck_scaling_test.log 2>&1 \
    || { echo "[FATAL] test_scaling.py failed -- see logs_lens/precheck_scaling_test.log"; tail -60 logs_lens/precheck_scaling_test.log; exit 1; }
tail -3 logs_lens/precheck_scaling_test.log

echo "[$(ts)] pre-check: 5-step CUDA smoke of every GPU arm"
rm -rf logs_lens/scaling_cuda_smoke
CUDA_VISIBLE_DEVICES=0 "$PY" baseline_dcgan_r3.py --n-train 3 --n-holdout 4 --n-gen 2 --steps 5 \
    --log-every 5 --device cuda --out logs_lens/scaling_cuda_smoke/dcgan > logs_lens/precheck_scaling_cuda.log 2>&1 \
    || { echo "[FATAL] baseline_dcgan_r3.py does not run on cuda"; tail -30 logs_lens/precheck_scaling_cuda.log; exit 1; }
CUDA_VISIBLE_DEVICES=0 "$PY" baseline_rbm_r3.py --n-train 3 --n-holdout 4 --n-gen 2 --steps 5 \
    --log-every 5 --n-pool 1000 --gen-sweeps 5 --gen-keep 2 --gen-chunk 2 --device cuda \
    --out logs_lens/scaling_cuda_smoke/rbm >> logs_lens/precheck_scaling_cuda.log 2>&1 \
    || { echo "[FATAL] baseline_rbm_r3.py does not run on cuda"; tail -30 logs_lens/precheck_scaling_cuda.log; exit 1; }
CUDA_VISIBLE_DEVICES=0 "$PY" train_iqp_affine_r3.py --M 4 --objective mmdflat --prior resample \
    --n-train 3 --n-holdout 4 --n-gen 2 --steps 5 --theta0-steps 5 --infer-restarts 3 \
    --infer-steps 5 --infer-chunk 2 --log-every 5 --device cuda \
    --out logs_lens/scaling_cuda_smoke/iqp >> logs_lens/precheck_scaling_cuda.log 2>&1 \
    || { echo "[FATAL] train_iqp_affine_r3.py --M 4 --objective mmdflat --prior resample does not run on cuda"; tail -30 logs_lens/precheck_scaling_cuda.log; exit 1; }
rm -rf logs_lens/scaling_cuda_smoke
echo "[$(ts)] pre-checks passed"

rm -rf "$OUT"
mkdir -p "$OUT"

# ---------------------------------------------------------------- reuse N=100 from round 3 / 3b
# See the header comment. Verifies n_train/n_gen/n_holdout/seed/steps/arm in the SOURCE
# meta.json before copying; any mismatch (or a missing/incomplete source) falls back to a
# fresh run of that one (arm, N=100) cell -- never a silent partial copy.
N100_DONE=0
try_reuse() {   # $1=src dir  $2=dest name under $OUT  $3=expected arm  $4=expected steps
    local src="$1" dst="$OUT/$2"
    if [ -f "$src/meta.json" ]; then
        local ok
        ok=$("$PY" -c "
import json
try:
    m = json.load(open('$src/meta.json'))
    print('yes' if (m.get('n_train') == 100 and m.get('n_gen') == 200 and m.get('n_holdout') == 200
                     and m.get('seed') == 0 and m.get('steps') == $4 and m.get('arm') == '$3') else 'no')
except Exception:
    print('no')
")
        local ngen; ngen=$(ls "$src"/generated_*.npy 2>/dev/null | wc -l)
        if [ "$ok" = "yes" ] && [ "$ngen" -eq 200 ]; then
            cp -r "$src" "$dst"
            echo "[$(ts)] REUSE $2 <- $src (config verified identical, complete; saved a GPU job)"
            N100_DONE=$((N100_DONE + 1))
            return 0
        fi
        echo "[$(ts)] NOTE: $src present but not a verified match (ok=$ok ngen=$ngen) -- will run $2 fresh"
        return 1
    fi
    echo "[$(ts)] NOTE: $src has no meta.json -- will run $2 fresh"
    return 1
}
SKIP_DCGAN100=0; SKIP_RBM100=0; SKIP_IQP100=0
try_reuse "$R3OUT/dcgan_n100" "dcgan_n100" "DCGAN-32" 20000 && SKIP_DCGAN100=1
try_reuse "$R3OUT/rbm_n100" "rbm_n100" "RBM-10bit" 20000 && SKIP_RBM100=1
try_reuse "$R3BOUT/iqp_m4_mmdflat_resample_n100" "iqp_n100" "IQP-affine-M4-MMDflat-resample" 4000 && SKIP_IQP100=1
echo "[$(ts)] reused $N100_DONE/3 of the N=100 cells from round 3 / 3b"

# ---------------------------------------------------------------- build the job list: TAG|DEVICE|CMD
JOBS=()
add_job() { JOBS+=("$1|$2|$3"); }

DCGAN_COMMON="--n-holdout 200 --n-gen 200 --device cuda --seed 0"
RBM_COMMON="--n-holdout 200 --n-gen 200 --device cuda --seed 0"
# round 3b's IQP_COMMON (train_iqp_affine_r3.py's own defaults already match every one of
# these; spelled out anyway so the log records what ran, per that script's own convention).
IQP_COMMON="--n-holdout 200 --n-gen 200 --device cuda --seed 0 --lr 0.02 --steps 4000 \
 --theta0-steps 1000 --infer-restarts 8 --infer-steps 1000 --infer-lr 0.005 --infer-chunk 50 \
 --log-every 500"

# N descending, RBM/DCGAN/IQP within each N: front-loads the worst case (a crash or OOM at the
# largest N surfaces in wave 1, not after 20+ minutes of smaller-N jobs) and, since RBM's cost
# grows fastest with N, keeps the two heaviest waves close in size (see the time-budget table).
for N in 90 70 50 30 10; do
    add_job "rbm_n$N" cuda \
        "$PY baseline_rbm_r3.py $RBM_COMMON --n-train $N --steps 20000 --out $OUT/rbm_n$N"
    add_job "dcgan_n$N" cuda \
        "$PY baseline_dcgan_r3.py $DCGAN_COMMON --n-train $N --steps 20000 --out $OUT/dcgan_n$N"
    add_job "iqp_n$N" cuda \
        "$PY train_iqp_affine_r3.py $IQP_COMMON --n-train $N --M 4 --objective mmdflat \
         --prior resample --out $OUT/iqp_n$N"
done
[ "$SKIP_RBM100" -eq 1 ] || add_job "rbm_n100" cuda \
    "$PY baseline_rbm_r3.py $RBM_COMMON --n-train 100 --steps 20000 --out $OUT/rbm_n100"
[ "$SKIP_DCGAN100" -eq 1 ] || add_job "dcgan_n100" cuda \
    "$PY baseline_dcgan_r3.py $DCGAN_COMMON --n-train 100 --steps 20000 --out $OUT/dcgan_n100"
[ "$SKIP_IQP100" -eq 1 ] || add_job "iqp_n100" cuda \
    "$PY train_iqp_affine_r3.py $IQP_COMMON --n-train 100 --M 4 --objective mmdflat \
     --prior resample --out $OUT/iqp_n100"

EXPECTED_JOBS=$((18 - N100_DONE))
echo "[$(ts)] ${#JOBS[@]} jobs queued (expected $EXPECTED_JOBS = 18 - $N100_DONE reused), running 8-wide"
if [ "${#JOBS[@]}" -ne "$EXPECTED_JOBS" ]; then
    echo "[FATAL] job list is ${#JOBS[@]}, expected $EXPECTED_JOBS -- the arm/N grid changed, refusing to run"
    exit 1
fi

# ---------------------------------------------------------------- run 8-wide, wait between batches
GPU_SLOTS=8
i=0
pids=()
tags=()
fail=0
for job in "${JOBS[@]}"; do
    tag="${job%%|*}"
    rest="${job#*|}"
    device="${rest%%|*}"
    cmd="${rest#*|}"
    slot=$((i % GPU_SLOTS))
    echo "[$(ts)] START $tag (slot $slot, device=$device)"
    if [ "$device" = "cuda" ]; then
        (CUDA_VISIBLE_DEVICES=$slot bash -c "$cmd" > "logs_lens/scaling_${tag}.log" 2>&1) &
    else
        (bash -c "$cmd" > "logs_lens/scaling_${tag}.log" 2>&1) &
    fi
    pids+=($!)
    tags[$i]="$tag"
    i=$((i + 1))
    if [ $((i % GPU_SLOTS)) -eq 0 ]; then
        for j in "${!pids[@]}"; do
            wait "${pids[$j]}" || { echo "[WARN] job ${tags[$((i - GPU_SLOTS + j))]} failed, see logs_lens/scaling_${tags[$((i - GPU_SLOTS + j))]}.log"; fail=1; }
        done
        pids=()
    fi
done
for j in "${!pids[@]}"; do
    idx=$((i - ${#pids[@]} + j))
    wait "${pids[$j]}" || { echo "[WARN] job ${tags[$idx]} failed, see logs_lens/scaling_${tags[$idx]}.log"; fail=1; }
done

echo "[$(ts)] all ${#JOBS[@]} jobs finished (fail=$fail); $N100_DONE/3 N=100 cells were reused from round 3/3b"
echo "[$(ts)] run dirs written: $(find "$OUT" -mindepth 1 -maxdepth 1 -type d | wc -l) (expected 18)"

# ---------------------------------------------------------------- evaluate
echo "[$(ts)] START evaluate_r3b over outputs_lens_scaling"
"$PY" evaluate_r3b.py --outputs-dir "$OUT" --n-holdout 200 --n-train-max 100 \
    --random-pair-arm "IQP-affine-M4-MMDflat-resample" --seed 0 --skip-missing \
    --results "$REPO/results_scaling.jsonl" \
    > logs_lens/evaluate_scaling.log 2>&1 \
    || { echo "[WARN] evaluate_r3b failed"; tail -40 logs_lens/evaluate_scaling.log; fail=1; }
tail -5 logs_lens/evaluate_scaling.log

# ---------------------------------------------------------------- plot
echo "[$(ts)] START paper_suite/scaling/plot_scaling.py"
"$PY" paper_suite/scaling/plot_scaling.py --results "$REPO/results_scaling.jsonl" \
    > logs_lens/plot_scaling.log 2>&1 \
    || { echo "[WARN] plot_scaling.py failed"; tail -40 logs_lens/plot_scaling.log; fail=1; }
cat logs_lens/plot_scaling.log

echo "[$(ts)] DONE submit_scaling.sbatch (fail=$fail)"
exit "$fail"
