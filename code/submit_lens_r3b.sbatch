#!/bin/bash
#SBATCH -J iqp_lens_r3b
#SBATCH -A <YOUR_ACCOUNT>
#SBATCH -p 8gpus
#SBATCH --gres=gpu:8
#SBATCH -c 32
#SBATCH --mem=128G
#SBATCH -t 02:00:00
#SBATCH -o <REPO_ROOT>/code/logs_lens/sbatch-r3b-%j.out
#SBATCH -e <REPO_ROOT>/code/logs_lens/sbatch-r3b-%j.err
#
# Round 3b of the IQP generative study on DeepLense Model II: LEARNED latent directions.
# Pre-registration: docs/IQP_LENS_ROUND3_PREREGISTRATION.md, section "Round 3b" plus the
# "Round 3b amendment (2026-09-07)" that fixes the arm grid from the completed mechanism test.
# Written and checked BEFORE this ran; predictions P18-P23 and the decision rule are in it.
#
# Round 3b touches NOTHING round 3 uses. Round 3's own 28-job sbatch is queued against
# train_iqp_r3.py / controls_r3.py / evaluate_r3.py / data_lens_r3.py / codec10.py /
# test_r3.py; this script only calls the three NEW files (train_iqp_affine_r3.py,
# controls_pca_r3.py, evaluate_r3b.py), which IMPORT those and modify none of them. The
# output tree (outputs_lens_r3b), the results file (results_r3b.jsonl) and every log name
# (r3b_*) are separate, so the two rounds can run at the same time on different nodes.
#
# ---------------------------------------------------------------------------- job list
# 11 arms x N in {11, 100} = 22 jobs, classes POOLED (as in round 3).
#
#   arm label                       | script                  | what it is
#   --------------------------------+-------------------------+-------------------------------
#   IQP-affine-M2-KL-gauss          | train_iqp_affine_r3.py  | GENERATION working point: the
#   IQP-affine-M2-KL-resample       |                         | mechanism test's 100%-ring M,
#                                   |                         | two priors (P19)
#   IQP-affine-M2-auto3-gauss       | train_iqp_affine_r3.py  | dispersion check: the paper's
#                                   |                         | kernel traded corr for e1 sd
#   IQP-affine-M4-KL-gauss          | train_iqp_affine_r3.py  | PROJECTION working point (0.911
#   IQP-affine-M4-KL-resample       |                         | in the mechanism test), 2 priors
#   IQP-affine-M4-MMDflat-resample  | train_iqp_affine_r3.py  | hardware-compatible objective
#   IQP-affine-M4-auto3-resample    | train_iqp_affine_r3.py  | the paper's kernel (P21)
#   IQP-affine-M8-KL-zpen-resample  | train_iqp_affine_r3.py  | capacity test (P23): does a 0.3
#                                   |                         | z-penalty lift the M = 8 ceiling
#                                   |                         | that ||z|| > coherence radius set
#   CTRL-pca2-gauss                 | controls_pca_r3.py      | the classical LINEAR analogue at
#   CTRL-pca4-gauss                 |                         | the same M and the same priors --
#   CTRL-pca4-resample              |                         | the arm the decision rule names
#
# Every arm writes round 3's output contract (generated_%03d.npy + crops/ + meta.json), is
# scored by the same evaluate_r3 functions against the same 200 held-out images, and reports
# the round-3b extra: held-out PROJECTION corr (best reconstruction reachable inside its own
# latent space, z optimised with everything else frozen; closed form for the PCA arms).
#
# ------------------------------------------------------------------------- time budget
# MEASURED on the login node, 2026-09-07, /usr/bin/python3, 4 torch threads, n = 10, float64:
#
#   theta0 seed (batch 1)                       2.82 ms/step  x 1,000 =    2.8 s
#   joint step  N = 11   M = 4  KL              4.17 ms/step  x 4,000 =   17 s
#   joint step  N = 11   M = 4  MMDflat/auto3   4.89 ms/step  x 4,000 =   20 s
#   joint step  N = 100  M = 4  KL             18.55 ms/step  x 4,000 =   74 s
#   joint step  N = 100  M = 4  MMDflat/auto3  25.55 ms/step  x 4,000 =  102 s
#   joint step  N = 100  M = 8  KL             18.89 ms/step  x 4,000 =   76 s
#   held-out projection: 200 images x 8 restarts = 1,600 rows, run as 4 chunks of 400 rows,
#                        30.34 ms/step x 1,000 steps x 4                =  121 s   (every arm)
#   data load per job (N = 100 + 200 held-out crops)                    =  ~10 s
#   PCA control (SVD of 100 x 1,024 + 200 closed-form projections)      =   < 5 s
#
# => per job, CPU numbers used as the CONSERVATIVE bound (the FWHT butterfly at n = 10 is
#    launch-bound on GPU; the batched joint/projection steps should be faster on GPU, never
#    slower -- and the 5-step CUDA smoke below is what checks that assumption did not turn
#    into a crash):
#
#      IQP N = 100 MMDflat/auto3   ~ 3.9 min   <- longest
#      IQP N = 100 KL              ~ 3.4 min
#      IQP N = 11  any objective   ~ 2.5 min
#      the six PCA controls        < 0.5 min each at either N
#
# Jobs run 8-wide with `wait` between batches (a batch costs its slowest member):
#   batch 1 (8)  the eight IQP arms at N = 100                          ~ 3.9 min
#   batch 2 (8)  the eight IQP arms at N = 11                           ~ 2.5 min
#   batch 3 (6)  the six PCA controls                                   ~ 0.5 min
#   + pre-checks (py_compile + test_r3b.py, MEASURED at 9.3 s wall) + a 5-step CUDA smoke
#     (~1 min) + evaluate_r3b over 22 (or 50, if round 3 has finished) run dirs with
#     conv_mmd2 computed twice per row (~2-4 min)
#   TOTAL ~ 13 min at the conservative CPU bound.  Requesting 2 h (~9x headroom).
set -uo pipefail

REPO=<REPO_ROOT>/code
cd "$REPO"
mkdir -p logs_lens

module purge; module load cuda/13.0
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
PY=/usr/bin/python3
OUT="$REPO/outputs_lens_r3b"
R3OUT="$REPO/outputs_lens_r3"
ts() { date +"%Y-%m-%d %H:%M:%S"; }

echo "[SLURM] job=$SLURM_JOB_ID node=$(hostname) part=$SLURM_JOB_PARTITION gpus=$($PY -c 'import torch;print(torch.cuda.device_count())')"

# ---------------------------------------------------------------- fail-fast: dataset + cache
"$PY" -c "import data_lens_r3 as d; m = d.build_cache(); print('[r3b] cache ok, N=11 class counts =', m['train11_class_counts'], 'holdout halves =', m['holdout_firsthalf_class_counts'], m['holdout_secondhalf_class_counts'])" \
    || { echo "[FATAL] the round-3 cache could not be built (dataset missing?). Aborting before using any 8gpus budget."; exit 1; }

# ---------------------------------------------------------------- fail-fast pre-checks
# Only the THREE NEW files are compiled: round 3's scripts belong to a queued job and this
# script must not touch them (importing them is read-only and is exercised by test_r3b.py).
echo "[$(ts)] pre-check: py_compile the round-3b files"
for f in train_iqp_affine_r3.py controls_pca_r3.py evaluate_r3b.py test_r3b.py; do
    "$PY" -m py_compile "$f" || { echo "[FATAL] py_compile failed for $f"; exit 1; }
done

echo "[$(ts)] pre-check: test_r3b.py"
"$PY" test_r3b.py > logs_lens/precheck_r3b_test.log 2>&1 \
    || { echo "[FATAL] test_r3b.py failed -- see logs_lens/precheck_r3b_test.log"; tail -40 logs_lens/precheck_r3b_test.log; exit 1; }
tail -3 logs_lens/precheck_r3b_test.log

echo "[$(ts)] pre-check: 5-step CUDA smoke of the GPU arm"
rm -rf logs_lens/r3b_cuda_smoke
CUDA_VISIBLE_DEVICES=0 "$PY" train_iqp_affine_r3.py --M 4 --objective kl --prior gauss \
    --n-train 3 --n-holdout 4 --n-gen 2 --steps 5 --theta0-steps 5 --infer-restarts 3 \
    --infer-steps 5 --infer-chunk 2 --log-every 5 --device cuda \
    --out logs_lens/r3b_cuda_smoke/kl > logs_lens/precheck_r3b_cuda.log 2>&1 \
    || { echo "[FATAL] train_iqp_affine_r3.py does not run on cuda"; tail -30 logs_lens/precheck_r3b_cuda.log; exit 1; }
CUDA_VISIBLE_DEVICES=0 "$PY" train_iqp_affine_r3.py --M 8 --objective auto3 --prior resample \
    --zpen 0.3 --n-train 3 --n-holdout 4 --n-gen 2 --steps 5 --theta0-steps 5 \
    --infer-restarts 3 --infer-steps 5 --infer-chunk 2 --log-every 5 --device cuda \
    --out logs_lens/r3b_cuda_smoke/auto3 >> logs_lens/precheck_r3b_cuda.log 2>&1 \
    || { echo "[FATAL] --objective auto3 / --zpen does not run on cuda"; tail -30 logs_lens/precheck_r3b_cuda.log; exit 1; }
rm -rf logs_lens/r3b_cuda_smoke
echo "[$(ts)] pre-checks passed"

rm -rf "$OUT"
mkdir -p "$OUT"

# ---------------------------------------------------------------- build the job list: TAG|DEVICE|CMD
JOBS=()
add_job() { JOBS+=("$1|$2|$3"); }

# --lr 0.02, --steps 4000, --infer-steps 1000 at --infer-lr 0.005, 8 restarts: the amendment's
# settings, spelled out here rather than left to the defaults so the log records what ran.
IQP_COMMON="--n-holdout 200 --n-gen 200 --device cuda --seed 0 --lr 0.02 --steps 4000 \
 --theta0-steps 1000 --infer-restarts 8 --infer-steps 1000 --infer-lr 0.005 --infer-chunk 50 \
 --log-every 500"
CTL_COMMON="--n-holdout 200 --n-gen 200 --seed 0"

# --- batch 1: the eight IQP arms at N = 100 (the longest) ------------------------------
for N in 100 11; do
    add_job "iqp_m2_kl_gauss_n$N" cuda \
        "$PY train_iqp_affine_r3.py $IQP_COMMON --n-train $N --M 2 --objective kl \
         --prior gauss --out $OUT/iqp_m2_kl_gauss_n$N"
    add_job "iqp_m2_kl_resample_n$N" cuda \
        "$PY train_iqp_affine_r3.py $IQP_COMMON --n-train $N --M 2 --objective kl \
         --prior resample --out $OUT/iqp_m2_kl_resample_n$N"
    add_job "iqp_m2_auto3_gauss_n$N" cuda \
        "$PY train_iqp_affine_r3.py $IQP_COMMON --n-train $N --M 2 --objective auto3 \
         --prior gauss --sigmas auto3 --out $OUT/iqp_m2_auto3_gauss_n$N"
    add_job "iqp_m4_kl_gauss_n$N" cuda \
        "$PY train_iqp_affine_r3.py $IQP_COMMON --n-train $N --M 4 --objective kl \
         --prior gauss --out $OUT/iqp_m4_kl_gauss_n$N"
    add_job "iqp_m4_kl_resample_n$N" cuda \
        "$PY train_iqp_affine_r3.py $IQP_COMMON --n-train $N --M 4 --objective kl \
         --prior resample --out $OUT/iqp_m4_kl_resample_n$N"
    add_job "iqp_m4_mmdflat_resample_n$N" cuda \
        "$PY train_iqp_affine_r3.py $IQP_COMMON --n-train $N --M 4 --objective mmdflat \
         --prior resample --out $OUT/iqp_m4_mmdflat_resample_n$N"
    add_job "iqp_m4_auto3_resample_n$N" cuda \
        "$PY train_iqp_affine_r3.py $IQP_COMMON --n-train $N --M 4 --objective auto3 \
         --prior resample --sigmas auto3 --out $OUT/iqp_m4_auto3_resample_n$N"
    # the capacity arm: z-penalty 0.3 instead of 1e-3, hence the -zpen tag in its arm name
    add_job "iqp_m8_kl_zpen_resample_n$N" cuda \
        "$PY train_iqp_affine_r3.py $IQP_COMMON --n-train $N --M 8 --objective kl \
         --zpen 0.3 --prior resample --out $OUT/iqp_m8_kl_zpen_resample_n$N"
done

# --- batch 3: the six PCA controls (numpy only, no GPU) --------------------------------
for N in 100 11; do
    add_job "pca2_gauss_n$N" cpu \
        "$PY controls_pca_r3.py $CTL_COMMON --n-train $N --M 2 --prior gauss \
         --out $OUT/pca2_gauss_n$N"
    add_job "pca4_gauss_n$N" cpu \
        "$PY controls_pca_r3.py $CTL_COMMON --n-train $N --M 4 --prior gauss \
         --out $OUT/pca4_gauss_n$N"
    add_job "pca4_resample_n$N" cpu \
        "$PY controls_pca_r3.py $CTL_COMMON --n-train $N --M 4 --prior resample \
         --out $OUT/pca4_resample_n$N"
done

echo "[$(ts)] ${#JOBS[@]} jobs queued (expected 22 = 11 arms x 2 N), running 8-wide"
if [ "${#JOBS[@]}" -ne 22 ]; then
    echo "[FATAL] job list is ${#JOBS[@]}, expected 22 -- the arm matrix changed, refusing to run"
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
        (CUDA_VISIBLE_DEVICES=$slot bash -c "$cmd" > "logs_lens/r3b_${tag}.log" 2>&1) &
    else
        (bash -c "$cmd" > "logs_lens/r3b_${tag}.log" 2>&1) &
    fi
    pids+=($!)
    tags[$i]="$tag"
    i=$((i + 1))
    if [ $((i % GPU_SLOTS)) -eq 0 ]; then
        for j in "${!pids[@]}"; do
            wait "${pids[$j]}" || { echo "[WARN] job ${tags[$((i - GPU_SLOTS + j))]} failed, see logs_lens/r3b_${tags[$((i - GPU_SLOTS + j))]}.log"; fail=1; }
        done
        pids=()
    fi
done
for j in "${!pids[@]}"; do
    idx=$((i - ${#pids[@]} + j))
    wait "${pids[$j]}" || { echo "[WARN] job ${tags[$idx]} failed, see logs_lens/r3b_${tags[$idx]}.log"; fail=1; }
done

echo "[$(ts)] all ${#JOBS[@]} jobs finished (fail=$fail)"
echo "[$(ts)] run dirs written: $(find "$OUT" -mindepth 1 -maxdepth 1 -type d | wc -l) (expected 22)"

# ---------------------------------------------------------------- evaluate
# Round 3's tree is merged into the table ONLY if it is complete (28 run dirs, each with a
# meta.json). Round 3's own job may be running right now, and scoring a half-written tree
# would put partial arms in the same table as finished ones -- the unfair comparison the
# pre-registration's "mark incomplete runs, do not compare them" rule exists to prevent.
R3_DIRS=0
if [ -d "$R3OUT" ]; then
    R3_DIRS=$(find "$R3OUT" -mindepth 2 -maxdepth 2 -name meta.json | wc -l)
fi
if [ "$R3_DIRS" -eq 28 ]; then
    EVAL_DIRS=("$OUT" "$R3OUT")
    echo "[$(ts)] round 3 is complete ($R3_DIRS run dirs) -- scoring rounds 3 and 3b in ONE table"
else
    EVAL_DIRS=("$OUT")
    echo "[$(ts)] round 3 has $R3_DIRS/28 finished run dirs -- scoring round 3b ALONE (merge later with:"
    echo "        $PY evaluate_r3b.py --outputs-dir $OUT $R3OUT --results $REPO/results_r3b.jsonl )"
fi

echo "[$(ts)] START evaluate_r3b"
"$PY" evaluate_r3b.py --outputs-dir "${EVAL_DIRS[@]}" --n-holdout 200 --n-train-max 100 \
    --random-pair-arm "IQP-affine-M4-KL-gauss" --seed 0 --skip-missing \
    --results "$REPO/results_r3b.jsonl" \
    > logs_lens/evaluate_r3b.log 2>&1 \
    || { echo "[WARN] evaluate_r3b failed"; tail -40 logs_lens/evaluate_r3b.log; fail=1; }
cat logs_lens/evaluate_r3b.log

echo "[$(ts)] DONE submit_lens_r3b.sbatch (fail=$fail)"
exit "$fail"
