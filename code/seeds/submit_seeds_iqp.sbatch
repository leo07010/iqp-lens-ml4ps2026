#!/bin/bash
#SBATCH -J iqp_seeds
#SBATCH -A <YOUR_ACCOUNT>
#SBATCH -p dev
#SBATCH --gres=gpu:8
#SBATCH -c 32
#SBATCH --mem=128G
#SBATCH -t 02:30:00
#SBATCH -o <HOME>/gsoc2026-deeplense-qml/experiments/13_iqp_turbulence_replication/logs_lens/sbatch-seeds-iqp-%j.out
#SBATCH -e <HOME>/gsoc2026-deeplense-qml/experiments/13_iqp_turbulence_replication/logs_lens/sbatch-seeds-iqp-%j.err
#
# Seed replicates 1-4 of every IQP number in the ML4PS-2026 paper, so each can be quoted as
# mean +- sd over five seeds (0-4).  Seed 0 already exists (results_r3.jsonl / results_r3b.jsonl
# / results_class.jsonl / results_scaling.jsonl) and is NOT re-run: aggregate_seeds.py copies
# those rows in with seed = 0.
#
# ------------------------------------------------------------------ what a "seed" changes
# NOTHING about the data.  data_lens_r3.load_r3 accepts a `seed` argument for signature
# symmetry but fixes the split at R3_SPLIT_SEED = 0 (data_lens_r3.py:264-267, 295-296), and
# data_lens_class.install derives its per-class split from D.R3_SPLIT_SEED as well
# (data_lens_class.py:128-129).  The training subset, the latent pool and the 200 held-out
# images are therefore bit-identical across seeds, and NO --model-seed clone script is needed:
# every training script's --seed already reaches only
#   * model initialisation      (train_iqp_r3.py:438-446, train_iqp_affine_r3.py:413/455)
#   * mini-batch / restart draws
#   * the latent prior draw     (train_iqp_r3.py:530/579, train_iqp_affine_r3.py:486)
#   * the generation draw       (--- gen_rng = default_rng(seed + 999))
#   * the auto3 kernel bandwidth's Monte-Carlo code sample (train_iqp_r3.py:431-432) --
#     a training-pipeline quantity, re-estimated per seed exactly as seed 0 estimated it.
# The pre-check below asserts the first sentence rather than trusting it.
#
# EVALUATION is identical for every seed: same 200 held-out images, same CONV_SEED = 0
# (evaluate_r3.py:80), same evaluate_r3b scorer, same --seed 0 for the random-pairing
# reference row.  evaluate_r3b.score_run IS evaluate_r3.score_run, verified to reproduce
# evaluate_r3's numbers bit-for-bit on the arms both scored (RBM/DCGAN at N = 100).
#
# ------------------------------------------------------------------------------ job list
# 4 seeds x 13 pooled arms + 4 seeds x 6 per-class arms = 76 runs.
#
#   tag                              | script                    | table it feeds
#   ---------------------------------+---------------------------+---------------------------
#   iqp_paper_n{11,100}              | train_iqp_r3.py           | Table 2 row 1, and the
#                                    |   --protocol paper        |   frozen-core diagnostic
#                                    |                           |   of section 3.2
#   iqp_resample_n{11,100}           | train_iqp_r3.py           | Table 2 row 2
#                                    |   --protocol joint        |
#   iqp_m4_mmdflat_resample_n{10,11, | train_iqp_affine_r3.py    | Table 2 row 3 (N = 11, 100)
#     30,50,70,90,100}               |   --M 4 --objective       |   AND Fig. 5 scaling
#                                    |   mmdflat --prior resample|   (N = 10..100)
#   iqp_m4_auto3_resample_n{11,100}  | train_iqp_affine_r3.py    | Table 2 row 4
#   {axion,cdm,no_sub}_m4_mmdflat_   | train_iqp_affine_class.py | Table 3
#     resample_n{11,100}             |                           |
#
# The M = 4 flat arm is run ONCE per (N, seed) and feeds both Table 2 and Fig. 5.  That is not
# a shortcut: at seed 0 round 3b's N = 100 run and the scaling job's N = 100 run produced
# bit-identical pdf_js / conv_mmd2 / conv_mmd2_aligned / projection_corr_mean, so the two
# tables were already reading the same computation twice.
#
# RBM and DCGAN are NOT here.  Their step count is being re-derived from a convergence sweep
# (results_convergence.jsonl); seeds/submit_seeds_baselines.sbatch is written and waiting for
# that number.
#
# ------------------------------------------------------------------------- time budget
# MEASURED, seed 0, from logs_lens/r3_*.log and logs_lens/r3b_*.log (one GPU each):
#   iqp_paper_n100        2150 s   <- the critical path
#   iqp_paper_n11          386 s
#   iqp_resample_n100      143 s
#   iqp_resample_n11        ~60 s
#   iqp_m4_* (any N)      65-70 s
#   per-class m4            65 s
# Total GPU-seconds ~ 14,900 = 4.1 GPU-hours; on 8 GPUs with a self-balancing work queue the
# wall clock is bounded below by the longest single job, 2150 s = 36 min, and above by
# 14,900 / 8 + 2150 = 67 min.  Plus 5 evaluations (~2 min each) and the two diagnostics
# (5 x 53 s single-crop + seconds for frozen-core).  Requesting 2.5 h.
#
# Scheduling: a flock-guarded work queue rather than round 3's fixed batches of 8.  With four
# 36-minute jobs among 76, fixed batches would idle four GPUs for 30 minutes in batch 1.
#
# Partition: dev (4 h limit, 2.5 h requested).  Rounds 3 and 3b used 8gpus, but that
# partition's QOS caps submissions at 5 per user and this account already has 20 jobs on it,
# so 8gpus rejects the submission outright (QOSMaxSubmitJobPerUserLimit).  dev grants the
# same --gres=gpu:8 and its 4 h ceiling leaves 1.5 h of headroom over the measured budget.
# ---------------------------------------------------------------------------------------
set -u

ts() { date "+%F %T"; }
REPO=<HOME>/gsoc2026-deeplense-qml/experiments/13_iqp_turbulence_replication
cd "$REPO" || exit 1
PY=/usr/bin/python3
OUT_ROOT="$REPO/outputs_seeds"
RAW="$REPO/results_seeds_raw"
SEEDS=(1 2 3 4)
ALL_SEEDS=(0 1 2 3 4)
SCALING_N=(10 30 50 70 90)          # 100 and 11 come from the Table-2 cells of the same arm
CLASSES=(axion cdm no_sub)

mkdir -p logs_lens "$OUT_ROOT" "$RAW"
echo "[$(ts)] START submit_seeds_iqp.sbatch  job=${SLURM_JOB_ID:-none}  node=$(hostname)"
echo "[$(ts)] seeds: ${SEEDS[*]} (seed 0 is reused from the existing results files)"

# ---------------------------------------------------------------- fail-fast: syntax
echo "[$(ts)] pre-check: py_compile"
for f in train_iqp_r3.py train_iqp_affine_r3.py train_iqp_affine_class.py data_lens_r3.py \
         data_lens_class.py evaluate_r3.py evaluate_r3b.py evaluate_class.py \
         seeds/diag_single_crop_seeds.py seeds/diag_frozen_core_seeds.py aggregate_seeds.py; do
    "$PY" -m py_compile "$f" || { echo "[FATAL] py_compile failed for $f"; exit 1; }
done

# ---------------------------------------------------------------- fail-fast: flock
command -v flock >/dev/null 2>&1 || { echo "[FATAL] flock not available, the work queue needs it"; exit 1; }

# ---------------------------------------------------------------- fail-fast: the seed claim
# The whole design rests on "--seed does not move the data".  Assert it instead of trusting
# the docstring: load the split under three different seed arguments and byte-compare.
echo "[$(ts)] pre-check: the data split is invariant under --seed"
"$PY" - <<'PYEOF' || { echo "[FATAL] the data split MOVES with --seed -- stop, the design is wrong"; exit 1; }
import sys
sys.path.insert(0, "<HOME>/gsoc2026-deeplense-qml/experiments/13_iqp_turbulence_replication")
import numpy as np
import data_lens_r3, data_lens_class
ref = data_lens_r3.load_r3(100, seed=0, n_holdout=200)
for s in (1, 2, 3, 4):
    d = data_lens_r3.load_r3(100, seed=s, n_holdout=200)
    for name in ("train_images", "train_crops", "train_pmfs", "train_offsets", "train_totals",
                 "holdout_images", "holdout_crops", "holdout_offsets", "holdout_totals"):
        a, b = getattr(ref, name), getattr(d, name)
        if not np.array_equal(np.asarray(a), np.asarray(b)):
            raise SystemExit("load_r3.%s changed at seed=%d" % (name, s))
    if float(ref.S) != float(d.S):
        raise SystemExit("load_r3.S changed at seed=%d" % s)
pool = data_lens_r3.latent_pool()
print("[precheck] load_r3 identical for seed in 0..4 "
      "(train %s, holdout %s, latent pool %s, S=%.6f)"
      % (ref.train_images.shape, ref.holdout_images.shape, pool.shape, ref.S))
for c in ("axion", "cdm", "no_sub"):
    m = data_lens_class.class_meta(c)
    print("[precheck] class %-7s split_seed=%s rule=%r" % (c, m.get("split_seed"), m.get("split_rule")))
PYEOF

# ---------------------------------------------------------------- fail-fast: class caches
echo "[$(ts)] pre-check: the three per-class caches exist"
"$PY" data_lens_class.py || { echo "[FATAL] the per-class caches could not be built"; exit 1; }

# ---------------------------------------------------------------- fail-fast: CUDA smoke
echo "[$(ts)] pre-check: 5-step CUDA smoke of all three training scripts at a non-zero seed"
rm -rf logs_lens/seeds_cuda_smoke
CUDA_VISIBLE_DEVICES=0 "$PY" train_iqp_r3.py --n-train 3 --n-holdout 4 --n-gen 2 --seed 7 \
    --protocol paper --steps-core 5 --steps-lat 5 --log-every 5 --device cuda \
    --out logs_lens/seeds_cuda_smoke/paper > logs_lens/precheck_seeds_cuda.log 2>&1 \
    || { echo "[FATAL] train_iqp_r3.py --protocol paper does not run"; tail -30 logs_lens/precheck_seeds_cuda.log; exit 1; }
CUDA_VISIBLE_DEVICES=0 "$PY" train_iqp_r3.py --n-train 3 --n-holdout 4 --n-gen 2 --seed 7 \
    --protocol joint --objective mmd --prior resample --steps 5 --log-every 5 --device cuda \
    --out logs_lens/seeds_cuda_smoke/joint >> logs_lens/precheck_seeds_cuda.log 2>&1 \
    || { echo "[FATAL] train_iqp_r3.py --protocol joint does not run"; tail -30 logs_lens/precheck_seeds_cuda.log; exit 1; }
CUDA_VISIBLE_DEVICES=0 "$PY" train_iqp_affine_r3.py --n-train 3 --n-holdout 4 --n-gen 2 --seed 7 \
    --M 4 --objective mmdflat --prior resample --steps 5 --theta0-steps 5 --infer-restarts 2 \
    --infer-steps 5 --infer-chunk 2 --log-every 5 --device cuda \
    --out logs_lens/seeds_cuda_smoke/affine >> logs_lens/precheck_seeds_cuda.log 2>&1 \
    || { echo "[FATAL] train_iqp_affine_r3.py does not run"; tail -30 logs_lens/precheck_seeds_cuda.log; exit 1; }
CUDA_VISIBLE_DEVICES=0 "$PY" train_iqp_affine_class.py --class-name axion --n-train 3 \
    --n-holdout 4 --n-gen 2 --seed 7 --M 4 --objective mmdflat --prior resample --steps 5 \
    --theta0-steps 5 --infer-restarts 2 --infer-steps 5 --infer-chunk 2 --log-every 5 \
    --device cuda --out logs_lens/seeds_cuda_smoke/cls >> logs_lens/precheck_seeds_cuda.log 2>&1 \
    || { echo "[FATAL] train_iqp_affine_class.py does not run"; tail -30 logs_lens/precheck_seeds_cuda.log; exit 1; }
rm -rf logs_lens/seeds_cuda_smoke
echo "[$(ts)] pre-checks passed"

# ---------------------------------------------------------------- build the job list
# Ordered LONGEST FIRST so the greedy work queue approximates longest-processing-time-first.
JOBS=()
add_job() { JOBS+=("$1|$2"); }

for S in "${SEEDS[@]}"; do
    O="$OUT_ROOT/s$S"
    rm -rf "$O"; mkdir -p "$O"
    R3="--n-holdout 200 --n-gen 200 --device cuda --sigmas auto3 --seed $S"
    add_job "s${S}_iqp_paper_n100" \
        "$PY train_iqp_r3.py $R3 --n-train 100 --latent lowvar --protocol paper \
         --objective mmd --steps-core 30000 --steps-lat 3000 --out $O/iqp_paper_n100"
done
for S in "${SEEDS[@]}"; do
    O="$OUT_ROOT/s$S"
    R3="--n-holdout 200 --n-gen 200 --device cuda --sigmas auto3 --seed $S"
    add_job "s${S}_iqp_paper_n11" \
        "$PY train_iqp_r3.py $R3 --n-train 11 --latent lowvar --protocol paper \
         --objective mmd --steps-core 30000 --steps-lat 3000 --out $O/iqp_paper_n11"
done
for S in "${SEEDS[@]}"; do
    O="$OUT_ROOT/s$S"
    R3="--n-holdout 200 --n-gen 200 --device cuda --sigmas auto3 --seed $S"
    add_job "s${S}_iqp_resample_n100" \
        "$PY train_iqp_r3.py $R3 --n-train 100 --latent lowvar --protocol joint \
         --objective mmd --prior resample --steps 20000 --out $O/iqp_resample_n100"
    add_job "s${S}_iqp_resample_n11" \
        "$PY train_iqp_r3.py $R3 --n-train 11 --latent lowvar --protocol joint \
         --objective mmd --prior resample --steps 20000 --out $O/iqp_resample_n11"
done
for S in "${SEEDS[@]}"; do
    O="$OUT_ROOT/s$S"
    R3B="--n-holdout 200 --n-gen 200 --device cuda --seed $S --lr 0.02 --steps 4000 \
 --theta0-steps 1000 --infer-restarts 8 --infer-steps 1000 --infer-lr 0.005 --infer-chunk 50 \
 --log-every 500"
    for N in 100 11 "${SCALING_N[@]}"; do
        add_job "s${S}_iqp_m4_mmdflat_resample_n$N" \
            "$PY train_iqp_affine_r3.py $R3B --n-train $N --M 4 --objective mmdflat \
             --prior resample --out $O/iqp_m4_mmdflat_resample_n$N"
    done
    for N in 100 11; do
        add_job "s${S}_iqp_m4_auto3_resample_n$N" \
            "$PY train_iqp_affine_r3.py $R3B --n-train $N --M 4 --objective auto3 \
             --prior resample --sigmas auto3 --out $O/iqp_m4_auto3_resample_n$N"
    done
done
for S in "${SEEDS[@]}"; do
    OC="$OUT_ROOT/cls_s$S"
    rm -rf "$OC"; mkdir -p "$OC"
    CLS="--n-holdout 200 --n-gen 200 --device cuda --seed $S --lr 0.02 --steps 4000 \
 --theta0-steps 1000 --infer-restarts 8 --infer-steps 1000 --infer-lr 0.005 --infer-chunk 50 \
 --log-every 500 --M 4 --objective mmdflat --prior resample"
    for N in 100 11; do
        for C in "${CLASSES[@]}"; do
            add_job "s${S}_${C}_m4_mmdflat_resample_n$N" \
                "$PY train_iqp_affine_class.py --class-name $C $CLS --n-train $N \
                 --out $OC/${C}_m4_mmdflat_resample_n$N"
        done
    done
done

NJOBS=${#JOBS[@]}
EXPECTED=$(( ${#SEEDS[@]} * (13 + 6) ))
echo "[$(ts)] $NJOBS jobs queued (expected $EXPECTED = ${#SEEDS[@]} seeds x (13 pooled + 6 per-class))"
if [ "$NJOBS" -ne "$EXPECTED" ]; then
    echo "[FATAL] job list is $NJOBS, expected $EXPECTED -- the arm matrix changed, refusing to run"
    exit 1
fi

# ---------------------------------------------------------------- run: 8-slot work queue
QUEUE="$OUT_ROOT/queue.txt"
IDXFILE="$OUT_ROOT/next_idx"
LOCKFILE="$OUT_ROOT/queue.lock"
FAILFILE="$OUT_ROOT/failed.txt"
printf '%s\n' "${JOBS[@]}" > "$QUEUE"
echo 0 > "$IDXFILE"
: > "$LOCKFILE"
: > "$FAILFILE"

worker() {
    slot="$1"
    while true; do
        idx=$(flock "$LOCKFILE" -c "i=\$(cat '$IDXFILE'); echo \$((i+1)) > '$IDXFILE'; echo \$i")
        if [ -z "$idx" ] || [ "$idx" -ge "$NJOBS" ]; then break; fi
        line=$(sed -n "$((idx + 1))p" "$QUEUE")
        tag="${line%%|*}"
        cmd="${line#*|}"
        t_start=$(date +%s)
        echo "[$(ts)] slot $slot START [$((idx + 1))/$NJOBS] $tag"
        if CUDA_VISIBLE_DEVICES="$slot" bash -c "$cmd" > "logs_lens/seeds_${tag}.log" 2>&1; then
            echo "[$(ts)] slot $slot OK    $tag ($(( $(date +%s) - t_start )) s)"
        else
            echo "[$(ts)] slot $slot FAIL  $tag ($(( $(date +%s) - t_start )) s) -- logs_lens/seeds_${tag}.log"
            tail -15 "logs_lens/seeds_${tag}.log"
            echo "$tag" >> "$FAILFILE"
        fi
    done
}

GPU_SLOTS=8
for slot in $(seq 0 $((GPU_SLOTS - 1))); do worker "$slot" & done
wait
NFAIL=$(wc -l < "$FAILFILE")
echo "[$(ts)] all $NJOBS training jobs finished, $NFAIL failed"
[ "$NFAIL" -gt 0 ] && { echo "[$(ts)] failed tags:"; cat "$FAILFILE"; }
fail=$([ "$NFAIL" -gt 0 ] && echo 1 || echo 0)

for S in "${SEEDS[@]}"; do
    got=$(find "$OUT_ROOT/s$S" -mindepth 2 -maxdepth 2 -name meta.json 2>/dev/null | wc -l)
    gotc=$(find "$OUT_ROOT/cls_s$S" -mindepth 2 -maxdepth 2 -name meta.json 2>/dev/null | wc -l)
    echo "[$(ts)] seed $S: $got/13 pooled run dirs, $gotc/6 per-class run dirs"
done

# ---------------------------------------------------------------- evaluate, one pass per seed
# Identical protocol to seed 0: same 200 held-out images, same CONV_SEED, evaluator --seed 0
# for the random-pairing reference row (the reference must not move with the arm's seed).
for S in "${SEEDS[@]}"; do
    echo "[$(ts)] START evaluate_r3b for seed $S"
    "$PY" evaluate_r3b.py --outputs-dir "$OUT_ROOT/s$S" --n-holdout 200 --n-train-max 100 \
        --random-pair-arm "IQP-affine-M4-MMDflat-resample" --seed 0 --skip-missing \
        --results "$RAW/pooled_s$S.jsonl" > "logs_lens/seeds_evaluate_pooled_s$S.log" 2>&1 \
        || { echo "[WARN] evaluate_r3b failed for seed $S"; tail -30 "logs_lens/seeds_evaluate_pooled_s$S.log"; fail=1; }
    tail -1 "logs_lens/seeds_evaluate_pooled_s$S.log"

    echo "[$(ts)] START evaluate_class for seed $S"
    "$PY" evaluate_class.py --outputs-dir "$OUT_ROOT/cls_s$S" --classes "${CLASSES[@]}" \
        --n-holdout 200 --n-train-max 100 --seed 0 --results "$RAW/class_s$S.jsonl" \
        > "logs_lens/seeds_evaluate_class_s$S.log" 2>&1 \
        || { echo "[WARN] evaluate_class failed for seed $S"; tail -30 "logs_lens/seeds_evaluate_class_s$S.log"; fail=1; }
    tail -1 "logs_lens/seeds_evaluate_class_s$S.log"
done

# ---------------------------------------------------------------- section 3.2 diagnostics
# (a) single-crop full-parameter fit, re-run at all five seeds.  Seed 0 must reproduce
#     mech_tests/mech_test/results.json exactly (MMD-flat 0.999972, MMD-auto3 0.996214) --
#     that is this script's own correctness check, printed below.
# (b) frozen-core reconstruction of the other training images: a forward pass over the
#     paper-protocol run dirs, seed 0's from outputs_lens_r3, seeds 1-4's from this job.
DIAG="$REPO/results_seeds_diag.jsonl"
: > "$DIAG"
echo "[$(ts)] START section-3.2 diagnostics"
for S in "${ALL_SEEDS[@]}"; do
    OMP_NUM_THREADS=4 "$PY" -u seeds/diag_single_crop_seeds.py --seed "$S" --results "$DIAG" \
        >> logs_lens/seeds_diag_single_crop.log 2>&1 \
        || { echo "[WARN] diag_single_crop_seeds.py failed at seed $S"; tail -20 logs_lens/seeds_diag_single_crop.log; fail=1; }
done
grep -E "^  \[diag-a" logs_lens/seeds_diag_single_crop.log || true

PAPER_DIRS=("$REPO/outputs_lens_r3/iqp_paper_n11" "$REPO/outputs_lens_r3/iqp_paper_n100")
for S in "${SEEDS[@]}"; do
    PAPER_DIRS+=("$OUT_ROOT/s$S/iqp_paper_n11" "$OUT_ROOT/s$S/iqp_paper_n100")
done
OMP_NUM_THREADS=4 "$PY" -u seeds/diag_frozen_core_seeds.py --run-dir "${PAPER_DIRS[@]}" \
    --results "$DIAG" > logs_lens/seeds_diag_frozen_core.log 2>&1 \
    || { echo "[WARN] diag_frozen_core_seeds.py failed"; tail -20 logs_lens/seeds_diag_frozen_core.log; fail=1; }
cat logs_lens/seeds_diag_frozen_core.log

# ---------------------------------------------------------------- merge + aggregate
echo "[$(ts)] START aggregate_seeds.py"
"$PY" -u aggregate_seeds.py > logs_lens/seeds_aggregate.log 2>&1 \
    || { echo "[WARN] aggregate_seeds.py failed"; tail -40 logs_lens/seeds_aggregate.log; fail=1; }
tail -40 logs_lens/seeds_aggregate.log

echo "[$(ts)] DONE submit_seeds_iqp.sbatch (fail=$fail)"
exit "$fail"
