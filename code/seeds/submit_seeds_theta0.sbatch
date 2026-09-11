#!/bin/bash
#SBATCH -J iqp_seeds_theta0
#SBATCH -A <YOUR_ACCOUNT>
#SBATCH -p dev
#SBATCH --gres=gpu:2
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -o <REPO_ROOT>/code/logs_lens/sbatch-seeds-theta0-%j.out
#SBATCH -e <REPO_ROOT>/code/logs_lens/sbatch-seeds-theta0-%j.err
#
# theta0 warm-start ablation: does the hardware-compatible arm depend on a KL seeding step?
#
# The M = 4 flat-MMD arm is the one the paper calls hardware-compatible, because the flat
# Walsh kernel keeps the loss estimable from shots.  But its theta0 warm-start minimises KL
# against the mean training crop (train_iqp_affine_r3.py:457-460), and KL needs the full
# probability vector -- something a shot-based device cannot form.  So the published pipeline
# is not shot-estimable end to end, and a reviewer can ask whether the result survives
# without that step.
#
# This job answers it.  seeds/train_iqp_affine_r3_theta0obj.py is a copy of the trainer with
# one added flag, --theta0-objective, and nothing else changed; with the default (kl) it
# reproduces the original exactly.  Here it runs with --theta0-objective mmdflat, so every
# stage of the arm is flat-MMD and the whole pipeline is shot-estimable.  The comparison arm
# is the ORIGINAL M = 4 flat run at the same seed and N, already on disk.
#
# All five seeds, not just seed 0: the point of this whole seed exercise is that a single-seed
# difference cannot be read, and the ablation is cheap enough that there is no reason to
# report it in the weaker form.
#
#   arm name       IQP-affine-M4-MMDflat-resample-th0mmdflat   (tagged so it cannot collide
#                                                               with the original's rows)
#   grid           N in {11, 100} x seeds 0-4 = 10 runs
#   cost           ~70 s each (measured for the original arm) = ~12 min on one GPU
# ---------------------------------------------------------------------------------------
set -u

ts() { date "+%F %T"; }
REPO=<REPO_ROOT>/code
cd "$REPO" || exit 1
PY=/usr/bin/python3
OUT="$REPO/outputs_seeds_theta0"
RESULTS="$REPO/results_seeds_theta0.jsonl"
SEEDS=(0 1 2 3 4)

mkdir -p logs_lens
echo "[$(ts)] START submit_seeds_theta0.sbatch job=${SLURM_JOB_ID:-none} node=$(hostname)"

echo "[$(ts)] pre-check: py_compile"
for f in seeds/train_iqp_affine_r3_theta0obj.py evaluate_r3b.py aggregate_seeds.py; do
    "$PY" -m py_compile "$f" || { echo "[FATAL] py_compile failed for $f"; exit 1; }
done

# The fork must be the original with one flag added, so with --theta0-objective kl it has to
# reproduce the original bit-for-bit.  Check that before trusting anything it produces.
echo "[$(ts)] pre-check: the fork reproduces the original at --theta0-objective kl"
rm -rf logs_lens/seeds_theta0_check
SMOKE="--n-train 5 --n-holdout 6 --n-gen 3 --seed 0 --M 4 --objective mmdflat \
 --prior resample --steps 60 --theta0-steps 40 --infer-restarts 2 --infer-steps 20 \
 --infer-chunk 3 --log-every 100 --device cuda"
CUDA_VISIBLE_DEVICES=0 "$PY" train_iqp_affine_r3.py $SMOKE \
    --out logs_lens/seeds_theta0_check/orig > logs_lens/precheck_theta0.log 2>&1 \
    || { echo "[FATAL] the ORIGINAL trainer failed the smoke"; tail -30 logs_lens/precheck_theta0.log; exit 1; }
CUDA_VISIBLE_DEVICES=0 "$PY" seeds/train_iqp_affine_r3_theta0obj.py $SMOKE \
    --theta0-objective kl --out logs_lens/seeds_theta0_check/fork >> logs_lens/precheck_theta0.log 2>&1 \
    || { echo "[FATAL] the FORK failed the smoke"; tail -30 logs_lens/precheck_theta0.log; exit 1; }
"$PY" - <<'PYEOF' || { echo "[FATAL] the fork does NOT reproduce the original at --theta0-objective kl"; exit 1; }
import numpy as np, os, sys
base = "logs_lens/seeds_theta0_check"
bad = []
for f in ("theta0.npy", "Phi.npy", "z_train.npy", "generated_000.npy"):
    a = np.load(os.path.join(base, "orig", f))
    b = np.load(os.path.join(base, "fork", f))
    if not np.array_equal(a, b):
        bad.append("%s max|d|=%.3e" % (f, float(np.abs(a - b).max())))
if bad:
    raise SystemExit("fork != original: " + "; ".join(bad))
print("[precheck] fork reproduces the original bit-for-bit at --theta0-objective kl")
PYEOF
rm -rf logs_lens/seeds_theta0_check
echo "[$(ts)] pre-checks passed"

rm -rf "$OUT"; mkdir -p "$OUT"
COMMON="--n-holdout 200 --n-gen 200 --device cuda --lr 0.02 --steps 4000 \
 --theta0-steps 1000 --infer-restarts 8 --infer-steps 1000 --infer-lr 0.005 --infer-chunk 50 \
 --log-every 500 --M 4 --objective mmdflat --prior resample --theta0-objective mmdflat"

fail=0
i=0
pids=()
tags=()
for S in "${SEEDS[@]}"; do
    for N in 100 11; do
        TAG="s${S}_m4_mmdflat_th0mmdflat_n$N"
        slot=$((i % 2))
        echo "[$(ts)] START $TAG (gpu $slot)"
        ( CUDA_VISIBLE_DEVICES=$slot "$PY" seeds/train_iqp_affine_r3_theta0obj.py $COMMON \
            --seed "$S" --n-train "$N" --out "$OUT/$TAG" > "logs_lens/seedsth0_${TAG}.log" 2>&1 ) &
        pids+=($!)
        tags+=("$TAG")
        i=$((i + 1))
        if [ $((i % 2)) -eq 0 ]; then
            for j in "${!pids[@]}"; do
                wait "${pids[$j]}" || { echo "[WARN] ${tags[$j]} failed"; tail -15 "logs_lens/seedsth0_${tags[$j]}.log"; fail=1; }
            done
            pids=(); tags=()
        fi
    done
done
for j in "${!pids[@]}"; do
    wait "${pids[$j]}" || { echo "[WARN] ${tags[$j]} failed"; fail=1; }
done

echo "[$(ts)] run dirs: $(find "$OUT" -mindepth 2 -maxdepth 2 -name meta.json | wc -l) (expected 10)"

# Scored by the unchanged evaluate_r3b against the same 200 held-out images, evaluator seed 0,
# so these rows sit next to the original arm's rows on identical terms.
echo "[$(ts)] START evaluate_r3b"
"$PY" evaluate_r3b.py --outputs-dir "$OUT" --n-holdout 200 --n-train-max 100 \
    --random-pair-arm "IQP-affine-M4-MMDflat-resample-th0mmdflat" --seed 0 --skip-missing \
    --results "$RESULTS" > logs_lens/seeds_evaluate_theta0.log 2>&1 \
    || { echo "[WARN] evaluate_r3b failed"; tail -30 logs_lens/seeds_evaluate_theta0.log; fail=1; }
tail -1 logs_lens/seeds_evaluate_theta0.log

echo "[$(ts)] DONE submit_seeds_theta0.sbatch (fail=$fail)"
exit "$fail"
