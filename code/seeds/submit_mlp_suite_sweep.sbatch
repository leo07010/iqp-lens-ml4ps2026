#!/bin/bash
#SBATCH -J mlpsw
#SBATCH -A <YOUR_ACCOUNT>
#SBATCH -p dev
#SBATCH --gres=gpu:8
#SBATCH -c 32
#SBATCH --mem=128G
#SBATCH -t 01:30:00
#SBATCH -o <REPO_ROOT>/code/logs_mlpsuite/sweep-%j.out
#SBATCH -e <REPO_ROOT>/code/logs_mlpsuite/sweep-%j.err
#
# MLP suite, phase 1: the latent-width (M) and hidden-width (H) sweeps for the non-linear
# arm  theta_i = theta0 + g(z_i),  g a tanh MLP  (train_iqp_affine_mlp.py).
#
# Protocol frozen to part D / v11: data_lens_r3 split (R3_SPLIT_SEED = 0), the same 200
# held-out frames, seeds 0-4, --objective mmdflat, --prior resample, 1,000-step theta0 KL warm
# start, 4,000 fixed joint Adam steps, --gnorm rownorm, --g-input-scale 20.  Only M and H move.
#
#   M sweep : M in {1,2,3,4,6,8,16} x H in {32,8} x N in {11,100} x 5 seeds = 140
#   H sweep : M = 4, H in {4,16}    x N in {11,100} x 5 seeds        =  20   (H = 8 and 32
#             at M = 4 come from the M sweep above, so they are not run twice)
#
# M = 4, H = 32 is deliberately re-run here even though part D already has it: same script,
# same CLI, same seed, so it must reproduce part D bit for bit, and the phase-3 consistency
# check tests exactly that.
set -u
ts() { date "+%F %T"; }
REPO=<REPO_ROOT>/code
cd "$REPO" || exit 1
PY=/usr/bin/python3
OUT_ROOT=<WORK>/gsoc_iqp_mlpsuite/sweep
RESULTS="$REPO/results_seeds_mlp_suite.jsonl"
mkdir -p "$OUT_ROOT" logs_mlpsuite
echo "[$(ts)] START job=${SLURM_JOB_ID:-none} node=$(hostname)"
"$PY" -m py_compile train_iqp_affine_mlp.py evaluate_v11.py || exit 1

# ---------------------------------------------------------------- smoke: the two extremes
# M = 1 and M = 16 are the only untested shapes (part D ran M in {2, 4}); everything else in
# the grid is a shape that has already produced a full run.
for MM in 1 16; do
  "$PY" -u train_iqp_affine_mlp.py --objective mmdflat --prior resample --gnorm rownorm \
      --M "$MM" --hidden 32 --seed 0 --device cuda --n-train 11 --steps 200 --theta0-steps 50 \
      --n-gen 8 --n-holdout 8 --infer-steps 40 --infer-restarts 3 --log-every 100 \
      --out "$OUT_ROOT/smoke_M$MM" > "logs_mlpsuite/smoke_M$MM.log" 2>&1 \
      || { echo "[FATAL] smoke M=$MM failed"; tail -25 "logs_mlpsuite/smoke_M$MM.log"; exit 1; }
done
"$PY" - "$OUT_ROOT" <<'PYEOF' || exit 1
import json, os, sys
import numpy as np
root = sys.argv[1]
bad = []
for MM in (1, 16):
    d = os.path.join(root, "smoke_M%d" % MM)
    m = json.load(open(os.path.join(d, "meta.json")))
    a = np.load(os.path.join(d, "generated_000.npy"))
    curve = np.load(os.path.join(d, "loss_curve.npy"))
    z = np.load(os.path.join(d, "z_train.npy"))
    phi = np.load(os.path.join(d, "Phi.npy"))
    if a.shape != (64, 64): bad.append("M=%d frame %s" % (MM, a.shape))
    if z.shape != (11, MM): bad.append("M=%d z %s" % (MM, z.shape))
    if phi.shape != (MM, 1024): bad.append("M=%d Phi %s" % (MM, phi.shape))
    if not float(curve[-1, 1]) < float(curve[0, 1]): bad.append("M=%d loss flat" % MM)
    if m["train_corr_mean"] < 0.5: bad.append("M=%d train corr %.3f" % (MM, m["train_corr_mean"]))
    if m["projection_corr_mean"] is None: bad.append("M=%d no projection" % MM)
    print("[smoke] M=%-2d arm=%s params=%d train_corr=%.4f proj=%.4f nonlin=%.3f"
          % (MM, m["arm"], m["param_count"], m["train_corr_mean"],
             m["projection_corr_mean"], m["nonlinearity_ratio"]))
print("[smoke] %s" % ("OK" if not bad else "FAILED: %s" % bad))
raise SystemExit(1 if bad else 0)
PYEOF
echo "[$(ts)] smoke OK"

# ---------------------------------------------------------------- job list
JOBS=()
add_job() { JOBS+=("$1|$2"); }
COMMON="--objective mmdflat --prior resample --gnorm rownorm --g-input-scale 20 \
        --device cuda --n-gen 200 --n-holdout 200 --steps 4000"
for seed in 0 1 2 3 4; do
  for N in 11 100; do
    for H in 32 8; do
      for M in 1 2 3 4 6 8 16; do
        add_job "sw_M${M}_H${H}_n${N}_s${seed}" \
          "$PY -u train_iqp_affine_mlp.py $COMMON --M $M --hidden $H --n-train $N \
            --seed $seed --out $OUT_ROOT/s$seed/mlp_M${M}_H${H}_n$N"
      done
    done
    for H in 4 16; do
      add_job "sw_M4_H${H}_n${N}_s${seed}" \
        "$PY -u train_iqp_affine_mlp.py $COMMON --M 4 --hidden $H --n-train $N \
          --seed $seed --out $OUT_ROOT/s$seed/mlp_M4_H${H}_n$N"
    done
  done
done
NJOBS=${#JOBS[@]}
[ "$NJOBS" -eq 160 ] || { echo "[FATAL] built $NJOBS jobs, expected 160"; exit 1; }
echo "[$(ts)] $NJOBS runs queued"

QUEUE="$OUT_ROOT/queue.txt"; IDXFILE="$OUT_ROOT/next_idx"
LOCKFILE="$OUT_ROOT/queue.lock"; FAILFILE="$OUT_ROOT/failed.txt"
printf '%s\n' "${JOBS[@]}" > "$QUEUE"; echo 0 > "$IDXFILE"; : > "$LOCKFILE"; : > "$FAILFILE"
worker() {
    slot="$1"
    while true; do
        idx=$(flock "$LOCKFILE" -c "i=\$(cat '$IDXFILE'); echo \$((i+1)) > '$IDXFILE'; echo \$i")
        if [ -z "$idx" ] || [ "$idx" -ge "$NJOBS" ]; then break; fi
        line=$(sed -n "$((idx + 1))p" "$QUEUE")
        tag="${line%%|*}"; cmd="${line#*|}"; t0=$(date +%s)
        if CUDA_VISIBLE_DEVICES="$slot" bash -c "$cmd" > "logs_mlpsuite/${tag}.log" 2>&1; then
            echo "OK    $tag  ($(( $(date +%s) - t0 ))s)"
        else
            echo "FAIL  $tag"; echo "$tag" >> "$FAILFILE"; tail -12 "logs_mlpsuite/${tag}.log"
        fi
    done
}
for slot in $(seq 0 7); do worker "$slot" & done
wait
NFAIL=$(wc -l < "$FAILFILE")
echo "[$(ts)] training done, $NFAIL failed"
[ "$NFAIL" -eq 0 ] || { cat "$FAILFILE"; exit 1; }

# ---------------------------------------------------------------- v11 scoring
RUNS=()
for seed in 0 1 2 3 4; do
  for N in 11 100; do
    for H in 32 8; do for M in 1 2 3 4 6 8 16; do
      RUNS+=("$OUT_ROOT/s$seed/mlp_M${M}_H${H}_n$N"); done; done
    for H in 4 16; do RUNS+=("$OUT_ROOT/s$seed/mlp_M4_H${H}_n$N"); done
  done
done
"$PY" -u evaluate_v11.py --runs "${RUNS[@]}" --n-holdout 200 --results "$RESULTS" --quiet \
    > logs_mlpsuite/v11_sweep.log 2>&1 \
    || { echo "[WARN] evaluate_v11 failed"; tail -40 logs_mlpsuite/v11_sweep.log; exit 1; }
tail -3 logs_mlpsuite/v11_sweep.log
echo "[$(ts)] rows in $RESULTS: $(wc -l < "$RESULTS")"
echo "[$(ts)] DONE"
