#!/bin/bash
#SBATCH -J mlpctrl
#SBATCH -A <YOUR_ACCOUNT>
#SBATCH -p dev
#SBATCH --gres=gpu:8
#SBATCH -c 32
#SBATCH --mem=128G
#SBATCH -t 01:30:00
#SBATCH -o <REPO_ROOT>/code/logs_gridmlp/ctrl-%j.out
#SBATCH -e <REPO_ROOT>/code/logs_gridmlp/ctrl-%j.err
#
# INTERNAL CONTROL (not for the paper): the same latent model with the IQP circuit deleted.
#
#   quantum arm (train_iqp_affine_mlp.py)  q_i = |<x| U(theta0 + g(z_i)) |0>|^2
#   this control (train_mlpctrl.py)        q_i = softmax(theta0 + g(z_i))
#
# theta0, g, z, the z penalty, the prior, the projection, S, the paste, the objective and the
# 4,000-step Adam are all unchanged -- only the map from theta to a distribution differs.
# Two objectives (flat MMD computed exactly from the pmf's Walsh coefficients, and KL) and two
# widths H in {32, 4}; H = 4 gives 5,140 shared parameters, within 1% of the published affine
# IQP arm's 5,120, so it is parameter-matched as well as architecture-matched.
set -u
ts() { date "+%F %T"; }
REPO=<REPO_ROOT>/code
PAPER=<REPO_ROOT>/paper
cd "$REPO" || exit 1
PY=/usr/bin/python3
OUT_ROOT=<WORK>/gsoc_iqp_gridmlp/ctrl        # heavy .npy output stays off /home
RESULTS="$REPO/results_seeds_mlpctrl.jsonl"          # 40 small rows, kept with the others
mkdir -p "$OUT_ROOT" logs_gridmlp
echo "[$(ts)] START job=${SLURM_JOB_ID:-none} node=$(hostname)"
"$PY" -m py_compile train_mlpctrl.py evaluate_v11.py "$PAPER/make_seed_tables_d2.py" || exit 1

# ---------------------------------------------------------------- smoke gate
echo "[$(ts)] smoke"
"$PY" -u train_mlpctrl.py --objective mmdflat --prior resample --M 4 --hidden 32 --seed 0 \
    --device cuda --n-train 11 --steps 300 --theta0-steps 100 --n-gen 8 --n-holdout 8 \
    --infer-steps 60 --infer-restarts 3 --log-every 100 --out "$OUT_ROOT/smoke" \
    > logs_gridmlp/ctrl_smoke.log 2>&1 || { tail -30 logs_gridmlp/ctrl_smoke.log; exit 1; }
"$PY" - "$OUT_ROOT/smoke" <<'PYEOF' || exit 1
import json, os, sys
import numpy as np
d = sys.argv[1]
m = json.load(open(os.path.join(d, "meta.json")))
c = np.load(os.path.join(d, "crops", "generated_000.npy"))
a = np.load(os.path.join(d, "generated_000.npy"))
curve = np.load(os.path.join(d, "loss_curve.npy"))
bad = []
if a.shape != (64, 64): bad.append("frame %s" % (a.shape,))
if c.shape != (32, 32): bad.append("crop %s" % (c.shape,))
if not (np.isfinite(a).all() and a.min() >= 0): bad.append("frame not finite/non-negative")
if abs(a.sum() / m["S"] - 1.0) > 0.05: bad.append("flux %.4f S" % (a.sum() / m["S"]))
if not float(curve[-1, 1]) < float(curve[0, 1]): bad.append("loss did not fall")
if m["train_corr_mean"] < 0.5: bad.append("train corr %.3f" % m["train_corr_mean"])
if m["projection_corr_mean"] is None: bad.append("no projection")
if m.get("quantum") is not False: bad.append("meta does not say quantum=False")
if m["arm"] != "MLPctrl-H32-M4-MMDflat-resample": bad.append("arm %r" % m["arm"])
print("[smoke] arm=%s params=%d train_corr=%.4f proj=%.4f nonlin=%.3f loss %.3e -> %.3e"
      % (m["arm"], m["param_count"], m["train_corr_mean"], m["projection_corr_mean"],
         m["nonlinearity_ratio"], curve[0, 1], curve[-1, 1]))
print("[smoke] %s" % ("OK" if not bad else "FAILED: %s" % bad))
raise SystemExit(1 if bad else 0)
PYEOF
echo "[$(ts)] smoke OK"

# ---------------------------------------------------------------- the 40-run grid
JOBS=()
add_job() { JOBS+=("$1|$2"); }
COMMON="--prior resample --M 4 --device cuda --n-gen 200 --n-holdout 200 --steps 4000 --gnorm rownorm"
for seed in 0 1 2 3 4; do
  for N in 11 100; do
    for H in 32 4; do
      for OBJ in mmdflat kl; do
        add_job "ctrl_${OBJ}_h${H}_n${N}_s${seed}" \
          "$PY -u train_mlpctrl.py $COMMON --objective $OBJ --hidden $H --n-train $N \
            --seed $seed --out $OUT_ROOT/s$seed/ctrl_${OBJ}_h${H}_n$N"
      done
    done
  done
done
NJOBS=${#JOBS[@]}
[ "$NJOBS" -eq 40 ] || { echo "[FATAL] built $NJOBS jobs, expected 40"; exit 1; }
echo "[$(ts)] $NJOBS runs queued (5 seeds x 2 N x 2 H x 2 objectives)"

QUEUE="$OUT_ROOT/queue.txt"; IDXFILE="$OUT_ROOT/next_idx"
LOCKFILE="$OUT_ROOT/queue.lock"; FAILFILE="$OUT_ROOT/failed.txt"
printf '%s\n' "${JOBS[@]}" > "$QUEUE"; echo 0 > "$IDXFILE"; : > "$LOCKFILE"; : > "$FAILFILE"
worker() {
    slot="$1"
    while true; do
        idx=$(flock "$LOCKFILE" -c "i=\$(cat '$IDXFILE'); echo \$((i+1)) > '$IDXFILE'; echo \$i")
        if [ -z "$idx" ] || [ "$idx" -ge "$NJOBS" ]; then break; fi
        line=$(sed -n "$((idx + 1))p" "$QUEUE")
        tag="${line%%|*}"; cmd="${line#*|}"; t_start=$(date +%s)
        if CUDA_VISIBLE_DEVICES="$slot" bash -c "$cmd" > "logs_gridmlp/${tag}.log" 2>&1; then
            echo "OK    $tag  ($(( $(date +%s) - t_start ))s)"
        else
            echo "FAIL  $tag"; echo "$tag" >> "$FAILFILE"; tail -15 "logs_gridmlp/${tag}.log"
        fi
    done
}
for slot in $(seq 0 7); do worker "$slot" & done
wait
NFAIL=$(wc -l < "$FAILFILE")
echo "[$(ts)] training done, $NFAIL failed"
[ "$NFAIL" -eq 0 ] || { cat "$FAILFILE"; exit 1; }

# ---------------------------------------------------------------- v11 scoring + table
RUNS=()
for seed in 0 1 2 3 4; do
  for N in 11 100; do
    for H in 32 4; do
      for OBJ in mmdflat kl; do
        RUNS+=("$OUT_ROOT/s$seed/ctrl_${OBJ}_h${H}_n$N")
      done
    done
  done
done
fail=0
"$PY" -u evaluate_v11.py --runs "${RUNS[@]}" --n-holdout 200 --results "$RESULTS" \
    > logs_gridmlp/ctrl_v11.log 2>&1 \
    || { echo "[WARN] evaluate_v11 failed"; tail -40 logs_gridmlp/ctrl_v11.log; fail=1; }
tail -3 logs_gridmlp/ctrl_v11.log
echo "[$(ts)] rows: $(wc -l < "$RESULTS" 2>/dev/null || echo 0)"
"$PY" -u "$PAPER/make_seed_tables_d2.py" > logs_gridmlp/table_d2.log 2>&1 \
    || { echo "[WARN] make_seed_tables_d2 failed"; tail -40 logs_gridmlp/table_d2.log; fail=1; }
tail -8 logs_gridmlp/table_d2.log
echo "[$(ts)] DONE (fail=$fail)"
exit "$fail"
