#!/bin/bash
#SBATCH -J mlpsuite2
#SBATCH -A <YOUR_ACCOUNT>
#SBATCH -p dev
#SBATCH --gres=gpu:8
#SBATCH -c 32
#SBATCH --mem=128G
#SBATCH -t 03:00:00
#SBATCH -o <HOME>/gsoc2026-deeplense-qml/experiments/13_iqp_turbulence_replication/logs_mlpsuite/phase2-%j.out
#SBATCH -e <HOME>/gsoc2026-deeplense-qml/experiments/13_iqp_turbulence_replication/logs_mlpsuite/phase2-%j.err
#
# MLP suite, phase 2 -- everything that depends on knowing the best (M, H).
#
# The winner is chosen by seeds/pick_best_mlp.py from phase 1's N=11 rows, on the flux-aligned
# Conv-MMD2 and on PDF-JS.  If those two metrics disagree, BOTH combinations go forward; the
# rule was fixed before any number was seen.
#
#   Table 2 targets  best combos x {flat MMD, paper kernel auto3} x N in {11,100} x 5 seeds,
#                    each at the fixed 4,000 steps AND as a 16,000-step run with snapshots at
#                    1k/2k/4k/8k/16k, so phase 3 can run the same validation-selected
#                    checkpoint check the classical baselines get.
#   Scaling          best combo, flat, N in {10,30,50,70,90,100} x 5 seeds
#   Per-class        best combo, flat, axion / cdm / no_sub, N in {11,100} x 5 seeds
#   Ablations        best combo, flat: theta0 warm start by flat MMD instead of KL;
#                    projection restart picked by loss instead of correlation; and the restart
#                    ladder cut to 1 (zeros only) and 2 (zeros + nearest training image)
#   M sweep at N=20  M in {1,2,3,4,6,8,16}, H = 32, for the MLP version of Fig. 2(a).  The
#                    mechanism test projects onto 20 held-out images (mech_seeds.py build_data:
#                    load_r3(40, n_holdout=20)), so these runs use --n-holdout 20 too and the
#                    two projection curves are measured on the same 20 frames.  The v11 metrics
#                    are unaffected: evaluate_v11.py builds its own 200-frame context.
#
# Everything else is frozen to part D / v11: data_lens_r3 split, same 200 held-out frames,
# seeds 0-4, --prior resample, --gnorm rownorm, --g-input-scale 20, 1,000-step theta0 KL warm
# start (except where the ablation says otherwise), 4,000 fixed steps.
set -u
ts() { date "+%F %T"; }
REPO=<HOME>/gsoc2026-deeplense-qml/experiments/13_iqp_turbulence_replication
cd "$REPO" || exit 1
PY=/usr/bin/python3
ROOT=<WORK>/gsoc_iqp_mlpsuite
OUT=$ROOT/phase2
RESULTS="$REPO/results_seeds_mlp_suite.jsonl"
mkdir -p "$OUT" logs_mlpsuite
echo "[$(ts)] START job=${SLURM_JOB_ID:-none} node=$(hostname)"
"$PY" -m py_compile train_iqp_affine_mlp.py train_iqp_affine_mlp_abl.py seeds/run_mlp_class.py \
    seeds/pick_best_mlp.py evaluate_v11.py || exit 1

# ---------------------------------------------------------------- the winner(s)
mapfile -t BEST < <("$PY" seeds/pick_best_mlp.py 2> logs_mlpsuite/pick_best.log)
cat logs_mlpsuite/pick_best.log
[ "${#BEST[@]}" -ge 1 ] || { echo "[FATAL] pick_best_mlp.py returned nothing"; exit 1; }
echo "[$(ts)] best combos: ${BEST[*]}"
BM=$(echo "${BEST[0]}" | cut -d' ' -f1); BH=$(echo "${BEST[0]}" | cut -d' ' -f2)
echo "[$(ts)] primary combo for scaling / per-class / ablations: M=$BM H=$BH"

JOBS=()
add_job() { JOBS+=("$1|$2"); }
COMMON="--prior resample --gnorm rownorm --g-input-scale 20 --device cuda \
        --n-gen 200 --n-holdout 200"
SNAPS="--save-every 1000 2000 4000 8000 16000"

for seed in 0 1 2 3 4; do
  # ---- 1. Table 2 targets: every winning combo, both objectives, both step protocols ----
  for combo in "${BEST[@]}"; do
    M=$(echo "$combo" | cut -d' ' -f1); H=$(echo "$combo" | cut -d' ' -f2)
    for N in 11 100; do
      for OBJ in mmdflat auto3; do
        add_job "t2_${OBJ}_M${M}_H${H}_n${N}_s${seed}" \
          "$PY -u train_iqp_affine_mlp.py $COMMON --objective $OBJ --M $M --hidden $H \
            --n-train $N --seed $seed --steps 4000 --out $OUT/s$seed/t2_${OBJ}_M${M}_H${H}_n$N"
        add_job "t2es_${OBJ}_M${M}_H${H}_n${N}_s${seed}" \
          "$PY -u train_iqp_affine_mlp.py $COMMON --objective $OBJ --M $M --hidden $H \
            --n-train $N --seed $seed $SNAPS --out $OUT/s$seed/t2es_${OBJ}_M${M}_H${H}_n$N"
      done
    done
  done
  # ---- 2. scaling, primary combo, flat MMD ----
  for N in 10 30 50 70 90 100; do
    add_job "sc_n${N}_s${seed}" \
      "$PY -u train_iqp_affine_mlp.py $COMMON --objective mmdflat --M $BM --hidden $BH \
        --n-train $N --seed $seed --steps 4000 --out $OUT/s$seed/sc_M${BM}_H${BH}_n$N"
  done
  # ---- 3. per-class ----
  for CLS in axion cdm no_sub; do
    for N in 11 100; do
      add_job "cls_${CLS}_n${N}_s${seed}" \
        "$PY -u seeds/run_mlp_class.py $CLS $COMMON --objective mmdflat --M $BM --hidden $BH \
          --n-train $N --seed $seed --steps 4000 --out $OUT/s$seed/${CLS}_mlp_M${BM}_H${BH}_n$N"
    done
  done
  # ---- 4. ablations, primary combo, flat MMD ----
  for N in 11 100; do
    add_job "abl_th0mmd_n${N}_s${seed}" \
      "$PY -u train_iqp_affine_mlp_abl.py $COMMON --objective mmdflat --M $BM --hidden $BH \
        --n-train $N --seed $seed --steps 4000 --theta0-objective mmdflat \
        --arm-name IQP-mlpabl-th0mmd-H${BH}-M${BM} --out $OUT/s$seed/abl_th0mmd_n$N"
    add_job "abl_pickloss_n${N}_s${seed}" \
      "$PY -u train_iqp_affine_mlp_abl.py $COMMON --objective mmdflat --M $BM --hidden $BH \
        --n-train $N --seed $seed --steps 4000 --restart-pick loss \
        --arm-name IQP-mlpabl-pickloss-H${BH}-M${BM} --out $OUT/s$seed/abl_pickloss_n$N"
    for R in 1 2; do
      add_job "abl_R${R}_n${N}_s${seed}" \
        "$PY -u train_iqp_affine_mlp_abl.py $COMMON --objective mmdflat --M $BM --hidden $BH \
          --n-train $N --seed $seed --steps 4000 --infer-restarts $R \
          --arm-name IQP-mlpabl-R${R}-H${BH}-M${BM} --out $OUT/s$seed/abl_R${R}_n$N"
    done
  done
  # ---- 5. the N = 20 M sweep for the Fig. 2(a) counterpart ----
  for M in 1 2 3 4 6 8 16; do
    add_job "n20_M${M}_s${seed}" \
      "$PY -u train_iqp_affine_mlp.py $COMMON --objective mmdflat --M $M --hidden 32 \
        --n-train 20 --n-holdout 20 --seed $seed --steps 4000 \
        --out $OUT/s$seed/n20_M${M}_H32"
  done
done
NJOBS=${#JOBS[@]}
echo "[$(ts)] $NJOBS runs queued (${#BEST[@]} winning combo(s))"

QUEUE="$OUT/queue.txt"; IDXFILE="$OUT/next_idx"
LOCKFILE="$OUT/queue.lock"; FAILFILE="$OUT/failed.txt"
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
echo "[$(ts)] training done, $NFAIL of $NJOBS failed"

# ---------------------------------------------------------------- v11 scoring (append)
# Every directory that has a meta.json is scored, including the snapshot dirs the 16k runs
# emit; phase 3 decides which of those the validation block actually selected.
"$PY" - "$OUT" > "$OUT/v11_runs.txt" <<'PYEOF'
import os, sys
root = sys.argv[1]
out = []
for dirpath, dirnames, filenames in os.walk(root):
    if "meta.json" in filenames and os.path.basename(dirpath) != "crops":
        out.append(dirpath)
print("\n".join(sorted(out)))
PYEOF
mapfile -t RUNS < "$OUT/v11_runs.txt"
echo "[$(ts)] scoring ${#RUNS[@]} run dirs"
"$PY" -u evaluate_v11.py --runs "${RUNS[@]}" --n-holdout 200 --results "$RESULTS" --append \
    --quiet > logs_mlpsuite/v11_phase2.log 2>&1 \
    || { echo "[WARN] evaluate_v11 failed"; tail -40 logs_mlpsuite/v11_phase2.log; exit 1; }
tail -3 logs_mlpsuite/v11_phase2.log
echo "[$(ts)] rows in $RESULTS: $(wc -l < "$RESULTS")"
echo "[$(ts)] DONE (failed runs: $NFAIL)"
exit 0
