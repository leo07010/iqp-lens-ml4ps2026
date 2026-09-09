#!/bin/bash
#SBATCH -J gridmlp
#SBATCH -A <YOUR_ACCOUNT>
#SBATCH -p dev
#SBATCH --gres=gpu:8
#SBATCH -c 32
#SBATCH --mem=128G
#SBATCH -t 02:30:00
#SBATCH -o <HOME>/gsoc2026-deeplense-qml/experiments/13_iqp_turbulence_replication/logs_gridmlp/grid-%j.out
#SBATCH -e <HOME>/gsoc2026-deeplense-qml/experiments/13_iqp_turbulence_replication/logs_gridmlp/grid-%j.err
#
# Two candidate fixes for the v11 paper's stated limitations, on the round-3 protocol exactly
# as the current best IQP arm (IQP-affine-M4-MMDflat-resample) ran it:
#
#   (A) FINER GRID       train_iqp_affine_n12.py -- 12 qubits, the whole 64x64 frame on 4,096
#                        codes, no 32x32 crop.  Tests "the crop and the coarse grid are what
#                        limit the model".
#   (B) NON-LINEAR MAP   train_iqp_affine_mlp.py -- theta = theta0 + g(z) with g a tanh MLP
#                        instead of theta = theta0 + Phi^T z.  Tests "an affine-in-phase
#                        latent map is what limits the model".
#
# Shared with the reference arm and NOT varied here: the data split (data_lens_r3's
# R3_SPLIT_SEED = 0, and data_lens_n12 gets its images from that same loader), the same 200
# held-out frames, seeds 0-4, N in {11, 100}, --objective mmdflat, --prior resample, M = 4,
# 1,000-step theta0 KL warm start, 4,000 joint Adam steps, and the z-only held-out projection
# with 8 restarts.  Anything else would confound the comparison.
#
# Each arm is run under BOTH step protocols:
#   fixed    --steps 4000, the number Table 2 reports for every IQP arm.  This is the one
#            that must be reported; it is the only apples-to-apples cell.
#   es       --save-every 1000 2000 4000 8000 16000 out of ONE 16,000-step run, so
#            seeds/select_checkpoint.py can pick a step count on the validation block the way
#            the classical baselines do.  A 4,000-step slice of a 16,000-step cosine schedule
#            is NOT the fixed run (different LR trajectory), which is why both exist.
set -u
ts() { date "+%F %T"; }
REPO=<HOME>/gsoc2026-deeplense-qml/experiments/13_iqp_turbulence_replication
cd "$REPO" || exit 1
PY=/usr/bin/python3
OUT_ROOT=<WORK>/gsoc_iqp_gridmlp/seeds
mkdir -p "$OUT_ROOT" logs_gridmlp
echo "[$(ts)] START job=${SLURM_JOB_ID:-none} node=$(hostname)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# ---------------------------------------------------------------- pre-flight
"$PY" -m py_compile data_lens_n12.py train_iqp_affine_n12.py train_iqp_affine_mlp.py \
    evaluate_v11.py seeds/select_checkpoint.py || exit 1

# the arm seed must not move the data: check it, do not assume it
"$PY" - <<'PYEOF' || exit 1
import numpy as np, data_lens_n12 as D12, data_lens_r3 as D3
a = D12.load_n12(11, seed=0, n_holdout=200)
b = D12.load_n12(11, seed=4, n_holdout=200)
r = D3.load_r3(11, seed=0, n_holdout=200)
ok = (np.array_equal(a.train_pmfs, b.train_pmfs)
      and np.array_equal(a.holdout_pmfs, b.holdout_pmfs)
      and np.array_equal(a.train_images, r.train_images)
      and np.array_equal(a.holdout_images, r.holdout_images))
print("[pre] n12 split is seed-invariant and identical to data_lens_r3: %s" % ok)
raise SystemExit(0 if ok else 1)
PYEOF
echo "[$(ts)] pre-flight OK"

# ---------------------------------------------------------------- the job list
JOBS=()
add_job() { JOBS+=("$1|$2"); }

COMMON="--objective mmdflat --prior resample --device cuda --n-gen 200 --n-holdout 200"
SNAPS="--save-every 1000 2000 4000 8000 16000"

for seed in 0 1 2 3 4; do
  for N in 11 100; do
    S="$OUT_ROOT/s$seed"
    # ---- (A) 12-qubit fine grid, M = 4 ------------------------------------------------
    add_job "n12es_n${N}_s${seed}" \
      "$PY -u train_iqp_affine_n12.py $COMMON --M 4 --n-train $N --seed $seed $SNAPS \
        --out $S/n12es_m4flat_n$N"
    add_job "n12_n${N}_s${seed}" \
      "$PY -u train_iqp_affine_n12.py $COMMON --M 4 --n-train $N --seed $seed --steps 4000 \
        --out $S/n12_m4flat_n$N"
    # ---- (B) non-linear latent map, M = 4, calibrated magnitude ------------------------
    add_job "mlpes_n${N}_s${seed}" \
      "$PY -u train_iqp_affine_mlp.py $COMMON --M 4 --n-train $N --seed $seed $SNAPS \
        --gnorm rownorm --out $S/mlpes_m4flat_n$N"
    add_job "mlp_n${N}_s${seed}" \
      "$PY -u train_iqp_affine_mlp.py $COMMON --M 4 --n-train $N --seed $seed --steps 4000 \
        --gnorm rownorm --out $S/mlp_m4flat_n$N"
    # ---- (B') the same map with the affine magnitude law enforced exactly --------------
    add_job "mlpzs_n${N}_s${seed}" \
      "$PY -u train_iqp_affine_mlp.py $COMMON --M 4 --n-train $N --seed $seed --steps 4000 \
        --gnorm zscale --out $S/mlpzs_m4flat_n$N"
    # ---- (B'') M = 2: does a non-linear map need fewer latent directions? --------------
    add_job "mlpm2_n${N}_s${seed}" \
      "$PY -u train_iqp_affine_mlp.py $COMMON --M 2 --n-train $N --seed $seed --steps 4000 \
        --gnorm rownorm --out $S/mlpm2_m2flat_n$N"
    # ---- (A-diagnostic) the same 12-qubit arm started at the 10-qubit arm's OWN latent
    # scale.  The timing run showed the 12-qubit latents collapsing to ||z|| ~ 1e-3 (median
    # nearest-neighbour spacing 9e-4, against 4e-2 for the 10-qubit arm), so the fixed-protocol
    # result would be ambiguous between "the encoding cannot use a latent" and "the shared
    # z-init is simply in the wrong place at 4,096 codes".  This run starts z where the
    # 10-qubit arm ends up and changes nothing else.
    add_job "n12z_n${N}_s${seed}" \
      "$PY -u train_iqp_affine_n12.py $COMMON --M 4 --n-train $N --seed $seed --steps 4000 \
        --z-init-std 0.05 --arm-name IQP-n12-affine-M4-MMDflat-resample-zinit05 \
        --out $S/n12z_m4flat_n$N"
  done
done

NJOBS=${#JOBS[@]}
EXPECTED=70
if [ "$NJOBS" -ne "$EXPECTED" ]; then
  echo "[FATAL] built $NJOBS jobs, expected $EXPECTED"; exit 1
fi
echo "[$(ts)] $NJOBS runs queued (5 seeds x 2 N x 7 configurations)"

# ---------------------------------------------------------------- 8-GPU work queue
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
        if CUDA_VISIBLE_DEVICES="$slot" bash -c "$cmd" > "logs_gridmlp/${tag}.log" 2>&1; then
            echo "OK    $tag  ($(( $(date +%s) - t_start ))s)"
        else
            echo "FAIL  $tag  ($(( $(date +%s) - t_start ))s)"; echo "$tag" >> "$FAILFILE"
            tail -15 "logs_gridmlp/${tag}.log"
        fi
    done
}

GPU_SLOTS=8
for slot in $(seq 0 $((GPU_SLOTS - 1))); do worker "$slot" & done
wait
NFAIL=$(wc -l < "$FAILFILE")
echo "[$(ts)] training done, $NFAIL failed"
[ "$NFAIL" -eq 0 ] || { echo "[FATAL] failed runs:"; cat "$FAILFILE"; exit 1; }

# ---------------------------------------------------------------- scoring and tables
bash seeds/gridmlp_postprocess.sh
rc=$?
echo "[$(ts)] DONE (postprocess rc=$rc)"
exit "$rc"
