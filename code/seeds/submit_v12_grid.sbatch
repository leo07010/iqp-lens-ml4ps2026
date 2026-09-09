#!/bin/bash
#SBATCH -J v12_grid
#SBATCH -A <YOUR_ACCOUNT>
#SBATCH -p dev
#SBATCH --gres=gpu:8
#SBATCH -c 32
#SBATCH --mem=128G
#SBATCH -t 03:30:00
#SBATCH -o <HOME>/gsoc2026-deeplense-qml/experiments/13_iqp_turbulence_replication/logs_lens/sbatch-v12grid-%j.out
#SBATCH -e <HOME>/gsoc2026-deeplense-qml/experiments/13_iqp_turbulence_replication/logs_lens/sbatch-v12grid-%j.err
#
# The v12 IQP-side improvement grid (train_iqp_affine_v12.py), N = 11, seeds 0-4, scored
# under the UNCHANGED v11 protocol (evaluate_v11.py).  The baselines, the data split, the
# held-out set and the metrics are not touched; the comparison numbers are parsed by
# v12_aggregate.py out of paper/ml4ps2026/seed_tables.md, Table 2 (v11).
#
# ------------------------------------------------------------------------------ arm matrix
#  A-D  crop10 / affine, multi-kernel + Walsh low-pass + flux + lam frontier          14
#  E    pixel-space Gaussian MMD, with its spatial/flat frontier swept                  8
#  F    8-qubit 16x16 + fixed bilinear upsample (codec_v12 lowres8)                     5
#  G    Gray-coded axes (codec_v12 gray10)                                              4
#  H    small MLP latent map, H in {4, 8}                                               9
#  I    lowres8 + MLP H=8                                                               4
#  = 44 arms x 5 seeds = 220 runs
#
# Group sizes and the frontier sweeps are set by the two smoke jobs, whose seed-0 readings at
# N = 11 were: flux resampling alone takes raw MMD^2 from 0.0894 to ~0.045 (DCGAN 0.0509); the
# spatial objective wins PDF-JS (0.00094 vs DCGAN 0.00111) but loses aligned MMD^2 (0.112 vs
# 0.0274); their equal-magnitude mix lands at 0.00182 / 0.0431 / 0.0266, i.e. beating DCGAN on
# two of the three columns.  The spatial/flat weight is therefore the frontier and is swept.
#
# --------------------------------------------------------------------------- time and disk
# MEASURED on the two smoke jobs: ~70-80 s per run on one GPU, independent of N.
# 220 x 78 / 8 = 2,145 s = 36 min, plus ~20 min of evaluation.  3.5 h requested.
# Disk: 5.6 MB per run dir -> ~1.3 GB.  The guard below refuses to start under 2 GB free.
# Partition dev (4 h); 8gpus is at its QOSMaxSubmitJobPerUserLimit for this account.
# ---------------------------------------------------------------------------------------
set -u
ts() { date "+%F %T"; }
REPO=<HOME>/gsoc2026-deeplense-qml/experiments/13_iqp_turbulence_replication
cd "$REPO" || exit 1
PY=/usr/bin/python3
OUT_ROOT="$REPO/outputs_v12"
SEEDS=(0 1 2 3 4)
mkdir -p logs_lens "$OUT_ROOT"
echo "[$(ts)] START submit_v12_grid.sbatch job=${SLURM_JOB_ID:-none} node=$(hostname)"

"$PY" -m py_compile train_iqp_affine_v12.py codec_v12.py objective_spatial.py \
    evaluate_v11.py v12_aggregate.py v12_figs.py || { echo "[FATAL] py_compile"; exit 1; }
command -v flock >/dev/null 2>&1 || { echo "[FATAL] flock missing"; exit 1; }
AVAIL_KB=$(df -Pk "$REPO" | tail -1 | awk '{print $4}')
echo "[$(ts)] free space: $((AVAIL_KB / 1024)) MB"
[ "$AVAIL_KB" -lt 2000000 ] && { echo "[FATAL] < 2 GB free, refusing to write ~1.3 GB"; exit 1; }

COMMON="--n-train 11 --n-holdout 200 --n-gen 200 --device cuda --prior resample --lr 0.02 \
 --steps 4000 --theta0-steps 1000 --infer-restarts 8 --infer-steps 1000 --infer-lr 0.005 \
 --infer-chunk 50 --log-every 2000 --sigmas auto3"

JOBS=()
add_job() { JOBS+=("$1|$2"); }
btag() { if [ "$1" = "0" ]; then echo "mu0"; else echo "mu$1k$2"; fi; }
SF="--spatial-flat auto"          # spatial + flat, weight from the auto rule x --spatial-flat-scale

for S in "${SEEDS[@]}"; do
    O="$OUT_ROOT/s$S"
    mkdir -p "$O"
    B="$PY -u train_iqp_affine_v12.py $COMMON --seed $S"

    # -------- group 1: crop10 / affine, multi-kernel (levers A-D) ------------- 14 arms
    for M in 2 4; do
        for FLUX in mean resample; do
            for BC in "0:3" "1e-2:3"; do
                MU="${BC%%:*}"; K0="${BC##*:}"
                T="M${M}_multi_$(btag "$MU" "$K0")_flux$FLUX"
                add_job "s${S}_$T" "$B --M $M --objective multi --mu $MU --walsh-k0 $K0 \
                    --flux $FLUX --out $O/$T"
            done
        done
        for OBJ in mmdflat auto3; do              # attribution: lever C on a single kernel
            T="M${M}_${OBJ}_mu0_fluxresample"
            add_job "s${S}_$T" "$B --M $M --objective $OBJ --mu 0 --flux resample --out $O/$T"
        done
    done
    for LS in 0.25 4; do                          # lever A's flat/paper frontier
        T="M4_multix${LS}_mu0_fluxresample"
        add_job "s${S}_$T" "$B --M 4 --objective multi --lam-scale $LS --mu 0 \
            --flux resample --out $O/$T"
    done

    # -------- group 2: E, pixel-space Gaussian MMD ---------------------------- 8 arms
    # smoke 2: pure spatial wins PDF-JS (0.00094) but loses aligned MMD^2 (0.112); the
    # equal-magnitude mix sits at 0.00182 / 0.0266.  The scale IS the frontier, so sweep it.
    add_job "s${S}_M4_spatial_fluxresample" \
        "$B --M 4 --objective spatial --flux resample --out $O/M4_spatial_fluxresample"
    for FS in 0.25 1 4 16; do
        T="M4_spatialflat${FS}_fluxresample"
        add_job "s${S}_$T" "$B --M 4 --objective spatial $SF --spatial-flat-scale $FS \
            --flux resample --out $O/$T"
    done
    add_job "s${S}_M2_spatialflat1_fluxresample" \
        "$B --M 2 --objective spatial $SF --flux resample --out $O/M2_spatialflat1_fluxresample"
    add_job "s${S}_M4_spatialflat1_fluxmean" \
        "$B --M 4 --objective spatial $SF --flux mean --out $O/M4_spatialflat1_fluxmean"
    add_job "s${S}_M4_spatialflat1_mu1e-2k3_fluxresample" \
        "$B --M 4 --objective spatial $SF --mu 1e-2 --walsh-k0 3 --flux resample \
         --out $O/M4_spatialflat1_mu1e-2k3_fluxresample"

    # -------- group 3: F, 8 qubits on a 16x16 grid + fixed bilinear ----------- 5 arms
    for M in 2 4; do
        add_job "s${S}_lowres8_M${M}_mmdflat_fluxresample" \
            "$B --encoding lowres8 --M $M --objective mmdflat --flux resample \
             --out $O/lowres8_M${M}_mmdflat_fluxresample"
        add_job "s${S}_lowres8_M${M}_spatialflat1_fluxresample" \
            "$B --encoding lowres8 --M $M --objective spatial $SF --flux resample \
             --out $O/lowres8_M${M}_spatialflat1_fluxresample"
    done
    add_job "s${S}_lowres8_M2_spatialflat1_fluxmean" \
        "$B --encoding lowres8 --M 2 --objective spatial $SF --flux mean \
         --out $O/lowres8_M2_spatialflat1_fluxmean"

    # -------- group 4: G, Gray-coded axes ------------------------------------- 4 arms
    for FLUX in mean resample; do
        add_job "s${S}_gray10_M4_mmdflat_flux$FLUX" \
            "$B --encoding gray10 --M 4 --objective mmdflat --flux $FLUX \
             --out $O/gray10_M4_mmdflat_flux$FLUX"
    done
    for FS in 1 4; do
        add_job "s${S}_gray10_M4_spatialflat${FS}_fluxresample" \
            "$B --encoding gray10 --M 4 --objective spatial $SF --spatial-flat-scale $FS \
             --flux resample --out $O/gray10_M4_spatialflat${FS}_fluxresample"
    done

    # -------- group 5: H, small MLP latent map -------------------------------- 9 arms
    for H in 4 8; do
        for M in 2 4; do
            add_job "s${S}_mlpH${H}_M${M}_mmdflat_fluxresample" \
                "$B --latent-map mlp --hidden $H --M $M --objective mmdflat --flux resample \
                 --out $O/mlpH${H}_M${M}_mmdflat_fluxresample"
        done
    done
    for FS in 1 4; do
        add_job "s${S}_mlpH8_M4_spatialflat${FS}_fluxresample" \
            "$B --latent-map mlp --hidden 8 --M 4 --objective spatial $SF \
             --spatial-flat-scale $FS --flux resample \
             --out $O/mlpH8_M4_spatialflat${FS}_fluxresample"
    done
    add_job "s${S}_mlpH8_M4_multi_fluxresample" \
        "$B --latent-map mlp --hidden 8 --M 4 --objective multi --flux resample \
         --out $O/mlpH8_M4_multi_fluxresample"
    add_job "s${S}_mlpH8_M4_mmdflat_fluxmean" \
        "$B --latent-map mlp --hidden 8 --M 4 --objective mmdflat --flux mean \
         --out $O/mlpH8_M4_mmdflat_fluxmean"
    add_job "s${S}_mlpH4_M4_spatialflat1_fluxresample" \
        "$B --latent-map mlp --hidden 4 --M 4 --objective spatial $SF --flux resample \
         --out $O/mlpH4_M4_spatialflat1_fluxresample"

    # -------- group 6: I, lowres8 + MLP H=8 ----------------------------------- 4 arms
    for M in 2 4; do
        add_job "s${S}_lowres8_mlpH8_M${M}_mmdflat_fluxresample" \
            "$B --encoding lowres8 --latent-map mlp --hidden 8 --M $M --objective mmdflat \
             --flux resample --out $O/lowres8_mlpH8_M${M}_mmdflat_fluxresample"
        add_job "s${S}_lowres8_mlpH8_M${M}_spatialflat1_fluxresample" \
            "$B --encoding lowres8 --latent-map mlp --hidden 8 --M $M --objective spatial $SF \
             --flux resample --out $O/lowres8_mlpH8_M${M}_spatialflat1_fluxresample"
    done
done

NJOBS=${#JOBS[@]}
EXPECTED=$(( ${#SEEDS[@]} * 44 ))
echo "[$(ts)] $NJOBS runs queued (expected $EXPECTED = ${#SEEDS[@]} seeds x 44 arms)"
[ "$NJOBS" -ne "$EXPECTED" ] && { echo "[FATAL] arm matrix is $NJOBS, expected $EXPECTED"; exit 1; }

QUEUE="$OUT_ROOT/queue.txt"; IDXFILE="$OUT_ROOT/next_idx"
LOCKFILE="$OUT_ROOT/queue.lock"; FAILFILE="$OUT_ROOT/failed.txt"
printf '%s\n' "${JOBS[@]}" > "$QUEUE"; echo 0 > "$IDXFILE"; : > "$LOCKFILE"; : > "$FAILFILE"

worker() {
    slot="$1"
    while true; do
        idx=$(flock "$LOCKFILE" -c "i=\$(cat '$IDXFILE'); echo \$((i+1)) > '$IDXFILE'; echo \$i")
        if [ -z "$idx" ] || [ "$idx" -ge "$NJOBS" ]; then break; fi
        line=$(sed -n "$((idx + 1))p" "$QUEUE")
        tag="${line%%|*}"; cmd="${line#*|}"
        t_start=$(date +%s)
        if CUDA_VISIBLE_DEVICES="$slot" bash -c "$cmd" > "logs_lens/v12_${tag}.log" 2>&1; then
            echo "[$(ts)] slot $slot OK   [$((idx + 1))/$NJOBS] $tag ($(( $(date +%s) - t_start )) s)"
        else
            echo "[$(ts)] slot $slot FAIL [$((idx + 1))/$NJOBS] $tag ($(( $(date +%s) - t_start )) s)"
            tail -15 "logs_lens/v12_${tag}.log"
            echo "$tag" >> "$FAILFILE"
        fi
    done
}
for slot in $(seq 0 7); do worker "$slot" & done
wait
NFAIL=$(wc -l < "$FAILFILE")
echo "[$(ts)] all $NJOBS runs finished, $NFAIL failed"
[ "$NFAIL" -gt 0 ] && { echo "failed tags:"; cat "$FAILFILE"; }
fail=$([ "$NFAIL" -gt 0 ] && echo 1 || echo 0)
GOT=$(find "$OUT_ROOT" -mindepth 3 -maxdepth 3 -name meta.json | wc -l)
echo "[$(ts)] $GOT/$NJOBS run dirs carry a meta.json"
du -sh "$OUT_ROOT"; df -Pk "$REPO" | tail -1

echo "[$(ts)] START evaluate_v11 over $OUT_ROOT"
"$PY" -u evaluate_v11.py --outputs-dir "$OUT_ROOT" --n-holdout 200 --no-legacy --quiet \
    --results "$REPO/results_seeds_v12.jsonl" > logs_lens/v12_eval.log 2>&1 \
    || { echo "[WARN] evaluate_v11 failed"; tail -40 logs_lens/v12_eval.log; fail=1; }
tail -3 logs_lens/v12_eval.log
echo "[$(ts)] rows: $(wc -l < "$REPO/results_seeds_v12.jsonl" 2>/dev/null || echo 0)"

echo "[$(ts)] START v12_aggregate.py"
"$PY" -u v12_aggregate.py > logs_lens/v12_aggregate.log 2>&1 \
    || { echo "[WARN] aggregate failed"; tail -40 logs_lens/v12_aggregate.log; fail=1; }
tail -70 logs_lens/v12_aggregate.log

echo "[$(ts)] START v12_figs.py"
"$PY" -u v12_figs.py --n-train 11 > logs_lens/v12_figs.log 2>&1 \
    || { echo "[WARN] figs failed"; tail -20 logs_lens/v12_figs.log; fail=1; }
tail -5 logs_lens/v12_figs.log
echo "[$(ts)] DONE (fail=$fail)"
exit "$fail"
