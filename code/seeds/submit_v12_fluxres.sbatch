#!/bin/bash
#SBATCH -J v12_flux
#SBATCH -A <YOUR_ACCOUNT>
#SBATCH -p dev
#SBATCH --gres=gpu:1
#SBATCH -c 12
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o <HOME>/gsoc2026-deeplense-qml/experiments/13_iqp_turbulence_replication/logs_lens/sbatch-v12flux-%j.out
#SBATCH -e <HOME>/gsoc2026-deeplense-qml/experiments/13_iqp_turbulence_replication/logs_lens/sbatch-v12flux-%j.err
#
# Re-score every run the v12 tables quote under BOTH total-intensity conventions
# (fixed S = training mean, and S_i resampled from a Silverman KDE of the N training crop
# totals).  No training: evaluate_v12_flux.py rebuilds the frames from each run's saved
# crops/ + meta["gen_offsets"], so the fixed-S pass must reproduce the published frames
# bit-for-bit -- which --check asserts and the job fails on.
#
# 325 run dirs x 2 conventions, CPU only, ~10 min.
# ---------------------------------------------------------------------------------------
set -u
ts() { date "+%F %T"; }
REPO=<HOME>/gsoc2026-deeplense-qml/experiments/13_iqp_turbulence_replication
cd "$REPO" || exit 1
PY=/usr/bin/python3
OUT=<WORK>/gsoc_iqp_fluxres

mkdir -p "$OUT" "$REPO/logs_lens" || exit 1

echo "[$(ts)] host=$(hostname) job=${SLURM_JOB_ID:-none}"
"$PY" -c "import numpy, torch; print('numpy', numpy.__version__, '| torch', torch.__version__)" || exit 1

echo "[$(ts)] scoring both flux conventions"
"$PY" -u "$REPO/evaluate_v12_flux.py" \
    --results "$OUT/results_v12_fluxres.jsonl" \
    --threads 8 --check \
    > "$OUT/flux_eval_${SLURM_JOB_ID:-0}.log" 2>&1
rc=$?
tail -25 "$OUT/flux_eval_${SLURM_JOB_ID:-0}.log"
if [ $rc -ne 0 ]; then
    echo "[$(ts)] FAILED rc=$rc -- see $OUT/flux_eval_${SLURM_JOB_ID:-0}.log"
    exit $rc
fi

cp "$OUT/results_v12_fluxres.jsonl" "$REPO/results_v12_fluxres.jsonl" || exit 1
echo "[$(ts)] rows: $(wc -l < "$REPO/results_v12_fluxres.jsonl")"
echo "[$(ts)] done"
