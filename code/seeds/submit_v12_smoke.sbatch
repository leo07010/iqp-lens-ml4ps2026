#!/bin/bash
#SBATCH -J v12_smoke
#SBATCH -A <YOUR_ACCOUNT>
#SBATCH -p dev
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:40:00
#SBATCH -o <REPO_ROOT>/code/logs_lens/sbatch-v12smoke-%j.out
#SBATCH -e <REPO_ROOT>/code/logs_lens/sbatch-v12smoke-%j.err
#
# SMOKE for train_iqp_affine_v12.py, before any grid is submitted.
#
#   1  py_compile
#   2  5-step CUDA run of the most exotic corner (multi + walsh + flux resample, M=2)
#   3  REGRESSION: v12 with --objective mmdflat --mu 0 --flux mean must reproduce the
#      published seed-0 arm IQP-affine-M4-MMDflat-resample at N=11 (outputs_lens_r3b/
#      iqp_m4_mmdflat_resample_n11) -- the refactor must not have changed the arithmetic
#   4  TIMING: one full N=11 and one full N=100 v12 run at the grid's real settings
#   5  evaluate_v11 on the timing runs, so the row shape is known good before the grid
# ---------------------------------------------------------------------------------------
set -u
ts() { date "+%F %T"; }
REPO=<REPO_ROOT>/code
cd "$REPO" || exit 1
PY=/usr/bin/python3
SM="$REPO/outputs_v12_smoke"
mkdir -p logs_lens
rm -rf "$SM"; mkdir -p "$SM"
echo "[$(ts)] START job=${SLURM_JOB_ID:-none} node=$(hostname)"

echo "[$(ts)] 1/5 py_compile"
"$PY" -m py_compile train_iqp_affine_v12.py evaluate_v11.py || { echo "[FATAL] py_compile"; exit 1; }

echo "[$(ts)] 2/5 tiny CUDA run of the exotic corner"
CUDA_VISIBLE_DEVICES=0 "$PY" -u train_iqp_affine_v12.py \
    --n-train 11 --n-holdout 4 --n-gen 4 --seed 7 --M 2 \
    --objective multi --mu 1e-2 --walsh-k0 3 --flux resample --prior resample \
    --steps 20 --theta0-steps 20 --infer-restarts 3 --infer-steps 5 --infer-chunk 2 \
    --log-every 10 --device cuda --out "$SM/tiny" > logs_lens/v12_smoke_tiny.log 2>&1 \
    || { echo "[FATAL] tiny run failed"; tail -40 logs_lens/v12_smoke_tiny.log; exit 1; }
grep -E "^\[v12\]" logs_lens/v12_smoke_tiny.log
echo "[$(ts)] tiny run OK"

echo "[$(ts)] 3/5 regression against the published seed-0 M4 flat arm"
t0=$(date +%s)
CUDA_VISIBLE_DEVICES=0 "$PY" -u train_iqp_affine_v12.py \
    --n-train 11 --n-holdout 200 --n-gen 200 --seed 0 --M 4 \
    --objective mmdflat --mu 0 --flux mean --prior resample \
    --steps 4000 --lr 0.02 --theta0-steps 1000 --infer-restarts 8 --infer-steps 1000 \
    --infer-lr 0.005 --infer-chunk 50 --log-every 1000 --device cuda \
    --out "$SM/regress_flat_n11" > logs_lens/v12_smoke_regress.log 2>&1 \
    || { echo "[FATAL] regression run failed"; tail -40 logs_lens/v12_smoke_regress.log; exit 1; }
echo "[$(ts)] regression run took $(( $(date +%s) - t0 )) s"
"$PY" - <<'PYEOF'
import json, os, glob
import numpy as np
REPO = "<REPO_ROOT>/code"
ref = os.path.join(REPO, "outputs_lens_r3b", "iqp_m4_mmdflat_resample_n11")
new = os.path.join(REPO, "outputs_v12_smoke", "regress_flat_n11")
n = len(glob.glob(os.path.join(ref, "generated_*.npy")))
worst = 0.0
for i in range(n):
    a = np.load(os.path.join(ref, "generated_%03d.npy" % i))
    b = np.load(os.path.join(new, "generated_%03d.npy" % i))
    worst = max(worst, float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max()))
ma = json.load(open(os.path.join(ref, "meta.json")))
mb = json.load(open(os.path.join(new, "meta.json")))
print("[regress] %d images, max |ref - v12| = %.3e" % (n, worst))
for k in ("loss_final_data", "train_corr_mean", "projection_corr_mean", "S", "zpen_L0"):
    print("[regress] %-22s ref=%.10g  v12=%.10g" % (k, ma[k], mb[k]))
print("[regress] VERDICT:", "IDENTICAL" if worst == 0.0 else
      ("CLOSE (%.1e)" % worst if worst < 1e-4 else "DIFFERENT -- investigate"))
PYEOF

echo "[$(ts)] 4/5 timing: full v12 multi runs at N=11 and N=100"
for N in 11 100; do
    t0=$(date +%s)
    CUDA_VISIBLE_DEVICES=0 "$PY" -u train_iqp_affine_v12.py \
        --n-train "$N" --n-holdout 200 --n-gen 200 --seed 0 --M 4 \
        --objective multi --mu 1e-2 --walsh-k0 3 --flux resample --prior resample \
        --steps 4000 --lr 0.02 --theta0-steps 1000 --infer-restarts 8 --infer-steps 1000 \
        --infer-lr 0.005 --infer-chunk 50 --log-every 1000 --device cuda \
        --out "$SM/multi_n$N" > "logs_lens/v12_smoke_multi_n$N.log" 2>&1 \
        || { echo "[FATAL] multi N=$N failed"; tail -40 "logs_lens/v12_smoke_multi_n$N.log"; exit 1; }
    echo "[$(ts)] multi N=$N took $(( $(date +%s) - t0 )) s"
    grep -E "^\[v12\]" "logs_lens/v12_smoke_multi_n$N.log"
done

echo "[$(ts)] 5/5 evaluate_v11 on the three full runs"
"$PY" -u evaluate_v11.py --runs "$SM/regress_flat_n11" "$SM/multi_n11" "$SM/multi_n100" \
    --n-holdout 200 --no-legacy --results "$SM/results_smoke_v12.jsonl" \
    > logs_lens/v12_smoke_eval.log 2>&1 \
    || { echo "[FATAL] evaluate_v11 failed"; tail -40 logs_lens/v12_smoke_eval.log; exit 1; }
"$PY" - <<'PYEOF'
import json
p = "<REPO_ROOT>/code/outputs_v12_smoke/results_smoke_v12.jsonl"
print("%-46s %2s %10s %8s %8s %8s %8s %8s" % ("arm", "N", "PDF-JS", "MMD2", "MMD2_al", "gen_max", "clip%", "genTot"))
for line in open(p):
    r = json.loads(line)
    print("%-46s %2d %10.6f %8.5f %8.5f %8.3f %8.3f %8.2f"
          % (r["arm"][:46], r["n_train"], r["pdf_js"], r["conv_mmd2"], r["conv_mmd2_aligned"],
             r["gen_max"], 100 * r["gen_clip_frac"], r["gen_mean_total"]))
PYEOF
du -sh "$SM"
echo "[$(ts)] SMOKE DONE"
