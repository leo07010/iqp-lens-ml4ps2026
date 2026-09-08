#!/bin/bash
#SBATCH -J iqp_turb_baseline_align
#SBATCH -A <YOUR_ACCOUNT>
#SBATCH -p 8gpus
#SBATCH --gres=gpu:8
#SBATCH -c 32
#SBATCH --mem=128G
#SBATCH -t 01:00:00
#SBATCH -o <REPO_ROOT>/code/logs_d2/sbatch-align-%j.out
#SBATCH -e <REPO_ROOT>/code/logs_d2/sbatch-align-%j.err
#
# Baseline-alignment sweep (2026-09-06 addendum to submit_d2_main.sbatch): our RBM/DCGAN
# baselines beat the paper's reported Table 7/8 numbers by a wide margin. The paper gives
# only an architecture-level description of each baseline, no training recipe, so this job
# scans plausible training choices to see which (if any) reproduces the paper's numbers.
# baseline_rbm.py and baseline_dcgan.py are UNMODIFIED; baseline_rbm_variants.py (new, see
# its docstring) adds two independent axes for the RBM, and this script scans DCGAN's
# training-step count directly against the existing --steps flag.
#
# Job list (16 total, single seed=0 -- 2026-09-06 correction: no multi-seed for this
# alignment scan, it's a search over training-recipe choices, not a variance estimate):
#   RBM:    --latent {literal,biases} x --recipe {pcd_adam,cd1_sgd}, seed=0             = 4 jobs
#           (biases+pcd_adam is baseline_rbm.py's own main-run combo, re-run here as a
#           built-in sanity check -- baseline_rbm_variants.py reproduces it bit-for-bit,
#           verified on a CPU smoke run before this job list was written).
#   DCGAN:  --tier {51k,0.2m,1.2m} x --steps {300,1000,3000,20000}, seed=0               = 12 jobs
#           (--n-train 11, matching the paper's data-matched regime and the 11-anchor
#           protocol every other arm uses); arm name is patched post-hoc (patch_arm below)
#           to "DCGAN-<tier>-s<steps>" so evaluate.py's (arm, train_snapshots) grouping
#           never merges different step counts of the same tier together.
#
# Wall-clock estimate (basis: submit_d2_main.sbatch's measured RBM ~20 min/job, DCGAN
# steps=20000 ~8 min/job -- other DCGAN step counts scaled linearly from that + fixed
# per-job overhead for data loading/model init):
#   RBM    4 jobs x ~15-20 min            = ~70 min total compute
#   DCGAN  3 jobs x ~8 min   (steps=20000) = ~24 min
#          3 jobs x ~1.5 min (steps=3000)  = ~4.5 min
#          3 jobs x ~1 min   (steps=1000)  = ~3 min
#          3 jobs x ~0.5 min (steps=300)   = ~1.5 min
#   TOTAL ~103 min of compute; packed 8-wide in 2 lockstep batches (jobs ordered
#   longest-first below so batch 1 = the 4 RBM jobs + 3 DCGAN-20000 + 1 DCGAN-3000,
#   batch 2 = the remaining 8 short DCGAN jobs) gives an estimated wall-clock of
#   batch1(~20 min) + batch2(~1.5 min) =~ 22 min. Requesting 1 h leaves headroom.
#
# Ground truth is REUSED from outputs_d2_main/ground_truth (the main run's 200-tau d2
# holdout set) rather than regenerated -- same source, saves ~nothing risky to skip since
# make_ground_truth.py's d2 path draws from data_d2.py's own fixed split either way, but
# reuse avoids a redundant few-minute step and guarantees byte-identical ground truth
# against results_d2_main.jsonl for direct comparison.
set -uo pipefail

REPO=<REPO_ROOT>/code
cd "$REPO"
mkdir -p logs_d2

# ---------------------------------------------------------------- fail-fast: data_d2.py + main-run ground truth must exist
/usr/bin/python3 -c "import data_d2" || {
    echo "[FATAL] data_d2.py is not importable. Aborting before using any 8gpus budget."
    exit 1
}
GT_DIR="$REPO/outputs_d2_main/ground_truth"
if [ ! -d "$GT_DIR" ] || [ -z "$(ls -A "$GT_DIR" 2>/dev/null)" ]; then
    echo "[FATAL] $GT_DIR missing or empty -- run submit_d2_main.sbatch first (this job reuses its ground truth)."
    exit 1
fi

module purge; module load cuda/13.0
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
PY=/usr/bin/python3
OUT="$REPO/outputs_d2_baseline_align"
ts() { date +"%Y-%m-%d %H:%M:%S"; }

echo "[SLURM] job=$SLURM_JOB_ID node=$(hostname) part=$SLURM_JOB_PARTITION gpus=$($PY -c 'import torch;print(torch.cuda.device_count())')"

# ---------------------------------------------------------------- fail-fast pre-checks
echo "[$(ts)] pre-check: py_compile all .py"
for f in *.py; do
    "$PY" -m py_compile "$f" || { echo "[FATAL] py_compile failed for $f"; exit 1; }
done

echo "[$(ts)] pre-check: unit tests"
for t in test_iqp_exact.py test_baselines.py test_baseline_variants.py test_controls.py test_diagnostics.py test_data_d2.py test_d2_arms_smoke.py; do
    "$PY" "$t" > "logs_d2/precheck_${t%.py}.log" 2>&1 \
        || { echo "[FATAL] $t failed -- see logs_d2/precheck_${t%.py}.log"; tail -20 "logs_d2/precheck_${t%.py}.log"; exit 1; }
done
echo "[$(ts)] pre-checks passed"

rm -rf "$OUT"
mkdir -p "$OUT"

# ---------------------------------------------------------------- patch_arm: rewrite meta.json's "arm" field in place
# (baseline_dcgan.py has no flag to override its default arm="DCGAN-<tier>" name; exported
# so it's visible inside the `bash -c "$cmd"` child processes the packing loop below spawns.)
patch_arm() {  # $1=meta.json path, $2=new arm string
    "$PY" - "$1" "$2" <<'PYEOF'
import json
import sys

path, arm = sys.argv[1], sys.argv[2]
with open(path) as fh:
    meta = json.load(fh)
meta["arm"] = arm
with open(path, "w") as fh:
    json.dump(meta, fh, indent=2)
PYEOF
}
export -f patch_arm
export PY

# ---------------------------------------------------------------- build the job list: TAG|DEVICE|CMD
JOBS=()
add_job() { JOBS+=("$1|$2|$3"); }

# Single seed=0 throughout (2026-09-06 correction) -- this sweep searches over training
# recipes, not seed variance.
SEED=0

# RBM: longest-first ordering (pcd_adam ~20 min/job before cd1_sgd ~15-16 min/job) so the
# first lockstep batch of 8 is homogeneous -- see packing loop below.
for recipe in pcd_adam cd1_sgd; do
    for latent in literal biases; do
        tag="rbm_${latent}_${recipe}_seed${SEED}"
        add_job "$tag" cuda \
            "$PY baseline_rbm_variants.py --data d2 --out $OUT/$tag --device cuda --seed $SEED --latent $latent --recipe $recipe --n-holdout 200"
    done
done

# DCGAN: longest-first ordering (steps=20000 ~8 min/job down to steps=300 ~0.5 min/job).
for steps in 20000 3000 1000 300; do
    for tier in 51k 0.2m 1.2m; do
        tag="dcgan_${tier}_s${steps}_seed${SEED}"
        add_job "$tag" cuda \
            "$PY baseline_dcgan.py --data d2 --out $OUT/$tag --device cuda --seed $SEED --tier $tier --n-train 11 --steps $steps --n-gen 200 && patch_arm $OUT/$tag/meta.json DCGAN-${tier}-s${steps}"
    done
done

echo "[$(ts)] ${#JOBS[@]} jobs queued, running 8-wide across the allocated GPUs"

# ---------------------------------------------------------------- run 8-wide, wait between batches
GPU_SLOTS=8
i=0
pids=()
fail=0
for job in "${JOBS[@]}"; do
    tag="${job%%|*}"
    rest="${job#*|}"
    device="${rest%%|*}"
    cmd="${rest#*|}"
    slot=$((i % GPU_SLOTS))
    echo "[$(ts)] START $tag (slot $slot, device=$device)"
    if [ "$device" = "cuda" ]; then
        (CUDA_VISIBLE_DEVICES=$slot bash -c "$cmd" > "logs_d2/${tag}.log" 2>&1) &
    else
        (bash -c "$cmd" > "logs_d2/${tag}.log" 2>&1) &
    fi
    pids+=($!)
    tags[$i]="$tag"
    i=$((i + 1))
    if [ $((i % GPU_SLOTS)) -eq 0 ]; then
        for j in "${!pids[@]}"; do
            wait "${pids[$j]}" || { echo "[WARN] job ${tags[$((i - GPU_SLOTS + j))]} failed, see logs_d2/${tags[$((i - GPU_SLOTS + j))]}.log"; fail=1; }
        done
        pids=()
    fi
done
for j in "${!pids[@]}"; do
    idx=$((i - ${#pids[@]} + j))
    wait "${pids[$j]}" || { echo "[WARN] job ${tags[$idx]} failed, see logs_d2/${tags[$idx]}.log"; fail=1; }
done

echo "[$(ts)] all ${#JOBS[@]} jobs finished (fail=$fail)"

# ---------------------------------------------------------------- evaluate
echo "[$(ts)] START evaluate"
"$PY" evaluate.py --outputs-dir "$OUT" --ground-truth "$GT_DIR" \
    --results "$REPO/results_d2_baseline_align.jsonl" \
    > logs_d2/evaluate_baseline_align.log 2>&1 || { echo "[FATAL] evaluate failed"; tail -30 logs_d2/evaluate_baseline_align.log; exit 1; }
cat logs_d2/evaluate_baseline_align.log

# ---------------------------------------------------------------- summary table (single seed=0 -> one value per arm, no mean/std; plus paper's reported baseline numbers for reference)
echo "[$(ts)] START summary"
"$PY" - <<'PYEOF' > logs_d2/baseline_align_summary.log 2>&1
import json
from collections import defaultdict

rows = [json.loads(line) for line in open("results_d2_baseline_align.jsonl")]
groups = defaultdict(list)
for r in rows:
    groups[r["arm"]].append(r)

print("Baseline-alignment sweep results ({} rows, {} arms, seed=0 single-seed run)".format(len(rows), len(groups)))
print()
print("| Arm | PDF-JS | MMD2 (Conv2D) |")
print("|---|---|---|")
for arm in sorted(groups):
    g = groups[arm]
    if len(g) != 1:
        print("[WARN] arm {} has {} rows, expected 1 (single-seed run) -- printing first".format(arm, len(g)))
    r = g[0]
    print("| {} | {:.4g} | {:.4g} |".format(arm, r["pdf_js"], r["conv_mmd2"]))

print()
print("Paper's reported baseline numbers (Table 7/8 -- NOT computed by this run, for reference only):")
print("| Arm (paper) | PDF-JS | MMD2 (Conv2D) |")
print("|---|---|---|")
print("| RBM (paper) | 0.168770 | 1.472481 |")
print("| DCGAN@11 51k (paper) | 0.366758 | 1.397969 |")
print("| DCGAN@11 0.2M (paper) | 0.328273 | 1.407441 |")
print("| DCGAN@11 1.2M (paper) | 0.244852 | 1.446292 |")
PYEOF
cat logs_d2/baseline_align_summary.log

echo "[$(ts)] DONE submit_baseline_align.sbatch (fail=$fail)"
exit "$fail"
