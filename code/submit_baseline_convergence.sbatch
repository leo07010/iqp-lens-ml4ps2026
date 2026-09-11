#!/bin/bash
#SBATCH -J iqp_lens_conv
#SBATCH -A <YOUR_ACCOUNT>
#SBATCH -p dev
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o <REPO_ROOT>/code/logs_lens/sbatch-conv-%j.out
#SBATCH -e <REPO_ROOT>/code/logs_lens/sbatch-conv-%j.err
#
# Baseline convergence sweep (paper addendum, 2026-09-09).
#
# QUESTION: Table 2 trains both classical baselines for a fixed 20,000 steps, but at that point
# the RBM's recon MSE is still falling (~20-25 % over the last quarter at N = 100) and the
# DCGAN's losses are still oscillating.  Were the baselines simply stopped too early, and would
# a longer run overtake the IQP arms?
#
# METHOD: re-run each baseline for longer and score it at several step counts, then report it at
# its BEST step count per metric -- an advantage no IQP arm in Table 2 is given (every IQP arm
# is scored at one pre-registered step count).  If the baselines still lose under best-of-steps,
# the fixed-20k budget was not what decided Table 2.
#
#   RBM-10bit  N in {11, 100}, snapshots at 20k / 40k / 80k / 160k   ->  8 points
#   DCGAN-32   N in {11, 100}, snapshots every 5k up to 60k          -> 24 points
#
# Each (N, arm) is ONE training run with snapshots, not one run per step count: both training
# loops are sequential with a single RNG stream and no step-count-dependent schedule, so the
# state at step k is what a standalone `--steps k` run would end with.  The pre-check below
# byte-compares a short sweep snapshot against a standalone run to prove that.
#
# Scoring is the UNMODIFIED evaluate_r3b.py against the SAME 200 held-out images, the same
# seed-0 Conv2D encoder and the same intensity-aligned addendum as Table 2, so every number
# here is directly comparable to a Table 2 cell.
#
# Touches nothing that Table 2 depends on: new scripts (baseline_*_sweep.py,
# aggregate_convergence.py), a new output tree (outputs_convergence/) and new result files.
# baseline_rbm_r3.py / baseline_dcgan_r3.py / evaluate_r3*.py are imported read-only.
set -uo pipefail

REPO=<REPO_ROOT>/code
cd "$REPO" || exit 1
mkdir -p logs_lens

module purge; module load cuda/13.0
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
PY=/usr/bin/python3
OUT="$REPO/outputs_convergence"
ts() { date +"%Y-%m-%d %H:%M:%S"; }

echo "[SLURM] job=$SLURM_JOB_ID node=$(hostname) part=$SLURM_JOB_PARTITION gpus=$($PY -c 'import torch;print(torch.cuda.device_count())')"

# ---------------------------------------------------------------- fail-fast: imports compile
echo "[$(ts)] pre-check: py_compile"
for f in baseline_rbm_sweep.py baseline_dcgan_sweep.py aggregate_convergence.py \
         baseline_rbm_r3.py baseline_dcgan_r3.py evaluate_r3.py evaluate_r3b.py; do
    "$PY" -m py_compile "$f" || { echo "[FATAL] py_compile failed for $f"; exit 1; }
done
"$PY" -c "import baseline_rbm_sweep, baseline_dcgan_sweep, aggregate_convergence" \
    || { echo "[FATAL] the new modules do not import"; exit 1; }

# ---------------------------------------------------------------- fail-fast: dataset cache
"$PY" -c "import data_lens_r3 as d; m = d.build_cache(); print('[conv] cache ok, N=11 class counts =', m['train11_class_counts'])" \
    || { echo "[FATAL] the round-3 cache could not be built"; exit 1; }

# ---------------------------------------------------------------- EQUIVALENCE pre-check
# The one-run-many-snapshots design rests on one claim: the snapshot at step k holds what
# `baseline_*_r3.py --steps k` would have written.  It is checked here, before any real
# training, in two parts.
#
#   (1) MECHANISM, on CPU (deterministic hardware): a sweep snapshot must be BIT-IDENTICAL to a
#       standalone run of the same length.  Any leak -- an RNG draw consumed by the snapshot, a
#       BatchNorm running statistic updated while sampling, a stale train/eval flag -- shows up
#       here as a non-zero difference.  This is the test that can fail for a real reason.
#
#   (2) HARDWARE FLOOR, on the GPU the sweep will actually use: the SAME standalone command is
#       run twice, and the sweep is required not to differ from the first run by more than a
#       small multiple of how much those two identical runs differ from each other.  cuDNN's
#       transposed-convolution backward accumulates with atomics, so two identical CUDA GAN runs
#       are not bitwise equal and 2,000 steps of adversarial training amplify that; the RBM uses
#       no such op and is required to stay bit-identical on the GPU as well.
echo "[$(ts)] pre-check: snapshot == standalone run"
SMOKE="$REPO/logs_lens/conv_equiv_smoke"
rm -rf "$SMOKE"; mkdir -p "$SMOKE"

RBM_SMOKE="--n-train 3 --n-holdout 4 --n-gen 4 --log-every 1000 --n-pool 1000 --gen-sweeps 20 --gen-keep 10 --gen-chunk 2"
GAN_SMOKE="--n-train 3 --n-holdout 4 --n-gen 4 --log-every 1000"

# (1) mechanism, on CPU
"$PY" baseline_dcgan_r3.py $GAN_SMOKE --steps 2000 --device cpu --out "$SMOKE/gan_cpu_ref" \
    > "$SMOKE/gan_cpu_ref.log" 2>&1 || { echo "[FATAL] cpu standalone DCGAN smoke failed"; tail -20 "$SMOKE/gan_cpu_ref.log"; exit 1; }
"$PY" baseline_dcgan_sweep.py $GAN_SMOKE --steps 2000 --snap-every 1000 --device cpu \
    --out-prefix "$SMOKE/gan_cpu_sw" > "$SMOKE/gan_cpu_sw.log" 2>&1 \
    || { echo "[FATAL] cpu sweep DCGAN smoke failed"; tail -20 "$SMOKE/gan_cpu_sw.log"; exit 1; }
"$PY" baseline_rbm_r3.py $RBM_SMOKE --steps 2000 --device cpu --out "$SMOKE/rbm_cpu_ref" \
    > "$SMOKE/rbm_cpu_ref.log" 2>&1 || { echo "[FATAL] cpu standalone RBM smoke failed"; tail -20 "$SMOKE/rbm_cpu_ref.log"; exit 1; }
"$PY" baseline_rbm_sweep.py $RBM_SMOKE --snapshots 1000 2000 --device cpu \
    --out-prefix "$SMOKE/rbm_cpu_sw" > "$SMOKE/rbm_cpu_sw.log" 2>&1 \
    || { echo "[FATAL] cpu sweep RBM smoke failed"; tail -20 "$SMOKE/rbm_cpu_sw.log"; exit 1; }

# (2) hardware floor, on cuda: two identical standalone runs + the sweep
CUDA_VISIBLE_DEVICES=0 "$PY" baseline_dcgan_r3.py $GAN_SMOKE --steps 2000 --device cuda \
    --out "$SMOKE/gan_ref" > "$SMOKE/gan_ref.log" 2>&1 || { echo "[FATAL] cuda standalone DCGAN smoke failed"; tail -20 "$SMOKE/gan_ref.log"; exit 1; }
CUDA_VISIBLE_DEVICES=0 "$PY" baseline_dcgan_r3.py $GAN_SMOKE --steps 2000 --device cuda \
    --out "$SMOKE/gan_ref2" > "$SMOKE/gan_ref2.log" 2>&1 || { echo "[FATAL] cuda standalone DCGAN smoke (repeat) failed"; tail -20 "$SMOKE/gan_ref2.log"; exit 1; }
CUDA_VISIBLE_DEVICES=0 "$PY" baseline_dcgan_sweep.py $GAN_SMOKE --steps 2000 --snap-every 1000 \
    --device cuda --out-prefix "$SMOKE/gan_sw" > "$SMOKE/gan_sw.log" 2>&1 \
    || { echo "[FATAL] cuda sweep DCGAN smoke failed"; tail -20 "$SMOKE/gan_sw.log"; exit 1; }
CUDA_VISIBLE_DEVICES=0 "$PY" baseline_rbm_r3.py $RBM_SMOKE --steps 2000 --device cuda \
    --out "$SMOKE/rbm_ref" > "$SMOKE/rbm_ref.log" 2>&1 || { echo "[FATAL] cuda standalone RBM smoke failed"; tail -20 "$SMOKE/rbm_ref.log"; exit 1; }
CUDA_VISIBLE_DEVICES=0 "$PY" baseline_rbm_sweep.py $RBM_SMOKE --snapshots 1000 2000 \
    --device cuda --out-prefix "$SMOKE/rbm_sw" > "$SMOKE/rbm_sw.log" 2>&1 \
    || { echo "[FATAL] cuda sweep RBM smoke failed"; tail -20 "$SMOKE/rbm_sw.log"; exit 1; }

"$PY" - "$SMOKE" <<'PYCHK'
import glob, json, os, sys
import numpy as np

smoke = sys.argv[1]


def dist(a_dir, b_dir, loss_key):
    files = sorted(os.path.basename(p) for p in glob.glob(os.path.join(a_dir, "generated_*.npy")))
    if not files:
        return None, None, 0
    dmax = 0.0
    for f in files:
        a = np.load(os.path.join(a_dir, f)).astype(np.float64)
        b = np.load(os.path.join(b_dir, f)).astype(np.float64)
        if a.shape != b.shape:
            return float("inf"), float("inf"), len(files)
        dmax = max(dmax, float(np.abs(a - b).max()))
    ma = json.load(open(os.path.join(a_dir, "meta.json")))
    mb = json.load(open(os.path.join(b_dir, "meta.json")))
    return dmax, abs(float(ma[loss_key]) - float(mb[loss_key])), len(files)


bad = 0
# (1) MECHANISM, cpu -- must be exact
for tag, ref, sw, lk in (("RBM", "rbm_cpu_ref", "rbm_cpu_sw_s002k", "recon_mse_last"),
                         ("DCGAN", "gan_cpu_ref", "gan_cpu_sw_s002k", "lossG_last")):
    d, dl, nf = dist(os.path.join(smoke, ref), os.path.join(smoke, sw), lk)
    if d is None:
        print("[equiv] cpu  %-5s FATAL: no frames in %s" % (tag, ref)); bad = 1; continue
    ok = (d == 0.0 and dl == 0.0)
    print("[equiv] cpu  %-5s snapshot vs standalone: max|d|=%.3g  %s d=%.3g  over %d frames -> %s"
          % (tag, d, lk, dl, nf, "BIT-IDENTICAL" if ok else "MISMATCH"))
    if not ok:
        print("[equiv] cpu  %-5s FATAL: the snapshot mechanism itself is not equivalent" % tag)
        bad = 1

# (2) HARDWARE FLOOR, cuda
d_rr, dl_rr, nf = dist(os.path.join(smoke, "gan_ref"), os.path.join(smoke, "gan_ref2"), "lossG_last")
d_rs, dl_rs, _ = dist(os.path.join(smoke, "gan_ref"), os.path.join(smoke, "gan_sw_s002k"), "lossG_last")
print("[equiv] cuda DCGAN standalone vs standalone (same command, 2 processes): max|d|=%.3g  lossG d=%.3g"
      % (d_rr, dl_rr))
print("[equiv] cuda DCGAN standalone vs sweep snapshot:                          max|d|=%.3g  lossG d=%.3g"
      % (d_rs, dl_rs))
tol = max(3.0 * d_rr, 1e-5)
if d_rs > tol:
    print("[equiv] cuda DCGAN FATAL: %.3g exceeds 3x the run-to-run floor (%.3g)" % (d_rs, tol))
    bad = 1
else:
    print("[equiv] cuda DCGAN OK: the sweep is within the run-to-run reproducibility floor of "
          "the unmodified script on this GPU (cuDNN conv-transpose backward is non-deterministic; "
          "the CPU test above is what proves the mechanism)")

d, dl, nf = dist(os.path.join(smoke, "rbm_ref"), os.path.join(smoke, "rbm_sw_s002k"), "recon_mse_last")
ok = (d == 0.0 and dl == 0.0)
print("[equiv] cuda RBM   snapshot vs standalone: max|d|=%.3g  recon d=%.3g over %d frames -> %s"
      % (d, dl, nf, "BIT-IDENTICAL" if ok else "MISMATCH"))
if not ok:
    print("[equiv] cuda RBM   FATAL: the RBM has no non-deterministic op and must match exactly")
    bad = 1
sys.exit(bad)
PYCHK
[ $? -eq 0 ] || { echo "[FATAL] the snapshot design is not equivalent to standalone runs -- refusing to run the sweep"; exit 1; }
rm -rf "$SMOKE"
echo "[$(ts)] pre-checks passed"

# ---------------------------------------------------------------- the four training runs
rm -rf "$OUT"; mkdir -p "$OUT"
COMMON="--n-holdout 200 --n-gen 200 --device cuda --seed 0"

echo "[$(ts)] START 4 sweep runs (one GPU, run concurrently -- all four are small)"
CUDA_VISIBLE_DEVICES=0 "$PY" baseline_rbm_sweep.py $COMMON --n-train 11 \
    --snapshots 20000 40000 80000 160000 --out-prefix "$OUT/rbm_n11" \
    > logs_lens/conv_rbm_n11.log 2>&1 &
p1=$!
CUDA_VISIBLE_DEVICES=0 "$PY" baseline_rbm_sweep.py $COMMON --n-train 100 \
    --snapshots 20000 40000 80000 160000 --out-prefix "$OUT/rbm_n100" \
    > logs_lens/conv_rbm_n100.log 2>&1 &
p2=$!
CUDA_VISIBLE_DEVICES=0 "$PY" baseline_dcgan_sweep.py $COMMON --n-train 11 \
    --steps 60000 --snap-every 5000 --out-prefix "$OUT/dcgan_n11" \
    > logs_lens/conv_dcgan_n11.log 2>&1 &
p3=$!
CUDA_VISIBLE_DEVICES=0 "$PY" baseline_dcgan_sweep.py $COMMON --n-train 100 \
    --steps 60000 --snap-every 5000 --out-prefix "$OUT/dcgan_n100" \
    > logs_lens/conv_dcgan_n100.log 2>&1 &
p4=$!
fail=0
for pv in "rbm_n11:$p1" "rbm_n100:$p2" "dcgan_n11:$p3" "dcgan_n100:$p4"; do
    wait "${pv#*:}" || { echo "[WARN] ${pv%%:*} failed -- see logs_lens/conv_${pv%%:*}.log"; fail=1; }
done
NDIR=$(find "$OUT" -mindepth 1 -maxdepth 1 -type d | wc -l)
echo "[$(ts)] training done (fail=$fail); run dirs written: $NDIR (expected 32)"
[ "$NDIR" -eq 32 ] || { echo "[WARN] expected 32 snapshot directories"; fail=1; }

# ---------------------------------------------------------------- evaluate: same flow as Table 2
echo "[$(ts)] START evaluate_r3b over $OUT"
"$PY" evaluate_r3b.py --outputs-dir "$OUT" --n-holdout 200 --n-train-max 100 --seed 0 \
    --results "$REPO/results_convergence_eval.jsonl" \
    > logs_lens/evaluate_convergence.log 2>&1 \
    || { echo "[WARN] evaluate_r3b failed"; tail -40 logs_lens/evaluate_convergence.log; fail=1; }
tail -80 logs_lens/evaluate_convergence.log

# ---------------------------------------------------------------- aggregate + best-of-steps
echo "[$(ts)] START aggregate_convergence"
"$PY" aggregate_convergence.py --eval-results results_convergence_eval.jsonl \
    --out results_convergence.jsonl --expect 32 \
    > logs_lens/aggregate_convergence.log 2>&1 \
    || { echo "[WARN] aggregate_convergence failed"; tail -40 logs_lens/aggregate_convergence.log; fail=1; }
cat logs_lens/aggregate_convergence.log

echo "[$(ts)] DONE submit_baseline_convergence.sbatch (fail=$fail)"
exit "$fail"
