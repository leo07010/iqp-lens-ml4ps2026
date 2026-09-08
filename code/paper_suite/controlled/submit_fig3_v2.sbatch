#!/bin/bash
#SBATCH -J iqp_fig3v2
#SBATCH -A <YOUR_ACCOUNT>
#SBATCH -p dev
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -o <REPO_ROOT>/code/logs_controlled/fig3v2-%j.out
#SBATCH -e <REPO_ROOT>/code/logs_controlled/fig3v2-%j.err
#
# ===========================================================================================
# Fig. 3 of arXiv 2603.06440, SECOND ATTEMPT.  Fig. 4 is untouched (job 356097 produced it
# correctly); this job runs the Fig. 3 study and its two plots only.
#
# ------------------------------------------------------------------ what went wrong in v1
# Run 356097 built 140 targets whose 10^4-sample I_QCLI all fell in 0.000-0.063, BELOW the
# i.i.d.-uniform null of 0.070 at the same M -- no spread, empty envelope panel, and the
# residual trend inside that sliver ran the wrong way.  Two independent causes, both fixed
# here and both measured, not guessed:
#
#   (1) SPSA does not climb at this parameter count.  Its 2-evaluation gradient estimate has
#       variance growing with the number of parameters, and the v1 target rows show the
#       collapse directly: exact I_QCLI reached 0.231 at G^data = 140 but only 0.0037 at
#       G^data = 560 and 0.0082 at G^data = 1050 -- i.e. the MORE gates, the LESS it moved.
#       FIX: --target-optimizer adam maximises the SAME exact I_QCLI by autograd (the exact
#       FWHT simulator makes the objective differentiable), which is insensitive to the
#       parameter count.  Measured on the login node, n=16, 4-local, batch 3, 600 steps:
#           G= 140  ->  I_QCLI 0.345      G= 560  ->  0.382      G=1050  ->  0.408
#       and the plateau is reached inside ~50 steps.
#
#   (2) The x-axis was the 10^4-sample I_QCLI.  At n=16 that is 10^4 samples over 2^16 = 65536
#       outcomes, and the estimator's own i.i.d.-uniform floor (0.070) sits above the entire
#       band the targets occupy on it.  FIX: the target levels are laid out on, and the figure
#       is drawn against, the EXACT circuit I_QCLI; the sample value is still recorded for
#       every target and gets its own second figure, so the compression is documented rather
#       than hidden.
#
# ------------------------------------------------------- the ceiling, and why it is not 0.7
# The paper says its targets span up to ~0.7.  We do not reach that at n=16 and it is not an
# optimiser failure: at G^data = 560 the exact I_QCLI plateaus at 0.381-0.382 from Adam lr
# 0.05 AND lr 0.20, from N(0, 0.05), N(0, 0.5) and N(0, 2.0) initialisations -- nine runs, all
# landing on the same value.  The ceiling rises with gate count (0.345 -> 0.408 for 140 ->
# 1050 gates), so 1050-gate targets should top out near 0.41.  For contrast, the same code at
# n = 8 reaches 0.98 (the indicator's degenerate maximum, attained by the exact uniform
# distribution), so the ceiling is a property of what a <=4-local IQP circuit can do to the
# order spectrum at 16 qubits, not of the optimiser.  The script prints and records the
# attained maximum per gate count; whatever it is, the 20 targets of that gate count are laid
# out uniformly across [0, attained], which is the coverage Table 3 asks for.
#
# ------------------------------------------------------------------------- what is recorded
#   record="target"  one row per target: I_QCLI_exact + I_CCI_exact (circuit pmf),
#                    I_QCLI + I_CCI (10^4-sample pmf), sigmas, trajectory/snapshot provenance
#   record="fit"     one row per (target, learner):
#                      final_mmd2               trained on the 10^4-sample pmf (paper protocol)
#                      final_mmd2_exacttrained  CONTROL: same learner, same init, trained on
#                                               the exact target pmf
#                      final_mmd2_vs_exact      the sample-trained theta scored against the
#                                               exact pmf (free; separates finite-sample error
#                                               from genuine support mismatch)
#                      mmd2_floor_proxy         Prop. II.2 proxy: kernel-weighted data spectral
#                                               mass outside the learner's tunable support
#   record="null"    i.i.d.-uniform I_QCLI at the matched M
#   record="meta"    every setting, plus the attained exact-I_QCLI span
#
# ------------------------------------------------------------------------- time budget
# MEASURED on the GPU in run 356097 (n=16, float64, same node class):
#   Adam MMD step, batch 70                                    14.6 ms/step
#   SPSA step for target generation, batch 20, 1050 gates       4.9 ms/step
# An Adam target step is one forward + one backward ~ 1.5x an SPSA step ~ 7.4 ms.
#
#   target generation  7 gate counts x 400 steps x 7.4 ms                    ~  0.4 min
#   fits               3 learners x 2 arms x 2 chunks x 5000 steps x 14.6 ms ~ 14.6 min
#   sampling / sigmas / QCLI+CCI of 140 targets / uniform null               ~  1 min
#   pre-checks (py_compile + test_controlled.py, MEASURED 18.2 s) + 2 plots  ~  1 min
#   TOTAL ~ 17 min.  Requesting 1 h (~3.5x headroom).
# ===========================================================================================
set -uo pipefail

REPL=<REPO_ROOT>/code
DIR="$REPL/paper_suite/controlled"
REPO=<REPO_ROOT>
OUT="$REPL/outputs_controlled"
LOGS="$REPL/logs_controlled"
FIGS="$REPO/docs/figures"
cd "$DIR"
mkdir -p "$OUT" "$LOGS" "$FIGS"

module purge; module load cuda/13.0
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
PY=/usr/bin/python3
F3="$OUT/results_fig3_v2.jsonl"
DONE="$OUT/fig3_v2.done"
ts() { date +"%Y-%m-%d %H:%M:%S"; }

rm -f "$DONE"
echo "[SLURM] $(ts) job=${SLURM_JOB_ID:-none} node=$(hostname) part=${SLURM_JOB_PARTITION:-none}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

# ---------------------------------------------------------------- fail-fast + pre-checks
"$PY" - <<'EOF' || { echo "[FATAL] python environment broken. Aborting before using any dev budget."; exit 1; }
import sys
import numpy, torch, scipy, matplotlib
print("[env] python %s | numpy %s | torch %s | scipy %s | matplotlib %s"
      % (sys.version.split()[0], numpy.__version__, torch.__version__,
         scipy.__version__, matplotlib.__version__))
assert torch.cuda.is_available(), "no CUDA device visible"
print("[env] cuda device: %s" % torch.cuda.get_device_name(0))
EOF

echo "[pre] $(ts) py_compile"
"$PY" -m py_compile common.py fig3_support_mismatch.py fig4_qcli_max.py plot_fig34.py test_controlled.py \
    || { echo "[FATAL] py_compile failed"; exit 1; }

echo "[pre] $(ts) test_controlled.py"
"$PY" test_controlled.py 2>&1 | tail -40
if [ "${PIPESTATUS[0]}" -ne 0 ]; then echo "[FATAL] test_controlled.py failed"; exit 1; fi

# ---------------------------------------------------------------- Fig. 3 v2
echo "[fig3v2] $(ts) start -> $F3"
"$PY" fig3_support_mismatch.py \
    --n 16 --gates-target 140,280,420,560,700,840,1050 --learner-gates 50,100,150 \
    --locality-target 4 --locality-learner 2 \
    --n-targets 20 --n-samples 10000 \
    --target-optimizer adam --target-lr 0.05 --target-steps 400 --target-snapshots 400 \
    --target-qcli-max 0.7 --target-init-std 0.05 \
    --steps 5000 --lr 1e-4 --learner-init-std 0.01 --train-batch 70 \
    --exact-target-control --null-reps 5 \
    --dtype float64 --device cuda --seed 20260907 \
    --bin-width-report 0.09 --progress-every 1000 \
    --out "$F3" --done-marker "$DONE"
RC3=$?
echo "[fig3v2] $(ts) exit=$RC3  rows=$(wc -l < "$F3" 2>/dev/null || echo 0)"
if [ "$RC3" -ne 0 ]; then echo "[FATAL] fig3 v2 failed; not plotting"; exit "$RC3"; fi

# ---------------------------------------------------------------- figures
# main figure: exact-pmf x-axis (the axis on which the targets are actually separated)
echo "[plot] $(ts) main figure (exact I_QCLI axis)"
"$PY" plot_fig34.py --fig3 "$F3" \
    --out-fig3 "$FIGS/iqp_fig3_support_mismatch.png" \
    --x exact --yscale log --bin-width 0.09
RCA=$?

# companion figure: the paper's literal 10^4-sample x-axis, kept so the compression is visible
echo "[plot] $(ts) companion figure (10^4-sample I_QCLI axis)"
"$PY" plot_fig34.py --fig3 "$F3" \
    --out-fig3 "$FIGS/iqp_fig3_support_mismatch_sampleaxis.png" \
    --x sample --yscale log --bin-width 0.09
RCB=$?

echo "[done] $(ts) fig3=$RC3 plot_exact=$RCA plot_sample=$RCB"
echo "--- done marker ---"; cat "$DONE" 2>&1
ls -la "$F3" "$DONE" "$FIGS/iqp_fig3_support_mismatch.png" \
       "$FIGS/iqp_fig3_support_mismatch_sampleaxis.png" 2>&1
exit 0
