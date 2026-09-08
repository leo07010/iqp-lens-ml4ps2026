#!/bin/bash
#SBATCH -J iqp_ctrl34
#SBATCH -A <YOUR_ACCOUNT>
#SBATCH -p dev
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH -o <REPO_ROOT>/code/logs_controlled/ctrl34-%j.out
#SBATCH -e <REPO_ROOT>/code/logs_controlled/ctrl34-%j.err
#
# ===========================================================================================
# The two CONTROLLED SYNTHETIC studies of arXiv 2603.06440 -- no lens data involved.
#
#   Fig. 3 (Table 3, paper.txt:2766-2789)  n = 16; data-generating IQP circuits with locality
#       <= 4 and G^data in {140,280,420,560,700,840,1050}; SPSA on their parameters to spread
#       I_QCLI; 10^4 bitstrings per target = the target dataset; three fixed <=2-local learners
#       with G_n in {50,100,150} fitted by Adam, 5000 iterations, lr 1e-4, on the exact
#       closed-form MMD^2 of Appendix G.  Claim under test: the UPPER ENVELOPE of the achieved
#       MMD falls as the target's I_QCLI rises, and drops further as G_n rises.
#
#   Fig. 4 (Table 4, paper.txt:2809-2831)  n in {8,12,16}; 150 random 2-local gates; SPSA
#       maximising I_QCLI ONLY; ~20000 circuits sampled along the optimisation, each scored on
#       (I_QCLI, I_CCI).  Claim under test: I_CCI rises with I_QCLI although it is never in
#       the objective.
#
# ---------------------------------------------------------------------------- run order
#   0. fail-fast import check + py_compile of all five modules + test_controlled.py
#      (10 tests: 4 numerical cross-checks against the audited iqp_exact / diagnostics_qcli_cci
#      references, plus end-to-end smokes of both scripts and the plotter).  MEASURED 9.7 s
#      wall on the login node -- if it does not pass, the job aborts before touching the GPU.
#   1. GPU timing probe (see below) -- prints the real ms/step and SIZES Fig. 3 from it.
#   2. Fig. 4 first (cheap, ~5 min): if anything overruns later, this result is already on disk.
#   3. Fig. 3.  Its jsonl is flushed row by row and the fits are written learner by learner,
#      so even a wall-clock kill leaves the completed learners usable.
#   4. plot_fig34.py -> docs/figures/iqp_fig3_support_mismatch.png, iqp_fig4_qcli_cci.png
#
# SUPERSEDED FOR FIG. 3 (2026-09-08).  This file is kept as the faithful record of run 356097,
# which produced a good Fig. 4 but a degenerate Fig. 3 (all 140 targets landed in a 0.06-wide
# strip of the 10^4-sample QCLI axis, below that estimator's own uniform null).  The Fig. 3
# invocation below therefore pins --target-optimizer spsa explicitly, because the script's
# default has since moved to adam.  Re-run Fig. 3 with submit_fig3_v2.sbatch, not with this.
#
# ------------------------------------------------------------------------- time budget
# MEASURED on the login node, 2026-09-07, /usr/bin/python3 (3.9.21, torch 2.8.0+cu128),
# 8 torch threads, n = 16, float64 -- these are the CPU numbers the probe below replaces with
# the real GPU ones at run time:
#
#   fig4  SPSA step (2 exact QCLI evaluations), batch 3          69.2 ms/step
#   fig4  per-step record (cci_batch, batch 3)                    8.6 ms       (numpy path;
#         on GPU only the n + n(n-1)/2 = 136 Walsh coefficients and H_joint cross PCIe)
#   fig3  target-diversity SPSA step, batch 20                  149.9 ms/step  (93.5 in float32)
#   fig3  Adam MMD step, batch 20                               223.4 ms/step
#   fig3  Adam MMD step, batch 70                              1330.7 ms/step  (19.0 ms/target)
#
# Extrapolated cost at a DELIBERATELY PESSIMISTIC 5x GPU speed-up over 8 CPU threads (the
# FWHT butterfly is memory-bound, so 20-40x is the realistic figure on an H100):
#
#   fig4   3 sizes x 2 SPSA arms x 3333 steps x (69.2 + 8.6)/5 ms          ~  5 min
#   fig3   target build:  7 gate counts x 1200 steps x 149.9/5 ms          ~  4 min
#   fig3   sampling + three-sigma rule + QCLI/CCI of 140 targets           ~  1 min
#   fig3   training:  3 learners x 140 targets x 5000 steps x 19.0/5 ms    ~133 min  <- the risk
#   plot                                                                   < 1 min
#
# 133 min of training at the pessimistic bound would leave almost no headroom in a 3 h job, and
# the honest number is unknown until a GPU is in hand.  So step 1 MEASURES the real Adam
# ms/step on this node and derives --n-targets from it, capped at the 20 per gate count that
# this study was designed for and floored at 6 (below that the envelope has too few points per
# 0.09-wide bin to be meaningful).  At the realistic 20x speed-up the probe will return
# ~1.0 ms/target/step and the cap of 20 applies (training ~33 min, whole job ~45 min).
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
ts() { date +"%Y-%m-%d %H:%M:%S"; }

echo "[SLURM] $(ts) job=${SLURM_JOB_ID:-none} node=$(hostname) part=${SLURM_JOB_PARTITION:-none}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

# ---------------------------------------------------------------- 0. fail-fast + pre-checks
"$PY" - <<'EOF' || { echo "[FATAL] python environment broken (numpy/torch/scipy/matplotlib). Aborting before using any dev budget."; exit 1; }
import sys
import numpy, torch, scipy, matplotlib
print("[env] python %s | numpy %s | torch %s | scipy %s | matplotlib %s"
      % (sys.version.split()[0], numpy.__version__, torch.__version__,
         scipy.__version__, matplotlib.__version__))
assert torch.cuda.is_available(), "no CUDA device visible"
print("[env] cuda device: %s  (count %d)" % (torch.cuda.get_device_name(0), torch.cuda.device_count()))
EOF

echo "[pre] $(ts) py_compile"
"$PY" -m py_compile common.py fig3_support_mismatch.py fig4_qcli_max.py plot_fig34.py test_controlled.py \
    || { echo "[FATAL] py_compile failed"; exit 1; }

echo "[pre] $(ts) test_controlled.py (numerical cross-checks + end-to-end smokes)"
"$PY" test_controlled.py 2>&1 | tail -60
if [ "${PIPESTATUS[0]}" -ne 0 ]; then echo "[FATAL] test_controlled.py failed"; exit 1; fi

# ---------------------------------------------------------------- 1. GPU timing probe
echo "[probe] $(ts) measuring the real GPU step times on this node"
PROBE_LOG="$LOGS/probe-${SLURM_JOB_ID:-0}.txt"
"$PY" - > "$PROBE_LOG" 2>&1 <<'EOF'
import os, sys, time
import numpy as np, torch
sys.path.insert(0, os.getcwd())
import common
from common import (BatchIQP, kernel_c_by_weight, mmd2_from_expvals, popcount_index,
                    qcli_torch, random_gate_masks, spsa_maximize, cci_batch)

dev, n, B = "cuda", 16, 70
dt = torch.float64
pc = popcount_index(n, dev)
rng = np.random.default_rng(0)

# (a) Fig. 3 Adam MMD step at the real batch size
mt = np.stack([random_gate_masks(n, 4, 560, rng)[0] for _ in range(B)])
simT = BatchIQP(n, mt, device=dev, dtype=dt)
ez = common.fwht(simT.probs(simT.init_theta(B, rng, std=0.5))).detach()
ml, _ = random_gate_masks(n, 2, 100, rng)
siml = BatchIQP(n, ml, device=dev, dtype=dt)
th = siml.init_theta(B, rng, std=0.01).requires_grad_(True)
cw = torch.as_tensor(np.stack([kernel_c_by_weight(n, [1.5, 2.2, 2.8]) for _ in range(B)]),
                     dtype=dt, device=dev)
opt = torch.optim.Adam([th], lr=1e-4)
for _ in range(5):                                   # warm-up / autotune
    opt.zero_grad(set_to_none=True)
    mmd2_from_expvals(siml.expvals(th), ez, cw, n, pc).sum().backward(); opt.step()
torch.cuda.synchronize(); t0 = time.time()
for _ in range(30):
    opt.zero_grad(set_to_none=True)
    mmd2_from_expvals(siml.expvals(th), ez, cw, n, pc).sum().backward(); opt.step()
torch.cuda.synchronize()
ms_adam = 1e3 * (time.time() - t0) / 30.0

# (b) Fig. 4 SPSA step + per-step record at batch 3
m4 = np.stack([random_gate_masks(n, 2, 150, rng)[0] for _ in range(3)])
s4 = BatchIQP(n, m4, device=dev, dtype=dt)
th4 = s4.init_theta(3, rng, std=0.05)
gen = torch.Generator(device=torch.device(dev)); gen.manual_seed(1)
obj = lambda t: qcli_torch(s4.probs(t), n, pc)
spsa_maximize(obj, th4, 5, a=0.5, c=0.2, generator=gen)
torch.cuda.synchronize(); t0 = time.time()
spsa_maximize(obj, th4, 30, a=0.5, c=0.2, generator=gen)
torch.cuda.synchronize()
ms_spsa = 1e3 * (time.time() - t0) / 30.0
t0 = time.time()
for _ in range(10):
    cci_batch(s4.probs(th4), n)
ms_cci = 1e3 * (time.time() - t0) / 10.0

# (c) Fig. 3 target-diversity SPSA step at batch 20 (4-local, 1050 gates = the worst case)
m3 = np.stack([random_gate_masks(n, 4, 1050, rng)[0] for _ in range(20)])
s3 = BatchIQP(n, m3, device=dev, dtype=dt)
th3 = s3.init_theta(20, rng, std=0.05)
obj3 = lambda t: qcli_torch(s3.probs(t), n, pc)
spsa_maximize(obj3, th3, 5, a=1.0, c=0.2, generator=gen)
torch.cuda.synchronize(); t0 = time.time()
spsa_maximize(obj3, th3, 20, a=1.0, c=0.2, generator=gen)
torch.cuda.synchronize()
ms_tgt = 1e3 * (time.time() - t0) / 20.0

print("[probe] n=16 float64 GPU: Adam MMD step (batch 70) %.2f ms  (= %.3f ms/target/step)"
      % (ms_adam, ms_adam / 70.0))
print("[probe] n=16 float64 GPU: fig4 SPSA step (batch 3) %.2f ms; cci_batch %.2f ms"
      % (ms_spsa, ms_cci))
print("[probe] n=16 float64 GPU: fig3 target SPSA step (batch 20, 1050 gates) %.2f ms" % ms_tgt)
print("PROBE %.4f %.4f %.4f %.4f" % (ms_adam, ms_spsa, ms_cci, ms_tgt))
EOF
cat "$PROBE_LOG"
MS_ADAM=$(grep -m1 '^PROBE ' "$PROBE_LOG" | awk '{print $2}')
if [ -z "${MS_ADAM:-}" ]; then
    echo "[WARN] the probe did not return a number; falling back to the design default n-targets=12"
    NT=12
else
    # 3 learners x 7 gate counts x NT targets x 5000 Adam steps must fit FIG3_TRAIN_BUDGET_S.
    NT=$("$PY" -c "
ms_per_target = float('$MS_ADAM') / 70.0
budget_s = 6000.0                      # 100 min of the 180 min wall, the rest is headroom
per_target_s = 3 * 5000 * ms_per_target / 1000.0
nt = int(budget_s / (7.0 * per_target_s))
print(max(6, min(20, nt)))
")
fi
echo "[probe] $(ts) --n-targets set to $NT per gate count (7 x $NT targets, cap 20, floor 6)"

# ---------------------------------------------------------------- 2. Fig. 4 (cheap, runs first)
F4="$OUT/results_fig4.jsonl"
echo "[fig4] $(ts) start -> $F4"
"$PY" fig4_qcli_max.py \
    --sizes 8,12,16 --gates 150 --locality 2 \
    --seeds 3 --steps 3333 --record-every 1 --spsa-a 0.5,2.0 --spsa-c 0.2 \
    --init-std 0.05 --dtype float64 --device cuda --seed 20260907 \
    --progress-every 500 --out "$F4"
RC4=$?
echo "[fig4] $(ts) exit=$RC4  rows=$(wc -l < "$F4" 2>/dev/null || echo 0)"

# ---------------------------------------------------------------- 3. Fig. 3
F3="$OUT/results_fig3.jsonl"
echo "[fig3] $(ts) start -> $F3"
"$PY" fig3_support_mismatch.py \
    --n 16 --gates-target 140,280,420,560,700,840,1050 --learner-gates 50,100,150 \
    --locality-target 4 --locality-learner 2 \
    --n-targets "$NT" --n-samples 10000 \
    --target-optimizer spsa \
    --target-steps 1200 --target-snapshots 80 --spsa-a 1.0 --spsa-c 0.2 --target-init-std 0.05 \
    --steps 5000 --lr 1e-4 --learner-init-std 0.01 --train-batch 70 \
    --null-reps 5 --dtype float64 --device cuda --seed 20260907 \
    --progress-every 500 --out "$F3"
RC3=$?
echo "[fig3] $(ts) exit=$RC3  rows=$(wc -l < "$F3" 2>/dev/null || echo 0)"

# ---------------------------------------------------------------- 4. figures
echo "[plot] $(ts) rendering"
"$PY" plot_fig34.py --fig3 "$F3" --fig4 "$F4" \
    --out-fig3 "$FIGS/iqp_fig3_support_mismatch.png" \
    --out-fig4 "$FIGS/iqp_fig4_qcli_cci.png" \
    --x sample --yscale log --bin-width 0.09
RCP=$?

echo "[done] $(ts) fig4=$RC4 fig3=$RC3 plot=$RCP"
ls -la "$OUT" "$FIGS/iqp_fig3_support_mismatch.png" "$FIGS/iqp_fig4_qcli_cci.png" 2>&1
exit 0
