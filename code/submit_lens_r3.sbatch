#!/bin/bash
#SBATCH -J iqp_lens_r3
#SBATCH -A <YOUR_ACCOUNT>
#SBATCH -p 8gpus
#SBATCH --gres=gpu:8
#SBATCH -c 32
#SBATCH --mem=128G
#SBATCH -t 02:30:00
#SBATCH -o <REPO_ROOT>/code/logs_lens/sbatch-r3-%j.out
#SBATCH -e <REPO_ROOT>/code/logs_lens/sbatch-r3-%j.err
#
# Round 3 of the IQP generative study on DeepLense Model II.
# Pre-registration: docs/IQP_LENS_ROUND3_PREREGISTRATION.md (goal B: can this model family
# generate PHYSICALLY PLAUSIBLE lens images), including the pseudo-time (P15), hardware-
# compatible MMDflat (P16) and latent-prior (P17) addenda, plus the intensity-aligned
# "Scoring note" (every generated/training/held-out image individually rescaled to the
# held-out mean total before PDF-JS/MMD2/NN are recomputed, next to the primary scores).
# Written and checked BEFORE this ran; predictions P10-P17 and the decision rule are in it.
#
# Round 3 touches NOTHING from rounds 1-2: every script it calls is a *_r3.py / codec10.py
# file, and the output tree is its own. Round 2's job 347344 uses train_iqp.py / data_lens.py /
# evaluate_lens.py, which are read-only here (train_iqp_r3.py imports exactly three pure
# helper functions from train_iqp.py and nothing else).
#
# ---------------------------------------------------------------------------- job list
# 14 arms x N in {11, 100} = 28 jobs, classes POOLED (no --lens-class in round 3).
#
#   arm label                    | script              | what it is
#   -----------------------------+---------------------+-----------------------------------
#   IQP-lowvar-joint-MMD         | train_iqp_r3.py     | PRIMARY: 10 qubits, all 1,024
#                                |                     | Z-strings, low-order/high-variance
#                                |                     | latent, joint training, closed-form MMD
#   IQP-lowvar-joint-KL          | train_iqp_r3.py     | objective ablation (exact KL)
#   IQP-hamming-joint-MMD        | train_iqp_r3.py     | latent-selection ablation (P12)
#   IQP-lowvar-paper-MMD         | train_iqp_r3.py     | protocol ablation: the paper's
#                                |                     | sequential Algorithm 1, all params
#                                |                     | trained in stage 1
#   IQP-lowvar-pseudotime-MMD    | train_iqp_r3.py     | ADDENDUM: Algorithm 1 along the e1
#                                |                     | ordering; CONDITIONAL generation
#   IQP-lowvar-joint-MMDflat     | train_iqp_r3.py     | ADDENDUM (P16): joint/MMD with FLAT
#                                |                     | Walsh weights c_s = 2^-10 -- hardware-
#                                |                     | compatible counterpart to the KL arm
#   IQP-lowvar-joint-MMD-prior8  | train_iqp_r3.py     | ADDENDUM (P17): primary config, latent
#                                |                     | prior truncated to PCA rank 8
#   IQP-lowvar-joint-MMD-resample| train_iqp_r3.py     | ADDENDUM (P17): primary config, latent
#                                |                     | prior = resample-with-replacement +
#                                |                     | 0.2x-spacing Gaussian jitter
#   RBM-10bit                    | baseline_rbm_r3.py  | 10 x 1000 Bernoulli RBM (11,010 par)
#   DCGAN-32                     | baseline_dcgan_r3.py| 32x32 Radford stack (199,837 par)
#   CTRL-copy / CTRL-mean        | controls_r3.py      | memorisation ceiling / no-variation floor
#   CTRL-ring                    | controls_r3.py      | zero-learning 5-parameter physical model
#   CTRL-interp-e1               | controls_r3.py      | ADDENDUM: conditional zero-parameter control
#
# ------------------------------------------------------------------------- time budget
# MEASURED on the login node, 2026-09-06, /usr/bin/python3 with 4 torch threads, n = 10:
#
#   IQP joint  MMD : N = 11   5.13 ms/step   N = 100  23.36 ms/step   (batched over N images)
#   IQP joint  KL  : N = 11   4.33 ms/step   N = 100  18.70 ms/step
#   IQP paper/pseudotime (batch of 1 image): 3.49 ms/step
#   RBM  PCD-1     : N = 11   6.37 ms/step   N = 100  46.46 ms/step
#   DCGAN          : N = 11  14.54 ms/step   N = 100  27.58 ms/step
#   RBM generation (200 images x 512 chains x 200 Gibbs sweeps, chunk 25): 157 s
#   data load per job (N=100 + 200 held-out crops): ~10 s;  IQP generation: < 1 s
#
# => per job, CPU numbers used as the CONSERVATIVE bound (the FWHT butterfly at n = 10 is
#    launch-bound on GPU, so the sequential batch-of-1 arms do not speed up; the batched
#    joint arms and the RBM/DCGAN should be faster on GPU, never slower):
#
#      IQP-lowvar-paper-MMD       N = 11   3.5 min   N = 100  19.0 min   <- longest
#      IQP-lowvar-pseudotime-MMD  N = 11   3.5 min   N = 100  19.0 min   <- longest
#        (steps = 3e4 stage-1 + (N-1) x 3e3 stage-2 = 6.0e4 at N=11, 3.27e5 at N=100)
#      RBM-10bit                  N = 11   4.8 min   N = 100  18.6 min   (incl. 2.6 min Gibbs)
#      DCGAN-32                   N = 11   4.8 min   N = 100   9.2 min
#      IQP-lowvar-joint-MMD       N = 11   1.7 min   N = 100   7.8 min
#      IQP-hamming-joint-MMD      N = 11   1.7 min   N = 100   7.8 min
#      IQP-lowvar-joint-MMDflat   N = 11   1.7 min   N = 100   7.8 min   (same fwd/bwd shape as
#                                                                         joint-MMD, only c_s differs)
#      IQP-lowvar-joint-MMD-prior8   N = 11   1.7 min   N = 100   7.8 min   (prior8/resample only
#      IQP-lowvar-joint-MMD-resample N = 11   1.7 min   N = 100   7.8 min    change POST-training
#                                                                            generation, < 1 s extra)
#      IQP-lowvar-joint-KL        N = 11   1.5 min   N = 100   6.2 min
#      the four controls          < 0.5 min each at either N
#
# Jobs run 8-wide with `wait` between batches (a batch costs its slowest member), LONGEST
# FIRST:
#   batch 1 (8)  paper/pseudotime/RBM/DCGAN/joint x3 at N = 100 + DCGAN N = 11        ~19.0 min
#   batch 2 (8)  MMDflat/prior8/resample at N = 100 + paper/pseudotime/RBM at N = 11
#                + 2 controls at N = 100                                              ~ 7.8 min
#   batch 3 (8)  joint-MMD/hamming/joint-KL + MMDflat/prior8/resample at N = 11
#                + 2 controls at N = 100                                              ~ 1.7 min
#   batch 4 (4)  the remaining 4 controls at N = 11                                   ~ 0.5 min
#   + pre-checks (~1 min: py_compile + test_r3.py, MEASURED at 11.9 s wall, now includes the
#     addendum unit/smoke tests and the intensity-aligned block) + a 5-step CUDA smoke
#     (~1 min) + evaluate_r3 over 28 run dirs, two tables (primary + intensity-aligned),
#     conv_mmd2 now computed twice per row (~1-2 min)
#   TOTAL ~ 33 min at the conservative CPU bound.  Requesting 2.5 h (~4.5x headroom).
#
# Why a CUDA smoke before the batches: the whole round was developed and tested on the login
# node's CPU, so a device-placement bug would otherwise surface only after all 28 jobs had
# been dispatched. 5 steps at N = 3 costs under a minute and fails the whole script fast.
set -uo pipefail

REPO=<REPO_ROOT>/code
cd "$REPO"
mkdir -p logs_lens

module purge; module load cuda/13.0
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
PY=/usr/bin/python3
OUT="$REPO/outputs_lens_r3"
ts() { date +"%Y-%m-%d %H:%M:%S"; }

echo "[SLURM] job=$SLURM_JOB_ID node=$(hostname) part=$SLURM_JOB_PARTITION gpus=$($PY -c 'import torch;print(torch.cuda.device_count())')"

# ---------------------------------------------------------------- fail-fast: dataset + cache
"$PY" -c "import data_lens_r3 as d; m = d.build_cache(); print('[r3] cache ok, N=11 class counts =', m['train11_class_counts'], 'holdout halves =', m['holdout_firsthalf_class_counts'], m['holdout_secondhalf_class_counts'])" \
    || { echo "[FATAL] the round-3 cache could not be built (dataset missing?). Aborting before using any 8gpus budget."; exit 1; }

# ---------------------------------------------------------------- fail-fast: latent blocks
# Both latent files must exist BEFORE any arm starts: three of the five IQP arms read them,
# and regenerating them mid-run would silently change the pre-registered latent block.
for f in r3_latent_lowvar50.npy r3_latent_hamming50.npy; do
    [ -f "$f" ] || { echo "[FATAL] $f missing -- run '$PY latent_select_r3.py' first"; exit 1; }
done
"$PY" -c "
import latent_select_r3 as L, numpy as np
from iqp_exact import popcount_table
pc = popcount_table(10)
for name in ('lowvar', 'hamming'):
    m = L.load_latent_masks(name)
    print('[r3] latent %-8s: %d masks, |s| histogram %s' % (name, m.size, np.bincount(pc[m], minlength=11)[:3].tolist()))
" || { echo "[FATAL] latent blocks failed validation"; exit 1; }

# ---------------------------------------------------------------- fail-fast pre-checks
# Only the round-3 files are compiled here: round 2's job 347344 is still running against
# train_iqp.py / data_lens.py / evaluate_lens.py and this script must not touch them.
echo "[$(ts)] pre-check: py_compile the round-3 files"
for f in codec10.py data_lens_r3.py latent_select_r3.py train_iqp_r3.py controls_r3.py \
         baseline_rbm_r3.py baseline_dcgan_r3.py evaluate_r3.py test_r3.py; do
    "$PY" -m py_compile "$f" || { echo "[FATAL] py_compile failed for $f"; exit 1; }
done

echo "[$(ts)] pre-check: test_r3.py"
"$PY" test_r3.py > logs_lens/precheck_r3_test.log 2>&1 \
    || { echo "[FATAL] test_r3.py failed -- see logs_lens/precheck_r3_test.log"; tail -40 logs_lens/precheck_r3_test.log; exit 1; }
tail -3 logs_lens/precheck_r3_test.log

echo "[$(ts)] pre-check: 5-step CUDA smoke of every GPU arm"
rm -rf logs_lens/r3_cuda_smoke
CUDA_VISIBLE_DEVICES=0 "$PY" train_iqp_r3.py --n-train 3 --n-holdout 4 --n-gen 2 --steps 5 \
    --log-every 5 --device cuda --out logs_lens/r3_cuda_smoke/iqp > logs_lens/precheck_r3_cuda.log 2>&1 \
    || { echo "[FATAL] train_iqp_r3.py does not run on cuda"; tail -30 logs_lens/precheck_r3_cuda.log; exit 1; }
CUDA_VISIBLE_DEVICES=0 "$PY" train_iqp_r3.py --n-train 3 --n-holdout 4 --n-gen 2 --protocol pseudotime \
    --steps-core 5 --steps-lat 5 --log-every 5 --device cuda --out logs_lens/r3_cuda_smoke/pt >> logs_lens/precheck_r3_cuda.log 2>&1 \
    || { echo "[FATAL] --protocol pseudotime does not run on cuda"; tail -30 logs_lens/precheck_r3_cuda.log; exit 1; }
CUDA_VISIBLE_DEVICES=0 "$PY" baseline_rbm_r3.py --n-train 3 --n-holdout 4 --n-gen 2 --steps 5 \
    --log-every 5 --n-pool 1000 --gen-sweeps 5 --gen-keep 2 --gen-chunk 2 --device cuda \
    --out logs_lens/r3_cuda_smoke/rbm >> logs_lens/precheck_r3_cuda.log 2>&1 \
    || { echo "[FATAL] baseline_rbm_r3.py does not run on cuda"; tail -30 logs_lens/precheck_r3_cuda.log; exit 1; }
CUDA_VISIBLE_DEVICES=0 "$PY" baseline_dcgan_r3.py --n-train 3 --n-holdout 4 --n-gen 2 --steps 5 \
    --log-every 5 --device cuda --out logs_lens/r3_cuda_smoke/dcgan >> logs_lens/precheck_r3_cuda.log 2>&1 \
    || { echo "[FATAL] baseline_dcgan_r3.py does not run on cuda"; tail -30 logs_lens/precheck_r3_cuda.log; exit 1; }
rm -rf logs_lens/r3_cuda_smoke
echo "[$(ts)] pre-checks passed"

rm -rf "$OUT"
mkdir -p "$OUT"

# ---------------------------------------------------------------- build the job list: TAG|DEVICE|CMD
JOBS=()
add_job() { JOBS+=("$1|$2|$3"); }

IQP_COMMON="--n-holdout 200 --n-gen 200 --device cuda --seed 0 --sigmas auto3"
RBM_COMMON="--n-holdout 200 --n-gen 200 --device cuda --seed 0"
GAN_COMMON="--n-holdout 200 --n-gen 200 --device cuda --seed 0"
CTL_COMMON="--n-holdout 200 --n-gen 200 --seed 0"

# --- batch 1: the eight longest -------------------------------------------------------
add_job "iqp_paper_n100" cuda \
    "$PY train_iqp_r3.py $IQP_COMMON --n-train 100 --latent lowvar --protocol paper \
     --objective mmd --steps-core 30000 --steps-lat 3000 --out $OUT/iqp_paper_n100"
add_job "iqp_pseudotime_n100" cuda \
    "$PY train_iqp_r3.py $IQP_COMMON --n-train 100 --latent lowvar --protocol pseudotime \
     --objective mmd --steps-core 30000 --steps-lat 3000 --out $OUT/iqp_pseudotime_n100"
add_job "rbm_n100" cuda \
    "$PY baseline_rbm_r3.py $RBM_COMMON --n-train 100 --steps 20000 --out $OUT/rbm_n100"
add_job "dcgan_n100" cuda \
    "$PY baseline_dcgan_r3.py $GAN_COMMON --n-train 100 --steps 20000 --out $OUT/dcgan_n100"
add_job "iqp_joint_mmd_n100" cuda \
    "$PY train_iqp_r3.py $IQP_COMMON --n-train 100 --latent lowvar --protocol joint \
     --objective mmd --steps 20000 --out $OUT/iqp_joint_mmd_n100"
add_job "iqp_hamming_n100" cuda \
    "$PY train_iqp_r3.py $IQP_COMMON --n-train 100 --latent hamming --protocol joint \
     --objective mmd --steps 20000 --out $OUT/iqp_hamming_n100"
add_job "iqp_joint_kl_n100" cuda \
    "$PY train_iqp_r3.py $IQP_COMMON --n-train 100 --latent lowvar --protocol joint \
     --objective kl --steps 20000 --out $OUT/iqp_joint_kl_n100"
add_job "dcgan_n11" cuda \
    "$PY baseline_dcgan_r3.py $GAN_COMMON --n-train 11 --steps 20000 --out $OUT/dcgan_n11"

# --- batch 2: the three new addendum arms at N=100 + the rest of the N=11 longest -------
add_job "iqp_mmdflat_n100" cuda \
    "$PY train_iqp_r3.py $IQP_COMMON --n-train 100 --latent lowvar --protocol joint \
     --objective mmdflat --steps 20000 --out $OUT/iqp_mmdflat_n100"
add_job "iqp_prior8_n100" cuda \
    "$PY train_iqp_r3.py $IQP_COMMON --n-train 100 --latent lowvar --protocol joint \
     --objective mmd --prior gauss8 --steps 20000 --out $OUT/iqp_prior8_n100"
add_job "iqp_resample_n100" cuda \
    "$PY train_iqp_r3.py $IQP_COMMON --n-train 100 --latent lowvar --protocol joint \
     --objective mmd --prior resample --steps 20000 --out $OUT/iqp_resample_n100"
add_job "iqp_paper_n11" cuda \
    "$PY train_iqp_r3.py $IQP_COMMON --n-train 11 --latent lowvar --protocol paper \
     --objective mmd --steps-core 30000 --steps-lat 3000 --out $OUT/iqp_paper_n11"
add_job "iqp_pseudotime_n11" cuda \
    "$PY train_iqp_r3.py $IQP_COMMON --n-train 11 --latent lowvar --protocol pseudotime \
     --objective mmd --steps-core 30000 --steps-lat 3000 --out $OUT/iqp_pseudotime_n11"
add_job "rbm_n11" cuda \
    "$PY baseline_rbm_r3.py $RBM_COMMON --n-train 11 --steps 20000 --out $OUT/rbm_n11"
add_job "ctrl_copy_n100" cpu \
    "$PY controls_r3.py $CTL_COMMON --arm copy --n-train 100 --out $OUT/ctrl_copy_n100"
add_job "ctrl_mean_n100" cpu \
    "$PY controls_r3.py $CTL_COMMON --arm mean --n-train 100 --out $OUT/ctrl_mean_n100"

# --- batch 3: the cheap N=11 IQP arms (primary set + the three addendum arms) -----------
add_job "iqp_joint_mmd_n11" cuda \
    "$PY train_iqp_r3.py $IQP_COMMON --n-train 11 --latent lowvar --protocol joint \
     --objective mmd --steps 20000 --out $OUT/iqp_joint_mmd_n11"
add_job "iqp_hamming_n11" cuda \
    "$PY train_iqp_r3.py $IQP_COMMON --n-train 11 --latent hamming --protocol joint \
     --objective mmd --steps 20000 --out $OUT/iqp_hamming_n11"
add_job "iqp_joint_kl_n11" cuda \
    "$PY train_iqp_r3.py $IQP_COMMON --n-train 11 --latent lowvar --protocol joint \
     --objective kl --steps 20000 --out $OUT/iqp_joint_kl_n11"
add_job "iqp_mmdflat_n11" cuda \
    "$PY train_iqp_r3.py $IQP_COMMON --n-train 11 --latent lowvar --protocol joint \
     --objective mmdflat --steps 20000 --out $OUT/iqp_mmdflat_n11"
add_job "iqp_prior8_n11" cuda \
    "$PY train_iqp_r3.py $IQP_COMMON --n-train 11 --latent lowvar --protocol joint \
     --objective mmd --prior gauss8 --steps 20000 --out $OUT/iqp_prior8_n11"
add_job "iqp_resample_n11" cuda \
    "$PY train_iqp_r3.py $IQP_COMMON --n-train 11 --latent lowvar --protocol joint \
     --objective mmd --prior resample --steps 20000 --out $OUT/iqp_resample_n11"
add_job "ctrl_ring_n100" cpu \
    "$PY controls_r3.py $CTL_COMMON --arm ring --n-train 100 --out $OUT/ctrl_ring_n100"
add_job "ctrl_interp_e1_n100" cpu \
    "$PY controls_r3.py $CTL_COMMON --arm interp_e1 --n-train 100 --out $OUT/ctrl_interp_e1_n100"

# --- batch 4: the remaining four controls at N=11 (all sub-second) ----------------------
add_job "ctrl_copy_n11" cpu \
    "$PY controls_r3.py $CTL_COMMON --arm copy --n-train 11 --out $OUT/ctrl_copy_n11"
add_job "ctrl_mean_n11" cpu \
    "$PY controls_r3.py $CTL_COMMON --arm mean --n-train 11 --out $OUT/ctrl_mean_n11"
add_job "ctrl_ring_n11" cpu \
    "$PY controls_r3.py $CTL_COMMON --arm ring --n-train 11 --out $OUT/ctrl_ring_n11"
add_job "ctrl_interp_e1_n11" cpu \
    "$PY controls_r3.py $CTL_COMMON --arm interp_e1 --n-train 11 --out $OUT/ctrl_interp_e1_n11"

echo "[$(ts)] ${#JOBS[@]} jobs queued (expected 28 = 14 arms x 2 N), running 8-wide"
if [ "${#JOBS[@]}" -ne 28 ]; then
    echo "[FATAL] job list is ${#JOBS[@]}, expected 28 -- the arm matrix changed, refusing to run"
    exit 1
fi

# ---------------------------------------------------------------- run 8-wide, wait between batches
GPU_SLOTS=8
i=0
pids=()
tags=()
fail=0
for job in "${JOBS[@]}"; do
    tag="${job%%|*}"
    rest="${job#*|}"
    device="${rest%%|*}"
    cmd="${rest#*|}"
    slot=$((i % GPU_SLOTS))
    echo "[$(ts)] START $tag (slot $slot, device=$device)"
    if [ "$device" = "cuda" ]; then
        (CUDA_VISIBLE_DEVICES=$slot bash -c "$cmd" > "logs_lens/r3_${tag}.log" 2>&1) &
    else
        (bash -c "$cmd" > "logs_lens/r3_${tag}.log" 2>&1) &
    fi
    pids+=($!)
    tags[$i]="$tag"
    i=$((i + 1))
    if [ $((i % GPU_SLOTS)) -eq 0 ]; then
        for j in "${!pids[@]}"; do
            wait "${pids[$j]}" || { echo "[WARN] job ${tags[$((i - GPU_SLOTS + j))]} failed, see logs_lens/r3_${tags[$((i - GPU_SLOTS + j))]}.log"; fail=1; }
        done
        pids=()
    fi
done
for j in "${!pids[@]}"; do
    idx=$((i - ${#pids[@]} + j))
    wait "${pids[$j]}" || { echo "[WARN] job ${tags[$idx]} failed, see logs_lens/r3_${tags[$idx]}.log"; fail=1; }
done

echo "[$(ts)] all ${#JOBS[@]} jobs finished (fail=$fail)"
echo "[$(ts)] run dirs written: $(find "$OUT" -mindepth 1 -maxdepth 1 -type d | wc -l) (expected 28)"

# ---------------------------------------------------------------- evaluate: TWO tables
# Every arm is scored against the SAME 200 held-out images with the same metrics and the same
# seed-0 Conv2D encoder. The primary table carries three reference rows that decide how to
# read it:
#   HOLDOUT split-half   the floor of every distributional metric (two halves of the real data)
#   RANDOM-PAIRING (ref) the floor of the paired block (IQP-lowvar-joint-MMD's samples paired
#                        with held-out images at random) -- conditioning only means something
#                        above this row
#   CTRL-copy            the memorisation ceiling: NN-train ~ 0 by construction
# Read the printed NOTE lines before quoting PDF-JS / MMD2: both bin on the JOINT range of
# (held-out, that arm), so an out-of-range arm is scored on a stretched scale, and CTRL-copy is
# the only arm carrying the ~0.6% of light that falls outside the 32x32 crop window.
#
# A second, intensity-aligned table follows (Scoring note addendum, 2026-09-07): PDF-JS /
# Conv2D-MMD^2 / NN-train / NN-holdout recomputed with every image -- generated, training and
# held-out alike -- individually rescaled to the held-out mean total first, so the scale
# mismatch the NOTE lines flag cannot dominate those four numbers. CTRL-copy's NN-train stays
# ~0 there too (same rescaling rule applied to identical arrays). The physics score (primary
# table) is unaffected by alignment and is not repeated in the second table.
echo "[$(ts)] START evaluate_r3"
"$PY" evaluate_r3.py --outputs-dir "$OUT" --n-holdout 200 --n-train-max 100 \
    --random-pair-arm "IQP-lowvar-joint-MMD" --seed 0 \
    --results "$REPO/results_r3.jsonl" \
    > logs_lens/evaluate_r3.log 2>&1 \
    || { echo "[WARN] evaluate_r3 failed"; tail -40 logs_lens/evaluate_r3.log; fail=1; }
cat logs_lens/evaluate_r3.log

echo "[$(ts)] DONE submit_lens_r3.sbatch (fail=$fail)"
exit "$fail"
