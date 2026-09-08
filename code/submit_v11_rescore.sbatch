#!/bin/bash
#SBATCH -J iqp_v11_rescore
#SBATCH -A <YOUR_ACCOUNT>
#SBATCH -p dev
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -o <HOME>/gsoc2026-deeplense-qml/experiments/13_iqp_turbulence_replication/logs_lens/v11-rescore-%j.out
#SBATCH -e <HOME>/gsoc2026-deeplense-qml/experiments/13_iqp_turbulence_replication/logs_lens/v11-rescore-%j.err
#
# Re-score every seed-0 lens output under metric protocol v11 (metrics_v11.py / evaluate_v11.py).
# NEW files only: metrics.py, evaluate_r3.py, evaluate_r3b.py, evaluate_class.py and every
# existing results_*.jsonl are untouched, so the frozen v10 numbers stay reproducible.
#
# CPU-only work (the encoder is 3 conv layers on 400 64x64 images); the GPU is requested only
# because every partition on this cluster is a GPU partition.
set -euo pipefail

cd <HOME>/gsoc2026-deeplense-qml/experiments/13_iqp_turbulence_replication
PY=/usr/bin/python3

# fail fast: a broken interpreter must not waste the allocation (memory qc-llm-env-broken.md)
$PY - <<'PYCHK'
import numpy, torch, sys
print("[v11] python", sys.version.split()[0], "numpy", numpy.__version__, "torch", torch.__version__)
import metrics, metrics_v11, evaluate_v11, data_lens_r3, data_lens_class
print("[v11] imports OK")
PYCHK

export OMP_NUM_THREADS=8

echo "=== [1/2] re-score all seed-0 outputs + 200v200 floor (pooled + 3 classes) ==="
$PY -u evaluate_v11.py \
    --outputs-dir outputs_lens_r3 outputs_lens_r3b outputs_lens_scaling outputs_convergence outputs_lens_class \
    --results results_v11_seed0.jsonl \
    --floor --n-boot 20 --n-side 200 --draw-mode resample \
    --floor-results results_v11_floor.jsonl \
    --threads 8

echo "=== [2/2] pooled floor again with MUTUALLY EXCLUSIVE draws (20 x 400 disjoint images) ==="
$PY -u metrics_v11.py \
    --n-boot 20 --n-side 200 --draw-mode disjoint --classes \
    --results results_v11_floor_disjoint.jsonl \
    --threads 8

echo "=== done ==="
wc -l results_v11_seed0.jsonl results_v11_floor.jsonl results_v11_floor_disjoint.jsonl
