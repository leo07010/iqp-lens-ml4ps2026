#!/bin/bash
#SBATCH -J lens_r2_clip
#SBATCH -A <YOUR_ACCOUNT>
#SBATCH -p 8gpus
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 00:40:00
#SBATCH -o <REPO_ROOT>/code/logs_lens/clipped-%j.out
#SBATCH -e <REPO_ROOT>/code/logs_lens/clipped-%j.err
set -uo pipefail
module purge; module load cuda/13.0
cd <REPO_ROOT>/code
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
/usr/bin/python3 -m py_compile evaluate_lens_clipped.py || exit 1
for c in axion cdm no_sub; do
  echo "[$(date '+%F %T')] clipped eval $c"
  /usr/bin/python3 -u evaluate_lens_clipped.py --outputs-dir outputs_lens_pilot/$c outputs_lens_round2/$c --lens-class $c \
    --results results_lens_round2_clipped_$c.jsonl > logs_lens/evaluate_round2_clipped_$c.log 2>&1 || echo "[FAILED] $c"
done
echo "[$(date '+%F %T')] DONE"
