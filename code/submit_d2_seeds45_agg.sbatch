#!/bin/bash
#SBATCH -J iqp_turb_seeds45_agg
#SBATCH -A <YOUR_ACCOUNT>
#SBATCH -p dev
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH -o <REPO_ROOT>/code/logs_d2/seeds45agg-%j.out
#SBATCH -e <REPO_ROOT>/code/logs_d2/seeds45agg-%j.err
#
# 5-seed completion for the ML4PS-2026 Appendix-B numbers, part 3 of 3: AGGREGATION.
# Submitted with --dependency=afterany on parts 1 and 2, so it runs once both have finished
# whether or not either reported a partial failure -- the aggregator marks any quantity it
# could not fill to 5 seeds with its actual n_seeds instead of silently averaging fewer.
# Reads only; writes results_d2_seeds.jsonl and paper/ml4ps2026/seed_tables_c.md.
set -uo pipefail
REPO=<REPO_ROOT>/code
cd "$REPO"
export PYTHONUNBUFFERED=1
PY=/usr/bin/python3
ts() { date +"%Y-%m-%d %H:%M:%S"; }
echo "[SLURM] job=$SLURM_JOB_ID node=$(hostname) part=$SLURM_JOB_PARTITION"
"$PY" -m py_compile turb_aggregate_seeds.py || { echo "[FATAL] py_compile failed"; exit 1; }
echo "[$(ts)] inputs present:"
for f in results_d2_main.jsonl results_d2_last.jsonl results_d2_seeds_main.jsonl \
         results_d2_seeds_last.jsonl results_d2_seeds_extras.jsonl; do
    printf "  %-34s %s\n" "$f" "$([ -f "$f" ] && wc -l < "$f" || echo MISSING) lines"
done
"$PY" -u turb_aggregate_seeds.py 2>&1 | tee logs_d2/seeds45_aggregate.log
rc="${PIPESTATUS[0]}"
echo "[$(ts)] DONE submit_d2_seeds45_agg.sbatch (rc=$rc)"
exit "$rc"
