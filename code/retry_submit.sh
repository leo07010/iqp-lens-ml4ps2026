#!/bin/bash
# Retry sbatch every 10 min until accepted (QOS submit limit), up to MAX tries. Usage: retry_submit.sh <sbatch file> <tag>
set -u
F="$1"; TAG="$2"; MAX=${MAX:-72}; LOG=logs_lens/retry_submit_${TAG}.log
cd <HOME>/gsoc2026-deeplense-qml/experiments/13_iqp_turbulence_replication
for i in $(seq 1 "$MAX"); do
  out=$(sbatch "$F" 2>&1); rc=$?
  echo "$(date '+%F %T') try $i rc=$rc :: $out" >> "$LOG"
  if [ "$rc" -eq 0 ]; then echo "$out" | /bin/grep -oE '[0-9]+$' > logs_lens/submitted_${TAG}.jobid; echo "SUBMITTED $(cat logs_lens/submitted_${TAG}.jobid)"; exit 0; fi
  sleep 600
done
echo "GAVE UP"; exit 2
