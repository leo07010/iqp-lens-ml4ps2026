#!/bin/bash
# Login-node watcher: wait for the listed SLURM jobs (and any job whose name matches the patterns),
# then copy v12/MLP-suite outputs into the release repo, redact identifiers, commit and push.
JOBS="362973 363008"
E=<REPO_ROOT>/code
P=<REPO_ROOT>/paper
R=<REPO_ROOT>
LOG=$E/logs_lens/sync_release_watcher.log
echo "[$(date)] watcher start, jobs=$JOBS" >> $LOG
# wait for the initial jobs and for any later jobs named v12*/mlp* (dependency chains)
while true; do
  pending=0
  for j in $JOBS; do squeue -h -j $j 2>/dev/null | grep -q . && pending=1; done
  squeue -h -u <user> -o "%j" 2>/dev/null | grep -q -E "^(v12|mlp)" && pending=1
  [ $pending -eq 0 ] && break
  sleep 300
done
echo "[$(date)] jobs finished; syncing" >> $LOG
mkdir -p $R/paper/v11/figures $R/results $R/code/seeds
for f in seed_tables_e.md seed_tables_f.md seed_tables_d.md seed_tables_d2.md; do [ -f $P/$f ] && cp $P/$f $R/paper/v11/; done
cp $P/figures/diag_v12_*.png $P/figures/fig2_mlp_sweep.* $P/figures/figA_scaling_mlp.* $P/figures/fig_samples_mlp.* $R/paper/v11/figures/ 2>/dev/null
for f in $E/results_seeds_v12.jsonl $E/results_seeds_mlp_suite.jsonl $E/results_v12_*.jsonl; do [ -f $f ] && cp $f $R/results/; done
for f in $E/train_iqp_affine_v12*.py $E/objective_spatial.py $E/codec10_gray.py $E/v12_aggregate.py $E/v12_figs.py $E/train_mlpctrl.py; do [ -f $f ] && cp $f $R/code/; done
cp $E/seeds/submit_v12*.sbatch $E/seeds/submit_mlp_suite*.sbatch $E/seeds/*mlp*.py $R/code/seeds/ 2>/dev/null
cd $R || exit 1
# Redaction step. The original line ran a grep+sed over paper/v11, results and code that
# replaced the cluster home directory, scratch root, username and SLURM account with
# <HOME>, <WORK>, <user> and <YOUR_ACCOUNT>. The search patterns were themselves the
# identifiers being redacted, so they are not reproduced in this release.
git add -A
git commit -q -m "Auto-sync: v12 IQP-side arms and IQP+MLP suite results (5 seeds, v11 metrics)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" && git push -q origin main
echo "[$(date)] pushed $(git log --oneline | head -1)" >> $LOG
