#!/bin/bash
# Agent-driven AL loop — bash wrapper that calls opencode for each iteration.
# Each turn: agent sees results.tsv, modifies al_optimizer_chemprop.py, runs 1 iter.
set -e

source /home/ars3983/miniforge/bin/activate al-agent
export WANDB_MODE=disabled
cd /n/data1/hms/dbmi/farhat/aryan/AL/agent_active_learning

cp data/train_df_chemprop_orig.csv data/train_df_chemprop.csv
cp data/pool_df_chemprop_orig.csv data/pool_df_chemprop.csv

echo "=== Agent-Driven DMPNN Active Learning ==="
echo "Started at $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"

# Initialize
printf "commit\titeration\ttest_auroc\ttest_auprc\thit_rate\tstatus\tdescription\n" > results.tsv

# Setup turn
/home/ars3983/.opencode/bin/opencode run \
"You are an autonomous active learning researcher working on O2 HPC.
Current directory: $(pwd)
Branch: al/may17

DO THIS NOW:
1. git checkout -b al/may17 (remove/rename results.tsv first if it blocks you)
2. Read al_optimizer_chemprop.py (the file you modify) and program_chemprop.md
3. Initialize results.tsv with header: commit iteration test_auroc test_auprc hit_rate status description
4. Run iteration 0 (baseline): python -u al_optimizer_chemprop.py --train data/train_df_chemprop.csv --pool data/pool_df_chemprop.csv --test data/test_df_chemprop.csv --iters 1 --work_dir al_chemprop_runs > al_run.log 2>&1
5. Parse FINAL_METRICS from al_run.log
6. Append to results.tsv with status=baseline
7. git add -A && git commit -m 'iter 0: baseline (default weights)'
8. Report the test_auroc to me"

echo "Setup done at $(date)"

# Iteration loop
for iter in $(seq 1 9); do
    echo "=== Iteration $iter at $(date) ==="

    /home/ars3983/.opencode/bin/opencode run \
"AL iteration $iter. Read results.tsv to see the trajectory so far.
Based on the trends (AUROC, AUPRC, hit_rate), modify acquisition weights
(W_INHIBITION, W_UNCERTAINTY, W_NOVELTY, W_DIVERSITY) and/or hyperparameters
in al_optimizer_chemprop.py to improve test_auroc.

THEN:
1. git commit -am 'iter $iter: <describe your strategy>'
2. Run: python -u al_optimizer_chemprop.py --train data/train_df_chemprop.csv --pool data/pool_df_chemprop.csv --test data/test_df_chemprop.csv --iters 1 --work_dir al_chemprop_runs > al_run.log 2>&1
3. Parse FINAL_METRICS from al_run.log
4. Append to results.tsv with status=run
5. git add -A && git commit -m 'results: iter $iter'
6. Report test_auroc

Keep every commit. Never git reset. Continue to iteration $((iter+1))."

done

echo "=== All 10 iterations done at $(date) ==="
cat results.tsv
git push origin HEAD 2>/dev/null || echo "Push failed (no network?)"
